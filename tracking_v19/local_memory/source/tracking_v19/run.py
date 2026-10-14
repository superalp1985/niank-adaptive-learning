"""Factual local-memory trials with matched finite practice budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from tracking_v17.run import Session as PreviousSession, dump
from tracking_v17.world import EYES, combinations
from tracking_v19.learner import Agent, EyeMemory, bootstrap_mask, probe_starts


def array_digest(value):
    return np.frombuffer(hashlib.sha256(value.tobytes()).digest(), np.uint8).copy()


class Session(PreviousSession):
    def __init__(self, output, mode, seed, worlds=72, training=True,
                 heldout=False, device="cuda", feedback_enabled=None,
                 memory_seed=19019, floor=1 / 255 ** 2):
        super().__init__(output, mode, seed, worlds, training, heldout, device)
        self.agent = Agent(mode, device)
        self.memory = EyeMemory(worlds, mode, memory_seed)
        self.memory_seed, self.floor = memory_seed, float(floor)
        self.feedback_enabled = (training and mode != "calibration"
                                 if feedback_enabled is None else feedback_enabled)
        self.db.execute("CREATE TABLE evidence(body INTEGER,time INTEGER,start INTEGER,"
                        "normal_errors BLOB,omitted_errors BLOB,scores BLOB,"
                        "disagreement REAL,bootstrap BLOB,PRIMARY KEY(body,time))")

    def advance(self, steps, epochs=3):
        rollout = {k: [] for k in ("features", "full_features", "proprio", "actions",
            "groups", "action_logp", "route_logp", "value", "reward", "done",
            "next_rgb", "starts", "bootstrap")}
        fields = ("combo", "segment", "goal_id", "actions", "groups", "contact",
            "safe_contact", "collision", "impact", "spawned", "reward", "done",
            "visible", "locked", "lock_eye", "attention_peak", "reacquired",
            "reacquire_delay", "pred_error", "omitted_error", "retention_gate",
            "retention_utility", "progress", "body_position", "target_position",
            "heading", "utility_abs", "deviation_mean", "gain_mean", "gain_std",
            "prediction_disagreement", "probe_selected_fraction", "promoted",
            "selected_bits", "probe_start", "rotation_shift", "utility_before_digest",
            "utility_after_digest", "deviation_before_digest",
            "deviation_after_digest", "gate_digest", "full_features_digest")
        logs = {key: [] for key in fields}
        for _ in range(steps):
            x, proprio = self.features()
            full = self.memory.full_features.copy()
            rgb = self.rgb.copy()
            starts = probe_starts(self.worlds, self.step)
            groups_eyes = (starts[:, None] + np.arange(16)) % EYES
            probed_selected = np.take_along_axis(
                self.memory.selected, groups_eyes, 1).mean(1)
            before = array_digest(self.memory.utility)
            deviation_before = array_digest(self.memory.deviation)
            masks = np.stack([env.audit_visibility() for env in self.envs])
            position = np.array([tuple(e.body.position) for e in self.envs], np.float32)
            target = np.array([tuple(e.target.position) for e in self.envs], np.float32)
            heading = np.array([e.heading for e in self.envs], np.float32)
            selected = self.agent.act(x, proprio, greedy=not self.training)
            actions = selected["actions"].cpu().numpy().astype(np.int64)
            groups = selected["groups"].cpu().numpy().astype(np.int64)
            attention = selected["attention"].cpu().numpy()
            lock_eye = attention.argmax(1)
            visible = masks.any(1)
            locked = masks[np.arange(self.worlds), lock_eye]
            reacquired = np.zeros(self.worlds, bool)
            reacquire_delay = np.zeros(self.worlds, np.int32)
            for i, env in enumerate(self.envs):
                if env.goal_id != self.goal_ids[i]:
                    self.ever_locked[i] = False
                    self.reacquire_wait[i] = -1
                    self.goal_ids[i] = env.goal_id
                if self.ever_locked[i] and self.previous_lock[i] and not locked[i]:
                    self.reacquire_wait[i] = self.step
                if locked[i]:
                    if self.reacquire_wait[i] >= 0:
                        reacquired[i] = True
                        reacquire_delay[i] = self.step - self.reacquire_wait[i]
                        self.reacquire_wait[i] = -1
                    self.ever_locked[i] = True
            self.previous_lock, self.previous_visible = locked.copy(), visible.copy()
            outcomes = [env.step(int(a)) for env, a in zip(self.envs, actions)]
            after = np.stack([v[0][0] for v in outcomes])
            next_prop = np.stack([v[0][1] for v in outcomes])
            rewards = np.asarray([v[1] for v in outcomes], np.float32)
            dones = np.asarray([v[2] for v in outcomes], bool)
            infos = [v[3] for v in outcomes]
            evidence = self.agent.evidence(full, proprio, actions, starts, after, self.floor)
            scores = evidence["scores"].astype(np.float32)
            bootstrap = bootstrap_mask(np.arange(self.worlds), self.step,
                                       np.array([v["segment"] for v in infos]))
            if self.feedback_enabled:
                self.memory.feedback(scores, starts)
            gains = np.zeros((self.worlds, EYES), np.float32)
            np.put_along_axis(gains, groups_eyes, scores.mean(1)[:, None], 1)
            rows, trials = [], []
            for i, info in enumerate(infos):
                rows.append((i, self.step, info["segment"],
                    zlib.compress(rgb[i].tobytes(), 1), proprio[i].tobytes(), int(actions[i]),
                    zlib.compress(after[i].tobytes(), 1), next_prop[i].tobytes(),
                    zlib.compress(gains[i].tobytes(), 1), np.packbits(masks[i]).tobytes()))
                trials.append((i, self.step, int(starts[i]),
                    evidence["normal_local"][i].astype(np.float32).tobytes(),
                    evidence["omitted_local"][i].astype(np.float32).tobytes(),
                    scores[i].tobytes(), float(evidence["prediction_disagreement"][i]),
                    bootstrap[i].tobytes()))
                key = str(tuple(info["combo"]))
                if key not in self.stats:
                    self.stats[key] = dict(steps=0, contacts=0, safe_contacts=0,
                        collisions=0, impacts=0, spawns=0, visible=0, locked=0,
                        reacquisitions=0, reacquire_sum=0, prediction_sum=0.)
                record = self.stats[key]
                record["steps"] += 1
                for name, item in (("contacts", info["contact"]),
                    ("safe_contacts", info["safe_contact"]), ("collisions", info["collision"]),
                    ("impacts", info["impact"]), ("spawns", info["spawned"]),
                    ("visible", visible[i]), ("locked", locked[i]),
                    ("reacquisitions", reacquired[i]), ("reacquire_sum", reacquire_delay[i]),
                    ("prediction_sum", evidence["full_mse"][i])):
                    record[name] += float(item)
            self.db.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
            self.db.executemany("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?)", trials)
            for key, item in (("features", x), ("full_features", full),
                ("proprio", proprio), ("actions", actions), ("groups", groups),
                ("reward", rewards), ("done", dones), ("next_rgb", after),
                ("starts", starts), ("bootstrap", bootstrap)):
                rollout[key].append(item)
            for key in ("action_logp", "route_logp", "value"):
                rollout[key].append(selected[key].cpu().numpy())
            for key in ("combo", "segment", "goal_id", "contact", "safe_contact",
                        "collision", "impact", "spawned", "progress"):
                logs[key].append(np.asarray([v[key] for v in infos]))
            values = dict(actions=actions, groups=groups, reward=rewards, done=dones,
                visible=visible, locked=locked, lock_eye=lock_eye,
                attention_peak=attention.max(1), reacquired=reacquired,
                reacquire_delay=reacquire_delay, pred_error=evidence["full_mse"],
                omitted_error=evidence["omitted_mse"], retention_gate=x[:, 9].mean(1),
                retention_utility=self.memory.utility.mean(1),
                utility_abs=np.abs(self.memory.utility).mean(1),
                deviation_mean=self.memory.deviation.mean(1),
                gain_mean=scores.mean(1), gain_std=scores.std(1),
                prediction_disagreement=evidence["prediction_disagreement"],
                probe_selected_fraction=probed_selected,
                promoted=self.memory.promoted.sum(1),
                selected_bits=np.packbits(self.memory.selected, axis=1),
                probe_start=starts,
                rotation_shift=np.rint(next_prop[:, 2] * EYES / 2).astype(np.int16),
                utility_before_digest=before,
                utility_after_digest=array_digest(self.memory.utility),
                deviation_before_digest=deviation_before,
                deviation_after_digest=array_digest(self.memory.deviation),
                gate_digest=array_digest(x[:, 9]), full_features_digest=array_digest(full),
                body_position=position, target_position=target, heading=heading)
            for key, item in values.items():
                logs[key].append(item)
            self.step += 1
            for i, done in enumerate(dones):
                if done:
                    self.envs[i].reset()
                    self.reset_flags[i] = True
                    self.ever_locked[i] = self.previous_lock[i] = False
                    self.reacquire_wait[i] = -1
                    self.goal_ids[i] = self.envs[i].goal_id
        if self.training:
            data = {key: np.asarray(value) for key, value in rollout.items()}
            last_x, last_prop = self.features()
            self.latest_loss = self.agent.update(data, last_x, last_prop, epochs)
        telemetry = {key: np.asarray(value) for key, value in logs.items()}
        telemetry["start_step"] = np.array(self.step - steps)
        np.savez_compressed(self.output / f"transitions_{self.chunk:05d}.npz", **telemetry)
        self.db.commit()
        self.chunk += 1
        point = dict(step=self.step, experiences=self.step * self.worlds,
            contact_count=int(telemetry["contact"].sum()),
            safe_contacts=int(telemetry["safe_contact"].sum()),
            collision_rate=float(telemetry["collision"].mean()),
            impact_rate=float(telemetry["impact"].mean()),
            visible_lock=float(telemetry["locked"].sum() /
                               max(telemetry["visible"].sum(), 1)),
            preupdate_mse=float(telemetry["pred_error"].mean()),
            predictor_loss=self.latest_loss)
        for key in ("retention_gate", "utility_abs", "deviation_mean", "gain_mean",
                    "gain_std", "prediction_disagreement", "probe_selected_fraction", "promoted"):
            point[key] = float(telemetry[key].mean())
        self.curves.append(point)
        dump(self.output / "progress.json", point)
        return point

    def summary(self):
        return dict(super().summary(), feedback_enabled=self.feedback_enabled,
            training=self.training, memory_seed=self.memory_seed, floor=self.floor,
            experience_kind="training" if self.training else "development_observation")


def probe(session, output, config, feedback):
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state() if session.device == "cuda" else None
    trial = None
    try:
        trial = Session(output, session.mode, config["world_seed"] + 900000,
            len(combinations(True)), training=False, heldout=True, device=session.device,
            feedback_enabled=feedback, memory_seed=config["memory_seed"] + 900000,
            floor=session.floor)
        trial.agent.load(session.agent.state())
        for _ in range(config["probe_steps"] // config["rollout"]):
            trial.advance(config["rollout"])
        result = trial.summary()
        dump(trial.output / "summary.json", result)
        return result
    finally:
        if trial is not None:
            trial.close()
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state(gpu)


def prepare(config):
    output = ROOT / config["stage"]
    output.mkdir(exist_ok=False)
    sources = sorted(ROOT.glob("*.py")) + [ROOT / name for name in (
        "PROTOCOL_ZH.md", "MEMORY_RESEARCH_ZH.md", "ANTHROPIC_RESEARCH_ZH.md", "config.json")]
    sources += [REPO / name for name in ("tracking_v17/learner.py", "tracking_v17/world.py",
                 "tracking_v17/run.py", "tracking_v14/learner.py", "tracking_v14/world.py")]
    hashes = {}
    for source in sources:
        relative = source.relative_to(REPO)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    warm = REPO / config["warm_start"]
    shutil.copyfile(warm, output / "warm_start.pt")
    calibration = config["calibration_steps"] * config["worlds"]
    per_arm = config["milestones"][-1] * config["worlds"]
    dump(output / "manifest.json", dict(config=config, source=hashes,
        warm_start_sha256=hashlib.sha256(warm.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        warm_start_kind="weights_only_fresh_optimizers_worlds_and_traces",
        shared_calibration_experiences=calibration,
        new_training_experiences_per_arm=per_arm,
        unique_new_training_experiences=calibration + len(config["arms"]) * per_arm,
        total_ancestral_plus_new_per_arm=config["ancestor_experiences"] + calibration + per_arm))
    return output


def calibrate(config, output):
    torch.manual_seed(config["model_seed"])
    torch.cuda.manual_seed_all(config["model_seed"])
    session = Session(output / "calibration", "calibration", config["world_seed"] - 100000,
                      config["worlds"], memory_seed=config["memory_seed"])
    ancestor = torch.load(output / "warm_start.pt", map_location="cuda", weights_only=True)
    session.agent.load_ancestor(ancestor["model"])
    torch.save(dict(model=session.agent.state()), session.output / "initial.pt")
    try:
        while session.step < config["calibration_steps"]:
            session.advance(config["rollout"], config["epochs"])
        dump(session.output / "summary.json", session.summary())
        torch.save(dict(model=session.agent.state()), session.output / "model.pt")
        sums = []
        for normal, omitted in session.db.execute("SELECT normal_errors,omitted_errors FROM evidence"):
            sums.append(np.frombuffer(normal, np.float32) + np.frombuffer(omitted, np.float32))
        floor = max(float(np.quantile(np.asarray(sums).mean(1), config["floor_quantile"])),
                    1 / 255 ** 2)
        dump(output / "calibration.json", dict(floor=floor, quantile=config["floor_quantile"],
            rows=session.step * session.worlds, parameters=session.agent.parameters(),
            shared_model_sha256=hashlib.sha256((session.output / "model.pt").read_bytes()).hexdigest()))
        print(json.dumps(dict(event="calibration_complete", floor=floor)), flush=True)
    finally:
        session.close()


def worker(config, arm):
    output = ROOT / config["stage"]
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["model_seed"] + 1)
    torch.cuda.manual_seed_all(config["model_seed"] + 1)
    session = Session(output / arm, arm, config["world_seed"], config["worlds"],
                      memory_seed=config["memory_seed"], floor=calibration["floor"])
    warm = torch.load(output / "calibration" / "model.pt", map_location="cuda", weights_only=True)
    session.agent.load(warm["model"])
    torch.save(dict(model=session.agent.state()), session.output / "initial.pt")
    try:
        for milestone in [0, *config["milestones"]]:
            while session.step < milestone:
                point = session.advance(config["rollout"], config["epochs"])
                if session.chunk % 8 == 0:
                    print(json.dumps(dict(event="progress", arm=arm, **point)), flush=True)
            if milestone:
                torch.save(dict(model=session.agent.state(), step=session.step,
                    optimizers=[o.state_dict() for o in session.agent.optimizers],
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
                    session.output / f"checkpoint_{milestone}.pt")
            result = session.summary()
            result["probes"] = {}
            for feedback in config["probe_feedback"]:
                name = "adaptive" if feedback else "frozen"
                result["probes"][name] = probe(session,
                    session.output / f"probe_{milestone}_{name}", config, feedback)
            dump(session.output / f"result_{milestone}.json", result)
            print(json.dumps(dict(event="milestone", arm=arm, step=milestone,
                probes={k: {metric: v[metric] for metric in ("contact_per_minute",
                    "visible_lock", "collision_rate", "attacked_impact_rate")}
                        for k, v in result["probes"].items()})), flush=True)
        torch.save(dict(model=session.agent.state()), session.output / "model.pt")
        dump(session.output / "summary.json", session.summary())
        dump(session.output / "COMPLETE.json", dict(arm=arm, steps=session.step,
                                                 experiences=session.step * session.worlds))
    finally:
        session.close()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Use the existing E: CUDA Python")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("utility", "protected"))
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.arm:
        worker(config, args.arm)
        return
    output = prepare(config)
    calibrate(config, output)
    torch.cuda.empty_cache()
    children, logs = [], []
    try:
        for arm in config["arms"]:
            log = (output / f"{arm}.log").open("w", encoding="utf-8")
            logs.append(log)
            child = subprocess.Popen([sys.executable, "-m", "tracking_v19.run", "--arm", arm],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            children.append((arm, child))
        while any(child.poll() is None for _, child in children):
            failed = [(arm, child.returncode) for arm, child in children
                      if child.poll() is not None and child.returncode != 0]
            if failed:
                raise RuntimeError(f"Worker failed; preserve outputs: {failed}")
            time.sleep(2)
        if any(child.returncode for _, child in children):
            raise RuntimeError("Worker failed; preserve all outputs")
        dump(output / "COMPLETE.json", dict(arms=config["arms"],
             shared_calibration_experiences=config["calibration_steps"] * config["worlds"],
             additional_training_experiences=len(config["arms"]) *
             config["milestones"][-1] * config["worlds"]))
        print(json.dumps(dict(complete=True, stage=config["stage"])), flush=True)
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
