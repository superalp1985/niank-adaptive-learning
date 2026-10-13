"""Controlled feedback-coordinate experiment with immutable V17 networks/world."""

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
from tracking_v18.learner import EyeMemory


def array_digest(value):
    return np.frombuffer(hashlib.sha256(value.tobytes()).digest(), np.uint8).copy()


class Session(PreviousSession):
    def __init__(self, output, frame_mode, seed, worlds=72, training=True,
                 heldout=False, device="cuda", feedback_enabled=None):
        super().__init__(output, "revisable", seed, worlds, training, heldout, device)
        self.frame_mode = frame_mode
        self.feedback_enabled = training if feedback_enabled is None else feedback_enabled
        self.memory = EyeMemory(worlds, frame_mode)

    def advance(self, steps, epochs=3):
        rollout = {k: [] for k in ("features", "proprio", "actions", "groups",
                   "action_logp", "route_logp", "value", "reward", "done", "next_rgb")}
        logs = {k: [] for k in ("combo", "segment", "goal_id", "actions", "groups",
                 "contact", "safe_contact", "collision", "impact", "spawned",
                 "reward", "done", "visible", "locked", "lock_eye", "attention_peak",
                 "reacquired", "reacquire_delay", "pred_error", "omitted_error",
                 "retention_gate", "retention_utility", "progress",
                 "body_position", "target_position", "heading", "utility_abs",
                 "gain_abs", "rotation_shift", "gain_alignment_abs",
                 "utility_before_digest", "utility_after_digest", "gate_digest")}
        for _ in range(steps):
            x, proprio = self.features()
            rgb = self.rgb.copy()
            before_digest = array_digest(self.memory.utility)
            masks = np.stack([env.audit_visibility() for env in self.envs])
            body_position = np.array([tuple(env.body.position) for env in self.envs], np.float32)
            target_position = np.array([tuple(env.target.position) for env in self.envs], np.float32)
            heading = np.array([env.heading for env in self.envs], np.float32)
            selected = self.agent.act(x, proprio, greedy=not self.training)
            actions = selected["actions"].cpu().numpy().astype(np.int64)
            groups = selected["groups"].cpu().numpy().astype(np.int64)
            attention = selected["attention"].cpu().numpy()
            lock_eye = attention.argmax(1)
            visible, locked = masks.any(1), masks[np.arange(self.worlds), lock_eye]
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
            outcomes = [env.step(int(action)) for env, action in zip(self.envs, actions)]
            after = np.stack([o[0][0] for o in outcomes])
            next_prop = np.stack([o[0][1] for o in outcomes])
            rewards = np.array([o[1] for o in outcomes], np.float32)
            dones = np.array([o[2] for o in outcomes], bool)
            infos = [o[3] for o in outcomes]
            with torch.no_grad():
                normal, omitted = self.agent.predict(
                    torch.as_tensor(x, device=self.device),
                    torch.as_tensor(proprio, device=self.device), selected["actions"])
                label = torch.as_tensor(after, device=self.device).transpose(1, 2).float() / 255
                full_error = (normal - label).square().mean(1).cpu().numpy()
                omit_error = (omitted - label).square().mean(1).cpu().numpy()
            gain = omit_error - full_error
            shifts = np.rint(next_prop[:, 2] * EYES / 2).astype(np.int16)
            gain_alignment = np.array([np.abs(np.roll(g, int(s)) - g).mean()
                                       for g, s in zip(gain, shifts)], np.float32)
            if self.feedback_enabled:
                self.memory.feedback(gain, next_prop)
            rows = []
            for i, info in enumerate(infos):
                rows.append((i, self.step, info["segment"], zlib.compress(rgb[i].tobytes(), 1),
                             proprio[i].tobytes(), int(actions[i]),
                             zlib.compress(after[i].tobytes(), 1), next_prop[i].tobytes(),
                             zlib.compress(gain[i].astype(np.float32).tobytes(), 1),
                             np.packbits(masks[i]).tobytes()))
                key = str(tuple(info["combo"]))
                if key not in self.stats:
                    self.stats[key] = dict(steps=0, contacts=0, safe_contacts=0,
                        collisions=0, impacts=0, spawns=0, visible=0, locked=0,
                        reacquisitions=0, reacquire_sum=0, prediction_sum=0.)
                record = self.stats[key]
                record["steps"] += 1
                for name, item in (("contacts", info["contact"]),
                                   ("safe_contacts", info["safe_contact"]),
                                   ("collisions", info["collision"]),
                                   ("impacts", info["impact"]),
                                   ("spawns", info["spawned"]),
                                   ("visible", visible[i]), ("locked", locked[i]),
                                   ("reacquisitions", reacquired[i]),
                                   ("reacquire_sum", reacquire_delay[i]),
                                   ("prediction_sum", full_error[i].mean())):
                    record[name] += float(item)
            self.db.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
            for key, item in (("features", x), ("proprio", proprio), ("actions", actions),
                              ("groups", groups), ("reward", rewards), ("done", dones),
                              ("next_rgb", after)):
                rollout[key].append(item)
            for key in ("action_logp", "route_logp", "value"):
                rollout[key].append(selected[key].cpu().numpy())
            for key in ("combo", "segment", "goal_id", "contact", "safe_contact",
                        "collision", "impact", "spawned", "progress"):
                logs[key].append(np.asarray([v[key] for v in infos]))
            for key, value in (("actions", actions), ("groups", groups), ("reward", rewards),
                ("done", dones), ("visible", visible), ("locked", locked), ("lock_eye", lock_eye),
                ("attention_peak", attention.max(1)), ("reacquired", reacquired),
                ("reacquire_delay", reacquire_delay), ("pred_error", full_error.mean(1)),
                ("omitted_error", omit_error.mean(1)), ("retention_gate", x[:, 9].mean(1)),
                ("retention_utility", self.memory.utility.mean(1)),
                ("utility_abs", np.abs(self.memory.utility).mean(1)),
                ("gain_abs", np.abs(gain).mean(1)), ("rotation_shift", shifts),
                ("gain_alignment_abs", gain_alignment),
                ("utility_before_digest", before_digest),
                ("utility_after_digest", array_digest(self.memory.utility)),
                ("gate_digest", array_digest(x[:, 9])),
                ("body_position", body_position), ("target_position", target_position),
                ("heading", heading)):
                logs[key].append(value)
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
                     omitted_mse=float(telemetry["omitted_error"].mean()),
                     mean_retention_gate=float(telemetry["retention_gate"].mean()),
                     mean_retention_utility=float(telemetry["retention_utility"].mean()),
                     mean_abs_utility=float(telemetry["utility_abs"].mean()),
                     mean_abs_gain=float(telemetry["gain_abs"].mean()),
                     mean_alignment_change=float(telemetry["gain_alignment_abs"].mean()),
                     predictor_loss=self.latest_loss)
        self.curves.append(point)
        dump(self.output / "progress.json", point)
        return point

    def summary(self):
        return dict(super().summary(), frame_mode=self.frame_mode,
                    feedback_enabled=self.feedback_enabled, training=self.training,
                    experience_kind="training" if self.training else "development_observation")


def probe(session, output, seed, steps, feedback):
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state() if session.device == "cuda" else None
    trial = None
    try:
        trial = Session(output, session.frame_mode, seed, len(combinations(True)),
                        training=False, heldout=True, device=session.device,
                        feedback_enabled=feedback)
        trial.agent.load(session.agent.state())
        for _ in range(steps // 128):
            trial.advance(128)
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
    sources = sorted(ROOT.glob("*.py")) + [ROOT / "PROTOCOL_ZH.md", ROOT / "config.json"]
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
    dump(output / "manifest.json", dict(config=config, source=hashes,
         warm_start_sha256=hashlib.sha256(warm.read_bytes()).hexdigest(),
         gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         warm_start_kind="weights_only_fresh_optimizers_worlds_and_traces",
         new_training_experiences_per_arm=config["milestones"][-1] * config["worlds"],
         total_ancestral_plus_new_per_arm=config["ancestor_experiences"] +
                                        config["milestones"][-1] * config["worlds"]))
    return output


def worker(config, arm):
    output = ROOT / config["stage"]
    torch.manual_seed(config["model_seed"])
    torch.cuda.manual_seed_all(config["model_seed"])
    session = Session(output / arm, arm, config["world_seed"], config["worlds"])
    warm = torch.load(output / "warm_start.pt", map_location="cuda", weights_only=True)
    session.agent.load(warm["model"])
    torch.save(dict(model=session.agent.state()), session.output / "initial.pt")
    try:
        for milestone in [0, *config["milestones"]]:
            while session.step < milestone:
                point = session.advance(min(config["rollout"], milestone - session.step),
                                        config["epochs"])
                if session.chunk % 8 == 0:
                    print(json.dumps(dict(event="progress", arm=arm, **point)), flush=True)
            if milestone:
                torch.save(dict(model=session.agent.state(),
                                optimizers=[o.state_dict() for o in session.agent.optimizers],
                                step=session.step, memory=session.memory.__dict__,
                                torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state()),
                           session.output / f"checkpoint_{milestone}.pt")
            result = session.summary()
            result["probes"] = {}
            for feedback in config["probe_feedback"]:
                name = "adaptive" if feedback else "frozen"
                result["probes"][name] = probe(session,
                    session.output / f"probe_{milestone}_{name}",
                    config["world_seed"] + 900000, config["probe_steps"], feedback)
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
    parser.add_argument("--arm", choices=("legacy", "aligned"))
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.arm:
        worker(config, args.arm)
        return
    output = prepare(config)
    children, logs = [], []
    try:
        for arm in config["arms"]:
            log = (output / f"{arm}.log").open("w", encoding="utf-8")
            logs.append(log)
            child = subprocess.Popen([sys.executable, "-m", "tracking_v18.run", "--arm", arm],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            children.append((arm, child))
        while any(child.poll() is None for _, child in children):
            failed = [(arm, child.returncode) for arm, child in children
                      if child.poll() is not None and child.returncode != 0]
            if failed:
                raise RuntimeError(f"Worker failed; preserve all outputs: {failed}")
            time.sleep(2)
        failures = [(arm, child.returncode) for arm, child in children if child.returncode]
        if failures:
            raise RuntimeError(f"Worker failed: {failures}")
        dump(output / "COMPLETE.json", dict(arms=config["arms"],
             steps=config["milestones"][-1], new_training_experiences=2 *
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
