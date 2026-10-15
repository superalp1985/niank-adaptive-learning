"""GPU multiscene practice with delayed factual labels and equal replay budgets."""

from __future__ import annotations

import argparse
from collections import deque
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
from tracking_v17.run import dump
from tracking_v17.world import EYES, combinations
from tracking_v19.run import Session as PreviousSession
from tracking_v19.learner import bootstrap_mask, probe_starts
from tracking_v20.directory import EventDirectory
from tracking_v20.learner import Agent, HORIZON


class Pending:
    def __init__(self, worlds):
        self.queues = [deque() for _ in range(worlds)]

    def push(self, body, time, segment, features, full, proprio, action, group):
        self.queues[body].append(dict(body=body, time=time, segment=segment,
            features=features.copy(), full=full.copy(), proprio=proprio.copy(),
            action=int(action), group=int(group), target=np.zeros(5, np.float32), horizon=0))

    def observe(self, body, info, done):
        queue, result = self.queues[body], []
        for event in queue:
            if event["segment"] != info["segment"]:
                raise RuntimeError("Consequence window crossed a segment")
            event["target"][0] += .95 ** event["horizon"] * info["progress"] / 1.2
            event["target"][1:] = np.maximum(event["target"][1:], [
                info["contact"], info["safe_contact"], info["collision"], info["impact"]])
            event["horizon"] += 1
        while queue and (queue[0]["horizon"] >= HORIZON or done or info["contact"]):
            event = queue.popleft()
            event["target"][0] = np.clip(event["target"][0], -1, 1)
            result.append(event)
        return result


def batch_events(events):
    return dict(features=np.asarray([e["features"] for e in events], np.float32),
        proprio=np.asarray([e["proprio"] for e in events], np.float32),
        actions=np.asarray([e["action"] for e in events], np.int64),
        groups=np.asarray([e["group"] for e in events], np.int64),
        targets=np.asarray([e["target"] for e in events], np.float32))


class Session(PreviousSession):
    def __init__(self, output, mode, seed, config, training=True, heldout=False,
                 device="cuda", calibration=False):
        worlds = config["worlds"] if training else len(combinations(True))
        super().__init__(output, "utility", seed, worlds, training, heldout, device,
                         feedback_enabled=not calibration,
                         memory_seed=config["memory_seed"], floor=1 / 255 ** 2)
        self.mode, self.config, self.calibration = mode, config, calibration
        self.agent = Agent(mode, device, 0 if calibration else config["guidance"])
        self.directory = EventDirectory(worlds, "random" if calibration else mode,
            config["memory_seed"], config["directory_capacity"])
        self.pending = Pending(worlds)
        self.query_strength = np.zeros(worlds, np.float32)
        self.query_features = np.zeros((worlds, 11, EYES), np.float32)
        self.query_proprio = np.zeros((worlds, 6), np.float32)
        self.directory_insertions = self.retrieval_count = self.reopen_count = 0
        self.db.execute("CREATE TABLE outcomes(body INTEGER,time INTEGER,segment INTEGER,"
            "horizon INTEGER,targets BLOB,PRIMARY KEY(body,time))")
        self.db.execute("CREATE TABLE directory_events(body INTEGER,time INTEGER,segment INTEGER,"
            "features BLOB,full_features BLOB,proprio BLOB,action INTEGER,expert INTEGER,"
            "targets BLOB,PRIMARY KEY(body,time))")
        self.db.execute("CREATE TABLE retrievals(query_time INTEGER,body INTEGER,event_time INTEGER,"
            "event_segment INTEGER,distance REAL,random INTEGER,weight_before REAL,"
            "weight_after REAL,failed_query REAL)")

    def advance(self, steps, epochs=2):
        rollout = {key: [] for key in ("features", "full_features", "proprio", "actions",
            "groups", "action_logp", "route_logp", "value", "reward", "done",
            "next_rgb", "starts", "bootstrap", "guide_logits")}
        fields = ("combo", "segment", "goal_id", "actions", "groups", "contact",
            "safe_contact", "collision", "impact", "spawned", "reward", "done", "visible",
            "locked", "lock_eye", "reacquired", "reacquire_delay", "pred_error",
            "retention_gate", "progress", "selected_bits", "body_position", "target_position",
            "heading", "consequence_predictions", "query_strength")
        logs, matured = {key: [] for key in fields}, []
        for _ in range(steps):
            x, proprio = self.features()
            full, rgb = self.memory.full_features.copy(), self.rgb.copy()
            starts = probe_starts(self.worlds, self.step)
            masks = np.stack([env.audit_visibility() for env in self.envs])
            position = np.asarray([tuple(e.body.position) for e in self.envs], np.float32)
            target = np.asarray([tuple(e.target.position) for e in self.envs], np.float32)
            heading = np.asarray([e.heading for e in self.envs], np.float32)
            selected = self.agent.act(x, proprio, greedy=not self.training)
            actions = selected["actions"].cpu().numpy().astype(np.int64)
            groups = selected["groups"].cpu().numpy().astype(np.int64)
            predictions = selected["outcome_predictions"].cpu().numpy()
            lock_eye = selected["attention"].cpu().numpy().argmax(1)
            visible, locked = masks.any(1), masks[np.arange(self.worlds), lock_eye]
            reacquired = np.zeros(self.worlds, bool)
            reacquire_delay = np.zeros(self.worlds, np.int32)
            for i, env in enumerate(self.envs):
                if env.goal_id != self.goal_ids[i]:
                    self.ever_locked[i], self.reacquire_wait[i] = False, -1
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
            after, next_prop = (np.stack([o[0][j] for o in outcomes]) for j in (0, 1))
            rewards = np.asarray([o[1] for o in outcomes], np.float32)
            dones = np.asarray([o[2] for o in outcomes], bool)
            infos = [o[3] for o in outcomes]
            evidence = self.agent.evidence(full, proprio, actions, starts, after, self.floor)
            if self.feedback_enabled:
                self.memory.feedback(evidence["scores"], starts)
            bootstrap = bootstrap_mask(np.arange(self.worlds), self.step,
                                       np.asarray([v["segment"] for v in infos]))
            fresh, raw, trials = [], [], []
            for i, info in enumerate(infos):
                self.pending.push(i, self.step, info["segment"], x[i], full[i], proprio[i],
                                  actions[i], groups[i])
                fresh.extend(self.pending.observe(i, info, bool(dones[i])))
                raw.append((i, self.step, info["segment"], zlib.compress(rgb[i].tobytes(), 1),
                    proprio[i].tobytes(), int(actions[i]), zlib.compress(after[i].tobytes(), 1),
                    next_prop[i].tobytes(), b"", np.packbits(masks[i]).tobytes()))
                trials.append((i, self.step, int(starts[i]),
                    evidence["normal_local"][i].astype(np.float32).tobytes(),
                    evidence["omitted_local"][i].astype(np.float32).tobytes(),
                    evidence["scores"][i].astype(np.float32).tobytes(),
                    float(evidence["prediction_disagreement"][i]), bootstrap[i].tobytes()))
                key = str(tuple(info["combo"]))
                if key not in self.stats:
                    self.stats[key] = dict(steps=0, contacts=0, safe_contacts=0, collisions=0,
                        impacts=0, spawns=0, visible=0, locked=0, reacquisitions=0,
                        reacquire_sum=0, prediction_sum=0.)
                record = self.stats[key]
                record["steps"] += 1
                for name, value in (("contacts", info["contact"]),
                    ("safe_contacts", info["safe_contact"]), ("collisions", info["collision"]),
                    ("impacts", info["impact"]), ("spawns", info["spawned"]),
                    ("visible", visible[i]), ("locked", locked[i]), ("reacquisitions", reacquired[i]),
                    ("reacquire_sum", reacquire_delay[i]), ("prediction_sum", evidence["full_mse"][i])):
                    record[name] += float(value)
            self.db.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)", raw)
            self.db.executemany("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?)", trials)
            self.db.executemany("INSERT INTO outcomes VALUES(?,?,?,?,?)", [
                (e["body"], e["time"], e["segment"], e["horizon"], e["target"].tobytes())
                for e in fresh])
            directory_rows = []
            for e in fresh:
                # Keep a regular temporal sample and every factual consequence event.
                if e["time"] % 4 != e["body"] % 4 and not e["target"][1:].any():
                    continue
                self.directory.add(e["body"], e["time"], e["segment"], e["features"],
                    e["proprio"], e["action"], e["group"], e["target"], key_features=e["full"])
                directory_rows.append((e["body"], e["time"], e["segment"],
                    zlib.compress(e["features"].astype(np.float16).tobytes(), 1),
                    zlib.compress(e["full"].astype(np.float16).tobytes(), 1),
                    e["proprio"].tobytes(), e["action"], e["group"], e["target"].tobytes()))
            self.directory_insertions += len(directory_rows)
            self.db.executemany("INSERT INTO directory_events VALUES(?,?,?,?,?,?,?,?,?)", directory_rows)
            matured.extend(fresh)
            surprise = np.clip((evidence["full_mse"] - .02) / .02, 0, 1)
            strength = (np.asarray([v["collision"] + 3 * v["impact"] for v in infos], np.float32) +
                        surprise)
            higher = strength > self.query_strength
            self.query_features[higher], self.query_proprio[higher] = full[higher], proprio[higher]
            self.query_strength = np.maximum(self.query_strength, strength)
            for key, item in dict(features=x, full_features=full, proprio=proprio, actions=actions,
                groups=groups, reward=rewards, done=dones, next_rgb=after, starts=starts,
                bootstrap=bootstrap).items():
                rollout[key].append(item)
            for key in ("action_logp", "route_logp", "value", "guide_logits"):
                rollout[key].append(selected[key].cpu().numpy())
            for key in ("combo", "segment", "goal_id", "contact", "safe_contact",
                        "collision", "impact", "spawned", "progress"):
                logs[key].append(np.asarray([v[key] for v in infos]))
            for key, item in dict(actions=actions, groups=groups, reward=rewards, done=dones,
                visible=visible, locked=locked, lock_eye=lock_eye, reacquired=reacquired,
                reacquire_delay=reacquire_delay, pred_error=evidence["full_mse"],
                retention_gate=x[:, 9].mean(1), selected_bits=np.packbits(self.memory.selected, axis=1),
                body_position=position, target_position=target, heading=heading,
                consequence_predictions=predictions[np.arange(self.worlds), actions],
                query_strength=strength).items():
                logs[key].append(item)
            self.step += 1
            for i, done in enumerate(dones):
                if done:
                    self.envs[i].reset()
                    self.reset_flags[i] = True
                    self.ever_locked[i] = self.previous_lock[i] = False
                    self.reacquire_wait[i] = -1
                    self.goal_ids[i] = self.envs[i].goal_id
        replay_metrics = {}
        if self.training:
            last_x, last_prop = self.features()
            if not self.calibration:
                self.latest_loss = self.agent.update(
                    {key: np.asarray(v) for key, v in rollout.items()}, last_x, last_prop, epochs)
            fresh_metrics = self.agent.fit_consequences(batch_events(matured))
            replay_metrics.update(fresh_metrics)
            if not self.calibration:
                query = self.directory.key(self.query_features, self.query_proprio)
                replay, accesses = self.directory.sample(query, self.query_strength,
                                                         self.step, self.config["replay_budget"])
                replay_metrics.update({f"replay_{k}": v for k, v in
                                       self.agent.fit_consequences(replay, replay=True).items()})
                self.db.executemany("INSERT INTO retrievals VALUES(?,?,?,?,?,?,?,?,?)", [
                    (r["query_time"], r["body"], r["event_time"], r["event_segment"], r["distance"],
                     int(r["random"]), r["weight_before"], r["weight_after"], r["failed_query"])
                    for r in accesses])
                self.retrieval_count += len(accesses)
                self.reopen_count += sum(r["weight_before"] < .25 for r in accesses)
        self.query_strength.fill(0)
        telemetry = {key: np.asarray(v) for key, v in logs.items()}
        telemetry["start_step"] = np.array(self.step - steps)
        np.savez_compressed(self.output / f"transitions_{self.chunk:05d}.npz", **telemetry)
        self.db.commit()
        self.chunk += 1
        point = dict(step=self.step, experiences=self.step * self.worlds,
            contact_count=int(telemetry["contact"].sum()),
            safe_contacts=int(telemetry["safe_contact"].sum()),
            collision_rate=float(telemetry["collision"].mean()),
            impact_rate=float(telemetry["impact"].mean()),
            preupdate_mse=float(telemetry["pred_error"].mean()),
            directory_insertions=self.directory_insertions, retrievals=self.retrieval_count,
            reopened_accesses=self.reopen_count, **replay_metrics)
        self.curves.append(point)
        dump(self.output / "progress.json", point)
        return point

    def summary(self):
        return dict(super().summary(), mode=self.mode, calibration=self.calibration,
            guidance=self.agent.guidance, consequence_presentations=self.agent.consequence_presentations,
            replay_presentations=self.agent.replay_presentations,
            replay_policy_presentations=self.agent.replay_policy_presentations,
            directory_insertions=self.directory_insertions, retrieval_count=self.retrieval_count,
            reopened_accesses=self.reopen_count,
            pending_unlabelled=sum(len(q) for q in self.pending.queues),
            experience_kind="shared_head_calibration" if self.calibration else
                            ("training" if self.training else "development_observation"))


def probe(session, output, config, guidance=None):
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    trial = None
    try:
        trial = Session(output, session.mode, config["world_seed"] + 900000,
                        config, training=False, heldout=True)
        trial.agent.load(session.agent.state())
        if guidance is not None:
            trial.agent.guidance = guidance
        for _ in range(config["probe_steps"] // config["rollout"]):
            trial.advance(config["rollout"])
        result = trial.summary()
        dump(trial.output / "summary.json", result)
        return result
    finally:
        if trial:
            trial.close()
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(gpu)


def prepare(config):
    output = ROOT / config["stage"]
    output.mkdir(exist_ok=False)
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "PROTOCOL_ZH.md", ROOT / "config.json"]
    paths += [REPO / name for name in ("tracking_v19/run.py", "tracking_v19/learner.py",
        "tracking_v17/run.py", "tracking_v17/learner.py", "tracking_v17/world.py",
        "tracking_v14/learner.py", "tracking_v14/world.py")]
    hashes = {}
    for source in paths:
        relative = source.relative_to(REPO)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    warm = REPO / config["warm_start"]
    shutil.copyfile(warm, output / "warm_start.pt")
    dump(output / "manifest.json", dict(config=config, source=hashes,
        warm_start_sha256=hashlib.sha256(warm.read_bytes()).hexdigest(),
        warm_start_kind="weights_only_fresh_worlds_optimizers_and_traces",
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        unique_new_training_experiences=config["calibration_steps"] * config["worlds"] +
            len(config["arms"]) * config["milestones"][-1] * config["worlds"],
        ancestral_plus_new_per_arm=config["ancestor_experiences"] +
            (config["calibration_steps"] + config["milestones"][-1]) * config["worlds"]))
    return output


def calibrate(config, output):
    torch.manual_seed(config["model_seed"])
    torch.cuda.manual_seed_all(config["model_seed"])
    session = Session(output / "calibration", "random", config["world_seed"] - 100000,
                      config, calibration=True)
    session.agent.load(torch.load(output / "warm_start.pt", map_location="cuda",
                                  weights_only=True)["model"])
    try:
        while session.step < config["calibration_steps"]:
            session.advance(config["rollout"])
        torch.save(dict(model=session.agent.state()), session.output / "model.pt")
        dump(session.output / "summary.json", session.summary())
        reference = probe(session, output / "baseline_reference", config, guidance=0.)
        dump(output / "baseline_reference.json", reference)
        print(json.dumps(dict(event="head_calibration_complete", steps=session.step)), flush=True)
    finally:
        session.close()


def worker(config, arm):
    output = ROOT / config["stage"]
    torch.manual_seed(config["model_seed"] + 1)
    torch.cuda.manual_seed_all(config["model_seed"] + 1)
    session = Session(output / arm, arm, config["world_seed"], config)
    session.agent.load(torch.load(output / "calibration/model.pt", map_location="cuda",
                                  weights_only=True)["model"])
    torch.save(dict(model=session.agent.state()), session.output / "initial.pt")
    try:
        for milestone in [0, *config["milestones"]]:
            while session.step < milestone:
                point = session.advance(config["rollout"], config["epochs"])
                if session.chunk % 8 == 0:
                    print(json.dumps(dict(event="progress", arm=arm, **point)), flush=True)
            if milestone:
                torch.save(dict(model=session.agent.state(), step=session.step,
                    optimizers=[o.state_dict() for o in session.agent.optimizers]),
                    session.output / f"checkpoint_{milestone}.pt")
            result = session.summary()
            result["probe"] = probe(session, session.output / f"probe_{milestone}", config)
            dump(session.output / f"result_{milestone}.json", result)
            print(json.dumps(dict(event="milestone", arm=arm, step=milestone,
                safe_contacts=result["probe"]["safe_contacts"], contacts=result["probe"]["contacts"],
                collision=result["probe"]["collision_rate"],
                impact=result["probe"]["attacked_impact_rate"])), flush=True)
        torch.save(dict(model=session.agent.state()), session.output / "model.pt")
        dump(session.output / "summary.json", session.summary())
        dump(session.output / "COMPLETE.json", dict(arm=arm, steps=session.step))
    finally:
        session.close()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Use the existing E: CUDA Python")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("retrieval", "random"))
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if args.arm:
        worker(config, args.arm)
        return
    output = prepare(config)
    calibrate(config, output)
    torch.cuda.empty_cache()
    children, handles = [], []
    try:
        for arm in config["arms"]:
            handle = (output / f"{arm}.log").open("w", encoding="utf-8")
            handles.append(handle)
            child = subprocess.Popen([sys.executable, "-B", "-m", "tracking_v20.run", "--arm", arm],
                cwd=REPO, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            children.append((arm, child))
        while any(child.poll() is None for _, child in children):
            if any(child.poll() not in (None, 0) for _, child in children):
                raise RuntimeError("Arm failed; preserve all results")
            time.sleep(2)
        if any(child.returncode for _, child in children):
            raise RuntimeError("Arm failed; preserve all results")
        dump(output / "COMPLETE.json", dict(arms=config["arms"], complete=True))
        print(json.dumps(dict(complete=True, stage=config["stage"])), flush=True)
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
