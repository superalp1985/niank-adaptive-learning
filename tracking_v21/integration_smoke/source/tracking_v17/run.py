"""Matched continuous eye experiments, with complete raw SQLite experience."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from tracking_v17.world import World, EYES, DT, combinations
from tracking_v17.learner import Agent, EyeMemory


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class Session:
    def __init__(self, output, mode, seed, worlds=72, training=True, heldout=False,
                 device="cuda", eye_disabled=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.mode, self.training, self.device = mode, training, device
        self.worlds, self.step, self.chunk = worlds, 0, 0
        self.eye_disabled = eye_disabled
        options = combinations(heldout)
        self.envs = [World(seed + 1009 * i, combo=options[i % len(options)],
                          heldout=heldout) for i in range(worlds)]
        if training:
            for env in self.envs:
                env.fixed_combo = None
        self.agent, self.memory = Agent(mode, device), EyeMemory(worlds, mode)
        self.reset_flags = np.ones(worlds, bool)
        self.feature_step, self.saved_features = -1, None
        self.db = sqlite3.connect(self.output / "events.sqlite")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("CREATE TABLE events(body INTEGER,time INTEGER,segment INTEGER,"
                        "rgb BLOB,proprio BLOB,action INTEGER,next_rgb BLOB,"
                        "next_proprio BLOB,retention_gain BLOB,target_mask BLOB,"
                        "PRIMARY KEY(body,time))")
        self.stats, self.curves = {}, []
        self.reacquire_wait = np.full(worlds, -1, np.int32)
        self.ever_locked = np.zeros(worlds, bool)
        self.goal_ids = np.zeros(worlds, np.int32)
        self.previous_lock = np.zeros(worlds, bool)
        self.previous_visible = np.zeros(worlds, bool)
        self.start = time.perf_counter()
        self.latest_loss = None

    def features(self):
        if self.feature_step != self.step:
            measurements = [env.observe() for env in self.envs]
            self.rgb = np.stack([v[0] for v in measurements])
            self.proprio = np.stack([v[1] for v in measurements])
            observed_rgb = np.zeros_like(self.rgb) if self.eye_disabled else self.rgb
            self.saved_features = self.memory.features(observed_rgb, self.proprio,
                                                       self.reset_flags)
            self.reset_flags[:] = False
            self.feature_step = self.step
        return self.saved_features.copy(), self.proprio.copy()

    def advance(self, steps, epochs=3):
        rollout = {k: [] for k in ("features", "proprio", "actions", "groups",
                   "action_logp", "route_logp", "value", "reward", "done", "next_rgb")}
        logs = {k: [] for k in ("combo", "segment", "goal_id", "actions", "groups",
                 "contact", "safe_contact", "collision", "impact", "spawned",
                 "reward", "done", "visible", "locked", "lock_eye", "attention_peak",
                 "reacquired", "reacquire_delay", "pred_error", "omitted_error",
                 "retention_gate", "retention_utility", "progress",
                 "body_position", "target_position", "heading")}
        for _ in range(steps):
            x, proprio = self.features()
            rgb = self.rgb.copy()
            masks = np.stack([env.audit_visibility() for env in self.envs])
            body_position = np.array([tuple(env.body.position) for env in self.envs], np.float32)
            target_position = np.array([tuple(env.target.position) for env in self.envs], np.float32)
            heading = np.array([env.heading for env in self.envs], np.float32)
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
                if env.goal_id != self.goal_ids[i] or self.reset_flags[i]:
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
            if self.training:
                self.memory.feedback(gain)
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
                ("omitted_error", omit_error.mean(1)),
                ("retention_gate", x[:, 9].mean(1)), ("retention_utility", self.memory.utility.mean(1)),
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
        data = {key: np.asarray(value) for key, value in rollout.items()}
        if self.training:
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
                     predictor_loss=self.latest_loss)
        self.curves.append(point)
        dump(self.output / "progress.json", point)
        return point

    def summary(self):
        records = list(self.stats.values())
        total = lambda name: sum(v[name] for v in records)
        attacked = [v for key, v in self.stats.items() if ast.literal_eval(key)[2] != 0]
        attacked_steps = sum(v["steps"] for v in attacked)
        contact = total("contacts")
        seconds = self.step * self.worlds * DT
        per_scene = {}
        for key, v in self.stats.items():
            per_scene[key] = dict(**v,
                contact_per_minute=v["contacts"] * 60 / max(v["steps"] * DT, 1),
                collision_rate=v["collisions"] / max(v["steps"], 1),
                impact_rate=v["impacts"] / max(v["steps"], 1),
                visible_lock=v["locked"] / max(v["visible"], 1))
        return dict(mode=self.mode, eye_disabled=self.eye_disabled, worlds=self.worlds,
            step=self.step, experiences=self.step * self.worlds,
            parameters=self.agent.parameters(), contacts=int(contact),
            safe_contacts=int(total("safe_contacts")),
            safe_contact_fraction=total("safe_contacts") / max(contact, 1),
            contact_per_minute=contact * 60 / max(seconds, 1),
            collision_rate=total("collisions") / max(total("steps"), 1),
            impact_rate=total("impacts") / max(total("steps"), 1),
            attacked_impact_rate=sum(v["impacts"] for v in attacked) / max(attacked_steps, 1),
            attacked_steps=int(attacked_steps), visibility_steps=int(total("visible")),
            visible_lock=total("locked") / max(total("visible"), 1),
            reacquisitions=int(total("reacquisitions")),
            mean_reacquire_seconds=total("reacquire_sum") * DT / max(total("reacquisitions"), 1),
            prediction_mse=total("prediction_sum") / max(total("steps"), 1),
            update_presentations=self.agent.presentations,
            predictor_presentations=self.agent.predictor_presentations,
            local_update_batches=self.agent.updates,
            utility_feedback_steps=self.memory.updates,
            elapsed_seconds=time.perf_counter() - self.start,
            combo_stats=per_scene, curves=self.curves)

    def close(self):
        self.db.commit()
        self.db.close()


def probe(session, output, seed, steps, eye_disabled=False):
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    try:
        trial = Session(output, session.mode, seed, len(combinations(True)),
                        training=False, heldout=True, eye_disabled=eye_disabled)
        trial.agent.load(session.agent.state())
        for _ in range(steps // 128):
            trial.advance(128)
        result = trial.summary()
        dump(trial.output / "summary.json", result)
        trial.close()
        return result
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(gpu)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use the E: training Python")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "pilot.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    output.mkdir(exist_ok=False)
    source_paths = sorted(ROOT.glob("*.py")) + [ROOT / "PROTOCOL.md", ROOT / "pilot.json",
                                              ROOT.parent / "tracking_v14/learner.py"]
    sources = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source_paths}
    for source in source_paths:
        destination = output / "source" / source.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    dump(output / "manifest.json", dict(config=config, source=sources,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    results = []
    for mode in config["modes"]:
        torch.manual_seed(config["model_seed"])
        torch.cuda.manual_seed_all(config["model_seed"])
        session = Session(output / mode, mode, config["world_seed"], config["worlds"])
        torch.save(session.agent.state(), session.output / "initial.pt")
        initial = probe(session, session.output / "probe_0", config["world_seed"] + 900000,
                        config["probe_steps"])
        dump(session.output / "result_0.json", dict(milestone=0, probes=initial))
        for milestone in config["milestones"]:
            while session.step < milestone:
                point = session.advance(min(config["rollout"], milestone - session.step),
                                        config["epochs"])
                if session.chunk % 8 == 0:
                    print(json.dumps(dict(event="progress", mode=mode, **point)), flush=True)
            torch.save(dict(model=session.agent.state(),
                            optimizers=[o.state_dict() for o in session.agent.optimizers],
                            step=session.step, memory=session.memory.__dict__,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
                       session.output / f"checkpoint_{milestone}.pt")
            result = session.summary()
            result["probes"] = probe(session, session.output / f"probe_{milestone}",
                                     config["world_seed"] + 900000, config["probe_steps"])
            dump(session.output / f"result_{milestone}.json", result)
            results.append(result)
            dump(output / "results.json", results)
            print(json.dumps(dict(event="milestone", mode=mode, step=milestone,
                contact=result["probes"]["contact_per_minute"],
                lock=result["probes"]["visible_lock"],
                collision=result["probes"]["collision_rate"],
                impact=result["probes"]["attacked_impact_rate"])), flush=True)
        torch.save(dict(model=session.agent.state()), session.output / "model.pt")
        removal = probe(session, session.output / "probe_no_eyes",
                        config["world_seed"] + 900000, config["probe_steps"], True)
        dump(session.output / "sensor_removal.json", removal)
        session.close()
        del session
        torch.cuda.empty_cache()
    dump(output / "COMPLETE.json", dict(results=len(results), source=sources))


if __name__ == "__main__":
    main()
