"""Finite, recoverable parallel practice with independently scored learning."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sqlite3
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from tracking_v22.neural import Learner, retina
from tracking_v22.world import World, DT, combinations, canonical_checkpoint

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pack_batch(batch):
    buffer = io.BytesIO()
    arrays = [t.detach().cpu().numpy().astype(np.int64 if i == 1 else np.float16)
              for i, t in enumerate(batch)]
    np.savez_compressed(buffer, **{f"a{i}": array for i, array in enumerate(arrays)})
    return buffer.getvalue()


def unpack_batch(blob, device):
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return tuple(torch.as_tensor(data[f"a{i}"].copy(), device=device,
                                     dtype=torch.long if i == 1 else torch.float32)
                     for i in range(6))


def stack_records(records):
    return tuple(torch.stack([record[i] for record in records]) for i in range(6))


class Session:
    def __init__(self, output, config, training=True, learner=None):
        self.output, self.config, self.training = Path(output), config, training
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "events.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS facts(
              block INTEGER PRIMARY KEY, step INTEGER, priority REAL, context BLOB, batch BLOB);
            CREATE TABLE IF NOT EXISTS events(
              step INTEGER, body INTEGER, timestamp REAL, facts TEXT);
            CREATE TABLE IF NOT EXISTS choices(
              block INTEGER PRIMARY KEY, step INTEGER, mode INTEGER, outcome TEXT);
            CREATE TABLE IF NOT EXISTS runtime(
              id INTEGER PRIMARY KEY, step INTEGER, sha256 TEXT, payload BLOB);
        """)
        self.bodies = config["bodies"]
        self.learning_enabled = training
        self.agent = learner or Learner(self.bodies, variant=config["variant"])
        options = combinations(not training)
        self.worlds = [World(config["world_seed"]+i*1009,
                             combo=options[(i*13) % len(options)], heldout=not training)
                       for i in range(self.bodies)]
        self.executor = ThreadPoolExecutor(max_workers=min(4, self.bodies))
        self.step = self.block = 0
        self.pending = [deque() for _ in self.worlds]
        self.records = []
        self.record_windows = []
        self.curve = []
        self.choices = [0, 0, 0]
        self.memory_recalls = self.cold_recalls = 0
        self.metrics = dict(contact=0, safe_contact=0, collision=0, impact=0,
                            spawned=0, frames=0, deaths=0, progress=0.)
        self.start = time.perf_counter()
        self.elapsed_before = 0.
        self.current_actions = np.zeros(self.bodies, np.int64)
        self.pending_credit = {}
        self.latest_context = None
        self.last_training = {}
        self.loss_sum = self.loss_count = 0
        self.last_fact_id = -1
        self.action_decisions = 0
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(facts)")}
        if "windows" not in columns:
            self.db.execute("ALTER TABLE facts ADD COLUMN windows BLOB")
        observations = [w.observe() for w in self.worlds]
        self.install_observations(observations)

    def install_observations(self, observations, event_bins=None):
        rgb = np.stack([o[0] for o in observations])
        self.proprio = torch.as_tensor(np.stack([o[1] for o in observations]), device="cuda")
        with torch.no_grad():
            self.features, self.cues = retina(torch.as_tensor(rgb, device="cuda"))
            self.x, self.sensory, self.context = self.agent.perceive(
                self.features, self.proprio, event_bins=event_bins)
        if self.step == 0:
            for i in range(min(4, self.bodies)):
                Image.fromarray(rgb[i]).save(self.output / f"actual_input_{i}.png")

    def recall(self, context):
        rows = self.db.execute("SELECT block,priority,context FROM facts ORDER BY block DESC").fetchall()
        if not rows:
            return None
        # Relevance and experienced surprise influence access, not permanent deletion.
        ctx = context.detach().cpu().numpy().reshape(-1)
        scores = []
        for block, priority, blob in rows:
            old = np.frombuffer(blob, np.float32)
            similarity = float(ctx @ old / max(np.linalg.norm(ctx)*np.linalg.norm(old), 1e-6))
            scores.append(similarity + .2*np.log1p(priority) + np.random.gumbel(scale=.15))
        index = int(np.argmax(scores))
        block = rows[index][0]
        self.memory_recalls += 1
        self.cold_recalls += int(self.block-block > 8)
        blob = self.db.execute("SELECT batch FROM facts WHERE block=?", (block,)).fetchone()[0]
        return unpack_batch(blob, self.agent.device)

    def finish_block(self):
        if not self.records:
            return
        batch = stack_records(self.records)
        with torch.no_grad():
            predictions = self.agent.model(batch[0])
            selected = predictions[:, torch.arange(len(batch[0]), device="cuda"), batch[1]]
            error = F.smooth_l1_loss(selected, batch[2][None].expand_as(selected))
        windows = torch.tensor(self.record_windows, device="cuda")
        credit = self.agent.settle_trial(batch, windows=windows) if self.training else {}
        if credit:
            self.db.execute("INSERT INTO choices VALUES(?,?,?,?)",
                            (self.block-1, self.step, credit["mode"], json.dumps(credit)))
        mean_context = self.context.mean(0, keepdim=True)
        self.db.execute("INSERT INTO facts VALUES(?,?,?,?,?,?)",
                        (self.block, self.step, float(error),
                         mean_context.detach().cpu().numpy().astype(np.float32).tobytes(),
                         pack_batch(batch), np.array(self.record_windows, np.int64).tobytes()))
        if self.learning_enabled and self.agent.trial is None:
            mode = self.agent.select_mode(mean_context)
            self.choices[mode] += 1
            self.agent.begin_trial(mean_context, mode, decision_step=self.step)
            train_data = self.recall(mean_context) if mode == 2 else batch
            self.last_training = self.agent.train_batch(train_data, mode, epochs=2 if mode == 2 else 1)
        point = dict(step=self.step, block=self.block,
                     simulation_seconds=self.step*DT*self.bodies,
                     elapsed_seconds=self.elapsed_before+time.perf_counter()-self.start,
                     prequential_loss=float(error), updates=self.agent.updates,
                     choices=self.choices.copy(), recalls=self.memory_recalls,
                     cold_recalls=self.cold_recalls,
                     parameter_l2_change=self.agent.parameter_change(),
                     state_parameter_l2_change=self.agent.state_parameter_change(),
                     spike_parameter_l2_change=self.agent.spike_parameter_change(),
                     action_decisions=self.action_decisions,
                     state_updates=self.agent.state_updates,
                     meta_updates=self.agent.meta_updates,
                     local_updates=self.agent.local_updates,
                     training_losses=self.last_training,
                     state_norm=float(self.agent.latent.norm(dim=1).mean()),
                     spike_rate_hz=float(self.agent.spikes.total_spikes.sum()) /
                     max(self.step*DT*self.bodies*self.agent.spike_dim, .001),
                     meta_credit=credit, **self.metrics)
        self.curve.append(point)
        self.records = []
        self.record_windows = []
        self.block += 1
        self.db.commit()
        dump(self.output / "progress.json", point)
        if self.block % 4 == 0:
            print(json.dumps(dict(event="progress", variant=self.config["variant"],
                                  **point)), flush=True)

    def advance(self, count):
        for _ in range(count):
            old_prop = self.proprio.clone()
            old_latent = self.agent.latent.clone()
            if self.step % self.config["action_every"] == 0:
                self.action_decisions += self.bodies
                with torch.no_grad():
                    actions, _, _ = self.agent.act(
                        self.x, self.cues, self.proprio, self.step, self.training,
                        prior_gain=self.config.get("prior_gain", 1.))
                self.current_actions = actions.cpu().numpy()
            for i in range(self.bodies):
                self.pending[i].append(dict(
                    start=self.step, x=self.x[i].clone(), a=int(self.current_actions[i]),
                    sensory=self.sensory[i].clone(), latent=old_latent[i].clone(),
                    trace=self.agent.pretrace[i].clone(), before=old_prop[i].clone(),
                    episode=self.worlds[i].segment, collision=0., impact=0., contact=0.))
            results = list(self.executor.map(
                lambda item: item[0].step(int(item[1])), zip(self.worlds, self.current_actions)))
            self.step += 1
            observations = [r[0] for r in results]
            bins = np.zeros((self.bodies, 10), np.float32)
            has_events = False
            for i, (_, _, done, info) in enumerate(results):
                self.metrics["frames"] += 1
                for key in ("contact", "safe_contact", "collision", "impact", "spawned"):
                    self.metrics[key] += int(info[key])
                self.metrics["progress"] += info["progress"]
                for event in info.get("sensor_events", []):
                    # Relative to this world's current 10ms interval, including resets.
                    within = event["timestamp"] - (info["timestamp"]-DT)
                    tick = min(9, max(0, int(round(within/.001))-1))
                    bins[i, tick] += 1 if event["kind"] != "impact" else 2
                    has_events = True
                self.db.execute("INSERT INTO events VALUES(?,?,?,?)",
                                (self.step, i, info["timestamp"], json.dumps(info)))
                for pending in self.pending[i]:
                    for key in ("collision", "impact", "contact"):
                        pending[key] = max(pending[key], float(info[key]))
            self.install_observations(observations, torch.as_tensor(bins, device="cuda")
                                      if has_events else None)
            with torch.no_grad():
                current_outcome = self.agent.outcome(self.cues, old_prop, self.proprio)
                for i, (_, _, done, info) in enumerate(results):
                    while self.pending[i] and (
                        self.step-self.pending[i][0]["start"] >= self.config["horizon"] or done):
                        pending = self.pending[i].popleft()
                        target = current_outcome[i].clone()
                        target[4:7] = torch.tensor(
                            [pending["collision"], pending["impact"], pending["contact"]],
                            device="cuda")
                        target[7:9] = self.proprio[i, 5:7] - pending["before"][5:7]
                        self.records.append((pending["x"],
                            torch.tensor(pending["a"], device="cuda"), target,
                            pending["sensory"], pending["latent"], pending["trace"]))
                        self.record_windows.append((pending["start"], self.step,
                                                    i, pending["episode"]))
                        predicted = self.agent.model(pending["x"][None])[:, 0, pending["a"]].mean(0)
                        self.agent.error[i] = F.smooth_l1_loss(predicted, target).clamp(0, 1)
                    if done:
                        self.metrics["deaths"] += 1
                        self.worlds[i].reset()
                        self.agent.latent[i] = 0
                        self.agent.spikes.reset([i])
                        self.agent.pretrace[i] = 0
                        self.agent.error[i] = 0
                        observations[i] = self.worlds[i].observe()
            # Death is a new episode; refresh only the reset rows without extra clock ticks.
            done_indices = [i for i, r in enumerate(results) if r[2]]
            if done_indices:
                rgb = torch.as_tensor(np.stack([observations[i][0] for i in done_indices]),
                                      device="cuda")
                new_features, new_cues = retina(rgb)
                self.features[done_indices] = new_features
                for key in self.cues:
                    self.cues[key][done_indices] = new_cues[key]
                self.proprio[done_indices] = torch.as_tensor(
                    np.stack([observations[i][1] for i in done_indices]), device="cuda")
                self.x[done_indices] = torch.cat((new_features.flatten(1),
                    self.proprio[done_indices], self.agent.latent[done_indices],
                    self.agent.spikes.trace[done_indices]/10.), 1)
                sensory = torch.cat((new_features.mean(1), new_features.amax(1),
                                      self.proprio[done_indices], self.agent.error[done_indices]), 1)
                self.sensory[done_indices] = sensory
                self.agent.last_sensory[done_indices] = sensory
                self.agent.last_drive[done_indices] = 1.5 + .9*torch.tanh(
                    F.linear(sensory, self.agent.spike_weights))
                self.context = self.agent.make_context(self.proprio)
            if self.step % self.config["block"] == 0:
                self.finish_block()
        return self.summary()

    def summary(self):
        return dict(**self.metrics, step=self.step, bodies=self.bodies,
                    simulation_seconds=self.step*DT*self.bodies,
                    contacts_per_body_minute=self.metrics["contact"]/
                    max(self.step*DT*self.bodies/60, 1e-9),
                    collision_fraction=self.metrics["collision"]/max(self.metrics["frames"], 1),
                    impacts_per_body_minute=self.metrics["impact"]/
                    max(self.step*DT*self.bodies/60, 1e-9),
                    updates=self.agent.updates, choices=self.choices,
                    memory_recalls=self.memory_recalls, cold_recalls=self.cold_recalls,
                    parameter_l2_change=self.agent.parameter_change(),
                    state_parameter_l2_change=self.agent.state_parameter_change(),
                    spike_parameter_l2_change=self.agent.spike_parameter_change(),
                    action_decisions=self.action_decisions,
                    state_updates=self.agent.state_updates, meta_updates=self.agent.meta_updates,
                    local_updates=self.agent.local_updates,
                    unsettled_learning_choices=int(self.agent.trial is not None),
                    neural_ticks=self.agent.spikes.ticks, curve=self.curve,
                    elapsed_seconds=self.elapsed_before+time.perf_counter()-self.start)

    def checkpoint(self):
        world_blob = canonical_checkpoint(self.worlds)
        state = {key: value for key, value in vars(self).items()
                 if key not in ("db", "executor", "worlds", "start")}
        state["elapsed_before"] = self.elapsed_before+time.perf_counter()-self.start
        payload = dict(state=state, world_blob=world_blob,
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                       numpy_rng=np.random.get_state(), python_rng=random.getstate())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        self.db.execute("INSERT OR REPLACE INTO runtime VALUES(1,?,?,?)",
                        (self.step, hashlib.sha256(data).hexdigest(), data))
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    @classmethod
    def restore(cls, path):
        path = Path(path)
        db = sqlite3.connect(path / "events.sqlite")
        row = db.execute("SELECT step,sha256,payload FROM runtime WHERE id=1").fetchone()
        if row is None or hashlib.sha256(row[2]).hexdigest() != row[1]:
            raise RuntimeError("No complete hash-verified checkpoint")
        saved = torch.load(io.BytesIO(row[2]), weights_only=False)
        session = cls.__new__(cls)
        vars(session).update(saved["state"])
        session.output, session.db = path, db
        session.worlds = pickle.loads(saved["world_blob"])
        session.executor = ThreadPoolExecutor(max_workers=min(4, session.bodies))
        session.start = time.perf_counter()
        # Rows after the committed runtime belong to an interrupted partial continuation.
        db.execute("DELETE FROM events WHERE step>?", (session.step,))
        db.execute("DELETE FROM facts WHERE step>?", (session.step,))
        db.execute("DELETE FROM choices WHERE step>?", (session.step,))
        db.commit()
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        return session

    def close(self):
        self.executor.shutdown()
        self.db.commit()
        self.db.close()


def source_snapshot(path, config):
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + [
        REPO/"tracking_v17/world.py", REPO/"SUBJECTIVE_STATE_DESIGN_NOTES_ZH.md"]
    manifest = {}
    for source in paths:
        relative = source.relative_to(REPO)
        dest = path/"source"/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    dump(path/"manifest.json", dict(config=config, sources=manifest,
                                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                                    sensory="native_1024x512_100Hz",
                                    neural_dt=.0001, physics_dt=.001,
                                    state_checkpoint="explicit_solver_cache_canonicalization"))


def run(config, output):
    output = Path(output)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    if (output/"COMPLETE.json").exists():
        raise RuntimeError("Completed configuration must not be rerun")
    if (output/"events.sqlite").exists():
        session = Session.restore(output)
        if session.config != config:
            raise ValueError("Cannot resume with changed configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        source_snapshot(output, config)
        session = Session(output, config)
        session.checkpoint()
    try:
        while session.step < config["steps"]:
            session.advance(min(config["checkpoint_every"], config["steps"]-session.step))
            session.checkpoint()
        session.learning_enabled = False
        settle_end = config["steps"] + config.get("settling_steps", config["block"])
        if session.step < settle_end:
            session.advance(settle_end-session.step)
            if session.records:
                session.finish_block()
            session.checkpoint()
        summary = session.summary()
        dump(output/"summary.json", summary)
        torch.save(session.agent, output/"learner.pt")
        # Frozen probes use a separate state, RNG stream and no parameter changes.
        probe_config = dict(config, world_seed=config["world_seed"]+700000)
        probe_agent = copy.deepcopy(session.agent)
        probe_agent.trial = None
        probe_agent.latent.zero_()
        probe_agent.pretrace.zero_()
        probe_agent.error.zero_()
        probe_agent.last_sensory = probe_agent.last_drive = None
        probe_agent.spikes = type(probe_agent.spikes)(session.bodies, probe_agent.spike_dim,
                                                     device="cuda")
        probe_path = output/"frozen_probe"
        if (probe_path/"events.sqlite").exists():
            db = sqlite3.connect(probe_path/"events.sqlite")
            has_checkpoint = db.execute("SELECT count(*) FROM runtime").fetchone()[0] > 0
            db.close()
            if has_checkpoint:
                trial = Session.restore(probe_path)
            else:
                # Preserve interrupted evidence; begin a named fresh attempt.
                prior = output/f"interrupted_probe_{time.time_ns()}"
                probe_path.rename(prior)
                trial = Session(probe_path, probe_config, training=False, learner=probe_agent)
                trial.checkpoint()
        else:
            trial = Session(probe_path, probe_config, training=False, learner=probe_agent)
            trial.checkpoint()
        try:
            while trial.step < config["probe_steps"]:
                trial.advance(min(config["checkpoint_every"], config["probe_steps"]-trial.step))
                trial.checkpoint()
            probe = trial.summary()
            dump(trial.output/"summary.json", probe)
        finally:
            trial.close()
        dump(output/"COMPLETE.json", dict(training=summary, frozen_probe=probe))
        print(json.dumps(dict(event="complete", output=str(output),
                              train={k:v for k,v in summary.items() if k!="curve"},
                              probe={k:v for k,v in probe.items() if k!="curve"})), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output)
