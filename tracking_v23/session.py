"""Full-state continuation, with width-safe episode resets and batched error."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from tracking_v22.neural import retina
from tracking_v22.run import Session as LegacySession
from tracking_v23.neural import ActiveLearner
from tracking_v23.persistence import backup_database
from tracking_v23.vision import (
    HeadWorld, compose_input, encode_images, migrate_batch, migrate_input,
)


class Session(LegacySession):
    def __init__(self, output, config, training=True, learner=None):
        self.active_vision = config.get("arm") == "active"
        self.origin_step = 0
        self.origin_metrics = {}
        self.origin_updates = 0
        self.sensor_frames = []
        self.visual_totals = dict(samples=0, entropy=0., attention_l1_change=0.,
                                  head_travel=0., moving_frames=0,
                                  gaze_actions=[0, 0, 0])
        self.previous_attention = None
        if self.active_vision and learner is None:
            learner = ActiveLearner(config["bodies"])
        if learner is not None and not training:
            learner.trial = None
        super().__init__(output, config, training=training, learner=learner)
        if self.active_vision:
            self.worlds = [HeadWorld.upgrade(world) for world in self.worlds]
        self._create_visual_table()

    def _create_visual_table(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS vision(
            step INTEGER, body INTEGER, head_action INTEGER, body_action INTEGER,
            yaw REAL, velocity REAL, entropy REAL, attention_change REAL,
            attention BLOB, PRIMARY KEY(step,body))""")
        self.db.commit()

    @classmethod
    def fork(cls, source, output, config):
        backup_database(source, Path(output)/"events.sqlite")
        session = LegacySession.restore(output)
        return cls.initialize_fork(session, config)

    @classmethod
    def initialize_fork(cls, session, config):
        session.__class__ = cls
        if session.bodies != config["bodies"]:
            session.close()
            raise ValueError("Continuation body count must match the complete source")
        session.config = dict(session.config, **config)
        session.training = session.learning_enabled = True
        session.active_vision = config["arm"] == "active"
        session.origin_step = session.step
        session.origin_metrics = copy.deepcopy(session.metrics)
        session.origin_updates = session.agent.updates
        session.sensor_frames = []
        session.previous_attention = None
        session.visual_totals = dict(samples=0, entropy=0., attention_l1_change=0.,
                                     head_travel=0., moving_frames=0,
                                     gaze_actions=[0, 0, 0])
        session._create_visual_table()
        if session.active_vision:
            ActiveLearner.upgrade(session.agent, seed=config["migration_seed"])
            session.worlds = [HeadWorld.upgrade(world) for world in session.worlds]
            session.x = migrate_input(session.x)
            session.records = [migrate_batch(record) for record in session.records]
            for queue in session.pending:
                for row in queue:
                    row["x"] = migrate_input(row["x"])
        # Observe the exact current scene without advancing state, spikes or RNG.
        observations = [world.observe() for world in session.worlds]
        if session.active_vision:
            _, _, _, head, detail = session._encode(observations)
            legacy = torch.cat((session.x[:, :512], session.x[:, 900:]), 1)
            session.x = compose_input(legacy, head, detail)
        session._capture(observations, force=True)
        return session

    @classmethod
    def restore(cls, path):
        session = super().restore(path)
        session._create_visual_table()
        session.db.execute("DELETE FROM vision WHERE step>?", (session.step,))
        session.db.commit()
        if not session.training:
            session.agent.trial = None
            session.learning_enabled = False
        return session

    def _encode(self, observations):
        rgb = torch.as_tensor(np.stack([row[0] for row in observations]), device="cuda")
        proprio = torch.as_tensor(np.stack([row[1] for row in observations]), device="cuda")
        if self.active_vision:
            head = torch.as_tensor(np.stack([
                row[2] if len(row) > 2 else np.zeros(4, np.float32)
                for row in observations]), device="cuda")
            features, cues, detail = encode_images(rgb, head)
        else:
            features, cues = retina(rgb)
            head = detail = None
        return features, cues, proprio, head, detail

    def _capture(self, observations, force=False):
        relative = self.step-self.origin_step
        if force or self.step == 0:
            for i, observation in enumerate(observations[:4]):
                Image.fromarray(observation[0]).save(self.output/f"actual_input_{i}.png")
        cadence = self.config.get("video_every", 32 if not self.training else 2048)
        if (force or relative % cadence == 0) and len(self.sensor_frames) < 40:
            name = f"sensor_{relative:06d}.png"
            folder = self.output/"sensor_frames"
            folder.mkdir(exist_ok=True)
            Image.fromarray(observations[0][0]).resize((512, 256)).save(folder/name)
            if name not in self.sensor_frames:
                self.sensor_frames.append(name)

    @torch.no_grad()
    def install_observations(self, observations, event_bins=None):
        self.features, self.cues, self.proprio, head, detail = self._encode(observations)
        legacy, self.sensory, self.context = self.agent.perceive(
            self.features, self.proprio, event_bins=event_bins)
        self.x = compose_input(legacy, head, detail) if self.active_vision else legacy
        self._capture(observations)

    @torch.no_grad()
    def _refresh_rows(self, observations, indices):
        """Refresh resets/current source without an additional neural time tick."""
        features, cues, proprio, head, detail = self._encode([observations[i] for i in indices])
        self.features[indices] = features
        self.proprio[indices] = proprio
        for key in self.cues:
            self.cues[key][indices] = cues[key]
        legacy = torch.cat((features.flatten(1), proprio, self.agent.latent[indices],
                            self.agent.spikes.trace[indices]/10.), 1)
        self.x[indices] = (compose_input(legacy, head, detail)
                           if self.active_vision else legacy)
        sensory = torch.cat((features.mean(1), features.amax(1), proprio,
                             self.agent.error[indices]), 1)
        self.sensory[indices] = sensory
        self.agent.last_sensory[indices] = sensory
        self.agent.last_drive[indices] = 1.5+.9*torch.tanh(
            F.linear(sensory, self.agent.spike_weights))
        self.context = self.agent.make_context(self.proprio)

    def recall(self, context):
        batch = super().recall(context)
        return migrate_batch(batch) if self.active_vision and batch is not None else batch

    def _log_visual(self):
        if not self.active_vision:
            return
        with torch.no_grad():
            attention = self.agent.model.attention(self.x)
            entropy = -(attention*attention.clamp_min(1e-12).log()).sum(1)
            change = (torch.zeros(self.bodies, device="cuda")
                      if self.previous_attention is None else
                      (attention-self.previous_attention).abs().sum(1))
            self.previous_attention = attention.clone()
        distributions = attention.cpu().numpy().astype(np.float16)
        entropy, change = entropy.cpu().tolist(), change.cpu().tolist()
        self.visual_totals["samples"] += self.bodies
        self.visual_totals["entropy"] += sum(entropy)
        self.visual_totals["attention_l1_change"] += sum(change)
        for i, world in enumerate(self.worlds):
            gaze, body = divmod(int(self.current_actions[i]), 9)
            self.visual_totals["gaze_actions"][gaze] += 1
            self.db.execute("INSERT INTO vision VALUES(?,?,?,?,?,?,?,?,?)",
                            (self.step+1, i, gaze, body, world.head_yaw,
                             world.head_velocity, entropy[i], change[i],
                             distributions[i].tobytes()))

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
                if self.active_vision and self.config.get("head_hold", False):
                    self.current_actions %= 9
                self._log_visual()
            for i in range(self.bodies):
                self.pending[i].append(dict(
                    start=self.step, x=self.x[i].clone(), a=int(self.current_actions[i]),
                    sensory=self.sensory[i].clone(), latent=old_latent[i].clone(),
                    trace=self.agent.pretrace[i].clone(), before=old_prop[i].clone(),
                    episode=self.worlds[i].segment, collision=0., impact=0., contact=0.))
            results = list(self.executor.map(
                lambda item: item[0].step(int(item[1])), zip(self.worlds, self.current_actions)))
            self.step += 1
            observations = [row[0] for row in results]
            bins = np.zeros((self.bodies, 10), np.float32)
            has_events = False
            for i, (_, _, done, info) in enumerate(results):
                self.metrics["frames"] += 1
                for key in ("contact", "safe_contact", "collision", "impact", "spawned"):
                    self.metrics[key] += int(info[key])
                self.metrics["progress"] += info["progress"]
                self.metrics["collision_seconds"] += info["collision_duration"]
                self.metrics["impact_events"] += info["impact_count"]
                if self.active_vision:
                    self.visual_totals["head_travel"] += info["head_travel"]
                    self.visual_totals["moving_frames"] += int(abs(info["head_velocity"]) > 1e-9)
                for event in info.get("sensor_events", []):
                    within = event["timestamp"]-(info["timestamp"]-.01)
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
                # The old loop evaluates every matured row, but only the last
                # error per body survives. Batch those exact surviving rows.
                error_rows = []
                for i, (_, _, done, _) in enumerate(results):
                    last = None
                    while self.pending[i] and (
                            self.step-self.pending[i][0]["start"] >= self.config["horizon"] or done):
                        pending = self.pending[i].popleft()
                        target = current_outcome[i].clone()
                        target[4:7] = target.new_tensor(
                            [pending["collision"], pending["impact"], pending["contact"]])
                        target[7:9] = self.proprio[i, 5:7]-pending["before"][5:7]
                        self.records.append((pending["x"],
                            torch.tensor(pending["a"], device="cuda"), target,
                            pending["sensory"], pending["latent"], pending["trace"]))
                        self.record_windows.append((pending["start"], self.step,
                                                    i, pending["episode"]))
                        last = (i, pending["x"], pending["a"], target)
                    if last is not None and not done:
                        error_rows.append(last)
                    if done:
                        self.metrics["deaths"] += 1
                        self.worlds[i].reset()
                        self.agent.latent[i] = 0
                        self.agent.spikes.reset([i])
                        self.agent.pretrace[i] = 0
                        self.agent.error[i] = 0
                        observations[i] = self.worlds[i].observe()
                if error_rows:
                    prediction = self.agent.model(torch.stack([row[1] for row in error_rows]))
                    index = torch.arange(len(error_rows), device="cuda")
                    actions = torch.tensor([row[2] for row in error_rows], device="cuda")
                    predicted = prediction[:, index, actions].mean(0)
                    target = torch.stack([row[3] for row in error_rows])
                    error = F.smooth_l1_loss(predicted, target, reduction="none").mean(1).clamp(0, 1)
                    self.agent.error[[row[0] for row in error_rows], 0] = error
                done_indices = [i for i, result in enumerate(results) if result[2]]
                if done_indices:
                    self._refresh_rows(observations, done_indices)
            if self.step % self.config["block"] == 0:
                self.finish_block()
        return self.summary()

    def summary(self):
        result = super().summary()
        fresh = {key: value-self.origin_metrics.get(key, 0)
                 for key, value in self.metrics.items()}
        fresh_steps = self.step-self.origin_step
        fresh.update(steps=fresh_steps, updates=self.agent.updates-self.origin_updates,
                     contacts_per_body_minute=fresh["contact"]/max(fresh_steps*.01*self.bodies/60, 1e-9),
                     collision_fraction=fresh["collision"]/max(fresh["frames"], 1))
        totals = self.visual_totals
        result.update(origin_step=self.origin_step, new_steps=fresh_steps,
                      continuation=fresh,
                      vision=dict(totals,
                          mean_entropy=totals["entropy"]/max(totals["samples"], 1),
                          mean_attention_l1_change=totals["attention_l1_change"]/max(totals["samples"], 1),
                          motion_fraction=totals["moving_frames"]/max(fresh["frames"], 1)))
        return result

    def save_gif(self):
        images = []
        try:
            for name in self.sensor_frames:
                with Image.open(self.output/"sensor_frames"/name) as source:
                    images.append(source.copy())
            if images:
                images[0].save(self.output/"actual_sensor.gif", save_all=True,
                               append_images=images[1:], duration=200, loop=0)
        finally:
            for image in images:
                image.close()
