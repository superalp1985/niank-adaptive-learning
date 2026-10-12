"""Continuous physical events and revisable, learned resource allocation."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[key] = str(CACHE)
sys.path.insert(0, str(ROOT.parent))

import sqlite3
import zlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from retention_v5.experiment import physics, TOLERANCE

OPS = ("keep", "retrieve", "relearn", "compute", "admit", "fade", "retire")
SLOTS, CANDIDATES = 3, 8
MODES = ("independent", "immediate", "delayed")
MOMENTA = np.array((0.2, 0.45, 0.7, 0.9), np.float32)
FORCES = np.array((-0.24, -0.08, 0.08, 0.24), np.float32)


class Stream:
    """Simulator truth stays here; learners see images, body, executed outcomes."""

    def __init__(self, count, seed, heldout=False):
        self.count, self.rng, self.step = count, np.random.default_rng(seed), 0
        self.pairs = np.array([
            (m, f) for m in range(4) for f in range(4)
            if (((m, f) in ((0, 0), (3, 3))) == heldout)
        ])
        self.body = np.zeros((count, 5), np.float32)
        self.body[:, 0] = self.rng.uniform(0.2, 0.8, count)
        self.body[:, 1] = self.rng.uniform(-0.3, 0.3, count)
        self.body[:, 2] = self.rng.uniform(0.1, 0.9, count)
        self.body[:, 3] = self.rng.uniform(0.4, 0.7, count)
        self.body[:, 4] = self.rng.choice((-1., 1.), count)
        self.active = self.pairs[self.rng.integers(len(self.pairs), size=count)].copy()
        self.pending = self.active.copy()
        self.context = np.zeros(count, np.float32)
        self.observed_step = -1

    def observe(self):
        if self.observed_step == self.step:
            raise RuntimeError("Each state may be observed only once")
        self.observed_step = self.step
        phase = self.step % 32
        if phase == 0:
            self.pending = self.pairs[self.rng.integers(len(self.pairs), size=self.count)].copy()
            # A visible joint change takes effect after both preceding flashes.
            if self.step == 0:
                self.active = self.pending.copy()
            self.body[:, 2] = self.rng.uniform(0.1, 0.9, self.count)
            self.body[:, 3] = self.rng.uniform(0.4, 0.7, self.count)
        if phase == 8:
            self.active = self.pending.copy()
        self.context = np.full(self.count, (0.15, 0.55, 1., 0.35)[phase // 8], np.float32)
        image = np.zeros((self.count, 3, 6, 6), np.int8)
        if self.step % 4 == 0:
            image[:, 2] = self.rng.integers(-12, 13, (self.count, 6, 6), dtype=np.int8)
            if phase == 0:
                image[:, 0, 1:4, 1:4] = ((self.pending[:, 0] * 2 - 3) * 24)[:, None, None]
            if phase == 4:
                image[:, 1, 2, 2:4] = ((self.pending[:, 1] * 2 - 3) * 24)[:, None]
        sensor = np.concatenate((
            self.body.copy(), self.context[:, None],
            np.full((self.count, 1), np.sin(2 * np.pi * phase / 32), np.float32),
            np.full((self.count, 1), np.cos(2 * np.pi * phase / 32), np.float32),
        ), 1)
        return image, sensor

    def execute(self, actions):
        if self.observed_step != self.step:
            raise RuntimeError("Observe before executing")
        states, contacts, costs = physics(
            self.body, MOMENTA[self.active[:, 0]], FORCES[self.active[:, 1]], self.context,
        )
        rows = np.arange(self.count)
        actual = states[rows, actions].copy()
        contact = contacts[rows, actions].copy()
        actual_cost = costs[rows, actions].copy()
        regret = actual_cost - costs.min(1)
        self.body[:, :2] = actual
        self.step += 1
        # regret is evaluator-only; it must never train either network.
        return actual, contact, actual_cost, regret


class Archive:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, world INTEGER, time INTEGER, image BLOB)")
        self.reads = 0

    def append(self, step, images):
        ids = step * len(images) + np.arange(len(images))
        self.db.executemany("INSERT INTO events VALUES(?,?,?,?)", (
            (int(ids[i]), i, step, zlib.compress(image.tobytes(), 1))
            for i, image in enumerate(images)
        ))
        return ids

    def get(self, identity):
        row = self.db.execute("SELECT image FROM events WHERE id=?", (int(identity),)).fetchone()
        if row is None:
            raise KeyError(identity)
        self.reads += 1
        return np.frombuffer(zlib.decompress(row[0]), np.int8).reshape(3, 6, 6).copy()

    def close(self):
        self.db.commit()
        self.db.close()


class Memory:
    def __init__(self, count):
        self.count = count
        self.ids = np.full((count, CANDIDATES), -1, np.int64)
        self.times = np.zeros((count, CANDIDATES), np.int64)
        self.live = np.zeros((count, CANDIDATES), bool)
        self.cache = np.zeros((count, CANDIDATES, 3, 6, 6), np.int8)
        self.faded = np.full((count, CANDIDATES), -1, np.int64)
        self.accumulated_loss = np.zeros((count, CANDIDATES), np.float32)
        self.replays = np.zeros((count, CANDIDATES), np.float32)
        self.recovery = np.zeros((count, 4), np.float32)
        self.returns = self.fade_count = self.censored = 0

    def _fade(self, row, slot, step):
        self.live[row, slot] = False
        self.cache[row, slot] = 0
        self.faded[row, slot] = step
        self.accumulated_loss[row, slot] = 0
        self.replays[row, slot] = 0
        self.fade_count += 1

    def append(self, step, images, ids):
        for row in range(self.count):
            slot = (step // 4) % CANDIDATES
            if self.faded[row, slot] >= 0:
                self.censored += 1
            self.live[row, slot] = False
            if self.live[row].sum() >= SLOTS:
                old = np.flatnonzero(self.live[row])[np.argmin(self.times[row, self.live[row]])]
                self._fade(row, old, step)
            self.ids[row, slot], self.times[row, slot] = ids[row], step
            self.cache[row, slot] = images[row]
            self.live[row, slot] = True
            self.faded[row, slot] = -1
            self.accumulated_loss[row, slot] = self.replays[row, slot] = 0

    def apply(self, ops, pointers, step, archive):
        for row in range(self.count):
            slot = pointers[row]
            if ops[row] == 5:
                assert self.live[row, slot]
                self._fade(row, slot, step)
            elif ops[row] == 1:
                assert self.ids[row, slot] >= 0 and not self.live[row, slot]
                raw = archive.get(self.ids[row, slot])
                if self.faded[row, slot] >= 0:
                    evidence = np.array((
                        (step - self.faded[row, slot]) / 32,
                        self.accumulated_loss[row, slot],
                        self.replays[row, slot] / 8, 1.,
                    ), np.float32)
                    self.recovery[row] = 0.85 * self.recovery[row] + 0.15 * evidence
                    self.returns += 1
                if self.live[row].sum() >= SLOTS:
                    old = np.flatnonzero(self.live[row])[np.argmin(self.times[row, self.live[row]])]
                    self._fade(row, old, step)
                self.cache[row, slot] = raw
                self.live[row, slot] = True
                self.faded[row, slot] = -1

    def feedback(self, loss, replayed):
        missing = self.faded >= 0
        self.accumulated_loss += missing * loss[:, None]
        self.replays += missing * replayed[:, None]
        self.recovery *= 0.998

    def tensors(self, step, image, sensor, device="cuda"):
        indices = np.argsort(np.where(self.live, self.times, -100000), axis=1)[:, -SLOTS:]
        rows = np.arange(self.count)[:, None]
        live = self.live[rows, indices]
        images = self.cache[rows, indices].copy()
        images[~live] = 0
        ages = np.where(live, (step - self.times[rows, indices]) / 32, 0).astype(np.float32)
        return {
            "images": torch.as_tensor(np.concatenate((image[:, None], images), 1), device=device).float() / 96,
            "sensor": torch.as_tensor(sensor, device=device),
            "ages": torch.as_tensor(ages, device=device),
            "mask": torch.as_tensor(live, device=device).float(),
        }

    def metadata(self, step):
        return np.stack((
            np.clip((step - self.times) / 32, 0, 10),
            self.live.astype(np.float32),
            (self.ids >= 0).astype(np.float32),
        ), -1).astype(np.float32)


class Predictor(nn.Module):
    """Routed hidden-unit groups of two matrices, not whole-layer skipping."""

    def __init__(self, groups):
        super().__init__()
        self.groups, self.width = groups, 96
        self.visual = nn.Sequential(nn.Linear(108, 32), nn.SiLU())
        self.combine = nn.Sequential(nn.Linear(4 * 32 + 8 + 2 * SLOTS, 64), nn.SiLU())
        self.router = nn.Linear(64, groups)
        self.w1 = nn.Parameter(torch.randn(groups, 64, self.width) * 0.06)
        self.b1 = nn.Parameter(torch.zeros(groups, self.width))
        self.w2 = nn.Parameter(torch.randn(groups, self.width, 64) * 0.04)
        self.b2 = nn.Parameter(torch.zeros(groups, 64))
        self.head = nn.Linear(64, 9)

    def forward(self, batch, admitted, boost):
        x = self.visual(batch["images"].flatten(2)).flatten(1)
        x = self.combine(torch.cat((x, batch["sensor"], batch["ages"], batch["mask"]), 1))
        allowed = torch.arange(self.groups, device=x.device)[None] < admitted[:, None]
        logits = self.router(x).masked_fill(~allowed, -1e9)
        values, indices = logits.topk(2, dim=1)
        count = 1 + boost.long()
        weights = values.masked_fill(torch.arange(2, device=x.device)[None] >= count[:, None], -1e9).softmax(1)
        hidden = torch.zeros_like(x)
        # Only selected rows perform the second group's matrix multiplications.
        for slot in range(2):
            selected = torch.nonzero(count > slot, as_tuple=True)[0]
            if not len(selected):
                continue
            group = indices[selected, slot]
            first = torch.bmm(x[selected, None], self.w1[group]).squeeze(1) + self.b1[group]
            second = torch.bmm(F.silu(first)[:, None], self.w2[group]).squeeze(1) + self.b2[group]
            hidden.index_add_(0, selected, second * weights[selected, slot:slot + 1])
        # Single-choice routing uses a straight-through router factor.
        if self.training:
            selected_probability = logits.softmax(1).gather(1, indices[:, :1])
            factor = 1 + selected_probability - selected_probability.detach()
            hidden = hidden * factor
        output = self.head(F.silu(x + hidden))
        return output[:, :6].reshape(-1, 3, 2), output[:, 6:], indices

    def forward_macs(self, batch, boosts):
        shared = 4 * 108 * 32 + (4 * 32 + 8 + 2 * SLOTS) * 64 + 64 * self.groups + 64 * 9
        return batch * shared + (batch + int(boosts)) * (2 * 64 * self.width)

    def active_parameters(self, boost):
        shared = sum(p.numel() for name, p in self.named_parameters() if name not in ("w1", "w2", "b1", "b2"))
        return shared + (1 + np.asarray(boost)) * (2 * 64 * self.width + self.width + 64)


def loss_for(states, contacts, actions, actual, contact):
    rows = torch.arange(len(actions), device=actions.device)
    error = (states[rows, actions] - actual).square().mean(-1)
    loss = 4 * error + 0.2 * F.binary_cross_entropy_with_logits(contacts[rows, actions], contact, reduction="none")
    return loss, error


def choose(states, contacts, sensor):
    actions = states.new_tensor((-1., 0., 1.))
    cost = ((states[:, :, 0].clamp(0, 1) - sensor[:, 2:3]).square()
            + 0.025 * states[:, :, 1].square() + 0.005 * actions.square()
            + 0.6 * contacts.sigmoid())
    return cost.argmin(1)


class Controller(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "independent":
            self.memory = nn.Sequential(nn.Linear(12, 48), nn.Tanh(), nn.Linear(48, 5))
            self.capacity = nn.Sequential(nn.Linear(10, 48), nn.Tanh(), nn.Linear(48, 5))
        else:
            self.joint = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 8))
        self.pointer = nn.Sequential(nn.Linear(23, 24), nn.Tanh(), nn.Linear(24, 1))

    def distributions(self, state, metadata, valid):
        if self.mode != "delayed":
            state = state.clone()
            state[:, 16:] = 0
        if self.mode == "independent":
            mem = self.memory(state[:, :12])
            cap = self.capacity(torch.cat((state[:, :6], state[:, 12:16]), 1))
            # Two independent policies with fixed equal arbitration.
            a = mem[:, :4].log_softmax(1) - np.log(2)
            b = cap[:, :4].log_softmax(1) - np.log(2)
            logits = torch.stack((
                torch.logaddexp(a[:, 0], b[:, 0]), a[:, 1], a[:, 2],
                b[:, 1], b[:, 2], a[:, 3], b[:, 3],
            ), 1)
            value = (mem[:, 4] + cap[:, 4]) / 2
        else:
            out = self.joint(state)
            logits, value = out[:, :7], out[:, 7]
        logits = logits.masked_fill(~valid, -1e9)
        probs = 0.95 * logits.softmax(1) + 0.05 * valid.float() / valid.sum(1, keepdim=True)
        ptr_state = state
        if self.mode == "independent":
            ptr_state = state.clone()
            ptr_state[:, 12:] = 0
        pointers = self.pointer(torch.cat((ptr_state[:, None].expand(-1, CANDIDATES, -1), metadata), -1)).squeeze(-1)
        return torch.distributions.Categorical(probs=probs), value, pointers

    def act(self, state, metadata, valid, retrieve_mask, fade_mask, greedy=False):
        distribution, value, pointers = self.distributions(state, metadata, valid)
        op = distribution.probs.argmax(1) if greedy else distribution.sample()
        permitted = torch.ones_like(retrieve_mask)
        permitted = torch.where((op == 1)[:, None], retrieve_mask, permitted)
        permitted = torch.where((op == 5)[:, None], fade_mask, permitted)
        pd = torch.distributions.Categorical(logits=pointers.masked_fill(~permitted, -1e9))
        pointer = pd.probs.argmax(1) if greedy else pd.sample()
        used = ((op == 1) | (op == 5)).float()
        logp = distribution.log_prob(op) + used * pd.log_prob(pointer)
        entropy = distribution.entropy() + used * pd.entropy()
        return op, pointer, logp, value, entropy


def controller_inputs(memory, step, sensor, admitted, boost, last_loss, ema_loss,
                      replay_count, last_op, groups, device="cuda"):
    state = np.zeros((memory.count, 20), np.float32)
    state[:, :6] = np.column_stack((
        np.clip(last_loss, 0, 5), np.clip(ema_loss, 0, 5),
        sensor[:, 1], sensor[:, 2] - sensor[:, 0], sensor[:, 5], sensor[:, 6],
    ))
    meta = memory.metadata(step)
    state[:, 6:12] = np.column_stack((
        memory.live.mean(1), (memory.ids >= 0).mean(1),
        (meta[:, :, 0] * memory.live).sum(1) / np.maximum(memory.live.sum(1), 1),
        np.clip(replay_count / max(1, step // 4), 0, 1),
        (last_op == 1), (last_op == 5),
    ))
    state[:, 12:16] = np.column_stack((admitted / groups, boost, last_op == 4, last_op == 6))
    state[:, 16:] = np.clip(memory.recovery, 0, 5)
    retrieve = (memory.ids >= 0) & ~memory.live
    valid = np.ones((memory.count, len(OPS)), bool)
    valid[:, 1] = retrieve.any(1)
    valid[:, 2] = step > 0
    valid[:, 4] = admitted < groups
    valid[:, 5] = memory.live.any(1)
    valid[:, 6] = admitted > 2
    return tuple(torch.as_tensor(a, device=device) for a in (
        state, meta, valid, retrieve, memory.live.copy(),
    ))


def update_controller(optimizer, records, bootstrap, mode, gamma=0.97):
    target = bootstrap.detach()
    losses, entropies = [], []
    for logp, value, entropy, reward in reversed(records):
        target = reward if mode == "immediate" else reward + gamma**4 * target
        advantage = target - value
        losses.append((-logp * advantage.detach().clamp(-10, 10)
                       + 0.5 * advantage.square() - 0.025 * entropy).mean())
        entropies.append(entropy.mean().detach())
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite controller loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 2.)
    optimizer.step()
    return float(loss.detach()), float(torch.stack(entropies).mean())
