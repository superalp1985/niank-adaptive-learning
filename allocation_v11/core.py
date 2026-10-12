"""Mutable per-resource traces, with a matched pooled-information ablation."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from allocation_v10 import core as old

CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[key] = str(CACHE)

CANDIDATES, SLOTS, HORIZON = 16, 3, 32
OPS = old.OPS
physics, loss_for, choose = old.physics, old.loss_for, old.choose
MOMENTA, FORCES, TOLERANCE = old.MOMENTA, old.FORCES, old.TOLERANCE


class Stream(old.Stream):
    def __init__(self, count, seed, heldout=False, delay=20):
        super().__init__(count, seed, heldout)
        if delay not in (8, 20):
            raise ValueError("Use an inside/outside-cache delay")
        self.delay = delay

    def observe(self):
        if self.observed_step == self.step:
            raise RuntimeError("State already observed")
        self.observed_step = self.step
        phase = self.step % 48
        if phase == 0:
            self.pending = self.pairs[self.rng.integers(len(self.pairs), size=self.count)].copy()
            if self.step == 0:
                self.active = self.pending.copy()
            self.body[:, 2] = self.rng.uniform(0.1, 0.9, self.count)
            self.body[:, 3] = self.rng.uniform(0.4, 0.7, self.count)
        if phase == self.delay:
            self.active = self.pending.copy()
        self.context = np.full(self.count, (0.15, 0.55, 1., 0.35)[phase // 12], np.float32)
        image = np.zeros((self.count, 3, 6, 6), np.int8)
        if phase % 4 == 0:
            image[:, 2] = self.rng.integers(-12, 13, (self.count, 6, 6), dtype=np.int8)
            if phase == 0:
                image[:, 0, 1:4, 1:4] = ((self.pending[:, 0] * 2 - 3) * 24)[:, None, None]
            if phase == 4:
                image[:, 1, 2, 2:4] = ((self.pending[:, 1] * 2 - 3) * 24)[:, None]
        sensor = np.concatenate((
            self.body.copy(), self.context[:, None],
            np.full((self.count, 1), np.sin(2 * np.pi * phase / 48), np.float32),
            np.full((self.count, 1), np.cos(2 * np.pi * phase / 48), np.float32),
        ), 1)
        return image, sensor


class Archive(old.Archive):
    def __init__(self, path):
        import sqlite3
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, world INTEGER, time INTEGER, image BLOB)")
        self.reads = 0


class Memory(old.Memory):
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
        self.details = np.zeros((count, CANDIDATES, 6), np.float32)
        self.returns = self.fade_count = self.censored = self.expired_credit = 0

    def append(self, step, images, ids):
        slot = (step // 4) % CANDIDATES
        for row in range(self.count):
            if self.faded[row, slot] >= 0:
                self.censored += 1
            self.live[row, slot] = False
            if self.live[row].sum() >= SLOTS:
                old_slot = np.flatnonzero(self.live[row])[np.argmin(self.times[row, self.live[row]])]
                self._fade(row, old_slot, step)
            self.ids[row, slot], self.times[row, slot] = ids[row], step
            self.cache[row, slot] = images[row]
            self.live[row, slot] = True
            self.faded[row, slot] = -1
            self.accumulated_loss[row, slot] = self.replays[row, slot] = 0
            self.details[row, slot] = 0

    def apply(self, ops, pointers, step, archive):
        rows = np.flatnonzero(ops == 1)
        if len(rows):
            slots = pointers[rows]
            self.details[rows, slots, 0] = (step - self.faded[rows, slots]) / 64
            self.details[rows, slots, 1] = self.accumulated_loss[rows, slots] / np.maximum(
                step - self.faded[rows, slots], 1)
            self.details[rows, slots, 2] += 1
        super().apply(ops, pointers, step, archive)

    def metadata(self, step, mode="addressed"):
        basic = super().metadata(step)
        extra = self.details.copy()
        extra[:, :, 2:4] = np.log1p(extra[:, :, 2:4]) / 4
        extra = np.clip(extra, -8, 8)
        if mode == "pooled":
            valid = self.ids >= 0
            mean = (extra * valid[:, :, None]).sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
            extra[:] = mean[:, None]
        return np.concatenate((basic, extra), -1).astype(np.float32)

    def credit(self, ops, identities, returns):
        for row in np.flatnonzero((ops == 1) | (ops == 5)):
            slots = np.flatnonzero(self.ids[row] == identities[row])
            if not len(slots):
                self.expired_credit += 1
                continue
            slot = slots[0]
            self.details[row, slot, 3] += 1
            self.details[row, slot, 4] = 0.8 * self.details[row, slot, 4] + 0.2 * returns[row]
            self.details[row, slot, 5] = returns[row]


class GroupTrace:
    def __init__(self, count, groups):
        self.counts = np.zeros((count, groups), np.float32)
        self.value = np.zeros((count, groups), np.float32)
        self.last = np.full((count, groups), -1, np.int64)

    def features(self, step, mode):
        out = np.stack((np.log1p(self.counts) / 4, self.value,
                        np.minimum((step - self.last) / 64, 4),
                        (self.counts > 0).astype(np.float32)), -1)
        if mode == "pooled":
            out[:] = out.mean(1, keepdims=True)
        return out.astype(np.float32)

    def credit(self, uses, returns, step):
        selected = uses > 0
        self.counts += uses
        self.value[selected] = (0.8 * self.value + 0.2 * returns[:, None])[selected]
        self.last[selected] = step


class Predictor(old.Predictor):
    def __init__(self, groups):
        super().__init__(groups)
        self.trace_router = nn.Sequential(nn.Linear(68, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, batch, admitted, boost):
        x = self.visual(batch["images"].flatten(2)).flatten(1)
        x = self.combine(torch.cat((x, batch["sensor"], batch["ages"], batch["mask"]), 1))
        trace = self.trace_router(torch.cat((
            x[:, None].expand(-1, self.groups, -1), batch["group_features"]), -1)).squeeze(-1)
        allowed = torch.arange(self.groups, device=x.device)[None] < admitted[:, None]
        logits = (self.router(x) + trace).masked_fill(~allowed, -1e9)
        values, indices = logits.topk(2, dim=1)
        count = 1 + boost.long()
        weights = values.masked_fill(torch.arange(2, device=x.device)[None] >= count[:, None], -1e9).softmax(1)
        hidden = torch.zeros_like(x)
        for slot in range(2):
            selected = torch.nonzero(count > slot, as_tuple=True)[0]
            if len(selected):
                group = indices[selected, slot]
                first = torch.bmm(x[selected, None], self.w1[group]).squeeze(1) + self.b1[group]
                second = torch.bmm(F.silu(first)[:, None], self.w2[group]).squeeze(1) + self.b2[group]
                hidden.index_add_(0, selected, second * weights[selected, slot:slot + 1])
        if self.training:
            probability = logits.softmax(1).gather(1, indices[:, :1])
            hidden = hidden * (1 + probability - probability.detach())
        output = self.head(F.silu(x + hidden))
        return output[:, :6].reshape(-1, 3, 2), output[:, 6:], indices

    def forward_macs(self, batch, boosts):
        return super().forward_macs(batch, boosts) + batch * self.groups * (68 * 16 + 16)


class Controller(old.Controller):
    def __init__(self, mode):
        super().__init__("delayed")
        self.mode = mode
        self.pointer = nn.Sequential(nn.Linear(29, 24), nn.Tanh(), nn.Linear(24, 1))

    def distributions(self, state, metadata, valid):
        out = self.joint(state)
        logits, value = out[:, :7], out[:, 7]
        logits = logits.masked_fill(~valid, -1e9)
        probs = 0.95 * logits.softmax(1) + 0.05 * valid.float() / valid.sum(1, keepdim=True)
        features = torch.cat((state[:, None].expand(-1, CANDIDATES, -1), metadata), -1)
        return torch.distributions.Categorical(probs=probs), value, self.pointer(features).squeeze(-1)


def inputs(memory, step, sensor, admitted, boost, last_loss, ema_loss,
           replay_count, last_op, groups, mode, device="cuda"):
    data = list(old.controller_inputs(memory, step, sensor, admitted, boost, last_loss,
                                      ema_loss, replay_count, last_op, groups, device))
    data[1] = torch.as_tensor(memory.metadata(step, mode), device=device)
    return tuple(data)
