"""Independent predictors and factual values for bounded raw-record plans."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np
import torch
from torch import nn
from allocation_v11 import core as previous

CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
for key in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[key] = str(CACHE)

CANDIDATES, SLOTS, QUERY = 16, 3, 19
INPUT = 4 * 108 + 8 + 2 * SLOTS
PLAN_INPUT = QUERY + 2 * 6 + 3
physics, loss_for, choose = previous.physics, previous.loss_for, previous.choose
MOMENTA, FORCES, TOLERANCE = previous.MOMENTA, previous.FORCES, previous.TOLERANCE
Archive = previous.Archive


class Stream(previous.Stream):
    def __init__(self, count, seed, heldout=False):
        super().__init__(count, seed, heldout, delay=20)
        self.order = np.zeros(count, bool)

    def observe(self):
        if self.observed_step == self.step:
            raise RuntimeError("Each state may be observed only once")
        self.observed_step = self.step
        phase = self.step % 48
        if phase == 0:
            self.pending = self.pairs[self.rng.integers(len(self.pairs), size=self.count)].copy()
            self.order = self.rng.integers(2, size=self.count).astype(bool)
            if self.step == 0:
                self.active = self.pending.copy()
            self.body[:, 2] = self.rng.uniform(0.1, 0.9, self.count)
            self.body[:, 3] = self.rng.uniform(0.4, 0.7, self.count)
        if phase == 20:
            self.active = self.pending.copy()
            self.body[:, 1] = self.rng.uniform(-0.65, 0.65, self.count)
            self.body[:, 2] = self.rng.uniform(0.1, 0.9, self.count)
        self.context = np.full(self.count, 0.75, np.float32)
        image = np.zeros((self.count, 3, 6, 6), np.int8)
        if phase % 4 == 0:
            image[:, 2] = self.rng.integers(-12, 13, (self.count, 6, 6), dtype=np.int8)
            if phase in (0, 4):
                material = self.order == (phase == 0)
                image[material, 0, 1:4, 1:4] = (
                    (self.pending[material, 0] * 2 - 3) * 24)[:, None, None]
                image[~material, 1, 2, 2:4] = (
                    (self.pending[~material, 1] * 2 - 3) * 24)[:, None]
        sensor = np.concatenate((
            self.body.copy(), self.context[:, None],
            np.full((self.count, 1), np.sin(2 * np.pi * phase / 48), np.float32),
            np.full((self.count, 1), np.cos(2 * np.pi * phase / 48), np.float32),
        ), 1)
        return image, sensor


class Memory(previous.Memory):
    def __init__(self, count):
        super().__init__(count)
        self.summary = np.zeros((count, CANDIDATES, 3), np.float32)
        self.touch = np.full((count, CANDIDATES), -1, np.int64)

    def make_space(self, row, step):
        if self.live[row].sum() >= SLOTS:
            live = np.flatnonzero(self.live[row])
            self._fade(row, live[np.argmin(self.touch[row, live])], step)

    def append(self, step, images, ids):
        slot = (step // 4) % CANDIDATES
        for row in range(self.count):
            self.live[row, slot] = False
            self.make_space(row, step)
            self.ids[row, slot], self.times[row, slot] = ids[row], step
            self.cache[row, slot], self.live[row, slot] = images[row], True
            self.touch[row, slot], self.faded[row, slot] = step * 3, -1
            self.accumulated_loss[row, slot] = self.replays[row, slot] = 0
            self.details[row, slot] = 0
            self.summary[row, slot] = (images[row] != 0).any((1, 2))

    def apply(self, choices, plans, step, archive):
        reads = np.zeros(self.count, np.int64)
        for row, choice in enumerate(choices):
            kind, first, second = plans.items[int(choice)]
            if kind == 2:
                if not self.live[row, first]:
                    raise RuntimeError("Invalid fade plan")
                self._fade(row, first, step)
            elif kind == 1:
                for which, slot in enumerate((first, second)):
                    if slot < 0:
                        continue
                    if self.ids[row, slot] < 0 or self.live[row, slot]:
                        raise RuntimeError("Invalid retrieval plan")
                    raw = archive.get(self.ids[row, slot])
                    self.make_space(row, step)
                    self.cache[row, slot], self.live[row, slot] = raw, True
                    self.faded[row, slot], self.touch[row, slot] = -1, step * 3 + which + 1
                    reads[row] += 1
                    self.returns += 1
        return reads

    def numerical(self, step, image, sensor, device="cuda"):
        batch = self.tensors(step, image, sensor, device)
        return torch.cat((batch["images"].flatten(1), batch["sensor"],
                          batch["ages"], batch["mask"]), 1)


def query(memory, sensor, last_loss, ema_loss, step):
    live = memory.live
    counts = np.maximum(live.sum(1, keepdims=True), 1)
    cached_presence = (memory.cache != 0).any((3, 4))
    presence = (cached_presence * live[:, :, None]).sum(1) / counts
    ages = np.sort(np.where(live, (step - memory.times) / 64, -1), axis=1)[:, -SLOTS:]
    return np.concatenate((sensor, np.clip(last_loss[:, None], 0, 4),
                           np.clip(ema_loss[:, None], 0, 4), presence,
                           np.maximum(ages, 0), (ages >= 0).astype(np.float32)), 1).astype(np.float32)


class Plans:
    def __init__(self):
        self.items = [(0, -1, -1)]
        self.items += [(1, i, -1) for i in range(CANDIDATES)]
        self.items += [(1, i, j) for i, j in itertools.combinations(range(CANDIDATES), 2)]
        self.items += [(2, i, -1) for i in range(CANDIDATES)]
        self.array = np.asarray(self.items)

    def features(self, memory, state, step, mode, device="cuda", reads_enabled=True):
        age = np.clip((step - memory.times) / 64, 0, 2)
        content = memory.summary.copy()
        if mode == "state_time":
            content.fill(0)
            state = state.copy()
            state[:, 10:13] = 0
        if mode == "no_query_content":
            state = np.zeros_like(state)
        meta = np.concatenate((age[:, :, None], memory.live[:, :, None],
                               (memory.ids >= 0)[:, :, None], content), -1).astype(np.float32)
        padded = np.concatenate((meta, np.zeros((memory.count, 1, 6), np.float32)), 1)
        first, second = self.array[:, 1], self.array[:, 2]
        f, s = np.where(first < 0, CANDIDATES, first), np.where(second < 0, CANDIDATES, second)
        kind = np.eye(3, dtype=np.float32)[self.array[:, 0]]
        features = np.concatenate((
            np.broadcast_to(state[:, None], (memory.count, len(self.items), QUERY)),
            padded[:, f], padded[:, s],
            np.broadcast_to(kind, (memory.count, len(self.items), 3)),
        ), -1)
        valid = np.ones((memory.count, len(self.items)), bool)
        missing = (memory.ids >= 0) & ~memory.live
        for i, (kind, a, b) in enumerate(self.items[1:], 1):
            if kind == 1:
                valid[:, i] = missing[:, a] & (True if b < 0 else missing[:, b])
                if not reads_enabled:
                    valid[:, i] = False
            else:
                valid[:, i] = memory.live[:, a]
        return (torch.as_tensor(features, device=device),
                torch.as_tensor(valid, device=device))


class Value(nn.Module):
    def __init__(self, groups):
        super().__init__()
        size = QUERY + groups if groups else PLAN_INPUT
        self.net = nn.Sequential(nn.Linear(size, 32), nn.Tanh(),
                                 nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, features):
        return self.net(features.detach()).squeeze(-1)


class LocalPredictor(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(INPUT, 64), nn.SiLU(), nn.Linear(64, 32),
                          nn.SiLU(), nn.Linear(32, 9)) for _ in range(groups)])

    def forward(self, features, groups):
        output = features.new_zeros((len(features), 9))
        for index, expert in enumerate(self.experts):
            rows = torch.nonzero(groups == index, as_tuple=True)[0]
            if len(rows):
                output = output.index_copy(0, rows, expert(features[rows].detach()))
        return output[:, :6].reshape(-1, 3, 2), output[:, 6:]


def update_groups(model, optimizers, loss, groups):
    updates = []
    selected = groups.unique().tolist()
    for number, group in enumerate(selected):
        optimizer = optimizers[group]
        optimizer.zero_grad(set_to_none=True)
        parameters = list(model.experts[group].parameters())
        grads = torch.autograd.grad(loss[groups == group].mean(), parameters,
                                    retain_graph=number + 1 < len(selected))
        updates.append((optimizer, parameters, grads))
    for optimizer, parameters, grads in updates:
        for parameter, grad in zip(parameters, grads):
            parameter.grad = grad
        nn.utils.clip_grad_norm_(parameters, 2)
        optimizer.step()


def update_value(model, optimizer, features, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(features) - targets.detach()).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite local value loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2)
    optimizer.step()
    return float(loss.detach())


def select(scores, valid, rng, exploration):
    result = scores.masked_fill(~valid, -1e9).argmax(1).cpu().numpy()
    rows = np.flatnonzero(rng.random(len(result)) < exploration)
    if len(rows):
        masks = valid.cpu().numpy()
        for row in rows:
            result[row] = rng.choice(np.flatnonzero(masks[row]))
    return result


def macs(module):
    return sum(layer.in_features * layer.out_features for layer in module.modules()
               if isinstance(layer, nn.Linear))
