"""Bounded, body-local replay with reversible access to factual past events.

The caller supplies only matured outcome labels. Retrieval uses the observation
key and previous-query failure strength; it never reads the outcome labels.
"""

from __future__ import annotations

import math

import numpy as np


class EventDirectory:
    """Keep a FIFO window per body and sample a fixed total replay budget.

    ``features`` uses channel-first current RGB, previous RGB and history RGB
    in its first nine channels. ``key_features`` can supply their full ungated
    snapshot while ``features`` preserves the actual input used for the action.
    Keys preserve the locations of 16 circular eye sectors.

    ``sample`` splits its total budget equally among bodies with eligible past
    records, rotating the remainder. Failed queries in retrieval mode receive
    75% nearest-key visits and 25% uniform reconsideration. Other queries and
    random mode draw uniformly with exactly the same per-body opportunities.
    Draws are disjoint when the eligible record count covers the body budget.
    Repetition is explicit in logs when the budget exceeds available records.
    """

    BINS = 16
    KEY_DIM = BINS * 9 + 6
    TARGET_DIM = 5
    RANDOM_FRACTION = .25
    HALF_LIFE = 256.
    MIN_WEIGHT = 1e-4
    MAX_WEIGHT = 4.
    PROMOTION = .25
    REVISIT_PRIORITY = .05

    def __init__(self, worlds, mode, seed, capacity=512):
        if mode not in ("retrieval", "random"):
            raise ValueError("Unknown V20 event-directory mode")
        self.worlds = self._integer(worlds, "worlds", minimum=1)
        self.capacity = self._integer(capacity, "capacity", minimum=1)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        shape = (self.worlds, self.capacity)
        self._features = None
        self._feature_shape = None
        self._proprio = np.zeros((*shape, 6), np.float32)
        self._keys = np.zeros((*shape, self.KEY_DIM), np.float32)
        self._targets = np.zeros((*shape, self.TARGET_DIM), np.float32)
        self._actions = np.zeros(shape, np.int64)
        self._groups = np.zeros(shape, np.int64)
        self._times = np.full(shape, -1, np.int64)
        self._segments = np.full(shape, -1, np.int64)
        self._weight = np.zeros(shape, np.float32)
        self._weight_time = np.full(shape, -1, np.int64)
        self._last_access = np.full(shape, -1, np.int64)
        self._access_count = np.zeros(shape, np.int64)
        self._counts = np.zeros(self.worlds, np.int64)
        self._heads = np.zeros(self.worlds, np.int64)
        self._latest_time = np.full(self.worlds, -1, np.int64)
        self._last_query_time = np.full(self.worlds, -1, np.int64)
        self.evictions = np.zeros(self.worlds, np.int64)
        self._sample_calls = 0

    @staticmethod
    def _integer(value, name, minimum=0):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or
                not np.isfinite(value) or int(value) != value or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    @property
    def counts(self):
        """Return the number of resident records per body."""
        return self._counts.copy()

    def __len__(self):
        return int(self._counts.sum())

    @property
    def stored_bytes(self):
        arrays = (self._features, self._proprio, self._keys, self._targets,
                  self._actions, self._groups, self._times, self._segments,
                  self._weight, self._weight_time, self._last_access,
                  self._access_count, self._counts, self._heads,
                  self._latest_time, self._last_query_time, self.evictions)
        return sum(value.nbytes for value in arrays if value is not None)

    @classmethod
    def key(cls, features, proprio):
        """Build float32 keys from sensory channels and six proprio values."""
        x = np.asarray(features, np.float32)
        prop = np.asarray(proprio, np.float32)
        if (x.ndim < 2 or x.shape[-2] < 9 or x.shape[-1] < cls.BINS or
                x.shape[-1] % cls.BINS):
            raise ValueError("Expected (..., >=9 RGB channels, eyes divisible by 16)")
        if prop.shape != (*x.shape[:-2], 6):
            raise ValueError("Expected six proprio values per sensory observation")
        visual = x[..., :9, :]
        if not np.isfinite(visual).all() or not np.isfinite(prop).all():
            raise ValueError("Sensory key inputs must be finite")
        visual = np.clip(visual, 0, 1)
        current, previous, history = (visual[..., start:start + 3, :]
                                      for start in (0, 3, 6))
        signals = np.concatenate((2 * current - 1,
                                  2 * (current - previous),
                                  .75 * (2 * history - 1)), axis=-2)
        pooled = signals.reshape(*signals.shape[:-1], cls.BINS,
                                 signals.shape[-1] // cls.BINS).mean(-1)
        # Keep directional sector order; do not rotate or average across sectors.
        visual_key = pooled.reshape(*x.shape[:-2], cls.BINS * 9)
        visual_key = visual_key / np.float32(math.sqrt(cls.BINS * 3))
        prop_key = np.tanh(prop) * np.float32(.5 / math.sqrt(6))
        return np.concatenate((visual_key, prop_key), -1).astype(np.float32)

    def add(self, body, time, segment, features, proprio, action, group, targets,
            key_features=None):
        """Insert one matured event and return its stable ID and evicted ID.

        Targets are float32 ``[discounted_progress, contact, safe_contact,
        collision, impact]``. Horizon termination and maturity are the caller's
        responsibility. They are stored solely as replay labels.
        """
        body = self._integer(body, "body")
        if body >= self.worlds:
            raise ValueError("body is outside the directory")
        time = self._integer(time, "time")
        segment = self._integer(segment, "segment")
        action = self._integer(action, "action")
        group = self._integer(group, "group")
        if time <= self._latest_time[body]:
            raise ValueError("Event times must increase strictly within each body")
        x = np.asarray(features, np.float32)
        prop = np.asarray(proprio, np.float32)
        target = np.asarray(targets, np.float32)
        if x.ndim != 2:
            raise ValueError("add expects one channel-first sensory observation")
        if self._feature_shape is not None and x.shape != self._feature_shape:
            raise ValueError("Replay feature shape changed")
        if prop.shape != (6,) or target.shape != (self.TARGET_DIM,):
            raise ValueError("Expected six proprio values and five matured targets")
        if (not np.isfinite(x).all() or not np.isfinite(prop).all() or
                not np.isfinite(target).all() or
                np.abs(x).max(initial=0) > np.finfo(np.float16).max):
            raise ValueError("Replay values must be finite and features fit float16")
        key_source = x if key_features is None else np.asarray(key_features, np.float32)
        if key_source.ndim != 2 or key_source.shape[-1] != x.shape[-1]:
            raise ValueError("key_features must describe the same sensory observation")
        key = self.key(key_source, prop)
        if self._features is None:
            self._feature_shape = x.shape
            self._features = np.zeros(
                (self.worlds, self.capacity, *x.shape), np.float16)
        slot = int(self._heads[body])
        old_time = int(self._times[body, slot])
        evicted_id = None if old_time < 0 else (body, old_time)
        if evicted_id is not None:
            self.evictions[body] += 1
        self._features[body, slot] = x
        self._proprio[body, slot] = prop
        self._keys[body, slot] = key
        self._targets[body, slot] = target
        self._actions[body, slot] = action
        self._groups[body, slot] = group
        self._times[body, slot] = time
        self._segments[body, slot] = segment
        self._weight[body, slot] = 1
        self._weight_time[body, slot] = time
        self._last_access[body, slot] = -1
        self._access_count[body, slot] = 0
        self._latest_time[body] = time
        self._heads[body] = (slot + 1) % self.capacity
        self._counts[body] = min(self.capacity, self._counts[body] + 1)
        return dict(event_id=(body, time), slot=slot, evicted_id=evicted_id)

    def event_ids(self, body=None):
        """Resident stable ``(body, time)`` IDs, sorted by body and time."""
        if body is None:
            bodies = range(self.worlds)
        else:
            body = self._integer(body, "body")
            if body >= self.worlds:
                raise ValueError("body is outside the directory")
            bodies = (body,)
        records = [(item, int(time)) for item in bodies
                   for time in np.sort(self._times[item, self._times[item] >= 0])]
        return np.asarray(records, np.int64).reshape(-1, 2)

    def _empty_batch(self):
        shape = self._feature_shape or (11, 128)
        return dict(features=np.empty((0, *shape), np.float32),
                    proprio=np.empty((0, 6), np.float32),
                    actions=np.empty(0, np.int64),
                    groups=np.empty(0, np.int64),
                    targets=np.empty((0, self.TARGET_DIM), np.float32),
                    record_ids=np.empty((0, 2), np.int64))

    def sample(self, query_keys, failure_strength, now, budget=1536):
        """Return ``(batch, list_of_visit_logs)`` for a total, body-local budget.

        Only records with ``event_time < now`` are eligible. Empty directories
        return an empty batch. A body's recent failures do not increase its
        replay opportunities. Scalar ``now`` and ``failure_strength`` broadcast.
        """
        budget = self._integer(budget, "budget")
        keys = np.asarray(query_keys, np.float32)
        strengths = np.asarray(failure_strength, np.float32)
        clocks = np.asarray(now)
        if keys.shape == (self.KEY_DIM,) and self.worlds == 1:
            keys = keys[None]
        if keys.shape != (self.worlds, self.KEY_DIM) or not np.isfinite(keys).all():
            raise ValueError("Expected one finite query key per body")
        try:
            strengths = np.broadcast_to(strengths, (self.worlds,))
            clocks = np.broadcast_to(clocks, (self.worlds,))
        except ValueError as error:
            raise ValueError("Expected one failure strength and clock per body") from error
        if not np.isfinite(strengths).all() or (strengths < 0).any():
            raise ValueError("Failure strengths must be finite and nonnegative")
        if (not np.isfinite(clocks).all() or (clocks < 0).any() or
                (clocks != np.floor(clocks)).any()):
            raise ValueError("Query clocks must be nonnegative integers")
        clocks = clocks.astype(np.int64)
        if (clocks < self._last_query_time).any():
            raise ValueError("Query clocks cannot move backward")
        eligible = [np.flatnonzero((self._times[body] >= 0) &
                                   (self._times[body] < clocks[body]))
                    for body in range(self.worlds)]
        bodies = [body for body, slots in enumerate(eligible) if len(slots)]
        self._last_query_time[:] = clocks
        if budget == 0 or not bodies:
            return self._empty_batch(), []
        quota, remainder = divmod(budget, len(bodies))
        quotas = dict.fromkeys(bodies, quota)
        offset = self._sample_calls % len(bodies)
        for index in range(remainder):
            quotas[bodies[(offset + index) % len(bodies)]] += 1
        self._sample_calls += 1
        selected_bodies, selected_slots, logs = [], [], []
        for body in bodies:
            count = quotas[body]
            if count == 0:
                continue
            slots = eligible[body]
            query_time = int(clocks[body])
            elapsed = query_time - self._weight_time[body, slots]
            decay = np.exp(-np.float32(math.log(2) / self.HALF_LIFE) * elapsed)
            self._weight[body, slots] = np.maximum(
                self.MIN_WEIGHT, self._weight[body, slots] * decay)
            self._weight_time[body, slots] = query_time
            distance = np.sqrt(np.sum(
                (self._keys[body, slots] - keys[body]) ** 2, axis=1))
            failed = bool(strengths[body] > 0)
            strength = float(min(1, strengths[body]))
            if self.mode == "retrieval" and failed:
                random_count = int(math.ceil(count * self.RANDOM_FRACTION))
                ranked_count = count - random_count
                priority = distance - (self.REVISIT_PRIORITY * strength *
                                       np.log1p(self._weight[body, slots]))
                order = np.lexsort((self._times[body, slots], priority))
                ranked = np.resize(order, ranked_count)
                pool = np.setdiff1d(np.arange(len(slots)), np.unique(ranked),
                                    assume_unique=True)
                if len(pool) < random_count:
                    pool = np.arange(len(slots))
                random = self.rng.choice(pool, random_count,
                                         replace=random_count > len(pool))
                chosen = np.concatenate((ranked, random))
                random_flags = np.arange(count) >= ranked_count
            else:
                chosen = self.rng.choice(len(slots), count, replace=count > len(slots))
                random_flags = np.ones(count, bool)
            seen = set()
            for index, random_visit in zip(chosen, random_flags):
                slot = int(slots[index])
                event_time = int(self._times[body, slot])
                before = float(self._weight[body, slot])
                after = min(self.MAX_WEIGHT,
                            before + self.PROMOTION * (1 + strength))
                previous_access = int(self._last_access[body, slot])
                self._weight[body, slot] = after
                self._last_access[body, slot] = query_time
                self._access_count[body, slot] += 1
                event_id = (body, event_time)
                logs.append(dict(
                    body=body, query_time=query_time, event_time=event_time,
                    event_segment=int(self._segments[body, slot]),
                    distance=float(distance[index]), random=bool(random_visit),
                    weight_before=before, weight_after=float(after),
                    failed_query=failed, failure_strength=float(strengths[body]),
                    event_id=event_id, retrieval_id=f"{body}:{event_time}",
                    last_access_before=previous_access,
                    last_access_after=query_time,
                    access_count=int(self._access_count[body, slot]),
                    repeated=slot in seen))
                seen.add(slot)
                selected_bodies.append(body)
                selected_slots.append(slot)
        rows = (np.asarray(selected_bodies), np.asarray(selected_slots))
        batch = dict(features=self._features[rows].astype(np.float32),
                     proprio=self._proprio[rows].copy(),
                     actions=self._actions[rows].copy(),
                     groups=self._groups[rows].copy(),
                     targets=self._targets[rows].copy(),
                     record_ids=np.asarray([log["event_id"] for log in logs], np.int64))
        return batch, logs
