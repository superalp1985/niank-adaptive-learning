"""Active memory forgets; factual database records remain recoverable."""

import math
import zlib

import numpy as np

from tracking_v20.directory import EventDirectory
from tracking_v20.learner import outcome_utility


class Directory(EventDirectory):
    PROMOTION = 0.

    def __init__(self, worlds, seed, db, capacity=128):
        super().__init__(worlds, "retrieval", seed, capacity)
        self.db = db
        self.projection = np.random.default_rng(21021).normal(
            size=(self.KEY_DIM, 6)).astype(np.float32)
        self.benefits = np.zeros((worlds, capacity), np.float32)
        self.admitted = np.zeros((worlds, capacity), np.int64)
        self.trials = [None] * worlds
        self.baselines = np.zeros((worlds, 64), np.float32)
        self.cold_recalls = self.settled_trials = self.enabled_trials = 0
        self.benefit_updates = self.query_count = 0
        self.db.execute("CREATE TABLE IF NOT EXISTS memory_catalog("
            "body INTEGER,time INTEGER,bucket INTEGER,benefit REAL DEFAULT 0,"
            "tests INTEGER DEFAULT 0,PRIMARY KEY(body,time))")
        self.db.execute("CREATE INDEX IF NOT EXISTS memory_bucket "
                        "ON memory_catalog(body,bucket,time)")
        self.db.execute("CREATE TABLE IF NOT EXISTS memory_trials("
            "body INTEGER,start INTEGER,end INTEGER,enabled INTEGER,baseline REAL,"
            "actual REAL,estimate REAL,records TEXT,PRIMARY KEY(body,start))")
        self.db.execute("CREATE TABLE IF NOT EXISTS cold_recalls("
                        "body INTEGER,time INTEGER,recalled_at INTEGER)")

    def bucket(self, keys):
        bits = (np.asarray(keys) @ self.projection > 0).astype(np.int64)
        return (bits * (1 << np.arange(6))).sum(-1)

    def _slot(self, body, now):
        empty = np.flatnonzero(self._times[body] < 0)
        if len(empty):
            return int(empty[0])
        # Positive benefit buys longer residence, never permanent immunity.
        lifetime = self.HALF_LIFE * np.exp(4 * self.benefits[body].clip(-.5, .5))
        score = np.exp(-(now - self.admitted[body]) / lifetime)
        return int(np.argmin(score))

    def add(self, body, time, segment, features, proprio, action, group, targets,
            key_features=None):
        self._heads[body] = self._slot(body, time)
        result = super().add(body, time, segment, features, proprio, action,
                             group, targets, key_features)
        self.benefits[body, result["slot"]] = 0
        self.admitted[body, result["slot"]] = time
        self.db.execute("INSERT INTO memory_catalog(body,time,bucket) VALUES(?,?,?)",
            (int(body), int(time), int(self.bucket(self._keys[body, result["slot"]]))))
        return result

    def recall(self, body, event_time, now):
        if event_time >= now:
            raise ValueError("Cold memory must precede the query")
        if np.any(self._times[body] == event_time):
            return False
        row = self.db.execute(
            "SELECT d.segment,d.features,d.full_features,d.proprio,d.action,d.expert,"
            "d.targets,m.benefit FROM directory_events d JOIN memory_catalog m "
            "ON d.body=m.body AND d.time=m.time WHERE d.body=? AND d.time=?",
            (int(body), int(event_time))).fetchone()
        if row is None:
            return False
        segment, packed, full, prop, action, group, target, benefit = row
        shape = self._feature_shape
        features = np.frombuffer(zlib.decompress(packed), np.float16).reshape(shape)
        key_features = np.frombuffer(zlib.decompress(full), np.float16).reshape(shape)
        proprio = np.frombuffer(prop, np.float32)
        slot = self._slot(body, now)
        self.evictions[body] += int(self._times[body, slot] >= 0)
        self._features[body, slot] = features
        self._keys[body, slot] = self.key(key_features, proprio)
        self._proprio[body, slot] = proprio
        self._actions[body, slot], self._groups[body, slot] = action, group
        self._targets[body, slot] = np.frombuffer(target, np.float32)
        self._times[body, slot], self._segments[body, slot] = event_time, segment
        self._weight[body, slot], self._weight_time[body, slot] = 1., now
        self._last_access[body, slot], self._access_count[body, slot] = -1, 0
        self.benefits[body, slot] = benefit
        self.admitted[body, slot] = now
        self.cold_recalls += 1
        self.db.execute("INSERT INTO cold_recalls VALUES(?,?,?)",
                        (int(body), int(event_time), int(now)))
        return True

    def query(self, features, proprio, now):
        keys = self.key(features, proprio)
        buckets = self.bucket(keys)
        correction = np.zeros((self.worlds, 9), np.float32)
        self.query_count += self.worlds
        for body in range(self.worlds):
            slots = np.flatnonzero((self._times[body] >= 0) & (self._times[body] < now))
            if not len(slots):
                continue
            distance = np.sum((self._keys[body, slots] - keys[body]) ** 2, axis=1)
            chosen = np.argsort(distance)[:8]
            slots, distance = slots[chosen], distance[chosen]
            weights = np.exp(-distance / .08)
            if weights.max() < .01:
                continue
            values = outcome_utility(self._targets[body, slots]) / 9.
            weights *= .5 + 1 / (1 + np.exp(-6 * self.benefits[body, slots]))
            mass, score = np.zeros(9), np.zeros(9)
            for slot, weight, value in zip(slots, weights, values):
                action = self._actions[body, slot]
                mass[action] += weight
                score[action] += weight * value
            center = float((weights * values).sum() / max(weights.sum(), 1e-6))
            offset = 1.2 * np.tanh(3 * (score - mass * center) / (mass + 1.))
            trial = self.trials[body]
            if trial is None:
                bucket = int(buckets[body])
                trial = dict(start=int(now), bucket=bucket, enabled=bool(self.rng.random() < .5),
                    baseline=float(self.baselines[body, bucket]), records={})
                self.trials[body] = trial
            for slot, weight in zip(slots, weights):
                event_time = int(self._times[body, slot])
                trial["records"][event_time] = trial["records"].get(event_time, 0.) + float(weight)
            if trial["enabled"]:
                correction[body] = offset
        return correction

    def settle(self, body, event_time, target, now):
        trial = self.trials[body]
        if trial is None or trial["start"] != event_time:
            return
        if now < event_time:
            raise ValueError("Trial cannot settle before action")
        actual = float(outcome_utility(target) / 9.)
        estimate = float(np.clip(2 * (1 if trial["enabled"] else -1) *
                                 (actual - trial["baseline"]), -1, 1))
        bucket = trial["bucket"]
        self.baselines[body, bucket] = .95 * self.baselines[body, bucket] + .05 * actual
        total = max(sum(trial["records"].values()), 1e-6)
        import json
        records = []
        for past_time, weight in trial["records"].items():
            row = self.db.execute("SELECT benefit,tests FROM memory_catalog "
                                  "WHERE body=? AND time=?", (int(body), past_time)).fetchone()
            if row is None:
                continue
            benefit, tests = row
            rate = .2 * math.sqrt(weight / total)
            benefit = (1 - rate) * benefit + rate * estimate
            self.db.execute("UPDATE memory_catalog SET benefit=?,tests=? "
                            "WHERE body=? AND time=?",
                            (benefit, tests + 1, int(body), past_time))
            self.benefits[body, self._times[body] == past_time] = benefit
            self.benefit_updates += 1
            records.append([past_time, weight, benefit])
        self.db.execute("INSERT INTO memory_trials VALUES(?,?,?,?,?,?,?,?)",
            (int(body), trial["start"], int(now), int(trial["enabled"]), trial["baseline"],
             actual, estimate, json.dumps(records, separators=(",", ":"))))
        self.settled_trials += 1
        self.enabled_trials += int(trial["enabled"])
        self.trials[body] = None

    def sample(self, query_keys, failure_strength, now, budget=1536):
        for body, bucket in enumerate(self.bucket(query_keys)):
            active = set(self._times[body].tolist())
            # Same coarse sensory bucket plus a uniformly chosen historical time.
            times = self.db.execute("SELECT time FROM memory_catalog WHERE body=? "
                "AND bucket=? AND time<? ORDER BY time DESC LIMIT 64",
                (body, int(bucket), int(now))).fetchall()
            cold = [int(row[0]) for row in times if row[0] not in active]
            if cold:
                for past_time in self.rng.choice(cold, min(2, len(cold)), replace=False):
                    self.recall(body, int(past_time), now)
            cutoff = int(self.rng.integers(max(1, now)))
            row = self.db.execute("SELECT time FROM memory_catalog WHERE body=? AND time<=? "
                "ORDER BY time DESC LIMIT 1", (body, cutoff)).fetchone()
            if row and row[0] < now:
                self.recall(body, int(row[0]), now)
        return super().sample(query_keys, failure_strength, now, budget)
