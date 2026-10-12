"""Bounded past-evidence lookup; full raw experience remains in SQLite."""

import sqlite3
import zlib

import numpy as np


def coarse(obs):
    bearing = (np.arctan2(obs[:, 3], obs[:, 2]) + np.pi) / (np.pi / 4)
    obstacle = np.where(obs[:, [10, 13, 16]] > 0,
                        np.linalg.norm(obs[:, 8:17].reshape(-1, 3, 3)[:, :, :2],
                                       axis=-1), 2).min(1) < 0.15
    return bearing.astype(int) % 8, obstacle, (obs[:, [19, 22, 25]] > 0).any(1)


class Directory:
    def __init__(self, path, worlds, mode):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("CREATE TABLE IF NOT EXISTS events("
                        "id INTEGER PRIMARY KEY, world INTEGER, time INTEGER,"
                        "obs BLOB, action INTEGER, next BLOB, image BLOB)")
        self.count, self.mode, self.capacity = worlds, mode, 64 if mode == "directory" else 8
        self.ids = np.full((worlds, self.capacity), -1, np.int64)
        self.states = np.zeros((worlds, self.capacity, 32), np.float32)
        self.times = np.zeros((worlds, self.capacity), np.int64)
        self.utility = np.zeros((worlds, self.capacity), np.float32)
        self.active = [{} for _ in range(worlds)]
        self.next_id, self.reads, self.fades, self.reuses = 0, 0, 0, 0

    def owner(self, identifier):
        return self.db.execute("SELECT world FROM events WHERE id=?",
                               (int(identifier),)).fetchone()[0]

    def fetch(self, world, identifier):
        row = self.db.execute("SELECT world,obs,action,next FROM events WHERE id=?",
                              (int(identifier),)).fetchone()
        if row is None or row[0] != world:
            raise ValueError("Memory record belongs to another body or is missing")
        self.reads += 1
        return dict(obs=np.frombuffer(row[1], np.float32).copy(), action=row[2],
                    next=np.frombuffer(row[3], np.float32).copy())

    def _activate(self, world, identifier, record):
        cache = self.active[world]
        if identifier in cache:
            return
        if len(cache) >= 8:
            def score(item):
                positions = np.flatnonzero(self.ids[world] == item)
                return (self.utility[world, positions[0]] if len(positions) else -100,
                        item)
            del cache[min(cache, key=score)]
            self.fades += 1
        cache[identifier] = record

    def append(self, step, obs, actions, next_obs, raw):
        rows = []
        for world in range(self.count):
            identifier = self.next_id
            self.next_id += 1
            rows.append((identifier, world, step, obs[world].tobytes(), int(actions[world]),
                         next_obs[world].tobytes(), zlib.compress(raw[world].tobytes(), 1)))
            empty = np.flatnonzero(self.ids[world] < 0)
            if len(empty):
                slot = empty[0]
            elif self.mode == "recent":
                slot = np.argmin(self.times[world])
            else:
                age = np.maximum(step - self.times[world], 0)
                slot = np.argmin(self.utility[world] - 0.0005 * age)
            old = int(self.ids[world, slot])
            self.active[world].pop(old, None)
            self.ids[world, slot], self.times[world, slot] = identifier, step
            self.states[world, slot], self.utility[world, slot] = obs[world], 0.
            self._activate(world, identifier, dict(obs=obs[world].copy(),
                                                  action=int(actions[world]),
                                                  next=next_obs[world].copy()))
        self.db.executemany("INSERT INTO events VALUES(?,?,?,?,?,?,?)", rows)

    def retrieve(self, obs):
        evidence = np.zeros((self.count, 16), np.float32)
        selected = np.full(self.count, -1, np.int64)
        bearing, obstacle, attack = coarse(obs)
        for world in range(self.count):
            valid = self.ids[world] >= 0
            if not valid.any():
                continue
            distance = np.square(self.states[world] - obs[world]).mean(1)
            if self.mode == "directory":
                b, o, a = coarse(self.states[world])
                gap = np.minimum((b - bearing[world]) % 8, (bearing[world] - b) % 8)
                local = valid & (gap <= 1) & (o == obstacle[world]) & (a == attack[world])
                if local.any():
                    valid = local
                distance = distance - 0.03 * self.utility[world]
            slot = int(np.argmin(np.where(valid, distance, np.inf)))
            identifier = int(self.ids[world, slot])
            selected[world] = identifier
            record = self.active[world].get(identifier)
            if record is None:
                record = self.fetch(world, identifier)
                self._activate(world, identifier, record)
            delta = record["next"] - record["obs"]
            translation = delta[26:28]
            evidence[world, :2] = delta[2:4] + translation
            evidence[world, 2:4] = delta[17:19] + translation
            evidence[world, 4:6] = delta[20:22] + translation
            evidence[world, 6 + record["action"]] = 1.
            evidence[world, 15] = np.square(record["obs"] - obs[world]).mean()
            self.reuses += 1
        return selected, evidence

    def feedback(self, ids, predictive_gain):
        for world, identifier in enumerate(ids):
            if identifier < 0:
                continue
            slots = np.flatnonzero(self.ids[world] == identifier)
            if len(slots):
                slot = slots[0]
                self.utility[world, slot] = (0.9 * self.utility[world, slot] +
                                             0.1 * np.clip(predictive_gain[world], -1, 1))

    def close(self):
        self.db.commit()
        self.db.close()
