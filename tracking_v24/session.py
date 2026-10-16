"""V23 continuation with checkpointed, episode-bounded observation feedback."""
import io
import json
from pathlib import Path

import numpy as np
import torch

from tracking_v23.session import Session as OriginalSession
from tracking_v24.credit import ConsequenceWindows
from tracking_v24.neural import DelayedLearner


class CapturedExecutor:
    def __init__(self, delegate):
        self.delegate = delegate
        self.last_results = None

    def map(self, function, values):
        self.last_results = list(self.delegate.map(function, values))
        return self.last_results

    def shutdown(self):
        self.delegate.shutdown()


def pack_inputs(x, actions, before, episodes):
    output = io.BytesIO()
    np.savez_compressed(output, x=x.detach().cpu().numpy().astype(np.float16),
                        actions=np.asarray(actions, np.int64), before=before,
                        episodes=np.asarray(episodes, np.int64))
    return output.getvalue()


class Session(OriginalSession):
    @classmethod
    def initialize_fork(cls, session, config):
        session = super().initialize_fork(session, config)
        session.mechanism = config["mechanism"]
        session.delay = None
        session.delay_ready = []
        session.admit_delayed = True
        session.delayed_log = []
        if session.mechanism == "delayed":
            DelayedLearner.upgrade(session.agent, config)
            session.delay = ConsequenceWindows(session.bodies, config["delay_horizons"])
        session.executor = CapturedExecutor(session.executor)
        session._create_delayed_tables()
        return session

    def _create_delayed_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS gaze_inputs(step INTEGER PRIMARY KEY, batch BLOB);
            CREATE TABLE IF NOT EXISTS observed_state(
                step INTEGER, body INTEGER, energy REAL, integrity REAL, terminal INTEGER,
                PRIMARY KEY(step,body));
            CREATE TABLE IF NOT EXISTS delayed_outcomes(
                start INTEGER, end INTEGER, body INTEGER, episode INTEGER,
                horizon INTEGER, action INTEGER, target BLOB,
                PRIMARY KEY(start,body,horizon));
            CREATE TABLE IF NOT EXISTS delayed_updates(step INTEGER PRIMARY KEY, result TEXT);
        """)
        self.db.commit()

    @classmethod
    def restore(cls, path):
        session = super().restore(path)
        session.executor = CapturedExecutor(session.executor)
        session._create_delayed_tables()
        for table, column in (("gaze_inputs", "step"), ("observed_state", "step"),
                              ("delayed_outcomes", "end"), ("delayed_updates", "step")):
            comparison = ">=" if table == "gaze_inputs" else ">"
            session.db.execute(
                f"DELETE FROM {table} WHERE {column}{comparison}?", (session.step,))
        session.db.commit()
        return session

    def advance(self, count):
        if self.mechanism == "control":
            return super().advance(count)
        for _ in range(count):
            decision = (self.learning_enabled and self.admit_delayed
                        and self.step % self.config["action_every"] == 0)
            if decision:
                x, start = self.x.clone(), self.step
                before = self.proprio[:, 5:7].cpu().numpy().astype(np.float64)
                episodes = [world.segment for world in self.worlds]
            super().advance(1)
            results = self.executor.last_results
            if decision:
                actions = torch.as_tensor(self.current_actions, device=self.x.device)
                self.delay.start(start, x, actions, before, episodes)
                self.db.execute("INSERT INTO gaze_inputs VALUES(?,?)",
                                (start, pack_inputs(x, self.current_actions, before, episodes)))
            # These are the actual physical observations before any death reset.
            after = np.array([result[0][1][5:7] for result in results], np.float64)
            terminal = [result[2] for result in results]
            experienced = np.array([[result[3]["collision_duration"], result[3]["impact_count"],
                                      result[3]["contact"], result[3]["safe_contact"]]
                                     for result in results], np.float64)
            self.db.executemany("INSERT INTO observed_state VALUES(?,?,?,?,?)", [
                (self.step, i, float(state[0]), float(state[1]), int(terminal[i]))
                for i, state in enumerate(after)])
            rows = self.delay.advance(self.step, after, experienced, terminal)
            self.db.executemany("INSERT INTO delayed_outcomes VALUES(?,?,?,?,?,?,?)", [
                (row["start"], row["end"], row["body"], row["episode"], row["horizon"],
                 row["action"], row["target"].tobytes()) for row in rows])
            self.delay_ready.extend(rows)
            if self.step % self.config["block"] == 0:
                self.flush_delayed()
        return self.summary()

    def flush_delayed(self):
        if not self.delay_ready:
            return
        result = self.agent.train_delayed(self.delay_ready)
        result.update(step=self.step, min_start=min(row["start"] for row in self.delay_ready),
                      max_end=max(row["end"] for row in self.delay_ready))
        self.db.execute("INSERT INTO delayed_updates VALUES(?,?)",
                        (self.step, json.dumps(result)))
        self.delayed_log.append(result)
        self.delay_ready = []
        self.db.commit()

    def summary(self):
        result = super().summary()
        result["mechanism"] = getattr(self, "mechanism", None)
        delay = getattr(self, "delay", None)
        result["delayed"] = dict(
            pending=delay.pending_count() if delay is not None else 0,
            ready=len(getattr(self, "delay_ready", [])),
            updates=getattr(self.agent, "delayed_updates", 0),
            samples=getattr(self.agent, "delayed_samples", 0),
            horizon_samples=getattr(self.agent, "delayed_horizon_samples", []),
            learning_curve=getattr(self, "delayed_log", []))
        return result
