"""Behavioral contracts for reversible, consequence-tested memory."""

import sqlite3
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from tracking_v21.directory import Directory
from tracking_v21.learner import Agent


class MemoryContracts(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE directory_events(body,time,segment,features,"
                        "full_features,proprio,action,expert,targets,PRIMARY KEY(body,time))")
        self.memory = Directory(1, 42, self.db, capacity=2)
        self.x = np.random.default_rng(7).random((11, 128), dtype=np.float32)
        self.prop = np.zeros(6, np.float32)

    def tearDown(self):
        self.db.close()

    def add(self, time, action=1):
        import zlib
        target = np.array([.2, 0, 0, 0, 0], np.float32)
        self.memory.add(0, time, 0, self.x, self.prop, action, 0, target)
        packed = zlib.compress(self.x.astype(np.float16).tobytes())
        self.db.execute("INSERT INTO directory_events VALUES(?,?,?,?,?,?,?,?,?)",
                        (0, time, 0, packed, packed, self.prop.tobytes(),
                         action, 0, target.tobytes()))

    def test_evicted_record_can_return_from_cold_database(self):
        for time in range(3):
            self.add(time)
        self.assertNotIn((0, 0), map(tuple, self.memory.event_ids()))
        self.assertTrue(self.memory.recall(0, 0, now=4))
        self.assertIn((0, 0), map(tuple, self.memory.event_ids()))
        self.assertEqual(self.memory.cold_recalls, 1)

    def test_access_alone_does_not_increase_retention(self):
        self.add(0)
        before = self.memory.benefits.copy()
        self.memory.query(self.x[None], self.prop[None], now=1)
        np.testing.assert_array_equal(before, self.memory.benefits)

    def test_only_matured_matching_trial_can_change_retention(self):
        self.add(0)
        self.memory.query(self.x[None], self.prop[None], now=1)
        trial = self.memory.trials[0]
        enabled = trial["enabled"]
        before = self.memory.benefits.copy()
        self.memory.settle(0, 0, np.ones(5, np.float32), now=8)
        np.testing.assert_array_equal(before, self.memory.benefits)
        target = np.array([1 if enabled else -1, 0, 0, 0, 0], np.float32)
        self.memory.settle(0, 1, target, now=8)
        self.assertGreater(float(self.memory.benefits.max()), 0)
        self.assertIsNone(self.memory.trials[0])

    def test_future_events_are_not_available_to_query_or_recall(self):
        self.add(5)
        correction = self.memory.query(self.x[None], self.prop[None], now=4)
        np.testing.assert_array_equal(correction, 0)
        with self.assertRaises(ValueError):
            self.memory.recall(0, 5, now=4)

    def test_retrieval_never_crosses_body_identity(self):
        other = Directory(2, 13, self.db, capacity=2)
        other.add(0, 0, 0, self.x, self.prop, 1, 0, np.ones(5, np.float32))
        result = other.query(np.stack([self.x, self.x]),
                             np.stack([self.prop, self.prop]), now=1)
        np.testing.assert_array_equal(result[1], 0)
        self.assertIsNone(other.trials[1])


class PolicyContracts(unittest.TestCase):
    def test_logged_probability_includes_external_correction(self):
        torch.manual_seed(12)
        agent = Agent("reversible", "cpu")
        x = np.random.default_rng(3).random((3, 11, 128), dtype=np.float32)
        prop = np.zeros((3, 6), np.float32)
        external = np.tile(np.linspace(-.8, .8, 9), (3, 1)).astype(np.float32)
        selected = agent.act(x, prop, external=external)
        with torch.no_grad():
            for row, group in enumerate(selected["groups"]):
                logits = agent.experts[int(group)](
                    torch.as_tensor(x[row:row + 1]), torch.as_tensor(prop[row:row + 1]))
                policy = Categorical(logits=logits + selected["guide_logits"][row])
                self.assertAlmostEqual(float(policy.log_prob(selected["actions"][row])),
                                       float(selected["action_logp"][row]), places=6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
