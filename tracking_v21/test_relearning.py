"""Consequences can revise behavior in either direction without fabricated labels."""

import sqlite3
import unittest

import numpy as np
import torch

from tracking_v21.directory import Directory
from tracking_v21.learner import Agent


class RelearningContracts(unittest.TestCase):
    def test_failed_executed_action_is_suppressed_only_in_candidate(self):
        x = np.random.default_rng(13).random((8, 11, 128), dtype=np.float32)
        prop = np.zeros((8, 6), np.float32)
        target = np.tile(np.array([-.5, 0, 0, 1, 1], np.float32), (8, 1))
        batch = dict(features=x, proprio=prop, actions=np.full(8, 2, np.int64),
                     groups=np.zeros(8, np.int64), targets=target)
        for mode in ("random", "reversible"):
            torch.manual_seed(91)
            agent = Agent(mode, "cpu")
            probability = lambda: agent.experts[0](
                torch.as_tensor(x), torch.as_tensor(prop)).softmax(1)[:, 2].mean().item()
            before = probability()
            predictors = [[v.detach().clone() for v in p.parameters()]
                          for p in agent.predictors]
            agent.fit_consequences(batch, replay=True)
            after = probability()
            if mode == "reversible":
                self.assertLess(after, before)
                self.assertEqual(agent.negative_presentations, 8)
            else:
                self.assertEqual(after, before)
            self.assertEqual(agent.replay_policy_presentations, 0)
            for saved, predictor in zip(predictors, agent.predictors):
                self.assertTrue(all(torch.equal(a, b)
                                    for a, b in zip(saved, predictor.parameters())))

    def test_previously_retained_detail_can_lose_value_after_failed_practice(self):
        db = sqlite3.connect(":memory:")
        try:
            directory = Directory(1, 8, db, capacity=4)
            x = np.random.default_rng(2).random((11, 128), dtype=np.float32)
            prop = np.zeros(6, np.float32)
            directory.add(0, 0, 0, x, prop, 1, 0, np.zeros(5, np.float32))
            directory.query(x[None], prop[None], 1)
            directory.trials[0]["enabled"] = True
            directory.settle(0, 1, np.array([1, 1, 1, 0, 0], np.float32), 8)
            self.assertGreater(float(directory.benefits[0, 0]), 0)
            for clock in range(9, 200, 9):
                directory.query(x[None], prop[None], clock)
                directory.trials[0]["enabled"] = True
                directory.settle(0, clock, np.array([-1, 0, 0, 1, 1], np.float32), clock + 7)
            self.assertLess(float(directory.benefits[0, 0]), 0)
        finally:
            db.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
