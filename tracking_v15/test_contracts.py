"""Temporal and risk boundaries, beyond V14's inherited world contracts."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "learner.py").exists(), "V15 risk allocation is not implemented")

    def test_temporal_delta_stays_in_body_and_resets(self):
        from tracking_v15.learner import temporal_features
        now, before = np.zeros((2, 32), np.float32), np.zeros((2, 32), np.float32)
        now[0, 2], now[1, 2] = 0.6, -0.2
        before[:, 2] = 0.1
        result = temporal_features(now, before, np.zeros((2, 16), np.float32),
                                   np.array([False, True]))
        self.assertAlmostEqual(float(result[0, 34]), 0.5)
        np.testing.assert_array_equal(result[1, 32:64], np.zeros(32))

    def test_risk_correction_is_detached_and_action_specific(self):
        from tracking_v15.learner import Agent
        agent = Agent(device="cpu", mode="risk_alloc")
        x = torch.zeros(3, 80)
        with torch.no_grad():
            agent.risk[-1].weight.zero_()
            agent.risk[-1].bias.fill_(-6)
            agent.risk[-1].bias[2] = 6
        correction = agent.correction(x)
        self.assertFalse(correction.requires_grad)
        self.assertLess(float(correction[0, 1]), float(correction[0, 0]) - 1.9)
        agent.expert_logits(x, torch.zeros(3, dtype=torch.long)).sum().backward()
        self.assertTrue(all(p.grad is None for p in agent.risk.parameters()))

    def test_only_executed_action_risk_has_a_loss_gradient(self):
        from tracking_v15.learner import factual_risk_loss
        logits = torch.zeros(2, 9, 2, requires_grad=True)
        factual_risk_loss(logits, torch.tensor([1, 2]), torch.ones(2, 2)).backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[0, 1].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[1, 1].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[1, 2].abs().sum()), 0.)

    def test_stored_correction_survives_later_estimator_change(self):
        from tracking_v15.learner import Agent
        agent = Agent(device="cpu", mode="risk_alloc")
        agent.act(np.zeros((2, 80), np.float32))
        original = agent.logged_corrections[0].copy()
        with torch.no_grad():
            agent.risk[-1].bias[0] = 6
        np.testing.assert_array_equal(agent.logged_corrections[0], original)
        self.assertGreater(float(np.abs(
            agent.correction(torch.zeros(2, 80)).numpy() - original).max()), 1.)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
