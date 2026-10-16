import pickle
import unittest

import numpy as np
import torch

try:
    from tracking_v24.credit import ConsequenceWindows
except ImportError:
    ConsequenceWindows = None


class CreditChecks(unittest.TestCase):
    def buffer(self):
        self.assertIsNotNone(ConsequenceWindows, "Delayed credit is not implemented")
        return ConsequenceWindows(2, (2, 4))

    def test_future_only_targets_and_horizons(self):
        buffer = self.buffer()
        buffer.start(10, torch.zeros(2, 964), torch.tensor([1, 2]),
                     np.ones((2, 2)), [3, 7])
        first = buffer.advance(11, np.array([[.9, .8], [1., 1.]]),
                               np.array([[.01, 1, 0, 0], [0, 0, 0, 0]]), [False, False])
        self.assertEqual(first, [])
        rows = buffer.advance(12, np.array([[.8, .7], [1., 1.]]),
                              np.array([[0, 0, 1, 1], [0, 0, 0, 0]]), [False, False])
        self.assertEqual([(r["body"], r["horizon"]) for r in rows], [(0, 2), (1, 2)])
        np.testing.assert_allclose(rows[0]["target"], [-.2, -.3, .5, 1, 1, 1, 0])
        buffer.advance(13, np.ones((2, 2)), np.zeros((2, 4)), [False, False])
        rows = buffer.advance(14, np.ones((2, 2)), np.zeros((2, 4)), [False, False])
        self.assertEqual(len(rows), 2)
        self.assertEqual(buffer.pending_count(), 0)

    def test_terminal_flush_uses_pre_reset_state_and_does_not_cross_episodes(self):
        buffer = self.buffer()
        buffer.start(0, torch.zeros(2, 964), torch.tensor([4, 5]),
                     np.ones((2, 2)), [9, 20])
        rows = buffer.advance(1, np.array([[.7, 0.], [.99, 1.]]),
                              np.zeros((2, 4)), [True, False])
        self.assertEqual([(r["body"], r["horizon"]) for r in rows], [(0, 2), (0, 4)])
        self.assertTrue(all(r["episode"] == 9 and r["target"][-1] == 1 for r in rows))
        np.testing.assert_allclose(rows[0]["target"][:2], [-.3, -1])
        buffer.start(1, torch.ones(2, 964), torch.tensor([6, 7]),
                     np.ones((2, 2)), [10, 20])
        rows = buffer.advance(2, np.ones((2, 2)), np.zeros((2, 4)), [False, False])
        self.assertFalse(any(r["body"] == 0 and r["episode"] == 9 for r in rows))

    def test_partially_matured_queues_resume_exactly(self):
        buffer = self.buffer()
        buffer.start(0, torch.randn(2, 964), torch.tensor([3, 11]),
                     np.ones((2, 2)), [0, 0])
        buffer.advance(1, np.ones((2, 2)), np.zeros((2, 4)), [False, False])
        restored = pickle.loads(pickle.dumps(buffer))
        expected = buffer.advance(2, np.ones((2, 2)), np.zeros((2, 4)), [False, True])
        actual = restored.advance(2, np.ones((2, 2)), np.zeros((2, 4)), [False, True])
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left["x"], right["x"]))
            np.testing.assert_array_equal(left["target"], right["target"])
            self.assertEqual(left["action"], right["action"])


if __name__ == "__main__":
    unittest.main()
