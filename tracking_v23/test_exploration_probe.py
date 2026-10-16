from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tracking_v22.evaluate_stages import parameter_hash
from tracking_v23.exploration_probe import ExploringFrozenSession
from tracking_v23.session import backup_database
from tracking_v23.test_runtime import configuration


@unittest.skipUnless(torch.cuda.is_available(), "CUDA experiment runtime")
class ExplorationChecks(unittest.TestCase):
    def test_exploration_without_learning_and_exact_resume(self):
        torch.set_num_threads(4)
        torch.manual_seed(53017)
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as folder:
            root = Path(folder)
            session = ExploringFrozenSession(
                root/"original", dict(configuration(), arm="active"), training=False)
            before = parameter_hash(session.agent)
            try:
                session.advance(16)
                self.assertFalse(session.training)
                self.assertFalse(session.learning_enabled)
                self.assertIsNone(session.agent.trial)
                self.assertEqual(session.agent.updates, 0)
                self.assertEqual(session.agent.meta_updates, 0)
                self.assertEqual(session.agent.local_updates, 0)
                self.assertEqual(session.agent.state_updates, 0)
                self.assertGreater(session.visual_totals["head_travel"], 0)
                self.assertEqual(parameter_hash(session.agent), before)
                session.checkpoint()
                backup_database(session.output/"events.sqlite", root/"resume"/"events.sqlite")
                session.advance(4)
                expected_actions = session.current_actions.copy()
                expected_positions = [tuple(world.body.position) for world in session.worlds]
            finally:
                session.close()
            restored = ExploringFrozenSession.restore(root/"resume")
            try:
                restored.advance(4)
                np.testing.assert_array_equal(restored.current_actions, expected_actions)
                self.assertEqual([tuple(world.body.position) for world in restored.worlds],
                                 expected_positions)
                self.assertEqual(parameter_hash(restored.agent), before)
                self.assertFalse(restored.training)
                self.assertFalse(restored.learning_enabled)
            finally:
                restored.close()


if __name__ == "__main__":
    unittest.main()
