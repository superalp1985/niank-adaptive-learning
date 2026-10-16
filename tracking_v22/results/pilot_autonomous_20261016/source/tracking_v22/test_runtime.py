"""Check complete live/restore state equivalence, including plasticity."""
import hashlib
import shutil
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from tracking_v22.run import Session, run


@unittest.skipUnless(torch.cuda.is_available(), "CUDA experiment runtime")
class RuntimeChecks(unittest.TestCase):
    def test_episode_boundary_context(self):
        config = dict(bodies=2, variant="autonomous", world_seed=1900, block=1,
                      horizon=1, action_every=1, seed=99)
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as root:
            session = Session(Path(root)/"death", config)
            try:
                for world in session.worlds:
                    world.energy = 0.
                session.advance(1)
                self.assertTrue(torch.equal(session.context[:, 26],
                                            torch.ones(2, device="cuda")))
                self.assertTrue(torch.equal(session.context[:, 27],
                                            torch.ones(2, device="cuda")))
                self.assertEqual(session.metrics["deaths"], 2)
            finally:
                session.close()

    def test_probe_resume_after_completion_marker_interruption(self):
        config = dict(bodies=2, variant="fixed", world_seed=1910, block=4,
                      horizon=2, action_every=2, seed=99, steps=8,
                      checkpoint_every=4, probe_steps=4, settling_steps=4)
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as root:
            path = Path(root)/"run"
            run(config, path)
            first = (path/"frozen_probe"/"summary.json").read_text()
            (path/"COMPLETE.json").unlink()
            run(config, path)
            import json
            a, b = json.loads(first), json.loads((path/"frozen_probe"/"summary.json").read_text())
            for key in ("step", "contact", "collision", "frames", "updates", "choices"):
                self.assertEqual(a[key], b[key])

    def test_runtime_continuation(self):
        torch.set_num_threads(4)
        torch.manual_seed(99)
        config = dict(bodies=2, variant="autonomous", world_seed=1900, block=8,
                      horizon=4, action_every=2, seed=99)
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as root:
            original = Path(root)/"original"
            restored = Path(root)/"restored"
            session = Session(original, config)
            try:
                session.advance(16)
                session.checkpoint()
                restored.mkdir()
                shutil.copy2(original/"events.sqlite", restored/"events.sqlite")
                session.advance(8)
                expected = [np.array(w.body.position) for w in session.worlds]
                weights = {k:v.clone() for k,v in session.agent.model.state_dict().items()}
                spikes = session.agent.spikes.membrane.clone()
                choices = session.choices.copy()
                # All 24 physical intervals must correspond to 2400 neural ticks.
                self.assertEqual(session.agent.spikes.ticks, 2400)
            finally:
                session.close()
            resumed = Session.restore(restored)
            try:
                resumed.advance(8)
                for position, world in zip(expected, resumed.worlds):
                    np.testing.assert_array_equal(position, np.array(world.body.position))
                for key, value in weights.items():
                    self.assertTrue(torch.equal(value, resumed.agent.model.state_dict()[key]), key)
                self.assertTrue(torch.equal(spikes, resumed.agent.spikes.membrane))
                self.assertEqual(choices, resumed.choices)
            finally:
                resumed.close()


if __name__ == "__main__":
    unittest.main()
