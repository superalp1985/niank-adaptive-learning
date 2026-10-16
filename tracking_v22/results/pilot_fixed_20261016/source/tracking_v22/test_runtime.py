"""Check complete live/restore state equivalence, including plasticity."""
import hashlib
import shutil
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from tracking_v22.run import Session


@unittest.skipUnless(torch.cuda.is_available(), "CUDA experiment runtime")
class RuntimeChecks(unittest.TestCase):
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
