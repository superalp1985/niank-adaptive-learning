from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tracking_v23.evaluate import file_hash
from tracking_v24.test_runtime import RuntimeChecks, configuration

try:
    from tracking_v24.run import run
except ImportError:
    run = None


@unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime")
class RunnerChecks(unittest.TestCase):
    def test_finite_pair_freezes_probes_and_settles_all_credit(self):
        self.assertIsNotNone(run, "Finite V24 runner is not implemented")
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as temporary:
            root = Path(temporary)
            with redirect_stdout(io.StringIO()):
                fixture = RuntimeChecks()
                source = fixture.source(root)
                source_hash = file_hash(source)
                config = dict(configuration(), additional_steps=16, checkpoint_every=4,
                              milestone_every=8, stage_probe_steps=4, final_probe_steps=4,
                              probe_checkpoint_every=2, stage_scene_seed=8113,
                              final_scene_seed=98113, probe_seed=239, final_diagnostics=True)
                results = run(config, root/"pair", source)
            self.assertEqual(file_hash(source), source_hash)
            for result in results.values():
                self.assertEqual(result["training"]["new_steps"], 32)
                self.assertEqual(result["training"]["unsettled_learning_choices"], 0)
                self.assertEqual(result["training"]["delayed"]["pending"], 0)
                self.assertEqual(result["training"]["delayed"]["ready"], 0)
                self.assertTrue(result["final_probe"]["learned_parameters_unchanged"])
            self.assertEqual(results["control"]["training"]["delayed"]["updates"], 0)
            self.assertGreater(results["delayed"]["training"]["delayed"]["updates"], 0)
            self.assertEqual(set(results["delayed"]["diagnostics"]),
                             {"uniform", "head_hold", "delay_off"})
            from tracking_v24.audit import check
            audit = check(root/"pair"/"delayed")
            self.assertTrue(audit["physical_targets_reproduced"])
            self.assertTrue(audit["no_episode_crossing"])
            with self.assertRaisesRegex(RuntimeError, "Completed"):
                run(config, root/"pair", source)
            marker = json.loads((root/"pair"/"COMPLETE.json").read_text())
            self.assertEqual(set(marker["arms"]), {"control", "delayed"})


if __name__ == "__main__":
    unittest.main()
