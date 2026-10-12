"""Withdrawn, unexecuted test draft for the rejected sequential V13 protocol.

Retained as a design record only. See STATUS.md before any implementation.
"""

import copy
from contextlib import closing
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
core = None
if (ROOT / "core.py").exists():
    import core


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, "V13 mechanism is not implemented")

    def test_class_exploration_does_not_depend_on_record_multiplicity(self):
        plans = core.Plans()
        values = np.zeros((2, 33), np.float32)
        classes = np.zeros((2, 3), np.float32)
        valid = np.zeros((2, 33), bool)
        valid[:, [0, 1, 17]] = True
        valid[1, 1:17] = True
        probability = plans.probabilities(classes, values, valid, True, epsilon=1)
        for kind in range(3):
            np.testing.assert_allclose(probability[:, plans.kinds == kind].sum(1), 1 / 3)
        self.assertFalse(probability[~valid].any())
        flat = plans.probabilities(classes, values, valid, False, epsilon=1)
        self.assertGreater(flat[1, 1:17].sum(), flat[0, 1:17].sum())

    def test_credit_is_factual_delayed_and_baseline_is_saved(self):
        pending = core.Pending(8)
        pending.add(dict(start=0, baseline=np.array([-0.3], np.float32),
                         class_prediction=np.array([0.1], np.float32)))
        for index in range(7):
            self.assertEqual(pending.observe(index * 4, np.array([-0.1], np.float32)), [])
        result = pending.observe(28, np.array([-0.1], np.float32))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["end"], 31)
        np.testing.assert_allclose(result[0]["target"], 0.2, atol=1e-7)
        np.testing.assert_allclose(result[0]["record_target"], 0.1, atol=1e-7)

    def test_summary_is_sign_free_and_excluded_from_numerical_path(self):
        memory = core.Memory(2)
        raw = np.zeros((2, 3, 6, 6), np.int8)
        raw[0, 0, 1:4, 1:4], raw[1, 0, 1:4, 1:4] = -72, 72
        memory.append(0, raw, np.arange(2))
        np.testing.assert_array_equal(memory.summary[0], memory.summary[1])
        sensor = np.zeros((2, 8), np.float32)
        before = memory.numerical(0, raw, sensor, device="cpu")
        memory.summary.fill(1)
        torch.testing.assert_close(before, memory.numerical(0, raw, sensor, device="cpu"),
                                   rtol=0, atol=0)

    def test_stream_all_pairs_and_continuing_reversal(self):
        env = core.Stream(16, 100, 48)
        self.assertEqual(len(set(map(tuple, env.pairs))), 16)
        for step in range(97):
            _, sensor = env.observe()
            np.testing.assert_array_equal(sensor[:, 5], 0.35 if step // 48 % 2 == 0 else 0.95)
            if step % 48 == 20:
                expected = env.pending if step // 48 % 2 == 0 else 3 - env.pending
                np.testing.assert_array_equal(env.active, expected)
            env.execute(np.ones(16, np.int64))

    def test_only_selected_group_gets_gradients(self):
        torch.manual_seed(10)
        model = core.LocalPredictor(4)
        optimizers = [torch.optim.Adam(group.parameters()) for group in model.experts]
        original = copy.deepcopy(model.state_dict())
        groups = torch.full((3,), 1)
        states, contacts = model(torch.randn(3, core.INPUT), groups)
        loss, _ = core.loss_for(states, contacts, torch.arange(3), torch.zeros(3, 2), torch.zeros(3))
        core.update_groups(model, optimizers, loss, groups)
        for name, parameter in model.state_dict().items():
            if not name.startswith("experts.1."):
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
        self.assertEqual(len(optimizers[0].state), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_exact_resume_with_pending_credit_and_sqlite(self):
        from run import Practice
        torch.manual_seed(19)
        torch.cuda.manual_seed_all(19)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
            full = Practice(Path(tmp) / "full", "class_span", 10, 8, 4, 48)
            full.advance(20)
            full.save()
            restored_path = Path(tmp) / "split"
            restored_path.mkdir()
            with closing(sqlite3.connect(restored_path / "events.sqlite")) as target:
                full.archive.db.backup(target)
            full.advance(112)
            expected = {name: copy.deepcopy(getattr(full, name).state_dict())
                        for name in ("model", "selector", "class_value", "record_value")}
            totals = copy.deepcopy(full.total)
            history = copy.deepcopy(full.history)
            full.close()
            restored = Practice(restored_path, "class_span", 10, 8, 4, 48)
            restored.load(Path(tmp) / "full" / "resume.pt")
            restored.advance(112)
            for name, wanted in expected.items():
                for key, value in getattr(restored, name).state_dict().items():
                    torch.testing.assert_close(value, wanted[key], rtol=0, atol=0)
            self.assertEqual(totals, restored.total)
            for got, wanted in zip(restored.history, history):
                for name in got:
                    np.testing.assert_array_equal(got[name], wanted[name])
            restored.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    unittest.main(verbosity=2)
