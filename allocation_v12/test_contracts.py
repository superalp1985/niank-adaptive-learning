"""Contracts for numerical isolation, factual locality and exact continuation."""

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
if importlib.util.find_spec("core") is None:
    core = None
else:
    import core


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, "V12 local factual mechanism is not implemented")

    def test_summary_is_sign_free_and_not_in_numerical_inputs(self):
        memory = core.Memory(2)
        image = np.zeros((2, 3, 6, 6), np.int8)
        image[0, 0, 1:4, 1:4] = -72
        image[1, 0, 1:4, 1:4] = 72
        memory.append(0, image, np.arange(2))
        np.testing.assert_array_equal(memory.summary[0], memory.summary[1])
        sensor = np.zeros((2, 8), np.float32)
        before = memory.tensors(0, image, sensor, "cpu")
        before_query = core.query(memory, sensor, np.zeros(2), np.zeros(2), 0)
        memory.summary.fill(1)
        after = memory.tensors(0, image, sensor, "cpu")
        after_query = core.query(memory, sensor, np.zeros(2), np.zeros(2), 0)
        np.testing.assert_array_equal(before_query, after_query)
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)

    def test_pair_retrieval_is_raw_and_retained_after_insertion(self):
        (ROOT / "cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
            archive, memory = core.Archive(Path(tmp) / "events.sqlite"), core.Memory(2)
            for step in range(0, 21, 4):
                image = np.full((2, 3, 6, 6), step + 1, np.int8)
                memory.append(step, image, archive.append(step, image))
            self.assertFalse(memory.live[:, :2].any())
            plans = core.Plans()
            chosen = next(i for i, plan in enumerate(plans.items) if plan == (1, 0, 1))
            reads = memory.apply(np.full(2, chosen), plans, 20, archive)
            np.testing.assert_array_equal(reads, 2)
            self.assertEqual(archive.reads, 4)
            self.assertTrue(memory.live[:, :2].all())
            np.testing.assert_array_equal(memory.cache[:, 0], 1)
            np.testing.assert_array_equal(memory.cache[:, 1], 5)
            image.fill(99)
            memory.append(24, image, archive.append(24, image))
            self.assertTrue(memory.live[:, :2].all())
            np.testing.assert_array_equal(memory.times[:, :2], [[0, 4], [0, 4]])
            archive.close()

    def test_plan_legality_and_condition_ablation(self):
        plans, memory = core.Plans(), core.Memory(3)
        sensor = np.ones((3, 8), np.float32)
        query = core.query(memory, sensor, np.ones(3), np.ones(3), 0)
        features, valid = plans.features(memory, query, 0, "state_content", "cpu")
        self.assertTrue(valid[:, 0].all())
        self.assertEqual(int(valid.sum()), 3)
        image = np.zeros((3, 3, 6, 6), np.int8)
        image[:, 1, 2, 2:4] = 24
        memory.append(0, image, np.arange(3))
        memory.live[:] = False
        a, valid = plans.features(memory, query, 20, "state_content", "cpu")
        b, _ = plans.features(memory, query, 20, "state_time", "cpu")
        c, _ = plans.features(memory, query, 20, "no_query_content", "cpu")
        torch.testing.assert_close(a[:, :, :core.QUERY], b[:, :, :core.QUERY])
        self.assertGreater(float((a - b).abs().sum()), 0)
        self.assertEqual(float(c[:, :, :core.QUERY].abs().sum()), 0)
        self.assertEqual(int(valid.sum()), 6)

    def test_selected_group_updates_without_cross_module_gradients(self):
        torch.manual_seed(5)
        model, selector = core.LocalPredictor(4).cpu(), core.Value(0).cpu()
        optimizers = [torch.optim.Adam(group.parameters(), lr=0.001) for group in model.experts]
        original = copy.deepcopy(model.state_dict())
        x = torch.randn(5, core.INPUT)
        actions = torch.tensor([0, 1, 2, 0, 1])
        groups = torch.full((5,), 2)
        states, contacts = model(x, groups)
        loss, _ = core.loss_for(states, contacts, actions, torch.zeros(5, 2), torch.zeros(5))
        core.update_groups(model, optimizers, loss, groups)
        for name, tensor in model.state_dict().items():
            if not name.startswith("experts.2."):
                torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, original[n]) for n, p in model.state_dict().items()
                            if n.startswith("experts.2.")))
        self.assertTrue(all(p.grad is None for p in selector.parameters()))
        self.assertEqual(len(optimizers[0].state), 0)

    def test_only_executed_action_receives_physical_gradient(self):
        states = torch.randn(3, 3, 2, requires_grad=True)
        contact = torch.randn(3, 3, requires_grad=True)
        actions = torch.tensor([0, 1, 2])
        loss, _ = core.loss_for(states, contact, actions, torch.zeros(3, 2), torch.zeros(3))
        loss.sum().backward()
        for row, action in enumerate(actions):
            for other in range(3):
                if other != action:
                    self.assertEqual(float(states.grad[row, other].abs().sum()), 0)
                    self.assertEqual(float(contact.grad[row, other].abs()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_exact_resume_of_local_updates_and_external_database(self):
        from run import Practice
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        (ROOT / "cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
            full = Practice(Path(tmp) / "full", "state_content", 90, 8, 4)
            full.advance(32)
            full.save()
            state = Path(tmp) / "full" / "resume.pt"
            splitpath = Path(tmp) / "split"
            splitpath.mkdir()
            with closing(sqlite3.connect(splitpath / "events.sqlite")) as target:
                full.archive.db.backup(target)
            full.advance(96)
            expected = copy.deepcopy(full.model.state_dict())
            expected_op = copy.deepcopy(full.planner.state_dict())
            expected_route = copy.deepcopy(full.selector.state_dict())
            history, total = copy.deepcopy(full.history), copy.deepcopy(full.total)
            full.close()
            split = Practice(splitpath, "state_content", 90, 8, 4)
            split.load(state)
            split.advance(96)
            for model, wanted in ((split.model, expected), (split.planner, expected_op),
                                  (split.selector, expected_route)):
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, wanted[name], rtol=0, atol=0)
            for got, wanted in zip(split.history, history):
                for name in got:
                    np.testing.assert_array_equal(got[name], wanted[name])
            self.assertEqual(split.total, total)
            split.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    unittest.main(verbosity=2)
