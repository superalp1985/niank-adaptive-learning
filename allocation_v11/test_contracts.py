"""Focused temporal-credit, isolation, and exact-continuation checks."""

import copy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import torch

import core
from core import ROOT, Stream, Memory, Archive, Predictor, Controller, GroupTrace
from run import Practice


class Contracts(unittest.TestCase):
    def test_cues_exist_but_leave_cache_before_long_delay(self):
        for delay in (8, 20):
            env, memory = Stream(4, 88, delay=delay), Memory(4)
            with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
                archive = Archive(Path(tmp) / "events.sqlite")
                for step in range(delay + 1):
                    image, _ = env.observe()
                    if step % 4 == 0:
                        memory.append(step, image, archive.append(step, image))
                    if step == delay:
                        cue_live = memory.live[:, :2]
                        self.assertEqual(bool(cue_live.all()), delay == 8)
                        self.assertTrue((memory.ids[:, :2] >= 0).all())
                        raw = archive.get(memory.ids[0, 0])
                        self.assertNotEqual(int(raw[0].sum()), 0)
                    env.execute(np.zeros(4, np.int64))
                archive.close()

    def test_credit_never_attaches_to_replacement_record(self):
        memory = Memory(1)
        image = np.zeros((1, 3, 6, 6), np.int8)
        memory.append(0, image, np.array([5]))
        memory.credit(np.array([1]), np.array([5]), np.array([-2.]))
        self.assertEqual(memory.details[0, 0, 3], 1)
        memory.append(64, image, np.array([69]))
        memory.credit(np.array([1]), np.array([5]), np.array([10.]))
        self.assertEqual(memory.expired_credit, 1)
        self.assertEqual(float(memory.details[0, 0].sum()), 0)

    def test_pooling_erases_only_resource_history_identity(self):
        memory = Memory(2)
        memory.ids[:] = np.arange(core.CANDIDATES)
        memory.details[:, 3, 4] = -2
        addressed, pooled = memory.metadata(20, "addressed"), memory.metadata(20, "pooled")
        np.testing.assert_array_equal(addressed[:, :, :3], pooled[:, :, :3])
        self.assertNotEqual(addressed[0, 0, 7], addressed[0, 3, 7])
        np.testing.assert_array_equal(pooled[:, 0, 3:], pooled[:, 3, 3:])
        groups = GroupTrace(2, 9)
        uses = np.zeros((2, 9), np.float32)
        uses[:, 4] = 1
        groups.credit(uses, np.array([-1., -2.]), 31)
        self.assertEqual(groups.counts[0, 4], 1)
        self.assertEqual(groups.counts[0, 3], 0)
        np.testing.assert_array_equal(groups.features(32, "pooled")[:, 0],
                                      groups.features(32, "pooled")[:, 4])

    def test_predictions_only_use_live_raw_records_and_accessible_groups(self):
        torch.manual_seed(71)
        model = Predictor(9).eval()
        memory = Memory(4)
        image, sensor = Stream(4, 33).observe()
        batch = memory.tensors(0, image, sensor, "cpu")
        batch["group_features"] = torch.zeros(4, 9, 4)
        admitted, boost = torch.full((4,), 2), torch.zeros(4, dtype=torch.bool)
        first = model(batch, admitted, boost)[0].detach()
        with torch.no_grad():
            model.w1[2:].fill_(999)
            model.w2[2:].fill_(-999)
        second = model(batch, admitted, boost)[0].detach()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(float(batch["images"][:, 1:].abs().sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_maturity_and_exact_resume(self):
        torch.manual_seed(81)
        torch.cuda.manual_seed_all(81)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
            a, b = Path(tmp) / "full", Path(tmp) / "split"
            a.mkdir()
            b.mkdir()
            full = Practice(Predictor(9).cuda(), Controller("addressed").cuda(), a, 802, 8, 20)
            full.advance(32)
            self.assertEqual(len(full.credits), 1)
            self.assertEqual(full.credits[0]["step"], 0)
            self.assertEqual(full.credits[0]["completed_at"], 31)
            full.save(a / "resume.pt")
            with closing(sqlite3.connect(b / "events.sqlite")) as target:
                full.archive.db.backup(target)
            full.advance(96)
            expected = {k: v.detach().clone() for k, v in full.model.state_dict().items()}
            expected_controller = {k: v.detach().clone() for k, v in full.controller.state_dict().items()}
            expected_history = copy.deepcopy(full.history)
            expected_groups = full.groups.value.copy()
            expected_total = copy.deepcopy(full.total)
            full.close()
            split = Practice(Predictor(9).cuda(), Controller("addressed").cuda(), b, 802, 8, 20)
            split.load(a / "resume.pt")
            split.advance(96)
            for key, tensor in split.model.state_dict().items():
                torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
            for key, tensor in split.controller.state_dict().items():
                torch.testing.assert_close(tensor, expected_controller[key], rtol=0, atol=0)
            for got, expected_step in zip(split.history, expected_history):
                for key in got:
                    np.testing.assert_array_equal(got[key], expected_step[key])
            np.testing.assert_array_equal(split.groups.value, expected_groups)
            self.assertEqual(split.total, expected_total)
            split.close()

    def test_split_and_executed_only_gradients(self):
        train, held = Stream(4, 10), Stream(4, 10, heldout=True)
        self.assertFalse(set(map(tuple, train.pairs)) & set(map(tuple, held.pairs)))
        states = torch.randn(4, 3, 2, requires_grad=True)
        contacts = torch.randn(4, 3, requires_grad=True)
        actions = torch.tensor([0, 1, 2, 0])
        loss, _ = core.loss_for(states, contacts, actions, torch.zeros(4, 2), torch.zeros(4))
        loss.sum().backward()
        for row, chosen in enumerate(actions):
            for action in range(3):
                if action != chosen:
                    self.assertEqual(float(states.grad[row, action].abs().sum()), 0)
                    self.assertEqual(float(contacts.grad[row, action].abs()), 0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    unittest.main(verbosity=2)
