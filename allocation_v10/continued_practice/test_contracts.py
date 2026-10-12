"""Only engineering/scientific isolation checks for the new mechanism."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from core import (
    ROOT, Stream, Archive, Memory, Predictor, Controller, controller_inputs,
    loss_for, physics, MOMENTA, FORCES, update_controller,
)


class Contracts(unittest.TestCase):
    def test_pair_split_and_physical_continuity(self):
        a, b = Stream(32, 100), Stream(32, 100, True)
        self.assertFalse(set(map(tuple, a.pairs)) & set(map(tuple, b.pairs)))
        for step in range(40):
            image, sensor = a.observe()
            before = a.body.copy()
            states, contacts, costs = physics(before, MOMENTA[a.active[:, 0]], FORCES[a.active[:, 1]], a.context)
            action = np.arange(32) % 3
            actual, contact, cost, regret = a.execute(action)
            np.testing.assert_array_equal(actual, states[np.arange(32), action])
            np.testing.assert_array_equal(a.body[:, :2], actual)
            np.testing.assert_allclose(regret, cost - costs.min(1))

    def test_raw_recovery_and_missing_value_erasure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            archive = Archive(Path(directory) / "test.sqlite")
            memory = Memory(1)
            image = np.full((1, 3, 6, 6), 41, np.int8)
            ids = archive.append(0, image)
            memory.append(0, image, ids)
            memory.apply(np.array([5]), np.array([0]), 4, archive)
            self.assertEqual(int(memory.cache.sum()), 0)
            memory.feedback(np.array([0.2]), np.array([1]))
            memory.apply(np.array([1]), np.array([0]), 8, archive)
            np.testing.assert_array_equal(memory.cache[0, 0], image[0])
            self.assertEqual(memory.returns, 1)
            self.assertEqual(archive.reads, 1)
            archive.close()

    def test_only_executed_outcome_gets_gradient(self):
        states = torch.randn(4, 3, 2, requires_grad=True)
        contacts = torch.randn(4, 3, requires_grad=True)
        actions = torch.tensor([0, 1, 2, 0])
        loss, _ = loss_for(states, contacts, actions, torch.zeros(4, 2), torch.zeros(4))
        loss.sum().backward()
        for i, action in enumerate(actions):
            for other in range(3):
                if other != action:
                    self.assertEqual(float(states.grad[i, other].abs().sum()), 0)
                    self.assertEqual(float(contacts.grad[i, other].abs().sum()), 0)

    def test_inaccessible_parameter_groups_cannot_affect_prediction(self):
        torch.manual_seed(77)
        model = Predictor(9).eval()
        memory = Memory(8)
        image, sensor = Stream(8, 33).observe()
        batch = memory.tensors(0, image, sensor, "cpu")
        admitted = torch.full((8,), 2)
        boost = torch.zeros(8, dtype=torch.bool)
        first = model(batch, admitted, boost)[0].detach()
        with torch.no_grad():
            model.w1[2:].fill_(999)
            model.w2[2:].fill_(-999)
        second = model(batch, admitted, boost)[0].detach()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertGreater(model.forward_macs(8, 8), model.forward_macs(8, 0))

    def test_controller_masks_and_parameter_scale(self):
        image, sensor = Stream(4, 101).observe()
        memory = Memory(4)
        memory.append(0, image, np.arange(4))
        for mode in ("independent", "immediate", "delayed"):
            ctrl = Controller(mode)
            inputs = controller_inputs(memory, 0, sensor, np.full(4, 2), np.zeros(4),
                                       np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), 9, "cpu")
            for _ in range(10):
                op, pointer, logp, value, entropy = ctrl.act(*inputs)
                self.assertFalse(bool(((op == 1) | (op == 2) | (op == 6)).any()))
                self.assertTrue(bool(torch.isfinite(logp).all()))
        small, large = Predictor(9), Predictor(78)
        self.assertGreater(sum(p.numel() for p in large.parameters()), 7 * sum(p.numel() for p in small.parameters()))
        self.assertLess(large.active_parameters(False), 2 * small.active_parameters(False))

    def test_delayed_feedback_reaches_first_choice(self):
        def gradient(mode):
            parameter = torch.nn.Parameter(torch.tensor(0.))
            optimizer = torch.optim.SGD([parameter], lr=0.)
            records = [[parameter, parameter * 0, parameter * 0, torch.tensor(0.)],
                       [parameter * 0, parameter * 0, parameter * 0, torch.tensor(1.)]]
            update_controller(optimizer, records, torch.tensor(0.), mode)
            return float(parameter.grad)
        self.assertEqual(gradient("immediate"), 0.)
        self.assertLess(gradient("delayed"), 0.)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
