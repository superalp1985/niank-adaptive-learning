from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiment import (
    CACHE, Directory, factual_loss, load_model, load_observations, observation_batch,
    plan, slot_contributions, worlds, archive, physics, facts,
)


class PracticeContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_initial_body_does_not_depend_on_action_label(self):
        base = worlds(8, 3700000)
        for row, identity in enumerate(base.ids):
            rng = np.random.default_rng(int(identity) + 73000000)
            expected = (rng.uniform(0.15, 0.85), rng.uniform(-0.65, 0.65),
                        rng.uniform(0.15, 0.85), rng.uniform(0.4, 0.8),
                        -1.0 if row % 2 else 1.0)
            np.testing.assert_allclose(base.body[row], expected, atol=1e-7)

    def test_factual_loss_uses_only_the_executed_action(self):
        states = torch.randn(8, 3, 2)
        contacts = torch.randn(8, 3)
        actions = torch.ones(8, dtype=torch.long)
        truth = torch.randn(8, 2)
        contact_truth = torch.zeros(8)
        first, _ = factual_loss({"states": states, "contacts": contacts}, actions, truth, contact_truth)
        states[:, 0] += 100
        contacts[:, 2] += 100
        second, _ = factual_loss({"states": states, "contacts": contacts}, actions, truth, contact_truth)
        torch.testing.assert_close(first, second)

    def test_cached_input_really_erases_archived_images(self):
        base = worlds(4, 3701000)
        directory = Directory(base.images, np.zeros((4, 18)), "recurrent")
        live = directory.active_map()
        self.assertEqual(float(np.abs(directory.cached[~live]).sum()), 0)
        base.images[:] = 99
        self.assertFalse(np.any(directory.cached == 99))
        directory.fade(100)
        self.assertEqual(float(np.abs(directory.cached).sum()), 0)

    def test_faded_reference_can_return_through_real_database_read(self):
        base = worlds(4, 3702000)
        with tempfile.TemporaryDirectory(dir=CACHE) as directory_path:
            path = Path(directory_path) / "archive.sqlite"
            archive(path, base)
            directory = Directory(base.images, np.zeros((4, 18)), "consolidated")
            directory.fade(100)
            original = base.images.copy()
            base.images[:] = 0
            recovered, io = load_observations(path, base.ids, 18)
            gain = np.zeros((4, 18), np.float32)
            gain[:, :4] = 0.1
            directory.recover(101, np.arange(4), recovered, gain, np.full(4, 0.02))
            self.assertEqual(io["records"], 72)
            np.testing.assert_array_equal(directory.cached[:, :4], original[:, :4])
            self.assertTrue(directory.mask.all())

    def test_strength_can_rise_then_decay_without_permanent_class(self):
        base = worlds(4, 3703000)
        directory = Directory(base.images, np.zeros((4, 18)), "consolidated")
        directory.practice(1, np.full((4, 4), 0.1), np.full(4, 0.02))
        peak = directory.strength.copy()
        self.assertGreater(peak.sum(), 0)
        for step in range(2, 200):
            directory.practice(step, np.zeros((4, 4)), np.full(4, 0.02))
        self.assertLess(directory.strength.sum(), peak.sum() * 0.1)

    def test_retrieval_uses_observed_error_and_history_only(self):
        base = worlds(4, 3704000)
        directory = Directory(base.images, np.zeros((4, 18)), "recurrent")
        low = directory.request(0, np.zeros(4))
        high = directory.request(1, np.ones(4))
        self.assertFalse(low.any())
        self.assertTrue(high.all())

    def test_closed_loop_body_follows_chosen_physics(self):
        base = worlds(8, 3705000)
        state, _, _ = physics(base.body, *facts(base), np.ones(8))
        next_body = base.body.copy()
        actions = np.arange(8) % 3
        next_body[:, :2] = state[np.arange(8), actions]
        again, _, _ = physics(next_body, *facts(base), np.ones(8))
        self.assertGreater(float(np.abs(again - state).mean()), 0)

    def test_no_outcome_labels_in_controller_and_gradients_flow(self):
        base = worlds(4, 3706000)
        model = load_model(11, "cpu")
        batch = observation_batch(base.images, base.body, base.context, "cpu")
        self.assertEqual(set(batch), {"images", "body", "context"})
        output = model(batch, "quartet_delta")
        directory = Directory(base.images, output["utility"].detach().numpy(), "recurrent")
        batch = observation_batch(directory.cached, base.body, base.context, "cpu")
        prediction = model(batch, "quartet_delta", active_override=directory.override("cpu"))
        actions = plan(prediction, batch["body"])
        actual = torch.from_numpy(base.states[np.arange(4), actions.numpy()])
        contacts = torch.from_numpy(base.contacts[np.arange(4), actions.numpy()])
        loss, _ = factual_loss(prediction, actions, actual, contacts)
        gains = slot_contributions(model, batch, directory, actions, actual, contacts, loss.detach())
        self.assertFalse(gains.requires_grad)
        value = prediction["utility"].gather(1, directory.override("cpu")[0])
        total = loss.mean() + (value - gains).square().mean()
        total.backward()
        unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(unused, [])
        self.assertTrue(all(not p.requires_grad for p in model.action.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
