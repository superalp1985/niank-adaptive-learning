"""Contracts for reversible access, local evidence, and detached learning."""

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    learner = importlib.import_module("tracking_v19.learner")
except ModuleNotFoundError as error:
    if error.name != "tracking_v19.learner":
        raise
    learner = None

from tracking_v17.learner import Agent as AncestorAgent

EYES = 128
torch.set_num_threads(2)


def proprio(shifts):
    values = np.zeros((len(shifts), 6), np.float32)
    values[:, 2] = np.asarray(shifts) / 64
    return values


def observe(memory, shifts, reset=None, rgb=None):
    count = len(shifts)
    return memory.features(
        np.zeros((count, EYES, 3), np.uint8) if rgb is None else rgb,
        proprio(shifts),
        np.zeros(count, bool) if reset is None else np.asarray(reset, bool))


def clone_state(agent):
    return {name: {key: value.clone() for key, value in state.items()}
            for name, state in agent.state().items()}


class AvailableContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(learner, "V19 learner is not implemented")


class MemoryContracts(AvailableContracts):
    def test_input_group_feedback_is_not_shifted_into_output_coordinates(self):
        memory = learner.EyeMemory(1, "utility", 1)
        observe(memory, [3])
        memory.feedback(np.array([[.2, .4, .6]], np.float32), np.array([16]))
        self.assertTrue(np.allclose(memory.utility[0, 16:32], .04))
        self.assertEqual(np.count_nonzero(memory.utility), 16)
        self.assertTrue(np.allclose(memory.deviation[0, 16:32], .016329932))

    def test_old_evidence_and_selection_rotate_with_each_body(self):
        memory = learner.EyeMemory(2, "utility", 2)
        memory.utility[:, 0] = [.6, -.3]
        memory.deviation[:, 0] = [.2, .1]
        observe(memory, [1, -2])
        self.assertAlmostEqual(float(memory.utility[0, 127]), .6)
        self.assertAlmostEqual(float(memory.utility[1, 2]), -.3)
        self.assertAlmostEqual(float(memory.deviation[0, 127]), .2)
        self.assertAlmostEqual(float(memory.deviation[1, 2]), .1)

    def test_untested_groups_keep_existing_evidence(self):
        memory = learner.EyeMemory(1, "protected", 3)
        memory.utility[:] = .7
        memory.deviation[:] = .2
        memory.feedback(np.array([[0., 0., 0.]], np.float32), np.array([0]))
        self.assertTrue(np.allclose(memory.utility[0, :16], .63))
        self.assertTrue(np.allclose(memory.utility[0, 16:], .7))
        self.assertTrue(np.allclose(memory.deviation[0, 16:], .2))
        self.assertEqual(memory.updates, 1)

    def test_reset_only_clears_that_body_and_is_not_a_promotion(self):
        memory = learner.EyeMemory(2, "protected", 4)
        memory.utility[:] = .2
        memory.deviation[:] = .1
        memory.history[:] = .7
        memory.previous[:] = .4
        memory.age[:] = 10
        memory.selected[:] = True
        observe(memory, [2, 0], [True, False])
        self.assertFalse(memory.utility[0].any())
        self.assertFalse(memory.deviation[0].any())
        self.assertTrue(np.allclose(memory.utility[1], .2))
        self.assertFalse(memory.full_features[0, 3:9].any())
        self.assertFalse(memory.promoted[0].any())

    def test_each_body_has_56_ranked_and_8_disjoint_exploratory_accesses(self):
        memory = learner.EyeMemory(3, "utility", 5)
        memory.utility[:, :56] = 1
        observe(memory, [0, 0, 0])
        self.assertTrue((memory.selected.sum(1) == 64).all())
        self.assertTrue((memory.explored.sum(1) == 8).all())
        self.assertTrue(memory.selected[:, :56].all())
        self.assertFalse(memory.explored[:, :56].any())
        self.assertTrue((memory.explored <= memory.selected).all())

    def test_protection_can_promote_disputed_group_at_equal_mean_utility(self):
        base = learner.EyeMemory(1, "utility", 6)
        protected = learner.EyeMemory(1, "protected", 6)
        for memory in (base, protected):
            memory.utility[0, :56] = .1
            memory.deviation[0, 64:80] = .5
            observe(memory, [0])
        self.assertEqual(base.selected[0, 64:80].sum() <= 8, True)
        self.assertTrue(protected.selected[0, 64:80].all())

    def test_mask_does_not_alter_current_previous_or_age(self):
        memory = learner.EyeMemory(1, "utility", 7)
        memory.previous[:] = .4
        memory.history[:] = .8
        memory.age[:] = 12
        x = observe(memory, [0], rgb=np.full((1, 128, 3), 153, np.uint8))
        self.assertTrue(np.array_equal(x[:, :6], memory.full_features[:, :6]))
        self.assertTrue(np.array_equal(x[:, 10], memory.full_features[:, 10]))
        self.assertFalse(x[0, 6:10, ~memory.selected[0]].any())
        self.assertTrue(memory.full_features[0, 9, ~memory.selected[0]].any())

    def test_policy_mutation_cannot_corrupt_full_feature_cache(self):
        memory = learner.EyeMemory(1, "utility", 8)
        x = observe(memory, [0])
        expected = memory.full_features.copy()
        x[:] = 91
        self.assertTrue(np.array_equal(memory.full_features, expected))

    def test_trace_update_is_independent_of_access_selection(self):
        memories = [learner.EyeMemory(1, "utility", 9),
                    learner.EyeMemory(1, "utility", 10)]
        for memory in memories:
            memory.previous[:] = .2
            memory.history[:] = .8
            memory.age[:] = 4
            observe(memory, [0], rgb=np.full((1, 128, 3), 153, np.uint8))
        self.assertFalse(np.array_equal(memories[0].selected, memories[1].selected))
        self.assertTrue(np.array_equal(memories[0].history, memories[1].history))
        self.assertTrue((memories[0].history > .6).all())
        for memory in memories:
            for name in ("previous", "history", "age", "utility", "deviation"):
                self.assertEqual(getattr(memory, name).dtype, np.float32)

    def test_promotions_compare_with_rotated_previous_selection(self):
        memory = learner.EyeMemory(1, "utility", 11)
        observe(memory, [0])
        before = memory.selected.copy()
        observe(memory, [4])
        expected = memory.selected & ~np.roll(before, -4, axis=1)
        self.assertTrue(np.array_equal(memory.promoted, expected))

    def test_fixed_seed_exploration_revisits_every_low_rank_eye(self):
        memory = learner.EyeMemory(1, "utility", 12)
        memory.utility[0, :56] = 1
        visited = np.zeros(128, bool)
        for _ in range(256):
            observe(memory, [0])
            visited |= memory.explored[0]
        self.assertTrue(visited[56:].all())
        self.assertFalse(visited[:56].any())

    def test_calibration_feedback_cannot_change_evidence(self):
        memory = learner.EyeMemory(1, "calibration", 13)
        memory.feedback(np.ones((1, 3), np.float32), np.array([0]))
        self.assertFalse(memory.utility.any())
        self.assertFalse(memory.deviation.any())
        self.assertEqual(memory.updates, 0)

    def test_audit_schedule_covers_every_group_for_each_body(self):
        schedule = np.stack([learner.probe_starts(3, step) for step in range(8)])
        self.assertTrue(np.array_equal(schedule[0], [0, 48, 96]))
        for body in range(3):
            self.assertTrue(np.array_equal(np.sort(schedule[:, body]),
                                           np.arange(0, 128, 16)))


class PredictorContracts(AvailableContracts):
    def setUp(self):
        super().setUp()
        torch.manual_seed(19)
        self.agent = learner.Agent("utility", "cpu")

    def fixture(self, count=2):
        generator = torch.Generator().manual_seed(1901)
        x = torch.rand((count, 11, 128), generator=generator)
        prop = torch.zeros((count, 6))
        actions = torch.arange(count) % 9
        starts = np.array([0, 112][:count], np.int64)
        after = np.full((count, 128, 3), 102, np.uint8)
        return x, prop, actions, starts, after

    def test_bootstrap_identity_is_stable_across_order_and_calls(self):
        body, time, segment = np.arange(2048), np.arange(2048) * 3, np.arange(2048) % 7
        mask = learner.bootstrap_mask(body, time, segment)
        again = learner.bootstrap_mask(body[::-1], time[::-1], segment[::-1])[::-1]
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, (2048, 3))
        self.assertTrue(np.array_equal(mask, again))
        self.assertTrue(((mask.mean(0) > .75) & (mask.mean(0) < .85)).all())
        self.assertFalse(np.array_equal(mask[:, 0], mask[:, 1]))
        self.assertTrue(np.array_equal(mask, learner.bootstrap_mask(body, time, segment)))

    def test_local_omission_has_only_24_eye_receptive_field_including_seam(self):
        x, prop, action, starts, _ = self.fixture()
        normal, omitted = self.agent.predict(x, prop, action, starts)
        difference = (normal - omitted).abs().sum((1, 2))
        for row, outside in ((0, np.arange(20, 124)), (1, np.arange(4, 108))):
            self.assertEqual(float(difference[row, outside].max()), 0.)
            self.assertGreater(float(difference[row].sum()), 0.)
        erased = self.agent.erased(x, starts)
        self.assertTrue(torch.equal(erased[:, :6], x[:, :6]))
        self.assertEqual(int((erased[0, 6:11] == 0).sum()), 5 * 16)

    def test_evidence_matches_member_local_errors_and_never_updates_weights(self):
        x, prop, action, starts, after = self.fixture()
        before = clone_state(self.agent)
        result = self.agent.evidence(x, prop, action, starts, after, .01)
        with torch.no_grad():
            normal, omitted = self.agent.predict(x, prop, action, starts)
            label = torch.from_numpy(after).float().permute(0, 2, 1) / 255
            full = (normal[0, :, :, list(range(124, 128)) + list(range(20))] -
                    label[0, :, list(range(124, 128)) + list(range(20))]).square().mean((1, 2))
            lost = (omitted[0, :, :, list(range(124, 128)) + list(range(20))] -
                    label[0, :, list(range(124, 128)) + list(range(20))]).square().mean((1, 2))
        self.assertTrue(np.allclose(result["normal_local"][0], full.numpy()))
        self.assertTrue(np.allclose(result["omitted_local"][0], lost.numpy()))
        self.assertTrue(np.allclose(result["scores"][0],
                                    ((lost - full) / (lost + full + .01)).numpy()))
        self.assertEqual(result["scores"].shape, (2, 3))
        for name, state in before.items():
            for key, tensor in state.items():
                self.assertTrue(torch.equal(tensor, self.agent.state()[name][key]))
        self.assertTrue(all(p.grad is None for module in self.agent.modules
                            for p in module.parameters()))

    def test_ancestor_loading_is_reproducible_and_members_start_diverse(self):
        torch.manual_seed(91)
        ancestor = AncestorAgent("revisable", "cpu").state()
        self.agent.load_ancestor(ancestor)
        torch.manual_seed(92)
        other = learner.Agent("protected", "cpu")
        other.load_ancestor(ancestor)
        for name, state in self.agent.state().items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, other.state()[name][key]))
        for key, value in ancestor["predictor"].items():
            self.assertTrue(torch.equal(value, self.agent.predictors[0].state_dict()[key]))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(self.agent.predictors[0].parameters(),
                                self.agent.predictors[1].parameters())))

    def test_state_round_trip_preserves_all_ensemble_predictions(self):
        other = learner.Agent("protected", "cpu")
        other.load(self.agent.state())
        x, prop, action, starts, _ = self.fixture()
        expected = self.agent.predict(x, prop, action, starts)
        actual = other.predict(x, prop, action, starts)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        self.assertEqual(len(other.optimizers), 9)

    def test_predictor_gradient_does_not_cross_into_inputs_or_other_modules(self):
        x, prop, action, _, _ = self.fixture()
        x.requires_grad_()
        prop.requires_grad_()
        loss = self.agent.predictors[0](x, prop, action).square().mean()
        self.agent.local_step(self.agent.predictors[0], self.agent.optimizers[6], loss)
        self.assertIsNone(x.grad)
        self.assertIsNone(prop.grad)
        for module in self.agent.modules[:6] + self.agent.modules[7:]:
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_update_uses_persistent_member_masks_and_counts_actual_presentations(self):
        x, prop, _, starts, after = self.fixture()
        selected = self.agent.act(x.numpy(), prop.numpy())
        data = {
            "features": np.stack([x.numpy()] * 2),
            "full_features": np.stack([x.numpy()] * 2),
            "proprio": np.stack([prop.numpy()] * 2),
            "reward": np.array([[.3, -.2], [.1, .4]], np.float32),
            "done": np.zeros((2, 2), bool),
            "next_rgb": np.stack([after] * 2),
            "starts": np.stack([starts] * 2),
            "bootstrap": np.array([[[True, False, True], [False, False, True]],
                                   [[True, False, False], [False, False, False]]]),
        }
        for key in ("actions", "groups", "action_logp", "route_logp", "value"):
            data[key] = np.stack([selected[key].numpy()] * 2)
        skipped = {key: value.clone() for key, value in self.agent.predictors[1].state_dict().items()}
        loss = self.agent.update(data, x.numpy(), prop.numpy(), epochs=2)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(self.agent.presentations, 8)
        self.assertEqual(self.agent.predictor_presentations, 16)
        self.assertTrue(all(torch.equal(tensor, self.agent.predictors[1].state_dict()[key])
                            for key, tensor in skipped.items()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
