"""Guard against label leakage, unused parameters and fake archive recovery."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiment import (
    CACHE, StateRetentionModel, archive, generate, intervention, load_observations,
    metrics, training_loss, with_context,
)


class StateContracts(unittest.TestCase):
    def test_context_and_fine_sign_change_truth_not_body(self):
        base = generate(32, 950000, delay=26)
        safe, risk = with_context(base, 0.0), with_context(base, 1.0)
        np.testing.assert_array_equal(safe.images, risk.images)
        np.testing.assert_array_equal(safe.body, risk.body)
        self.assertGreater(float(np.abs(safe.states - risk.states).mean()), 0.0)
        for context, expected in ((0.0, 0.0), (1.0, 0.36)):
            first, second = with_context(base, context), with_context(base, context, True)
            np.testing.assert_allclose(np.abs(second.states[..., 1] - first.states[..., 1]), expected, atol=1e-6)

    def test_every_constructed_parameter_receives_training_gradients(self):
        data = generate(16, 951000).tensors("cpu")
        for gated, mode in ((False, "state_nogate"), (True, "state_gate")):
            model = StateRetentionModel(gated)
            output = model(data, mode, replay=True)
            self.assertEqual(output["single_events"]["states"].shape, (16, 18, 3, 2))
            loss, _ = training_loss(output, data)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual([name for name, p in model.named_parameters() if p.grad is None], [])
            if gated:
                self.assertGreater(float(model.gate[-1].weight.grad.abs().sum()), 0.0)
            else:
                self.assertFalse(hasattr(model, "gate"))

    def test_future_outcomes_and_cue_metadata_do_not_enter_predictions(self):
        batch = generate(16, 952000).tensors("cpu")
        model = StateRetentionModel(True).eval()
        a = model(batch, "state_gate")
        batch["states"] = batch["states"] + 10
        batch["contacts"] = 1 - batch["contacts"]
        batch["costs"] = -batch["costs"]
        batch["cue_times"] = batch["cue_times"] + 10
        b = model(batch, "state_gate")
        for name in ("action", "states", "contacts", "indices", "utility", "gate"):
            torch.testing.assert_close(a[name], b[name])

    def test_database_read_recovers_discarded_images(self):
        base = generate(4, 953000, delay=26)
        with tempfile.TemporaryDirectory(dir=CACHE) as directory:
            path = Path(directory) / "archive.sqlite"
            archive(path, base)
            active = np.tile(np.asarray([0, 10, 11, 12]), (4, 1))
            stale, io = load_observations(path, base.ids, 26, active_indices=active)
            self.assertEqual(io["records"], 16)
            self.assertEqual(float(np.abs(stale[:, 1:5, :2]).sum()), 0.0)
            original = base.images.copy()
            base.images[:] = 0
            restored, io = load_observations(path, base.ids, 26)
            np.testing.assert_array_equal(original, restored)
            self.assertEqual(io["records"], 104)
            self.assertEqual(io["bytes"], 104 * 3 * 6 * 6 * 4)

    def test_recovery_is_read_only_and_counts_database_io(self):
        base = generate(8, 954000, delay=26)
        model = StateRetentionModel(True).eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=CACHE) as directory:
            path = Path(directory) / "archive.sqlite"
            archive(path, base)
            archive_before = path.read_bytes()
            result, arrays = intervention(model, "state_gate", base, path)
            self.assertEqual(result["recovery_database_reads"]["records"], 8 * 26)
            self.assertEqual(result["stale_database_reads"]["records"], 8 * 4)
            self.assertEqual(arrays["old_active"].shape, (8, 4))
            self.assertEqual(archive_before, path.read_bytes())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(before[name], value)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_identical_costs_allow_different_correct_answers(self):
        base = generate(8, 955000)
        base.costs[:] = 0.5
        result, _ = metrics(StateRetentionModel(False), "state_nogate", base)
        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["regret"], 0.0)

    def test_shared_modules_match_paired_seed_initialization(self):
        torch.manual_seed(19)
        first = StateRetentionModel(False)
        torch.manual_seed(19)
        second = StateRetentionModel(True)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name])

    def test_id_ranges_disjoint_and_prefix_nested(self):
        first, second = generate(4, 1000000), generate(8, 1000000)
        np.testing.assert_array_equal(first.images, second.images[:4])
        groups = [set(generate(8, seed).ids) for seed in (1000000, 1010000, 1020000, 1030000, 1200000, 1240000)]
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                self.assertFalse(a & b)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
