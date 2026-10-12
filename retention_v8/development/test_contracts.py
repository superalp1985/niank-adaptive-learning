from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiment import (
    CACHE, OutcomeRetentionModel, archive, generate, metrics,
    outcome_difference_loss, quartet_data, recovery_probe, training_loss,
)


class OutcomeContracts(unittest.TestCase):
    def test_quartet_truth_matches_actual_state_differences(self):
        base = generate(8, 2500000)
        data = quartet_data(base)
        states = data.states.reshape(4, 8, 3, 2)
        np.testing.assert_allclose(states[1] - states[0], 0, atol=1e-6)
        np.testing.assert_allclose(np.abs(states[3, ..., 1] - states[2, ..., 1]), 0.36, atol=1e-6)
        batch = data.tensors("cpu")
        self.assertEqual(float(outcome_difference_loss({"states": batch["states"]}, batch)), 0)

    def test_both_objectives_use_all_parameters_without_target_gradients(self):
        batch = quartet_data(generate(8, 2501000)).tensors("cpu")
        for mode in ("quartet_point", "quartet_delta"):
            model = OutcomeRetentionModel()
            output = model(batch, mode, replay=True)
            self.assertFalse(output["omitted"]["states"].requires_grad)
            loss, _ = training_loss(output, batch, mode)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual([n for n, p in model.named_parameters() if p.grad is None], [])

    def test_labels_do_not_enter_frozen_prediction(self):
        batch = generate(8, 2502000).tensors("cpu")
        model = OutcomeRetentionModel()
        first = model(batch, "quartet_delta")
        for name in ("states", "costs", "contacts", "cue_times"):
            batch[name] += 10
        second = model(batch, "quartet_delta")
        for name in ("action", "states", "utility", "indices", "active_mask", "gates"):
            torch.testing.assert_close(first[name], second[name])

    def test_archive_recovery_is_read_only(self):
        data = generate(8, 2503000, delay=26)
        model = OutcomeRetentionModel().eval()
        saved = {n: value.clone() for n, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=CACHE) as directory:
            path = Path(directory) / "archive.sqlite"
            archive(path, data)
            before = path.read_bytes()
            result, _ = recovery_probe(model, "quartet_delta", data, path)
            self.assertEqual(result["recovery_database_reads"]["records"], 208)
            self.assertEqual(before, path.read_bytes())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(saved[name], value)

    def test_matched_initial_models_have_equal_parameter_count(self):
        a, b = OutcomeRetentionModel(), OutcomeRetentionModel()
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        self.assertEqual(metrics(a, "quartet_point", generate(8, 2504000))[0]["event_records_scored"], 18)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
