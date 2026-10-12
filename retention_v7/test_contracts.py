from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiment import (
    CACHE, EventRetentionModel, archive, exposure_data, generate, metrics,
    recovery_probe, seed_all, training_loss,
)


class EventContracts(unittest.TestCase):
    def test_exposure_has_matched_body_counts_and_real_context_truth(self):
        base = generate(8, 1800000)
        ordinary, paired = exposure_data(base, False), exposure_data(base, True)
        np.testing.assert_array_equal(ordinary.images, paired.images)
        np.testing.assert_array_equal(ordinary.body, paired.body)
        self.assertEqual(len(ordinary), len(paired))
        np.testing.assert_array_equal(paired.context[:8], 0)
        np.testing.assert_array_equal(paired.context[8:], 1)
        self.assertGreater(float(np.abs(paired.states[:8] - paired.states[8:]).mean()), 0)

    def test_all_parameters_used(self):
        batch = exposure_data(generate(8, 1801000), True).tensors("cpu")
        for gated, mode in ((False, "paired_plain"), (True, "paired_event")):
            model = EventRetentionModel(gated)
            output = model(batch, mode, replay=True)
            loss, _ = training_loss(output, batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual([name for name, p in model.named_parameters() if p.grad is None], [])
            self.assertEqual(output["single_events"]["states"].shape, (16, 18, 3, 2))

    def test_outcome_and_cue_metadata_isolation(self):
        batch = generate(8, 1802000).tensors("cpu")
        model = EventRetentionModel(True).eval()
        first = model(batch, "paired_event")
        for name in ("states", "contacts", "costs", "cue_times"):
            batch[name] = batch[name] + 100
        second = model(batch, "paired_event")
        for name in ("action", "states", "indices", "active_mask", "utility", "gates"):
            torch.testing.assert_close(first[name], second[name])

    def test_no_positive_value_means_empty_memory(self):
        base = generate(8, 1803000)
        model = EventRetentionModel(False)
        with torch.no_grad():
            model.value[-1].weight.zero_()
            model.value[-1].bias.fill_(-1)
        result, _ = metrics(model, "ordinary_plain", base)
        self.assertEqual(result["active_count"], 0)
        self.assertEqual(result["read_count"], 0)
        output = model(base.tensors("cpu"), "ordinary_plain")
        self.assertTrue(torch.isfinite(output["states"]).all())

    def test_true_archive_recovery_and_no_updates(self):
        base = generate(8, 1804000, delay=26)
        model = EventRetentionModel(True).eval()
        before = {n: p.clone() for n, p in model.state_dict().items()}
        with tempfile.TemporaryDirectory(dir=CACHE) as directory:
            path = Path(directory) / "archive.sqlite"
            archive(path, base)
            saved = path.read_bytes()
            result, _ = recovery_probe(model, "paired_event", base, path)
            self.assertEqual(result["recovery_database_reads"]["records"], 208)
            self.assertLessEqual(result["stale_database_reads"]["records"], 32)
            self.assertEqual(saved, path.read_bytes())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(before[name], value)

    def test_shared_initializations(self):
        seed_all(11)
        first = EventRetentionModel(False)
        seed_all(11)
        second = EventRetentionModel(True)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
