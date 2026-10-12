"""Scientific contracts: causality, archive preservation, scoring, gradients."""

import tempfile
import unittest
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import torch

from experiment import (
    CACHE, MODES, RetentionModel, archive, evaluate, generate, physics,
    select_records, training_loss,
)


class ExperimentContracts(unittest.TestCase):
    def test_nested_training_and_disjoint_ids(self):
        small, large = generate(8, 1000), generate(16, 1000)
        np.testing.assert_array_equal(small.images, large.images[:8])
        np.testing.assert_array_equal(small.states, large.states[:8])
        self.assertFalse(set(small.ids) & set(generate(8, 2000).ids))

    def test_contact_is_future_truth_not_input(self):
        dataset = generate(48, 8100)
        self.assertEqual(dataset.body.shape[1], 5)
        self.assertEqual(dataset.context.shape[1], 1)
        self.assertGreater(dataset.contacts.sum(), 0)
        self.assertLess(dataset.contacts.sum(), dataset.contacts.size)
        # Change contact labels only. All evaluation predictions must stay fixed.
        model = RetentionModel().eval()
        first = model(dataset.tensors("cpu"), "value")
        changed = dataset.tensors("cpu")
        changed["contacts"] = 1 - changed["contacts"]
        changed["costs"] = torch.flip(changed["costs"], (-1,))
        changed["cue_times"] = changed["cue_times"] + 5
        second = model(changed, "value")
        for name in ("action", "states", "contacts", "indices", "utility"):
            torch.testing.assert_close(first[name], second[name])
        self.assertNotIn("replay", second)

    def test_fine_cue_smaller_than_nuisance_but_changes_physics(self):
        dataset = generate(32, 8200)
        body = dataset.body
        risk = np.ones(len(dataset), np.float32)
        first, _, _ = physics(body, np.full(len(dataset), 0.9), np.full(len(dataset), -0.18), risk)
        second, _, _ = physics(body, np.full(len(dataset), 0.9), np.full(len(dataset), 0.18), risk)
        self.assertGreater(np.abs(first - second).mean(), 0.05)
        self.assertLess(np.abs(dataset.images[:, :, 1]).max(), dataset.images[:, :, 2].mean())

    def test_priority_buffer_has_causal_prefix_equivalence(self):
        priorities = torch.tensor([[0.6, 0.9, 0.1, 0.4, 0.7, 0.3]])
        retained = []
        for step in range(priorities.shape[1]):
            retained.append(step)
            retained = sorted(retained, key=lambda i: float(priorities[0, i]), reverse=True)[:3]
            expected = select_records(priorities[:, :step + 1], 3)[0].tolist()
            self.assertEqual(sorted(retained), expected)

    def test_archive_has_observations_not_targets(self):
        dataset = generate(4, 8400)
        with tempfile.TemporaryDirectory(dir=CACHE) as temporary:
            path = Path(temporary) / "archive.sqlite"
            archive(path, dataset)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0], 72)
                columns = [row[1] for row in connection.execute("PRAGMA table_info(episodes)")]
                self.assertNotIn("contacts", columns)
                blob = connection.execute("SELECT observation FROM events WHERE episode=? AND time=2", (int(dataset.ids[0]),)).fetchone()[0]
                np.testing.assert_array_equal(np.frombuffer(blob, np.float32), dataset.images[0, 2].ravel())

    def test_scientific_metrics_accept_equal_cost_answers(self):
        dataset = generate(32, 8500)
        dataset.costs[:] = 0.5
        for mode in MODES:
            metrics, _ = evaluate(RetentionModel(), mode, dataset, device="cpu")
            self.assertEqual(metrics["success"], 1.0)
            self.assertEqual(metrics["regret"], 0.0)

    def test_all_parameters_used_and_counterfactuals_detached(self):
        dataset = generate(16, 8600)
        parameters = []
        for mode in MODES:
            model = RetentionModel()
            batch = dataset.tensors("cpu")
            output = model(batch, mode, replay=True)
            self.assertFalse(output["omitted"]["action"].requires_grad)
            loss, _ = training_loss(output, batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
            self.assertEqual(unused, [], (mode, unused))
            self.assertGreater(float(model.value[-1].weight.grad.abs().sum()), 0)
            parameters.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(parameters)), 1)

    def test_fine_sign_intervention_has_context_dependent_truth(self):
        from probe_details import construct
        base = generate(32, 8700, delay=26)
        for risk, expected in ((0.0, 0.0), (1.0, 0.36)):
            original = construct(base, risk, False)
            flipped = construct(base, risk, True)
            np.testing.assert_array_equal(original.body, flipped.body)
            magnitude = np.abs(flipped.states[..., 1] - original.states[..., 1])
            np.testing.assert_allclose(magnitude, expected, atol=1e-6)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
