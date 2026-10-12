"""Small CPU contract tests; no CUDA experiment is launched by this module."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from learning_experiment import (
    Config, OBJECTIVES, PlainGRU, RateRNN, has_heldout_pair, load_matrices,
    loss_terms, make_splits, rollout, row_normalize, task_logits, transition,
)


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(matrix_path="unused", output_dir="unused")
        cls.train, cls.validation, cls.test = make_splits(cls.config)

    def test_independent_transition_truth_and_wraparound(self):
        self.assertEqual(transition(3, 0), 0)
        self.assertEqual(transition(0, 1), 3)
        self.assertEqual(transition(12, 2), 0)
        self.assertEqual(transition(0, 3), 12)
        self.assertEqual(rollout(0, (0, 2, 1, 3)), (1, 5, 4, 0))

    def test_all_splits_are_disjoint_unique_and_nested(self):
        datasets = [self.train, *self.validation.values(), *self.test.values()]
        all_ids = [identity for dataset in datasets for identity in dataset.identities()]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(self.train), 1024)
        self.assertEqual(self.train.take(slice(0, 64)).identities(),
                         self.train.take(slice(0, 256)).identities()[:64])

    def test_heldout_pair_and_length_are_separate_axes(self):
        self.assertTrue(all(not has_heldout_pair(tuple(row[1:]))
                            for row in self.train.identities()))
        for collection in (self.validation, self.test):
            for scenario, data in collection.items():
                self.assertTrue(all(has_heldout_pair(tuple(identity[1:])) == ("unseen" in scenario)
                                    for identity in data.identities()))
                lengths = set(data.lengths.tolist())
                self.assertLessEqual(lengths, {5, 6} if scenario.startswith("long") else {3, 4})

    def test_generation_is_reproducible_and_references_are_exact(self):
        repeated, _, _ = make_splits(self.config)
        self.assertEqual(self.train.fingerprint(), repeated.fingerprint())
        for dataset in [self.train, *self.test.values()]:
            for start, actions, length, states in zip(dataset.starts, dataset.actions,
                                                       dataset.lengths, dataset.states):
                self.assertEqual(rollout(int(start), tuple(actions[:length])), tuple(states[:length]))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[0., 2., -1.], [3., 0., 0.], [0., 0., 0.]], dtype=np.float32)
        self.config = Config(matrix_path="unused", output_dir="unused")

    def test_row_normalization_keeps_empty_row_and_sign(self):
        normalized = row_normalize(self.matrix)
        np.testing.assert_allclose(np.abs(normalized).sum(axis=1), [1., 1., 0.])
        np.testing.assert_array_equal(np.sign(normalized), np.sign(self.matrix))

    def test_learned_edges_preserve_sign_topology_and_row_norm(self):
        torch.manual_seed(5)
        model = RateRNN(self.matrix)
        np.testing.assert_allclose(model.recurrent_weight().detach(), row_normalize(self.matrix), atol=1e-7)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        actions = torch.tensor([[0, 1, 2], [2, 3, 0]])
        lengths = torch.tensor([3, 3])
        for _ in range(2):
            logits, _ = model(F.one_hot(torch.tensor([0, 5]), 16).float(), actions, lengths)
            loss = logits.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        weight = model.recurrent_weight().detach().numpy()
        np.testing.assert_array_equal(np.sign(weight), np.sign(self.matrix))
        np.testing.assert_allclose(np.abs(weight).sum(axis=1), [1., 1., 0.], atol=1e-6)

    def test_padding_cannot_change_terminal_state(self):
        for model in (RateRNN(self.matrix), PlainGRU(3)):
            frame = F.one_hot(torch.tensor([0, 2]), 16).float()
            lengths = torch.tensor([2, 3])
            actions = torch.tensor([[0, 1, -1, -1], [2, 3, 0, -1]])
            logits, hidden = model(frame, actions, lengths)
            torch.testing.assert_close(hidden[0, 1], hidden[0, 3])
            torch.testing.assert_close(hidden[1, 2], hidden[1, 3])
            torch.testing.assert_close(task_logits(logits, lengths),
                                       torch.stack((logits[0, 1], logits[1, 2])))

    def test_pair_has_equal_input_and_readout_initialization(self):
        alternate = self.matrix.T.copy()
        torch.manual_seed(9)
        first = RateRNN(self.matrix)
        torch.manual_seed(9)
        second = RateRNN(alternate)
        torch.testing.assert_close(first.initial.weight, second.initial.weight)
        torch.testing.assert_close(first.action_input.weight, second.action_input.weight)
        torch.testing.assert_close(first.readout.weight, second.readout.weight)

    def test_auxiliary_prediction_excludes_final_target(self):
        logits = torch.randn(2, 3, 16, requires_grad=True)
        hidden = torch.randn(2, 3, 3, requires_grad=True)
        states = torch.tensor([[1, 5, 4], [6, 2, 3]])
        lengths = torch.tensor([3, 3])
        first = loss_terms(logits, hidden, states, lengths, "task_predict", self.config)
        changed = states.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 16
        second = loss_terms(logits, hidden, changed, lengths, "task_predict", self.config)
        torch.testing.assert_close(first["prediction"], second["prediction"])
        self.assertFalse(torch.equal(first["task"], second["task"]))

    def test_all_objectives_finite_and_consistency_rejects_missing_view(self):
        model = RateRNN(self.matrix)
        frame = F.one_hot(torch.tensor([0, 5]), 16).float()
        actions = torch.tensor([[0, 1, 2], [2, 3, 0]])
        lengths = torch.tensor([3, 3])
        states = torch.tensor([rollout(0, (0, 1, 2)), rollout(5, (2, 3, 0))])
        logits, hidden = model(frame, actions, lengths)
        _, other = model(frame + .01, actions, lengths)
        for objective in OBJECTIVES:
            terms = loss_terms(logits, hidden, states, lengths, objective, self.config, other)
            self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        with self.assertRaises(ValueError):
            loss_terms(logits, hidden, states, lengths, "full", self.config)

    def test_matrix_loader_rejects_nonmatched_rewiring(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp) / "graph.npz"
            np.savez(location, ids=np.arange(3), matrix=self.matrix, rewired_matrix=self.matrix)
            _, metadata = load_matrices(str(location))
            self.assertTrue(metadata["signed_degree_preserved"])
            broken = self.matrix.copy()
            broken[0, 0] = 1
            np.savez(location, ids=np.arange(3), matrix=self.matrix, rewired_matrix=broken)
            with self.assertRaises(ValueError):
                load_matrices(str(location))


if __name__ == "__main__":
    unittest.main()
