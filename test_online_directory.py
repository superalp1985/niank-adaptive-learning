"""Small CPU-only contract tests; inference experiments require CUDA."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from online_directory import OnlineDirectory


def unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


def small_directory() -> OnlineDirectory:
    return OnlineDirectory(3, max_cells=6, device="cpu", allow_cpu_test=True)


def fill(directory: OnlineDirectory) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4001)
    centers = np.eye(3, dtype=np.float32)
    keys = unit(
        np.concatenate(
            [center[None] + 0.07 * rng.normal(size=(8, 3)) for center in centers]
        )
    )
    ids = np.arange(100, 124, dtype=np.int64)
    for start in range(0, len(keys), 8):
        directory.observe(keys[start : start + 8], ids[start : start + 8])
    return keys, ids


class OnlineDirectoryContracts(unittest.TestCase):
    def assert_integrity(self, directory: OnlineDirectory) -> None:
        state = directory.snapshot()
        labels = state["membership"]
        counts = state["counts"]
        offsets = state["offsets"]
        ids = state["record_ids"]
        packed = state["packed_positions"]
        np.testing.assert_array_equal(np.bincount(labels, minlength=len(counts)), counts)
        np.testing.assert_array_equal(np.diff(offsets), counts)
        self.assertEqual(offsets[0], 0)
        self.assertEqual(offsets[-1], len(ids))
        np.testing.assert_array_equal(np.sort(packed), np.arange(len(ids)))
        np.testing.assert_array_equal(np.sort(state["packed_record_ids"]), np.sort(ids))
        expected = np.zeros_like(state["running_sums"])
        np.add.at(expected, labels, directory.keys.numpy())
        np.testing.assert_allclose(state["running_sums"], expected, atol=2e-6, rtol=2e-6)

    def test_future_batches_do_not_change_prefix_membership(self) -> None:
        left, right = small_directory(), small_directory()
        prefix = unit([[1, 0.1, 0], [1, -0.1, 0], [1, 0, 0.1]])
        ids = np.asarray([10, 20, 30])
        left.observe(prefix, ids)
        right.observe(prefix, ids)
        left_prefix = left.snapshot()
        right_prefix = right.snapshot()
        np.testing.assert_array_equal(left_prefix["centroids"], right_prefix["centroids"])
        np.testing.assert_array_equal(left_prefix["membership"], right_prefix["membership"])
        right.observe(unit([[0, 1, 0], [0, 1, 0.1]]), np.asarray([80, 90]))
        np.testing.assert_array_equal(
            right.record_cluster[: len(ids)], left_prefix["membership"]
        )
        self.assertEqual(left.n, 3)
        self.assertEqual(right.n, 5)
        self.assertEqual(left.growth_curve[0], right.growth_curve[0])

    def test_cells_grow_at_most_once_per_observe_call(self) -> None:
        directory = small_directory()
        keys, _ = fill(directory)
        self.assertEqual(len(directory.growth_curve), 3)
        self.assertLessEqual(directory.cell_count, 3)
        previous = 0
        for point in directory.growth_curve:
            self.assertLessEqual(point["cells"] - previous, 1)
            previous = point["cells"]
        self.assertEqual(directory.n, len(keys))
        self.assert_integrity(directory)

    def test_route_only_scores_selected_members_and_all_cells_match_dense(self) -> None:
        directory = small_directory()
        keys, ids = fill(directory)
        query = keys[10]
        sparse = directory.route(query, 1)
        labels = directory.record_cluster
        allowed = ids[np.isin(labels, sparse.selected_cells)]
        np.testing.assert_array_equal(np.sort(sparse.candidate_ids), np.sort(allowed))
        self.assertLess(sparse.candidate_count, directory.n)
        self.assertIn(sparse.best_id, sparse.candidate_ids)
        all_cells = directory.route(query, 999)
        dense_position = int(np.argmax(keys @ query))
        self.assertEqual(all_cells.best_id, int(ids[dense_position]))
        self.assertAlmostEqual(all_cells.similarity, float(keys[dense_position] @ query), places=6)
        self.assertEqual(all_cells.candidate_count, directory.n)
        self.assertGreater(all_cells.work, sparse.work)

    def test_stale_and_insufficient_full_budget_do_no_hidden_work(self) -> None:
        directory = small_directory()
        fill(directory)
        before = directory.snapshot()
        stale = directory.maintain("stale")
        denied = directory.maintain("full", directory.n - 1)
        self.assertFalse(stale.executed)
        self.assertEqual(stale.work, 0)
        self.assertFalse(denied.executed)
        self.assertEqual(denied.visited, 0)
        self.assertEqual(denied.work, 0)
        np.testing.assert_array_equal(directory.record_cluster, before["membership"])
        self.assertEqual(directory.work["maintenance"], 0)

    def test_full_reindex_matches_frozen_pre_operation_centers_and_costs(self) -> None:
        directory = small_directory()
        fill(directory)
        active = directory.active_cells
        centers = directory.centroids.numpy().copy()[active]
        expected = active[np.argmax(directory.keys.numpy() @ centers.T, axis=1)]
        result = directory.maintain("full", directory.n)
        np.testing.assert_array_equal(directory.record_cluster, expected)
        self.assertTrue(result.executed)
        self.assertEqual(result.visited, directory.n)
        self.assertGreater(result.work, 0)
        self.assertAlmostEqual(directory.work["maintenance"], result.work)
        self.assert_integrity(directory)

    def test_incremental_budget_and_round_robin_preserve_unvisited_records(self) -> None:
        directory = small_directory()
        fill(directory)
        original = directory.record_cluster
        first = directory.maintain("incremental", 5)
        self.assertEqual(first.visited, 5)
        self.assertEqual(directory._maintenance_cursor, 5)
        np.testing.assert_array_equal(directory.record_cluster[5:], original[5:])
        second = directory.maintain("incremental", 5)
        self.assertEqual(second.visited, 5)
        self.assertEqual(directory._maintenance_cursor, 10)
        np.testing.assert_array_equal(directory.record_cluster[10:], original[10:])
        last = directory.maintain("incremental", 14)
        self.assertEqual(last.visited, 14)
        self.assertEqual(directory._maintenance_cursor, 0)
        self.assertGreater(last.work, 0)
        self.assert_integrity(directory)

    def test_bounded_and_full_maintenance_move_stale_members(self) -> None:
        def drifting_directory() -> OnlineDirectory:
            directory = OnlineDirectory(
                3, max_cells=2, device="cpu", allow_cpu_test=True
            )
            for rid, degrees in enumerate((0, 90, 40)):
                angle = np.deg2rad(degrees)
                vector = np.asarray([[np.cos(angle), np.sin(angle), 0]], dtype=np.float32)
                directory.observe(vector, np.asarray([rid]))
            angle = np.deg2rad(-40)
            later = np.tile([np.cos(angle), np.sin(angle), 0], (64, 1)).astype(np.float32)
            directory.observe(later, np.arange(10, 74))
            self.assertEqual(directory.record_cluster[2], 0)
            return directory

        bounded = drifting_directory()
        original = bounded.record_cluster
        result = bounded.maintain("incremental", 3)
        self.assertEqual(result.visited, 3)
        self.assertGreater(result.changed, 0)
        self.assertEqual(bounded.record_cluster[2], 1)
        np.testing.assert_array_equal(bounded.record_cluster[3:], original[3:])
        self.assert_integrity(bounded)
        full = drifting_directory()
        result = full.maintain("full")
        self.assertGreater(result.changed, 0)
        self.assertEqual(full.record_cluster[2], 1)
        self.assert_integrity(full)

    def test_duplicate_ids_and_invalid_vectors_are_atomic(self) -> None:
        directory = small_directory()
        directory.observe(unit([[1, 0, 0], [1, 0.1, 0]]), np.asarray([1, 2]))
        before = directory.snapshot()
        with self.assertRaises(ValueError):
            directory.observe(unit([[0, 1, 0]]), np.asarray([1]))
        with self.assertRaises(ValueError):
            directory.observe(np.zeros((1, 3), dtype=np.float32), np.asarray([3]))
        with self.assertRaises(ValueError):
            directory.observe(unit([[0, 1, 0], [0, 0, 1]]), np.asarray([3, 3]))
        self.assertEqual(directory.n, 2)
        np.testing.assert_array_equal(directory.record_cluster, before["membership"])
        self.assert_integrity(directory)

    def test_cpu_requires_explicit_test_opt_in_and_empty_queries_fail(self) -> None:
        with self.assertRaises(RuntimeError):
            OnlineDirectory(3, device="cpu")
        directory = small_directory()
        with self.assertRaises(RuntimeError):
            directory.route(np.asarray([1, 0, 0], dtype=np.float32), 1)
        with self.assertRaises(ValueError):
            directory.maintain("incremental")
        with self.assertRaises(ValueError):
            directory.maintain("incremental", -1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
