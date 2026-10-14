"""Contracts for body-local, factual and reversible V20 event retrieval."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracking_v20.directory import EventDirectory


def feature(value=.4):
    x = np.full((11, 128), value, np.float32)
    x[3:6] = .2
    x[6:9] = .3
    return x


def insert(directory, body, time, value=.4, targets=None, key_features=None):
    x = feature(value)
    prop = np.array([body, 0, 0, 0, 0, 0], np.float32)
    target = np.arange(5, dtype=np.float32) if targets is None else targets
    result = directory.add(body, time, time // 16, x, prop, time % 9, body % 4,
                           target, key_features=key_features)
    return result, x, prop


def query(directory, value=.4):
    return np.stack([directory.key(feature(value),
                                  np.array([body, 0, 0, 0, 0, 0], np.float32))
                     for body in range(directory.worlds)])


class KeyContracts(unittest.TestCase):
    def test_key_is_150_dimensions_float32_and_batch_matches_single(self):
        x = np.stack([feature(), feature(.7)])
        prop = np.zeros((2, 6), np.float32)
        keys = EventDirectory.key(x, prop)
        self.assertEqual(keys.shape, (2, 150))
        self.assertEqual(keys.dtype, np.float32)
        self.assertTrue(np.array_equal(keys[0], EventDirectory.key(x[0], prop[0])))

    def test_rotated_visual_has_a_different_key(self):
        x = feature()
        x[:9, 0:8] += .2
        self.assertFalse(np.array_equal(
            EventDirectory.key(x, np.zeros(6)),
            EventDirectory.key(np.roll(x, 8, axis=1), np.zeros(6))))

    def test_key_ignores_non_visual_gate_channels(self):
        x = feature()
        original = EventDirectory.key(x, np.zeros(6))
        x[9:] = 901
        self.assertTrue(np.array_equal(original, EventDirectory.key(x, np.zeros(6))))

    def test_proprio_delta_and_history_each_change_the_key(self):
        x = feature()
        prop = np.zeros(6, np.float32)
        original = EventDirectory.key(x, prop)
        changed_prop = prop.copy()
        changed_prop[2] = .4
        self.assertFalse(np.array_equal(original, EventDirectory.key(x, changed_prop)))
        for channel in (0, 3, 6):
            changed = x.copy()
            changed[channel] += .1
            self.assertFalse(np.array_equal(original, EventDirectory.key(changed, prop)))


class IngestionContracts(unittest.TestCase):
    def test_fifo_capacity_and_stable_ids_survive_slot_reuse(self):
        directory = EventDirectory(2, "retrieval", 1, capacity=3)
        for time in range(3):
            insert(directory, 0, time)
        insert(directory, 1, 0)
        result, _, _ = insert(directory, 0, 3)
        self.assertEqual(result["event_id"], (0, 3))
        self.assertEqual(result["evicted_id"], (0, 0))
        self.assertTrue(np.array_equal(directory.event_ids(), [[0, 1], [0, 2],
                                                              [0, 3], [1, 0]]))
        self.assertTrue(np.array_equal(directory.counts, [3, 1]))
        self.assertEqual(len(directory), 4)
        self.assertEqual(directory.evictions.tolist(), [1, 0])
        with self.assertRaises(ValueError):
            insert(directory, 0, 0)

    def test_features_are_float16_labels_and_keys_float32_and_copy_inputs(self):
        directory = EventDirectory(1, "random", 2)
        _, x, prop = insert(directory, 0, 0)
        self.assertEqual(directory._features.dtype, np.float16)
        self.assertEqual(directory._proprio.dtype, np.float32)
        self.assertEqual(directory._keys.dtype, np.float32)
        self.assertEqual(directory._targets.dtype, np.float32)
        x[:] = 7
        prop[:] = 9
        batch, _ = directory.sample(query(directory), [0], 8, budget=1)
        self.assertTrue(np.allclose(batch["features"][0], feature(), atol=.0002))
        self.assertTrue(np.array_equal(batch["proprio"][0], np.zeros(6)))
        self.assertEqual(batch["features"].dtype, np.float32)

    def test_key_features_are_full_while_replay_keeps_actual_masked_input(self):
        directory = EventDirectory(1, "retrieval", 3)
        full = feature(.8)
        masked = full.copy()
        masked[6:9] = 0
        directory.add(0, 0, 0, masked, np.zeros(6), 1, 2, np.zeros(5),
                      key_features=full)
        batch, logs = directory.sample(directory.key(full, np.zeros(6)), 1, 8, 4)
        self.assertEqual(logs[0]["distance"], 0.)
        self.assertTrue(np.allclose(batch["features"][0], masked, atol=.0002))
        self.assertTrue(np.allclose(directory._keys[0, 0],
                                    directory.key(full, np.zeros(6))))

    def test_targets_cannot_change_ranking_or_access_weight(self):
        directories = [EventDirectory(1, "retrieval", 4) for _ in range(2)]
        for index, directory in enumerate(directories):
            for time in range(12):
                insert(directory, 0, time, value=time / 12,
                       targets=np.full(5, (1 if index else -1) * time, np.float32))
        results = [directory.sample(query(directory, .45), 1, 20, 8)
                   for directory in directories]
        self.assertTrue(np.array_equal(results[0][0]["record_ids"],
                                       results[1][0]["record_ids"]))
        self.assertEqual(results[0][1], results[1][1])
        self.assertFalse(np.array_equal(results[0][0]["targets"],
                                        results[1][0]["targets"]))

    def test_duplicate_or_backward_time_does_not_mutate_resident_record(self):
        directory = EventDirectory(1, "random", 5)
        insert(directory, 0, 4)
        for time in (4, 3):
            with self.assertRaises(ValueError):
                insert(directory, 0, time)
        self.assertTrue(np.array_equal(directory.event_ids(), [[0, 4]]))


class SamplingContracts(unittest.TestCase):
    def fixture(self, worlds=3, mode="retrieval", count=16, capacity=512):
        directory = EventDirectory(worlds, mode, 91, capacity=capacity)
        for body in range(worlds):
            for time in range(count):
                insert(directory, body, time, value=.1 + time / (count + 2))
        return directory

    def test_default_budget_exact_body_local_and_repeats_explicit(self):
        directory = self.fixture(count=8)
        batch, logs = directory.sample(query(directory), [1, 0, .3], 16)
        self.assertEqual(len(logs), 1536)
        self.assertEqual(len(batch["features"]), 1536)
        self.assertEqual(np.bincount([row["body"] for row in logs]).tolist(),
                         [512, 512, 512])
        self.assertTrue(any(row["repeated"] for row in logs))
        for log, prop, record_id, group in zip(logs, batch["proprio"],
                                             batch["record_ids"], batch["groups"]):
            self.assertEqual(int(prop[0]), log["body"])
            self.assertEqual(int(group), log["body"] % 4)
            self.assertEqual(tuple(record_id), log["event_id"])
            self.assertEqual(log["retrieval_id"],
                             f'{log["body"]}:{log["event_time"]}')

    def test_enough_records_are_disjoint_with_exact_random_reconsideration(self):
        directory = self.fixture(worlds=2, count=40)
        _, logs = directory.sample(query(directory), [1, 1], 50, budget=32)
        for body in range(2):
            rows = [row for row in logs if row["body"] == body]
            self.assertEqual(len(rows), 16)
            self.assertEqual(len({row["event_id"] for row in rows}), 16)
            self.assertEqual(sum(row["random"] for row in rows), 4)
            self.assertFalse(any(row["repeated"] for row in rows))

    def test_future_and_current_records_never_enter_batch(self):
        directory = self.fixture(worlds=2, count=12)
        _, logs = directory.sample(query(directory), 1, [5, 7], budget=32)
        self.assertEqual(len(logs), 32)
        for row in logs:
            self.assertLess(row["event_time"], row["query_time"])
        with self.assertRaises(ValueError):
            directory.sample(query(directory), 1, [4, 7], budget=1)

    def test_failed_queries_retrieve_nearest_nonfailed_queries_are_uniform(self):
        directory = self.fixture(worlds=2, count=20)
        keys = query(directory, .1)
        _, logs = directory.sample(keys, [1, 0], 25, budget=8)
        body_zero = [row for row in logs if row["body"] == 0]
        self.assertEqual([row["event_time"] for row in body_zero[:3]], [0, 1, 2])
        self.assertFalse(any(row["random"] for row in body_zero[:3]))
        body_one = [row for row in logs if row["body"] == 1]
        self.assertTrue(all(row["random"] and not row["failed_query"]
                            for row in body_one))

    def test_random_and_retrieval_have_identical_query_opportunity_counts(self):
        directories = [self.fixture(mode=mode, count=12)
                       for mode in ("retrieval", "random")]
        opportunity = np.zeros((2, 3), np.int64)
        for query_time in range(20, 26):
            for index, directory in enumerate(directories):
                _, logs = directory.sample(query(directory), [1, 0, 1],
                                           query_time, budget=10)
                opportunity[index] += np.bincount([row["body"] for row in logs],
                                                 minlength=3)
                if directory.mode == "random":
                    self.assertTrue(all(row["random"] for row in logs))
        self.assertTrue(np.array_equal(opportunity[0], opportunity[1]))
        self.assertEqual(opportunity[0].tolist(), [20, 20, 20])

    def test_random_mode_does_not_depend_on_query_or_failure_strength(self):
        first = self.fixture(worlds=2, mode="random")
        second = self.fixture(worlds=2, mode="random")
        a, _ = first.sample(query(first, .2), [0, 0], 24, budget=20)
        b, _ = second.sample(query(second, .9), [100, 2], 24, budget=20)
        self.assertTrue(np.array_equal(a["record_ids"], b["record_ids"]))

    def test_weights_decay_access_promotes_and_random_reopens_dormant_record(self):
        directory = self.fixture(worlds=1, count=8)
        directory._weight[0, :8] = 0
        _, logs = directory.sample(query(directory), 1, 512, budget=8)
        self.assertEqual(len({row["event_id"] for row in logs}), 8)
        self.assertEqual(sum(row["random"] for row in logs), 2)
        for row in logs:
            self.assertAlmostEqual(row["weight_before"], directory.MIN_WEIGHT)
            self.assertGreater(row["weight_after"], row["weight_before"])
            self.assertEqual(row["last_access_after"], 512)
            self.assertEqual(row["access_count"], 1)
        _, next_logs = directory.sample(query(directory), 0, 1024, budget=8)
        for row in next_logs:
            self.assertLess(row["weight_before"], logs[0]["weight_after"])
            self.assertEqual(row["last_access_before"], 512)
            self.assertEqual(row["access_count"], 2)

    def test_empty_body_is_not_filled_with_another_bodys_events(self):
        directory = EventDirectory(3, "retrieval", 101)
        insert(directory, 1, 2)
        batch, logs = directory.sample(query(directory), [1, 1, 1], 10, budget=7)
        self.assertEqual(len(batch["features"]), 7)
        self.assertTrue(all(row["body"] == 1 for row in logs))
        self.assertEqual(directory.counts.tolist(), [0, 1, 0])

    def test_empty_or_no_past_records_return_typed_empty_batch(self):
        directory = EventDirectory(1, "random", 102)
        for budget in (0, 12):
            batch, logs = directory.sample(query(directory), 1, 0, budget)
            self.assertEqual(len(batch["features"]), 0)
            self.assertEqual(batch["features"].dtype, np.float32)
            self.assertEqual(batch["targets"].shape, (0, 5))
            self.assertEqual(logs, [])
        insert(directory, 0, 0)
        batch, logs = directory.sample(query(directory), 1, 0, 12)
        self.assertEqual(len(batch["features"]), 0)
        self.assertEqual(logs, [])

    def test_capacity_bytes_remain_bounded_after_many_insertions(self):
        directory = self.fixture(worlds=1, count=4, capacity=4)
        allocated = directory.stored_bytes
        for time in range(4, 40):
            insert(directory, 0, time)
        self.assertEqual(directory.stored_bytes, allocated)
        self.assertEqual(len(directory), 4)
        self.assertTrue(np.array_equal(directory.event_ids(0),
                                       [[0, 36], [0, 37], [0, 38], [0, 39]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
