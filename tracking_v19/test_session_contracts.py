"""Small CPU sessions verify persisted evidence, cache boundaries, and resets."""

import copy
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tracking_v17.run import dump
from tracking_v19.run import Session

try:
    from tracking_v19.audit import session as audit_session
except ModuleNotFoundError as error:
    if error.name != "tracking_v19.audit":
        raise
    audit_session = None

torch.set_num_threads(2)


def clone_state(agent):
    return {name: {key: tensor.clone() for key, tensor in state.items()}
            for name, state in agent.state().items()}


class SessionContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit_session, "Independent V19 auditor is not implemented")
        cache = REPO / "cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="v19_contract_", dir=cache)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_session(self, name, mode="utility", training=True, feedback=None):
        torch.manual_seed(19031)
        return Session(self.root / name, mode, 681000, worlds=2, training=training,
                       device="cpu", feedback_enabled=feedback, memory_seed=1919,
                       floor=.01)

    def finish(self, trial):
        dump(trial.output / "summary.json", trial.summary())
        trial.close()
        return audit_session(trial.output)

    def test_rollout_cache_and_one_body_reset_pass_independent_reconstruction(self):
        for mode in ("utility", "protected"):
            with self.subTest(mode=mode):
                trial = self.make_session(mode, mode)
                try:
                    trial.envs[0].steps = 1023
                    trial.advance(4, epochs=1)
                    arrays = {name: getattr(trial.memory, name).copy() for name in
                              ("previous", "history", "age", "utility", "deviation",
                               "selected", "full_features", "promoted")}
                    random_state = copy.deepcopy(trial.memory.rng.bit_generator.state)
                    x, prop = trial.features()
                    repeated, repeated_prop = trial.features()
                    self.assertTrue(np.array_equal(x, repeated))
                    self.assertTrue(np.array_equal(prop, repeated_prop))
                    for name, expected in arrays.items():
                        self.assertTrue(np.array_equal(getattr(trial.memory, name), expected),
                                        name)
                    self.assertEqual(random_state, trial.memory.rng.bit_generator.state)
                    x[:] = 90
                    self.assertTrue(np.array_equal(trial.features()[0], repeated))
                    trial.advance(3, epochs=1)
                    with np.load(trial.output / "transitions_00000.npz") as data:
                        self.assertTrue(data["done"][0, 0])
                        self.assertFalse(data["done"][0, 1])
                        self.assertTrue(np.array_equal(data["segment"][:, 0], [0, 1, 1, 1]))
                        self.assertTrue(np.array_equal(data["segment"][:, 1], [0, 0, 0, 0]))
                    result = self.finish(trial)
                    trial = None
                    self.assertEqual(result["rows"], 14)
                    self.assertEqual(result["frame_digests_checked"], 42)
                finally:
                    if trial is not None:
                        trial.close()

    def test_frozen_sessions_never_change_weights_with_or_without_feedback(self):
        trajectories = []
        for mode in ("utility", "protected"):
            for feedback in (False, True):
                with self.subTest(mode=mode, feedback=feedback):
                    trial = self.make_session(f"{mode}_{feedback}", mode, False, feedback)
                    before = clone_state(trial.agent)
                    try:
                        trial.advance(4)
                        trial.advance(3)
                        self.assertEqual(trial.memory.updates, 7 if feedback else 0)
                        self.assertEqual(trial.agent.updates, 0)
                        self.assertEqual(trial.agent.presentations, 0)
                        self.assertEqual(trial.agent.predictor_presentations, 0)
                        for module, tensors in before.items():
                            for name, tensor in tensors.items():
                                self.assertTrue(torch.equal(tensor, trial.agent.state()[module][name]))
                        self.assertTrue(all(parameter.grad is None for module in trial.agent.modules
                                            for parameter in module.parameters()))
                        if feedback:
                            self.assertTrue(trial.memory.utility.any())
                        else:
                            self.assertFalse(trial.memory.utility.any())
                            self.assertFalse(trial.memory.deviation.any())
                            chunks = []
                            for path in sorted(trial.output.glob("transitions_*.npz")):
                                with np.load(path) as data:
                                    chunks.append({key: data[key].copy() for key in
                                                   ("actions", "groups", "reward", "pred_error",
                                                    "selected_bits", "full_features_digest")})
                            trajectories.append(chunks)
                        result = self.finish(trial)
                        trial = None
                        self.assertEqual(result["rows"], 14)
                    finally:
                        if trial is not None:
                            trial.close()
        for first, second in zip(*trajectories):
            for key in first:
                self.assertTrue(np.array_equal(first[key], second[key]), key)

    def test_auditor_rejects_changed_scores_bootstrap_and_selected_bits(self):
        trial = self.make_session("tamper", training=False, feedback=True)
        try:
            trial.advance(4)
            trial.advance(3)
            self.finish(trial)
        finally:
            if trial.db:
                try:
                    trial.close()
                except sqlite3.ProgrammingError:
                    pass
        database = trial.output / "events.sqlite"
        with sqlite3.connect(database) as db:
            scores, bootstrap = db.execute(
                "SELECT scores,bootstrap FROM evidence WHERE body=0 AND time=0").fetchone()
            changed = np.frombuffer(scores, np.float32).copy()
            changed[0] += .1
            db.execute("UPDATE evidence SET scores=? WHERE body=0 AND time=0",
                       (changed.tobytes(),))
        with self.assertRaises(AssertionError):
            audit_session(trial.output)
        with sqlite3.connect(database) as db:
            changed = np.frombuffer(bootstrap, bool).copy()
            changed[0] = ~changed[0]
            db.execute("UPDATE evidence SET scores=?,bootstrap=? WHERE body=0 AND time=0",
                       (scores, changed.tobytes()))
        with self.assertRaises(AssertionError):
            audit_session(trial.output)
        with sqlite3.connect(database) as db:
            db.execute("UPDATE evidence SET bootstrap=? WHERE body=0 AND time=0", (bootstrap,))
        path = trial.output / "transitions_00000.npz"
        with np.load(path) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["selected_bits"][0, 0, 0] ^= np.uint8(1)
        np.savez_compressed(path, **arrays)
        with self.assertRaises(AssertionError):
            audit_session(trial.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
