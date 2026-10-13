"""Physical-direction evidence must stay in its observation's coordinates."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking_v17.world import EYES
from tracking_v18.learner import EyeMemory


def prop(shifts):
    result = np.zeros((len(shifts), 6), np.float32)
    result[:, 2] = np.asarray(shifts) * (2 / EYES)
    return result


def observe(memory, shifts, reset=None):
    worlds = len(shifts)
    return memory.features(np.zeros((worlds, EYES, 3), np.uint8), prop(shifts),
                           np.zeros(worlds, bool) if reset is None else reset)


class FrameContracts(unittest.TestCase):
    def test_next_frame_new_evidence_keeps_its_measured_direction(self):
        for shift, eye in ((1, 10), (-3, 41), (4, 0), (-4, EYES - 1)):
            with self.subTest(shift=shift, eye=eye):
                memory = EyeMemory(1, "aligned")
                gain = np.zeros((1, EYES), np.float32)
                gain[0, eye] = .04
                memory.feedback(gain, prop([shift]))
                observe(memory, [shift])
                self.assertEqual(int(memory.utility[0].argmax()), eye)
                self.assertAlmostEqual(float(memory.utility[0, eye]), .004, places=8)

    def test_old_evidence_rotates_but_new_evidence_does_not(self):
        memory = EyeMemory(1, "aligned")
        memory.utility[0, 30] = .02
        gain = np.zeros((1, EYES), np.float32)
        gain[0, 70] = -.04
        memory.feedback(gain, prop([3]))
        observe(memory, [3])
        self.assertAlmostEqual(float(memory.utility[0, 27]), .018, places=8)
        self.assertAlmostEqual(float(memory.utility[0, 70]), -.004, places=8)
        self.assertEqual(int(np.count_nonzero(memory.utility)), 2)

    def test_each_body_owns_its_rotation(self):
        memory = EyeMemory(3, "aligned")
        gain = np.zeros((3, EYES), np.float32)
        gain[np.arange(3), [2, 40, 127]] = [.02, -.03, .04]
        memory.feedback(gain, prop([5, -2, 0]))
        observe(memory, [5, -2, 0])
        self.assertTrue(np.array_equal(np.abs(memory.utility).argmax(1), [2, 40, 127]))

    def test_successive_rotations_preserve_old_and_new_physical_evidence(self):
        memory = EyeMemory(1, "aligned")
        gain = np.zeros((1, EYES), np.float32)
        gain[0, 0] = .04
        memory.feedback(gain, prop([2]))
        observe(memory, [2])
        gain[:] = 0
        gain[0, 50] = .02
        memory.feedback(gain, prop([-3]))
        observe(memory, [-3])
        self.assertAlmostEqual(float(memory.utility[0, 3]), .0036, places=8)
        self.assertAlmostEqual(float(memory.utility[0, 50]), .002, places=8)

    def test_reset_discards_old_segment_feedback_only_for_that_body(self):
        memory = EyeMemory(2, "aligned")
        gain = np.full((2, EYES), .04, np.float32)
        memory.feedback(gain, prop([1, -1]))
        observe(memory, [1, -1], np.array([True, False]))
        self.assertEqual(float(memory.utility[0].sum()), 0)
        self.assertTrue((memory.utility[1] > 0).all())

    def test_legacy_arm_preserves_documented_extra_rotation(self):
        memory = EyeMemory(1, "legacy")
        gain = np.zeros((1, EYES), np.float32)
        gain[0, 10] = .04
        memory.feedback(gain, prop([1]))
        observe(memory, [1])
        self.assertEqual(int(memory.utility[0].argmax()), 9)

    def test_zero_rotation_arms_are_identical(self):
        legacy, aligned = EyeMemory(2, "legacy"), EyeMemory(2, "aligned")
        gain = np.random.default_rng(18).normal(0, .1, (2, EYES)).astype(np.float32)
        for memory in (legacy, aligned):
            memory.feedback(gain, prop([0, 0]))
            observe(memory, [0, 0])
        self.assertTrue(np.array_equal(legacy.utility, aligned.utility))


class SessionContracts(unittest.TestCase):
    def test_legacy_session_matches_v17_actions_gain_and_update(self):
        import torch
        from tracking_v17.run import Session as PreviousSession
        from tracking_v18.run import Session
        states, data = [], []
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as temporary:
            for cls, mode in ((PreviousSession, "revisable"), (Session, "legacy")):
                torch.manual_seed(181)
                trial = cls(Path(temporary) / mode, mode, 681000, worlds=2, device="cpu")
                try:
                    trial.advance(4, epochs=1)
                    states.append(trial.agent.state())
                    with np.load(trial.output / "transitions_00000.npz") as values:
                        data.append({k: values[k].copy() for k in
                                     ("actions", "groups", "pred_error", "retention_gate",
                                      "retention_utility", "locked", "reward")})
                finally:
                    trial.close()
        for key in data[0]:
            self.assertTrue(np.array_equal(data[0][key], data[1][key]), key)
        for module, tensors in states[0].items():
            for key, tensor in tensors.items():
                self.assertTrue(torch.equal(tensor, states[1][module][key]), (module, key))

    def test_cached_features_do_not_rotate_utility_twice(self):
        from tracking_v18.run import Session
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as temporary:
            trial = Session(Path(temporary) / "aligned", "aligned", 681000,
                            worlds=2, device="cpu")
            try:
                trial.advance(3, epochs=1)
                before = trial.memory.utility.copy()
                x, _ = trial.features()
                repeat, _ = trial.features()
                self.assertTrue(np.array_equal(before, trial.memory.utility))
                self.assertTrue(np.array_equal(x, repeat))
            finally:
                trial.close()

    def test_frozen_weights_can_receive_factual_feedback_without_gradient_updates(self):
        import torch
        from tracking_v18.run import Session
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as temporary:
            trial = Session(Path(temporary) / "aligned", "aligned", 681000,
                            worlds=2, training=False, device="cpu", feedback_enabled=True)
            before = {name: {k: v.clone() for k, v in state.items()}
                      for name, state in trial.agent.state().items()}
            try:
                trial.advance(4)
                self.assertEqual(trial.memory.updates, 4)
                self.assertEqual(trial.agent.updates, 0)
                for module, state in before.items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(value, trial.agent.state()[module][key]))
            finally:
                trial.close()

    def test_raw_frame_audit_covers_rollout_cache_and_segment_reset(self):
        from tracking_v17.run import dump
        from tracking_v18.audit import session as audit_session
        from tracking_v18.run import Session
        with tempfile.TemporaryDirectory(dir=ROOT / "tracking_v18") as temporary:
            for mode in ("legacy", "aligned"):
                trial = Session(Path(temporary) / mode, mode, 681000, worlds=2, device="cpu")
                try:
                    trial.envs[0].steps = 1023
                    trial.advance(4, epochs=1)
                    trial.advance(3, epochs=1)
                    dump(trial.output / "summary.json", trial.summary())
                finally:
                    trial.close()
                result = audit_session(trial.output)
                self.assertEqual(result["rows"], 14)
                self.assertEqual(result["frame_digests_checked"], 21)


if __name__ == "__main__":
    unittest.main(verbosity=2)
