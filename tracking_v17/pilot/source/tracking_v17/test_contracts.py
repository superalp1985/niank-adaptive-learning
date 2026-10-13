"""Boundaries that can invalidate a compound-eye learning experiment."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from tracking_v17.world import World, EYES, SEGMENT, combinations


def clear_view(seed=1):
    env = World(seed, combo=(0, 0, 0))
    env.body.position, env.body.velocity = (0., 0.), (0., 0.)
    env.heading = 0.
    for body, _ in env.distractors:
        body.position = (3., 3.)
    env._cache = None
    return env


class VisionContracts(unittest.TestCase):
    def test_rear_and_circular_seam_are_visible(self):
        env = clear_view()
        env.target.position = (-2., 0.)
        rgb, _ = env.observe()
        self.assertTrue(env.audit_visibility()[EYES // 2])
        self.assertGreater(int(rgb[EYES // 2, 1]), 150)
        env.target.position, env._cache = (2., 0.), None
        env.observe()
        mask = env.audit_visibility()
        self.assertTrue(mask[0] and mask[-1], "360-degree seam has a blind gap")

    def test_nearest_solid_surface_occludes_marked_target(self):
        env = clear_view()
        env.target.position = (2., 0.)
        env.obstacles.append(env.circle((1., 0.), .45, 2))
        rgb, prop = env.observe()
        self.assertFalse(env.audit_visibility().any())
        self.assertEqual(rgb.shape, (128, 3))
        self.assertEqual(prop.shape, (6,))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_contact_and_impact_do_not_end_practice(self):
        env = clear_view()
        env.body.position = tuple(env.target.position)
        _, _, done, info = env.step(0)
        self.assertTrue(info["contact"])
        self.assertFalse(done)
        bullet, shape = env.circle(env.body.position, .12, 4, sensor=True)
        shape.filter = type(env.body_shape.filter)(categories=8, mask=1)
        env.projectiles.append((bullet, shape))
        _, _, done, info = env.step(0)
        self.assertTrue(info["impact"])
        self.assertFalse(done)
        self.assertLess(env.steps, SEGMENT)

    def test_all_training_combinations_have_parallel_slots(self):
        seen, novel = combinations(), combinations(True)
        self.assertEqual((len(seen), len(novel)), (72, 24))
        self.assertFalse(set(seen) & set(novel))
        for axis, count in enumerate((6, 4, 4)):
            self.assertEqual({c[axis] for c in seen}, set(range(count)))
            self.assertEqual({c[axis] for c in novel}, set(range(count)))

    def test_attack_intervals_are_not_fixed_periods(self):
        env = World(7, combo=(0, 0, 1))
        launches = []
        for step in range(200):
            _, _, _, info = env.step(0)
            if info["spawned"]:
                launches.append(step)
        self.assertGreater(len(launches), 3)
        self.assertGreater(len(set(np.diff(launches))), 1)


class LearningContracts(unittest.TestCase):
    def test_raw_sqlite_and_transition_telemetry_cover_same_physical_steps(self):
        from tracking_v17.run import Session
        with tempfile.TemporaryDirectory(dir=ROOT.parent / "cache") as temporary:
            session = Session(Path(temporary) / "trial", "revisable", 171091,
                              worlds=2, device="cpu")
            point = session.advance(4, epochs=1)
            self.assertEqual(point["experiences"], 8)
            self.assertEqual(session.db.execute("SELECT count(*) FROM events").fetchone()[0], 8)
            self.assertEqual(session.db.execute("SELECT length(target_mask) FROM events").fetchone()[0], 16)
            with np.load(session.output / "transitions_00000.npz") as d:
                self.assertEqual(d["actions"].shape, (4, 2))
                self.assertEqual(d["body_position"].shape, (4, 2, 2))
                self.assertEqual(int(d["start_step"]), 0)
            session.close()

    def test_direction_is_not_erased_by_circular_pooling(self):
        from tracking_v17.learner import EyeNetwork
        torch.manual_seed(17)
        net = EyeNetwork(9)
        x = torch.zeros(1, 11, EYES)
        x[0, 1, :3] = 1
        forward = net(x, torch.zeros(1, 6))
        behind = net(torch.roll(x, EYES // 2, 2), torch.zeros(1, 6))
        self.assertFalse(torch.allclose(forward, behind, atol=1e-6))

    def test_rotation_aligns_own_history_and_reset_clears_it(self):
        from tracking_v17.learner import EyeMemory
        memory = EyeMemory(2, "fixed")
        rgb = np.zeros((2, EYES, 3), np.uint8)
        rgb[0, 0] = [20, 230, 30]
        prop = np.zeros((2, 6), np.float32)
        memory.features(rgb, prop, np.ones(2, bool))
        prop[0, 2] = 2 / EYES
        x = memory.features(np.zeros_like(rgb), prop, np.zeros(2, bool))
        self.assertGreater(x[0, 6:9, -1].sum(), 0.)
        self.assertEqual(float(x[1, 6:9].sum()), 0.)
        x = memory.features(np.zeros_like(rgb), prop, np.ones(2, bool))
        self.assertEqual(float(x[:, 6:9].sum()), 0.)

    def test_prediction_updates_do_not_backpropagate_into_policy(self):
        from tracking_v17.learner import Agent
        agent = Agent("fixed", device="cpu")
        x = torch.rand(2, 11, EYES, requires_grad=True)
        prop = torch.zeros(2, 6)
        action = torch.tensor([1, 2])
        normal, omitted = agent.predict(x, prop, action)
        (normal.square().mean() + omitted.square().mean()).backward()
        self.assertIsNone(x.grad)
        self.assertTrue(any(p.grad is not None for p in agent.predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.experts.parameters()))

    def test_memory_feedback_uses_observed_local_prediction_gain(self):
        from tracking_v17.learner import EyeMemory
        memory = EyeMemory(1, "revisable")
        memory.feedback(np.full((1, EYES), .04, np.float32))
        self.assertTrue((memory.utility > 0).all())
        for _ in range(12):
            memory.feedback(np.full((1, EYES), -.04, np.float32))
        self.assertTrue((memory.utility < 0).all())
        fixed = EyeMemory(1, "fixed")
        fixed.feedback(np.full((1, EYES), .04, np.float32))
        self.assertEqual(float(fixed.utility.sum()), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
