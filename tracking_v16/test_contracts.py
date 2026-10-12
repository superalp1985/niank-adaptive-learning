"""Executed-episode reward contracts."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "tracking_v14"))
sys.path.insert(0, str(ROOT.parent / "tracking_runtime"))


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "run.py").exists(), "V16 safe contact is not implemented")

    def test_whole_episode_collision_changes_contact_feedback(self):
        from tracking_v16.run import World
        env = World(91, combo=(0, 0, 0), safe=True)
        env.body.position, env.target.position = (0, 0), (2, 2)
        env.add_obstacle((0, 0), 0.3)
        env.step(0)
        self.assertEqual(float(env.observe()[30]), 1.)
        env.body.position = tuple(env.target.position)
        _, reward, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertFalse(info["safe_success"])
        self.assertAlmostEqual(reward - info["base_reward"], -8.)

    def test_clean_contact_receives_actual_safe_feedback(self):
        from tracking_v16.run import World
        env = World(93, combo=(0, 0, 0), safe=True)
        env.body.position = tuple(env.target.position)
        _, reward, _, info = env.step(0)
        self.assertTrue(info["safe_success"])
        self.assertAlmostEqual(reward - info["base_reward"], 16.)
        env.reset()
        self.assertEqual(float(env.observe()[30]), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
