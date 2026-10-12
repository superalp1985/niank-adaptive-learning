"""Behavioral contracts for factual mixed-scene tracking."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "tracking_runtime"))

import numpy as np
import torch


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "world.py").exists(),
                        "V14 mixed-scene world has not been implemented")

    def test_split_covers_every_factor_pair_without_overlap(self):
        from world import combinations
        train, test = combinations(False), combinations(True)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(train), 18)
        self.assertEqual(len(test), 9)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            expected = {(a, b) for a in range(3) for b in range(3)}
            self.assertEqual({(c[i], c[j]) for c in train}, expected)
            self.assertEqual({(c[i], c[j]) for c in test}, expected)

    def test_contact_requires_reaching_the_target(self):
        from world import World
        env = World(17, combo=(0, 0, 0))
        env.body.position = (0, 0)
        env.body.velocity = (0, 0)
        env.target.position = (2, 0)
        _, _, done, info = env.step(0)
        self.assertFalse(done)
        self.assertFalse(info["success"])
        env.body.position = tuple(env.target.position)
        _, _, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue(info["success"])

    def test_real_obstacle_and_projectile_contact(self):
        from world import World
        env = World(21, combo=(0, 0, 0))
        env.body.position = (0, 0)
        env.body.velocity = (0, 0)
        env.target.position = (2, 2)
        env.add_obstacle((0, 0), 0.3)
        _, _, _, info = env.step(0)
        self.assertTrue(info["collision"])
        env = World(22, combo=(0, 0, 0))
        env.body.position = (0, 0)
        env.target.position = (2, 2)
        env.add_projectile((0, 0), (0, 0))
        _, _, done, info = env.step(0)
        self.assertTrue(info["impact"])
        self.assertTrue(done)
        self.assertFalse(info["success"])

    def test_hidden_motion_program_is_not_a_sensor(self):
        from world import World
        env = World(18, combo=(0, 0, 0))
        before = env.observe().copy()
        env.combo = (0, 2, 2)
        env.target.velocity = (0.9, -0.8)
        np.testing.assert_array_equal(env.observe(), before)

    def test_cold_record_recovery_preserves_ownership(self):
        from memory import Directory
        cache = ROOT / "cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            memory = Directory(Path(folder) / "events.sqlite", 2, "directory")
            obs = np.zeros((2, 32), np.float32)
            obs[0, 0], obs[1, 0] = 0.5, -0.5
            raw = np.zeros((2, 3, 16, 16), np.uint8)
            for step in range(12):
                memory.append(step, obs, np.zeros(2, np.int64), obs, raw)
            ids, evidence = memory.retrieve(obs)
            self.assertTrue(all(memory.owner(int(identifier)) == row
                                for row, identifier in enumerate(ids)))
            wanted = memory.fetch(0, 0)
            np.testing.assert_array_equal(wanted["obs"], obs[0])
            self.assertRaises(ValueError, memory.fetch, 1, 0)
            memory.close()

    def test_unselected_experts_receive_no_gradient(self):
        from learner import Agent
        torch.manual_seed(1)
        agent = Agent(64, 4, device="cpu")
        obs = torch.randn(8, 64)
        selected = torch.ones(8, dtype=torch.long)
        output = agent.expert_logits(obs, selected)
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in agent.experts[1].parameters()))
        for index in (0, 2, 3):
            self.assertTrue(all(p.grad is None for p in agent.experts[index].parameters()))
        self.assertTrue(all(p.grad is None for p in agent.router.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.value.parameters()))

    def test_terminal_gae_does_not_import_reset_value(self):
        from learner import advantages
        reward = torch.tensor([[1.], [2.]])
        done = torch.tensor([[True], [False]])
        value = torch.zeros(2, 1)
        advantage, returns = advantages(reward, done, value, torch.tensor([100.]),
                                        gamma=0.5, lam=1.)
        torch.testing.assert_close(advantage[0], torch.tensor([1.]))
        torch.testing.assert_close(returns[1], torch.tensor([52.]))

    def test_runner_counts_world_steps_and_archives_preaction_observation(self):
        from run import Session
        cache = ROOT / "cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            session = Session(Path(folder), "recent", 71, worlds=2,
                              device="cpu", training=False)
            raw_before = session.envs[0].raster().copy()
            session.advance(2, update=False)
            self.assertEqual(session.step, 2)
            self.assertEqual(session.summary()["new_experiences"], 4)
            self.assertEqual(session.memory.db.execute(
                "SELECT COUNT(*) FROM events").fetchone()[0], 4)
            import zlib
            image = session.memory.db.execute(
                "SELECT image FROM events WHERE id=0").fetchone()[0]
            np.testing.assert_array_equal(
                np.frombuffer(zlib.decompress(image), np.uint8).reshape(3, 16, 16),
                raw_before)
            session.close()

    def test_probe_assigns_every_heldout_combination(self):
        from run import Session
        from world import combinations
        cache = ROOT / "cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            session = Session(Path(folder), "recent", 73, worlds=9,
                              device="cpu", training=False, heldout=True,
                              fixed_combos=combinations(True))
            self.assertEqual({env.combo for env in session.envs}, set(combinations(True)))
            session.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
