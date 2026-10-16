import math
import pickle
import unittest

import numpy as np
import torch

from tracking_v22.neural import retina
from tracking_v22.world import World, canonical_checkpoint
from tracking_v23.vision import (
    HeadWorld, HEAD_LIMIT, HEAD_SPEED, compose_input, encode_images, migrate_input,
)

torch.set_num_threads(4)


class VisionChecks(unittest.TestCase):
    def scene(self):
        world = HeadWorld(123, combo=(0, 0, 0))
        world.body.position = (0, 0)
        world.body.velocity = (0, 0)
        world.target.position = (2, 0)
        world.heading = 0.
        world.invalidate_observation()
        return world

    def test_head_motion_is_independent_of_body_motion(self):
        head, body = self.scene(), self.scene()
        for _ in range(4):
            a, _, _, ai = head.step(12)
            b, _, _, bi = body.step(3)
            self.assertEqual(tuple(head.body.position), tuple(body.body.position))
            self.assertEqual(head.heading, body.heading)
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(ai["body_action"], 3)
        self.assertAlmostEqual(head.head_yaw, HEAD_SPEED*.04)
        self.assertEqual(body.head_yaw, 0.)
        self.assertNotEqual(head.head_yaw, head.heading)

    def test_bounded_finite_head_motion_and_reset(self):
        world = self.scene()
        world.head_yaw = HEAD_LIMIT-.001
        world.step(9)
        self.assertEqual(world.head_yaw, HEAD_LIMIT)
        self.assertLessEqual(abs(world.head_velocity), HEAD_SPEED)
        world.step(9)
        self.assertEqual(world.head_velocity, 0.)
        world.step(18)
        self.assertLess(world.head_yaw, HEAD_LIMIT)
        self.assertAlmostEqual(world.timestamp, .03)
        world.reset()
        np.testing.assert_array_equal(world.head_proprio(), np.zeros(4))
        for action in (-1, 27, 1.5):
            with self.assertRaises(ValueError):
                world.step(action)

    def test_panorama_preserves_every_pixel(self):
        world = self.scene()
        original = world.observe()[0]
        for yaw in (-1.3, -.45, .8, HEAD_LIMIT):
            world.head_yaw = yaw
            observed = world.observe()[0]
            self.assertEqual(observed.shape, (512, 1024, 3))
            np.testing.assert_array_equal(np.roll(observed, world.head_pixel, axis=1), original)

    def test_body_features_cues_and_outcomes_invariant_to_head_yaw(self):
        world = self.scene()
        initial = world.observe()
        expected_features, expected_cues = retina(torch.from_numpy(initial[0][None]))
        old_detail = None
        differences = []
        for yaw in (0., -.6, .9, HEAD_LIMIT):
            world.head_yaw = yaw
            rgb, proprio, head = world.observe()
            features, cues, detail = encode_images(torch.from_numpy(rgb[None]),
                                                   torch.from_numpy(head[None]))
            self.assertTrue(torch.equal(features, expected_features))
            for key in cues:
                self.assertTrue(torch.equal(cues[key], expected_cues[key]), key)
            if old_detail is not None:
                differences.append(float((detail-old_detail).abs().sum()))
            old_detail = detail
        self.assertTrue(any(value > 1 for value in differences))

    def test_canonical_restored_world_retains_head_and_callbacks(self):
        world = self.scene()
        world.step(10)
        worlds = [world]
        restored = pickle.loads(canonical_checkpoint(worlds))[0]
        for action in (19, 9, 14):
            a, ar, ad, ai = worlds[0].step(action)
            b, br, bd, bi = restored.step(action)
            for left, right in zip(a, b):
                np.testing.assert_array_equal(left, right)
            self.assertEqual((ar, ad, ai), (br, bd, bi))
        restored.target.position = restored.body.position
        self.assertTrue(restored.step(0)[3]["contact"])

    def test_legacy_upgrade_keeps_world_rng_and_physics(self):
        world = World(54, combo=(0, 0, 0))
        before = world.observe()
        rng = pickle.dumps(world.rng.bit_generator.state)
        HeadWorld.upgrade(world)
        after = world.observe()
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(pickle.dumps(world.rng.bit_generator.state), rng)

    def test_insertion_preserves_spikes_and_missing_history(self):
        old = torch.arange(576.).repeat(3, 1)
        new = migrate_input(old)
        self.assertEqual(new.shape, (3, 964))
        self.assertTrue(torch.equal(new[:, :512], old[:, :512]))
        self.assertTrue(torch.equal(new[:, 900:], old[:, 512:]))
        self.assertTrue(torch.equal(new[:, -32:], old[:, -32:]))
        self.assertEqual(int(torch.count_nonzero(new[:, 512:900])), 0)
        composed = compose_input(old, torch.ones(3, 4), torch.full((3, 384), 2.))
        self.assertTrue(torch.equal(composed[:, -32:], old[:, -32:]))


if __name__ == "__main__":
    unittest.main()
