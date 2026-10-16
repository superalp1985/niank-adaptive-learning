"""Physical sensor contract checks; run with unittest."""
import pickle
import unittest

import numpy as np

from tracking_v22.world import HEIGHT, WIDTH, World, canonical_checkpoint


class WorldTests(unittest.TestCase):
    def scene(self):
        world = World(41, combo=(0, 0, 0))
        world.body.position = (0, 0)
        world.body.velocity = (0, 0)
        world.heading = 0.
        for body, _ in world.distractors:
            body.position = (-3, 3)
        world.target.position = (2, 0)
        world.distance = 2.
        world.invalidate_observation()
        return world

    def test_native_2d_shape_and_cross(self):
        world = self.scene()
        rgb, proprio = world.observe()
        self.assertEqual(rgb.shape, (512, 1024, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(proprio.shape, (8,))
        audit = world.audit_visibility()
        self.assertGreater(audit['target_pixels'], 500)
        self.assertGreater(audit['marker_pixels'], 30)
        mask = world._marker_mask
        self.assertGreater(np.count_nonzero(mask.any(axis=0)), 8)
        self.assertGreater(np.count_nonzero(mask.any(axis=1)), 8)
        self.assertFalse(np.array_equal(rgb[HEIGHT // 2], rgb[HEIGHT // 2 + 16]))

    def test_full_circle_and_seam(self):
        world = self.scene()
        world.observe()
        mask = world._target_mask
        self.assertTrue(mask[:, :30].any())
        self.assertTrue(mask[:, -30:].any())
        for angle in (0., np.pi / 2, np.pi, 3 * np.pi / 2):
            world.target.position = (2 * np.cos(angle), 2 * np.sin(angle))
            world.invalidate_observation()
            world.observe()
            column = int(angle / (2 * np.pi) * WIDTH) % WIDTH
            self.assertTrue(world._target_mask[:, column].any())

    def test_nearest_depth_occlusion(self):
        world = self.scene()
        world.obstacles.append(world.circle((1., 0.), .4, 2))
        world.invalidate_observation()
        self.assertEqual(world.audit_visibility()['target_pixels'], 0)

    def test_physical_clocks_and_dt_invariance(self):
        a, b = self.scene(), self.scene()
        b.physics_dt = .0005
        before = a.observe()
        initial_frames = a.frames
        for _ in range(30):
            a.step(2)
            b.step(2)
        self.assertAlmostEqual(a.timestamp, .30, places=9)
        self.assertEqual(a.frames, initial_frames + 30)
        self.assertLess((a.body.position - b.body.position).length, .003)
        np.testing.assert_array_equal(before[1], [0, 0, 0, 0, 0, 1, 1, 0])
        timestamp = a.timestamp
        rng = pickle.dumps(a.rng.bit_generator.state)
        a.observe()
        a.audit_visibility()
        self.assertEqual(a.timestamp, timestamp)
        self.assertEqual(pickle.dumps(a.rng.bit_generator.state), rng)
        self.assertTrue((a.observe()[1][5:7] <= 1).all())
        self.assertLess(a.energy, 1.)

    def test_impact_health_and_contact_energy(self):
        world = self.scene()
        projectile = world.circle((0., 0.), .12, 4, sensor=True)
        projectile[1].filter = world.body_shape.filter.__class__(categories=8, mask=1)
        world.projectiles.append(projectile)
        world.energy = .5
        _, _, _, info = world.step(0)
        self.assertTrue(info['impact'])
        self.assertLess(world.integrity, 1.)
        self.assertEqual(world.observe()[1][4], 1.)
        world.target.position = world.body.position
        world.invalidate_observation()
        energy = world.energy
        _, _, _, info = world.step(0)
        self.assertTrue(info['contact'])
        self.assertGreater(world.energy, energy)
        self.assertEqual(world.observe()[1][7], 1.)

    def test_canonical_checkpoint_continuation(self):
        world = self.scene()
        for action in (1, 2, 0, 7):
            world.step(action)
        worlds = [world]
        snapshot = canonical_checkpoint(worlds)
        world = worlds[0]
        restored = pickle.loads(snapshot)[0]
        np.testing.assert_array_equal(world.observe()[0], restored.observe()[0])
        self.assertEqual(world.audit_visibility(), restored.audit_visibility())
        for action in (3, 0, 4, 1):
            a, ra, da, ia = world.step(action)
            b, rb, db, ib = restored.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual((ra, da, ia), (rb, db, ib))

    def test_checkpoint_overlapping_contact_and_moving_obstacle(self):
        # Deep overlap exercises the contact bias omitted by raw Pymunk pickle.
        for obstacle_velocity in ((0., 0.), (.15, .25)):
            with self.subTest(obstacle_velocity=obstacle_velocity):
                world = self.scene()
                obstacle = world.circle((.25, 0.), .32, 2)
                obstacle[0].velocity = obstacle_velocity
                world.obstacles.append(obstacle)
                world.step(1)
                self.assertTrue(world.collision)
                worlds = [world]
                list_identity = id(worlds)
                before_rgb, before_proprio = world.observe()
                before_rng = pickle.dumps(world.rng.bit_generator.state)
                before_time = world.timestamp
                before_position = tuple(world.body.position)
                snapshot = canonical_checkpoint(worlds)
                self.assertEqual(id(worlds), list_identity)
                fresh = worlds[0]
                resumed = pickle.loads(snapshot)[0]
                self.assertIsNot(fresh, world)
                self.assertEqual(fresh.timestamp, before_time)
                self.assertEqual(tuple(fresh.body.position), before_position)
                self.assertEqual(pickle.dumps(fresh.rng.bit_generator.state), before_rng)
                np.testing.assert_array_equal(fresh.observe()[0], before_rgb)
                np.testing.assert_array_equal(fresh.observe()[1], before_proprio)
                for _ in range(10):
                    a, ra, da, ia = fresh.step(1)
                    b, rb, db, ib = resumed.step(1)
                    self.assertTrue(ia['collision'])
                    np.testing.assert_array_equal(a[0], b[0])
                    np.testing.assert_array_equal(a[1], b[1])
                    self.assertEqual(tuple(fresh.body.position), tuple(resumed.body.position))
                    self.assertEqual(tuple(fresh.body.velocity), tuple(resumed.body.velocity))
                    self.assertEqual((ra, da, ia), (rb, db, ib))

    def test_restored_collision_handlers(self):
        world = self.scene()
        world.target.position = world.body.position
        restored = pickle.loads(pickle.dumps(world))
        self.assertTrue(restored.step(0)[3]['contact'])
        self.assertFalse(world.contact)
        projectile = restored.circle(tuple(restored.body.position), .12, 4, sensor=True)
        restored.projectiles.append(projectile)
        self.assertTrue(restored.step(0)[3]['impact'])

    def test_sensor_onsets_are_timed_and_sustained_collision_is_not_repeated(self):
        world = self.scene()
        world.obstacles.append(world.circle((.25, 0.), .32, 2))
        projectile = world.circle((.34, 0.), .12, 4, sensor=True)
        projectile[0].velocity = (-10., 0.)
        world.projectiles.append(projectile)
        info = world.step(0)[3]
        self.assertEqual([event['kind'] for event in info['sensor_events']], ['collision', 'impact'])
        timestamps = [event['timestamp'] for event in info['sensor_events']]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertLess(timestamps[0], timestamps[1])
        for event in info['sensor_events']:
            self.assertEqual(set(event), {'timestamp', 'kind'})
            self.assertGreater(event['timestamp'], 0.)
            self.assertLessEqual(event['timestamp'], world.timestamp)
            self.assertAlmostEqual(event['timestamp'] / .001, round(event['timestamp'] / .001), places=8)
        self.assertAlmostEqual(info['collision_duration'], .01)
        following = world.step(0)[3]
        self.assertTrue(following['collision'])
        self.assertEqual(following['sensor_events'], [])
        world.target.position = world.body.position
        contact = world.step(0)[3]
        self.assertIn('contact', [event['kind'] for event in contact['sensor_events']])

    def test_distinct_projectile_hits_are_charged_separately(self):
        world = self.scene()
        for y in (-.05, .05):
            world.projectiles.append(world.circle((0., y), .12, 4, sensor=True))
        info = world.step(0)[3]
        self.assertEqual(info['impact_count'], 2)
        self.assertEqual(sum(event['kind'] == 'impact' for event in info['sensor_events']), 2)
        self.assertAlmostEqual(world.integrity, .76 + .0002 * .01)


if __name__ == '__main__':
    unittest.main()
