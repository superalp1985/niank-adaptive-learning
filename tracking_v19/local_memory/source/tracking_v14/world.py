"""Reduced physical tracking worlds with visible, nonprivileged sensors."""

from pathlib import Path
import itertools
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tracking_runtime"))
import pymunk

DT, LIMIT, OBS = 0.12, 128, 32
ACTIONS = np.array([(0, 0)] + [(np.cos(i * np.pi / 4), np.sin(i * np.pi / 4))
                             for i in range(8)], np.float32)


def combinations(heldout):
    return [c for c in itertools.product(range(3), repeat=3)
            if ((sum(c) % 3 == 0) == heldout)]


class World:
    def __init__(self, seed, combo=None, heldout=False):
        self.rng = np.random.default_rng(seed)
        self.fixed_combo, self.heldout = combo, heldout
        self.episode = -1
        self.reset()

    def _circle(self, position, radius, kind, body_type, sensor=False):
        body = pymunk.Body(1, 1, body_type=body_type)
        body.position = tuple(position)
        shape = pymunk.Circle(body, radius)
        shape.collision_type, shape.sensor = kind, sensor
        shape.elasticity, shape.friction = 0.0, 0.0
        self.space.add(body, shape)
        return body, shape

    def add_obstacle(self, position, radius):
        body, shape = self._circle(position, radius, 2, pymunk.Body.STATIC)
        self.obstacles.append((body, shape))

    def add_projectile(self, position, velocity):
        body, shape = self._circle(position, 0.09, 4, pymunk.Body.KINEMATIC, True)
        shape.filter = pymunk.ShapeFilter(categories=8, mask=1)
        body.velocity = tuple(velocity)
        self.projectiles.append((body, shape))

    def _collision(self, arbiter, space, data):
        self.collision = True

    def _target_contact(self, arbiter, space, data):
        self.touched = True

    def _impact(self, arbiter, space, data):
        self.impact = True

    def reset(self):
        self.episode += 1
        options = combinations(self.heldout)
        self.combo = self.fixed_combo or options[self.rng.integers(len(options))]
        self.space = pymunk.Space()
        self.space.iterations = 10
        self.space.on_collision(1, 2, pre_solve=self._collision)
        self.space.on_collision(1, 3, pre_solve=self._target_contact)
        self.space.on_collision(1, 4, pre_solve=self._impact)
        self.obstacles, self.projectiles = [], []
        self.body, self.body_shape = self._circle(
            self.rng.uniform(-2.3, 2.3, 2), 0.16, 1, pymunk.Body.DYNAMIC)
        self.body_shape.filter = pymunk.ShapeFilter(categories=1)
        target = self.rng.uniform(-2.3, 2.3, 2)
        while np.linalg.norm(target - np.array(self.body.position)) < 1.8:
            target = self.rng.uniform(-2.3, 2.3, 2)
        self.target, self.target_shape = self._circle(
            target, 0.22, 3, pymunk.Body.KINEMATIC, True)
        self.target_shape.filter = pymunk.ShapeFilter(categories=4, mask=1)
        angle = self.rng.uniform(0, 2 * np.pi)
        self.target_direction = np.array([np.cos(angle), np.sin(angle)]) * 0.3
        self.turn = self.rng.choice([-1., 1.]) * 0.055
        for a, b in (((-3, -3), (3, -3)), ((3, -3), (3, 3)),
                     ((3, 3), (-3, 3)), ((-3, 3), (-3, -3))):
            wall = pymunk.Segment(self.space.static_body, a, b, 0.05)
            wall.collision_type, wall.friction = 2, 0.0
            self.space.add(wall)
        layout = self.combo[0]
        points = []
        if layout == 1:
            points = [(self.rng.uniform(-1.8, 1.8, 2), 0.30) for _ in range(5)]
        elif layout == 2:
            gap = self.rng.choice([-1., 1.])
            points = [(np.array([0., y]), 0.32) for y in np.arange(-2.4, 2.5, 0.65)
                      if abs(y - gap) > 0.55]
        for point, radius in points:
            if min(np.linalg.norm(point - np.array(self.body.position)),
                   np.linalg.norm(point - target)) > radius + 0.4:
                self.add_obstacle(point, radius)
        self.steps, self.collision, self.impact, self.touched = 0, False, False, False
        self.distance = np.linalg.norm(np.array(self.target.position) -
                                       np.array(self.body.position))
        self.attack_offset = int(self.rng.integers(20))
        return self.observe()

    def observe(self):
        position = np.array(self.body.position)
        obs = np.zeros(OBS, np.float32)
        obs[:2] = np.array(self.body.velocity) / 1.4
        obs[2:4] = (np.array(self.target.position) - position) / 6
        obs[4:8] = np.array([3 + position[0], 3 - position[0],
                            3 + position[1], 3 - position[1]]) / 6
        for items, start in ((self.obstacles, 8), (self.projectiles, 17)):
            nearby = sorted(items, key=lambda item: (item[0].position -
                            self.body.position).length)[:3]
            for i, (body, shape) in enumerate(nearby):
                offset = start + 3 * i
                obs[offset:offset + 2] = (np.array(body.position) - position) / 6
                obs[offset + 2] = shape.radius
        obs[26:28] = position / 6
        obs[28], obs[29] = 1., min(obs[4:8])
        obs[30], obs[31] = float(self.collision), float(self.impact)
        return obs

    def raster(self):
        raw = np.zeros((3, 16, 16), np.uint8)
        def draw(position, channel, value, cross=False):
            xy = np.clip(((np.array(position) + 3) / 6 * 15).astype(int), 0, 15)
            x, y = xy
            raw[channel, y, x] = value
            if cross:
                raw[channel, max(y - 1, 0):min(y + 2, 16), x] = value
                raw[channel, y, max(x - 1, 0):min(x + 2, 16)] = value
        draw(self.body.position, 0, 255)
        draw(self.target.position, 1, 255, cross=True)
        for body, _ in self.obstacles:
            draw(body.position, 0, 110, cross=True)
        for body, _ in self.projectiles:
            draw(body.position, 2, 255)
        return raw

    def step(self, action):
        if not 0 <= int(action) < len(ACTIONS):
            raise ValueError("Invalid action")
        self.collision, self.impact, self.touched = False, False, False
        self.body.velocity = tuple(0.65 * np.array(self.body.velocity) +
                                   0.35 * 1.4 * ACTIONS[int(action)])
        motion = self.combo[1]
        if motion == 2:
            c, s = np.cos(self.turn), np.sin(self.turn)
            self.target_direction = np.array([[c, -s], [s, c]]) @ self.target_direction
        if motion:
            for axis in range(2):
                if abs(self.target.position[axis]) > 2.6:
                    self.target_direction[axis] = -np.sign(self.target.position[axis]) * abs(
                        self.target_direction[axis])
            self.target.velocity = tuple(self.target_direction)
        else:
            self.target.velocity = (0, 0)
        if self.combo[2] and self.steps >= 12 and self.steps % 20 == self.attack_offset:
            axis = int(self.rng.integers(2))
            spawn = self.rng.uniform(-2.7, 2.7, 2)
            spawn[axis] = self.rng.choice([-2.85, 2.85])
            aim = (np.array(self.body.position) if self.combo[2] == 1
                   else self.rng.uniform(-1.5, 1.5, 2))
            vector = aim - spawn
            self.add_projectile(spawn, vector / max(np.linalg.norm(vector), 1e-6) * 1.65)
        for _ in range(3):
            self.space.step(DT / 3)
        for body, shape in list(self.projectiles):
            if max(abs(body.position.x), abs(body.position.y)) > 3.2:
                self.space.remove(shape, body)
                self.projectiles.remove((body, shape))
        distance = np.linalg.norm(np.array(self.target.position) -
                                  np.array(self.body.position))
        success = self.touched and not self.impact
        progress = self.distance - distance
        reward = (2 * progress - 0.015 - 0.004 * float(action != 0)
                  - 0.4 * self.collision - 2 * self.impact + 8 * success)
        self.steps += 1
        done = bool(success or self.impact or self.steps >= LIMIT)
        self.distance = distance
        info = dict(success=bool(success), collision=bool(self.collision),
                    impact=bool(self.impact), contact=bool(self.touched),
                    progress=float(progress), distance=float(distance),
                    duration=self.steps, combo=self.combo, episode=self.episode,
                    timeout=bool(self.steps >= LIMIT and not success and not self.impact))
        return self.observe(), float(reward), done, info
