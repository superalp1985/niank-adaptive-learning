"""Planar 360-degree RGB eyes and continuous Pymunk practice."""

from pathlib import Path
import itertools
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tracking_runtime"))
import pymunk

DT, SEGMENT, EYES = .12, 1024, 128
ANGLES = np.arange(EYES) * (2 * np.pi / EYES)
BASE_RAYS = np.stack((np.cos(ANGLES), np.sin(ANGLES)), 1)
ACTIONS = np.array([(0., 0.)] + [(np.cos(i * np.pi / 4), np.sin(i * np.pi / 4))
                               for i in range(8)], np.float32)


def combinations(heldout=False):
    return [c for c in itertools.product(range(6), range(4), range(4))
            if ((sum(c) % 4 == 0) == heldout)]


class World:
    def __init__(self, seed, combo=None, heldout=False):
        self.rng = np.random.default_rng(seed)
        self.fixed_combo, self.heldout = combo, heldout
        self.segment = -1
        self.reset()

    def circle(self, position, radius, kind, dynamic=False, sensor=False):
        body = pymunk.Body(1, 1, body_type=pymunk.Body.DYNAMIC if dynamic
                           else pymunk.Body.KINEMATIC)
        body.position = tuple(position)
        shape = pymunk.Circle(body, radius)
        shape.collision_type, shape.sensor = kind, sensor
        shape.friction, shape.elasticity = 0., 0.
        self.space.add(body, shape)
        return body, shape

    def _collision(self, arbiter, space, data):
        self.collision = True

    def _contact(self, arbiter, space, data):
        self.contact = True

    def _impact(self, arbiter, space, data):
        self.impact = True
        for shape in arbiter.shapes:
            if shape.collision_type == 4:
                self.hit_bodies.add(shape.body)

    def free_point(self, away=None):
        for _ in range(200):
            p = self.rng.uniform(-3.4, 3.4, 2)
            clear = all(np.linalg.norm(p - np.array(b.position)) > s.radius + .4
                        for b, s in self.obstacles)
            if clear and (away is None or np.linalg.norm(p - away) > 2):
                return p
        return np.array([-3.3, -3.3])

    def reset(self):
        self.segment += 1
        options = combinations(self.heldout)
        self.combo = self.fixed_combo or options[int(self.rng.integers(len(options)))]
        self.space = pymunk.Space()
        self.space.iterations = 10
        self.space.on_collision(1, 2, pre_solve=self._collision)
        self.space.on_collision(1, 3, pre_solve=self._contact)
        self.space.on_collision(1, 4, pre_solve=self._impact)
        self.obstacles, self.projectiles, self.distractors = [], [], []
        for a, b in (((-4, -4), (4, -4)), ((4, -4), (4, 4)),
                     ((4, 4), (-4, 4)), ((-4, 4), (-4, -4))):
            wall = pymunk.Segment(self.space.static_body, a, b, .05)
            wall.collision_type = 2
            self.space.add(wall)
        layout = self.combo[0]
        points = []
        if layout in (1, 5):
            points = [(self.rng.uniform(-2.8, 2.8, 2), self.rng.uniform(.2, .45))
                      for _ in range(8)]
        elif layout == 2:
            gap = self.rng.uniform(-1.8, 1.8)
            points = [(np.array([0., y]), .32) for y in np.arange(-3.3, 3.4, .65)
                      if abs(y - gap) > .65]
        elif layout == 3:
            points = [(np.array([x, y]), .32)
                      for x, gap in ((-1.25, -1.5), (1.25, 1.5))
                      for y in np.arange(-3., 3.1, .65) if abs(y - gap) > .65]
        elif layout == 4:
            points = [(np.array([x, y]), .32) for x in (-1.5, 1.5)
                      for y in np.arange(-1.5, 1.6, .6)]
            points += [(np.array([x, -1.5]), .32) for x in np.arange(-.9, 1., .6)]
        for p, r in points:
            self.obstacles.append(self.circle(p, r, 2))
        self.body, self.body_shape = self.circle(self.free_point(), .16, 1, True)
        self.body_shape.filter = pymunk.ShapeFilter(categories=1)
        self.target, self.target_shape = self.circle(
            self.free_point(np.array(self.body.position)), .30, 3, sensor=True)
        self.target_shape.filter = pymunk.ShapeFilter(categories=4, mask=1)
        for _ in range(3):
            b, s = self.circle(self.free_point(), .30, 5, sensor=True)
            s.filter = pymunk.ShapeFilter(categories=16, mask=0)
            self.distractors.append((b, s))
        angle = self.rng.uniform(0, 2 * np.pi)
        self.target_direction = np.array([np.cos(angle), np.sin(angle)]) * self.rng.uniform(.2, .65)
        self.turn = self.rng.choice([-1., 1.]) * self.rng.uniform(.025, .10)
        self.heading = float(self.rng.uniform(-np.pi, np.pi))
        self.rotation, self.steps = 0., 0
        self.collision = self.impact = self.contact = False
        self.hit_bodies = set()
        self.since_contact_collision = False
        self.attack_wait = int(self.rng.geometric(.055))
        self.attack_events, self.goal_id = 0, 0
        self.distance = (self.target.position - self.body.position).length
        self._cache = None
        return self.observe()

    def rays(self):
        c, s = np.cos(self.heading), np.sin(self.heading)
        return BASE_RAYS @ np.array([[c, s], [-s, c]])

    def render(self):
        origin, rays = np.array(self.body.position), self.rays()
        with np.errstate(divide="ignore", invalid="ignore"):
            walls = np.where(rays > 0, (4 - origin) / rays, (-4 - origin) / rays)
        walls = np.where(np.abs(rays) > 1e-10, walls, np.inf)
        depth = np.maximum(walls, 0).min(1)
        rgb = np.tile(np.array([90., 95., 105.]), (EYES, 1))
        owner = np.full(EYES, -1, np.int16)
        objects = ([(self.target, self.target_shape, 0)] +
                   [(b, s, 1) for b, s in self.obstacles] +
                   [(b, s, 2) for b, s in self.projectiles] +
                   [(b, s, 3 + i) for i, (b, s) in enumerate(self.distractors)])
        if objects:
            relative = np.stack([np.array(b.position) - origin for b, _, _ in objects])
            radius = np.array([s.radius for _, s, _ in objects])
            projection = rays @ relative.T
            perp2 = np.maximum((relative ** 2).sum(1)[None] - projection ** 2, 0)
            valid = (perp2 <= radius ** 2) & (projection > 0)
            entry = projection - np.sqrt(np.maximum(radius ** 2 - perp2, 0))
            entry = np.where(valid, np.maximum(entry, 0), np.inf)
            closest = entry.argmin(1)
            distance = entry[np.arange(EYES), closest]
            visible = distance < depth
            depth[visible] = distance[visible]
            for index, (body, shape, kind) in enumerate(objects):
                pixels = visible & (closest == index)
                if not pixels.any():
                    continue
                owner[pixels] = kind
                if kind == 0:
                    center = np.arctan2(relative[index, 1], relative[index, 0])
                    ray_angle = ANGLES[pixels] + self.heading
                    offset = np.arctan2(np.sin(ray_angle - center), np.cos(ray_angle - center))
                    extent = np.arcsin(min(shape.radius / max(np.linalg.norm(relative[index]), .001), 1))
                    # The center stripe is a visible marker; no semantic channel is returned.
                    stripe = np.abs(offset) < .32 * extent
                    rgb[pixels] = np.where(stripe[:, None], [235., 235., 220.], [35., 205., 65.])
                elif kind == 1:
                    rgb[pixels] = [125., 100., 145.]
                elif kind == 2:
                    rgb[pixels] = [240., 65., 45.]
                else:
                    rgb[pixels] = [45., 170., 75.] if kind % 2 else [160., 190., 150.]
            # Untagged surfaces produce physical RGB only. Occlusion uses nearest hit.
        illumination = .88 + .12 * np.sin(self.steps * .021 + self.segment)
        shading = .6 + .4 * np.exp(-depth / 4)
        rgb = np.clip(rgb * illumination * shading[:, None], 0, 255).astype(np.uint8)
        return rgb, owner == 0

    def observe(self):
        if self._cache is None:
            rgb, target_visible = self.render()
            c, s = np.cos(self.heading), np.sin(self.heading)
            velocity = np.array(self.body.velocity) @ np.array([[c, -s], [s, c]])
            proprio = np.array([*(velocity / 1.5), self.rotation / np.pi,
                                float(self.collision), float(self.impact),
                                float(self.since_contact_collision)], np.float32)
            self._cache = (rgb, proprio, target_visible)
        return self._cache[0].copy(), self._cache[1].copy()

    def audit_visibility(self):
        self.observe()
        return self._cache[2].copy()

    def _spawn_attack(self):
        axis = int(self.rng.integers(2))
        spawn = self.rng.uniform(-3.8, 3.8, 2)
        spawn[axis] = self.rng.choice([-3.85, 3.85])
        aim = (np.array(self.body.position) if self.combo[2] in (1, 3)
               else self.rng.uniform(-2.5, 2.5, 2))
        direction = aim - spawn
        angle = np.arctan2(direction[1], direction[0])
        speed = self.rng.uniform(1.5, 3.2)
        for deviation in ((-.16, 0., .16) if self.combo[2] == 3 else (0.,)):
            body, shape = self.circle(spawn, .12, 4, sensor=True)
            shape.filter = pymunk.ShapeFilter(categories=8, mask=1)
            body.velocity = (speed * np.cos(angle + deviation), speed * np.sin(angle + deviation))
            self.projectiles.append((body, shape))
        self.attack_events += 1

    def step(self, action):
        if not 0 <= int(action) < 9:
            raise ValueError("Invalid action")
        self.collision = self.impact = self.contact = False
        self.hit_bodies = set()
        c, s = np.cos(self.heading), np.sin(self.heading)
        vector = ACTIONS[int(action)] @ np.array([[c, s], [-s, c]])
        self.body.velocity = tuple(.65 * np.array(self.body.velocity) + .35 * 1.5 * vector)
        velocity = np.array(self.body.velocity)
        self.rotation = 0.
        if np.linalg.norm(velocity) > .15:
            desired = np.arctan2(velocity[1], velocity[0])
            difference = np.arctan2(np.sin(desired - self.heading), np.cos(desired - self.heading))
            self.rotation = float(np.clip(difference, -.18, .18))
            self.heading += self.rotation
        motion = self.combo[1]
        if motion in (2, 3):
            if motion == 3 and self.rng.random() < .045:
                self.turn = self.rng.uniform(-.25, .25)
            c, s = np.cos(self.turn), np.sin(self.turn)
            self.target_direction = self.target_direction @ np.array([[c, s], [-s, c]])
        if motion:
            for axis in range(2):
                if abs(self.target.position[axis]) > 3.5:
                    self.target_direction[axis] = -np.sign(self.target.position[axis]) * abs(self.target_direction[axis])
            self.target.velocity = tuple(self.target_direction)
        if self.combo[0] == 5:
            for i, (body, _) in enumerate(self.obstacles):
                body.velocity = (0., .28 * np.cos(self.steps * .025 + i))
        spawned = False
        if self.combo[2]:
            self.attack_wait -= 1
            if self.attack_wait <= 0:
                self._spawn_attack()
                self.attack_wait = int(self.rng.geometric(.055))
                spawned = True
        for _ in range(3):
            self.space.step(DT / 3)
        for body, shape in list(self.projectiles):
            if body in self.hit_bodies or max(abs(body.position.x), abs(body.position.y)) > 4.2:
                self.space.remove(shape, body)
                self.projectiles.remove((body, shape))
        self.since_contact_collision |= self.collision
        distance = (self.target.position - self.body.position).length
        progress = self.distance - distance
        success = self.contact and not self.impact
        safe = success and not self.since_contact_collision
        reward = (2 * progress - .015 - .004 * (action != 0) - .4 * self.collision
                  - 3 * self.impact + 8 * success + 4 * safe)
        self.steps += 1
        info = dict(contact=bool(success), safe_contact=bool(safe),
                    collision=bool(self.collision), impact=bool(self.impact),
                    spawned=spawned, progress=float(progress), distance=float(distance),
                    combo=self.combo, segment=self.segment, time=self.steps, goal_id=self.goal_id)
        if success:
            self.target.position = tuple(self.free_point(np.array(self.body.position)))
            self.goal_id += 1
            self.since_contact_collision = False
        if self.impact:
            self.body.velocity = tuple(.3 * np.array(self.body.velocity))
        self.distance = (self.target.position - self.body.position).length
        self._cache = None
        return self.observe(), float(reward), self.steps >= SEGMENT, info
