"""Native spherical RGB sensor over planar Pymunk motion.

Pixel centers: azimuth=(u+.5)*2*pi/1024 relative to heading; +azimuth is
body-left (+y), so forward crosses the left/right seam. Elevation is
pi/2-(v+.5)*pi/512 (top up). All pixels trace independent 3D rays.
The observer and sphere centers are z=.35; obstacles are vertical cylinders.
Physics is explicitly 2.5D: z does not affect motion or contact collision.
Only RGB and eight physical body sensors are returned to the policy.
Info sensor_events carries identity-free contact/collision/impact onsets at
physics substep end timestamps; sustained touch does not create repeat onsets.
Collision wear/reward integrate touching duration; projectile damage counts
distinct hitting bodies once per frame, including simultaneous hits.
"""
import math
import pickle

import numpy as np

from tracking_v17.world import World as SceneWorld, ACTIONS, combinations

DT, PHYSICS_DT = .01, .001
WIDTH, HEIGHT = 1024, 512
SEGMENT = 12_288  # 122.88 simulated seconds, matching the old segment duration.
CAMERA_Z = .35
AZIMUTH = (np.arange(WIDTH, dtype=np.float32) + .5) * (2 * np.pi / WIDTH)
ELEVATION = np.pi / 2 - (np.arange(HEIGHT, dtype=np.float32) + .5) * (np.pi / HEIGHT)
COS_A, SIN_A = np.cos(AZIMUTH), np.sin(AZIMUTH)
COS_E, SIN_E = np.cos(ELEVATION), np.sin(ELEVATION)


def canonical_checkpoint(worlds):
    """Replace live worlds by checkpoint-canonical worlds and return saved bytes.

    Use only between completed action batches, with a mutable list of worlds:
    ``world_blob = canonical_checkpoint(worlds)``. Store these exact bytes in
    the runtime checkpoint; resume with ``worlds = pickle.loads(world_blob)``.
    The list keeps its identity, but callers must release old World references.

    Pymunk does not serialize all solver state (notably contact bias velocity).
    A raw dump followed by continuing the original world can therefore diverge
    from loading it during penetration/contact. Both paths here start as loads
    of identical bytes. This explicitly resets unsaved numerical solver caches
    at every checkpoint, without advancing simulated time or consuming RNG.
    It preserves physical positions/velocities, sensors, and the random state;
    it does not promise equivalence to an uninterrupted, unsaved trajectory.
    """
    if not isinstance(worlds, list):
        raise TypeError('canonical_checkpoint requires a mutable list of worlds')
    snapshot = pickle.dumps(worlds, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(snapshot)
    worlds[:] = restored
    return snapshot


class World(SceneWorld):
    physics_dt = PHYSICS_DT

    def reset(self):
        self.timestamp = 0.
        self.frames = 0
        self._frame_timestamp = None
        self.energy = self.integrity = 1.
        self.next_turn = float(self.rng.exponential(1 / .3837))
        self.next_attack = float(self.rng.exponential(1 / .4715))
        self.sensor_events = []
        self._substep_timestamp = 0.
        self._substep_collision = False
        result = super().reset()
        self._register_collision_callbacks()
        # Legacy layout sampling is reused; all rates below are in seconds.
        self.turn /= .12
        return result

    def _register_collision_callbacks(self):
        self.space.on_collision(1, 2, begin=self._collision_begin, pre_solve=self._collision)
        self.space.on_collision(1, 3, begin=self._contact_begin, pre_solve=self._contact)
        self.space.on_collision(1, 4, begin=self._impact_begin, pre_solve=self._impact)

    def _sensor_onset(self, kind):
        # A begin callback occurs once per touching shape pair, until separation.
        # Only body-sensor kind and physical timestamp leave the environment.
        self.sensor_events.append(dict(timestamp=self._substep_timestamp, kind=kind))

    def _collision_begin(self, arbiter, space, data):
        self._sensor_onset('collision')

    def _contact_begin(self, arbiter, space, data):
        self._sensor_onset('contact')

    def _impact_begin(self, arbiter, space, data):
        self._sensor_onset('impact')

    def _collision(self, arbiter, space, data):
        self._substep_collision = True
        super()._collision(arbiter, space, data)

    def invalidate_observation(self):
        """Invalidate after explicit audit-only scene edits."""
        self._cache = None

    def _background(self):
        """Exact floor and square-wall ray distances, independent of objects."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        dx = COS_A * c - SIN_A * s
        dy = SIN_A * c + COS_A * s
        x, y = self.body.position
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.where(dx > 0, (4 - x) / dx, (-4 - x) / dx)
            ty = np.where(dy > 0, (4 - y) / dy, (-4 - y) / dy)
            wall = np.maximum(np.minimum(tx, ty), .001)[None, :] / COS_E[:, None]
            floor = np.where(SIN_E < 0, -CAMERA_Z / SIN_E, np.inf)[:, None]
        depth = np.minimum(wall, floor).astype(np.float32)
        rgb = np.empty((HEIGHT, WIDTH, 3), np.uint8)
        rgb[:] = (82, 91, 107)
        rgb[floor < wall] = (48, 55, 61)
        return rgb, depth

    def _patch(self, relative, radius, cylinder=False):
        distance = max(float(np.linalg.norm(relative)), 1e-7)
        center = (math.atan2(relative[1], relative[0]) - self.heading) % (2 * math.pi)
        half = math.asin(min(radius / distance, 1.)) if distance > radius else math.pi
        lo = math.floor((center - half) * WIDTH / (2 * math.pi) - 1)
        hi = math.ceil((center + half) * WIDTH / (2 * math.pi) + 1)
        columns = np.arange(lo, min(hi + 1, lo + WIDTH)) % WIDTH
        if cylinder:
            near = max(distance - radius, .001)
            bottom = math.atan2(-CAMERA_Z, near)
            top = math.atan2(1.1 - CAMERA_Z, near)
        else:
            bottom, top = -half, half
        v0 = max(0, math.floor((math.pi / 2 - top) * HEIGHT / math.pi) - 1)
        v1 = min(HEIGHT, math.ceil((math.pi / 2 - bottom) * HEIGHT / math.pi) + 1)
        return np.arange(v0, v1), columns

    def render(self):
        rgb, depth = self._background()
        owner = np.full((HEIGHT, WIDTH), -1, np.int16)
        marker = np.zeros((HEIGHT, WIDTH), bool)
        origin = np.array(self.body.position)
        c, s = math.cos(self.heading), math.sin(self.heading)
        objects = ([(self.target, self.target_shape, 0)] +
                   [(b, shape, 1) for b, shape in self.obstacles] +
                   [(b, shape, 2) for b, shape in self.projectiles] +
                   [(b, shape, 3 + i) for i, (b, shape) in enumerate(self.distractors)])
        threat_counts = []
        for index, (body, shape, kind) in enumerate(objects):
            rel = np.array(body.position) - origin
            radius = shape.radius
            rows, cols = self._patch(rel, radius, cylinder=kind == 1)
            if not len(rows) or not len(cols):
                continue
            ray_x = (COS_A[cols] * c - SIN_A[cols] * s)[None, :] * COS_E[rows, None]
            ray_y = (SIN_A[cols] * c + COS_A[cols] * s)[None, :] * COS_E[rows, None]
            projection = ray_x * rel[0] + ray_y * rel[1]
            if kind == 1:
                horizontal2 = COS_E[rows, None] ** 2
                discriminant = projection ** 2 - horizontal2 * (float(rel @ rel) - radius ** 2)
                entry = (projection - np.sqrt(np.maximum(discriminant, 0))) / horizontal2
                exit_hit = (projection + np.sqrt(np.maximum(discriminant, 0))) / horizontal2
                entry = np.where(entry > 1e-5, entry, exit_hit)
                hit_z = CAMERA_Z + entry * SIN_E[rows, None]
                valid = (discriminant >= 0) & (entry > 0) & (hit_z >= 0) & (hit_z <= 1.1)
                # Cylinder top/bottom caps are also actual depth-tested surfaces.
                with np.errstate(divide='ignore', invalid='ignore'):
                    cap = np.where(SIN_E[rows, None] > 0,
                                   (1.1 - CAMERA_Z) / SIN_E[rows, None],
                                   -CAMERA_Z / SIN_E[rows, None])
                cap_ok = ((cap * ray_x - rel[0]) ** 2 +
                          (cap * ray_y - rel[1]) ** 2 <= radius ** 2) & (cap > 0)
                entry = np.minimum(np.where(valid, entry, np.inf), np.where(cap_ok, cap, np.inf))
            else:
                discriminant = projection ** 2 - (float(rel @ rel) - radius ** 2)
                near = projection - np.sqrt(np.maximum(discriminant, 0))
                far = projection + np.sqrt(np.maximum(discriminant, 0))
                entry = np.where(near > 1e-5, near, far)
                entry = np.where((discriminant >= 0) & (entry > 0), entry, np.inf)
            grid = np.ix_(rows, cols)
            old_depth = depth[grid]
            visible = entry < old_depth
            if not visible.any():
                continue
            pixels = rgb[grid]
            local_marker = np.zeros(visible.shape, bool)
            if kind == 0 or kind >= 3:
                # Surface coordinates on the observer-facing sphere. A cross needs
                # both vertical and horizontal structure, including away from equator.
                distance = max(float(np.linalg.norm(rel)), 1e-6)
                surface = np.where(np.isfinite(entry), entry, 0.)
                lateral = (surface * ray_x - rel[0]) * (-rel[1] / distance)
                lateral += (surface * ray_y - rel[1]) * (rel[0] / distance)
                vertical = surface * SIN_E[rows, None]
                a, b = np.abs(lateral) / radius, np.abs(vertical) / radius
                if kind == 0:
                    local_marker = ((a < .17) & (b < .80)) | ((b < .17) & (a < .80))
                elif kind % 2:
                    local_marker = (np.abs(a - b) < .14) & (a < .73) & (b < .73)
                else:
                    local_marker = (a > .48) & (a < .72) & (b < .72)
                color = np.empty_like(pixels)
                color[:] = (35, 205, 65)
                color[local_marker] = (242, 245, 230)
            else:
                color = np.empty_like(pixels)
                color[:] = (142, 84, 180) if kind == 1 else (245, 52, 42)
            # Range shading preserves saturated object hue and marker contrast.
            shade = (.72 + .28 * np.exp(-np.minimum(entry, 40) / 6)).astype(np.float32)
            color = (color * shade[..., None]).astype(np.uint8)
            pixels[visible] = color[visible]
            rgb[grid] = pixels
            old_depth[visible] = entry[visible]
            depth[grid] = old_depth
            patch_owner = owner[grid]
            patch_owner[visible] = index
            owner[grid] = patch_owner
            patch_marker = marker[grid]
            patch_marker[visible] = local_marker[visible] & (kind == 0)
            marker[grid] = patch_marker
        self._target_mask = owner == 0
        self._marker_mask = marker
        projectile_indices = [i for i, obj in enumerate(objects) if obj[2] == 2]
        for index in projectile_indices:
            threat_counts.append(int(np.count_nonzero(owner == index)))
        self._visibility = dict(target_pixels=int(self._target_mask.sum()),
                                marker_pixels=int(marker.sum()),
                                threat_pixels=sum(threat_counts),
                                smallest_threat_pixels=min(threat_counts, default=0),
                                projectile_pixels=threat_counts)
        return rgb, self._target_mask

    def observe(self):
        if self._cache is None:
            rgb, _ = self.render()
            c, s = math.cos(self.heading), math.sin(self.heading)
            vx, vy = self.body.velocity
            proprio = np.array([(vx * c + vy * s) / 1.5,
                                (-vx * s + vy * c) / 1.5,
                                self.rotation / np.pi, float(self.collision),
                                float(self.impact), self.energy, self.integrity,
                                float(self.contact)], np.float32)
            self._cache = (rgb, proprio)
            if self._frame_timestamp != self.timestamp:
                self.frames += 1
                self._frame_timestamp = self.timestamp
        return self._cache[0].copy(), self._cache[1].copy()

    def audit_visibility(self):
        if self._cache is None:
            self.observe()
        return dict(self._visibility, timestamp=self.timestamp, frames=self.frames)

    def step(self, action):
        if int(action) != action or not 0 <= int(action) < 9:
            raise ValueError('Action must be an integer from 0 to 8')
        if self.physics_dt <= 0 or self.physics_dt > PHYSICS_DT:
            raise ValueError('Physics timestep must be in (0, .001] seconds')
        self.collision = self.impact = self.contact = False
        self.hit_bodies = set()
        self.sensor_events = []
        collision_duration = 0.
        c, s = math.cos(self.heading), math.sin(self.heading)
        ax, ay = ACTIONS[int(action)]
        desired = np.array([ax * c - ay * s, ax * s + ay * c]) * 1.5
        spawned = False
        start = self.timestamp
        elapsed = 0.
        while elapsed < DT - 1e-12:
            h = min(self.physics_dt, DT - elapsed)
            now = start + elapsed
            velocity = np.array(self.body.velocity)
            self.body.velocity = tuple(desired + (velocity - desired) * math.exp(-h / .2786))
            if np.linalg.norm(velocity) > .15:
                angle = math.atan2(velocity[1], velocity[0])
                difference = math.atan2(math.sin(angle - self.heading), math.cos(angle - self.heading))
                self.rotation = float(np.clip(difference / h, -1.5, 1.5))
                self.heading = math.atan2(math.sin(self.heading + self.rotation * h),
                                          math.cos(self.heading + self.rotation * h))
            else:
                self.rotation = 0.
            motion = self.combo[1]
            if motion == 3 and now >= self.next_turn:
                self.turn = float(self.rng.uniform(-.25, .25) / .12)
                self.next_turn = now + float(self.rng.exponential(1 / .3837))
            if motion in (2, 3):
                tc, ts = math.cos(self.turn * h), math.sin(self.turn * h)
                tx, ty = self.target_direction
                self.target_direction = np.array([tx * tc - ty * ts, tx * ts + ty * tc])
            if motion:
                for axis in range(2):
                    if abs(self.target.position[axis]) > 3.5:
                        self.target_direction[axis] = -np.sign(self.target.position[axis]) * abs(self.target_direction[axis])
                self.target.velocity = tuple(self.target_direction)
            if self.combo[0] == 5:
                for i, (body, _) in enumerate(self.obstacles):
                    body.velocity = (0., .28 * math.cos(now * (.025 / .12) + i))
            if self.combo[2] and now >= self.next_attack:
                self._spawn_attack()
                self.next_attack = now + float(self.rng.exponential(1 / .4715))
                spawned = True
            self._substep_timestamp = start + min(DT, elapsed + h)
            self._substep_collision = False
            self.space.step(h)
            collision_duration += h * self._substep_collision
            elapsed += h
        self.timestamp = start + DT
        for body, shape in list(self.projectiles):
            if body in self.hit_bodies or max(abs(body.position.x), abs(body.position.y)) > 4.2:
                self.space.remove(shape, body)
                self.projectiles.remove((body, shape))
        self.since_contact_collision |= self.collision
        distance = (self.target.position - self.body.position).length
        progress = self.distance - distance
        success = self.contact and not self.impact
        safe = success and not self.since_contact_collision
        self.energy = float(np.clip(self.energy - DT * (.008 + .012 * (action != 0))
                                    + .18 * success, 0., 1.))
        impact_count = len(self.hit_bodies)
        self.integrity = float(np.clip(self.integrity - .12 * impact_count
                                       - .025 * collision_duration + .0002 * DT, 0., 1.))
        reward = (2 * progress - DT * (.015 + .004 * (action != 0)) / .12
                  - .4 / .12 * collision_duration - 3 * impact_count + 8 * success + 4 * safe)
        self.steps += 1
        info = dict(timestamp=self.timestamp, time=self.timestamp, physics_dt=self.physics_dt,
                    contact=bool(success), safe_contact=bool(safe), collision=bool(self.collision),
                    impact=bool(self.impact), spawned=spawned, progress=float(progress),
                    impact_count=impact_count, collision_duration=float(collision_duration),
                    sensor_events=[dict(event) for event in self.sensor_events],
                    distance=float(distance), combo=self.combo, segment=self.segment, goal_id=self.goal_id)
        if success:
            self.target.position = tuple(self.free_point(np.array(self.body.position)))
            self.goal_id += 1
            self.since_contact_collision = False
        if self.impact:
            self.body.velocity = tuple(.3 * np.array(self.body.velocity))
        self.distance = (self.target.position - self.body.position).length
        self.invalidate_observation()
        observation = self.observe()
        info['visibility'] = self.audit_visibility()
        done = self.steps >= SEGMENT or self.integrity <= 0 or self.energy <= 0
        return observation, float(reward), bool(done), info

    def __getstate__(self):
        state = self.__dict__.copy()
        # Frame caches are derived data; rebuild them after restoring physics.
        state['_cache'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Pymunk callbacks must target this restored world, not a copied callback owner.
        self._register_collision_callbacks()
