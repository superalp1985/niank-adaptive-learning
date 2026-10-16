"""Native panorama rotation and a head-centered detail observation.

Rotation is quantized to one native horizontal pixel. Every panoramic pixel
is retained. The inverse rotation uses only reported head proprioception,
before the existing retina; motor and consequence directions stay body-relative.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from tracking_v22.neural import retina
from tracking_v22.world import World as LegacyWorld, WIDTH, DT, PHYSICS_DT

HEAD_LIMIT = math.pi / 2
HEAD_SPEED = math.pi
HEAD_DIM, DETAIL_DIM, EXTRA_DIM = 4, 384, 388
INPUT_DIM = 576 + EXTRA_DIM


class HeadWorld(LegacyWorld):
    def reset(self):
        self.head_yaw = self.head_velocity = self.head_travel = 0.
        return super().reset()

    @classmethod
    def upgrade(cls, world):
        if not isinstance(world, cls):
            world.__class__ = cls
            world.head_yaw = world.head_velocity = world.head_travel = 0.
        return world

    @property
    def head_pixel(self):
        return int(round(self.head_yaw * WIDTH / (2 * math.pi)))

    def head_proprio(self):
        # cos(yaw)-1 makes a missing historical head observation all zeros.
        return np.array([math.sin(self.head_yaw), math.cos(self.head_yaw)-1,
                         self.head_velocity/HEAD_SPEED, self.head_yaw/HEAD_LIMIT],
                        np.float32)

    def observe(self):
        rgb, proprio = super().observe()
        return np.roll(rgb, -self.head_pixel, axis=1), proprio, self.head_proprio()

    def step(self, action):
        if int(action) != action or not 0 <= int(action) < 27:
            raise ValueError("Joint action must be an integer from 0 to 26")
        gaze, body = divmod(int(action), 9)
        before = self.head_yaw
        requested = (0., HEAD_SPEED, -HEAD_SPEED)[gaze]
        elapsed = 0.
        while elapsed < DT - 1e-12:
            h = min(PHYSICS_DT, DT-elapsed)
            self.head_yaw = float(np.clip(self.head_yaw+requested*h,
                                          -HEAD_LIMIT, HEAD_LIMIT))
            elapsed += h
        self.head_velocity = (self.head_yaw-before)/DT
        travel = abs(self.head_yaw-before)
        self.head_travel += travel
        observation, reward, done, info = super().step(body)
        info.update(body_action=body, head_action=gaze, head_yaw=self.head_yaw,
                    head_velocity=self.head_velocity, head_travel=travel,
                    head_pixel=self.head_pixel)
        return observation, reward, done, info


def migrate_input(x):
    """Insert unavailable observations; the original last 32 spikes stay last."""
    if x.shape[-1] == INPUT_DIM:
        return x
    if x.shape[-1] != 576:
        raise ValueError(f"Unsupported input width {x.shape[-1]}")
    return torch.cat((x[..., :512], x.new_zeros((*x.shape[:-1], EXTRA_DIM)),
                      x[..., 512:]), -1)


def migrate_batch(batch):
    return (migrate_input(batch[0]), *batch[1:])


def encode_images(images, head):
    """Only RGB and four head sensors enter; no world/object state is accepted."""
    yaw = torch.atan2(head[:, 0], head[:, 1]+1.)
    pixels = (yaw * WIDTH/(2*math.pi)).round().long()
    # Gathering is an exact circular pixel permutation, including the seam.
    columns = (torch.arange(WIDTH, device=images.device)[None]-pixels[:, None]) % WIDTH
    body_images = images.gather(2, columns[:, None, :, None].expand_as(images))
    features, cues = retina(body_images)
    forward = torch.cat((images[:, 128:384, -128:],
                         images[:, 128:384, :128]), dim=2)
    detail = F.adaptive_avg_pool2d(forward.permute(0, 3, 1, 2).float()/255.,
                                  (8, 16)).flatten(1)
    return features, cues, detail


def compose_input(legacy, head, detail):
    if head.shape[-1] != HEAD_DIM or detail.shape[-1] != DETAIL_DIM:
        raise ValueError("Invalid head or detail dimensions")
    return torch.cat((legacy[:, :512], head, detail, legacy[:, 512:]), dim=1)
