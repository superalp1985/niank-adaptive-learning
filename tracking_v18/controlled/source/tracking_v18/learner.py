"""V17 memory with an explicit next-observation feedback boundary."""

import numpy as np

from tracking_v17.learner import EyeMemory as PreviousMemory
from tracking_v17.world import EYES


class EyeMemory(PreviousMemory):
    def __init__(self, worlds, frame_mode):
        super().__init__(worlds, "revisable")
        if frame_mode not in ("legacy", "aligned"):
            raise ValueError("Unknown feedback coordinate mode")
        self.frame_mode = frame_mode

    def feedback(self, local_gain, next_proprio):
        if self.frame_mode == "aligned":
            shifts = np.rint(next_proprio[:, 2] * EYES / 2).astype(int)
            # Store next-frame evidence in the current trace's coordinates.
            local_gain = np.stack([np.roll(gain, int(shift))
                                   for gain, shift in zip(local_gain, shifts)])
        super().feedback(local_gain)
