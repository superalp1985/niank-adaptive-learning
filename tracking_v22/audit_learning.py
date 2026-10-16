"""Measure actual sensor-to-controller signal and LIF numerical sensitivity."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tracking_v22.neural import retina, SpikeState
from tracking_v22.sensory_audit import clean_world


def main(output):
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for distance in (1.5, 4., 8.):
        for angle in (0., .7, 2., 3.5, 5.3):
            world = clean_world()
            # Center scenes are used below8m; diagonal positions allow an8m span.
            direction = np.array([math.cos(angle), math.sin(angle)])
            if distance == 8.:
                world.body.position = tuple(-direction*3.0)
                distance_actual = 6.
            else:
                distance_actual = distance
            world.target.position = tuple(np.array(world.body.position)+direction*distance_actual)
            world.heading = .13
            for body, _ in world.distractors:
                body.position = (-3., 2.5)
            world.invalidate_observation()
            image, _ = world.observe()
            with torch.no_grad():
                feature, cues = retina(torch.as_tensor(image[None], device="cuda"))
            vector = cues["direction"][0].cpu().numpy()
            predicted = math.atan2(vector[1], vector[0])
            truth = angle-world.heading
            error = abs(math.atan2(math.sin(predicted-truth), math.cos(predicted-truth)))
            samples.append(dict(distance=distance_actual, angle=angle,
                                strength=float(cues["target_strength"][0]),
                                direction_error_deg=math.degrees(error),
                                visibility=world.audit_visibility()))
    # Same-color distractors with different 2D markings must not resemble a cross.
    world = clean_world()
    world.target.position = (10., 10.)
    world.distractors[0][0].position = (1.5, .5)
    world.distractors[1][0].position = (-1.5, 1.)
    world.invalidate_observation()
    rgb, _ = world.observe()
    _, distractor_cues = retina(torch.as_tensor(rgb[None], device="cuda"))
    timing = []
    for dt in (.0001, .00005):
        state = SpikeState(4, 32, dt=dt)
        generator = torch.Generator().manual_seed(18)
        drive = 1.2 + torch.rand((100, 4, 32), generator=generator)
        for frame in drive:
            state.advance(frame)
        timing.append(dict(dt=dt, count=float(state.total_spikes.sum()),
                           membrane=state.membrane.tolist(), trace=state.trace.tolist()))
    batch = torch.as_tensor(np.stack([rgb]*8), device="cuda")
    retina(batch)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        retina(batch)
    torch.cuda.synchronize()
    result = dict(samples=samples,
                  distractor_max_strength=float(distractor_cues["target_strength"][0]),
                  neural_resolution=timing,
                  relative_spike_count_difference=abs(timing[0]["count"]-timing[1]["count"])/
                  timing[1]["count"],
                  retina_seconds_per_8_frames=(time.perf_counter()-start)/10)
    (output/"learning_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="neural_resolution"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
