"""Save native model inputs and small physical sensor/timing audit artifacts."""
import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from tracking_v22.world import DT, PHYSICS_DT, World, canonical_checkpoint


def clean_world():
    w = World(42, combo=(0, 0, 0))
    w.body.position, w.body.velocity, w.heading = (0., 0.), (0., 0.), 0.
    w.target.position = (0., 2.)
    for b, _ in w.distractors:
        b.position = (-3., -3.)
    w.distance = 2.
    w.invalidate_observation()
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panels, report = [], {}

    def save(label, world):
        world.invalidate_observation()
        rgb, proprio = world.observe()
        image = Image.fromarray(rgb)
        image.save(args.output / f'{label}.png')
        panels.append((label, image))
        report[label] = dict(world.audit_visibility(), proprio=proprio.tolist())

    w = clean_world()
    w.distractors[0][0].position = (2., 1.)
    w.distractors[1][0].position = (-2., 1.)
    save('near_target_and_pattern_distractors', w)
    w = clean_world()
    w.body.position, w.target.position = (-3.3, -3.3), (3.3, 3.3)
    for body, _ in w.distractors:
        body.position = (-3., 3.)
    threat = w.circle((3.2, 2.), .12, 4, sensor=True)
    w.projectiles.append(threat)
    save('far_target_smallest_threat', w)
    w = clean_world()
    w.target.position = (2., 0.)
    save('forward_seam', w)
    w = clean_world()
    obstacle = w.circle((0., 1.), .42, 2)
    w.obstacles.append(obstacle)
    save('fully_occluded', w)
    obstacle[0].position = (.4, 1.)
    save('partial_occlusion', w)
    obstacle[0].position = (1., 1.)
    save('reappeared', w)
    w = clean_world()
    projectile = w.circle((-.6, 1.), .12, 4, sensor=True)
    projectile[0].velocity = (3., 0.)
    w.projectiles.append(projectile)
    w.target.velocity = (.4, 0.)
    for frame in range(4):
        save(f'constant_motion_{frame:02d}', w)
        for _ in range(5):
            w.step(0)
    mosaic = Image.new('RGB', (2048, len(panels) // 2 * 540), (16, 18, 20))
    draw = ImageDraw.Draw(mosaic)
    for i, (label, image) in enumerate(panels):
        x, y = (i % 2) * 1024, (i // 2) * 540
        draw.text((x + 8, y + 6), label, fill=(235, 235, 235))
        mosaic.paste(image, (x, y + 28))
    mosaic.save(args.output / 'model_input_mosaic.png')
    benchmark = World(7, combo=(1, 2, 3))
    times = []
    for i in range(100):
        start = time.perf_counter()
        benchmark.step(i % 9)
        times.append(time.perf_counter() - start)
    live_worlds = [benchmark]
    world_blob = canonical_checkpoint(live_worlds)
    benchmark = live_worlds[0]
    restored = pickle.loads(world_blob)[0]
    deterministic = True
    for action in (1, 4, 0, 7, 5):
        a, r1, d1, i1 = benchmark.step(action)
        b, r2, d2, i2 = restored.step(action)
        deterministic &= np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        deterministic &= (r1, d1, i1) == (r2, d2, i2)
    report['timing'] = dict(observation_dt=DT, physics_dt=PHYSICS_DT,
                            benchmark_frames=len(times), mean_frame_ms=1000 * float(np.mean(times)),
                            median_frame_ms=1000 * float(np.median(times)),
                            canonical_checkpoint_continuation_identical=bool(deterministic),
                            checkpoint_numerics='live and resumed worlds both load identical saved bytes; unsaved contact solver caches reset at checkpoint')
    report['geometry'] = dict(width=1024, height=512, horizontal_degrees=360,
                              vertical_degrees=180, native_ray_pixels=True,
                              azimuth='(u+.5)*2pi/1024; body forward at seam; body left at u=256',
                              elevation='pi/2-(v+.5)*pi/512; top up',
                              limitation='2.5D planar collisions; spheres and cylinders rendered in 3D; finite pixels can miss tiny distant markers')
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), timing=report['timing'],
                         far=report['far_target_smallest_threat']), indent=2))


if __name__ == '__main__':
    main()
