"""Summarize measured attention/head trajectories without loading the learner."""
import argparse
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def inspect(path):
    path = Path(path).resolve()
    database = sqlite3.connect((path/"events.sqlite").as_uri()+"?mode=ro", uri=True)
    try:
        rows = database.execute(
            "SELECT step,body,head_action,body_action,yaw,velocity,entropy,"
            "attention_change,attention FROM vision ORDER BY step,body").fetchall()
        if not rows:
            raise ValueError("No recorded attention decisions")
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        origin = manifest["source_runtime"]["runtime_step"]
        end = origin+manifest["config"]["additional_steps"]
        rows = [row for row in rows if origin < row[0] <= end]
    finally:
        database.close()
    distributions = np.stack([np.frombuffer(row[8], np.float16).astype(np.float32) for row in rows])
    sums = distributions.sum(1)
    if not np.allclose(sums, 1., atol=.001) or not np.isfinite(distributions).all():
        raise AssertionError("Invalid stored attention probabilities")
    distributions /= sums[:, None]
    steps = np.array([row[0]-origin for row in rows])
    bodies = np.array([row[1] for row in rows])
    gaze = np.array([row[2] for row in rows])
    yaw = np.array([row[4] for row in rows])
    velocity = np.array([row[5] for row in rows])
    entropy = np.array([row[6] for row in rows])
    variation = np.array([row[7] for row in rows])
    outward_at_limit = (((gaze == 1) & (yaw >= np.pi/2-1e-6)) |
                        ((gaze == 2) & (yaw <= -np.pi/2+1e-6)))
    # The multiplier, rather than argmax alone, reveals how weak/strong selection is.
    multipliers = .25+.75*64*distributions
    stages = []
    for low, high in ((0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)):
        mask = (steps > low) & (steps <= high)
        if not mask.any():
            continue
        stages.append(dict(
            new_steps_from=low, new_steps_to=high,
            actual_last_step=int(steps[mask].max()),
            decisions=int(mask.sum()),
            entropy_mean=float(entropy[mask].mean()),
            entropy_uniform=float(np.log(64)),
            attention_change_l1_mean=float(variation[mask].mean()),
            max_sector_probability_mean=float(distributions[mask].max(1).mean()),
            multiplier_min=float(multipliers[mask].min()),
            multiplier_max=float(multipliers[mask].max()),
            multiplier_top_mean=float(multipliers[mask].max(1).mean()),
            absolute_head_degrees_mean=float(np.abs(yaw[mask]).mean()*180/np.pi),
            head_limit_fraction=float((np.abs(yaw[mask]) >= np.pi/2-1e-6).mean()),
            sampled_preaction_moving_fraction=float((np.abs(velocity[mask]) > 1e-9).mean()),
            outward_command_at_limit_fraction=float(outward_at_limit[mask].mean()),
            head_action_counts=np.bincount(gaze[mask], minlength=3).tolist()))
    by_body = []
    for body in np.unique(bodies):
        mask = bodies == body
        by_body.append(dict(
            body=int(body), decisions=int(mask.sum()),
            entropy_mean=float(entropy[mask].mean()),
            attention_change_l1_mean=float(variation[mask].mean()),
            multiplier_top_mean=float(multipliers[mask].max(1).mean()),
            absolute_head_degrees_mean=float(np.abs(yaw[mask]).mean()*180/np.pi),
            head_limit_fraction=float((np.abs(yaw[mask]) >= np.pi/2-1e-6).mean()),
            sampled_preaction_moving_fraction=float((np.abs(velocity[mask]) > 1e-9).mean()),
            outward_command_at_limit_fraction=float(outward_at_limit[mask].mean()),
            head_action_counts=np.bincount(gaze[mask], minlength=3).tolist()))
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    selected = np.flatnonzero(bodies == 0)[::max(1, int((bodies == 0).sum()/2000))]
    times = steps[selected]*.01
    span = max(.1, float(np.max(np.abs(multipliers[selected]-1))))
    picture = axes[0].imshow(multipliers[selected].T, origin="lower", aspect="auto",
                            extent=(times[0], times[-1], 0, 360), cmap="viridis",
                            vmin=max(.25, 1-span), vmax=1+span)
    figure.colorbar(picture, ax=axes[0], label="Actual feature multiplier")
    axes[0].set(title="Body 0: learned spatial emphasis (all sectors retained)",
                ylabel="Body-relative azimuth (degrees)")
    axes[1].plot(times, yaw[selected]*180/np.pi, linewidth=.7, color="#ab3c59")
    axes[1].set(ylabel="Relative head yaw (degrees)", ylim=(-95, 95))
    axes[2].plot(times, entropy[selected], linewidth=.8, color="#286c85")
    axes[2].axhline(np.log(64), color="#777777", linestyle="--", label="Uniform")
    axes[2].set(ylabel="Attention entropy", xlabel="New practice seconds per body")
    axes[2].legend()
    figure.savefig(path/"attention_head_process.png", dpi=160)
    plt.close(figure)
    report = dict(stages=stages, by_body=by_body, total_decisions=len(rows),
                  settling_frames_excluded=True,
                  note="Attention entropy and head travel are activity measures, not task success. "
                       "Yaw and velocity are sampled before the logged command is applied; "
                       "outward-at-limit commands indicate blocked requests, not actual travel.")
    (path/"attention_measurements.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    inspect(parser.parse_args().path)
