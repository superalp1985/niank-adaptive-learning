"""Export learning-process measurements without treating short runs as verdicts."""
import argparse
import io
import json
import os
from pathlib import Path
import sqlite3

os.environ.setdefault("MPLCONFIGDIR", "E:/niank-temp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracking_v22.run import dump


def stored_measurements(path):
    db = sqlite3.connect((path/"events.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    movement = []
    try:
        for block, step, blob in db.execute("SELECT block,step,batch FROM facts ORDER BY block"):
            with np.load(io.BytesIO(blob), allow_pickle=False) as batch:
                features = batch["a0"].astype(np.float32)
                actions = batch["a1"]
                counts = np.bincount(actions, minlength=9)
                movement.append(dict(block=block, step=step, window_starts=len(actions),
                    stop_action_fraction=float((actions == 0).mean()),
                    mean_speed=float(np.linalg.norm(features[:, 512:514], axis=1).mean()*1.5),
                    action_counts=counts.tolist()))
        credits = [json.loads(row[0]) for row in
                   db.execute("SELECT outcome FROM choices ORDER BY step")]
        if any(row["validation_min_start"] < row["decision_step"] for row in credits):
            raise RuntimeError("Learning credit contains a pre-decision window")
        per_mode = []
        for mode, name in enumerate(("observe", "recent", "recall")):
            selected = [row for row in credits if row["mode"] == mode]
            per_mode.append(dict(mode=name, trials=len(selected),
                negative=sum(row["gain"] < 0 for row in selected),
                mean_gain=float(np.mean([row["gain"] for row in selected])) if selected else None,
                retained_updates=sum(row.get("retained_updates", 0) for row in selected)))
        return dict(
            movement_by_block=movement, credit_by_mode=per_mode,
            settled_choices=len(credits),
            matured_window_starts=sum(row["window_starts"] for row in movement),
            strict_future_windows=True,
            interpretation="Movement uses stored window-start proprioception; overlapping future windows are not independent trials.")
    finally:
        db.close()


def motion_summary(rows):
    count = sum(row["window_starts"] for row in rows)
    return dict(window_starts=count,
        mean_speed=sum(row["window_starts"]*row["mean_speed"] for row in rows)/max(count, 1),
        stop_action_fraction=sum(row["window_starts"]*row["stop_action_fraction"]
                                 for row in rows)/max(count, 1))


def build(path):
    path = Path(path)
    summary = json.loads((path/"training_only_summary.json").read_text())
    curve = summary["curve"]
    steps = np.array([r["step"] for r in curve])
    elapsed = steps*.01
    outcomes = []
    start = dict(step=0, contact=0, collision_seconds=0., impact_events=0,
                 updates=0, choices=[0, 0, 0])
    for point in curve:
        if point["step"] % 512:
            continue
        seconds = (point["step"]-start["step"])*.01*summary["bodies"]
        outcomes.append(dict(start=start["step"], end=point["step"],
            body_seconds=seconds,
            contacts=point["contact"]-start["contact"],
            contacts_per_body_minute=(point["contact"]-start["contact"])*60/seconds,
            collision_time_fraction=(point["collision_seconds"]-start["collision_seconds"])/seconds,
            impacts=point["impact_events"]-start["impact_events"],
            updates=point["updates"], choices=point["choices"],
            window_updates=point["updates"]-start["updates"],
            window_choices=[a-b for a, b in zip(point["choices"], start["choices"])],
            learning_rate=point["current_lr"], prior_fraction=point["prior_fraction"],
            prequential_loss=point["prequential_loss"]))
        start = point
    stages = []
    for complete in sorted((path/"stage_probes").glob("stage_*/COMPLETE.json")):
        item = json.loads(complete.read_text())
        result = item["frozen"]
        motion = motion_summary(stored_measurements(complete.parent)["movement_by_block"])
        stages.append(dict(training_step=item["training_step"],
                           probe_body_seconds=result["simulation_seconds"],
                           contacts=result["contact"], safe_contacts=result["safe_contact"],
                           collision_time_fraction=result["collision_seconds"]/result["simulation_seconds"],
                           impacts=result["impact_events"], prior_fraction=result["prior_fraction"],
                           parameters_unchanged=item["learned_parameters_unchanged"],
                           movement=motion))
    final_probe = json.loads((path/"frozen_probe"/"summary.json").read_text())
    final_motion = motion_summary(stored_measurements(path/"frozen_probe")["movement_by_block"])
    dump(path/"process_measurements.json", dict(training_windows=outcomes,
        frozen_stages=stages, stored_facts=stored_measurements(path),
        final_frozen_observation=dict(steps=final_probe["step"], contacts=final_probe["contact"],
            collision_time_fraction=final_probe["collision_seconds"]/final_probe["simulation_seconds"],
            impacts=final_probe["impact_events"], movement=final_motion),
        training_seconds_per_body=summary["step"]*.01,
        aggregate_training_body_seconds=summary["simulation_seconds"],
        interpretation="Descriptive trajectories; changing scenes and prior reliance are not controlled causal effects."))
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    colors = ["#167d9a", "#d15842", "#658536"]
    axes[0, 0].plot(elapsed, [r["prequential_loss"] for r in curve], color=colors[0])
    axes[0, 0].set_title("Next-window prediction loss (practice)")
    for i, name in enumerate(("observe", "recent learning", "recall")):
        axes[0, 1].plot(elapsed, [r["choices"][i] for r in curve], label=name, color=colors[i])
    axes[0, 1].set_title("Accumulated learning choices")
    axes[0, 1].legend()
    axes[1, 0].plot(elapsed, [r["current_lr"] for r in curve], color=colors[0])
    axes[1, 0].set_title("Autonomously adjusted predictor learning rate")
    axes[1, 1].plot(elapsed, [r["prior_fraction"] for r in curve], color=colors[1])
    axes[1, 1].set_title("Motor-prior multiplier (scheduled)")
    window_time = np.array([r["end"] for r in outcomes])*.01
    axes[2, 0].plot(window_time, [r["contacts_per_body_minute"] for r in outcomes],
                    marker="o", color=colors[2])
    axes[2, 0].set_title("Contacts per body-minute in each 512-step window")
    axes[2, 1].plot(window_time, [r["collision_time_fraction"] for r in outcomes],
                    marker="o", color=colors[1])
    axes[2, 1].set_title("Fraction of body-time spent in contact with obstacles")
    for ax in axes.flat:
        ax.set_xlabel("Simulated seconds per body")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("V22-B: learning, adaptation and physical consequences", fontsize=15)
    fig.savefig(path/"learning_process.png", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(windows=len(outcomes), stages=stages,
                          plot=str(path/"learning_process.png"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    build(parser.parse_args().run)
