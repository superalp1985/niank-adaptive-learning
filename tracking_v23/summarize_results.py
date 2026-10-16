"""Compact observed outcomes and a process figure; no success classification."""
import argparse
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(summary):
    return {key: summary[key] for key in (
        "contact", "safe_contact", "collision_fraction", "impact_events",
        "contacts_per_body_minute", "progress")}


def practice_intervals(path, origin, steps, bodies):
    connection = sqlite3.connect((path/"events.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    intervals = []
    body_rows = [dict(body=index, contact=0, safe_contact=0, collision_frames=0,
                      impact_events=0, frames=0) for index in range(bodies)]
    try:
        for lower in range(0, steps, 8192):
            upper = min(lower+8192, steps)
            tally = dict(new_steps_from=lower, new_steps_to=upper, contact=0,
                         safe_contact=0, collision_frames=0, impact_events=0, frames=0)
            for body, raw in connection.execute(
                    "SELECT body,facts FROM events WHERE step>? AND step<=?",
                    (origin+lower, origin+upper)):
                event = json.loads(raw)
                for target in (tally, body_rows[body]):
                    target["contact"] += event["contact"]
                    target["safe_contact"] += event["safe_contact"]
                    target["collision_frames"] += event["collision"]
                    target["impact_events"] += event["impact_count"]
                    target["frames"] += 1
            tally["collision_fraction"] = tally["collision_frames"]/max(tally["frames"], 1)
            tally["contacts_per_body_minute"] = tally["contact"]/max(tally["frames"]*.01/60, 1e-9)
            intervals.append(tally)
    finally:
        connection.close()
    return dict(intervals=intervals, by_body=body_rows)


def summarize(root):
    root = Path(root)
    result = {}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"baseline": "#3768a2", "active": "#be4a59"}
    for arm in ("baseline", "active"):
        complete = read(root/arm/"COMPLETE.json")
        train = read(root/arm/"training_only_summary.json")
        stage = {key: metrics(value["frozen"]) for key, value in complete["stages"].items()}
        final = complete["final_probe"]
        diagnostics = {key: metrics(value["frozen"])
                       for key, value in complete["diagnostics"].items()}
        fresh = train["continuation"]
        result[arm] = dict(
            new_training_steps_per_body=train["new_steps"],
            new_training_seconds_per_body=train["new_steps"]*.01,
            new_training_body_seconds=train["new_steps"]*.01*train["bodies"],
            new_training=fresh, vision=train["vision"],
            stage_probes=stage,
            independent_final_probe=metrics(final["frozen"]),
            independent_final_scenes=final["scenes"],
            diagnostics=diagnostics,
            attention_input_dependence=final["input_dependence"],
            frozen_checks_all_passed=all(item["learned_parameters_unchanged"] for item in
                list(complete["stages"].values())+[final]+list(complete["diagnostics"].values())),
            practice_process=practice_intervals(root/arm, train["origin_step"],
                                               train["new_steps"], train["bodies"]),
        )
        exploration = root/arm/"exploring_frozen_probe"/"COMPLETE.json"
        if exploration.exists():
            observed = read(exploration)
            result[arm]["exploratory_stochastic_probe"] = dict(
                metrics=metrics(observed["frozen"]),
                scenes=observed["scenes"],
                learned_parameters_unchanged=observed["learned_parameters_unchanged"],
                identity=observed["identity"],
                note="Added after the baseline midpoint; supplements the planned greedy probes.")
        result[arm]["final_probe_vision"] = final["frozen"]["vision"]
        result[arm]["diagnostic_vision"] = {
            key: value["frozen"]["vision"] for key, value in complete["diagnostics"].items()}
        origin = train["origin_step"]
        curve = train["curve"]
        points = [point for point in curve if point["step"] >= origin]
        steps = np.array([point["step"] for point in points])
        seconds = (steps-origin)*.01
        color = colors[arm]
        axes[0, 0].plot(seconds, [point["contact"]-train["contact"]+fresh["contact"]
                                  for point in points], label=arm, color=color)
        collisions = np.array([point["collision"] for point in points])
        frames = np.array([point["frames"] for point in points])
        # Nonoverlapping 1024-step windows avoid overstating noisy 64-step blocks.
        sampled = np.arange(0, len(points), 16)
        if len(sampled) > 1:
            axes[0, 1].plot(seconds[sampled[1:]],
                           np.diff(collisions[sampled])/np.maximum(np.diff(frames[sampled]), 1),
                           label=arm, color=color)
        axes[1, 0].plot(seconds, [point["prequential_loss"] for point in points],
                       alpha=.65, color=color, label=arm)
        stage_keys = sorted(int(key) for key in stage)
        axes[1, 1].plot(np.array(stage_keys)*.01,
                       [stage[str(key)]["contact"] for key in stage_keys],
                       marker="o", color=color, label=arm)
    for axis, title, ylabel in zip(axes.flat,
            ("New practice: cumulative contacts", "Practice: collision frame fraction",
             "Loss on fresh experienced outcomes", "Frozen common scenes: contacts"),
            ("Contacts (8 bodies)", "Collision fraction", "Smooth L1 loss",
             "Contacts / 5.12 seconds per body")):
        axis.set(title=title, xlabel="New training seconds per body", ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("V23: extended practice and active vision\n"
                   "Two exploration arms, one source state; no causal or convergence claim",
                   fontsize=13)
    figure.savefig(root/"learning_process.png", dpi=160)
    plt.close(figure)
    (root/"measurements.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
