"""Full seed summaries and paired outcomes; no best-seed selection."""

from collections import defaultdict
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize():
    records = []
    for stage in ("development", "confirmation"):
        path = ROOT / stage / "results.json"
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                row["stage"] = stage
                records.append(row)
    groups = defaultdict(list)
    for row in records:
        groups[(row["stage"], row["unique"], row["cycles"], row["mode"])].append(row)
    summary = []
    for (stage, unique, cycles, mode), rows in groups.items():
        metric_set = "final" if stage == "confirmation" else "development"
        features = {
            "success": [r[metric_set]["success"] for r in rows],
            "regret": [r[metric_set]["regret"] for r in rows],
            "state_mse": [r[metric_set]["state_mse"] for r in rows],
            "active_count": [r[metric_set]["active_count"] for r in rows],
            "risk_gain": [r["detail"]["1"]["effect_gain"] for r in rows],
            "safe_spurious": [r["detail"]["0"]["predicted_effect_mae"] for r in rows],
            "risk_retention": [r["detail"]["1"]["fine_retained"] for r in rows],
            "safe_retention": [r["detail"]["0"]["fine_retained"] for r in rows],
            "stale_success": [r["recovery"]["stale_success"] for r in rows],
            "recovered_success": [r["recovery"]["recovered_success"] for r in rows],
            "recovery_gain": [r["recovery"]["recovered_success"] - r["recovery"]["stale_success"] for r in rows],
            "forgotten_episodes": [r["recovery"]["forgotten_episodes"] for r in rows],
            "forgotten_recovery_gain": [
                None if r["recovery"]["forgotten_stale_success"] is None else
                r["recovery"]["forgotten_recovered_success"] - r["recovery"]["forgotten_stale_success"]
                for r in rows
            ],
        }
        if stage == "confirmation":
            for label in ("left_layout", "texture_shift"):
                features[f"{label}_success"] = [r[label]["success"] for r in rows]
        summary.append({
            "stage": stage, "unique": unique, "cycles": cycles, "mode": mode,
            "presentations": unique * cycles * 4,
            "seeds": [r["seed"] for r in rows],
            "means": {key: mean_or_none(values) for key, values in features.items()},
            "seed_values": features,
            "parameters": rows[0]["parameters"],
            "mean_training_seconds": float(np.mean([r["seconds"] for r in rows])),
        })
    paired = []
    for stage in ("development", "confirmation"):
        pairs = defaultdict(dict)
        for row in records:
            if row["stage"] == stage:
                pairs[(row["unique"], row["cycles"], row["seed"])][row["mode"]] = row
        for (unique, cycles, seed), rows in pairs.items():
            if len(rows) != 2:
                continue
            point, delta = rows["quartet_point"], rows["quartet_delta"]
            metric_set = "final" if stage == "confirmation" else "development"
            paired.append({
                "stage": stage, "unique": unique, "cycles": cycles, "seed": seed,
                "success_gain": delta[metric_set]["success"] - point[metric_set]["success"],
                "risk_gain_difference": delta["detail"]["1"]["effect_gain"] - point["detail"]["1"]["effect_gain"],
                "safe_spurious_difference": delta["detail"]["0"]["predicted_effect_mae"] - point["detail"]["0"]["predicted_effect_mae"],
                "state_mse_difference": delta[metric_set]["state_mse"] - point[metric_set]["state_mse"],
            })
    (ROOT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "PAIRED.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    return summary, paired


def plot(summary):
    rows = [r for r in summary if r["stage"] == "confirmation"]
    if not rows:
        rows = summary
    conditions = sorted(set((r["unique"], r["cycles"]) for r in rows),
                        key=lambda pair: (pair[0] * pair[1], pair[0]))
    labels = [f"{u} histories x {c} cycles\n{u*c*4:,} presentations" for u, c in conditions]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    features = ("success", "risk_gain", "safe_spurious", "risk_retention", "safe_retention", "recovery_gain")
    titles = ("Physical action success", "Risky fine-cue response (ideal 1)",
              "Safe spurious response (ideal 0)", "Risky fine-reference retention",
              "Safe fine-reference retention", "SQLite recovery gain")
    for ax, key, title in zip(axes.flat, features, titles):
        for mode, color in (("quartet_point", "#2563eb"), ("quartet_delta", "#dc2626")):
            selected = [r for r in rows if r["mode"] == mode]
            x = [conditions.index((r["unique"], r["cycles"])) for r in selected]
            y = [r["means"][key] for r in selected]
            low = [r["means"][key] - min(r["seed_values"][key]) for r in selected]
            high = [max(r["seed_values"][key]) - r["means"][key] for r in selected]
            ax.errorbar(x, y, yerr=[low, high], fmt="o-", capsize=3, label=mode, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=7)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Means and min/max across optimization seeds; shared evaluation episodes", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "outcomes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    summary, paired = summarize()
    plot(summary)
    print(json.dumps({"summary": summary, "paired": paired}))
