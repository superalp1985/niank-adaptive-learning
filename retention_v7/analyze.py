"""Summarize every retained run without treating seed x episode as independent."""

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

def collect():
    records = []
    for stage in ("development", "development_volume", "confirmation"):
        path = ROOT / stage / "results.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                row["stage"] = stage
                records.append(row)
    groups = defaultdict(list)
    for row in records:
        groups[(row["stage"], row["unique"], row["cycles"], row["mode"])].append(row)
    summary = []
    for (stage, unique, cycles, mode), rows in groups.items():
        key = "final" if stage == "confirmation" else "development"
        features = {
            "success": [r[key]["success"] for r in rows],
            "regret": [r[key]["regret"] for r in rows],
            "state_mse": [r[key]["state_mse"] for r in rows],
            "active_count": [r[key]["active_count"] for r in rows],
            "risk_gain": [r["detail"]["1"]["effect_gain"] for r in rows],
            "safe_spurious": [r["detail"]["0"]["predicted_effect_mae"] for r in rows],
            "risk_retention": [r["detail"]["1"]["fine_retained"] for r in rows],
            "safe_retention": [r["detail"]["0"]["fine_retained"] for r in rows],
            "recovery_gain": [r["recovery"]["recovered_success"] - r["recovery"]["stale_success"] for r in rows],
            "stale_success": [r["recovery"]["stale_success"] for r in rows],
            "recovered_success": [r["recovery"]["recovered_success"] for r in rows],
        }
        summary.append({
            "stage": stage, "unique": unique, "cycles": cycles, "mode": mode,
            "presentations": unique * cycles * 2,
            "seeds": [r["seed"] for r in rows],
            "means": {key: float(np.mean(values)) for key, values in features.items()},
            "seed_ranges": {key: [float(min(values)), float(max(values))] for key, values in features.items()},
            "parameters": rows[0]["parameters"],
        })
    (ROOT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot(summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    colors = {"ordinary_plain": "#52525b", "ordinary_event": "#dc2626",
              "paired_plain": "#2563eb", "paired_event": "#059669"}
    labels = []
    for row in summary:
        label = f"{row['stage']} {row['unique']} x {row['cycles']}"
        if label not in labels:
            labels.append(label)
    for ax, key, title in zip(axes.flat,
                              ("success", "risk_gain", "safe_spurious", "recovery_gain"),
                              ("Physical action success", "Risky fine-cue response gain (ideal 1)",
                               "Safe spurious velocity response (ideal 0)", "SQLite recovery action gain")):
        for mode, color in colors.items():
            rows = [r for r in summary if r["mode"] == mode]
            x = [labels.index(f"{r['stage']} {r['unique']} x {r['cycles']}") for r in rows]
            y = [r["means"][key] for r in rows]
            ax.plot(x, y, "o-", color=color, label=mode)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "outcomes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    summary = collect()
    plot(summary)
    print(json.dumps(summary))
