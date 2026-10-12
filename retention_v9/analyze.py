"""Aggregate complete repeated-practice runs and display the time course."""

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


def summarize():
    output = ROOT / "confirmation"
    if not (output / "COMPLETE.json").exists():
        raise RuntimeError("Confirmation is incomplete")
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    examples = []
    for result in results:
        grouped[(result["steps"], result["mode"])].append(result)
        target = output / f"t{result['steps']}_s{result['seed']}_{result['mode']}"
        with np.load(target / "transitions.npz") as data:
            returns = data["fine_returned"].sum(0)
            examples.append({
                "mode": result["mode"], "steps": result["steps"], "seed": result["seed"],
                "worlds_with_two_or_more_fine_returns": int((returns >= 2).sum()),
                "max_fine_returns_in_one_world": int(returns.max()),
                "mean_fine_active_fraction": float(data["fine_active"].mean()),
                "mean_evidence_strength": float(data["strength"].mean()),
                "mean_replay_loss_reduction": float(data["replay_improvement"][data["retrieval"]].mean())
                if data["retrieval"].any() else None,
            })
    summary = []
    for (steps, mode), rows in grouped.items():
        fields = {
            "prequential_success": [r["prequential"]["success"] for r in rows],
            "first_success": [r["first_32"]["success"] for r in rows],
            "last_success": [r["last_32"]["success"] for r in rows],
            "regret": [r["prequential"]["regret"] for r in rows],
            "actual_cost": [r["prequential"]["actual_cost"] for r in rows],
            "state_mse": [r["prequential"]["state_mse"] for r in rows],
            "retrieval_fraction": [r["prequential"]["retrieval_fraction"] for r in rows],
            "archive_reads": [r["prequential"]["archive_records_read"] for r in rows],
            "fine_drops": [r["prequential"]["fine_drop_events"] for r in rows],
            "fine_returns": [r["prequential"]["fine_return_events"] for r in rows],
            "active_references": [r["prequential"]["mean_active_references"] for r in rows],
            "probe_initial_success": [r["initial_probe"]["success"] for r in rows],
            "probe_final_success": [r["final_probe"]["success"] for r in rows],
            "probe_final_left": [r["final_probe"]["left_success"] for r in rows],
            "probe_final_right": [r["final_probe"]["right_success"] for r in rows],
        }
        summary.append({
            "steps": steps, "mode": mode, "seeds": [r["seed"] for r in rows],
            "worlds_per_seed": rows[0]["worlds"],
            "means": {key: float(np.mean(value)) for key, value in fields.items()},
            "seed_values": fields,
        })
    (ROOT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "REVISIT_SUMMARY.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return results, summary


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    colors = {"fixed": "#52525b", "periodic": "#2563eb",
              "recurrent": "#dc2626", "consolidated": "#059669"}
    rows = [row for row in results if row["steps"] == 512]
    for mode, color in colors.items():
        selected = [row for row in rows if row["mode"] == mode]
        phases = [row["phases"] for row in selected]
        x = [phase["start"] + 16 for phase in phases[0]]
        for ax, field, label in zip(axes.flat,
                                    ("success", "state_mse", "retrieval_fraction", "mean_active_references"),
                                    ("Pre-feedback action success", "Chosen-state prediction MSE",
                                     "Post-feedback retrieval fraction", "Live raw references")):
            values = np.asarray([[phase[field] for phase in run] for run in phases])
            ax.plot(x, values.mean(0), color=color, label=mode)
            ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.10)
            ax.set_title(label)
            ax.set_xlabel("Practice step (one action per world)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Three initialization/world seeds; means and min/max, not confidence intervals")
    fig.tight_layout()
    fig.savefig(ROOT / "practice_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results, summary = summarize()
    plot(results)
    print(json.dumps(summary))
