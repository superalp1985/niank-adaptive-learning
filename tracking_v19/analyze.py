"""Summarize practice and reversible-access diagnostics without seed selection."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracking_v17.run import dump
from tracking_v17.world import DT, EYES

ROOT = Path(__file__).resolve().parent


def phase(path, start, end):
    fields = ("contact", "safe_contact", "collision", "impact", "visible", "locked")
    totals = {key: 0 for key in fields}
    averages = {key: [] for key in ("pred_error", "omitted_error", "retention_gate",
        "utility_abs", "deviation_mean", "gain_mean", "gain_std",
        "prediction_disagreement", "promoted")}
    count = attacked_steps = attacked_impacts = 0
    # These are access opportunities, not evidence of behavioral causal recovery.
    eligible = {"positive": 0, "nonpositive": 0}
    accessed = {"positive": 0, "nonpositive": 0}
    previous = None
    all_probe_count = low_access_positive = 0
    for chunk in sorted(path.glob("transitions_*.npz")):
        with np.load(chunk) as d:
            chunk_start = int(d["start_step"])
            if not start <= chunk_start < end:
                continue
            count += d["actions"].size
            for key in fields:
                totals[key] += int(d[key].sum())
            attacked = d["combo"][:, :, 2] != 0
            attacked_steps += int(attacked.sum())
            attacked_impacts += int((attacked & d["impact"]).sum())
            for key in averages:
                averages[key].append(float(d[key].mean()))
            all_probe_count += d["gain_mean"].size
            low_access_positive += int(((d["probe_selected_fraction"] < .5) &
                                        (d["gain_mean"] > 0)).sum())
            selections = np.unpackbits(d["selected_bits"], axis=2).astype(bool)
            for t in range(len(selections)):
                selected = selections[t]
                if previous is not None:
                    prior_eligible, indices, good, old_done = previous
                    now = np.take_along_axis(selected, indices, 1)
                    for name, condition in (("positive", good), ("nonpositive", ~good)):
                        valid = prior_eligible & condition[:, None] & ~old_done[:, None]
                        eligible[name] += int(valid.sum())
                        accessed[name] += int((valid & now).sum())
                indices = (d["probe_start"][t, :, None] + np.arange(16)) % EYES
                not_selected = ~np.take_along_axis(selected, indices, 1)
                next_indices = (indices - d["rotation_shift"][t, :, None]) % EYES
                previous = (not_selected, next_indices, d["gain_mean"][t] > 0,
                            d["done"][t])
    return dict(start=start, end=end, experiences=count, totals=totals,
        contact_per_minute=totals["contact"] * 60 / (count * DT),
        safe_contact_per_minute=totals["safe_contact"] * 60 / (count * DT),
        safe_contact_fraction=totals["safe_contact"] / max(totals["contact"], 1),
        collision_rate=totals["collision"] / count,
        attacked_impact_rate=attacked_impacts / max(attacked_steps, 1),
        visible_lock=totals["locked"] / max(totals["visible"], 1),
        means={k: float(np.mean(v)) for k, v in averages.items()},
        low_access_positive_probe_fraction=low_access_positive / max(all_probe_count, 1),
        reconsideration={k: dict(eligible_eyes=eligible[k], next_access=accessed[k],
            next_access_fraction=accessed[k] / max(eligible[k], 1)) for k in eligible})


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    assert (output / "COMPLETE.json").exists()
    audit = json.loads((output / "AUDIT.json").read_text(encoding="utf-8"))
    milestones = [0, *config["milestones"]]
    results, phases = {}, {}
    for arm in config["arms"]:
        path = output / arm
        results[arm] = [json.loads((path / f"result_{s}.json").read_text(encoding="utf-8"))
                        for s in milestones]
        phases[arm] = [phase(path, lo, hi) for lo, hi in zip(milestones[:-1], milestones[1:])]
    metrics = ["contact_per_minute", "safe_contact_fraction", "collision_rate",
               "attacked_impact_rate", "visible_lock", "prediction_mse"]
    first, second = config["arms"]
    for key in metrics:
        assert results[first][0]["probes"]["frozen"][key] == results[second][0]["probes"]["frozen"][key]
    scene_deltas = {}
    for kind in ("frozen", "adaptive"):
        left, right = (results[arm][-1]["probes"][kind]["combo_stats"] for arm in config["arms"])
        scene_deltas[kind] = {scene: {key: right[scene][key] - left[scene][key]
            for key in ("contact_per_minute", "collision_rate", "impact_rate", "visible_lock")}
            for scene in left}
    result = dict(results=results, phases=phases, final_scene_deltas=scene_deltas,
        all_audited_rows=audit["all_rows"],
        limitation="One shared initialization; inspected development scenarios; no confirmatory significance.")
    dump(output / "ANALYSIS.json", result)
    colors = {first: "#a63a59", second: "#087f8c"}
    labels = ("Contacts / body-minute", "Safe fraction of contacts", "Collision steps (%)",
              "Impact steps in attack scenes (%)", "Visible attention hits (%)", "Next RGB MSE")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, metric, label in zip(axes.flat, metrics, labels):
        scale = 100 if metric.endswith("_rate") or metric == "visible_lock" else 1
        for arm in config["arms"]:
            for kind, style in (("adaptive", "-"), ("frozen", "--")):
                y = [v["probes"][kind][metric] * scale for v in results[arm]]
                axis.plot(milestones, y, style, marker="o", markersize=4,
                          color=colors[arm], label=f"{arm} / {kind}")
        axis.set_title(label, fontsize=11)
        axis.set_xlabel("Additional practice steps per body")
        axis.set_xticks(milestones)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("V19: matched access budget, revisable local evidence", fontsize=13)
    fig.savefig(ROOT / "PROCESS_CURVES.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    for axis, metric, title in zip(axes, ("contact_count", "collision_rate", "preupdate_mse"),
        ("Practice contacts / body-minute", "Practice collision rate", "Preupdate next RGB MSE")):
        for arm in config["arms"]:
            curves = results[arm][-1]["curves"]
            x = np.array([c["step"] for c in curves])
            y = np.array([c[metric] for c in curves])
            if metric == "contact_count":
                y = y * 60 / (config["rollout"] * config["worlds"] * DT)
            window = 8
            axis.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, "valid"),
                      color=colors[arm], label=arm)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Additional practice steps per body")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(ROOT / "PRACTICE_CURVES.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(phases=phases, rows=audit["all_rows"])), flush=True)


if __name__ == "__main__":
    main()
