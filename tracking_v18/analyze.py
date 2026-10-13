"""Learning-path summaries and figures; run only after both arms finish."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracking_v17.run import dump
from tracking_v17.world import DT

ROOT = Path(__file__).resolve().parent


def phase(path, start, end):
    fields = ("contact", "safe_contact", "collision", "impact", "visible", "locked")
    totals = {k: 0 for k in fields}
    steps = attacked_steps = attacked_impacts = 0
    errors, utility, gain, alignment = [], [], [], []
    for chunk in sorted(path.glob("transitions_*.npz")):
        with np.load(chunk) as d:
            chunk_start = int(d["start_step"])
            if not start <= chunk_start < end:
                continue
            steps += d["actions"].size
            for key in fields:
                totals[key] += int(d[key].sum())
            attacked = d["combo"][:, :, 2] != 0
            attacked_steps += int(attacked.sum())
            attacked_impacts += int((d["impact"] & attacked).sum())
            errors.append(float(d["pred_error"].mean()))
            utility.append(float(d["utility_abs"].mean()))
            gain.append(float(d["gain_abs"].mean()))
            alignment.append(float(d["gain_alignment_abs"].mean()))
    return dict(start=start, end=end, experiences=steps, totals=totals,
        contact_per_minute=totals["contact"] * 60 / (steps * DT),
        safe_contact_fraction=totals["safe_contact"] / max(totals["contact"], 1),
        collision_rate=totals["collision"] / steps,
        attacked_impact_rate=attacked_impacts / max(attacked_steps, 1),
        visible_lock=totals["locked"] / max(totals["visible"], 1),
        preupdate_mse=float(np.mean(errors)), mean_abs_utility=float(np.mean(utility)),
        mean_abs_gain=float(np.mean(gain)), mean_alignment_change=float(np.mean(alignment)))


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    assert (output / "COMPLETE.json").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "AUDIT.json").read_text(encoding="utf-8"))
    milestones = [0, *config["milestones"]]
    results, phases = {}, {}
    for arm in config["arms"]:
        path = output / arm
        results[arm] = [json.loads((path / f"result_{step}.json").read_text(encoding="utf-8"))
                        for step in milestones]
        phases[arm] = [phase(path, start, end)
                       for start, end in zip(milestones[:-1], milestones[1:])]
    metrics = ["contact_per_minute", "safe_contact_fraction", "collision_rate",
               "attacked_impact_rate", "visible_lock", "prediction_mse"]
    initial = {arm: {key: results[arm][0]["probes"]["frozen"][key] for key in metrics}
               for arm in config["arms"]}
    assert initial["legacy"] == initial["aligned"]
    for lfile in sorted((output / "legacy" / "probe_0_frozen").glob("transitions_*.npz")):
        with np.load(lfile) as left, np.load(output / "aligned" / "probe_0_frozen" / lfile.name) as right:
            for key in ("actions", "groups", "reward", "locked", "pred_error"):
                assert np.array_equal(left[key], right[key]), (lfile, key)
    dump(output / "ANALYSIS.json", dict(manifest=manifest, results=results, phases=phases,
         initial_frozen_transitions_identical=True,
         all_audited_rows=audit["all_rows"], frame_digests_checked=audit["frame_digests_checked"]))
    colors = {"legacy": "#bc4560", "aligned": "#168177"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    labels = ("Contacts / body-minute", "Safe fraction of contacts",
              "Collision steps (%)", "Impact steps in attack scenes (%)",
              "Visible attention hits (%)", "Next RGB prediction MSE")
    for axis, metric, label in zip(axes.flat, metrics, labels):
        scale = 100 if metric.endswith("_rate") or metric == "visible_lock" else 1
        for arm in config["arms"]:
            for kind, style in (("frozen", "-"), ("adaptive", "--")):
                y = [r["probes"][kind][metric] * scale for r in results[arm]]
                axis.plot(milestones, y, style, marker="o", markersize=4,
                          color=colors[arm], label=f"{arm} / {kind}")
        axis.set_title(label, fontsize=11)
        axis.set_xlabel("Additional practice steps per body")
        axis.set_xticks(milestones)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("V18: one coordinate repair; same warm start and practice budget", fontsize=13)
    fig.savefig(ROOT / "PROCESS_CURVES.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    for axis, metric, title in zip(axes, ("contact_per_minute", "collision_rate", "preupdate_mse"),
                                   ("Practice contacts / body-minute", "Practice collision rate", "Preupdate next RGB MSE")):
        for arm in config["arms"]:
            curves = results[arm][-1]["curves"]
            x = np.array([c["step"] for c in curves])
            if metric == "contact_per_minute":
                y = np.array([c["contact_count"] * 60 /
                             (config["rollout"] * config["worlds"] * DT) for c in curves])
            else:
                y = np.array([c[metric] for c in curves])
            window = 8
            axis.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, "valid"),
                      color=colors[arm], label=arm)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Additional practice steps per body")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(ROOT / "PRACTICE_CURVES.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(analyzed=True, rows=audit["all_rows"],
                          phases=phases)), flush=True)


if __name__ == "__main__":
    main()
