"""Aggregate whole-run results, costs and matched observed compute prefixes."""

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import ROOT, OPS, TOLERANCE
from run import dump


def summarize(stage):
    folder = ROOT / stage
    manifest = json.loads((folder / "manifest.json").read_text())
    config = manifest["config"]
    rows = []
    for run in config["runs"]:
        target = folder / f'g{run["groups"]}_s{run["seed"]}_w{run["world_seed"]}_{run["mode"]}'
        if not (target / "result.json").exists():
            continue
        result = json.loads((target / "result.json").read_text())
        if "composition" not in result:
            continue
        result["path"] = str(target)
        with np.load(target / "transitions.npz") as data:
            result["late_operations"] = {name: float((data["op"][-128:] == i).mean()) for i, name in enumerate(OPS)}
            result["entropy_first"] = float(data["entropy"][:8].mean())
            result["entropy_last"] = float(data["entropy"][-8:].mean())
        rows.append(result)
    groups = []
    for size in sorted(set(x["group_reserve"] for x in rows)):
        for mode in ("independent", "immediate", "delayed"):
            selected = [x for x in rows if x["group_reserve"] == size and x["mode"] == mode]
            if not selected:
                continue
            def aggregate(values):
                a = np.array(values)
                return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
                        "std": float(a.std(ddof=1)) if len(a) > 1 else None}
            groups.append(dict(
                groups=size, mode=mode, runs=len(selected),
                total_parameters=selected[0]["total_parameters"],
                online=aggregate([x["success"] for x in selected]),
                online_last=aggregate([x["last_success"] for x in selected]),
                seen=aggregate([x["seen"]["success"] for x in selected]),
                composition=aggregate([x["composition"]["success"] for x in selected]),
                state_mse=aggregate([x["last_mse"] for x in selected]),
                estimated_macs=aggregate([x["estimated_total_macs"] for x in selected]),
                active_parameters=aggregate([x["mean_active_predictor_parameters"] for x in selected]),
                retrieved=aggregate([x["retrievals"] for x in selected]),
                replayed=aggregate([x["totals"]["replay_presentations"] for x in selected]),
                accessible=aggregate([x["final_accessible_groups"] for x in selected]),
                late_operations={name: float(np.mean([x["late_operations"][name] for x in selected])) for name in OPS},
                entropy_last=aggregate([x["entropy_last"] for x in selected]),
            ))
    matched = []
    for size, seed, world in sorted(set(
        (r["group_reserve"], r["model_seed"], r["environment_seed"]) for r in rows
    )):
        selected = [r for r in rows if (r["group_reserve"], r["model_seed"], r["environment_seed"]) == (size, seed, world)]
        if len(selected) != 3:
            continue
        cap = min(r["curves"][-1]["estimated_macs"] for r in selected)
        for result in selected:
            with np.load(Path(result["path"]) / "transitions.npz") as source:
                used = source["cumulative_estimated_macs"]
                end = int(np.searchsorted(used, cap, side="right"))
                regret = source["regret"][max(0, end - 256):end]
                matched.append(dict(groups=size, seed=seed, world_seed=world, mode=result["mode"],
                                    common_estimated_mac_cap=cap, end_step=end,
                                    success=float((regret <= TOLERANCE).mean()),
                                    actual_estimated_macs=int(used[end - 1])))
    summary = dict(stage=stage, completed_runs=len(rows), expected_runs=len(config["runs"]),
                   groups=groups, matched_estimated_mac_prefixes=matched)
    dump(folder / "SUMMARY.json", summary)
    if not groups:
        return
    fig, axes = plt.subplots(1, len(set(r["group_reserve"] for r in rows)), figsize=(11, 4), squeeze=False)
    colors = dict(independent="#555555", immediate="#cc5544", delayed="#228866")
    for ax, size in zip(axes[0], sorted(set(r["group_reserve"] for r in rows))):
        for mode in ("independent", "immediate", "delayed"):
            selected = [r for r in rows if r["group_reserve"] == size and r["mode"] == mode]
            if not selected:
                continue
            x = np.array([p["presentations"] for p in selected[0]["curves"]])
            ys = np.array([[p["success"] for p in r["curves"]] for r in selected])
            ax.plot(x, ys.mean(0), label=mode, color=colors[mode])
            ax.fill_between(x, ys.min(0), ys.max(0), color=colors[mode], alpha=0.12)
        ax.set(title=f"Reserve {size} groups", xlabel="New physical transitions", ylabel="Last 128-step success")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=140)
    plt.close(fig)
    for row in groups:
        print(json.dumps({k: row[k] for k in ("groups", "mode", "runs", "online", "online_last", "seen", "composition", "late_operations")}))


if __name__ == "__main__":
    summarize(sys.argv[1])
