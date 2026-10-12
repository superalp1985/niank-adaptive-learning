"""Frozen checkpoints only: does a tiny remembered cue affect physical prediction?"""

import json
from collections import defaultdict

import numpy as np
import torch

from experiment import ROOT, MODES, RetentionModel, archive, generate, physics


@torch.no_grad()
def predictions(model, dataset, mode):
    tensors = dataset.tensors()
    output = model(tensors, mode, replay=False)
    return {
        "velocity": output["states"][..., 1].cpu().numpy(),
        "action": output["action"].argmax(-1).cpu().numpy(),
        "fine_retained": (
            output["indices"] == tensors["cue_times"][:, 1:2]
        ).any(-1).cpu().numpy(),
    }


def construct(base, risk, flip):
    episode = base.take(slice(None))
    episode.images = episode.images.copy()
    episode.context = np.full_like(episode.context, risk)
    count = len(base)
    rows = np.arange(count)
    if flip:
        episode.images[rows, base.cue_times[:, 1], 1] *= -1
    coarse = base.images[rows, base.cue_times[:, 0], 0, 1, 1] > 0
    fine = episode.images[rows, base.cue_times[:, 1], 1, 2, 2] > 0
    momentum = np.where(coarse, 0.9, 0.25).astype(np.float32)
    force = np.where(fine, 0.18, -0.18).astype(np.float32)
    episode.states, episode.contacts, episode.costs = physics(
        episode.body, momentum, force, episode.context[:, 0]
    )
    return episode


def main():
    torch.set_num_threads(2)
    output = ROOT / "confirmation"
    assert (output / "COMPLETE.json").exists()
    frozen = json.loads((output / "final_results.json").read_text())
    base = generate(256, 810000, delay=26, scenario="detail_intervention")
    for scenario, seed, delay in (
        ("ordinary", 710000, 18), ("long_delay", 720000, 34),
        ("left_layout", 730000, 18), ("texture_shift", 740000, 34),
        ("development", 610000, 18),
    ):
        path = output / f"archive_{scenario}.sqlite"
        if not path.exists():
            archive(path, generate(256 if scenario == "development" else 512, seed, delay=delay, scenario=scenario))
    records = []
    for cohort in frozen:
        location = output / f"u{cohort['unique']}_r{cohort['repeats']}_s{cohort['seed']}"
        for mode in MODES:
            model = RetentionModel().cuda().eval()
            model.load_state_dict(torch.load(location / f"{mode}.pt", map_location="cuda", weights_only=True))
            metrics = {}
            for risk in (0.0, 1.0):
                original = construct(base, risk, False)
                flipped = construct(base, risk, True)
                first, second = predictions(model, original, mode), predictions(model, flipped, mode)
                truth = flipped.states[..., 1] - original.states[..., 1]
                effect = second["velocity"] - first["velocity"]
                align = (effect * truth).sum() / max(1e-8, float((truth**2).sum()))
                metrics[str(int(risk))] = {
                    "predicted_velocity_effect_mae": float(np.abs(effect).mean()),
                    "true_velocity_effect_mae": float(np.abs(truth).mean()),
                    "effect_mse": float(np.mean((effect - truth) ** 2)),
                    "effect_gain": float(align) if risk else None,
                    "fine_retained": float((first["fine_retained"].mean() + second["fine_retained"].mean()) / 2),
                }
            records.append({
                "unique": cohort["unique"], "repeats": cohort["repeats"],
                "seed": cohort["seed"], "mode": mode, "contexts": metrics,
            })
    (ROOT / "DETAIL_INTERVENTION.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    groups = defaultdict(list)
    for record in records:
        groups[(record["unique"], record["repeats"], record["mode"])].append(record)
    summary = []
    for (unique, repeats, mode), rows in groups.items():
        summary.append({
            "unique": unique, "repeats": repeats, "mode": mode,
            "risk_effect_gain": float(np.mean([row["contexts"]["1"]["effect_gain"] for row in rows])),
            "risk_effect_mse": float(np.mean([row["contexts"]["1"]["effect_mse"] for row in rows])),
            "safe_spurious_velocity_effect": float(np.mean([row["contexts"]["0"]["predicted_velocity_effect_mae"] for row in rows])),
            "risk_detail_retention": float(np.mean([row["contexts"]["1"]["fine_retained"] for row in rows])),
        })
    (ROOT / "DETAIL_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
