"""Check whether selection itself encodes a cue even when its image is absent."""

import json

import numpy as np
import torch

from experiment import ROOT, OutcomeRetentionModel, generate, metrics
from retention_v6.experiment import with_context


def main():
    output = ROOT / "confirmation"
    if not (output / "COMPLETE.json").exists():
        raise RuntimeError("Wait for all frozen training runs")
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    base = generate(256, manifest["data"]["detail"]["ids"][0], delay=26, scenario="detail")
    torch.set_num_threads(2)
    records = []
    for result in results:
        if result["unique"] != 2048 or result["cycles"] not in (8, 32):
            continue
        mode = result["mode"]
        model = OutcomeRetentionModel().cuda().eval()
        target = output / f"u2048_c{result['cycles']}_s{result['seed']}_{mode}"
        model.load_state_dict(torch.load(target / "model.pt", map_location="cuda", weights_only=True))
        for context in (0.0, 1.0):
            original, flipped = with_context(base, context), with_context(base, context, True)
            _, a = metrics(model, mode, original)
            _, b = metrics(model, mode, flipped)
            override = (
                torch.from_numpy(a["active_indices"]).cuda(),
                torch.from_numpy(a["active_mask"]).cuda(),
            )
            _, fixed = metrics(model, mode, flipped, active_override=override)
            effect = b["velocity"] - a["velocity"]
            fixed_effect = fixed["velocity"] - a["velocity"]
            truth = flipped.states[..., 1] - original.states[..., 1]
            masks = {
                "both_fine_active": a["fine_retained"] & b["fine_retained"],
                "only_one_fine_active": a["fine_retained"] ^ b["fine_retained"],
                "neither_fine_active": ~a["fine_retained"] & ~b["fine_retained"],
            }
            subsets = {}
            for name, mask in masks.items():
                subsets[name] = {
                    "episodes": int(mask.sum()),
                    "predicted_effect_mae": float(np.abs(effect[mask]).mean()) if mask.any() else None,
                    "effect_gain": float((effect[mask] * truth[mask]).sum() / (truth[mask]**2).sum())
                    if mask.any() and context else None,
                    "fixed_directory_effect_mae": float(np.abs(fixed_effect[mask]).mean()) if mask.any() else None,
                }
            active_sets_a = [set(indices[mask].tolist()) for indices, mask in zip(a["active_indices"], a["active_mask"])]
            active_sets_b = [set(indices[mask].tolist()) for indices, mask in zip(b["active_indices"], b["active_mask"])]
            changes = np.asarray([x != y for x, y in zip(active_sets_a, active_sets_b)])
            records.append({
                "mode": mode, "seed": result["seed"], "cycles": result["cycles"],
                "context": context, "active_directory_change_fraction": float(changes.mean()),
                "adaptive_response_gain": float((effect * truth).sum() / (truth**2).sum()) if context else None,
                "fixed_directory_response_gain": float((fixed_effect * truth).sum() / (truth**2).sum()) if context else None,
                "fixed_directory_spurious_effect": float(np.abs(fixed_effect).mean()) if not context else None,
                "fine_in_original_directory": float(a["fine_retained"].mean()),
                "subsets": subsets,
            })
    (ROOT / "REFERENCE_CODE_PROBE.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoints": len(records) // 2, "records": len(records)}))


if __name__ == "__main__":
    main()
