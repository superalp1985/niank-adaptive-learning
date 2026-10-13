"""Measure gate changes on factual traces; shadow gates are not policy rollouts."""

import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from tracking_v17.run import dump
from tracking_v17.world import EYES

ROOT = Path(__file__).resolve().parent


def one_arm(path, milestones, arm):
    worlds = json.loads((path / "summary.json").read_text(encoding="utf-8"))["worlds"]
    utility = np.zeros((worlds, EYES), np.float32)
    shadow = np.zeros_like(utility)
    age = np.zeros_like(utility)
    previous = np.zeros((worlds, EYES, 3), np.float32)
    previous_segment = np.full(worlds, -1)
    phases = [dict(start=start, end=end, sector_presentations=0, gate_sum=0.,
                   fixed_gate_sum=0., delta_sum=0., shadow_delta_sum=0.,
                   delta_gt_001=0, delta_gt_01=0, shadow_delta_gt_001=0,
                   positive_gain=0, negative_gain=0, scenes={})
              for start, end in zip([0, *milestones[:-1]], milestones)]
    db = sqlite3.connect(f"file:{path / 'events.sqlite'}?mode=ro", uri=True)
    try:
        for chunk in sorted(path.glob("transitions_*.npz")):
            with np.load(chunk) as d:
                start = int(d["start_step"])
                phase = next(p for p in phases if p["start"] <= start < p["end"])
                length = len(d["actions"])
                rows = db.execute("SELECT rgb,proprio,next_proprio,retention_gain "
                                  "FROM events WHERE time>=? AND time<? ORDER BY time,body",
                                  (start, start + length)).fetchall()
                rgb = np.frombuffer(b"".join(zlib.decompress(r[0]) for r in rows),
                                    np.uint8).reshape(length, worlds, EYES, 3)
                prop = np.frombuffer(b"".join(r[1] for r in rows), np.float32).reshape(length, worlds, 6)
                after_prop = np.frombuffer(b"".join(r[2] for r in rows), np.float32).reshape(length, worlds, 6)
                gain = np.frombuffer(b"".join(zlib.decompress(r[3]) for r in rows),
                                     np.float32).reshape(length, worlds, EYES)
                for t in range(length):
                    reset = d["segment"][t] != previous_segment
                    rotation = np.rint(prop[t, :, 2] * EYES / 2).astype(int)
                    indices = (np.arange(EYES)[None] + rotation[:, None]) % EYES
                    utility = np.take_along_axis(utility, indices, 1)
                    shadow = np.take_along_axis(shadow, indices, 1)
                    age = np.take_along_axis(age, indices, 1)
                    previous = np.take_along_axis(previous, indices[:, :, None], 1)
                    utility[reset] = shadow[reset] = age[reset] = 0
                    previous[reset] = 0
                    fixed = np.exp(-age / 32)
                    gate = np.clip(fixed * np.clip(1 + utility * 12, .1, 1.9), 0, 1)
                    shadow_gate = np.clip(fixed * np.clip(1 + shadow * 12, .1, 1.9), 0, 1)
                    delta, shadow_delta = np.abs(gate - fixed), np.abs(gate - shadow_gate)
                    phase["sector_presentations"] += gate.size
                    phase["gate_sum"] += float(gate.sum(dtype=np.float64))
                    phase["fixed_gate_sum"] += float(fixed.sum(dtype=np.float64))
                    phase["delta_sum"] += float(delta.sum(dtype=np.float64))
                    phase["shadow_delta_sum"] += float(shadow_delta.sum(dtype=np.float64))
                    phase["delta_gt_001"] += int((delta > .001).sum())
                    phase["delta_gt_01"] += int((delta > .01).sum())
                    phase["shadow_delta_gt_001"] += int((shadow_delta > .001).sum())
                    phase["positive_gain"] += int((gain[t] > 0).sum())
                    phase["negative_gain"] += int((gain[t] < 0).sum())
                    current = rgb[t].astype(np.float32) / 255
                    age = np.where(np.abs(current - previous).mean(2) > .07, 0, age + 1)
                    previous = current
                    rotation = np.rint(after_prop[t, :, 2] * EYES / 2).astype(int)
                    indices = (np.arange(EYES)[None] - rotation[:, None]) % EYES
                    corrected = np.take_along_axis(gain[t], indices, 1)
                    real_feed, shadow_feed = ((corrected, gain[t]) if arm == "aligned"
                                               else (gain[t], corrected))
                    utility = .9 * utility + .1 * np.clip(real_feed, -.08, .08)
                    shadow = .9 * shadow + .1 * np.clip(shadow_feed, -.08, .08)
                    assert np.allclose(np.abs(utility).mean(1), d["utility_abs"][t], atol=1e-9)
                    previous_segment = d["segment"][t].copy()
                combos, counts = np.unique(d["combo"].reshape(-1, 3), axis=0, return_counts=True)
                for combo, count in zip(combos, counts):
                    key = str(tuple(int(v) for v in combo))
                    phase["scenes"][key] = phase["scenes"].get(key, 0) + int(count)
    finally:
        db.close()
    for phase in phases:
        count = phase["sector_presentations"]
        phase.update(mean_gate=phase["gate_sum"] / count,
                     mean_fixed_gate=phase["fixed_gate_sum"] / count,
                     mean_abs_gate_change_vs_fixed=phase["delta_sum"] / count,
                     fraction_gate_change_gt_001=phase["delta_gt_001"] / count,
                     fraction_gate_change_gt_01=phase["delta_gt_01"] / count,
                     mean_abs_alignment_shadow_gate_change=phase["shadow_delta_sum"] / count,
                     fraction_shadow_gate_change_gt_001=phase["shadow_delta_gt_001"] / count,
                     fraction_positive_gain=phase["positive_gain"] / count,
                     fraction_negative_gain=phase["negative_gain"] / count,
                     scene_count=len(phase["scenes"]))
    return phases


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    result = dict(
        method="Same factual gain/RGB/actions; opposite alignment for shadow utility only. "
               "No shadow policy or predictor rollout; no causal behavior attribution.",
        arms={})
    for arm in config["arms"]:
        result["arms"][arm] = one_arm(output / arm, config["milestones"], arm)
        print(json.dumps(dict(arm=arm, phases=[{k: p[k] for k in
              ("start", "end", "mean_abs_gate_change_vs_fixed", "fraction_gate_change_gt_01",
               "mean_abs_alignment_shadow_gate_change", "scene_count")}
              for p in result["arms"][arm]])), flush=True)
    dump(output / "MECHANISM_DIAGNOSTICS.json", result)


if __name__ == "__main__":
    main()
