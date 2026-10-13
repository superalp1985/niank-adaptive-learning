"""Rebuild frame equations directly from raw records, without EyeMemory."""

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from tracking_v17.run import dump
from tracking_v17.world import EYES, DT

ROOT = Path(__file__).resolve().parent


def digest(value):
    return np.frombuffer(hashlib.sha256(value.tobytes()).digest(), np.uint8)


def _session(path, db):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    worlds, step = summary["worlds"], 0
    utility = np.zeros((worlds, EYES), np.float32)
    age = np.zeros_like(utility)
    previous = np.zeros((worlds, EYES, 3), np.float32)
    previous_segment = np.full(worlds, -1)
    collision_history = np.zeros(worlds, bool)
    accumulated = {k: 0 for k in ("contact", "safe_contact", "collision", "impact",
                   "visible", "locked", "reacquired", "reacquire_delay")}
    attacked_steps = attacked_impacts = 0
    gain_abs_sum = utility_abs_sum = alignment_abs_sum = 0.
    large_gain_count = rotated_rows = 0
    for chunk in sorted(path.glob("transitions_*.npz")):
        with np.load(chunk) as d:
            assert int(d["start_step"]) == step
            length = len(d["actions"])
            rows = db.execute("SELECT body,time,segment,rgb,proprio,action,next_rgb,"
                              "next_proprio,retention_gain,target_mask FROM events "
                              "WHERE time>=? AND time<? ORDER BY time,body",
                              (step, step + length)).fetchall()
            assert len(rows) == length * worlds
            keys = np.array([(r[0], r[1], r[2], r[5]) for r in rows], np.int64)
            assert np.array_equal(keys[:, 0], np.tile(np.arange(worlds), length))
            assert np.array_equal(keys[:, 1], np.repeat(np.arange(step, step + length), worlds))
            assert np.array_equal(keys[:, 2], d["segment"].ravel())
            assert np.array_equal(keys[:, 3], d["actions"].ravel())
            rgb_bytes = [zlib.decompress(r[3]) for r in rows]
            after_bytes = [zlib.decompress(r[6]) for r in rows]
            assert all(len(v) == EYES * 3 for v in rgb_bytes + after_bytes)
            assert all(len(r[4]) == len(r[7]) == 24 for r in rows)
            rgb = np.frombuffer(b"".join(rgb_bytes), np.uint8).reshape(length, worlds, EYES, 3)
            prop = np.frombuffer(b"".join(r[4] for r in rows), np.float32).reshape(length, worlds, 6)
            next_prop = np.frombuffer(b"".join(r[7] for r in rows), np.float32).reshape(length, worlds, 6)
            gains = np.frombuffer(b"".join(zlib.decompress(r[8]) for r in rows),
                                  np.float32).reshape(length, worlds, EYES)
            assert np.isfinite(prop).all() and np.isfinite(next_prop).all()
            assert np.isfinite(gains).all()
            assert np.allclose(gains.mean(2), d["omitted_error"] - d["pred_error"], atol=1e-7)
            masks = np.unpackbits(np.frombuffer(b"".join(r[9] for r in rows), np.uint8))
            masks = masks.reshape(length, worlds, EYES).astype(bool)
            assert np.array_equal(masks.any(2), d["visible"])
            locked = np.take_along_axis(masks, d["lock_eye"][:, :, None], 2)[:, :, 0]
            assert np.array_equal(locked, d["locked"])
            shifts = np.rint(next_prop[:, :, 2] * EYES / 2).astype(int)
            assert np.array_equal(shifts, d["rotation_shift"])
            for t in range(length):
                reset = d["segment"][t] != previous_segment
                rotation = np.rint(prop[t, :, 2] * EYES / 2).astype(int)
                indices = (np.arange(EYES)[None] + rotation[:, None]) % EYES
                utility = np.take_along_axis(utility, indices, 1)
                age = np.take_along_axis(age, indices, 1)
                previous = np.take_along_axis(previous, indices[:, :, None], 1)
                utility[reset] = age[reset] = 0
                previous[reset] = 0
                assert np.array_equal(digest(utility), d["utility_before_digest"][t]), (path, step + t, "before")
                gate = np.clip(np.exp(-age / 32) * np.clip(1 + utility * 12, .1, 1.9), 0, 1)
                assert np.array_equal(digest(gate), d["gate_digest"][t]), (path, step + t, "gate")
                current = rgb[t].astype(np.float32) / 255
                change = np.abs(current - previous).mean(2)
                age = np.where(change > .07, 0, age + 1)
                previous = current.copy()
                feedback_indices = (np.arange(EYES)[None] - shifts[t, :, None]) % EYES
                corrected_gain = np.take_along_axis(gains[t], feedback_indices, 1)
                alignment = np.abs(corrected_gain - gains[t]).mean(1)
                assert np.allclose(alignment, d["gain_alignment_abs"][t], atol=1e-9)
                if summary["feedback_enabled"]:
                    feed = corrected_gain if summary["frame_mode"] == "aligned" else gains[t]
                    utility = .9 * utility + .1 * np.clip(feed, -.08, .08)
                assert np.array_equal(digest(utility), d["utility_after_digest"][t]), (path, step + t, "after")
                assert np.allclose(np.abs(utility).mean(1), d["utility_abs"][t], atol=1e-9)
                collision_history[reset] = False
                collision_history |= d["collision"][t]
                assert np.array_equal(d["contact"][t] & ~collision_history, d["safe_contact"][t])
                collision_history[d["contact"][t]] = False
                previous_segment = d["segment"][t].copy()
            reward = (2 * d["progress"] - .015 - .004 * (d["actions"] != 0)
                      - .4 * d["collision"] - 3 * d["impact"]
                      + 8 * d["contact"] + 4 * d["safe_contact"])
            assert np.allclose(reward, d["reward"], atol=2e-6)
            gain_abs_sum += float(np.abs(gains).sum(dtype=np.float64))
            utility_abs_sum += float(d["utility_abs"].sum(dtype=np.float64))
            alignment_abs_sum += float(d["gain_alignment_abs"].sum(dtype=np.float64))
            large_gain_count += int((np.abs(gains) > .001).sum())
            rotated_rows += int((shifts != 0).sum())
            for key in accumulated:
                accumulated[key] += int(d[key].sum())
            attacked = d["combo"][:, :, 2] != 0
            attacked_steps += int(attacked.sum())
            attacked_impacts += int((attacked & d["impact"]).sum())
            step += length
    count = step * worlds
    assert count == summary["experiences"] == db.execute("SELECT count(*) FROM events").fetchone()[0]
    recomputed = dict(contacts=accumulated["contact"], safe_contacts=accumulated["safe_contact"],
        collision_rate=accumulated["collision"] / count,
        impact_rate=accumulated["impact"] / count,
        contact_per_minute=accumulated["contact"] * 60 / (count * DT),
        visibility_steps=accumulated["visible"],
        visible_lock=accumulated["locked"] / max(accumulated["visible"], 1),
        attacked_steps=attacked_steps, attacked_impact_rate=attacked_impacts / max(attacked_steps, 1),
        reacquisitions=accumulated["reacquired"],
        mean_reacquire_seconds=accumulated["reacquire_delay"] * DT /
                               max(accumulated["reacquired"], 1))
    for key, value in recomputed.items():
        assert np.isclose(value, summary[key], atol=1e-8), (path, key)
    return dict(path=path.relative_to(ROOT).as_posix(), rows=count,
                frame_digests_checked=step * 3, recomputed=recomputed,
                mean_abs_sector_gain=gain_abs_sum / (count * EYES),
                fraction_abs_gain_gt_001=large_gain_count / (count * EYES),
                mean_abs_utility=utility_abs_sum / count,
                mean_alignment_change=alignment_abs_sum / count,
                fraction_nonzero_rotation=rotated_rows / count)


def session(path):
    db = sqlite3.connect(f"file:{path / 'events.sqlite'}?mode=ro", uri=True)
    try:
        return _session(path, db)
    finally:
        db.close()


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    source = {name: hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == value
              for name, value in manifest["source"].items()}
    assert all(source.values()), source
    warm_hash = hashlib.sha256((output / "warm_start.pt").read_bytes()).hexdigest()
    assert warm_hash == manifest["warm_start_sha256"]
    import torch
    initial = [torch.load(output / arm / "initial.pt", weights_only=True)["model"]
               for arm in config["arms"]]
    warm = torch.load(output / "warm_start.pt", weights_only=True)["model"]
    for name, state in warm.items():
        for key, tensor in state.items():
            assert all(torch.equal(tensor, model[name][key]) for model in initial)
    results = []
    for arm in config["arms"]:
        base = output / arm
        for path in [base, *sorted(base.glob("probe_*"))]:
            result = session(path)
            results.append(result)
            print(json.dumps(dict(audited=result["path"], rows=result["rows"],
                                  frame_digests=result["frame_digests_checked"])), flush=True)
    dump(output / "AUDIT.json", dict(source_unchanged=source,
         same_warm_start_tensors=True, sessions=results,
         all_rows=sum(v["rows"] for v in results),
         frame_digests_checked=sum(v["frame_digests_checked"] for v in results),
         audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()
