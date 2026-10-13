"""Independently reconstruct access and feedback from persisted raw records."""

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
EYES, MEMBERS, DT = 128, 3, .12


def digest(value):
    return np.frombuffer(hashlib.sha256(value.tobytes()).digest(), np.uint8)


def bootstrap_identity(body, time, segment):
    """Scalar unsigned arithmetic, independent of the training vector routine."""
    mask = (1 << 64) - 1
    result = []
    for member in range(1, 4):
        value = (((int(body) + 1) * 0x9E3779B97F4A7C15) & mask)
        value ^= ((int(time) + 1) * 0xBF58476D1CE4E5B9) & mask
        value ^= ((int(segment) + 1) * 0x94D049BB133111EB) & mask
        value ^= (member * 0xD6E8FEB86659FD93) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        result.append((value >> 11) < int(.8 * (1 << 53)))
    return result


def _close(actual, expected, label, atol=1e-8):
    assert np.allclose(actual, expected, rtol=1e-6, atol=atol), label


def _session(path, db):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    worlds, step = int(summary["worlds"]), 0
    shape = (worlds, EYES)
    utility, deviation, age = (np.zeros(shape, np.float32) for _ in range(3))
    previous = np.zeros((*shape, 3), np.float32)
    history = np.zeros_like(previous)
    selected = np.zeros(shape, bool)
    previous_segment = np.full(worlds, -1)
    previous_done = np.ones(worlds, bool)
    last_after = last_next_prop = None
    collision_history = np.zeros(worlds, bool)
    previous_lock = np.zeros(worlds, bool)
    ever_locked = np.zeros(worlds, bool)
    goal_ids = np.zeros(worlds, np.int64)
    wait = np.full(worlds, -1, np.int64)
    rng = np.random.default_rng(summary["memory_seed"])
    totals = {key: 0 for key in ("contact", "safe_contact", "collision", "impact",
              "visible", "locked", "reacquired", "reacquire_delay")}
    attacked_steps = attacked_impacts = bootstrap_count = 0
    prediction_sum = utility_sum = deviation_sum = gain_sum = 0.
    promoted_count = rotated_rows = local_batches = 0
    for chunk in sorted(path.glob("transitions_*.npz")):
        with np.load(chunk) as d:
            assert int(d["start_step"]) == step, (path, "chunk sequence")
            length = len(d["actions"])
            rows = db.execute("SELECT body,time,segment,rgb,proprio,action,next_rgb,"
                "next_proprio,retention_gain,target_mask FROM events "
                "WHERE time>=? AND time<? ORDER BY time,body", (step, step + length)).fetchall()
            evidence = db.execute("SELECT body,time,start,normal_errors,omitted_errors,"
                "scores,disagreement,bootstrap FROM evidence WHERE time>=? AND time<? "
                "ORDER BY time,body", (step, step + length)).fetchall()
            assert len(rows) == len(evidence) == length * worlds, (path, "row count")
            keys = np.asarray([(r[0], r[1], r[2], r[5]) for r in rows], np.int64)
            assert np.array_equal(keys[:, 0], np.tile(np.arange(worlds), length))
            assert np.array_equal(keys[:, 1], np.repeat(np.arange(step, step + length), worlds))
            assert np.array_equal(keys[:, 2], d["segment"].ravel())
            assert np.array_equal(keys[:, 3], d["actions"].ravel())
            assert np.array_equal(np.asarray([r[:2] for r in evidence]), keys[:, :2])
            rgb_raw = [zlib.decompress(r[3]) for r in rows]
            after_raw = [zlib.decompress(r[6]) for r in rows]
            assert all(len(v) == EYES * 3 for v in rgb_raw + after_raw)
            assert all(len(r[4]) == len(r[7]) == 24 and len(r[9]) == 16 for r in rows)
            assert all(len(r[3]) == len(r[4]) == len(r[5]) == 12 and
                       len(r[7]) == 3 for r in evidence)
            rgb = np.frombuffer(b"".join(rgb_raw), np.uint8).reshape(length, worlds, EYES, 3)
            after = np.frombuffer(b"".join(after_raw), np.uint8).reshape(rgb.shape)
            prop = np.frombuffer(b"".join(r[4] for r in rows), np.float32).reshape(length, worlds, 6)
            next_prop = np.frombuffer(b"".join(r[7] for r in rows), np.float32).reshape(prop.shape)
            gains = np.frombuffer(b"".join(zlib.decompress(r[8]) for r in rows),
                                 np.float32).reshape(length, worlds, EYES)
            starts = np.asarray([r[2] for r in evidence]).reshape(length, worlds)
            full_error, omitted_error, scores = [
                np.frombuffer(b"".join(r[column] for r in evidence), np.float32)
                .reshape(length, worlds, MEMBERS) for column in (3, 4, 5)]
            disagreement = np.asarray([r[6] for r in evidence]).reshape(length, worlds)
            bootstrap = np.frombuffer(b"".join(r[7] for r in evidence), bool).reshape(scores.shape)
            expected_bootstrap = np.asarray([bootstrap_identity(r[0], r[1], r[2])
                                             for r in rows]).reshape(scores.shape)
            assert np.array_equal(bootstrap, expected_bootstrap), (path, "bootstrap identity")
            assert np.array_equal(starts, d["probe_start"])
            expected_starts = ((np.arange(step, step + length)[:, None] +
                                3 * np.arange(worlds)[None]) % 8) * 16
            assert np.array_equal(starts, expected_starts), (path, "probe coverage")
            for value in (prop, next_prop, full_error, omitted_error, scores, gains, disagreement):
                assert np.isfinite(value).all(), (path, "nonfinite evidence")
            assert (full_error >= 0).all() and (omitted_error >= 0).all()
            assert (disagreement >= 0).all()
            calculated_scores = (omitted_error - full_error) / (
                omitted_error + full_error + summary["floor"])
            _close(scores, calculated_scores, (path, "relative score"), atol=2e-7)
            _close(scores.mean(2), d["gain_mean"], (path, "gain mean"))
            _close(scores.std(2), d["gain_std"], (path, "gain deviation"))
            _close(disagreement, d["prediction_disagreement"], (path, "prediction disagreement"))
            masks = np.unpackbits(np.frombuffer(b"".join(r[9] for r in rows), np.uint8))
            masks = masks.reshape(length, worlds, EYES).astype(bool)
            assert np.array_equal(masks.any(2), d["visible"])
            locked = np.take_along_axis(masks, d["lock_eye"][:, :, None], 2)[:, :, 0]
            assert np.array_equal(locked, d["locked"])
            if "rotation_shift" in d:
                assert np.array_equal(np.rint(next_prop[:, :, 2] * EYES / 2).astype(int),
                                      d["rotation_shift"])
            for t in range(length):
                current_step = step + t
                label = (str(path), current_step)
                reset = d["segment"][t] != previous_segment
                assert np.array_equal(reset, previous_done), (*label, "reset timing")
                assert np.array_equal(d["segment"][t][reset], previous_segment[reset] + 1)
                if last_after is not None:
                    assert np.array_equal(rgb[t][~reset], last_after[~reset]), (*label, "RGB continuity")
                    assert np.array_equal(prop[t][~reset], last_next_prop[~reset]), (*label, "proprio continuity")
                rotation = np.rint(prop[t, :, 2] * EYES / 2).astype(int)
                indices = (np.arange(EYES)[None] + rotation[:, None]) % EYES
                for value in (utility, deviation, age, previous, history, selected):
                    index = indices[:, :, None] if value.ndim == 3 else indices
                    value[:] = np.take_along_axis(value, index, axis=1)
                    value[reset] = 0
                old_selected = selected.copy()
                assert np.array_equal(digest(utility), d["utility_before_digest"][t]), (*label, "utility before")
                assert np.array_equal(digest(deviation), d["deviation_before_digest"][t]), (*label, "deviation before")
                base_gate = np.exp(-age / 32)
                current = rgb[t].astype(np.float32) / 255
                full = np.empty((worlds, 11, EYES), np.float32)
                full[:, :3] = current.transpose(0, 2, 1)
                full[:, 3:6] = previous.transpose(0, 2, 1)
                full[:, 6:9] = (history * base_gate[:, :, None]).transpose(0, 2, 1)
                full[:, 9] = base_gate
                full[:, 10] = np.minimum(age / 128, 1)
                assert np.array_equal(digest(full), d["full_features_digest"][t]), (*label, "full features")
                priority = utility.copy()
                if summary["mode"] == "protected":
                    priority += deviation
                elif summary["mode"] == "calibration":
                    priority.fill(0)
                ties = rng.random(shape)
                selected.fill(False)
                for body in range(worlds):
                    order = np.asarray(sorted(range(EYES),
                        key=lambda eye: (-float(priority[body, eye]), ties[body, eye])))
                    selected[body, order[:56]] = True
                    extra = rng.choice(order[56:], 8, replace=False)
                    assert len(np.unique(extra)) == 8
                    selected[body, extra] = True
                assert (selected.sum(1) == 64).all()
                assert np.array_equal(np.packbits(selected, axis=1), d["selected_bits"][t]), (*label, "selected")
                promoted = selected & ~old_selected
                promoted[reset] = False
                assert np.array_equal(promoted.sum(1), d["promoted"][t]), (*label, "promoted")
                gate = base_gate * selected
                assert np.array_equal(digest(gate), d["gate_digest"][t]), (*label, "gate")
                _close(gate.mean(1), d["retention_gate"][t], (*label, "mean gate"))
                group_eyes = (starts[t, :, None] + np.arange(16)) % EYES
                fraction = np.take_along_axis(selected, group_eyes, 1).mean(1)
                _close(fraction, d["probe_selected_fraction"][t], (*label, "probed access"))
                group_gain = np.zeros(shape, np.float32)
                means, deviations = scores[t].mean(1), scores[t].std(1)
                np.put_along_axis(group_gain, group_eyes, means[:, None], 1)
                assert np.array_equal(group_gain, gains[t]), (*label, "input-group gain")
                if summary["feedback_enabled"] and summary["mode"] != "calibration":
                    for body in range(worlds):
                        eye = group_eyes[body]
                        utility[body, eye] = .9 * utility[body, eye] + .1 * means[body]
                        deviation[body, eye] = .9 * deviation[body, eye] + .1 * deviations[body]
                assert np.array_equal(digest(utility), d["utility_after_digest"][t]), (*label, "utility after")
                assert np.array_equal(digest(deviation), d["deviation_after_digest"][t]), (*label, "deviation after")
                _close(utility.mean(1), d["retention_utility"][t], (*label, "mean utility"))
                _close(np.abs(utility).mean(1), d["utility_abs"][t], (*label, "absolute utility"))
                _close(deviation.mean(1), d["deviation_mean"][t], (*label, "mean deviation"))
                change = np.abs(current - previous).mean(2)
                age = np.where(change > .07, 0, age + 1).astype(np.float32)
                keep = np.where(change > .07, .96 * base_gate, .65 * base_gate)
                history = (keep[:, :, None] * history +
                           (1 - keep[:, :, None]) * current).astype(np.float32)
                previous = current.copy()
                collision_history[reset] = False
                collision_history |= d["collision"][t]
                assert not (d["contact"][t] & d["impact"][t]).any()
                assert np.array_equal(d["contact"][t] & ~collision_history, d["safe_contact"][t])
                collision_history[d["contact"][t]] = False
                previous_lock[reset] = ever_locked[reset] = False
                wait[reset] = -1
                new_goal = reset | (d["goal_id"][t] != goal_ids)
                ever_locked[new_goal] = False
                wait[new_goal] = -1
                goal_ids = d["goal_id"][t].copy()
                lost = ever_locked & previous_lock & ~locked[t]
                wait[lost] = current_step
                reacquired = locked[t] & (wait >= 0)
                delay = np.where(reacquired, current_step - wait, 0)
                assert np.array_equal(reacquired, d["reacquired"][t])
                assert np.array_equal(delay, d["reacquire_delay"][t])
                wait[locked[t]] = -1
                ever_locked |= locked[t]
                previous_lock = locked[t].copy()
                previous_segment = d["segment"][t].copy()
                previous_done = d["done"][t].copy()
                last_after, last_next_prop = after[t], next_prop[t]
                promoted_count += int(promoted.sum())
                rotated_rows += int((rotation != 0).sum())
            reward = (2 * d["progress"] - .015 - .004 * (d["actions"] != 0)
                      - .4 * d["collision"] - 3 * d["impact"]
                      + 8 * d["contact"] + 4 * d["safe_contact"])
            _close(reward, d["reward"], (path, "reward"), atol=2e-6)
            for key in totals:
                totals[key] += int(d[key].sum())
            attacked = d["combo"][:, :, 2] != 0
            attacked_steps += int(attacked.sum())
            attacked_impacts += int((attacked & d["impact"]).sum())
            prediction_sum += float(d["pred_error"].sum(dtype=np.float64))
            utility_sum += float(d["utility_abs"].sum(dtype=np.float64))
            deviation_sum += float(d["deviation_mean"].sum(dtype=np.float64))
            gain_sum += float(np.abs(scores).sum(dtype=np.float64))
            bootstrap_count += int(bootstrap.sum())
            local_batches += int(np.ceil(length * worlds / 768))
            step += length
    count = step * worlds
    assert step == summary["step"] and count == summary["experiences"] and count > 0
    assert count == db.execute("SELECT count(*) FROM events").fetchone()[0]
    assert count == db.execute("SELECT count(*) FROM evidence").fetchone()[0]
    recomputed = dict(contacts=totals["contact"], safe_contacts=totals["safe_contact"],
        safe_contact_fraction=totals["safe_contact"] / max(totals["contact"], 1),
        collision_rate=totals["collision"] / count, impact_rate=totals["impact"] / count,
        contact_per_minute=totals["contact"] * 60 / max(count * DT, 1),
        visibility_steps=totals["visible"], visible_lock=totals["locked"] / max(totals["visible"], 1),
        attacked_steps=attacked_steps, attacked_impact_rate=attacked_impacts / max(attacked_steps, 1),
        reacquisitions=totals["reacquired"],
        mean_reacquire_seconds=totals["reacquire_delay"] * DT / max(totals["reacquired"], 1),
        prediction_mse=prediction_sum / count)
    for key, value in recomputed.items():
        _close(value, summary[key], (path, key))
    expected_feedback = step if summary["feedback_enabled"] and summary["mode"] != "calibration" else 0
    assert summary["utility_feedback_steps"] == expected_feedback
    if summary["training"]:
        epochs = summary["update_presentations"] / count
        assert epochs == int(epochs) and epochs > 0
        assert summary["predictor_presentations"] == 2 * bootstrap_count * epochs
        assert summary["local_update_batches"] == local_batches * epochs
    else:
        assert summary["update_presentations"] == summary["predictor_presentations"] == summary["local_update_batches"] == 0
    return dict(path=str(path.relative_to(REPO)), rows=count, frame_digests_checked=step * 6,
        recomputed=recomputed, bootstrap_presentations=bootstrap_count,
        mean_abs_member_gain=gain_sum / (count * MEMBERS),
        mean_abs_utility=utility_sum / count, mean_deviation=deviation_sum / count,
        promoted_accesses=promoted_count, fraction_rotated=rotated_rows / count)


def session(path):
    path = Path(path)
    db = sqlite3.connect(f"file:{path / 'events.sqlite'}?mode=ro", uri=True)
    try:
        return _session(path, db)
    finally:
        db.close()


def main():
    import torch
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {}
    for name, expected in manifest["source"].items():
        sources[name] = dict(snapshot=sha(output / "source" / name) == expected,
                             current=sha(REPO / name) == expected)
        assert all(sources[name].values()), (name, sources[name])
    assert sha(output / "warm_start.pt") == manifest["warm_start_sha256"]
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    assert sha(output / "calibration" / "model.pt") == calibration["shared_model_sha256"]
    shared = torch.load(output / "calibration" / "model.pt", map_location="cpu", weights_only=True)["model"]
    for arm in config["arms"]:
        initial = torch.load(output / arm / "initial.pt", map_location="cpu", weights_only=True)["model"]
        assert initial.keys() == shared.keys()
        for name, tensors in shared.items():
            assert initial[name].keys() == tensors.keys()
            assert all(torch.equal(tensor, initial[name][key]) for key, tensor in tensors.items())
    with sqlite3.connect(f"file:{output / 'calibration' / 'events.sqlite'}?mode=ro", uri=True) as db:
        sums = [np.frombuffer(a, np.float32) + np.frombuffer(b, np.float32)
                for a, b in db.execute("SELECT normal_errors,omitted_errors FROM evidence")]
    expected_floor = max(float(np.quantile(np.asarray(sums).mean(1), config["floor_quantile"])),
                         1 / 255 ** 2)
    assert expected_floor == calibration["floor"] and len(sums) == calibration["rows"]
    first, second = [output / arm / "probe_0_frozen" for arm in config["arms"]]
    first_chunks, second_chunks = sorted(first.glob("transitions_*.npz")), sorted(second.glob("transitions_*.npz"))
    assert len(first_chunks) == len(second_chunks) > 0
    for a, b in zip(first_chunks, second_chunks):
        with np.load(a) as left, np.load(b) as right:
            assert set(left.files) == set(right.files)
            assert all(np.array_equal(left[key], right[key]) for key in left.files)
    paths = [output / "calibration"]
    for arm in config["arms"]:
        paths.extend([output / arm, *sorted((output / arm).glob("probe_*"))])
    results = []
    for path in paths:
        result = session(path)
        results.append(result)
        print(json.dumps(dict(audited=result["path"], rows=result["rows"],
                              frame_digests=result["frame_digests_checked"])), flush=True)
    current_python = {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(ROOT.glob("*.py"))}
    result = dict(source_verification=sources, same_calibration_tensors=True,
        same_initial_frozen_trajectory=True, calibrated_floor=expected_floor,
        sessions=results, all_rows=sum(v["rows"] for v in results),
        frame_digests_checked=sum(v["frame_digests_checked"] for v in results),
        audit_source_sha256=sha(Path(__file__)), current_python_sha256=current_python,
        supplementary_python=[name for name in current_python if name not in manifest["source"]],
        limits=["Reconstructs logged local-error algebra, not every neural prediction.",
                "Prediction-error authenticity is not independently replayed from per-step weights.",
                "Registered sources and their snapshots are checked; later analysis files are recorded separately."])
    temporary = output / "AUDIT.tmp"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output / "AUDIT.json")


if __name__ == "__main__":
    main()
