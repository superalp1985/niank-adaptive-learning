"""Batch full-record audit, avoiding scalar NumPy dispatch per transition."""

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from tracking_v17.run import dump, ROOT
from tracking_v17.world import EYES, DT


def session(path):
    summary_path = path / "summary.json"
    if not summary_path.exists():
        summary_path = path / "result_16384.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    worlds, step = summary["worlds"], 0
    db = sqlite3.connect(f"file:{path / 'events.sqlite'}?mode=ro", uri=True)
    accumulated = {k: 0 for k in ("contact", "safe_contact", "collision", "impact",
                   "visible", "locked", "reacquired", "reacquire_delay")}
    collision_history = np.zeros(worlds, bool)
    previous_segment = np.full(worlds, -1)
    attacked_steps = attacked_impacts = 0
    for chunk in sorted(path.glob("transitions_*.npz")):
        with np.load(chunk) as d:
            assert int(d["start_step"]) == step
            length = len(d["actions"])
            for t in range(length):
                collision_history[d["segment"][t] != previous_segment] = False
                collision_history |= d["collision"][t]
                assert np.array_equal(d["contact"][t] & ~collision_history, d["safe_contact"][t])
                collision_history[d["contact"][t]] = False
                previous_segment = d["segment"][t].copy()
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
            rgb = [zlib.decompress(r[3]) for r in rows]
            after = [zlib.decompress(r[6]) for r in rows]
            assert all(len(v) == EYES * 3 for v in rgb + after)
            assert all(len(r[4]) == len(r[7]) == 24 for r in rows)
            prop = np.frombuffer(b"".join(r[4] for r in rows), np.float32)
            after_prop = np.frombuffer(b"".join(r[7] for r in rows), np.float32)
            assert np.isfinite(prop).all() and np.isfinite(after_prop).all()
            gains = np.frombuffer(b"".join(zlib.decompress(r[8]) for r in rows),
                                  np.float32).reshape(length, worlds, EYES)
            assert np.isfinite(gains).all()
            assert np.allclose(gains.mean(2), d["omitted_error"] - d["pred_error"], atol=1e-7)
            masks = np.unpackbits(np.frombuffer(b"".join(r[9] for r in rows), np.uint8))
            masks = masks.reshape(length, worlds, EYES).astype(bool)
            assert np.array_equal(masks.any(2), d["visible"])
            locked = np.take_along_axis(masks, d["lock_eye"][:, :, None], 2)[:, :, 0]
            assert np.array_equal(locked, d["locked"])
            for key in accumulated:
                accumulated[key] += int(d[key].sum())
            attacked = d["combo"][:, :, 2] != 0
            attacked_steps += int(attacked.sum())
            attacked_impacts += int((attacked & d["impact"]).sum())
            assert np.isfinite(d["reward"]).all() and np.isfinite(d["pred_error"]).all()
            step += length
    count = step * worlds
    assert count == summary["experiences"] == db.execute("SELECT count(*) FROM events").fetchone()[0]
    db.close()
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
    return dict(path=str(path.relative_to(ROOT)), rows=count, recomputed=recomputed)


def main():
    pilot = ROOT / "pilot"
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    source = {name: hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == digest
              for name, digest in manifest["source"].items()}
    assert all(source.values())
    results = []
    for mode in manifest["config"]["modes"]:
        base = pilot / mode
        results.append(session(base))
        print(json.dumps(dict(audited=str(base.relative_to(ROOT)), rows=results[-1]["rows"])), flush=True)
        for probe in sorted(base.glob("probe_*")):
            results.append(session(probe))
            print(json.dumps(dict(audited=str(probe.relative_to(ROOT)), rows=results[-1]["rows"])), flush=True)
    dump(pilot / "AUDIT.json", dict(source_unchanged=source, sessions=results,
                                   all_rows=sum(v["rows"] for v in results),
                                   audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()
