"""Independent telemetry/SQLite chronology and metric audit."""

import ast
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from tracking_v17.run import dump
from tracking_v17.world import EYES, DT


def audit_session(path):
    db = sqlite3.connect(f"file:{path / 'events.sqlite'}?mode=ro", uri=True)
    chunks = sorted(path.glob("transitions_*.npz"))
    summary_path = path / "summary.json"
    if not summary_path.exists():
        summary_path = path / "result_16384.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    worlds, step = summary["worlds"], 0
    totals = dict(contacts=0, safe_contacts=0, collisions=0, impacts=0, visible=0,
                  locked=0, attacked_steps=0, attacked_impacts=0, reacquired=0, delays=0)
    collisions_since_goal = np.zeros(worlds, bool)
    previous_segment = np.full(worlds, -1)
    checked = 0
    for file in chunks:
        with np.load(file) as d:
            assert int(d["start_step"]) == step
            length = len(d["actions"])
            for t in range(length):
                changed = d["segment"][t] != previous_segment
                collisions_since_goal[changed] = False
                collisions_since_goal |= d["collision"][t]
                expected_safe = d["contact"][t] & ~collisions_since_goal
                assert np.array_equal(expected_safe, d["safe_contact"][t]), (file, t)
                collisions_since_goal[d["contact"][t]] = False
                previous_segment = d["segment"][t].copy()
            rows = db.execute("SELECT body,time,segment,rgb,proprio,action,next_rgb,"
                              "next_proprio,retention_gain,target_mask FROM events "
                              "WHERE time>=? AND time<? ORDER BY time,body",
                              (step, step + length)).fetchall()
            assert len(rows) == length * worlds
            for i, row in enumerate(rows):
                body, clock, segment, rgb, prop, action, after, after_prop, gain, mask = row
                t = i // worlds
                assert body == i % worlds and clock == step + t
                assert segment == d["segment"][t, body] and action == d["actions"][t, body]
                assert len(zlib.decompress(rgb)) == len(zlib.decompress(after)) == EYES * 3
                assert len(prop) == len(after_prop) == 6 * 4
                actual_mask = np.unpackbits(np.frombuffer(mask, np.uint8)).astype(bool)
                assert bool(actual_mask.any()) == bool(d["visible"][t, body])
                assert bool(actual_mask[d["lock_eye"][t, body]]) == bool(d["locked"][t, body])
                local_gain = np.frombuffer(zlib.decompress(gain), np.float32)
                assert len(local_gain) == EYES and np.isfinite(local_gain).all()
                assert np.isclose(local_gain.mean(),
                    d["omitted_error"][t, body] - d["pred_error"][t, body], atol=1e-7)
                checked += 1
            attacked = d["combo"][:, :, 2] != 0
            for key, field in (("contacts", "contact"), ("safe_contacts", "safe_contact"),
                               ("collisions", "collision"), ("impacts", "impact"),
                               ("visible", "visible"), ("locked", "locked"),
                               ("reacquired", "reacquired"), ("delays", "reacquire_delay")):
                totals[key] += int(d[field].sum())
            totals["attacked_steps"] += int(attacked.sum())
            totals["attacked_impacts"] += int((attacked & d["impact"]).sum())
            assert np.isfinite(d["pred_error"]).all() and np.isfinite(d["reward"]).all()
            assert (d["locked"] <= d["visible"]).all()
            assert (d["safe_contact"] <= d["contact"]).all()
            step += length
    assert checked == summary["experiences"] == step * worlds
    assert db.execute("SELECT count(*) FROM events").fetchone()[0] == checked
    assert db.execute("SELECT count(*) FROM events WHERE body<0 OR body>=?",
                      (worlds,)).fetchone()[0] == 0
    recomputed = dict(contacts=totals["contacts"], safe_contacts=totals["safe_contacts"],
        collision_rate=totals["collisions"] / checked, impact_rate=totals["impacts"] / checked,
        visibility_steps=totals["visible"],
        visible_lock=totals["locked"] / max(totals["visible"], 1),
        contact_per_minute=totals["contacts"] * 60 / (checked * DT),
        attacked_steps=totals["attacked_steps"],
        attacked_impact_rate=totals["attacked_impacts"] / max(totals["attacked_steps"], 1),
        reacquisitions=totals["reacquired"],
        mean_reacquire_seconds=totals["delays"] * DT / max(totals["reacquired"], 1))
    for key, value in recomputed.items():
        assert np.isclose(value, summary[key], atol=1e-8), (path, key, value, summary[key])
    db.close()
    return dict(path=str(path.relative_to(ROOT)), rows=checked, metrics=recomputed)


def main():
    output = ROOT / "pilot"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    source_checks = {}
    for name, digest in manifest["source"].items():
        source_checks[name] = hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == digest
    assert all(source_checks.values()), "Training source changed after declared run"
    sessions = []
    for mode in manifest["config"]["modes"]:
        path = output / mode
        sessions.append(audit_session(path))
        for probe in sorted(path.glob("probe_*")):
            sessions.append(audit_session(probe))
        print(json.dumps(dict(audited_mode=mode,
                              rows=sum(s["rows"] for s in sessions if mode in s["path"]))), flush=True)
    dump(output / "AUDIT.json", dict(source_unchanged=source_checks,
                                    all_rows=sum(s["rows"] for s in sessions), sessions=sessions))


if __name__ == "__main__":
    main()
