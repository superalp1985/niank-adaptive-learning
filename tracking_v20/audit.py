"""Replay factual consequence labels and validate historical access before archiving."""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from tracking_v17.run import dump

ROOT = Path(__file__).resolve().parent


def session_audit(path):
    fields = ("progress", "contact", "safe_contact", "collision", "impact", "done", "segment")
    arrays = {key: [] for key in fields}
    predictions = []
    for source in sorted(path.glob("transitions_*.npz")):
        with np.load(source, allow_pickle=False) as data:
            for key in fields:
                arrays[key].append(data[key])
            predictions.append(data["consequence_predictions"])
    data = {key: np.concatenate(values) for key, values in arrays.items()}
    steps, worlds = data["progress"].shape
    expected = np.zeros((steps, worlds, 5), np.float32)
    horizon = np.zeros((steps, worlds), np.int16)
    live = np.ones((steps, worlds), bool)
    for offset in range(8):
        if offset >= steps:
            break
        count = steps - offset
        use = live[:count]
        contribution = (.95 ** offset * data["progress"][offset:] / 1.2)
        expected[:count, :, 0] = np.where(
            use, expected[:count, :, 0] + contribution, expected[:count, :, 0])
        for channel, key in enumerate(("contact", "safe_contact", "collision", "impact"), 1):
            expected[:count, :, channel] = np.maximum(
                expected[:count, :, channel], use * data[key][offset:])
        horizon[:count] += use
        live[:count] &= ~data["done"][offset:] & ~data["contact"][offset:]
    expected[:, :, 0] = np.clip(expected[:, :, 0], -1, 1)
    predicted = np.concatenate(predictions)
    database = sqlite3.connect(path / "events.sqlite")
    try:
        counts = {table: database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("events", "evidence", "outcomes", "directory_events", "retrievals")}
        if counts["events"] != steps * worlds or counts["evidence"] != counts["events"]:
            raise AssertionError("Raw experience count mismatch")
        checked, maximum = 0, 0.
        errors = np.zeros(5, np.float64)
        for body, time, segment, length, raw in database.execute(
                "SELECT body,time,segment,horizon,targets FROM outcomes ORDER BY time,body"):
            target = np.frombuffer(raw, np.float32)
            if len(target) != 5 or not np.isfinite(target).all():
                raise AssertionError("Invalid consequence target")
            if segment != data["segment"][time, body] or length != horizon[time, body]:
                raise AssertionError("Crossed temporal horizon or segment")
            delta = float(np.abs(target - expected[time, body]).max())
            maximum = max(maximum, delta)
            if delta > 2e-6:
                raise AssertionError(f"Factual target differs: {body},{time},{delta}")
            errors += (predicted[time, body] - target) ** 2
            checked += 1
        invalid = database.execute(
            "SELECT COUNT(*) FROM retrievals r LEFT JOIN directory_events d "
            "ON d.body=r.body AND d.time=r.event_time "
            "WHERE d.time IS NULL OR r.event_time>=r.query_time OR "
            "r.event_segment!=d.segment OR r.query_time>? OR r.weight_after<r.weight_before",
            (steps,)).fetchone()[0]
        if invalid:
            raise AssertionError("Invalid historical retrieval")
        row_counts = list(database.execute(
            "SELECT query_time,COUNT(*) FROM retrievals GROUP BY query_time ORDER BY query_time"))
        if any(count != 1536 for _, count in row_counts):
            raise AssertionError("Unequal per-rollout replay budget")
        premature = database.execute(
            "SELECT COUNT(*) FROM retrievals r JOIN outcomes o "
            "ON o.body=r.body AND o.time=r.event_time "
            "WHERE o.time+o.horizon>r.query_time").fetchone()[0]
        if premature:
            raise AssertionError("Retrieved consequence before it was observed")
        return dict(steps=steps, worlds=worlds, counts=counts,
            consequence_labels_checked=checked, maximum_target_difference=maximum,
            consequence_channel_mse=(errors / max(checked, 1)).tolist(),
            replay_batches=len(row_counts), invalid_retrievals=invalid,
            premature_labels=premature)
    finally:
        database.close()


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = ROOT / config["stage"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for relative, sha in manifest["source"].items():
        source = output / "source" / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise AssertionError("Immutable source snapshot differs")
        if hashlib.sha256((ROOT.parent / relative).read_bytes()).hexdigest() != sha:
            raise AssertionError("Running source changed since start")
    results = {}
    for path in sorted(output.rglob("events.sqlite")):
        name = path.parent.relative_to(output).as_posix()
        results[name] = session_audit(path.parent)
        print(json.dumps(dict(session=name, **results[name])), flush=True)
    budget = [results[arm]["counts"]["retrievals"] for arm in config["arms"]]
    if len(set(budget)) != 1:
        raise AssertionError("Different arm replay budgets")
    result = dict(source_verified=True, equal_replay_budget=budget[0],
                  sessions=results, complete=True)
    dump(output / "AUDIT.json", result)


if __name__ == "__main__":
    main()
