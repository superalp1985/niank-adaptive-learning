"""Recompute physical truth and delayed credits from saved trajectories."""

import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

from core import ROOT, physics, MOMENTA, FORCES, TOLERANCE, HORIZON
from run import dump


def join(paths):
    parts = []
    for path in sorted(paths):
        with np.load(path) as source:
            parts.append({key: source[key] for key in source.files})
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def audit_directory(path):
    data = join(path.glob("transitions_*.npz"))
    decision = join(path.glob("decisions_*.npz"))
    credit = join(path.glob("credits_*.npz"))
    steps, count = data["action"].shape
    rows = np.arange(count)[None]
    states, contacts, costs = physics(
        data["body"].reshape(-1, 5),
        MOMENTA[data["pair"][:, :, 0].reshape(-1)],
        FORCES[data["pair"][:, :, 1].reshape(-1)],
        data["context"].reshape(-1))
    states, contacts, costs = states.reshape(steps, count, 3, 2), contacts.reshape(steps, count, 3), costs.reshape(steps, count, 3)
    times = np.arange(steps)[:, None]
    actions = data["action"]
    np.testing.assert_allclose(data["actual"], states[times, rows, actions], rtol=0, atol=1e-7)
    np.testing.assert_allclose(data["contact"], contacts[times, rows, actions], rtol=0, atol=0)
    np.testing.assert_allclose(data["cost"], costs[times, rows, actions], rtol=0, atol=1e-7)
    np.testing.assert_allclose(data["regret"], data["cost"] - costs.min(-1), rtol=0, atol=1e-7)
    np.testing.assert_allclose(data["mse"], ((data["predicted"] - data["actual"])**2).mean(-1), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(data["body"][1:, :, :2], data["actual"][:-1])
    ops = np.repeat(decision["ops"], 4, axis=0)
    reward = -data["cost"] - 0.5 * data["mse"] - 0.002 * (1 + data["boost"].astype(int)) - 0.0002 * data["admitted"]
    reward[::4] -= 0.01 * (decision["ops"] == 1) + 0.01 * (decision["ops"] == 2) + 0.02 * (decision["ops"] == 4)
    np.testing.assert_array_equal(data["boost"], ops == 3)
    np.testing.assert_array_equal(credit["completed_at"] - credit["step"] + 1, HORIZON)
    step_lookup = {int(step): i for i, step in enumerate(data["step"])}
    decision_lookup = {int(step): i for i, step in enumerate(decision["step"])}
    for index, step in enumerate(credit["step"]):
        start = step_lookup[int(step)]
        expected = np.zeros(count, np.float32)
        for offset in range(HORIZON):
            expected += 0.97**offset * reward[start + offset]
        np.testing.assert_allclose(credit["returns"][index], expected, rtol=1e-6, atol=1e-6)
        which = decision_lookup[int(step)]
        np.testing.assert_array_equal(credit["ops"][index], decision["ops"][which])
        np.testing.assert_array_equal(credit["identities"][index], decision["identities"][which])
        uses = np.zeros_like(credit["uses"][index])
        for offset in range(4):
            for slot in range(2):
                selected = np.arange(count) if slot == 0 else np.flatnonzero(data["boost"][start + offset])
                np.add.at(uses, (selected, data["route"][start + offset, selected, slot]), 1)
        np.testing.assert_array_equal(credit["uses"][index], uses)
    for i in range(len(decision["step"])):
        ptr = decision["pointers"][i]
        live = decision["live"][i, np.arange(count), ptr]
        identities = decision["directory_ids"][i, np.arange(count), ptr]
        np.testing.assert_array_equal(identities, decision["identities"][i])
        assert ((decision["ops"][i] != 1) | ((identities >= 0) & ~live)).all()
        assert ((decision["ops"][i] != 5) | live).all()
    connection = sqlite3.connect(f"file:{(path / 'events.sqlite').as_posix()}?mode=ro", uri=True)
    try:
        events = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert events == len(decision["step"]) * count
    finally:
        connection.close()
    return dict(path=str(path.relative_to(ROOT)), transitions=steps * count, raw_events=events,
                matured_decisions=len(credit["step"]) * count,
                success=float((data["regret"] <= TOLERANCE).mean()))


def main(stage):
    root = ROOT / stage
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, sha in manifest["sha256"].items():
        assert hashlib.sha256((ROOT.parent / relative).read_bytes()).hexdigest() == sha
    checked = [audit_directory(path.parent) for path in sorted(root.rglob("transitions_00.npz"))]
    result = dict(directories=len(checked), transitions=sum(v["transitions"] for v in checked),
                  matured_decisions=sum(v["matured_decisions"] for v in checked),
                  checks=checked, source_hashes_match=True)
    dump(root / "AUDIT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
