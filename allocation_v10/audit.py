"""Recompute recorded physics and metadata without training any model."""

from pathlib import Path
import hashlib
import json
import sqlite3
import sys

import numpy as np

from core import ROOT, MOMENTA, FORCES, OPS, physics, TOLERANCE
from run import dump


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        path = folder / ("physics_dependency.py" if name == "physics_dependency" else name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    checked, decisions = 0, 0
    results = []
    for path in sorted(folder.glob("g*/**/transitions.npz")):
        result = json.loads((path.parent / "result.json").read_text())
        with np.load(path) as source:
            data = {k: source[k] for k in source.files}
        body = data["body"]
        steps, count = body.shape[:2]
        states, contact, costs = physics(
            body.reshape(-1, 5), MOMENTA[data["pair"][:, :, 0].ravel()],
            FORCES[data["pair"][:, :, 1].ravel()], data["context"].ravel(),
        )
        action = data["action"].ravel()
        row = np.arange(len(action))
        np.testing.assert_allclose(states[row, action], data["actual"].reshape(-1, 2), rtol=0, atol=1e-6)
        np.testing.assert_allclose(costs[row, action], data["cost"].ravel(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(costs[row, action] - costs.min(1), data["regret"].ravel(), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(body[1:, :, :2], data["actual"][:-1])
        if "predicted" in data:
            np.testing.assert_allclose(((data["predicted"] - data["actual"])**2).mean(-1), data["mse"], atol=1e-6)
        pairs = set(map(tuple, data["pair"].reshape(-1, 2).tolist()))
        reserved = {(0, 0), (3, 3)}
        assert (pairs <= reserved) if result["heldout"] else not (pairs & reserved)
        score = float((data["regret"] <= TOLERANCE).mean())
        assert abs(score - result["success"]) < 1e-9
        op = data["op"]
        selected_live = np.take_along_axis(data["memory_before"], data["pointer"][..., None], 2)[..., 0]
        assert np.all(selected_live[op == 5])
        assert not np.any(selected_live[op == 1])
        assert np.all(data["pointer_id"][op == 1] >= 0)
        assert result["retrievals"] == int((op == 1).sum())
        assert sum(result["operation_counts"].values()) == op.size
        assert int(data["boost"].sum()) + steps * count == result["totals"]["selected_groups"]
        if result["training"]:
            assert result["totals"]["main_presentations"] == steps * count
            assert result["totals"]["replay_presentations"] == int((op == 2).sum())
        db = sqlite3.connect(path.parent / "events.sqlite")
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        columns = [row[1] for row in db.execute("PRAGMA table_info(events)")]
        assert columns == ["id", "world", "time", "image"]
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == (steps // 4) * count
        db.close()
        checked += steps * count
        decisions += op.size
        results.append({"path": str(path.parent.relative_to(folder)), "transitions": steps * count, "success": score})
    result = {"source_hashes": True, "transitions": checked, "decisions": decisions, "runs_and_probes": results}
    dump(folder / "AUDIT.json", result)
    print(json.dumps({"audited_transitions": checked, "audited_decisions": decisions, "artifacts": len(results)}))


if __name__ == "__main__":
    audit(ROOT / sys.argv[1])
