"""Verify all V7 final outcomes independently from stored summary metrics."""

from contextlib import closing
import hashlib
import json
import sqlite3

import numpy as np

from experiment import ROOT, generate, TOLERANCE


def main():
    output = ROOT / "confirmation"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert json.loads((output / "COMPLETE.json").read_text())["models"] == len(results) == 24
    sha = hashlib.sha256((ROOT / "experiment.py").read_bytes()).hexdigest()
    assert sha == manifest["sha256"]
    assert hashlib.sha256((output / "experiment.py").read_bytes()).hexdigest() == sha
    for version, expected in manifest["dependencies"].items():
        assert hashlib.sha256((ROOT.parent / version / "experiment.py").read_bytes()).hexdigest() == expected
        assert hashlib.sha256((output / f"dependency_{version}.py").read_bytes()).hexdigest() == expected
    data = generate(512, manifest["data"]["final"]["ids"][0], scenario="final")
    assert data.fingerprint() == manifest["data"]["final"]["sha256"]
    ids = [set(range(v["ids"][0], v["ids"][1] + 1)) for v in manifest["data"].values()]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not a & b
    for result in results:
        target = output / f"u{result['unique']}_c{result['cycles']}_s{result['seed']}_{result['mode']}"
        with np.load(target / "final_predictions.npz") as prediction:
            np.testing.assert_array_equal(prediction["ids"], data.ids)
            regret = data.costs[np.arange(len(data)), prediction["action"]] - data.costs.min(-1)
            np.testing.assert_allclose(regret, prediction["regret"], atol=1e-7, rtol=0)
            np.testing.assert_array_equal(regret <= TOLERANCE, prediction["success"])
            assert abs(float((regret <= TOLERANCE).mean()) - result["final"]["success"]) < 1e-10
    for name in manifest["data"]:
        with closing(sqlite3.connect(output / f"{name}.sqlite")) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            columns = {row[1] for row in connection.execute("PRAGMA table_info(episodes)")}
            assert columns == {"id", "body", "context"}
    record = {
        "models": 24, "prediction_files": 24, "physical_scoring": "passed",
        "source_hashes": "match", "shared_ids": True, "split_ids_disjoint": True,
        "database_integrity": "ok", "future_labels_in_database": False,
    }
    (ROOT / "AUDIT.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
