"""Read-only numerical and provenance audit of completed V3 artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def audit_learning(folder):
    frozen = json.loads((folder / "FROZEN.json").read_text(encoding="utf-8"))
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    assert results["status"] == "complete"
    for name, digest in frozen["source_hashes"].items():
        candidate = folder / "source" / name
        if not candidate.exists():
            candidate = ROOT / name
        assert hashlib.sha256(candidate.read_bytes()).hexdigest() == digest, candidate
    matrix = Path(frozen["config"]["matrix_path"])
    assert hashlib.sha256(matrix.read_bytes()).hexdigest() == frozen["matrix_sha256"]
    counts = 0
    for record in results["records"]:
        assert record["presentations"] == record["unique_examples"] * record["repetitions"]
        for split, scenarios in record["metrics"].items():
            for scenario, metrics in scenarios.items():
                if not metrics["predictions_file"]:
                    continue
                path = folder / "predictions" / metrics["predictions_file"]
                with np.load(path, allow_pickle=False) as saved:
                    correct = int(np.sum(saved["prediction"] == saved["truth"]))
                    assert correct == metrics["correct"]
                    assert len(saved["truth"]) == metrics["questions"]
                    assert abs(correct / len(saved["truth"]) - metrics["accuracy"]) < 1e-12
                    all_steps = np.all(
                        (saved["step_predictions"] == saved["states"]) | (saved["states"] < 0),
                        axis=1,
                    )
                    assert int(all_steps.sum()) == metrics["all_steps_correct"]
                    probabilities = saved["probabilities"]
                    np.testing.assert_allclose(probabilities.sum(1), 1, atol=2e-6)
                    assert np.isfinite(probabilities).all()
                    brier = np.sum(
                        (probabilities - np.eye(16)[saved["truth"]]) ** 2, axis=1
                    )
                    assert abs(float(brier.mean()) - metrics["final_brier"]) < 1e-6
                    counts += len(saved["truth"])
    return {"folder": folder.name, "records": len(results["records"]),
            "predictions_checked": counts, "status": "passed"}


def audit_graph(path):
    with np.load(path, allow_pickle=False) as saved:
        first, second = saved["matrix"], saved["rewired_matrix"]
        assert first.shape == second.shape == (len(saved["ids"]), len(saved["ids"]))
        for sign in (1, -1):
            for axis in (0, 1):
                np.testing.assert_array_equal(
                    (first * sign > 0).sum(axis), (second * sign > 0).sum(axis)
                )
        np.testing.assert_array_equal(np.sort(first.ravel()), np.sort(second.ravel()))
        return {"file": path.name, "nodes": len(saved["ids"]),
                "edges": int(np.count_nonzero(first)), "status": "passed"}


def main():
    manifest = json.loads((ROOT / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    source = ROOT.parents[1] / "external" / "fly-brain" / manifest["commit"]
    for item in manifest["files"]:
        path = source / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    report = {
        "downloads": "all pinned files match",
        "graphs": [audit_graph(path) for path in sorted(ROOT.glob("connectome_*.npz"))],
        "learning": [audit_learning(path.parent) for path in sorted(ROOT.glob("learning*/results.json"))],
    }
    (ROOT / "ARTIFACT_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
