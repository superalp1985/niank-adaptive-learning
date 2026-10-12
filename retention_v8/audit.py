"""Independently check physical scoring, snapshots, archives and frozen probes."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from experiment import (
    ROOT, OutcomeRetentionModel, detail_probe, generate, metrics, recovery_probe,
)
from retention_v5.experiment import TOLERANCE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    output = ROOT / "confirmation"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert complete["models"] == len(results) == 24
    assert sha(ROOT / "experiment.py") == sha(output / "experiment.py") == manifest["sha256"]
    for version, expected in manifest["dependencies"].items():
        assert sha(ROOT.parent / version / "experiment.py") == expected
        assert sha(output / f"dependency_{version}.py") == expected
    torch.set_num_threads(2)
    truth = {
        "final": generate(512, manifest["data"]["final"]["ids"][0], scenario="final"),
        "left_layout": generate(512, manifest["data"]["left_layout"]["ids"][0], scenario="left_layout"),
        "texture_shift": generate(512, manifest["data"]["texture_shift"]["ids"][0], delay=34, scenario="texture_shift"),
    }
    for name, data in truth.items():
        assert data.fingerprint() == manifest["data"][name]["sha256"]
    episode_ids = [set(range(v["ids"][0], v["ids"][1] + 1)) for v in manifest["data"].values()]
    for i, a in enumerate(episode_ids):
        for b in episode_ids[i + 1:]:
            assert not a & b
    checks = []
    for result in results:
        target = output / f"u{result['unique']}_c{result['cycles']}_s{result['seed']}_{result['mode']}"
        for name, data in truth.items():
            with np.load(target / f"{name}_predictions.npz") as prediction:
                np.testing.assert_array_equal(prediction["ids"], data.ids)
                actions = prediction["action"]
                regret = data.costs[np.arange(len(data)), actions] - data.costs.min(-1)
                np.testing.assert_allclose(prediction["regret"], regret, rtol=0, atol=1e-7)
                np.testing.assert_array_equal(prediction["success"], regret <= TOLERANCE)
                assert abs(float((regret <= TOLERANCE).mean()) - result[name]["success"]) < 1e-10
        checks.append({"unique": result["unique"], "cycles": result["cycles"],
                       "seed": result["seed"], "mode": result["mode"], "physical_scoring": "passed"})
    for name in manifest["data"]:
        with closing(sqlite3.connect(output / f"{name}.sqlite")) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(episodes)")}
            assert columns == {"id", "context", "body"}
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    # Independently reload two high-volume paired checkpoints for numerical replay.
    replayed = []
    detail = generate(256, manifest["data"]["detail"]["ids"][0], delay=26, scenario="detail")
    recovery = generate(256, manifest["data"]["recovery"]["ids"][0], delay=26, scenario="recovery")
    for mode in ("quartet_point", "quartet_delta"):
        result = next(r for r in results if r["unique"] == 2048 and r["cycles"] == 32 and r["seed"] == 23 and r["mode"] == mode)
        target = output / f"u2048_c32_s23_{mode}"
        model = OutcomeRetentionModel().cuda().eval()
        model.load_state_dict(torch.load(target / "model.pt", map_location="cuda", weights_only=True))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        final, _ = metrics(model, mode, truth["final"])
        probe = detail_probe(model, mode, detail)
        transition, _ = recovery_probe(model, mode, recovery, output / "recovery.sqlite")
        for key in ("success", "regret", "state_mse", "active_count"):
            assert abs(final[key] - result["final"][key]) < 1e-6
        for context in ("0", "1"):
            for key in ("effect_mse", "predicted_effect_mae", "fine_retained"):
                assert abs(probe[context][key] - result["detail"][context][key]) < 1e-6
        for key in ("stale_success", "recovered_success", "stale_regret", "recovered_regret"):
            assert abs(transition[key] - result["recovery"][key]) < 1e-6
        for name, value in model.state_dict().items():
            torch.testing.assert_close(before[name], value)
        replayed.append(mode)
    audit = {
        "models": len(results), "physical_prediction_files_verified": len(results) * 3,
        "data_ids_disjoint": True, "all_source_hashes_match": True,
        "archive_schema_has_no_future_labels": True, "sqlite_integrity": "ok",
        "frozen_replay_modes": replayed, "weight_updates_during_replay": 0,
        "checks": checks,
    }
    (ROOT / "AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "checks"}))


if __name__ == "__main__":
    run()
