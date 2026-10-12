"""Batch independent archive/physics audit; no training or simulation rerun."""

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

ROOT = None


def folder_audit(folder):
    files = sorted(folder.glob("transitions_*.npz"))
    stats = dict(transitions=0, episodes=0, success_count=0, safe_success_count=0,
                 impact_count=0, collision_steps=0, reward_sum=0., mse_sum=0.,
                 risk_error_sum=0., scenes={})
    collided = None
    with sqlite3.connect(folder / "events.sqlite") as db:
        cursor = db.execute("SELECT id,world,time,obs,action,next,image FROM events ORDER BY id")
        for path in files:
            with np.load(path) as data:
                worlds = data["reward"].shape[1]
                size = data["reward"].size
                if collided is None:
                    collided = np.zeros(worlds, bool)
                rows = cursor.fetchmany(size)
                if len(rows) != size:
                    raise AssertionError("Missing raw records")
                identities = np.array([r[:3] for r in rows], np.int64)
                expected = np.arange(stats["transitions"], stats["transitions"] + size)
                np.testing.assert_array_equal(identities[:, 0], expected)
                np.testing.assert_array_equal(identities[:, 1], expected % worlds)
                np.testing.assert_array_equal(identities[:, 2], expected // worlds)
                raw_before = np.stack([np.frombuffer(r[3], np.float32) for r in rows])
                raw_after = np.stack([np.frombuffer(r[5], np.float32) for r in rows])
                np.testing.assert_array_equal(raw_before,
                                               data["features"][:, :, :32].reshape(-1, 32))
                np.testing.assert_array_equal(raw_after, data["next_obs"].reshape(-1, 32))
                np.testing.assert_array_equal(np.array([r[4] for r in rows]),
                                               data["actions"].ravel())
                if any(len(zlib.decompress(r[6])) != 768 for r in rows):
                    raise AssertionError("Bad raw raster")
                recalled = data["retrieved"].ravel()
                if np.any(recalled >= expected) or np.any(
                        (recalled >= 0) & (recalled % worlds != expected % worlds)):
                    raise AssertionError("Future/cross-body recall")
                np.testing.assert_array_equal(data["success"], data["contact"] & ~data["impact"])
                np.testing.assert_array_equal(
                    data["done"], data["success"] | data["impact"] | (data["duration"] >= 128))
                distance_before = np.linalg.norm(raw_before[:, 2:4].astype(np.float64),
                                                axis=1) * 6
                distance_after = np.linalg.norm(raw_after[:, 2:4].astype(np.float64),
                                               axis=1) * 6
                reward = (2 * (distance_before - distance_after) - 0.015
                          - 0.004 * (data["actions"].ravel() != 0)
                          - 0.4 * data["collision"].ravel()
                          - 2 * data["impact"].ravel() + 8 * data["success"].ravel())
                np.testing.assert_allclose(data["reward"].ravel(), reward, atol=2e-6, rtol=2e-5)
                safe = np.zeros_like(data["success"])
                for step in range(len(data["reward"])):
                    collided |= data["collision"][step]
                    safe[step] = data["success"][step] & ~collided
                    collided[data["done"][step]] = False
                stats["transitions"] += size
                for key, source in (("episodes", "done"), ("success_count", "success"),
                                    ("impact_count", "impact"), ("collision_steps", "collision")):
                    stats[key] += int(data[source].sum())
                stats["safe_success_count"] += int(safe.sum())
                stats["reward_sum"] += float(data["reward"].astype(np.float64).sum())
                stats["mse_sum"] += float(data["pred_error"].astype(np.float64).sum())
                if "risk_pred" in data.files:
                    labels = np.stack((data["collision"], data["impact"]), -1)
                    stats["risk_error_sum"] += float(np.square(data["risk_pred"] - labels).sum())
                combos = data["combo"].reshape(-1, 3)
                for combo in np.unique(combos, axis=0):
                    mask = (combos == combo).all(1)
                    key = str(tuple(map(int, combo)))
                    scene = stats["scenes"].setdefault(key, dict(steps=0, episodes=0,
                        success=0, safe_success=0, impact=0, collision=0))
                    scene["steps"] += int(mask.sum())
                    for key, source in (("episodes", "done"), ("success", "success"),
                                        ("impact", "impact"), ("collision", "collision")):
                        scene[key] += int(data[source].ravel()[mask].sum())
                    scene["safe_success"] += int(safe.ravel()[mask].sum())
        if cursor.fetchone() is not None:
            raise AssertionError("Unaccounted raw records")
    stats["success"] = stats["success_count"] / max(stats["episodes"], 1)
    stats["safe_success"] = stats["safe_success_count"] / max(stats["episodes"], 1)
    stats["impact"] = stats["impact_count"] / max(stats["episodes"], 1)
    stats["collision_step_rate"] = stats["collision_steps"] / max(stats["transitions"], 1)
    stats["reward"] = stats["reward_sum"] / max(stats["transitions"], 1)
    stats["mse"] = stats["mse_sum"] / max(stats["transitions"], 1)
    stats["risk_brier"] = stats["risk_error_sum"] / max(2 * stats["transitions"], 1)
    stats["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in files + [folder / "events.sqlite"]}
    return stats


def main():
    output = ROOT / "pilot"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, value in manifest["source"].items():
        if hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() != value:
            raise AssertionError(f"Source changed: {name}")
    results = {}
    for mode in manifest["config"]["modes"]:
        results[mode] = {"training": folder_audit(output / mode)}
        for step in [0] + manifest["config"]["milestones"]:
            results[mode][str(step)] = {name: folder_audit(output / mode / f"probe_{step}" / name)
                                       for name in ("seen", "heldout")}
            extra = output / mode / f"probe_{step}" / "heldout_no_risk"
            if extra.exists():
                results[mode][str(step)]["heldout_no_risk"] = folder_audit(extra)
        print(json.dumps(dict(audited=mode, transitions=results[mode]["training"]["transitions"])),
              flush=True)
    (output / "AUDIT.json").write_text(json.dumps(dict(results=results,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
