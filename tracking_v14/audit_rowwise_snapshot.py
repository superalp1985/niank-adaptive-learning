"""Initial rowwise audit, preserved as a stopped diagnostic.

This implementation was intentionally stopped before completion because its
per-record Python loop is too slow for the full archive.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent


def audit_folder(folder):
    files = sorted(folder.glob("transitions_*.npz"))
    if not files:
        raise ValueError(f"No transitions: {folder}")
    count = episodes = success = impacts = collisions = 0
    reward = mse = 0.
    scenes = {}
    with sqlite3.connect(folder / "events.sqlite") as db:
        stored = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        rows = db.execute("SELECT id,world,time,obs,action,next,image FROM events ORDER BY id")
        for path in files:
            with np.load(path) as data:
                shape = data["reward"].shape
                worlds = shape[1]
                for key in ("done", "success", "collision", "impact", "duration",
                            "episode", "pred_error", "retrieved"):
                    if data[key].shape != shape:
                        raise AssertionError(f"Invalid shape: {path}:{key}")
                if not np.isfinite(data["features"]).all():
                    raise AssertionError("Nonfinite sensor features")
                if not np.array_equal(data["success"], data["contact"] & ~data["impact"]):
                    raise AssertionError("Success disagrees with physical contact")
                expected_done = data["success"] | data["impact"] | (data["duration"] >= 128)
                if not np.array_equal(data["done"], expected_done):
                    raise AssertionError("Terminal flags disagree with physical outcomes")
                # Reward distance is reconstructed from measured relative positions.
                old_distance = np.linalg.norm(data["features"][:, :, 2:4], axis=-1) * 6
                new_distance = np.linalg.norm(data["next_obs"][:, :, 2:4], axis=-1) * 6
                expected_reward = (2 * (old_distance - new_distance) - 0.015
                    - 0.004 * (data["actions"] != 0) - 0.4 * data["collision"]
                    - 2 * data["impact"] + 8 * data["success"])
                np.testing.assert_allclose(data["reward"], expected_reward,
                                           atol=2e-6, rtol=2e-5)
                for step in range(shape[0]):
                    for world in range(worlds):
                        row = next(rows)
                        identifier, owner, time, obs, action, after, image = row
                        if identifier != count or owner != world or time != count // worlds:
                            raise AssertionError("Raw record order/ownership/time mismatch")
                        np.testing.assert_array_equal(
                            np.frombuffer(obs, np.float32), data["features"][step, world, :32])
                        np.testing.assert_array_equal(
                            np.frombuffer(after, np.float32), data["next_obs"][step, world])
                        if action != int(data["actions"][step, world]):
                            raise AssertionError("Executed action differs from raw database")
                        if len(zlib.decompress(image)) != 3 * 16 * 16:
                            raise AssertionError("Corrupt raw image")
                        recalled = int(data["retrieved"][step, world])
                        if recalled >= identifier or (recalled >= 0 and recalled % worlds != world):
                            raise AssertionError("Future or cross-body memory access")
                        key = str(tuple(map(int, data["combo"][step, world])))
                        scene = scenes.setdefault(key, dict(steps=0, episodes=0,
                            success=0, impact=0, collision=0, reward=0., mse=0.))
                        scene["steps"] += 1
                        scene["episodes"] += int(data["done"][step, world])
                        scene["success"] += int(data["success"][step, world])
                        scene["impact"] += int(data["impact"][step, world])
                        scene["collision"] += int(data["collision"][step, world])
                        scene["reward"] += float(data["reward"][step, world])
                        scene["mse"] += float(data["pred_error"][step, world])
                        count += 1
                episodes += int(data["done"].sum())
                success += int(data["success"].sum())
                impacts += int(data["impact"].sum())
                collisions += int(data["collision"].sum())
                reward += float(data["reward"].astype(np.float64).sum())
                mse += float(data["pred_error"].astype(np.float64).sum())
        if next(rows, None) is not None or stored != count:
            raise AssertionError("Raw database contains unaccounted transitions")
    return dict(transitions=count, episodes=episodes, success_count=success,
                success=success / max(episodes, 1), impact_count=impacts,
                impact=impacts / max(episodes, 1), collision_steps=collisions,
                collision_step_rate=collisions / max(count, 1),
                reward=reward / max(count, 1), mse=mse / max(count, 1),
                scenes=scenes,
                sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files + [folder / "events.sqlite"]})


def main():
    output = ROOT / "pilot"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["source"].items():
        if hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Training source changed: {name}")
    results = {}
    for mode in manifest["config"]["modes"]:
        results[mode] = {"training": audit_folder(output / mode)}
        for milestone in [0] + manifest["config"]["milestones"]:
            results[mode][str(milestone)] = {
                name: audit_folder(output / mode / f"probe_{milestone}" / name)
                for name in ("seen", "heldout")}
    (output / "AUDIT.json").write_text(json.dumps(
        dict(results=results, audit_source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()), indent=2), encoding="utf-8")
    print(json.dumps({mode: result["8192"]["heldout"]["success"]
                      for mode, result in results.items()}))


if __name__ == "__main__":
    main()
