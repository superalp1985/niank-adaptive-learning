"""Recompute all closed-loop facts, scores and directory return counts."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from experiment import ROOT, TOLERANCE, checkpoint, facts, frozen_probe, load_model, physics, worlds


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "confirmation"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert json.loads((output / "COMPLETE.json").read_text())["runs"] == len(results) == 24
    assert manifest["sha256"] == digest(ROOT / "experiment.py") == digest(output / "experiment.py")
    for name, expected in manifest["dependencies"].items():
        assert expected == digest(ROOT.parent / name / "experiment.py") == digest(output / f"dependency_{name}.py")
    torch.set_num_threads(2)
    transition_count = 0
    lifetime_checks = 0
    identities = []
    for seed in manifest["seeds"]:
        assert manifest["checkpoints"][str(seed)] == digest(checkpoint(seed))
        data = manifest["data"][str(seed)]
        identities.extend([set(range(start, end + 1)) for start, end in (data["world_ids"], data["probe_ids"])])
    for i, a in enumerate(identities):
        for b in identities[i + 1:]:
            assert not a & b
    for result in results:
        seed, steps, mode = result["seed"], result["steps"], result["mode"]
        data = manifest["data"][str(seed)]
        base = worlds(result["worlds"], data["world_ids"][0])
        assert base.fingerprint() == data["world_sha256"]
        momentum, force = facts(base)
        target = output / f"t{steps}_s{seed}_{mode}"
        with np.load(target / "transitions.npz") as records:
            body = records["body"]
            count = body.shape[1]
            truth, contact, costs = physics(
                body.reshape(-1, 5), np.tile(momentum, steps), np.tile(force, steps),
                records["context"].reshape(-1),
            )
            actions = records["action"].reshape(-1)
            row = np.arange(len(actions))
            chosen = truth[row, actions].reshape(steps, count, 2)
            np.testing.assert_allclose(chosen, records["actual_state"], rtol=0, atol=1e-7)
            np.testing.assert_allclose(contact[row, actions].reshape(steps, count), records["actual_contact"], rtol=0, atol=0)
            np.testing.assert_allclose(body[1:, :, :2], chosen[:-1], rtol=0, atol=0)
            np.testing.assert_allclose(costs.reshape(steps, count, 3), records["all_action_costs"], rtol=0, atol=1e-7)
            regret = (costs[row, actions] - costs.min(-1)).reshape(steps, count)
            np.testing.assert_allclose(regret, records["regret"], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(regret <= TOLERANCE, records["success"])
            error = ((records["prediction"] - chosen)**2).mean(-1)
            np.testing.assert_allclose(error, records["state_mse"], rtol=1e-6, atol=1e-7)
            assert abs(float((regret <= TOLERANCE).mean()) - result["prequential"]["success"]) < 1e-10
            np.testing.assert_array_equal(records["weight_update_index"], np.tile(np.arange(steps)[:, None], (1, count)))
            live = np.zeros((steps, count, 18), bool)
            saved_indices = records["indices"]
            saved_masks = records["mask"]
            saved_returns = records["returned_count"]
            for step in range(steps):
                for world in range(count):
                    live[step, world, saved_indices[step, world][saved_masks[step, world]]] = True
            dropped = live[:-1] & ~live[1:]
            np.testing.assert_array_equal(dropped.sum(-1), records["dropped_count"][:-1])
            was_dropped = np.zeros((count, 18), bool)
            for step in range(steps - 1):
                returned = live[step + 1] & ~live[step] & was_dropped
                np.testing.assert_array_equal(returned.sum(-1), saved_returns[step])
                was_dropped |= dropped[step]
            lifetime_checks += steps - 1
            if mode == "fixed":
                assert not records["retrieval"].any()
                assert not records["dropped_count"].any()
            if steps == 512:
                with np.load(output / f"t128_s{seed}_{mode}" / "transitions.npz") as short:
                    np.testing.assert_array_equal(records["action"][:128], short["action"])
                    np.testing.assert_allclose(records["prediction"][:128], short["prediction"], atol=1e-7, rtol=1e-6)
            with closing(sqlite3.connect(target / "practice.sqlite")) as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert connection.execute("SELECT COUNT(*) FROM practice").fetchone()[0] == count * steps
                columns = {entry[1] for entry in connection.execute("PRAGMA table_info(practice)")}
                assert not {"all_action_costs", "cue_times", "unexecuted_outcomes"} & columns
                stored = connection.execute(
                    "SELECT next_state, action FROM practice WHERE world=? AND step=?",
                    (int(base.ids[0]), steps - 1),
                ).fetchone()
                np.testing.assert_array_equal(np.frombuffer(stored[0], np.float32), chosen[-1, 0])
                assert stored[1] == int(records["action"][-1, 0])
            transition_count += steps * count
    replay = []
    for mode in ("fixed", "consolidated"):
        seed = 23
        data = manifest["data"][str(seed)]
        probe = worlds(256, data["probe_ids"][0], scenario="frozen_probe")
        assert probe.fingerprint() == data["probe_sha256"]
        model = load_model(seed)
        target = output / f"t512_s{seed}_{mode}"
        model.load_state_dict(torch.load(target / "model.pt", map_location="cuda", weights_only=True))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        observed = frozen_probe(model, probe)
        recorded = next(r["final_probe"] for r in results if r["seed"] == seed and r["steps"] == 512 and r["mode"] == mode)
        for key in ("success", "regret", "state_mse", "left_success", "right_success"):
            assert abs(observed[key] - recorded[key]) < 1e-6
        for name, value in model.state_dict().items():
            torch.testing.assert_close(before[name], value)
        replay.append(mode)
    result = {
        "runs": 24, "verified_physical_transitions": transition_count,
        "directory_transitions_checked": lifetime_checks,
        "source_and_dependency_hashes": "match", "data_ids_disjoint": True,
        "closed_loop_continuity": "passed", "short_long_prefixes": "identical",
        "practice_sqlite": "integrity_and_observed_outcomes_passed",
        "frozen_replay_modes": replay, "replay_weight_updates": 0,
    }
    (ROOT / "AUDIT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
