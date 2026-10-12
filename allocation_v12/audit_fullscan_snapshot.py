"""Independent trajectory, raw access, and factual operation-target audit."""

import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib

import numpy as np

from core import ROOT, CANDIDATES, SLOTS, QUERY, Plans, TOLERANCE
from run import dump


def physical_outcomes(body, pair, risk):
    momentum = np.array((0.2, 0.45, 0.7, 0.9), np.float32)[pair[:, 0]]
    force = np.array((-0.24, -0.08, 0.08, 0.24), np.float32)[pair[:, 1]]
    actions = np.array((-1., 0., 1.), np.float32)
    velocity = (momentum[:, None] * body[:, 1:2] + 0.35 * actions[None]
                + risk[:, None] * force[:, None])
    position = body[:, 0:1] + body[:, 3:4] * velocity
    contact = np.where(body[:, 4:5] > 0, position >= 1, position <= 0)
    position = np.clip(position, 0, 1)
    cost = ((position - body[:, 2:3])**2 + 0.025 * velocity**2
            + 0.005 * actions[None]**2 + 0.6 * contact).astype(np.float32)
    return np.stack((position, velocity), -1).astype(np.float32), contact.astype(np.float32), cost


def join(paths):
    parts = []
    for path in sorted(paths):
        with np.load(path) as data:
            parts.append({name: data[name] for name in data.files})
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def audit_directory(path):
    data = join(path.glob("transitions_*.npz"))
    decisions = join(path.glob("decisions_*.npz"))
    outcomes = join(path.glob("outcomes_*.npz"))
    steps, count = data["action"].shape
    results = sorted(path.glob("result_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    result = json.loads(results[-1].read_text())
    rows, times = np.arange(count)[None], np.arange(steps)[:, None]
    states, contacts, costs = physical_outcomes(
        data["body"].reshape(-1, 5), data["pair"].reshape(-1, 2),
        data["context"].reshape(-1))
    states = states.reshape(steps, count, 3, 2)
    contacts, costs = contacts.reshape(steps, count, 3), costs.reshape(steps, count, 3)
    np.testing.assert_array_equal(data["step"], np.arange(steps))
    action = data["action"]
    np.testing.assert_allclose(data["actual"], states[times, rows, action], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(data["contact"], contacts[times, rows, action])
    np.testing.assert_allclose(data["cost"], costs[times, rows, action], rtol=0, atol=1e-7)
    np.testing.assert_allclose(data["regret"], data["cost"] - costs.min(-1), rtol=0, atol=1e-7)
    np.testing.assert_allclose(data["mse"], ((data["predicted"] - data["actual"])**2).mean(-1), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(data["body"][1:, :, 0], data["actual"][:-1, :, 0])
    uninterrupted = data["step"][1:] % 48 != 20
    np.testing.assert_array_equal(data["body"][1:][uninterrupted, :, 1],
                                  data["actual"][:-1][uninterrupted, :, 1])
    np.testing.assert_array_equal(decisions["step"], np.arange(0, steps, 4))
    np.testing.assert_array_equal(outcomes["step"], decisions["step"])
    np.testing.assert_array_equal(outcomes["completed_at"], decisions["step"] + 3)
    reward = np.zeros_like(outcomes["reward"])
    for offset in range(4):
        reward += (-data["cost"][offset::4] - 0.5 * data["mse"][offset::4]) / 4
    reward -= 0.002 * decisions["reads"]
    np.testing.assert_allclose(outcomes["reward"], reward, rtol=1e-6, atol=1e-7)

    live = np.zeros((count, CANDIDATES), bool)
    identities = np.full((count, CANDIDATES), -1, np.int64)
    event_times = np.zeros((count, CANDIDATES), np.int64)
    access = np.full((count, CANDIDATES), -1, np.int64)
    presence = np.zeros((count, CANDIDATES, 3), np.float32)
    plan_items = ([(0, -1, -1)] + [(1, i, -1) for i in range(CANDIDATES)]
                  + [(1, i, j) for i in range(CANDIDATES) for j in range(i + 1, CANDIDATES)]
                  + [(2, i, -1) for i in range(CANDIDATES)])
    assert Plans().items == plan_items and len(plan_items) == 153
    expected_reads, current_cycle_cue_reads, change_window_reads = 0, 0, 0
    connection = sqlite3.connect(f"file:{(path / 'events.sqlite').as_posix()}?mode=ro", uri=True)
    try:
        raw_events = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert raw_events == len(decisions["step"]) * count
        for index, step in enumerate(decisions["step"]):
            slot = (int(step) // 4) % CANDIDATES
            records = connection.execute("SELECT id,world,image FROM events WHERE time=? ORDER BY world",
                                         (int(step),)).fetchall()
            assert len(records) == count
            for row, (identity, world, raw) in enumerate(records):
                assert world == row and identity == step * count + row
                live[row, slot] = False
                if live[row].sum() >= SLOTS:
                    old = np.flatnonzero(live[row])
                    live[row, old[np.argmin(access[row, old])]] = False
                live[row, slot] = True
                identities[row, slot], event_times[row, slot], access[row, slot] = identity, step, step * 3
                raw_image = np.frombuffer(zlib.decompress(raw), np.int8).reshape(3, 6, 6)
                presence[row, slot] = (raw_image != 0).any((1, 2))
            np.testing.assert_array_equal(decisions["live"][index], live)
            np.testing.assert_array_equal(decisions["ids"][index], identities)
            query_presence = ((presence * live[:, :, None]).sum(1) / np.maximum(
                live.sum(1, keepdims=True), 1)).astype(np.float32)
            query_ages = np.sort(np.where(live, (step - event_times) / 64, -1), axis=1)[:, -SLOTS:]
            sensor = np.concatenate((
                data["body"][int(step)], data["context"][int(step), :, None],
                np.full((count, 1), np.sin(2 * np.pi * (step % 48) / 48), np.float32),
                np.full((count, 1), np.cos(2 * np.pi * (step % 48) / 48), np.float32),
            ), 1)
            selected_plans = np.asarray(plan_items)[decisions["choices"][index]]
            features = decisions["plan_features"][index]
            np.testing.assert_array_equal(features[:, -3:], np.eye(3)[selected_plans[:, 0]])
            for which in range(2):
                candidate = selected_plans[:, which + 1]
                available_rows = np.flatnonzero(candidate >= 0)
                chosen = candidate[available_rows]
                expected_meta = np.zeros((count, 6), np.float32)
                expected_meta[available_rows, 0] = np.minimum(
                    (step - event_times[available_rows, chosen]) / 64, 2)
                expected_meta[available_rows, 1] = live[available_rows, chosen]
                expected_meta[available_rows, 2] = identities[available_rows, chosen] >= 0
                if result["mode"] != "state_time":
                    expected_meta[available_rows, 3:] = presence[available_rows, chosen]
                np.testing.assert_array_equal(
                    features[:, QUERY + which * 6:QUERY + (which + 1) * 6], expected_meta)
            if result["mode"] == "no_query_content":
                np.testing.assert_array_equal(features[:, :QUERY], 0)
            else:
                np.testing.assert_array_equal(features[:, :8], sensor)
                expected_presence = 0 if result["mode"] == "state_time" else query_presence
                np.testing.assert_array_equal(features[:, 10:13], expected_presence)
                np.testing.assert_array_equal(features[:, 13:16], np.maximum(query_ages, 0))
                np.testing.assert_array_equal(features[:, 16:19], query_ages >= 0)
            for row, (kind, first, second) in enumerate(selected_plans):
                selected = (first, second)
                if kind == 1:
                    assert result["reads_enabled"]
                    assert identities[row, first] >= 0 and not live[row, first]
                    assert second < 0 or (identities[row, second] >= 0 and not live[row, second])
                elif kind == 2:
                    assert live[row, first]
                reads = 0
                if kind == 1:
                    for which, candidate in enumerate(selected):
                        if candidate < 0:
                            continue
                        # The selected raw row must still exist in the external DB.
                        assert connection.execute("SELECT COUNT(*) FROM events WHERE id=? AND world=?",
                                                  (int(identities[row, candidate]), row)).fetchone()[0] == 1
                        if live[row].sum() >= SLOTS:
                            old = np.flatnonzero(live[row])
                            live[row, old[np.argmin(access[row, old])]] = False
                        live[row, candidate], access[row, candidate] = True, step * 3 + which + 1
                        cycle_start = (step // 48) * 48
                        current_cycle_cue_reads += int(
                            event_times[row, candidate] in (cycle_start, cycle_start + 4))
                        change_window_reads += int(step % 48 == 20)
                        reads += 1
                elif kind == 2:
                    live[row, first] = False
                assert reads == decisions["reads"][index, row]
                expected_reads += reads
            cue_slots = [((step // 48) * 48 // 4 + i) % CANDIDATES for i in range(2)]
            available = live[:, cue_slots].all(1)
            for offset in range(4):
                np.testing.assert_array_equal(data["both_cues"][int(step) + offset], available)
    finally:
        connection.close()
    assert expected_reads == result["retrievals"]
    value_error = decisions["estimated_score"] - outcomes["reward"]
    return dict(path=str(path.relative_to(ROOT)), transitions=steps * count,
                raw_events=raw_events, operation_windows=len(outcomes["step"]) * count,
                reads=expected_reads, success=float((data["regret"] <= TOLERANCE).mean()),
                current_cycle_cue_reads=current_cycle_cue_reads,
                other_event_reads=expected_reads - current_cycle_cue_reads,
                change_window_reads=change_window_reads,
                selected_plan_value_rmse=float(np.sqrt((value_error**2).mean())),
                selected_plan_value_bias=float(value_error.mean()),
                mean_factual_plan_reward=float(outcomes["reward"].mean()),
                largest_group_fraction=float(np.bincount(
                    data["group"].ravel(), minlength=4).max() / data["group"].size))


def audit_mode(output, mode, final_step):
    target = output / mode
    assert (target / f"result_{final_step}.json").is_file(), f"Mode still training: {mode}"
    directories = sorted(path.parent for path in target.rglob("transitions_00.npz"))
    inputs = sorted(path for directory in directories for path in directory.iterdir()
                    if path.name == "events.sqlite" or path.name.startswith(
                        ("transitions_", "decisions_", "outcomes_", "result_")))
    input_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in inputs}
    audit_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = output / f"AUDIT_{mode}.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved["input_sha256"] == input_hashes, "Audited inputs changed"
        assert saved["audit_sha256"] == audit_hash, "Audit implementation changed"
        checked = saved["checks"]
        assert {item["path"] for item in checked} == {
            str(directory.relative_to(ROOT)) for directory in directories}
    else:
        checked = []
        for directory in directories:
            checked.append(audit_directory(directory))
            print(json.dumps({"audited_directory": checked[-1]["path"],
                              "transitions": checked[-1]["transitions"]}), flush=True)
        dump(receipt, dict(mode=mode, audit_sha256=audit_hash,
                           input_sha256=input_hashes, checks=checked))
    return checked


def main(stage, mode=None):
    output = ROOT / stage
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, sha in manifest["sha256"].items():
        assert hashlib.sha256((ROOT.parent / relative).read_bytes()).hexdigest() == sha
    config = manifest["config"]
    modes = [mode] if mode is not None else config["modes"]
    assert all(item in config["modes"] for item in modes)
    if mode is None:
        assert (output / "COMPLETE.json").is_file(), "Training incomplete"
    checked = [item for selected in modes for item in audit_mode(
        output, selected, config["milestones"][-1])]
    assert len(checked) == len(modes) * (2 + 2 * len(config["milestones"]))
    result = dict(directories=len(checked), transitions=sum(v["transitions"] for v in checked),
                  operation_windows=sum(v["operation_windows"] for v in checked),
                  reads=sum(v["reads"] for v in checked), source_hashes_match=True,
                  independent_physics=True, raw_directory_and_query_reconstruction=True,
                  checks=checked)
    if mode is None:
        dump(output / "AUDIT.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
