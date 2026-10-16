"""Read-only verification of all experienced delayed targets and source bytes."""
import argparse
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from tracking_v23.audit_results import check as original_check
from tracking_v23.evaluate import file_hash
from tracking_v23.persistence import atomic_json as dump


def check(path):
    path = Path(path)
    original = original_check(path)
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sources"].items():
        if file_hash(path/"source"/relative) != expected:
            raise AssertionError("Executed source snapshot changed: "+relative)
    db_path = path/"events.sqlite"
    before = file_hash(db_path)
    db = sqlite3.connect(db_path.resolve().as_uri()+"?mode=ro", uri=True)
    try:
        step, _, raw = db.execute("SELECT step,sha256,payload FROM runtime").fetchone()
        runtime = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)["state"]
        origin, bodies = runtime["origin_step"], runtime["bodies"]
        if runtime.get("delay") is None:
            if db.execute("SELECT count(*) FROM delayed_outcomes").fetchone()[0]:
                raise AssertionError("Control unexpectedly has delayed training")
            return dict(original=original, delayed_targets=0, source_hashes=True)
        if runtime["delay"].pending_count() or runtime["delay_ready"]:
            raise AssertionError("Unsettled delayed consequences")
        length = step-origin
        prefix = np.zeros((length+1, bodies, 4), np.float64)
        episodes = np.full((length+1, bodies), -1, np.int64)
        for event_step, body, raw in db.execute(
                "SELECT step,body,facts FROM events WHERE step>?", (origin,)):
            event = json.loads(raw)
            index = event_step-origin
            prefix[index, body] = [event["collision_duration"], event["impact_count"],
                                   event["contact"], event["safe_contact"]]
            episodes[index, body] = event["segment"]
        prefix = prefix.cumsum(0)
        state = np.full((length+1, bodies, 2), np.nan)
        terminal = np.zeros((length+1, bodies), bool)
        for event_step, body, energy, integrity, done in db.execute(
                "SELECT step,body,energy,integrity,terminal FROM observed_state"):
            state[event_step-origin, body] = [energy, integrity]
            terminal[event_step-origin, body] = done
        if not np.isfinite(state[1:]).all():
            raise AssertionError("Missing pre-reset bodily observations")
        terminal_prefix = terminal.cumsum(0)
        inputs = {}
        for start, raw in db.execute("SELECT step,batch FROM gaze_inputs"):
            with np.load(io.BytesIO(raw), allow_pickle=False) as batch:
                if batch["x"].shape != (bodies, 964) or not np.isfinite(batch["x"]).all():
                    raise AssertionError("Invalid delayed input")
                inputs[start] = {key: batch[key].copy() for key in
                                 ("actions", "before", "episodes")}
        counts = dict.fromkeys(manifest["config"]["delay_horizons"], 0)
        truncated = 0
        seen = set()
        for start, end, body, episode, horizon, action, raw in db.execute(
                "SELECT start,end,body,episode,horizon,action,target FROM delayed_outcomes"):
            a, b = start-origin, end-origin
            if not (0 <= a < b <= length and end-start <= horizon):
                raise AssertionError("Invalid future-only delayed interval")
            source = inputs[start]
            if action != source["actions"][body] or episode != source["episodes"][body]:
                raise AssertionError("Input/action/episode alignment failed")
            if episodes[a+1, body] != episode or episodes[b, body] != episode:
                raise AssertionError("Delayed credit crossed an episode")
            nterminal = terminal_prefix[b, body]-terminal_prefix[a, body]
            if nterminal != int(terminal[b, body]):
                raise AssertionError("Credit extends beyond a terminal")
            if end-start < horizon and not terminal[b, body]:
                raise AssertionError("Unexplained shortened window")
            delta = prefix[b, body]-prefix[a, body]
            expected = np.concatenate((
                state[b, body]-source["before"][body], [delta[0]/((end-start)*.01)],
                delta[1:], [float(terminal[b, body])]))
            target = np.frombuffer(raw, np.float32)
            np.testing.assert_allclose(target, expected, rtol=1e-6, atol=1e-7)
            key = (start, body, horizon)
            if key in seen:
                raise AssertionError("Duplicate delayed target")
            seen.add(key)
            counts[horizon] += 1
            truncated += end-start < horizon
        expected_count = len(inputs)*bodies
        if set(counts.values()) != {expected_count}:
            raise AssertionError("A selected observation lacks a horizon target")
        updates = [json.loads(row[0]) for row in db.execute(
            "SELECT result FROM delayed_updates ORDER BY step")]
        agent = runtime["agent"]
        if (sum(row["samples"] for row in updates) != len(seen)
                or agent.delayed_samples != len(seen)
                or agent.delayed_updates != len(updates)*manifest["config"]["delay_epochs"]):
            raise AssertionError("Auxiliary optimization count differs from physical records")
        result = dict(original=original, delayed_targets=len(seen), horizon_counts=counts,
                      terminal_truncated_targets=truncated, auxiliary_updates=agent.delayed_updates,
                      source_hashes=True, no_episode_crossing=True,
                      physical_targets_reproduced=True)
    finally:
        db.close()
    if file_hash(db_path) != before:
        raise AssertionError("Read-only audit changed the database")
    return dict(result, database_sha256=before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {name: check(args.root/name) for name in ("control", "delayed")}
    dump(args.root/"audit.json", results)
    print(json.dumps({key: dict(targets=value["delayed_targets"],
                               frames=value["original"]["physical_events_checked"])
                      for key, value in results.items()}), flush=True)
