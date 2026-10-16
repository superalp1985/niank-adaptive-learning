"""Read-only checks of physical learning records from either continuation arm."""
import argparse
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(path):
    path = Path(path)
    db_path = path / "events.sqlite"
    before = digest(db_path)
    db = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise AssertionError("SQLite integrity check failed")
        last_step, bodies = db.execute(
            "SELECT MAX(step), MAX(body)+1 FROM events").fetchone()
        flags = np.zeros((last_step+1, bodies, 3), np.int32)
        seen = np.zeros((last_step+1, bodies), bool)
        head_travel = head_samples = head_limits = 0
        head_choices = np.zeros(3, np.int64)
        episodes, goals, combinations = {}, {}, set()
        for step, body, raw in db.execute("SELECT step,body,facts FROM events"):
            if seen[step, body]:
                raise AssertionError("Duplicate physical event row")
            seen[step, body] = True
            event = json.loads(raw)
            flags[step, body] = [event[k] for k in ("collision", "impact", "contact")]
            for catalogue, key in (
                    (episodes, (body, event["segment"])),
                    (goals, (body, event["segment"], event["goal_id"]))):
                first, last = catalogue.get(key, (step, step))
                catalogue[key] = (min(first, step), max(last, step))
            combinations.add(tuple(event["combo"]))
            if "head_yaw" in event:
                head_samples += 1
                head_travel += event["head_travel"]
                head_limits += abs(event["head_yaw"]) >= np.pi/2-1e-6
                head_choices[event["head_action"]] += 1
        if not seen[1:].all():
            raise AssertionError("Missing physical frames")
        prefix = flags.cumsum(axis=0)
        windows_checked = 0
        widths = {}
        for raw, window_blob in db.execute(
                "SELECT batch,windows FROM facts ORDER BY block"):
            windows = np.frombuffer(window_blob, np.int64).reshape(-1, 4)
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                labels = data["a2"][:, 4:7]
                width = data["a0"].shape[1]
                if not all(np.isfinite(data[f"a{i}"]).all() for i in range(6)):
                    raise AssertionError("Nonfinite learning record")
                if width not in (576, 964):
                    raise AssertionError("Unexpected feature width")
                maximum_action = 26 if width == 964 else 8
                if np.any(data["a1"] < 0) or np.any(data["a1"] > maximum_action):
                    raise AssertionError("Action outside recorded sensor layout")
                widths[width] = widths.get(width, 0)+len(labels)
            start, end, body = windows[:, 0], windows[:, 1], windows[:, 2]
            if np.any(start < 0) or np.any(end > last_step) or np.any(end <= start):
                raise AssertionError("Invalid consequence window")
            physical = prefix[end, body]-prefix[start, body] > 0
            if not np.array_equal(labels, physical):
                raise AssertionError("Consequence labels differ from physical events")
            windows_checked += len(windows)

        rate = .0008
        rolling = deque(maxlen=12)
        trials = negatives = changes = retained = 0
        counts = np.zeros(3, np.int64)
        returns = np.zeros(3, np.float64)
        for raw, in db.execute("SELECT outcome FROM choices ORDER BY step"):
            result = json.loads(raw)
            if result["validation_min_start"] < result["decision_step"]:
                raise AssertionError("Learning credit used pre-decision experience")
            if result["reverted"] or result["retained_updates"] != result["attempted_updates"]:
                raise AssertionError("Unexpected rollback of a learning trial")
            mode = result["mode"]
            counts[mode] += 1
            returns[mode] += .2*(result["score"]-returns[mode])
            retained += result["retained_updates"]
            if result["attempted_updates"]:
                trials += 1
                negatives += result["gain"] < 0
                rolling.append(result["gain"])
                previous = rate
                if len(rolling) == 12 and trials % 4 == 0:
                    average = np.mean(rolling)
                    if average < -.03:
                        rate = max(.00005, rate*.85)
                    elif average > .03:
                        rate = min(.003, rate*1.10)
                changes += rate != previous
            if abs(rate-result["learning_rate"]) > 1e-12:
                raise AssertionError("Learning-rate changes do not follow recorded feedback")

        runtime_step, runtime_sha, runtime = db.execute(
            "SELECT step,sha256,payload FROM runtime WHERE id=1").fetchone()
        if hashlib.sha256(runtime).hexdigest() != runtime_sha or runtime_step != last_step:
            raise AssertionError("Runtime checkpoint is incomplete")
        saved = torch.load(io.BytesIO(runtime), map_location="cpu", weights_only=False)
        agent = saved["state"]["agent"]
        origin = saved["state"].get("origin_step", 0)
        if agent.trial is not None:
            raise AssertionError("A learning decision remains unsettled")
        if not np.array_equal(counts, agent.mode_counts):
            raise AssertionError("Mode counts differ from experienced choices")
        if not np.allclose(returns, agent.mode_returns, rtol=0, atol=1e-12):
            raise AssertionError("Learning preferences differ from recorded feedback")
        if retained != agent.accepted_updates or negatives != agent.negative_trials:
            raise AssertionError("Retained update counters disagree")
        if changes != len(agent.lr_changes):
            raise AssertionError("Learning-rate history length disagrees")
        metrics = saved["state"]["metrics"]
        if metrics["frames"] != int(seen.sum()):
            raise AssertionError("Saved frame count disagrees")
        event_totals = flags.sum((0, 1))
        for key, value in zip(("collision", "impact", "contact"), event_totals):
            if metrics[key] != int(value):
                raise AssertionError(f"Saved physical total disagrees: {key}")
        parameter_finite = all(torch.isfinite(value).all().item()
                               for value in agent.model.state_dict().values())
        if not parameter_finite:
            raise AssertionError("Nonfinite predictor parameters")
        result = dict(
            physical_events_checked=int(seen.sum()),
            consequence_windows_checked=windows_checked,
            feature_width_counts=widths,
            source_and_new_records_included=True,
            settled_choices=int(counts.sum()),
            learning_trials=trials,
            negative_trials_retained=int(negatives),
            retained_updates=retained,
            learning_rate_changes_reproduced=changes,
            final_learning_rate=rate,
            mode_counts_reproduced=counts.tolist(),
            mode_returns_reproduced=returns.tolist(),
            complete_runtime_step=runtime_step,
            continuation_origin_step=origin,
            fresh_body_frames=(runtime_step-origin)*bodies,
            fresh_seconds_per_body=(runtime_step-origin)*.01,
            physical_scene_combinations=len(combinations),
            episodes_with_fresh_experience=sum(last > origin for first, last in episodes.values()),
            new_episode_starts=sum(first > origin for first, last in episodes.values()),
            targets_with_fresh_experience=sum(last > origin for first, last in goals.values()),
            finite_predictor_parameters=True,
            head_frame_samples=head_samples,
            head_travel_radians=float(head_travel),
            head_limit_frame_fraction=head_limits/max(head_samples, 1),
            head_action_frame_counts=head_choices.tolist(),
            limitation="Record consistency is not evidence of better behavior.")
    finally:
        db.close()
    after = digest(db_path)
    if before != after:
        raise AssertionError("Read-only audit changed source database")
    return dict(result, database_sha256=before, database_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {str(path): check(path) for path in args.runs}
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
