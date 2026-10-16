"""Check completed physical-feedback learning without changing the experiment."""
import argparse
from collections import deque
import copy
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from tracking_v22.run import dump, unpack_batch


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


@torch.no_grad()
def selection_scores(agent, context):
    neural = agent.meta(context).mean(0).cpu().numpy()
    bonus = .04*np.sqrt(np.log(agent.mode_counts.sum()+2)/(agent.mode_counts+1))
    return neural+agent.mode_returns+bonus


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    db_path = run/"events.sqlite"
    expected_hash = digest(db_path)
    db = sqlite3.connect(db_path.as_uri()+"?mode=ro", uri=True)
    torch.set_num_threads(4)
    try:
        summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("Database integrity failure")
        agent = torch.load(run/"learner.pt", map_location="cpu", weights_only=False)
        flags = np.zeros((summary["step"]+1, summary["bodies"], 3), dtype=np.int64)
        event_count = 0
        for step, body, raw in db.execute("SELECT step,body,facts FROM events"):
            event = json.loads(raw)
            flags[step, body] = [int(event[k]) for k in ("collision", "impact", "contact")]
            event_count += 1
        if event_count != summary["frames"]:
            raise RuntimeError("Event count does not cover all body frames")
        prefix = flags.cumsum(axis=0)
        windows_checked = 0
        for blob, window_blob in db.execute("SELECT batch,windows FROM facts ORDER BY block"):
            windows = np.frombuffer(window_blob, dtype=np.int64).reshape(-1, 4)
            with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                observed = data["a2"][:, 4:7]
            start, end, body = windows[:, 0], windows[:, 1], windows[:, 2]
            actual = (prefix[end, body]-prefix[start, body]) > 0
            if not np.array_equal(observed, actual):
                raise RuntimeError("Training consequence labels differ from physical events")
            windows_checked += len(windows)

        credits = [json.loads(row[0]) for row in
                   db.execute("SELECT outcome FROM choices ORDER BY step")]
        rolling = deque(maxlen=12)
        rate, trials, negatives, changes = .0008, 0, 0, 0
        counts = np.zeros(3, dtype=np.int64)
        returns = np.zeros(3, dtype=np.float64)
        for credit in credits:
            if credit["validation_min_start"] < credit["decision_step"]:
                raise RuntimeError("A learning decision used earlier-starting experiences")
            mode = credit["mode"]
            counts[mode] += 1
            returns[mode] += .2*(credit["score"]-returns[mode])
            if credit["attempted_updates"]:
                trials += 1
                negatives += credit["gain"] < 0
                rolling.append(credit["gain"])
                before = rate
                if len(rolling) == 12 and trials % 4 == 0:
                    mean = float(np.mean(rolling))
                    if mean < -.03:
                        rate = max(.00005, rate*.85)
                    elif mean > .03:
                        rate = min(.003, rate*1.10)
                changes += before != rate
            if abs(rate-credit["learning_rate"]) > 1e-12:
                raise RuntimeError("Learning-rate history does not follow recorded feedback")
        if not np.array_equal(counts, agent.mode_counts) or not np.allclose(
                returns, agent.mode_returns, rtol=0, atol=1e-12):
            raise RuntimeError("Recorded facts do not reproduce final learning preferences")
        if negatives != agent.negative_trials or changes != len(agent.lr_changes):
            raise RuntimeError("Recorded feedback does not reproduce adaptive counters")

        replays = []
        for stage in (2048, 4096, 6144):
            base = torch.load(run/f"learner_step_{stage:05d}.pt",
                              map_location="cpu", weights_only=False)
            base.device = torch.device("cpu")
            context = base.trial["context"].clone()
            row = db.execute("SELECT batch,windows FROM facts WHERE step=?",
                             (stage+64,)).fetchone()
            batch = unpack_batch(row[0], "cpu")
            windows = torch.from_numpy(np.frombuffer(row[1], dtype=np.int64).copy().reshape(-1, 4))
            recorded = next(c for c in credits if c["decision_step"] == stage)
            actual, altered = copy.deepcopy(base), copy.deepcopy(base)
            actual_credit = actual.settle_trial(batch, windows=windows)
            replay_error = abs(actual_credit["gain"]-recorded["gain"])
            if replay_error > .002:
                raise RuntimeError("Stored-window replay differs beyond float16 tolerance")
            counterfactual = tuple(t.clone() for t in batch)
            # Alter only the physical outcome labels, not the starting context,
            # candidate action, network or optimizer. This is an offline check.
            counterfactual[2][:, 4:7] = 1-counterfactual[2][:, 4:7]
            counterfactual[2][:, 7:9] *= -1
            altered_credit = altered.settle_trial(counterfactual, windows=windows)
            real_scores = selection_scores(actual, context)
            altered_scores = selection_scores(altered, context)
            score_delta = float(np.max(np.abs(real_scores-altered_scores)))
            if score_delta <= 1e-8:
                raise RuntimeError("Physical outcome changes have no learning-score effect")
            replays.append(dict(decision_step=stage, mode=recorded["mode"],
                recorded_gain=recorded["gain"], replay_gain=actual_credit["gain"],
                float16_replay_error=replay_error,
                counterfactual_gain=altered_credit["gain"],
                selection_score_max_difference=score_delta,
                actual_selected_mode=actual.select_mode(context, explore=False),
                counterfactual_selected_mode=altered.select_mode(context, explore=False)))
        result = dict(
            physical_events_checked=event_count,
            consequence_windows_matched=windows_checked,
            settled_choices=len(credits), completed_learning_trials=trials,
            negative_trials_retained=negatives,
            learning_rate_changes_reproduced=changes, final_learning_rate=rate,
            preference_counts_reproduced=counts.tolist(),
            preference_returns_reproduced=returns.tolist(),
            feedback_replays=replays,
            limitation="Offline mechanism sensitivity, not an improvement or generalization claim. "
                        "Changing scores need not change the winning action in every context.")
    finally:
        db.close()
    if digest(db_path) != expected_hash:
        raise RuntimeError("The audit changed the training database")
    result.update(training_database_sha256=expected_hash, training_database_unchanged=True)
    dump(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.run, args.output)
