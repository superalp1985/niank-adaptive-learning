"""Post-hoc lock persistence and unfinished recovery accounting."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def diagnose(path):
    pieces = []
    for file in sorted(path.glob("transitions_*.npz")):
        with np.load(file) as d:
            pieces.append({k: d[k].copy() for k in ("locked", "visible", "goal_id", "segment",
                           "contact", "collision", "impact", "done")})
    data = {k: np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
    worlds = data["locked"].shape[1]
    durations, completed, abandoned, occluded_gaps = [], [], [], []
    pending_at_end, ever_count = 0, 0
    for body in range(worlds):
        run = 0
        known = False
        gap_start = None
        gap_occluded = False
        previous_goal, previous_segment = None, None
        for step in range(len(data["locked"])):
            goal, segment = int(data["goal_id"][step, body]), int(data["segment"][step, body])
            if (goal, segment) != (previous_goal, previous_segment):
                if run:
                    durations.append(run)
                if gap_start is not None:
                    abandoned.append(step - gap_start)
                known, run, gap_start = False, 0, None
                previous_goal, previous_segment = goal, segment
            locked = bool(data["locked"][step, body])
            visible = bool(data["visible"][step, body])
            if locked:
                if not known:
                    ever_count += 1
                    known = True
                if gap_start is not None:
                    completed.append(step - gap_start)
                    if gap_occluded:
                        occluded_gaps.append(step - gap_start)
                    gap_start = None
                run += 1
            else:
                if run:
                    durations.append(run)
                    run = 0
                if known and gap_start is None:
                    gap_start, gap_occluded = step, not visible
                elif gap_start is not None:
                    gap_occluded |= not visible
        if run:
            durations.append(run)
        if gap_start is not None:
            pending_at_end += 1
    return dict(path=str(path.relative_to(ROOT)), recorded_steps=len(data["locked"]),
        lock_runs=len(durations), median_lock_run_steps=float(np.median(durations)) if durations else 0,
        lock_runs_at_least_10_steps=sum(v >= 10 for v in durations),
        goals_ever_attention_locked=ever_count,
        recovered_gaps=len(completed), abandoned_gaps=len(abandoned),
        pending_recovery_at_end=pending_at_end,
        recovered_gaps_that_included_occlusion=len(occluded_gaps),
        median_recovery_steps=float(np.median(completed)) if completed else 0,
        caveat="Post-hoc attention metrics; completion-conditioned latency can hide abandoned recovery.")


def main():
    results = {}
    for mode in ("fixed", "revisable"):
        path = ROOT / "pilot" / mode / "probe_16384"
        if not path.exists():
            continue
        results[mode] = diagnose(path)
    (ROOT / "pilot/LOCK_DIAGNOSTICS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
