"""Descriptive paired scores and observable cache/retrieval behavior."""

import json
import sys

import numpy as np

from audit import join
from core import ROOT
from run import dump


def behavior(path, delay):
    decisions = join(path.glob("decisions_*.npz"))
    count = decisions["ops"].shape[1]
    recovered, read_count, critical_both, critical_total = 0, 0, 0, 0
    for index, step in enumerate(decisions["step"]):
        ids, live = decisions["directory_ids"][index], decisions["live"][index].copy()
        ops, ptr = decisions["ops"][index], decisions["pointers"][index]
        event_times = ids // count
        for row in range(count):
            slot = ptr[row]
            if ops[row] == 1:
                read_count += 1
                recovered += int(step - event_times[row, slot] >= 12)
                if live[row].sum() >= 3:
                    old = np.flatnonzero(live[row])[np.argmin(event_times[row, live[row]])]
                    live[row, old] = False
                live[row, slot] = True
            elif ops[row] == 5:
                live[row, slot] = False
        if step >= 48 and step % 48 == delay:
            cycle = (step // 48) * 48
            for row in range(count):
                active_times = set(event_times[row, live[row]].tolist())
                critical_both += int(cycle in active_times and cycle + 4 in active_times)
                critical_total += 1
    return dict(retrievals=read_count, retrievals_age_at_least_12=recovered,
                both_cues_available_at_change=critical_both,
                change_events=critical_total,
                both_cues_fraction=critical_both / max(critical_total, 1))


def main(stage):
    root = ROOT / stage
    results = json.loads((root / "results.json").read_text())
    rows = []
    for result in results:
        path = root / f"delay{result['delay']}_{result['mode']}" / f"probe_{result['step']}"
        row = dict(mode=result["mode"], delay=result["delay"], steps=result["step"],
                   new_experiences=result["totals"]["new_experiences"],
                   parameters=result["total_parameters"], used_groups=result["used_groups"],
                   changed_groups=result["changed_input_groups"],
                   estimated_total_macs=result["estimated_total_macs"],
                   replay_presentations=result["totals"]["replay_presentations"])
        for name in ("seen", "composition"):
            probe = result["probes"][name]
            row[name] = {key: probe[key] for key in ("success", "critical_success", "mse", "critical_mse", "regret")}
            row[name]["behavior"] = behavior(path / name, result["delay"])
        rows.append(row)
    contrasts = []
    for delay in sorted({r["delay"] for r in rows}):
        for steps in sorted({r["steps"] for r in rows}):
            pair = {r["mode"]: r for r in rows if r["delay"] == delay and r["steps"] == steps}
            contrasts.append(dict(
                delay=delay, steps=steps,
                composition_delta_pp=100 * (pair["addressed"]["composition"]["success"] - pair["pooled"]["composition"]["success"]),
                critical_delta_pp=100 * (pair["addressed"]["composition"]["critical_success"] - pair["pooled"]["composition"]["critical_success"])))
    summary = dict(rows=rows, contrasts=contrasts,
                   interpretation="One seed per condition; descriptive exploration, not independent-sample significance.",
                   within_delay_only="Physical risk varies with phase; cross-delay differences are not a pure delay intervention.")
    dump(root / "SUMMARY.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
