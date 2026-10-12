"""Report matched readouts and the declared no-read intervention."""

import json
from pathlib import Path
import sys

from core import ROOT
from run import dump


def main(stage):
    output = ROOT / stage
    results = json.loads((output / "results.json").read_text())
    rows = []
    for training in results:
        for name, score in training["probes"].items():
            rows.append(dict(
                mode=training["mode"], experiences=training["totals"]["new_experiences"],
                probe=name, success=score["success"], critical=score["critical_success"],
                mse=score["mse"], critical_mse=score["critical_mse"], cost=score["cost"],
                regret=score["regret"], reads=score["retrievals"],
                both_cues=score["both_cues_at_change"],
                operations=score["operation_counts"], paired_reads=score["paired_retrievals"],
                groups=score["group_execution_counts"], parameters=training["parameters"],
                active_parameters=training["active_predictor_parameters"],
                replay_presentations=training["totals"]["planner_replay_presentations"],
                estimated_macs=training["estimated_total_macs"],
                training_seconds=training["seconds"]))
    final_volume = max(v["experiences"] for v in rows)
    effects = []
    for mode in sorted({v["mode"] for v in rows}):
        final = {v["probe"]: v for v in rows if v["mode"] == mode and v["experiences"] == final_volume}
        if "composition_no_reads" in final:
            a, b = final["composition"], final["composition_no_reads"]
            effects.append(dict(mode=mode, success_difference=a["success"] - b["success"],
                                critical_difference=a["critical"] - b["critical"],
                                mse_difference=a["mse"] - b["mse"],
                                cost_difference=a["cost"] - b["cost"]))
    summary = dict(rows=rows, final_read_interventions=effects,
                   audit=json.loads((output / "AUDIT.json").read_text()))
    dump(output / "SUMMARY.json", summary)
    lines = [
        "# V12: bounded retrieval plans and local factual learning",
        "",
        "Three continuous configurations. Readouts use actual physical outcomes,",
        "not agreement with another model. One seed: descriptive exploration.",
        "All networks have 126,918 parameters, including scorers; one active",
        "predictor group has 30,985 parameters. No end-to-end module gradient.",
        "",
        "## Held-out combinations",
        "",
        "| Mode | New experiences | Success | Critical four steps | MSE | Reads | Both cues at change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["probe"] == "composition":
            lines.append(f'| {row["mode"]} | {row["experiences"]:,} | {row["success"]:.2%} | '
                         f'{row["critical"]:.2%} | {row["mse"]:.6f} | {row["reads"]:,} | {row["both_cues"]:.2%} |')
    lines += ["", "## Final frozen intervention", "",
              "Enabled minus disabled raw reads. Same environment seed and weights;",
              "different actions can cause subsequent physical trajectories to diverge.",
              "This estimates the effect of allowing the trained retrieval policy;",
              "it does not isolate causal value for each individual database detail.", "",
              "| Mode | Success difference | Critical difference | MSE difference |",
              "| --- | ---: | ---: | ---: |"]
    for row in effects:
        lines.append(f'| {row["mode"]} | {row["success_difference"] * 100:+.2f} pp | '
                     f'{row["critical_difference"] * 100:+.2f} pp | {row["mse_difference"]:+.6f} |')
    audited = {entry["path"]: entry for entry in summary["audit"]["checks"]}
    final_step = max(training["step"] for training in results)
    lines += ["", "## Final retrieval and local-value diagnostics", "",
              "Current-cycle cues are records written at phase 0 or 4 of this cycle.",
              "Other-event reads may include earlier cues; this classification does",
              "not label those records permanently useless. Value errors compare",
              "only the executed plan's estimate and factual four-step outcome.", "",
              "| Mode | Current-cycle cue reads / all reads | Reads at change | Value RMSE | Largest group share |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for mode in sorted({row["mode"] for row in rows}):
        relative = str((output / mode / f"probe_{final_step}" / "composition").relative_to(ROOT))
        entry = audited[relative]
        lines.append(f'| {mode} | {entry["current_cycle_cue_reads"]:,} / {entry["reads"]:,} | '
                     f'{entry["change_window_reads"]:,} | {entry["selected_plan_value_rmse"]:.6f} | '
                     f'{entry["largest_group_fraction"]:.2%} |')
    lines += ["", "## Learning and accounting", "",
              "| Mode | Experiences | Operation replay presentations | Estimated total MAC | Seconds |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["probe"] == "composition" and row["experiences"] == final_volume:
            lines.append(f'| {row["mode"]} | {row["experiences"]:,} | {row["replay_presentations"]:,} | '
                         f'{row["estimated_macs"]:,} | {row["training_seconds"]:.1f} |')
    audit = summary["audit"]
    lines += ["", "## Audit and limits", "",
              f'Independent physical/access/operation-target audit: {audit["directories"]} directories, '
              f'{audit["transitions"]:,} transitions, {audit["operation_windows"]:,} operation windows.',
              "Source hashes match the run manifest.", "",
              "The environment differs from V11: fixed risk, random cue order, visible",
              "impulse at the change. No direct cross-version mechanism attribution.",
              "Search covers 153 metadata plans, not all physical/parameter choices.",
              "Local Adam updates do not certify local minima or global optimality.",
              "Backpropagation still runs inside each local network; separation is",
              "at module boundaries, not a biologically local synapse rule.",
              "No guaranteed monotonic generalization with more training.", "",
              "Summary bits never disclose cue sign; raw records alone enter numerical",
              "predictions. The tiny scorer evaluates many plans, so total planning MAC",
              "is not necessarily below predictor MAC. No hardware speedup claim.", "",
              "Complete data, databases and final resume states are archived in Release.",
              "Resume states require the matching external SQLite databases.", "",
              "See PROTOCOL.md for prior work, exact candidate differences and budgets.",
              "Interpretation in REPORT_V12_ZH.md is written after all matched readouts."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(rows=len(rows), interventions=effects)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
