"""Report fixed V20 behavior, practice curves and factual retrieval recurrence."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DT = .12
FINAL_STEP = 12288
STAGES = ((0, 4096), (4096, 8192), (8192, FINAL_STEP))
ANALYSIS_CONFIG = dict(
    schema_version=3, arms=["random", "retrieval"], final_step=FINAL_STEP,
    stages=[list(stage) for stage in STAGES], dt_seconds=DT,
    failure_followup_steps=32,
    failure_followup_eligibility="Full future window in the same segment, regardless of contact.",
    consequence_horizon=8, low_weight_threshold=.25,
    expected_worlds=72, expected_rollout=128, expected_epochs=2,
    expected_calibration_steps=1024, expected_segment_steps=1024,
    expected_factual_experiences_per_arm=884736,
    expected_ppo_presentations_per_arm=1769472,
    expected_replay_presentations_per_arm=147456,
    expected_replay_queries_per_arm=96, expected_replay_per_query=1536)
METRICS = ("safe_contact_per_minute", "contact_per_minute", "safe_contact_fraction",
           "collision_rate", "attacked_impact_rate", "prediction_mse")
COUNTERS = ("experiences", "update_presentations", "predictor_presentations",
            "consequence_presentations", "replay_presentations",
            "replay_policy_presentations", "local_update_batches",
            "directory_insertions", "retrieval_count", "reopened_accesses",
            "pending_unlabelled")
COLORS = {"random": "#b24762", "retrieval": "#008575"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def require_audit(output):
    audit = read_json(output / "AUDIT.json")
    if audit.get("complete") is not True or audit.get("source_verified") is not True:
        raise ValueError("Analysis requires a complete, source-verified AUDIT.json")
    return audit


def analysis_provenance(output):
    encoded_config = json.dumps(ANALYSIS_CONFIG, sort_keys=True, separators=(",", ":")).encode()
    return dict(analysis_script="tracking_v20/analyze.py",
                analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                analysis_config=ANALYSIS_CONFIG,
                analysis_config_sha256=hashlib.sha256(encoded_config).hexdigest(),
                training_manifest_sha256=hashlib.sha256(
                    (output / "manifest.json").read_bytes()).hexdigest(),
                audit_sha256=hashlib.sha256((output / "AUDIT.json").read_bytes()).hexdigest(),
                training_manifest_modified=False,
                counter_scope="Agent construction resets counters; state/load transfer weights only. "
                              "Shared calibration presentations are separate from each arm.")


def check_fixed_config(config):
    expected = dict(worlds=72, rollout=128, epochs=2, calibration_steps=1024,
                    replay_budget=1536, horizon=8, milestones=[4096, FINAL_STEP])
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Fixed V20 analysis expected {key}={value!r}")
    if set(config["arms"]) != {"random", "retrieval"} or len(config["arms"]) != 2:
        raise ValueError("Fixed V20 analysis requires the two original arms")


def pending_tail(path, horizon=8):
    needed = horizon - 1
    pieces, final_done, final_contact = [], None, None
    for source in reversed(sorted(path.glob("transitions_*.npz"))):
        with np.load(source, allow_pickle=False) as data:
            done, contact = data["done"].astype(bool), data["contact"].astype(bool)
            if final_done is None:
                final_done, final_contact = done[-1].copy(), contact[-1].copy()
            take = min(needed, len(done))
            pieces.append(done[-take:] | contact[-take:])
            needed -= take
        if not needed:
            break
    if final_done is None:
        raise ValueError(f"No final consequence-window log: {path}")
    closes = np.concatenate(list(reversed(pieces)))
    pending = np.zeros(closes.shape[1], np.int64)
    for terminal in closes:
        pending += 1
        pending[terminal] = 0
    return dict(pending_expected_from_tail=int(pending.sum()),
                pending_per_body=pending.tolist(),
                final_reset_bodies=int(final_done.sum()),
                final_contact_bodies=int(final_contact.sum()),
                tail_steps_inspected=len(closes),
                interpretation="Queues persist across rollout updates. Mature at eight observations "
                               "or truncate on factual contact/reset; no manual final flush.")


def validate_budget(config, counters, recurrence, audit, phases, pending, calibration,
                    calibration_pending):
    check_fixed_config(config)
    expected_facts = ANALYSIS_CONFIG["expected_factual_experiences_per_arm"]
    expected_ppo = ANALYSIS_CONFIG["expected_ppo_presentations_per_arm"]
    expected_replay = ANALYSIS_CONFIG["expected_replay_presentations_per_arm"]
    expected_queries = list(range(config["rollout"], FINAL_STEP + 1, config["rollout"]))
    validated = {}

    def equal(name, actual, expected):
        if actual != expected:
            raise ValueError(f"Budget mismatch {name}: actual={actual!r}, expected={expected!r}")

    for arm in config["arms"]:
        counter, retrieval, session = counters[arm], recurrence[arm], audit["sessions"][arm]
        counts = session["counts"]
        equal(f"{arm}.audited_steps", session["steps"], FINAL_STEP)
        equal(f"{arm}.audited_worlds", session["worlds"], config["worlds"])
        equal(f"{arm}.summary_facts", counter["experiences"], expected_facts)
        equal(f"{arm}.audited_facts", counts["events"], expected_facts)
        equal(f"{arm}.audited_evidence", counts["evidence"], expected_facts)
        equal(f"{arm}.npz_facts", sum(point["experiences"] for point in phases[arm]), expected_facts)
        equal(f"{arm}.ppo_presentations", counter["update_presentations"], expected_ppo)
        equal(f"{arm}.replay_presentations", counter["replay_presentations"], expected_replay)
        equal(f"{arm}.retrieval_summary", counter["retrieval_count"], expected_replay)
        equal(f"{arm}.retrieval_audit", counts["retrievals"], expected_replay)
        equal(f"{arm}.retrieval_logs", retrieval["visits"], expected_replay)
        equal(f"{arm}.audited_replay_batches", session["replay_batches"], len(expected_queries))
        query_counts = {int(time): count for time, count in retrieval["query_visit_counts"].items()}
        equal(f"{arm}.replay_query_times", sorted(query_counts), expected_queries)
        for time, count in query_counts.items():
            equal(f"{arm}.query_{time}_budget", count, config["replay_budget"])
        matured = int(counts["outcomes"])
        equal(f"{arm}.audited_labels_checked", session["consequence_labels_checked"], matured)
        equal(f"{arm}.pending_tail", counter["pending_unlabelled"],
              pending[arm]["pending_expected_from_tail"])
        equal(f"{arm}.facts_minus_matured_labels", expected_facts - matured,
              counter["pending_unlabelled"])
        equal(f"{arm}.consequence_presentations", counter["consequence_presentations"],
              matured + expected_replay)
        equal(f"{arm}.directory_insertions", counter["directory_insertions"], counts["directory_events"])
        equal(f"{arm}.low_weight_visits", counter["reopened_accesses"], retrieval["low_weight_visits"])
        equal(f"{arm}.orphaned_references", retrieval["orphaned_event_references"], 0)
        if not 0 <= counter["replay_policy_presentations"] <= expected_replay:
            raise ValueError(f"Invalid {arm} factual-imitation presentation count")
        validated[arm] = dict(factual_experiences=expected_facts, ppo_presentations=expected_ppo,
            replay_presentations=expected_replay, replay_queries=len(query_counts),
            replay_per_query=config["replay_budget"], matured_new_labels=matured,
            pending_unlabelled=counter["pending_unlabelled"],
            consequence_presentations=matured + expected_replay,
            factual_imitation_presentations=counter["replay_policy_presentations"],
            final_window=pending[arm], all_counts_verified=True)
    shared = config["calibration_steps"] * config["worlds"]
    calibration_session = audit["sessions"]["calibration"]
    calibration_counts = calibration_session["counts"]
    equal("calibration.steps", calibration["step"], config["calibration_steps"])
    equal("calibration.worlds", calibration["worlds"], config["worlds"])
    equal("calibration.audited_steps", calibration_session["steps"], config["calibration_steps"])
    equal("calibration.audited_worlds", calibration_session["worlds"], config["worlds"])
    equal("calibration.facts", calibration["experiences"], shared)
    equal("calibration.audited_facts", calibration_counts["events"], shared)
    equal("calibration.audited_evidence", calibration_counts["evidence"], shared)
    equal("calibration.labels_checked", calibration_session["consequence_labels_checked"],
          calibration_counts["outcomes"])
    equal("calibration.pending_tail", calibration["pending_unlabelled"],
          calibration_pending["pending_expected_from_tail"])
    equal("calibration.matured_plus_pending", calibration_counts["outcomes"] +
          calibration["pending_unlabelled"], shared)
    equal("calibration.consequence_presentations", calibration["consequence_presentations"],
          calibration_counts["outcomes"])
    equal("calibration.ppo_presentations", calibration["update_presentations"], 0)
    equal("calibration.replay_presentations", calibration["replay_presentations"], 0)
    equal("calibration.audited_retrievals", calibration_counts["retrievals"], 0)
    equal("calibration.audited_replay_batches", calibration_session["replay_batches"], 0)
    return dict(epochs=config["epochs"], replay_budget_per_rollout=config["replay_budget"],
        new_factual_experiences_per_arm=expected_facts, shared_calibration_experiences=shared,
        shared_calibration_matured_labels=int(calibration_counts["outcomes"]),
        shared_calibration_pending_unlabelled=int(calibration["pending_unlabelled"]),
        shared_calibration_consequence_presentations=int(calibration["consequence_presentations"]),
        shared_calibration_final_window=calibration_pending,
        shared_calibration_counter_carried_into_arm=False,
        unique_new_training_experiences=shared + len(config["arms"]) * expected_facts,
        ancestral_plus_new_per_arm=config["ancestor_experiences"] + shared + expected_facts,
        expected_ppo_presentations_per_arm=expected_ppo,
        expected_replay_presentations_per_arm=expected_replay,
        expected_replay_queries_per_arm=len(expected_queries),
        expected_replay_per_query=config["replay_budget"], verified_arms=validated,
        all_counts_verified=True)


def summary_metrics(summary):
    experiences = int(summary["experiences"])
    minutes = experiences * DT / 60
    if experiences <= 0:
        raise ValueError("Behavior summary has no observed body time")
    result = {key: float(summary[key]) for key in METRICS
              if key in summary and key != "safe_contact_per_minute"}
    result.update(contacts=int(summary["contacts"]), safe_contacts=int(summary["safe_contacts"]),
                  safe_contact_per_minute=summary["safe_contacts"] / minutes,
                  contact_per_minute=summary["contacts"] / minutes,
                  observed_body_minutes=minutes, experiences=experiences,
                  steps_per_body=int(summary["step"]), bodies=int(summary["worlds"]))
    result["collisions"] = int(round(summary["collision_rate"] * experiences))
    result["impacts"] = int(round(summary["impact_rate"] * experiences))
    result["attacked_steps"] = int(summary["attacked_steps"])
    result["attacked_impacts"] = int(round(summary["attacked_impact_rate"] *
                                          summary["attacked_steps"]))
    return result


def metrics_delta(left, right):
    return {key: left[key] - right[key] for key in METRICS}


def phase(path, start, end):
    fields = ("contact", "safe_contact", "collision", "impact", "visible", "locked")
    totals = dict.fromkeys(fields, 0)
    moments = dict.fromkeys(("pred_error", "progress", "retention_gate", "query_strength"), 0.)
    count = attacked_steps = attacked_impacts = covered_steps = 0
    body_contacts = body_safe = None
    for source in sorted(path.glob("transitions_*.npz")):
        with np.load(source, allow_pickle=False) as data:
            chunk_start = int(data["start_step"])
            chunk_steps, bodies = data["contact"].shape
            lo, hi = max(start - chunk_start, 0), min(end - chunk_start, chunk_steps)
            if hi <= lo:
                continue
            if chunk_start + lo != start + covered_steps:
                raise ValueError(f"Missing or overlapping transitions: {source}")
            if body_contacts is None:
                body_contacts = np.zeros(bodies, np.int64)
                body_safe = np.zeros(bodies, np.int64)
            covered_steps += hi - lo
            observations = (hi - lo) * bodies
            count += observations
            for key in fields:
                totals[key] += int(data[key][lo:hi].sum())
            body_contacts += data["contact"][lo:hi].sum(0).astype(np.int64)
            body_safe += data["safe_contact"][lo:hi].sum(0).astype(np.int64)
            attacked = data["combo"][lo:hi, :, 2] != 0
            attacked_steps += int(attacked.sum())
            attacked_impacts += int((attacked & data["impact"][lo:hi].astype(bool)).sum())
            for key in moments:
                moments[key] += float(data[key][lo:hi].sum(dtype=np.float64))
    if covered_steps != end - start or not count:
        raise ValueError(f"Incomplete phase {path}, {start}:{end}")
    minutes_per_body = covered_steps * DT / 60
    rates = body_safe / minutes_per_body
    return dict(start=start, end=end, steps_per_body=covered_steps,
                bodies=len(body_safe), experiences=count, totals=totals,
                observed_body_minutes=count * DT / 60,
                contact_per_minute=totals["contact"] * 60 / (count * DT),
                safe_contact_per_minute=totals["safe_contact"] * 60 / (count * DT),
                safe_contact_fraction=totals["safe_contact"] / max(totals["contact"], 1),
                collision_rate=totals["collision"] / count,
                impact_rate=totals["impact"] / count,
                attacked_steps=attacked_steps, attacked_impacts=attacked_impacts,
                attacked_impact_rate=attacked_impacts / max(attacked_steps, 1),
                visible_lock=totals["locked"] / max(totals["visible"], 1),
                prediction_mse=moments["pred_error"] / count,
                means={key: value / count for key, value in moments.items()},
                safe_contacts_per_body=body_safe.tolist(),
                contacts_per_body=body_contacts.tolist(),
                safe_contacts_per_body_minute=rates.tolist(),
                safe_rate_body_min=float(rates.min()), safe_rate_body_median=float(np.median(rates)),
                safe_rate_body_max=float(rates.max()))


def chunks(path):
    result = []
    for source in sorted(path.glob("transitions_*.npz")):
        with np.load(source, allow_pickle=False) as data:
            count = data["contact"].size
            attacked = data["combo"][:, :, 2] != 0
            result.append(dict(start=int(data["start_step"]),
                end=int(data["start_step"]) + len(data["contact"]), experiences=count,
                safe_contacts=int(data["safe_contact"].sum()),
                contacts=int(data["contact"].sum()),
                safe_contact_per_minute=float(data["safe_contact"].sum() * 60 / (count * DT)),
                contact_per_minute=float(data["contact"].sum() * 60 / (count * DT)),
                collision_rate=float(data["collision"].mean()),
                attacked_impact_rate=float((attacked & data["impact"].astype(bool)).sum() /
                                          max(int(attacked.sum()), 1)),
                prediction_mse=float(data["pred_error"].mean())))
    return result


def retrieval_recurrence(path):
    previous, recurrent_ids, low_weight_ids, low_weight_recurrent_ids = {}, set(), set(), set()
    same_query_duplicates = later_visits = low_weight_visits = referenced_rows = 0
    random_visits = failed_visits = 0
    distance_sum = lag_sum = 0.
    per_body, per_query = {}, {}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
        orphaned = database.execute(
            "SELECT COUNT(*) FROM retrievals r LEFT JOIN directory_events d "
            "ON d.body=r.body AND d.time=r.event_time WHERE d.time IS NULL").fetchone()[0]
        rows = database.execute(
            "SELECT query_time,body,event_time,event_segment,distance,random,"
            "weight_before,weight_after,failed_query FROM retrievals "
            "ORDER BY query_time,body,rowid")
        for query_time, body, event_time, segment, distance, random, before, after, failed in rows:
            event_id = (body, event_time)
            referenced_rows += 1
            random_visits += bool(random)
            failed_visits += bool(failed)
            distance_sum += distance
            per_body[body] = per_body.get(body, 0) + 1
            per_query[query_time] = per_query.get(query_time, 0) + 1
            if event_id in previous:
                previous_query = previous[event_id]
                if previous_query == query_time:
                    same_query_duplicates += 1
                else:
                    later_visits += 1
                    recurrent_ids.add(event_id)
                    lag_sum += query_time - previous_query
                    if event_id in low_weight_ids:
                        low_weight_recurrent_ids.add(event_id)
            if before < ANALYSIS_CONFIG["low_weight_threshold"]:
                low_weight_visits += 1
                low_weight_ids.add(event_id)
            previous[event_id] = query_time
    denominator = max(referenced_rows, 1)
    return dict(visits=referenced_rows, unique_referenced_events=len(previous),
                orphaned_event_references=int(orphaned), same_query_duplicate_visits=same_query_duplicates,
                later_query_revisits=later_visits, events_revisited_in_later_queries=len(recurrent_ids),
                later_revisit_fraction=later_visits / denominator,
                mean_later_revisit_delay_steps=lag_sum / max(later_visits, 1),
                low_weight_visits=low_weight_visits,
                unique_low_weight_visited_events=len(low_weight_ids),
                low_weight_events_revisited_in_later_queries=len(low_weight_recurrent_ids),
                low_weight_threshold=ANALYSIS_CONFIG["low_weight_threshold"],
                low_weight_access_definition="Weight_before < 0.25 at an ordinary permitted visit; "
                                             "the event was never disabled or moved to cold storage.",
                random_visits=random_visits, random_fraction=random_visits / denominator,
                failed_query_visits=failed_visits, failed_query_fraction=failed_visits / denominator,
                mean_key_distance=distance_sum / denominator,
                body_visit_counts={str(key): value for key, value in sorted(per_body.items())},
                query_visit_counts={str(key): value for key, value in sorted(per_query.items())},
                interpretation="Same historical event ID accessed again; not a new environmental "
                               "encounter and not evidence of causal behavioral recovery.")


def followup_kind(data, kind, window):
    failed = data[kind].astype(bool)
    starts = failed.copy()
    starts[1:] &= (~failed[:-1] | (data["segment"][1:] != data["segment"][:-1]) |
                   (data["goal_id"][1:] != data["goal_id"][:-1]))
    times, bodies = np.nonzero(starts)
    same_step_contact = data["contact"][times, bodies].astype(bool)
    same_step_reset = data["done"][times, bodies].astype(bool)
    active = ~(same_step_reset | same_step_contact)
    future_end = times + window
    eligible = active & (future_end < len(failed))
    ids = np.flatnonzero(eligible)
    eligible[ids] &= (data["segment"][future_end[ids], bodies[ids]] ==
                      data["segment"][times[ids], bodies[ids]])
    short_window = active & ~eligible
    observed = np.zeros(len(times), np.int16)
    contacted = np.zeros(len(times), bool)
    safe = np.zeros(len(times), bool)
    delays = np.zeros(len(times), np.int16)
    segments, goals = data["segment"][times, bodies], data["goal_id"][times, bodies]
    for offset in range(1, window + 1):
        within = times + offset < len(failed)
        ids = np.flatnonzero(active & within)
        active[~within] = False
        if not len(ids):
            continue
        now, body = times[ids] + offset, bodies[ids]
        same = ((data["segment"][now, body] == segments[ids]) &
                (data["goal_id"][now, body] == goals[ids]))
        active[ids[~same]] = False
        ids, now, body = ids[same], now[same], body[same]
        observed[ids] += 1
        contact = data["contact"][now, body].astype(bool)
        contacted[ids] |= contact
        safe[ids] |= contact & data["safe_contact"][now, body].astype(bool)
        delays[ids[contact]] = offset
        stopped = contact | data["done"][now, body].astype(bool)
        active[ids[stopped]] = False
    if np.any(eligible & ~((observed == window) | contacted)):
        raise ValueError("A full followup window ended without contact or a segment reset")
    if kind == "collision" and safe.any():
        raise ValueError("Same-goal safe contact after collision contradicts the world definition")
    eligible_contact = eligible & contacted
    eligible_safe = eligible & safe
    return dict(failure_episode_starts=int(len(times)),
                same_step_contact_at_episode_start=int(same_step_contact.sum()),
                same_step_reset_at_episode_start=int(same_step_reset.sum()),
                followup_eligible_episodes=int(eligible.sum()),
                censored_or_no_future_window=int((~eligible).sum()),
                episodes_with_later_first_contact=int(eligible_contact.sum()),
                later_first_contact_fraction=float(eligible_contact.sum() / max(int(eligible.sum()), 1)),
                mean_later_first_contact_delay_steps=float(delays[eligible_contact].mean())
                                                    if eligible_contact.any() else None,
                mean_later_first_contact_delay_seconds=float(delays[eligible_contact].mean() * DT)
                                                      if eligible_contact.any() else None,
                first_contacts_marked_safe=int(eligible_safe.sum()),
                safe_fraction_of_first_contacts=float(eligible_safe.sum() /
                                                      max(int(eligible_contact.sum()), 1)),
                short_window_episodes=int(short_window.sum()),
                short_window_later_first_contacts=int((short_window & contacted).sum()),
                eligibility_definition="Failure starts with a full future window in the same "
                                       "segment; eligibility does not depend on later contact. "
                                       "Short windows are separate, including their successes.",
                same_goal_post_collision_safe_contacts_zero_by_definition=kind == "collision")


def failure_followups(path, window=32):
    fields = ("contact", "safe_contact", "collision", "impact", "done", "segment", "goal_id")
    pieces = {key: [] for key in fields}
    for source in sorted(path.glob("transitions_*.npz")):
        with np.load(source, allow_pickle=False) as data:
            for key in fields:
                pieces[key].append(data[key])
    if not pieces["contact"]:
        raise ValueError(f"No failure-followup observations: {path}")
    data = {key: np.concatenate(values) for key, values in pieces.items()}
    if (data["safe_contact"].astype(bool) & ~data["contact"].astype(bool)).any():
        raise ValueError("Safe contact without a successful contact contradicts the world definition")
    return dict(window_steps=window, window_seconds=window * DT,
                by_failure={kind: followup_kind(data, kind, window) for kind in ("collision", "impact")},
                condition="Each type is counted separately from its consecutive-episode start. "
                          "Preselect starts with 32 future steps available in the same segment. "
                          "Observe within the same body and goal until first contact or reset; "
                          "report shorter windows separately, regardless of success. Types can overlap.",
                world_contact_definition="contact = target sensor contact and no same-step impact. "
                                         "safe_contact = contact and no collision since the last "
                                         "successful contact/reset. A collision taints the same goal "
                                         "until successful contact; impact alone does not persist "
                                         "in the safety flag.",
                interpretation="Primary outcome is the later first contact. Same-goal post-collision "
                               "safe contact must be zero by definition. No causal retrieval recovery claim.")


def scene_deltas(results):
    first, second = (results[arm][-1]["probe"]["combo_stats"] for arm in ("random", "retrieval"))
    result = {}
    for scene in first:
        left, right = first[scene], second[scene]
        result[scene] = dict(
            safe_contacts=int(right["safe_contacts"] - left["safe_contacts"]),
            contacts=int(right["contacts"] - left["contacts"]),
            safe_contact_per_minute=(right["safe_contacts"] * 60 / (right["steps"] * DT) -
                                     left["safe_contacts"] * 60 / (left["steps"] * DT)),
            collision_rate=right["collision_rate"] - left["collision_rate"],
            impact_rate=right["impact_rate"] - left["impact_rate"])
    return result


def plot_analysis(output, arms, milestones, development, practice):
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7), constrained_layout=True)
    titles = ("Safe contacts / body-minute", "Contacts / body-minute", "Safe contact fraction (%)",
              "Collision steps (%)", "Impact steps in attack scenes (%)", "Next RGB MSE")
    for axis, key, title in zip(axes.flat, METRICS, titles):
        scale = 100 if key.endswith("_rate") or key == "safe_contact_fraction" else 1
        for arm in arms:
            axis.plot(milestones, [point[key] * scale for point in development[arm]],
                      marker="o", color=COLORS[arm], label=arm)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Additional practice steps per body")
        axis.set_xticks(milestones)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("V20: development observations under equal replay budgets", fontsize=12)
    figure.savefig(output / "DEVELOPMENT_CURVES.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for axis, key, title in zip(axes, ("safe_contact_per_minute", "collision_rate",
                                      "attacked_impact_rate"),
                               ("Practice safe contacts / body-minute", "Practice collision steps (%)",
                                "Practice impact steps in attack scenes (%)")):
        scale = 1 if key == "safe_contact_per_minute" else 100
        for arm in arms:
            points = practice[arm]
            x = np.array([point["end"] for point in points])
            y = np.array([point[key] * scale for point in points])
            window = min(8, len(y))
            axis.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, "valid"),
                      color=COLORS[arm], label=arm)
        for boundary in (4096, 8192):
            axis.axvline(boundary, color="#777777", alpha=.3, linewidth=.8)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Additional practice steps per body")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    figure.suptitle("V20: 8-rollout moving averages of actual practice", fontsize=12)
    figure.savefig(output / "PRACTICE_CURVES.png", dpi=150)
    plt.close(figure)


def report(output, result):
    config = result["config"]
    development = result["development"]
    lines = ["# V20 行动后果检索与重学结果", "",
        "实验日期：2026-10-14。本报告固定比较 random 与 retrieval；完整失败和成功记录均保留。",
        "",
        f"共享后果头预热 {config['calibration_steps']:,} 步；每组 {FINAL_STEP:,} 步、"
        f"{config['worlds']} 个身体，PPO 每轮 {config['epochs']} 遍；每 {config['rollout']} 步"
        f"回放预算 {config['replay_budget']:,} 条。DT={DT} 秒，触达率以所有身体累计分钟为分母。",
        "",
        "## 开发观察的实际行为", "",
        "| 实践步数 | 组别 | 安全触达/触达 | 安全触达/身体分钟 | 碰撞步占比 | 攻击场景受击步占比 |",
        "|---:|---|---:|---:|---:|---:|"]
    for index, milestone in enumerate(result["milestones"]):
        for arm in config["arms"]:
            point = development[arm][index]
            lines.append(f"| {milestone:,} | {arm} | {point['safe_contacts']}/{point['contacts']} | "
                         f"{point['safe_contact_per_minute']:.4f} | {100 * point['collision_rate']:.3f}% | "
                         f"{100 * point['attacked_impact_rate']:.3f}% |")
    delta = result["final_retrieval_minus_random"]
    lines += ["", f"最终 retrieval 比 random 的安全触达/身体分钟差为 "
              f"{delta['safe_contact_per_minute']:+.4f}；碰撞差为 "
              f"{100 * delta['collision_rate']:+.3f} 个百分点；攻击场景受击差为 "
              f"{100 * delta['attacked_impact_rate']:+.3f} 个百分点。"]
    if result["baseline_reference"] is not None:
        baseline = result["baseline_reference"]
        lines += ["", f"同种子祖先策略基线（关闭后果引导）：安全触达 "
                  f"{baseline['safe_contacts']}/{baseline['contacts']}，"
                  f"{baseline['safe_contact_per_minute']:.4f}/身体分钟，碰撞 "
                  f"{100 * baseline['collision_rate']:.3f}%，攻击场景受击 "
                  f"{100 * baseline['attacked_impact_rate']:.3f}%。"]
        for arm in config["arms"]:
            change = result["final_minus_baseline"][arm]
            lines.append(f"{arm} 相对基线：安全触达/身体分钟 {change['safe_contact_per_minute']:+.4f}，"
                         f"碰撞 {100 * change['collision_rate']:+.3f} 个百分点，"
                         f"攻击场景受击 {100 * change['attacked_impact_rate']:+.3f} 个百分点。")
    lines += ["", "开发观察包括同一批 24 个已查看组合，每次 1,024 步；模型参数在观察期间固定，"
              "局部访问状态仍随事实变化。这些结果不能称独立泛化检验。",
              "", "## 三段实践", "",
              "| 组别 | 身体步范围 | 安全触达 | 安全触达/身体分钟 | 碰撞步占比 | 攻击场景受击步占比 |",
              "|---|---|---:|---:|---:|---:|"]
    for arm in config["arms"]:
        for point in result["phases"][arm]:
            lines.append(f"| {arm} | {point['start']:,}–{point['end']:,} | "
                         f"{point['totals']['safe_contact']:,} | {point['safe_contact_per_minute']:.4f} | "
                         f"{100 * point['collision_rate']:.3f}% | "
                         f"{100 * point['attacked_impact_rate']:.3f}% |")
    lines += ["", "## 事实与呈现预算", "",
              f"单组新增身体事实 {result['budget']['new_factual_experiences_per_arm']:,} 条，"
              f"共享预热事实 {result['budget']['shared_calibration_experiences']:,} 条；"
              f"单组祖先加本轮事实 {result['budget']['ancestral_plus_new_per_arm']:,} 条。"
              "重复回放是额外呈现，不计作新的环境事实。",
              f"审计、汇总和日志已逐项核对：每臂新事实 884,736、PPO 呈现 1,769,472、"
              "回放呈现 147,456；96 个查询时刻各 1,536 条。后果头呈现应等于"
              "成熟的新事实标签加回放呈现。未通过任何一项校验则不生成正式报告。",
              f"共享后果头预热呈现 {result['budget']['shared_calibration_consequence_presentations']:,} 次。"
              "保存和加载的 state 只有模块权重，没有计数；新建每臂 Agent 后计数从零开始，"
              "共享预热呈现没有混入每臂计数。"]
    for arm in config["arms"]:
        counters = result["training_counters"][arm]
        lines.append(f"{arm}：PPO 呈现 {counters['update_presentations']:,}，回放呈现 "
                     f"{counters['replay_presentations']:,}，其中正向事实效用的策略模仿呈现 "
                     f"{counters['replay_policy_presentations']:,}，后果头总呈现 "
                     f"{counters['consequence_presentations']:,}。")
        verified = result["budget"]["verified_arms"][arm]
        lines.append(f"{arm}：成熟的新事实标签 {verified['matured_new_labels']:,}，"
                     f"末尾待标注 {verified['pending_unlabelled']:,}；尾部日志中最后一步"
                     f"重置身体数 {verified['final_window']['final_reset_bodies']}。"
                     "待标注数通过审计标签差额和最后 7 步截断记录交叉核对，"
                     "没有把未完成窗口当成成熟标签。")
    lines += ["", "场景长度在 `world.py` 中固定为 1,024 步，12,288 步正好在第 12 个"
              "场景结尾；`done` 会关闭全部窗口。`advance` 之间窗口继续保留，"
              "程序没有手动清空末尾队列。模仿呈现由正向事实效用决定，允许两组不同，单列报告。"]
    lines += ["", "## 检索复查和失败后的接触", ""]
    for arm in config["arms"]:
        recurrence = result["retrieval_recurrence"][arm]
        followups = result["post_failure_followups"][arm]
        lines.append(f"{arm}：{recurrence['visits']:,} 次访问引用 "
                     f"{recurrence['unique_referenced_events']:,} 条历史事件；同一事件在后续查询轮"
                     f"再次访问 {recurrence['later_query_revisits']:,} 次，涉及 "
                     f"{recurrence['events_revisited_in_later_queries']:,} 条事件。访问前权重小于 0.25 的"
                     f"低权重访问 {recurrence['low_weight_visits']:,} 次；同一轮重复访问 "
                     f"{recurrence['same_query_duplicate_visits']:,} 次单列。")
        for kind, label in (("collision", "碰撞"), ("impact", "受击")):
            event = followups["by_failure"][kind]
            lines.append(f"{arm}：{label}新连续段 {event['failure_episode_starts']:,} 个；"
                         f"失败发生时已有完整后续 32 步场景预算的段 "
                         f"{event['followup_eligible_episodes']:,} 个，其中同一身体、场景和目标内"
                         f"后来首次触达 {event['episodes_with_later_first_contact']:,} 个"
                         f"（{100 * event['later_first_contact_fraction']:.2f}%），"
                         f"触达中的安全标记 {event['first_contacts_marked_safe']:,} 个。"
                         f"发生{label}的同一步已经触达 {event['same_step_contact_at_episode_start']:,} 个，"
                         "不混入后续首次触达分母。")
            lines.append(f"{arm}：{label}后不足 32 步的短窗口另有 "
                         f"{event['short_window_episodes']:,} 个，其中后来首次触达 "
                         f"{event['short_window_later_first_contacts']:,} 个；成功与失败均不进入主比例。")
    lines += ["", "低权重事件始终允许访问，0.25 阈值只是离线计数组；这些记录不表示关闭后重开，"
              "也不是从冷库恢复。", "",
              "安全触达定义是成功触达且上次触达以来没有碰撞。碰撞标记持续到成功触达，"
              "因此碰撞后同一目标的安全触达按定义必为零，不能用这个零值评价恢复能力。"
              "受击标记只代表当前步，受击后同一目标仍可能安全触达。碰撞和受击连续段分开统计，"
              "两类允许重叠；本小节次要跟踪指标使用后来首次成功触达，窗口截断在首次触达或重置。",
              "", "再次引用历史事件表示复查；不代表环境中的新相似遭遇，也不证明检索导致行为恢复。"
              "失败后触达统计同样只是实际后续事件，不能单独归因于检索。",
              "", "## 结论范围", "",
              "本轮检验有界事件目录、真实执行动作后果学习和有限引导后的实际行为差异。"
              "单个共享初始化、两条实践轨迹、已反复查看的开发场景，不支持确认性显著性判断。"
              "后果预测受当前行为轨迹影响，未执行动作没有已验证反事实标签。"
              "这些记录不证明形成了生物层级结构或真实生物动力学。", "",
              "图表：`DEVELOPMENT_CURVES.png`、`PRACTICE_CURVES.png`。"
              "机器可读计数、逐场景差值和观察曲线见 `ANALYSIS.json`。",
              "源与后果窗口审计见 `AUDIT.json`。"
              "分析脚本 SHA-256、固定分析配置、训练 manifest 摘要和审计摘要另存于 "
              "`ANALYSIS_MANIFEST.json`，原训练 manifest 保持原样。", ""]
    (output / "REPORT_V20_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    current_config = read_json(ROOT / "config.json")
    output = (args.output or ROOT / current_config["stage"]).resolve()
    if not (output / "COMPLETE.json").is_file():
        raise FileNotFoundError("Both fixed arms must finish before analysis")
    completion = read_json(output / "COMPLETE.json")
    if completion.get("complete") is not True:
        raise ValueError("Both fixed arms must finish before analysis")
    audit = require_audit(output)
    manifest = read_json(output / "manifest.json")
    config = manifest["config"]
    check_fixed_config(config)
    if sorted(completion.get("arms", [])) != sorted(config["arms"]):
        raise ValueError("Completion record does not contain both fixed arms")
    milestones = [0, *config["milestones"]]
    results, development, curves, development_chunks, phases = {}, {}, {}, {}, {}
    counters, recurrence, followups, pending = {}, {}, {}, {}
    for arm in config["arms"]:
        path = output / arm
        results[arm] = [read_json(path / f"result_{step}.json") for step in milestones]
        if any(int(results[arm][0][key]) != 0 for key in COUNTERS):
            raise ValueError(f"Arm counters must start from zero after weight-only load: {arm}")
        summary = read_json(path / "summary.json")
        if int(summary["step"]) != FINAL_STEP or int(summary["worlds"]) != config["worlds"]:
            raise ValueError(f"Unfinished arm: {arm}")
        counters[arm] = {key: int(summary[key]) for key in COUNTERS}
        development[arm] = [summary_metrics(result["probe"]) for result in results[arm]]
        curves[arm] = chunks(path)
        development_chunks[arm] = {str(step): chunks(path / f"probe_{step}")
                                   for step in milestones}
        phases[arm] = [phase(path, lo, hi) for lo, hi in STAGES]
        recurrence[arm] = retrieval_recurrence(path / "events.sqlite")
        followups[arm] = failure_followups(path)
        pending[arm] = pending_tail(path, config["horizon"])
    calibration = read_json(output / "calibration" / "summary.json")
    calibration_pending = pending_tail(output / "calibration", config["horizon"])
    budget = validate_budget(config, counters, recurrence, audit, phases, pending,
                             calibration, calibration_pending)
    provenance = analysis_provenance(output)
    baseline_path = output / "baseline_reference.json"
    baseline = summary_metrics(read_json(baseline_path)) if baseline_path.is_file() else None
    final = {arm: development[arm][-1] for arm in config["arms"]}
    result = dict(config=config, milestones=milestones, dt_seconds=DT,
        results=results, development=development, development_chunk_curves=development_chunks,
        practice_chunk_curves=curves, phases=phases, training_counters=counters, budget=budget,
        initial_behavior_identical=all(development["random"][0][key] ==
                                       development["retrieval"][0][key] for key in METRICS),
        final_retrieval_minus_random=metrics_delta(final["retrieval"], final["random"]),
        final_scene_deltas=scene_deltas(results), baseline_reference=baseline,
        final_minus_baseline={arm: metrics_delta(final[arm], baseline) for arm in config["arms"]}
                             if baseline else None,
        retrieval_recurrence=recurrence, post_failure_followups=followups,
        audit=audit, analysis_provenance=provenance,
        limitations=["One shared initialization and one practice trajectory per arm.",
                     "The 24 development combinations have been repeatedly inspected.",
                     "Outcome labels are factual executed-action targets under the current policy.",
                     "Retrieval recurrence and post-failure contacts do not prove causal recovery.",
                     "No evidence here establishes biological hierarchy or biological dynamics."])
    write_json(output / "ANALYSIS.json", result)
    write_json(output / "ANALYSIS_MANIFEST.json", provenance)
    plot_analysis(output, config["arms"], milestones, development, curves)
    report(output, result)
    print(json.dumps(dict(output=str(output), final=final,
                          final_retrieval_minus_random=result["final_retrieval_minus_random"],
                          budget=budget), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
