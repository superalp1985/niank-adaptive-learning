"""Read completed V24 records; keep activity, prediction and task outcomes distinct."""
import argparse
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracking_v23.persistence import atomic_json as dump


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def open_records(path):
    return sqlite3.connect((Path(path)/"events.sqlite").resolve().as_uri()+"?mode=ro",
                           uri=True)


def physical_tally():
    return dict(frames=0, contact=0, safe_contact=0, collision_frames=0,
                collision_seconds=0., impact_events=0, head_travel=0.,
                moving_frames=0, progress=0.)


def finish_tally(tally):
    frames = tally["frames"]
    return dict(tally, collision_fraction=tally["collision_frames"]/max(frames, 1),
                head_motion_fraction=tally["moving_frames"]/max(frames, 1),
                contacts_per_body_minute=tally["contact"]/max(frames*.01/60, 1e-9),
                safe_contacts_per_body_minute=tally["safe_contact"]/max(frames*.01/60, 1e-9))


def physical_process(db, training, width):
    origin, steps, bodies = (training[key] for key in ("origin_step", "new_steps", "bodies"))
    quarters = [physical_tally() for _ in range((steps+width-1)//width)]
    by_body = [physical_tally() for _ in range(bodies)]
    combinations, episodes = [set() for _ in range(bodies)], [set() for _ in range(bodies)]
    for step, body, raw in db.execute(
            "SELECT step,body,facts FROM events WHERE step>? AND step<=? ORDER BY step,body",
            (origin, origin+steps)):
        event = json.loads(raw)
        interval = quarters[(step-origin-1)//width]
        for target in (interval, by_body[body]):
            target["frames"] += 1
            target["contact"] += int(event["contact"])
            target["safe_contact"] += int(event["safe_contact"])
            target["collision_frames"] += int(event["collision"])
            target["collision_seconds"] += event["collision_duration"]
            target["impact_events"] += event["impact_count"]
            target["head_travel"] += event["head_travel"]
            target["moving_frames"] += int(abs(event["head_velocity"]) > 1e-9)
            target["progress"] += event["progress"]
        combinations[body].add(tuple(event["combo"]))
        episodes[body].add(event["segment"])
    if sum(item["frames"] for item in quarters) != steps*bodies:
        raise AssertionError("Incomplete training-only physical records")
    points = {point["step"]: point for point in training["curve"]}
    fresh = training["continuation"]
    start_deaths = training["deaths"]-fresh["deaths"]
    previous_deaths = start_deaths
    intervals = []
    for index, tally in enumerate(quarters):
        low, high = index*width, min((index+1)*width, steps)
        point = points[origin+high]
        item = dict(finish_tally(tally), new_steps_from=low, new_steps_to=high,
                    deaths=point["deaths"]-previous_deaths)
        previous_deaths = point["deaths"]
        choices = [json.loads(row[0]) for row in db.execute(
            "SELECT outcome FROM choices WHERE step>? AND step<=?",
            (origin+low, origin+high))]
        trials = [choice for choice in choices if choice["attempted_updates"]]
        item["learning"] = dict(
            settled_choices=len(choices),
            mode_counts=[sum(choice["mode"] == mode for choice in choices) for mode in range(3)],
            attempted_trials=len(trials),
            negative_trials_retained=sum(choice["gain"] < 0 for choice in trials),
            predictor_updates=sum(choice["retained_updates"] for choice in choices),
            mean_prediction_gain=(float(np.mean([choice["gain"] for choice in trials]))
                                  if trials else None),
            scope="Grouped by credit settlement; prediction gain is not behavioral utility.")
        intervals.append(item)
    for event_key, summary_key in (
            ("contact", "contact"), ("safe_contact", "safe_contact"),
            ("collision_frames", "collision"), ("impact_events", "impact_events")):
        if sum(item[event_key] for item in intervals) != fresh[summary_key]:
            raise AssertionError("Physical records differ from summary: "+event_key)
    if sum(item["deaths"] for item in intervals) != fresh["deaths"]:
        raise AssertionError("Death counts differ from training-only summary")
    return dict(intervals=intervals, by_body=[
        dict(finish_tally(item), body=i, encountered_combinations=sorted(combinations[i]),
             encountered_episode_count=len(episodes[i]))
        for i, item in enumerate(by_body)])


def attention_process(db, origin, steps, bodies, width):
    rows = db.execute(
        "SELECT step,body,head_action,yaw,velocity,entropy,attention_change,attention "
        "FROM vision WHERE step>? AND step<=? ORDER BY step,body",
        (origin, origin+steps)).fetchall()
    if not rows:
        raise AssertionError("Missing attention and head decisions")
    distributions = np.stack([np.frombuffer(row[7], np.float16).astype(np.float32)
                              for row in rows])
    sums = distributions.sum(1)
    if not np.isfinite(distributions).all() or not np.allclose(sums, 1, atol=.001):
        raise AssertionError("Invalid attention probabilities")
    distributions /= sums[:, None]
    relative = np.array([row[0]-origin for row in rows])
    body = np.array([row[1] for row in rows])
    command = np.array([row[2] for row in rows])
    yaw = np.array([row[3] for row in rows])
    velocity = np.array([row[4] for row in rows])
    entropy = np.array([row[5] for row in rows])
    change = np.array([row[6] for row in rows])
    limited = np.abs(yaw) >= np.pi/2-1e-6
    outward = (((command == 1) & (yaw >= np.pi/2-1e-6))
               | ((command == 2) & (yaw <= -np.pi/2+1e-6)))
    multiplier = .25+.75*64*distributions

    def values(mask):
        return dict(decisions=int(mask.sum()), entropy_mean=float(entropy[mask].mean()),
                    attention_change_l1_mean=float(change[mask].mean()),
                    multiplier_top_mean=float(multiplier[mask].max(1).mean()),
                    head_limit_fraction=float(limited[mask].mean()),
                    sampled_preaction_motion_fraction=float((np.abs(velocity[mask]) > 1e-9).mean()),
                    outward_command_at_limit_fraction=float(outward[mask].mean()),
                    head_action_counts=np.bincount(command[mask], minlength=3).tolist())

    intervals = [dict(values((relative > low) & (relative <= min(low+width, steps))),
                      new_steps_from=low, new_steps_to=min(low+width, steps))
                 for low in range(0, steps, width)]
    trace = (body == 0) & (np.arange(len(body)) % (bodies*max(1, steps//8000)) == 0)
    return dict(intervals=intervals,
                by_body=[dict(values(body == i), body=i) for i in range(bodies)],
                total_decisions=len(rows), entropy_uniform=float(np.log(64)),
                note="Yaw/velocity precede the command; activity is not evidence of utility."), dict(
                    seconds=relative[trace]*.01, multiplier=multiplier[trace],
                    yaw=yaw[trace]*180/np.pi, entropy=entropy[trace])


def delayed_process(db, origin, steps, width):
    rows = [dict(json.loads(raw), step=step) for step, raw in db.execute(
        "SELECT step,result FROM delayed_updates ORDER BY step")]
    if not rows:
        return dict(updates=0, samples=0, horizons={}, intervals=[])
    statistics, zero_loss_sum, scalar_count = {}, 0., 0
    for horizon, start, end, action, raw in db.execute(
            "SELECT horizon,start,end,action,target FROM delayed_outcomes ORDER BY end"):
        target = np.frombuffer(raw, np.float32).astype(np.float64)
        absolute = np.abs(target)
        loss = np.where(absolute < 1, .5*absolute**2, absolute-.5)
        zero_loss_sum += float(loss.sum())
        scalar_count += len(target)
        item = statistics.setdefault(horizon, dict(
            targets=0, truncated=0, duration_sum=0., sums=np.zeros(7),
            positives=np.zeros(7), zero_loss_sum=0., action_counts=np.zeros(27, np.int64)))
        item["targets"] += 1
        item["truncated"] += end-start < horizon
        item["duration_sum"] += (end-start)*.01
        item["sums"] += target
        item["positives"] += target > 0
        item["zero_loss_sum"] += float(loss.sum())
        item["action_counts"][action] += 1
    horizons = {}
    for horizon, item in statistics.items():
        count = item["targets"]
        horizons[str(horizon)] = dict(
            targets=count, terminal_truncated=item["truncated"],
            mean_realized_seconds=item["duration_sum"]/count,
            mean_target=(item["sums"]/count).tolist(),
            positive_target_fraction=(item["positives"]/count).tolist(),
            joint_action_counts=item["action_counts"].tolist(),
            zero_predictor_smooth_l1=item["zero_loss_sum"]/(count*7))
    intervals = []
    for low in range(0, steps+width, width):
        high = min(low+width, steps) if low < steps else steps+512
        selected = [row for row in rows if origin+low < row["step"] <= origin+high]
        samples = sum(row["samples"] for row in selected)
        intervals.append(dict(
            new_steps_from=low, new_steps_to=high, settling_only=low >= steps,
            updates=len(selected), samples=samples,
            sample_weighted_preupdate_loss=(
                sum(row["loss"]*row["samples"] for row in selected)/samples if samples else None)))
    samples = sum(row["samples"] for row in rows)
    if samples != sum(item["targets"] for item in statistics.values()):
        raise AssertionError("Auxiliary updates and target counts differ")
    return dict(updates=len(rows), samples=samples, horizons=horizons, intervals=intervals,
                sample_weighted_preupdate_loss=sum(row["loss"]*row["samples"] for row in rows)/samples,
                zero_predictor_smooth_l1=zero_loss_sum/scalar_count,
                target_order=["energy_change", "integrity_change", "collision_time_fraction",
                              "impacts", "contacts", "safe_contacts", "terminal"],
                note="Windows overlap. Online loss mixes horizons and changing representations; "
                     "a lower loss is not evidence of better gaze credit.")


def probe(probe_result):
    frozen = probe_result["frozen"]
    return dict(
        **{key: frozen[key] for key in (
            "contact", "safe_contact", "collision_fraction", "impact_events", "deaths", "progress")},
        vision=frozen["vision"], scenes=probe_result["scenes"],
        learned_parameters_unchanged=probe_result["learned_parameters_unchanged"],
        input_dependence=probe_result["input_dependence"])


def figures(root, results, traces):
    colors = {"control": "#287a84", "delayed": "#b44860"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, result in results.items():
        intervals = result["practice"]["intervals"]
        time = np.array([row["new_steps_to"] for row in intervals])*.01
        color = colors[name]
        axes[0, 0].plot(time, [row["safe_contact"] for row in intervals], "o-",
                        label=name, color=color)
        axes[0, 1].plot(time, [row["collision_fraction"] for row in intervals], "o-",
                        label=name, color=color)
        axes[1, 0].plot(time, [row["impact_events"] for row in intervals], "o-",
                        label=name, color=color)
        stages = result["stage_probes"]
        keys = sorted(map(int, stages))
        axes[1, 1].plot(np.array(keys)*.01, [stages[str(k)]["safe_contact"] for k in keys],
                        "o-", label=name, color=color)
    for axis, title, ylabel in zip(axes.flat,
            ("Practice: safe contacts per quarter", "Practice: collision fraction",
             "Practice: impact events per quarter", "Frozen common scenes: safe contacts"),
            ("Contacts (8 bodies)", "Collision frames / all frames", "Impact events",
             "Contacts / 5.12 s per body")):
        axis.set(title=title, ylabel=ylabel, xlabel="New practice seconds per body")
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("V24: same complete source state, additional practice and delayed gaze feedback")
    figure.savefig(root/"learning_process.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(3, 2, figsize=(13, 8), constrained_layout=True)
    for column, (name, trace) in enumerate(traces.items()):
        axes[0, column].imshow(trace["multiplier"].T, origin="lower", aspect="auto",
            extent=(trace["seconds"][0], trace["seconds"][-1], 0, 360), cmap="viridis",
            vmin=.25, vmax=5)
        axes[0, column].set(title=name+": body 0 feature multiplier (clipped at 5)",
                            ylabel="Body azimuth (degrees)")
        axes[1, column].plot(trace["seconds"], trace["yaw"], linewidth=.6, color=colors[name])
        axes[1, column].set(ylabel="Relative head yaw (degrees)", ylim=(-95, 95))
        axes[2, column].plot(trace["seconds"], trace["entropy"], linewidth=.6, color=colors[name])
        axes[2, column].axhline(np.log(64), linestyle="--", color="#777777")
        axes[2, column].set(ylabel="Spatial attention entropy",
                            xlabel="New practice seconds per body")
    figure.savefig(root/"attention_head_process.png", dpi=160)
    plt.close(figure)


def summarize(root):
    root = Path(root).resolve()
    if not (root/"COMPLETE.json").exists():
        raise RuntimeError("Analysis requires a completed pair; do not report partial runs as final")
    results, traces = {}, {}
    for name in ("control", "delayed"):
        arm = root/name
        complete = read(arm/"COMPLETE.json")
        training = read(arm/"training_only_summary.json")
        if training["new_steps"] != complete["config"]["additional_steps"]:
            raise AssertionError("Training-only boundary differs")
        db = open_records(arm)
        try:
            physical = physical_process(db, training, 8192)
            attention, traces[name] = attention_process(
                db, training["origin_step"], training["new_steps"], training["bodies"], 8192)
            delayed = delayed_process(db, training["origin_step"], training["new_steps"], 8192)
        finally:
            db.close()
        expected_decisions = training["new_steps"]//complete["config"]["action_every"]*training["bodies"]
        if attention["total_decisions"] != expected_decisions:
            raise AssertionError("Missing training attention decisions")
        stage_probes = {key: probe(value) for key, value in complete["stages"].items()}
        final = probe(complete["final_probe"])
        diagnostics = {key: probe(value) for key, value in complete["diagnostics"].items()}
        results[name] = dict(
            source_runtime=complete["source_runtime"],
            new_training_steps_per_body=training["new_steps"],
            new_training_seconds_per_body=training["new_steps"]*.01,
            new_training=training["continuation"], practice=physical, attention=attention,
            delayed=delayed, stage_probes=stage_probes, independent_final_probe=final,
            diagnostics=diagnostics,
            frozen_checks_all_passed=all(value["learned_parameters_unchanged"] for value in
                list(stage_probes.values())+[final]+list(diagnostics.values())),
            settling_steps_per_body=complete["training"]["new_steps"]-training["new_steps"],
            final_delayed_updates=complete["training"]["delayed"]["updates"],
            final_predictor_updates=complete["training"]["continuation"]["updates"])
    if results["control"]["source_runtime"] != results["delayed"]["source_runtime"]:
        raise AssertionError("Branches do not share a complete source state")
    first = [results[name]["stage_probes"]["0"] for name in results]
    for key in ("contact", "safe_contact", "collision_fraction", "impact_events", "deaths", "progress"):
        if first[0][key] != first[1][key]:
            raise AssertionError("Zero-initialized branches differ at their frozen start: "+key)
    dump(root/"measurements.json", results)
    figures(root, results, traces)
    print(json.dumps({name: dict(training=value["new_training"],
                                final_probe={key: value["independent_final_probe"][key]
                                    for key in ("contact", "safe_contact", "collision_fraction",
                                                "impact_events")},
                                delayed_updates=value["final_delayed_updates"])
                      for name, value in results.items()}), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
