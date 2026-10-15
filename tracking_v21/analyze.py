"""Read completed factual runs, audit credit, and report trajectories."""

import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from tracking_v17.run import dump
from tracking_v20.audit import session_audit
from tracking_v20.learner import outcome_utility

ROOT = Path(__file__).resolve().parent


def memory_audit(path):
    db = sqlite3.connect(path / "events.sqlite")
    try:
        total = db.execute("SELECT COUNT(*) FROM memory_catalog").fetchone()[0]
        facts = db.execute("SELECT COUNT(*) FROM directory_events").fetchone()[0]
        missing = db.execute("SELECT COUNT(*) FROM memory_catalog m "
            "LEFT JOIN directory_events d ON m.body=d.body AND m.time=d.time "
            "WHERE d.time IS NULL").fetchone()[0]
        if total != facts or missing:
            raise AssertionError("Cold catalog does not cover every stored fact")
        checked = enabled = references = 0
        midtrial = positive_to_negative = negative_to_positive = 0
        last_benefit, visits = {}, {}
        for body, start, end, on, baseline, actual, estimate, raw in db.execute(
            "SELECT * FROM memory_trials ORDER BY body,start"):
            row = db.execute("SELECT horizon,targets FROM outcomes WHERE body=? AND time=?",
                             (body, start)).fetchone()
            if row is None or start + row[0] - 1 != end:
                raise AssertionError("Trial used incomplete or mismatched consequence")
            target = np.frombuffer(row[1], np.float32)
            expected = float(outcome_utility(target) / 9.)
            if abs(expected - actual) > 1e-7:
                raise AssertionError("Trial utility differs from factual outcome")
            credit = np.clip(2 * (1 if on else -1) * (actual - baseline), -1, 1)
            if abs(credit - estimate) > 1e-7:
                raise AssertionError("Credit rule differs")
            records = json.loads(raw)
            total_weight = max(sum(item[1] for item in records), 1e-6)
            for past, weight, benefit in records:
                row = db.execute("SELECT horizon FROM outcomes WHERE body=? AND time=?",
                                 (body, past)).fetchone()
                # References accumulate throughout the trial, not just at its start.
                if row is None or past + row[0] > end or weight <= 0:
                    raise AssertionError("Future label or invalid body-local memory credit")
                if not np.isfinite([weight, benefit]).all():
                    raise AssertionError("Nonfinite memory credit")
                midtrial += int(past + row[0] > start)
                previous = last_benefit.get((body, past), 0.)
                rate = .2 * math.sqrt(weight / total_weight)
                expected_benefit = (1 - rate) * previous + rate * estimate
                if abs(expected_benefit - benefit) > 1e-10:
                    raise AssertionError("Retention update differs from credit recurrence")
                positive_to_negative += int(previous > 0 and benefit < 0)
                negative_to_positive += int(previous < 0 and benefit > 0)
                last_benefit[body, past] = benefit
                visits[body, past] = visits.get((body, past), 0) + 1
                references += 1
            checked += 1
            enabled += on
        for body, past, benefit, tests in db.execute(
                "SELECT body,time,benefit,tests FROM memory_catalog WHERE tests>0"):
            if visits.get((body, past)) != tests or last_benefit[body, past] != benefit:
                raise AssertionError("Database retention differs from recorded practice")
        invalid = db.execute(
            "SELECT COUNT(*) FROM cold_recalls c LEFT JOIN outcomes o "
            "ON c.body=o.body AND c.time=o.time "
            "WHERE o.time IS NULL OR o.time+o.horizon>c.recalled_at").fetchone()[0]
        if invalid:
            raise AssertionError("Invalid cold recall")
        values = db.execute("SELECT MIN(benefit),MAX(benefit),SUM(tests>0),SUM(benefit>0),"
                            "SUM(benefit<0) FROM memory_catalog").fetchone()
        return dict(catalog_records=total, trials_checked=checked, enabled=enabled,
            credited_references=references, invalid_cold_recalls=invalid,
            cold_recalls=db.execute("SELECT COUNT(*) FROM cold_recalls").fetchone()[0],
            benefit_min=values[0], benefit_max=values[1], tested_records=values[2],
            positive_records=values[3], negative_records=values[4],
            positive_to_negative=positive_to_negative,
            negative_to_positive=negative_to_positive,
            references_maturing_during_trial=midtrial,
            temporal_check="mature_by_trial_end; exact_per_query_time_not_logged")
    finally:
        db.close()


def phases(path, width=4096):
    keys = ("safe_contact", "contact", "collision", "impact", "combo", "pred_error")
    joined = {key: [] for key in keys}
    for file in sorted(path.glob("transitions_*.npz")):
        with np.load(file, allow_pickle=False) as data:
            for key in keys:
                joined[key].append(data[key])
    arrays = {key: np.concatenate(value) for key, value in joined.items()}
    result = []
    for start in range(0, len(arrays["contact"]), width):
        a = {key: value[start:start + width] for key, value in arrays.items()}
        attacked = a["combo"][..., 2] != 0
        count = a["contact"].size
        result.append(dict(start=start, end=start + len(a["contact"]),
            experiences=count, safe_contacts=int(a["safe_contact"].sum()),
            contacts=int(a["contact"].sum()),
            safe_per_body_minute=float(a["safe_contact"].mean() * 500),
            collision_rate=float(a["collision"].mean()),
            attacked_impact_rate=float(a["impact"][attacked].mean()),
            prediction_mse=float(a["pred_error"].mean())))
    return result


def plot(output, phase_data, boundary=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (key, title) in zip(axes.flat, (
        ("safe_per_body_minute", "Safe contacts / body minute"),
        ("collision_rate", "Collision steps"),
        ("attacked_impact_rate", "Impact steps in attacked scenes"),
        ("prediction_mse", "Online prediction MSE"))):
        for arm, color in (("random", "#4b5563"), ("reversible", "#008b76")):
            points = phase_data[arm]
            ax.plot([p["end"] for p in points], [p[key] for p in points],
                    marker="o", color=color, label=arm)
        ax.set(title=title, xlabel="New steps per body")
        ax.grid(alpha=.2)
        if boundary is not None:
            ax.axvline(boundary, color="#b45309", linestyle=":", linewidth=1)
    axes[0, 0].legend()
    if boundary is not None:
        fig.suptitle(f"Weights retained; runtime reset at step {boundary}")
    fig.tight_layout()
    fig.savefig(output / "learning.png", dpi=160)
    plt.close(fig)


def main():
    config = json.loads((ROOT / "config.json").read_text())
    output = ROOT / config["stage"]
    if not (output / "COMPLETE.json").exists():
        raise RuntimeError("Full run is not complete")
    manifest = json.loads((output / "manifest.json").read_text())
    for name, sha in manifest["sources"].items():
        for source in (ROOT.parent / name, output / "source" / name):
            if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
                raise AssertionError(f"Frozen source changed: {name}")
    audits = {}
    for path in sorted(output.rglob("events.sqlite")):
        name = path.parent.relative_to(output).as_posix()
        audits[name] = session_audit(path.parent)
        print(json.dumps(dict(audited=name, labels=audits[name]["consequence_labels_checked"])),
              flush=True)
    credit = memory_audit(output / "reversible")
    replay = [audits[arm]["counts"]["retrievals"] for arm in config["arms"]]
    if len(set(replay)) != 1:
        raise AssertionError("Replay counts differ")
    phase_data = {arm: phases(output / arm) for arm in config["arms"]}
    results = {arm: {str(step): json.loads(
        (output / arm / f"result_{step}.json").read_text())["probe"]
        for step in [0, *config["milestones"]]} for arm in config["arms"]}
    shifts = {arm: [json.loads((output / arm / f"final_shift_{i}.json").read_text())
                   for i in (0, 1)] for arm in config["arms"]}
    dump(output / "ANALYSIS.json", dict(phases=phase_data, development=results, shifts=shifts))
    dump(output / "AUDIT.json", dict(complete=True, frozen_sources=True,
        factual_sessions=audits, memory_credit=credit, replay_per_arm=replay[0]))
    plot(output, phase_data)
    lines = ["# V21 实测结果", "", "2026-10-15。完整配置一次运行；保留全部原始事实。",
             "", "## 练习过程", "",
             "| 每身体步数区间 | 随机安全触达 | 可逆目录安全触达 | 随机碰撞 | 可逆碰撞 |",
             "|---|---:|---:|---:|---:|"]
    for a, b in zip(phase_data["random"], phase_data["reversible"]):
        lines.append(f"| {a['start']}–{a['end']} | {a['safe_contacts']} | "
            f"{b['safe_contacts']} | {a['collision_rate']:.3%} | {b['collision_rate']:.3%} |")
    lines += ["", "## 权重冻结后的开发观察", "",
              "此表关闭事件目录动作修正、参数更新；旧组合已用于开发。",
              "", "| 每身体训练步数 | 随机安全/全部触达 | 可逆安全/全部触达 |",
              "|---|---:|---:|"]
    for step in [0, *config["milestones"]]:
        a, b = (results[arm][str(step)] for arm in config["arms"])
        lines.append(f"| {step} | {a['safe_contacts']}/{a['contacts']} | "
                     f"{b['safe_contacts']}/{b['contacts']} |")
    lines += ["", "## 保留物理条件", "", "仅在训练结束后运行，未用于本轮调参。",
              "", "| 条件 | 随机安全/全部触达 | 可逆安全/全部触达 | "
              "随机碰撞/受击 | 可逆碰撞/受击 |", "|---|---:|---:|---:|---:|"]
    for i in (0, 1):
        a, b = (shifts[arm][i] for arm in config["arms"])
        lines.append(f"| {i} | {a['safe_contacts']}/{a['contacts']} | "
            f"{b['safe_contacts']}/{b['contacts']} | "
            f"{a['collision_rate']:.3%}/{a['attacked_impact_rate']:.3%} | "
            f"{b['collision_rate']:.3%}/{b['attacked_impact_rate']:.3%} |")
    lines += ["", "## 机制与口径", "",
        f"冷库记录 {credit['catalog_records']:,}；召回 {credit['cold_recalls']:,} 次；"
        f"实际后果试验 {credit['trials_checked']:,} 段，其中开启 {credit['enabled']:,} 段。",
        f"有后果信用记录 {credit['tested_records']:,}；当前正值 {credit['positive_records']:,}、"
        f"负值 {credit['negative_records']:,}。访问本身不奖励。",
        f"每组新增物理经验 {config['worlds'] * config['milestones'][-1]:,}，"
        f"回放 {replay[0]:,} 次。实际策略呈现差异见各组 summary.json。",
        "两组都从同一校准权重出发，128 条活跃目录。候选同时改变目录、动作修正、"
        "失败回放，结果只能解释为组合差异，不能单独证明遗忘作用。",
        "单种子、简化平面物理、有限视觉；不声称模拟真实大脑动力学。",
        "后果开关提供有噪声的整段贡献估计，不能证明每条记录的独立因果价值。",
        "", "原始事实标签、来源哈希、回放时序、冷库召回与信用递推均已核验。"
        "段内引用只记录累计权重，能核对段结束前已成熟，不能逐次审计调用时间；"
        "完整机器数据见 ANALYSIS.json、AUDIT.json。"]
    (output / "RESULT_TABLES_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    post = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "analyze.py",)}
    dump(output / "postprocessing_sources.json", post)


if __name__ == "__main__":
    main()
