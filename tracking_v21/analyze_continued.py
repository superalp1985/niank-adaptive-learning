"""Account separately for aborted practice and the weights-only continuation."""

import argparse
import hashlib
import json
from pathlib import Path

from tracking_v17.run import dump
from tracking_v20.audit import session_audit
from tracking_v21.analyze import memory_audit, phases, plot

ROOT = Path(__file__).resolve().parent


def verify_sources(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        for path in (ROOT.parent / name, output / "source" / name):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise AssertionError("Frozen source differs: " + name)
    return manifest


def audit_sessions(output):
    result = {}
    for path in sorted(output.rglob("events.sqlite")):
        if "uncommitted" in path.parts or "interrupted_observations" in path.parts:
            continue
        name = path.parent.relative_to(output).as_posix()
        result[name] = session_audit(path.parent)
        print(json.dumps(dict(audited=str(output.name) + "/" + name,
                              labels=result[name]["consequence_labels_checked"])), flush=True)
    return result


def interrupted():
    output = ROOT / "reversible_practice"
    verify_sources(output)
    result = dict(training_complete=False, sources_verified=True,
                  sessions=audit_sessions(output),
                  credit=memory_audit(output / "reversible"),
                  phases={arm: phases(output / arm) for arm in ("random", "reversible")})
    result["unmatured_at_interruption"] = {
        arm: row["counts"]["events"] - row["counts"]["outcomes"]
        for arm, row in result["sessions"].items() if "/" not in arm}
    dump(output / "INTERRUPTED_AUDIT.json", result)
    return result


def completed():
    output = ROOT / "continued_practice"
    complete = json.loads((output / "COMPLETE.json").read_text())
    if not complete["complete"]:
        raise RuntimeError("Continuation incomplete")
    manifest = verify_sources(output)
    config = manifest["config"]
    previous = ROOT / "reversible_practice"
    for arm, expected in manifest["ancestors"].items():
        if hashlib.sha256((previous / arm / "checkpoint_4096.pt").read_bytes()).hexdigest() != expected:
            raise AssertionError("Ancestral weights changed")
    old_file = previous / "INTERRUPTED_AUDIT.json"
    old = json.loads(old_file.read_text()) if old_file.exists() else interrupted()
    sessions = audit_sessions(output)
    credit = memory_audit(output / "reversible")
    practice = {arm: phases(output / arm) for arm in config["arms"]}
    lineage = {}
    summaries, probes, shifts = {}, {}, {}
    for arm in config["arms"]:
        lineage[arm] = [old["phases"][arm][0]] + [
            dict(point, start=point["start"] + 4096, end=point["end"] + 4096)
            for point in practice[arm]]
        summaries[arm] = json.loads((output / arm / "summary.json").read_text())
        if summaries[arm]["pending_unlabelled"] or sessions[arm]["steps"] != 20480:
            raise AssertionError("Incomplete continuation")
        if sessions[arm]["counts"]["retrievals"] != 245760:
            raise AssertionError("Replay budget mismatch")
        counts = sessions[arm]["counts"]
        if counts["events"] != counts["outcomes"]:
            raise AssertionError("Missing final factual labels")
        for name in ("safe_contacts", "contacts"):
            if sum(point[name] for point in practice[arm]) != summaries[arm][name]:
                raise AssertionError("Summary differs from raw practice: " + name)
        probes[arm] = {str(step): json.loads(
            (previous / arm / f"result_{step}.json").read_text())["probe"]
            for step in (0, 4096)}
        for step in config["milestones"]:
            probes[arm][str(step + 4096)] = json.loads(
                (output / arm / f"result_{step}.json").read_text())["probe"]
        shifts[arm] = [json.loads((output / arm / f"final_shift_{i}.json").read_text())
                       for i in (0, 1)]
        for observation in [*probes[arm].values(), *shifts[arm]]:
            if observation["update_presentations"] or observation["replay_presentations"]:
                raise AssertionError("Frozen observation unexpectedly trained")
    analysis = dict(continuation_boundary=4096, practice=practice, lineage_phases=lineage,
        development=probes, shifts=shifts, summaries=summaries,
        lineage_experiences_per_arm=72 * 24576,
        new_continuation_experiences_per_arm=72 * 20480,
        aborted_extra_experiences={arm: (old["sessions"][arm]["steps"] - 4096) * 72
                                   for arm in config["arms"]})
    dump(output / "ANALYSIS.json", analysis)
    dump(output / "AUDIT.json", dict(complete=True, sources_verified=True,
        factual_sessions=sessions, credit=credit,
        original_interrupted_audit_sha256=hashlib.sha256(old_file.read_bytes()).hexdigest()))
    plot(output, lineage, boundary=4096)
    lines = ["# V21 续训实测表", "",
        "4096 步处权重保留、优化器与运行记忆重置。不能解释为不中断训练。",
        "", "## 同等实践长度", "",
        "| 累计步数 | 随机安全触达 | 可逆安全触达 | 随机碰撞 | 可逆碰撞 |",
        "|---|---:|---:|---:|---:|"]
    for a, b in zip(lineage["random"], lineage["reversible"]):
        lines.append(f"| {a['start']}-{a['end']} | {a['safe_contacts']} | {b['safe_contacts']} | "
                     f"{a['collision_rate']:.3%} | {b['collision_rate']:.3%} |")
    lines += ["", "## 关闭事件目录修正后的权重观察", "",
        "| 累计步数 | 随机安全/总触达 | 可逆安全/总触达 | 随机碰撞/受击 | 可逆碰撞/受击 |",
        "|---|---:|---:|---:|---:|"]
    for step in (0, 4096, 12288, 24576):
        a, b = (probes[arm][str(step)] for arm in config["arms"])
        lines.append(f"| {step} | {a['safe_contacts']}/{a['contacts']} | "
            f"{b['safe_contacts']}/{b['contacts']} | "
            f"{a['collision_rate']:.3%}/{a['attacked_impact_rate']:.3%} | "
            f"{b['collision_rate']:.3%}/{b['attacked_impact_rate']:.3%} |")
    lines += ["", "## 最后才运行的保留物理条件", "",
        "| 条件 | 随机安全/总触达 | 可逆安全/总触达 | 随机碰撞/受击 | 可逆碰撞/受击 |",
        "|---|---:|---:|---:|---:|"]
    for profile in (0, 1):
        a, b = (shifts[arm][profile] for arm in config["arms"])
        lines.append(f"| {profile} | {a['safe_contacts']}/{a['contacts']} | "
            f"{b['safe_contacts']}/{b['contacts']} | "
            f"{a['collision_rate']:.3%}/{a['attacked_impact_rate']:.3%} | "
            f"{b['collision_rate']:.3%}/{b['attacked_impact_rate']:.3%} |")
    lines += ["", "## 记忆可逆性", "",
        f"冷库 {credit['catalog_records']:,} 条，召回 {credit['cold_recalls']:,} 次；"
        f"成熟后果试验 {credit['trials_checked']:,} 段。",
        f"保留价值由正转负 {credit['positive_to_negative']:,} 次，"
        f"由负转正 {credit['negative_to_positive']:,} 次。",
        "价值反转证明机制在变化，不等于证明每次变化都是正确的。",
        "实际效用来自真实物理后果，不是参考模型答案一致率。",
        "段内引用记录只有累计权重，不能声称逐次调用时间均经独立审计。",
        "单种子、组合机制、运行重置和简化二维视觉限制详见报告。"]
    (output / "RESULT_TABLES_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    post = {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "analyze.py", ROOT / "analyze_continued.py",
                         ROOT.parent / "tracking_v20/audit.py")}
    dump(output / "postprocessing_sources.json", post)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interrupted-only", action="store_true")
    args = parser.parse_args()
    interrupted() if args.interrupted_only else completed()
