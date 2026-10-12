"""Read frozen experiment outputs, audit, plot, and report without model tuning."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache" / "matplotlib")

import hashlib
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ("full", "fifo", "surprise", "value", "anti_value", "none")
COLORS = {
    "full": "#555555", "fifo": "#6554a4", "surprise": "#dd7d25",
    "value": "#13865b", "anti_value": "#c4444c", "none": "#2982b6",
}
METRICS = (
    "success", "state_mse", "regret", "fine_retained_risk", "fine_retained_safe",
    "coarse_retained", "contact_balanced_accuracy", "collision", "nuisance_records",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_interval(arrays: list[np.ndarray], seed: int = 20013) -> list[float]:
    """Two-way resampling: optimization seeds and shared held-out episode IDs."""
    data = np.stack(arrays)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(2000):
        chosen_seeds = rng.integers(0, len(data), len(data))
        chosen_episodes = rng.integers(0, data.shape[1], data.shape[1])
        means.append(float(data[chosen_seeds][:, chosen_episodes].mean()))
    return list(map(float, np.quantile(means, (0.025, 0.975))))


def main() -> None:
    output = ROOT / "confirmation"
    assert (output / "COMPLETE.json").exists(), "Do not analyze an incomplete cohort grid"
    records = read(output / "final_results.json")
    manifest = read(output / "frozen_manifest.json")
    snapshot = (output / "source_snapshot.py").read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == manifest["source_sha256"]
    assert len(records) == 12
    groups = defaultdict(list)
    for record in records:
        groups[(record["unique"], record["repeats"])].append(record)
    assert len(groups) == 4
    summary = []
    for (unique, repeats), cohorts in groups.items():
        cohorts.sort(key=lambda item: item["seed"])
        assert [item["seed"] for item in cohorts] == [11, 23, 37]
        row = {"unique": unique, "repeats": repeats, "presentations": unique * repeats, "scenarios": {}}
        for scenario in ("ordinary", "long_delay", "left_layout", "texture_shift"):
            row["scenarios"][scenario] = {}
            for mode in MODES:
                values = [item["scenarios"][scenario][mode] for item in cohorts]
                metrics = {}
                for name in METRICS:
                    data = np.asarray([item[name] for item in values])
                    metrics[name] = {
                        "mean": float(data.mean()), "min": float(data.min()), "max": float(data.max()),
                        "seeds": data.tolist(),
                    }
                row["scenarios"][scenario][mode] = metrics
            row["scenarios"][scenario]["paired_value_minus_full"] = {}
            for name in ("success", "state_mse", "regret"):
                deltas = []
                for item in cohorts:
                    directory = output / f"u{unique}_r{repeats}_s{item['seed']}"
                    with np.load(directory / f"predictions_{scenario}_value.npz") as value:
                        with np.load(directory / f"predictions_{scenario}_full.npz") as full:
                            np.testing.assert_array_equal(value["ids"], full["ids"])
                            deltas.append(value[name].astype(float) - full[name].astype(float))
                row["scenarios"][scenario]["paired_value_minus_full"][name] = {
                    "mean": float(np.mean(deltas)), "bootstrap_95": paired_interval(deltas),
                    "seed_means": [float(delta.mean()) for delta in deltas],
                    "note": "Only three optimization seeds; interval is exploratory.",
                }
        summary.append(row)
    path = ROOT / "SUMMARY_V5.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = [f"{row['unique']} x {row['repeats']}" for row in summary]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    for axis, metric, title, factor in (
        (axes[0, 0], "success", "Physical-action success (%)", 100),
        (axes[0, 1], "state_mse", "Next-state prediction MSE", 1),
        (axes[1, 0], "fine_retained_risk", "Small cue retained when physically relevant (%)", 100),
        (axes[1, 1], "regret", "Physical action regret (lower is better)", 1),
    ):
        for mode in MODES:
            averages = np.asarray([row["scenarios"]["ordinary"][mode][metric]["mean"] for row in summary]) * factor
            low = np.asarray([row["scenarios"]["ordinary"][mode][metric]["min"] for row in summary]) * factor
            high = np.asarray([row["scenarios"]["ordinary"][mode][metric]["max"] for row in summary]) * factor
            axis.plot(range(len(summary)), averages, marker="o", color=COLORS[mode], label=mode)
            if mode in ("full", "value"):
                axis.fill_between(range(len(summary)), low, high, color=COLORS[mode], alpha=0.12)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(len(summary)), labels)
        axis.set_xlabel("Unique episodes x repetitions")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    figure.suptitle("Frozen V5 confirmation: 3 optimization seeds; shaded seed ranges", fontsize=12)
    figure.savefig(ROOT / "volume_results.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for mode in ("full", "fifo", "surprise", "value", "anti_value", "none"):
        histories = [read(output / f"u512_r32_s{seed}" / "history.json")[mode] for seed in (11, 23, 37)]
        x = [item["presentations"] for item in histories[0]]
        for axis, name, title in (
            (axes[0], "success", "Development physical success"),
            (axes[1], "state_mse", "Development physical prediction MSE"),
        ):
            y = np.asarray([[entry["development"][name] for entry in history] for history in histories])
            axis.plot(x, y.mean(0), marker="o", color=COLORS[mode], label=mode)
            axis.set_title(title)
            axis.set_xlabel("Training presentations (512 unique episodes)")
            axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    figure.savefig(ROOT / "learning_curves.png", dpi=150)
    plt.close(figure)

    lines = [
        "# V5 实验结果：训练量与有价值的遗忘", "",
        "日期：2026-10-12。72 个模型配置完成：4 种训练量 × 6 种记忆策略 × 3 个优化种子。",
        "每种配置在普通、长延迟、左侧墙布局、改变干扰纹理四个独立场景上评估，各 512 个经历。",
        "这是视觉线索、身体状态和物理反应的简化任务，不是完整果蝇身体仿真。", "",
        "上一轮接触指标存在输入泄漏，参数计数也包含未使用模块；本轮独立修正，不拿旧数字做提升基线。",
        "本轮评分由物理结果决定：动作成本距最优不超过 0.02 即通过，允许多个正确动作。", "",
        "## 普通场景", "",
        "|独立经历×重复|总呈现|全部保留|价值保留|时间 FIFO|意外程度|反向价值|无记忆|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        scenario = row["scenarios"]["ordinary"]
        scores = [f"{scenario[mode]['success']['mean']:.2%}" for mode in ("full", "value", "fifo", "surprise", "anti_value", "none")]
        lines.append(f"|{row['unique']}×{row['repeats']}|{row['presentations']:,}|" + "|".join(scores) + "|")
    lines.extend([
        "", "以上为三个优化种子均值，不能把 1,536 次评分当作 1,536 次独立训练。",
        "所有模型结构和参数量相同。普通场景全部保留读 18 条；价值、FIFO、意外程度和反向价值保留 4 条，读 2 条。",
        "本轮仍编码和打分全部观察，回放和价值训练有额外成本；不能说实际计算减少了 8/9。", "",
        "## 预测与细节", "",
        "|独立经历×重复|全部保留 MSE|价值保留 MSE|风险细节保留|安全场景细节保留|粗线索保留|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        scene = row["scenarios"]["ordinary"]
        value = scene["value"]
        lines.append(
            f"|{row['unique']}×{row['repeats']}|{scene['full']['state_mse']['mean']:.5f}|"
            f"{value['state_mse']['mean']:.5f}|{value['fine_retained_risk']['mean']:.2%}|"
            f"{value['fine_retained_safe']['mean']:.2%}|{value['coarse_retained']['mean']:.2%}|"
        )
    lines.extend([
        "", "细节仅是两个低幅度像素，干扰信号更亮且不可预测。",
        "细节留下了不等于已经被正确使用；配套物理预测误差和动作后果才是能力指标。",
        "安全场景仍保留细节说明当前机制尚未实现充分按自身状态压缩，不能宣称已经实现 FAME。", "",
        "## 迁移与失效", "",
        "|独立经历×重复|长延迟 全/价值|左布局 全/价值|纹理改变 全/价值|",
        "|---|---:|---:|---:|",
    ])
    for row in summary:
        cells = []
        for scenario in ("long_delay", "left_layout", "texture_shift"):
            scene = row["scenarios"][scenario]
            cells.append(f"{scene['full']['success']['mean']:.2%} / {scene['value']['success']['mean']:.2%}")
        lines.append(f"|{row['unique']}×{row['repeats']}|" + "|".join(cells) + "|")
    high = next(row for row in summary if row["unique"] == 512 and row["repeats"] == 32)
    pair = high["scenarios"]["ordinary"]["paired_value_minus_full"]["success"]
    lines.extend([
        "", f"512×32 的价值减全部保留差为 {pair['mean']*100:+.2f} 个百分点；"
        f"配对探索性 bootstrap 区间为 [{pair['bootstrap_95'][0]*100:+.2f}, {pair['bootstrap_95'][1]*100:+.2f}]。",
        "区间同时重采样优化种子和相同测试经历；仅三个优化种子，不能作为最终显著性结论。", "",
        "## 保留的失败", "",
        "- discovery：任务动作偏斜，全部策略退化成固定动作；70.70% 只是固定基线。",
        "- discovery_v2：价值策略保护粗线索，但风险细节保留仅 17.72%；这是细节保护失败。",
        "- discovery_v3：按通道尺度归一化，开发风险细节保留达到 100%，据此冻结本轮确认协议。",
        "- anti_value：刻意保留低价值事件，所有结果公开，不为成功删去。", "",
        "## 解释边界", "",
        "64×64 与 512×8，以及 512×32 与 2048×8，分别保持总呈现和优化步数相同，比较多样性。",
        "512×8 与 512×32 比较同一经历池增加重复的作用。",
        "当前结果只能归于这个任务和这些条件，不能归于人脑动力学、通用推理或完整果蝇连接结构。",
        "价值由训练期同一模型的物理损失反馈学习；评估时不读取标签，数据库也不向它提供参考答案。", "",
        "最优动作均衡通过物理场景拒绝采样实现，身体状态仍可能与材料存在采样相关；无记忆基线和配对细节翻转用于暴露这类捷径。",
        "结构化视觉通道明确区分材质、微弱线索与纹理，是有意简化；尚未证明在自然图像中自行发现重要特征。", "",
    ])
    detail_path = ROOT / "DETAIL_SUMMARY.json"
    if detail_path.exists():
        details = read(detail_path)
        lines.extend([
            "## 细节翻转诊断", "",
            "这是确认集训练结束后的独立只读诊断，不用于参数选择。256 个新经历，26 步延迟。",
            "保持身体状态、粗线索和纹理不变，翻转细线索；风险状态下真实速度变化幅度 0.36，安全状态下为 0。",
            "响应增益 1 表示平均预测变化达到真实变化，0 表示不响应；只看保留率不能判断使用是否正确。", "",
            "|独立经历×重复|全保留风险响应增益|价值风险响应增益|价值安全场景错误变化|",
            "|---|---:|---:|---:|",
        ])
        for row in summary:
            subset_details = {
                item["mode"]: item for item in details
                if item["unique"] == row["unique"] and item["repeats"] == row["repeats"]
            }
            lines.append(
                f"|{row['unique']}×{row['repeats']}|"
                f"{subset_details['full']['risk_effect_gain']:.3f}|"
                f"{subset_details['value']['risk_effect_gain']:.3f}|"
                f"{subset_details['value']['safe_spurious_velocity_effect']:.4f}|"
            )
        lines.extend([
            "", "训练增加后价值策略确实开始响应细节，但高预算风险响应仍不足，安全状态仍有不应出现的响应。",
            "这是重要的失败：当前价值学习保护信息，却没有充分学会按自身状态决定保留与使用。",
            "这一诊断也说明目前的泛化提升有限；不能把保留关键细节等同于理解其物理作用。", "",
        ])
    lines.extend([
        "## 文件", "",
        "- PROTOCOL.md：变量、因果边界与冻结规则。",
        "- SUMMARY_V5.json：种子范围、四场景、配对差值区间。",
        "- confirmation/：源码哈希、训练曲线、检查点、逐经历评分。",
        "- confirmation/observations.sqlite：原始观察，活跃目录遗忘后仍可查询。",
        "- volume_results.png、learning_curves.png：训练量和学习过程图。", "",
        "下一轮优先检验：用状态条件反事实训练价值和读取机制，减少安全状态的错误响应；引入状态变化后的数据库恢复。",
        "之后再加入分层目录索引、训练数量扩大与成熟身体环境接口；本轮不声称已完成它们。", "",
    ])
    (ROOT / "REPORT_V5_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    total_bytes = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())
    (ROOT / "AUDIT.json").write_text(json.dumps({
        "complete": True, "cohorts": len(records), "models": len(records) * len(MODES),
        "source_snapshot_hash_verified": True,
        "root_bytes": total_bytes, "root_mib": total_bytes / 1024**2,
        "prediction_artifacts": len(list(output.glob("u*/predictions_*.npz"))),
        "data_ids_disjoint": [[510000, 512047], [610000, 610255], [710000, 710511],
                              [720000, 720511], [730000, 730511], [740000, 740511],
                              [810000, 810255]],
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "models": 72, "root_mib": round(total_bytes / 1024**2, 2),
        "ordinary": [
            {"unique": row["unique"], "repeats": row["repeats"],
             "success": {mode: round(row["scenarios"]["ordinary"][mode]["success"]["mean"], 4) for mode in MODES}}
            for row in summary
        ],
    }))


if __name__ == "__main__":
    main()
