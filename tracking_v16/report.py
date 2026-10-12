"""Whole-episode outcome curves, with a matched additional-practice control."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    results = json.loads((ROOT / "pilot" / "AUDIT.json").read_text(encoding="utf-8"))["results"]
    lines = ["# V16：完整安全到达与学习量", "",
        "日期：2026-10-13。两组都从V15连续观察组的同一最终权重开始，",
        "在新世界、新优化器和空记忆中额外学习393216次新经历。",
        "每组累计权重所经历的物理数据量为786432条；不是原轨迹精确续训。",
        "两组都增加可观察的本回合碰撞历史；只有safe_contact改变回合反馈。",
        "预测、行动、路由、价值和风险模块各自更新；风险不参与动作分配。",
        "",
        "| 模式 | 额外步/身体 | 接触成功 | 全回合无碰撞接触 | 受击 | 碰撞步 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for mode, payload in results.items():
        xs = [0, 1024, 4096, 8192]
        values = [payload[str(step)]["heldout"] for step in xs]
        for step, value in zip(xs, values):
            lines.append(f"| {mode} | {step:,} | {value['success']:.2%} | "
                         f"{value['safe_success']:.2%} | {value['impact']:.2%} | "
                         f"{value['collision_step_rate']:.2%} |")
        for ax, key in zip(axes, ("success", "safe_success", "impact", "collision_step_rate")):
            ax.plot(xs, [v[key] for v in values], marker="o", label=mode)
            ax.set_xlabel("Additional steps / body")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
    for ax, title, metric in zip(axes, ("Target contact", "Contact without collision",
                               "Impacts / episodes", "Collision steps / steps"),
                                ("success", "safe_success", "impact", "collision_step_rate")):
        ax.set_title(title, fontsize=10)
        maximum = max(payload[str(step)]["heldout"][metric]
                      for payload in results.values() for step in (0, 1024, 4096, 8192))
        ax.set_ylim(0, max(maximum * 1.08, 0.01))
    fig.tight_layout()
    fig.savefig(ROOT / "PROCESS_CURVES_CORRECTED.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 最终逐场景", "",
              "| 模式 | 布局,运动,攻击 | 接触/结束 | 无碰撞接触/结束 | 受击/结束 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for mode, payload in results.items():
        for scene, v in payload["8192"]["heldout"]["scenes"].items():
            lines.append(f"| {mode} | {scene} | {v['success']}/{v['episodes']} | "
                         f"{v['safe_success']}/{v['episodes']} | {v['impact']}/{v['episodes']} |")
    lines += ["", "## 场景覆盖与失败过程", ""]
    for mode, payload in results.items():
        scenes = payload["8192"]["heldout"]["scenes"].values()
        macro = sum(v["safe_success"] / max(v["episodes"], 1) for v in scenes) / 9
        lines.append(f"{mode}的逐场景等权无碰撞接触率：{macro:.2%}。")
    lines += ["", "安全反馈的安全接触改善在逐场景等权后仍存在，但不代表所有",
              "指标同时改善：受击和碰撞步均较对照差，应报告为混合结果。",
              "",
              "POSTHOC_DIAGNOSTICS.json是已执行轨迹的事后诊断，没有参与训练。",
              "碰撞后失败回合平均接触障碍29.5步（对照）和43.43步（安全反馈）。",
              "两组都未使用停止动作，不能归因为学会停住。安全反馈失败回合",
              "平均时长更短，却在障碍上持续接触更久；这提示遇阻后的恢复",
              "行为仍有问题，不证明某一架构缺陷。",
              "",
              "下一步应研究真实碰撞后形成可修订的局部通路/绕行子目标记忆，",
              "而非继续增加固定风险扣分或只放大最终成功奖励。此方向尚未实现。"]
    lines += ["", "## 解释", "",
              "safe_contact的额外反馈只在实际目标接触时，根据已发生的碰撞",
              "历史给予+16或-8；不能直接告诉它应当选择哪个动作。",
              "原物理奖励和额外反馈分别从原始转移独立重算，全部数据通过审计。",
              "单种子探索，不凭回合次数制造独立样本显著性。",
              "",
              "V14/V15/V16都在同一场景家族开发；留出组合已反复查看。",
              "新物理样本不能代替独立新任务结构验证，当前能力不等于通用追踪。",
              "观测仍为理想化物体位置，图像只存档，尚未验证摄像头识别、",
              "遮挡和真实机器人身体，也不声称模拟了真实脑动力学。",
              "",
              "完整数据、初始与最终权重保留于远端归档，逐片读回哈希后清理本地。",
              "上传与清理凭据见archives/research-v16-safe-contact-20261013。"]
    (ROOT / "REPORT_V16_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
