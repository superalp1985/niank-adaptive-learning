"""Generate descriptive process figures and reports from independently audited data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "pilot" / "AUDIT.json").read_text(encoding="utf-8"))["results"]
    colors = dict(recent="#147d92", directory="#b44c38")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    lines = [
        "# V14：多场景实体追踪首轮",
        "",
        "日期：2026-10-13。两组单初始化探索；不是统计显著性或机器人部署证明。",
        "48个独立身体持续混合18类训练组合，冻结评价另外9类组合。",
        "各类组合覆盖布局、目标运动和攻击；没有共同A-B-A-B课程。",
        "每个探针固定每类1024时间步；以下成功率分母是已结束回合，",
        "仍在进行的9个回合另行保留，不将时间步当作回合。",
        "",
        "| 记忆 | 每身体学习步 | 新经历 | 未见组合接触成功 | 受击/已结束回合 | 碰撞时间步 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, payload in data.items():
        xs = [0, 1024, 4096, 8192]
        values = [payload[str(step)]["heldout"] for step in xs]
        for step, value in zip(xs, values):
            lines.append(f"| {mode} | {step:,} | {step * 48:,} | "
                         f"{value['success']:.2%} | {value['impact']:.2%} | "
                         f"{value['collision_step_rate']:.2%} |")
        for ax, key in zip(axes, ("success", "impact", "collision_step_rate")):
            ax.plot(xs, [v[key] for v in values], marker="o", color=colors[mode], label=mode)
            ax.set_xlabel("New physical experiences / body")
            ax.grid(alpha=0.2)
            ax.set_ylim(bottom=0)
    for ax, title in zip(axes, ("Held-out target contact", "Projectile impacts / episodes",
                              "Obstacle-contact steps / steps")):
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "PROCESS_CURVES.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 逐场景最终记录", "",
              "| 记忆 | 场景(布局,运动,攻击) | 接触/结束 | 受击/结束 | 碰撞步/总步 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for mode, payload in data.items():
        final = payload["8192"]["heldout"]
        for scene, value in final["scenes"].items():
            lines.append(f"| {mode} | {scene} | {value['success']}/{value['episodes']} | "
                         f"{value['impact']}/{value['episodes']} | "
                         f"{value['collision']}/{value['steps']} |")
    lines += ["", "## 解释边界", "",
              "成功来自实际目标接触；受击同帧的接触不算成功。全部物理转移、",
              "奖励、终止、实际动作、原始数据库记录及记忆身份已由audit.py重算。",
              "这里没有最优动作标签，没有用稠密模型一致性作正确率。",
              "",
              "目录价值是进展与预测误差的关联估计，不是目录操作收益的因果证明。",
              "两组都使用当前可见几何与过去转移；预测器只影响目录估值，",
              "尚未直接参与安全行动选择。高接触率不能掩盖碰撞和受击。",
              "",
              "观测是理想化可见物体测量，图片仅存档；未验证摄像头识别、",
              "遮挡、真实机器人动力学或生物神经活动。每帧只报告最近3个障碍",
              "和攻击物；仍需针对拥挤/遮挡设计新场景。",
              "四个行动专家、路由、价值和预测模块彼此断梯度，",
              "共81,993个参数。单种子和有限场景不足以支持普遍泛化结论。",
              "",
              "所有探针在反馈学习前冻结模型/目录估值；可积累当前观察记忆。",
              "未见指组合未用于训练，不代表每个组成因素都全新。",
              "本轮留出结果已经查看，下一轮只能作为开发依据，需新的评价种子。",
              "",
              "## 完整数据", "",
              "上传、远端分片SHA-256验证和本地清理凭据放在archives目录。",
              "原始观察永久保留于远端归档，不因目录淡出而删除。"]
    (ROOT / "REPORT_V14_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
