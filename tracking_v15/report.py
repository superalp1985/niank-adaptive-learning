"""Descriptive V15 process evidence, including no-risk intervention."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    results = json.loads((ROOT / "pilot" / "AUDIT.json").read_text(encoding="utf-8"))["results"]
    lines = ["# V15：连续观察与事实风险分配", "",
        "日期：2026-10-13。单初始化、两组开发迭代；不是独立泛化确认或实机验证。",
        "每组48个身体、393216次新经历，连续混合18种训练组合。",
        "9种留出组合家族在V14已被查看，V15使用新的预先声明世界种子；",
        "不能把同一家族反复开发后的分数声称为独立确认。",
        "风险模型只训练实际执行动作后的碰撞和受击，未执行动作无环境标签。",
        "行动、路由、价值、预测和风险模型各自反向更新，模块之间断梯度。",
        "",
        "| 模式 | 每身体学习步 | 接触成功 | 全回合无碰撞接触 | 受击 | 碰撞步 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for mode, payload in results.items():
        xs = [0, 1024, 4096, 8192]
        values = [payload[str(step)]["heldout"] for step in xs]
        for step, value in zip(xs, values):
            lines.append(f"| {mode} | {step:,} | {value['success']:.2%} | "
                f"{value['safe_success']:.2%} | {value['impact']:.2%} | "
                f"{value['collision_step_rate']:.2%} |")
        for axis, key in zip(axes, ("success", "safe_success", "impact", "collision_step_rate")):
            axis.plot(xs, [v[key] for v in values], marker="o", label=mode)
            axis.grid(alpha=0.2)
            axis.set_xlabel("Physical steps / body")
            axis.set_ylim(bottom=0)
            axis.legend(frameon=False)
    for axis, title in zip(axes, ("Target contact", "Contact without collision",
                                  "Impacts / finished episodes", "Collision steps / steps")):
        axis.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "PROCESS_CURVES.png", dpi=150)
    plt.close(fig)
    lines += ["", "## 冻结权重禁用风险输出", "",
              "| 同一最终权重 | 接触成功 | 全回合无碰撞接触 | 受击 | 碰撞步 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in (("heldout", "启用风险分配"), ("heldout_no_risk", "禁用风险分配")):
        value = results["risk_alloc"]["8192"][key]
        lines.append(f"| {label} | {value['success']:.2%} | {value['safe_success']:.2%} | "
                     f"{value['impact']:.2%} | {value['collision_step_rate']:.2%} |")
    lines += ["", "使用相同外生种子。动作改变会改变回合结束和随后随机消费，",
              "不是逐时间步相同轨迹的配对因果估计；只能作为风险输出是否被",
              "实际使用的有限干预证据。",
              "",
              "## 各场景的最终后果", "",
              "| 模式 | 场景(布局,运动,攻击) | 接触/结束 | 无碰撞接触/结束 | 受击/结束 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for mode, payload in results.items():
        for scene, value in payload["8192"]["heldout"]["scenes"].items():
            lines.append(f"| {mode} | {scene} | {value['success']}/{value['episodes']} | "
                         f"{value['safe_success']}/{value['episodes']} | "
                         f"{value['impact']}/{value['episodes']} |")
    lines += ["", "## 证据边界", "",
              "每帧风险概率在动作之前保存，风险监督取真实下一帧后果。",
              "PPO重复更新使用动作发生时保存的风险修正，风险模型的后续改变",
              "不会篡改旧策略概率。自然频率回放有额外学习量，单独记录；",
              "两组容量和回放预算相同。没有最优动作、未来攻击程序或隐藏场景标签。",
              "",
              "utility仍为事实进展与预测误差的关联估计；动态目录不等于",
              "已证明它产生广义生物学习。即时风险并不保证长时安全，",
              "固定2/4分配系数也不是学习到的全局最优。",
              "每类探针1024时间步，分母为已结束回合，仍在进行的回合单列。",
              "",
              "每帧最近三物体按距离排序，差分可能错配物体身份；",
              "当前仍是理想化物体位置测量，图像仅归档，尚未训练视觉感知。",
              "真实摄像头、遮挡、复杂身体和真实机器人未验证。",
              "",
              "完整数据上传并逐片读回SHA-256校验后才清理，凭据在archives目录。"]
    (ROOT / "REPORT_V15_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
