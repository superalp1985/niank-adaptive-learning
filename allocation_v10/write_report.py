"""Produce the research report from completed, audited artifacts."""

import json
from pathlib import Path
import numpy as np

from core import ROOT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{value * 100:.2f}%"


def main():
    confirmed = read(ROOT / "confirmation" / "SUMMARY.json")
    continued = read(ROOT / "continued_practice" / "SUMMARY.json")
    audits = [read(ROOT / name / "AUDIT.json") for name in ("development", "confirmation", "continued_practice")]
    assert confirmed["completed_runs"] == confirmed["expected_runs"] == 18
    assert continued["completed_runs"] == continued["expected_runs"] == 6
    labels = {"independent": "独立决策", "immediate": "即时共同决策", "delayed": "延迟共同决策"}
    lines = [
        "# V10：共同分配与参数储备实验",
        "",
        "日期：2026-10-12。全部本轮运行结束。结论：尚未证明共同分配算法有效，保留负结果。",
        "",
        "## 做了什么",
        "",
        "实现了连续视觉事件、真实物理反馈、原始记录 SQLite、可学习检索/淡出指针，",
        "以及查回、重学、增加活跃计算、开放/收回参数组的共同控制器。",
        "粗算与分层精度的 Qwen 路线未在本轮实现。",
        "",
        "共完成 3 个开发运行、18 个确认运行、6 个继续实践运行。",
        "确认阶段每个运行 262,144 次新物理转移，3 个初始化/世界条件、2 档参数储备。",
        "条件为 (模型23,世界420100)、(模型37,世界420100)、(模型23,世界430100)，",
        "分别改变模型与世界；这些部分共享随机条件，下面的均值和范围只作描述，不作显著性证明。",
        "继续实践载入既有权重，每个模型追加 786,432 次经历，累计 1,048,576 次；没有重跑旧前缀。",
        "",
        "成功率由实际行动的物理代价判定：距离该状态最优代价不超过 0.02。",
        "网络只从已执行行动的后果学习。稠密模型答案一致性不参与评分。",
        "训练有 10% 随机动作探索；冻结探测没有随机动作探索、没有梯度更新，且禁用重学操作。",
        "冻结探测允许使用过去已发生的误差作为历史状态。",
        "",
        "## 确认结果",
        "",
        "| 参数储备 | 机制 | 全程在线 | 末256步在线 | 已见组合新世界 | 未见物理组合 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in confirmed["groups"]:
        lines.append(f'| {item["total_parameters"]:,} | {labels[item["mode"]]} | '
                     f'{pct(item["online"]["mean"])} | {pct(item["online_last"]["mean"])} | '
                     f'{pct(item["seen"]["mean"])} | {pct(item["composition"]["mean"])} |')
    lines += [
        "",
        "未见组合仅指材料/作用力的 (0,0)、(3,3) 两种组合未参与训练；单个因素均见过。",
        "这是当前简化物理任务的组合测试，不代表开放世界或语言泛化。",
        "",
        "128K 储备下延迟共同决策的组合均分略高，但在 1M 储备下落后于独立决策。",
        "主假设没有稳定跨规模支持。增大总储备也没有带来一致的在线收益。",
        "",
        "## 继续实践结果",
        "",
        "以下只有一个初始化条件，使用相同新世界。Adam、身体与活跃目录统一重新初始化；",
        "因此不能把变化全部归因于学习量。该阶段也不能与从零训练的全程在线分数直接比较。",
        "",
        "| 参数储备 | 机制 | 末256步在线 | 已见组合新世界 | 未见物理组合 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in continued["groups"]:
        lines.append(f'| {item["total_parameters"]:,} | {labels[item["mode"]]} | '
                     f'{pct(item["online_last"]["mean"])} | {pct(item["seen"]["mean"])} | '
                     f'{pct(item["composition"]["mean"])} |')
    lines += [
        "",
        "## 成本与容量",
        "",
        "| 储备组数 | 机制 | 活跃预测参数均值 | 查回次数 | 额外重学样本 | 末期可访问组数 | 估算MAC十亿 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in confirmed["groups"]:
        lines.append(f'| {item["groups"]} | {labels[item["mode"]]} | '
                     f'{item["active_parameters"]["mean"]:,.0f} | {item["retrieved"]["mean"]:,.0f} | '
                     f'{item["replayed"]["mean"]:,.0f} | {item["accessible"]["mean"]:.2f} | '
                     f'{item["estimated_macs"]["mean"] / 1e9:.2f} |')
    lines += [
        "",
        "控制器参数已计入总参数；活跃预测参数只计预测器，调度、反向传播和优化器另有开销。",
        "总参数是预留存储，开放/收回只是改变访问范围，不是新建/销毁参数。",
        "同一组参数在世界间共享，也不能说每次开放的是全新未训练知识。",
        "逐组对比初始化与最终权重：9组储备平均8.67组的输入矩阵发生更新；",
        "78组储备平均27.44组发生更新，最多37组。详见 confirmation/LEARNED_GROUPS.json。",
        "因此不能把这一轮当作128K与1M参数均充分训练后的规模定律比较。",
        "Adam 操作仍使用稠密参数存储；MAC为算法估算，不能称为实际硬件提速。",
        "相同经历数量下，额外学习和检索的实际用量不同，因此不是严格等计算实验。",
        "",
        "另按实际记录截取共同估算MAC上限内的训练前缀，末256步分数如下：",
        "",
        "| 储备组数 | 独立决策 | 即时共同决策 | 延迟共同决策 |",
        "| --- | --- | --- | --- |",
    ]
    for size in (9, 78):
        values = []
        for mode in ("independent", "immediate", "delayed"):
            values.append(np.mean([r["success"] for r in confirmed["matched_estimated_mac_prefixes"]
                                   if r["groups"] == size and r["mode"] == mode]))
        lines.append(f'| {size} | {" | ".join(pct(v) for v in values)} |')
    lines += [
        "",
        "这是已有轨迹的实测前缀比较，不含 optimizer 内存流量或 SQLite 代价的严格配平；",
        "各方法截至上限所接触的新经历数量不同，不把它写成同时等数据、等计算。",
        "",
        "## 发现的限制与下一次改动",
        "",
        "1. 控制器短期训练只有64次更新，追加后累计256次；训练期操作仍需检查是否形成条件选择，",
        "   不能用预测网络已经学会物理代替证明调度器有效。",
        "2. 默认新记录直接进入3槽缓存，而关键线索8步后生效，通常还在约12步覆盖范围内。",
        "   环境可能容许绕开长期检索需求。很少检索可以是合理选择，不是遗忘/恢复理论被否定。",
        "3. 参数储备变大不等于有效使用更多表示能力；每次仍只计算1或2个矩阵组。",
        "4. 新事件自动入缓存、满时淘汰最老项、只向最近8个事件发起查询，仍是工程规则。",
        "   本轮没有完成整个目录的自组织，也没有证明细节天然分级已经形成。",
        "5. 当前恢复状态来自实际缺席期间的经历和损失，不能因而声称某条细节导致了这些损失。",
        "",
        "下一轮优先修正这个尚未得到支持的联合分配机制：让某条经历的恢复成本跟随该经历，",
        "并以晚到反馈更新它和参数组的共同选择，而不是仅把几项恢复统计平均进控制器。",
        "环境要让某些后果跨过当前缓存窗口，但后果发生前仍有可辨认线索；保持新旧条件配对，",
        "避免人为指定哪些细节永久重要。应分别测缓存可覆盖与不能覆盖的条件，",
        "用最终物理行为检验这项计算规则的差异，不重做已有记忆论文的常规结论。",
        "",
        "目前不把8M参数扫描列为自动下一步：先解决资源是否被有效选择以及任务信息需求，",
        "否则只是扩大预留张量。已有失败和原始数据库全部保留。",
        "",
        "## 核验与文件",
        "",
        f'- 6项契约检查通过；独立重算 {sum(a["transitions"] for a in audits):,} 次物理转移，',
        f'  检查 {sum(a["decisions"] for a in audits):,} 次调度决策、原始数据库、预算、划分及源码哈希。',
        "- 两档延迟控制器检查点在 CPU 上回放冻结预测，与原 GPU 预测的误差界见 CHECKPOINT_AUDIT.json。",
        "- 新代码、数据、模型、日志、缓存全部位于 E 盘；未下载新权重，未启动交互推理 CLI。",
        "- 收尾时本轮目录约0.52 GiB；训练及检查任务均已结束。",
        "",
        "入口：`PROTOCOL.md`、`CONTINUATION_DECISION_ZH.md`。",
        "确认：`confirmation/SUMMARY.json`、`confirmation/AUDIT.json`、`confirmation/learning_curves.png`。",
        "追加：`continued_practice/SUMMARY.json`、`continued_practice/AUDIT.json`。",
        "各运行保留 `model.pt`、`events.sqlite`、`transitions.npz`、`result.json`。",
        "",
        "结论边界：本轮完成了新候选机制与规模实验，尚未得到有效新算法或原创性的证明。",
        "",
    ]
    (ROOT / "REPORT_V10_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(ROOT / "REPORT_V10_ZH.md"), "lines": len(lines)}))


if __name__ == "__main__":
    main()
