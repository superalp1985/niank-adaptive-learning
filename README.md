# Niank Adaptive Learning

自适应稀疏学习研究：动态激活、时序目录、遗忘与重学、参数分配和分层精度。
通过物理仿真提出、检验和修改算法，保留失败结果。

## 当前状态

- [V10 研究报告](allocation_v10/REPORT_V10_ZH.md)
- [V11 研究报告](allocation_v11/REPORT_V11_ZH.md)
- [V12 研究报告](allocation_v12/REPORT_V12_ZH.md)
- [最新继续入口：实体追踪](RESEARCH_STATUS_20261013_TRACKING_ZH.md)
- [V14 多场景实体追踪](tracking_v14/REPORT_V14_ZH.md)
- [V15 连续观察与事实风险](tracking_v15/REPORT_V15_ZH.md)
- [V16 安全到达与学习量](tracking_v16/REPORT_V16_ZH.md)
- [V17 360度复眼与方位错误诊断](tracking_v17/REPORT_V17_ZH.md)
- [V18 方位修复与持续学习过程](tracking_v18/REPORT_V18_ZH.md)
- [V13调研：多场景共同学习，尚未实验](adaptation_v13/RESEARCH_REVIEW_20261013_ZH.md)
- [归档和清理记录](ARCHIVE_STATUS_20261012_ZH.md)
- [研究原则](RESEARCH_PRINCIPLES_ZH.md)
- [已有工作与候选算法](NOVEL_ALGORITHM_PLAN_20261012_ZH.md)
- [遗忘与反复分级原则](RETENTION_PRINCIPLE_20261012_ZH.md)
- [局部学习与全局最优边界](LOCAL_LEARNING_CONSTRAINT_20261012_ZH.md)

V10 已完成 27 个运行。延迟共同决策没有稳定优于独立决策。
约128K与992K为总参数储备，尚非真实动态创建/销毁参数。
实验只检验软件可实现的机制，不声称复现生物脑动力学。

V11 已完成 4 个配置，反馈随具体经历和矩阵组保存。
早期优势未持续，最终未见组合分数低于平均反馈对照；
冻结策略未学会查回旧线索。保留全部负结果，不宣称已获得有效新算法。
对照的路由端还有平移抵消限制，见
[V11 解释补充](V11_INTERPRETATION_LIMIT_20261012_ZH.md)。

V12 已完成三组，每组连续学习至1,048,576次新经历。
屏蔽检索计划当前状态查询的版本最终未见组合成功率90.21%，
冻结权重后禁用数据库读取降至74.19%，预测误差与物理代价也变差。
完整状态+内容版本只有79.33%，不支持“输入更完整必然更好”。
这是单种子探索结果；局部模块使用各自事实反馈，不保证全局最优。
全部3,468,288次转移的独立审计通过，学习曲线和限制见V12报告。

2026-10-13：V13顺序反转训练草案已撤回，未实现或运行。
随后V14-V16已完成六组多场景实体追踪实验。更多实践提高安全接触；
固定风险扣分失败。安全回合反馈改善无碰撞接触，也增加了碰撞后的
连续接触代价，保留这项取舍。下一步优先遇阻后的绕行和恢复。
所有报告关注跨场景的学习过程和更优选择，不将最终得分作为唯一目标。

V18已完成两组共享V17权重的方位修复比较，每组新增589824条实际经历。
1376256条训练/观察记录及全部方位摘要通过核验。修正组最终碰撞更少，
但接触和攻击应对落后于旧对照，实际练习期也未形成优势。当前门控变化
平均偏弱；下一步研究价值信号，不能用预测误差替代实际行动质量。

## 数据与恢复

Git 保存代码、协议、报告、配置、统计摘要和归档清单。
原始数据库、轨迹、模型和完整运行快照保存在 GitHub Releases。
大文件上传并从远端重新下载校验后，才清理本地已归档副本。

每个版本的 `archives/<tag>/manifest.json` 保存原始文件 SHA-256、
归档分片 SHA-256、大小和恢复路径。恢复单个版本：

```powershell
python -B tools/archive_snapshot.py restore --tag <tag> --destination E:\research-restored
```

发布流程和本地保留规则见 [ARCHIVE_POLICY.md](ARCHIVE_POLICY.md)。
历史二进制不存在于当前 checkout 时，应先恢复所需版本，不能将缺少数据库误判为实验未做。

## 运行

V14-V16使用Python、NumPy、PyTorch 2.7.1+cu128、Pymunk 7.1.0和CUDA GPU；
较早版本使用PyTorch 2.3.1+cu118。
报告绘图使用 Matplotlib；早期连接组读取另外需要 pandas、pyarrow。
运行实验前阅读对应目录的协议。已完成配置不自动重跑；新实验使用新输出目录。

V10 工程契约检查：

```powershell
python -B allocation_v10/test_contracts.py
```

源码中的历史绝对路径描述原始实验环境，并非跨机器安装保证。
上游资源归属和固定版本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
