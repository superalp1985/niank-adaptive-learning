# Niank Adaptive Learning

自适应稀疏学习研究：动态激活、时序目录、遗忘与重学、参数分配和分层精度。
通过物理仿真提出、检验和修改算法，保留失败结果。

## 当前状态

- [V10 研究报告](allocation_v10/REPORT_V10_ZH.md)
- [V11 研究报告](allocation_v11/REPORT_V11_ZH.md)
- [最新继续入口](RESEARCH_STATUS_20261012_V11_ZH.md)
- [归档和清理记录](ARCHIVE_STATUS_20261012_ZH.md)
- [研究原则](RESEARCH_PRINCIPLES_ZH.md)
- [已有工作与候选算法](NOVEL_ALGORITHM_PLAN_20261012_ZH.md)
- [遗忘与反复分级原则](RETENTION_PRINCIPLE_20261012_ZH.md)

V10 已完成 27 个运行。延迟共同决策没有稳定优于独立决策。
约128K与992K为总参数储备，尚非真实动态创建/销毁参数。
实验只检验软件可实现的机制，不声称复现生物脑动力学。

V11 已完成 4 个配置，反馈随具体经历和矩阵组保存。
早期优势未持续，最终未见组合分数低于平均反馈对照；
冻结策略未学会查回旧线索。保留全部负结果，不宣称已获得有效新算法。

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

现有运行使用 Python、NumPy、PyTorch 2.3.1+cu118 和 CUDA GPU。
报告绘图使用 Matplotlib；早期连接组读取另外需要 pandas、pyarrow。
运行实验前阅读对应目录的协议。已完成配置不自动重跑；新实验使用新输出目录。

V10 工程契约检查：

```powershell
python -B allocation_v10/test_contracts.py
```

源码中的历史绝对路径描述原始实验环境，并非跨机器安装保证。
上游资源归属和固定版本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
