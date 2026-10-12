# Research workflow

直接，不拖泥带水。保留失败，不把细节永久定性为有用或无用。

实验结束后必须上传代码、报告和完整数据归档到
`superalp1985/niank-adaptive-learning`。
使用 `tools/archive_snapshot.py` 建立清单、上传和远端下载校验，
再使用 `tools/cleanup_verified.ps1` 清理已核验的本地旧数据。
上传或校验失败时保留原件；不要继续产生大量新结果。

数据、缓存、临时归档、训练输出均放 E 盘。
只保留必要工作集，不能删除共享 CUDA 环境、原始模型或其他项目。
每次真正修改算法，使用新的实验目录；不要重跑已完成配置。
