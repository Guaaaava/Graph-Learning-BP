# GIB4CL

合作定位图选择、GNN 训练与粒子信念传播实验工程。当前生产工程位于 [Graph-Learning-BP](Graph-Learning-BP/README.md)。本目录可整体作为 GitHub 仓库上传，并在新主机上独立使用。

## 目录

- `Graph-Learning-BP/`：生产代码、配置、测试、运行文档与实验工具。
- `planning/`：固定测距噪声及 4090 迁移的检查与后续计划。
- `experiment-reference/`：三个旧实验运行器的源码参考，不作为新实验入口。
- `GIB-Chap234-1110.tex`：与当前来源登记对应的论文草稿。
- `SOURCE-ORIGINS.json`：整理时的源文件出处；`FINAL-MANIFEST.json`：交付文件的校验摘要。

未包含历史数据、结果、模型权重、Python 环境、缓存、旧 Git 历史、SSH 信息及 legacy 代码。核心工程运行不依赖旧工作区。保留来源登记与数学参考测试，不需要下载 NEBP 或恢复旧快照。

## 当前实验设置

本次只整理工程，保留现有算法与配置行为。**正式配置仍为 `σ(d)=0.5+0.05d`，尚未切换为固定 1 m；正式目标系数仍待显式确定。请先完成 [新实验计划](planning/固定噪声与4090实验计划.md) 中的配置修改和 4090 验收，再开始正式训练。** smoke 配置可用于流程检查，其结果不代表正式性能。

## 安装与检查

在仓库根目录创建独立环境，并使用该环境的 Python。建议 Python 3.11 或 3.12；新主机的实际依赖组合需验收。先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装适合主机驱动的 PyTorch 2.7.1 CUDA 构建，再安装工程依赖。`requirements.txt` 是现有版本基线，不是 4090 的完整环境锁定文件。

```text
cd Graph-Learning-BP
python -m pip install -r requirements.txt
python -B -m gib4cl --help
python -B -m tools.accept_cpu --output-root outputs/acceptance-01
```

CPU 验收包含完整回归、训练恢复、独立指标审计及多种子 smoke 重放；不代表 GPU 性能验收。失败时查看相应输出目录中的日志。生成目录已被 Git 忽略。

```text
python -B -m gib4cl run --config configs/smoke-pipeline.json --data-root data/demo --output-root outputs/demo --device cpu --threads 1
```

## 保留的补充实现

- `tools/bp_cached.py`：原有 BP 缓存/性能工具，保持当前测试与工具接口。
- `tools/bp_cached_fast.py` 与 `tools/bp_distance.py`：从旧正式实验快照提取的二维 float64 距离加速实现。可通过 `from tools.bp_cached_fast import CachedParticleBP` 显式使用，默认 `cache_bytes=0`；尚未改成统一评价入口的默认后端，需要在 4090 重新验收。
- `tools/baseline_topologies.py`：森林与中心 BFS 单树选边函数，不依赖历史运行目录。中心单树沿用 100 m 场景、中心锚点 `(50,50)`，不连通时返回 `None`。固定噪声下 `variance_msf` 为按边 ID 打破平局的等权森林。

新正式实验应建立新的输出目录和冻结协议。不要把旧运行器的 5090 环境身份检查取消后作为同一次运行续接。

更详细的模块说明和命令见 [工程 README](Graph-Learning-BP/README.md)；本次实际验证范围见 [交付检查](DELIVERY-CHECK.md)。
