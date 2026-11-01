# GIB4CL 可交付实验工程

当前目录只保留一套生产实现：`gib4cl/`。原有 BP、GNN_learning、NBP_ST 兼容目录、旧 FIM、NEBP、旧阶段脚本和快照均集中在 `legacy/`，不再是顶层入口，也不参与发布包运行。

第二章树／单环粒子 BP 独立实验已提供[运行说明](docs/tree-loop-experiment.md)与[实验计划](docs/tree-loop-overconfidence-plan.md)。该入口不训练 GNN，支持精确后验参照、数值预实验、成对统计和左右对照出图。

关于 finite support、后验可信区间、覆盖率和 NEES 卡方检验的读法，见[指标与图形解读](docs/tree-loop-reader-guide.md)；七节点树和 3／4／5 环拓展见[拓展实验说明](docs/tree-loop-extensions.md)。

当前论文图表暂不使用 NEES，校准图采用阴影置信带，正文与附录的组织见[选图建议](docs/tree-loop-paper-selection.md)。

当前只有一个生产场景：四角加中心五个锚点、智能体在整个正方形区域均匀分布、LOS 加性高斯测距。所有测量边使用同一个 `comm_radius`，默认 30 m。正式配置为 [experiment.json](configs/experiment.json)；[smoke-pipeline.json](configs/smoke-pipeline.json) 使用相同规则的小规模数据检查流程。正式 BP 预算、验证规则及损失系数的选择流程见 [实验选择方案](docs/experiment-selection.md)。

本地只检查场景、Dense/BFS 和示例选边掩码时，可运行 `python -B -m tools.preview_network --num-agents 30`，输出图片、配置及原始图。脚本位于 `tools/`，操作说明见 [本地模块测试](docs/local-module-tests.md)。

| 目录/文件 | 用途 | 源代码发布包 |
| --- | --- | --- |
| gib4cl/ | 数据、GNN、论文目标、BP、指标、训练及多种子编排 | 包含 |
| configs/ | 单次实验及 suite 配置 | 包含 |
| tests/ | 当前回归、独立数学参照、便携验收审计和来源校验数据 | 包含 |
| tools/ | 来源核验、打包/校验、完整 CPU 验收 | 包含 |
| docs/ | 当前算法、配置、运行和验收说明 | 包含 |
| requirements.txt、pytest.ini、属性文件 | 依赖列表、测试发现及换行规则 | 包含 |
| legacy/ | 历史源码、兼容层、外部原始参考、旧记录与快照 | 排除 |
| .venv-data/、data/、outputs/、dist/、.git/、.vscode/ | 本地环境、生成产物、发布产物和工作区元数据 | 排除 |

请使用白名单打包程序，不需要手动挑选文件：

```text
python -B -m tools.package_project build --output-root dist/my-release-01
python -B -m tools.package_project verify --archive dist/my-release-01/Graph-Learning-BP.zip
```

输出 ZIP、包内/包外相同的 PACKAGE-MANIFEST.json 以及 ZIP 的 SHA-256 文件。包只包含当前源代码和验收所需文件，所有文件逐项记录大小与校验值。

解压后进入 Graph-Learning-BP，先激活装有项目依赖的 Python 环境，再执行：

```text
python -B -m tools.package_project verify --directory .
python -B -m tools.accept_cpu --output-root outputs/acceptance-01
```

完整 CPU 验收会先检查发布包所有源码的校验值，再执行数据/目标/GNN/BP/指标回归、独立恢复与指标审计，以及真实多种子面板的恢复、幂等和整套重放。验收使用新输出目录；失败以非零退出码结束。

本机源码目录已有解释器为 `.\.venv-data\Scripts\python.exe`，可替换上述 python；该环境不打入 ZIP。详细步骤和通过条件见 [完整验收说明](docs/acceptance.md)。

训练与评价统一入口：

```text
python -B -m gib4cl --help
python -B -m gib4cl run --config configs/smoke-pipeline.json --data-root data/demo --output-root outputs/demo --device cpu --threads 1
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --plan
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --output-root outputs/panel --device cpu --threads 1
```

用户配置只接受当前一种结构，分别定义数据、训练、BP、bootstrap 种子，不再填写配置、模型或 BP 版本。数据和完整检查点保留读取格式校验，结果保留指标协议；无读取校验用途的运行与结果版本标签已删除，模型权重只保存参数张量。旧场景字段、旧种子别名和旧数据列表不再转换；请按新配置在新目录生成数据。场景配置精简产生新的指纹，不复用旧场景数据；随后移除算法标签不改变当前数据指纹。固定数据的多训练种子统计与图级不确定性分开，Dense/BFS 的重复副本不增加独立样本数量。

本工程已具备完整的 CPU 小规模验收入口；正式损失系数、BP 预算、独立依赖环境和服务器 GPU 数值/资源仍需相应验收。smoke 用于流程检查。

文档：[运行](docs/unified-experiment.md)、[种子与编排](docs/delivery-and-seeds.md)、[数据](docs/data-contract.md)、[目标](docs/training-objective.md)、[GNN](docs/gnn-sampling.md)、[BP](docs/particle-bp.md)、[指标](docs/evaluation-metrics.md)、[论文对应](docs/paper-implementation-map.md)、[来源](docs/provenance.md)。
