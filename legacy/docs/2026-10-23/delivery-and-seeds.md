# 目录交付、独立随机种子与实验编排

本轮完成用户指定的两项：生产/历史/参考代码分类，以及实验随机种子拆分和多次运行编排。GPU、独立依赖环境安装、正式参数和正式性能验证仍不属于本轮已完成范围。当前本地严格验收见 [delivery-verification.json](delivery-verification.json)。

## 1. 当前目录与唯一实现

```text
gib4cl/
  config.py                    v2 配置与独立种子
  runtime.py                   执行、来源、原子保存与随机状态
  data/                        数据契约、模拟、生成、加载
  learning/                    GNN、论文目标、训练和检查点
  inference/                   粒子 BP、多源 BFS 森林
  evaluation/                  指标、拓扑选择、报告、可选绘图
  experiments/                 主入口、种子面板编排、独立子进程执行
configs/
  *.json                       单次实验配置
  suites/*.json                固定数据、多训练/BP 种子面板
tests/
  test_*.py                    回归和边界检查
  references/                  独立数学参照
  verify_stage*.py              可复验的阶段审计
tools/
  provenance.py                迁移来源、核心算法与参考文件校验
  accept_cpu.py                一条命令完成当前 CPU 严格验收
legacy/                        旧 FIM 目标和旧配置常量
references/NEBP/               原始外部参考源码，逐字保留
GNN_learning/、BP/、NBP_ST/     旧导入路径/启动兼容，均无第二套算法
docs/                          当前说明、阶段证据、不可变快照
data/、outputs/                生成产物，默认不进 Git
```

正式调用统一使用 `python -m gib4cl`。三个旧源码目录保留薄兼容层，便于已有调用和数据类型导入迁移；新开发应直接导入 `gib4cl`。例如：

| 旧位置 | 当前唯一实现 |
| --- | --- |
| GNN_learning/experiment_config.py | [gib4cl/config.py](../gib4cl/config.py) |
| GNN_learning/generate_network.py、generate_datasets.py、dataset.py、data_schema.py | [gib4cl/data/](../gib4cl/data/) |
| GNN_learning/model.py、objectives.py、train.py、checkpoints.py | [gib4cl/learning/](../gib4cl/learning/) |
| BP/particle_bp.py | [particle_bp.py](../gib4cl/inference/particle_bp.py) |
| NBP_ST/BFS_tree.py | [spanning_tree.py](../gib4cl/inference/spanning_tree.py) |
| GNN_learning/metrics.py、topology.py、evaluate.py | [gib4cl/evaluation/](../gib4cl/evaluation/) |
| GNN_learning/experiment.py | [cli.py](../gib4cl/experiments/cli.py) |
| BP/reference.py | [独立参照](../tests/references/particle_posterior.py) |
| GNN_learning/loss.py 的旧 FIM/CRLB、config.py 常量 | [legacy/](../legacy/) |
| NEBP/ | [references/NEBP/](../references/NEBP/) |

生产入口、训练和评价均不导入 legacy、NEBP、测试参照或旧路径。旧 loss 入口默认只导出当前 compute_gib_loss，显式索取旧函数时才延迟导入历史代码并发出弃用提示。绘图模块的 matplotlib/NetworkX 为可选依赖，不进入训练/评价主流程。

`pytest.ini` 将默认测试发现范围固定为 tests，避免将外部 NEBP 的测试/训练程序混入本项目验收。

整理前源码保存在 `docs/baselines/2026-10-23-pre-layout/`，第一步快照继续保留。完整路径对应见 [source-layout.json](source-layout.json)。三份核心算法通过去除导入语句后的完整 AST 与已验证快照比较，除导入位置外语义未改；同时复跑所有数学参照。NEBP 八个源码文件与第一步快照逐字一致。

`.gitattributes` 将生产源码和配置等文本固定为 LF，避免不同系统检出时只因换行转换改变源码校验值；NEBP 与不可变快照标为保留原始字节，不转换其换行。最终候选版本应连同该属性文件一起保存。

## 2. 四类独立种子

当前配置协议为 `gib4cl.config.v2`：

```json
{
  "config_version": "gib4cl.config.v2",
  "seeds": {
    "data": 100,
    "training": 101,
    "bp": 700,
    "bootstrap": 1700
  }
}
```

| 字段 | 控制范围 | 更改时的数据指纹 |
| --- | --- | --- |
| seeds.data | 各 split/样本的图规模、位置、初值、测距等数据随机流 | 改变 |
| seeds.training | 参数初始化、训练边采样、训练/验证 DataLoader 随机状态 | 不变 |
| seeds.bp | 每图、每 BP 重复的先验支持，Dense/BFS/GNN 共享 | 不变 |
| seeds.bootstrap | 图聚类区间和训练种子区间的重抽，分别使用不同派生命名空间 | 不变 |

种子须为 [0, 2^63) 的整数，拒绝 bool。数据样本仍由数据种子、split、offset、样本编号派生；BP 支持仍绑定 dataset_id/sample_id/重复编号，保持遍历顺序不影响结果。

数据指纹继续使用历史的编码 `hash({data, seed: seeds.data})`。六份已有场景/验证配置的数据指纹未改变，可以显式复用原数据。固定数据意味着复用已经生成并校验的数据文件；重新生成时创建时间/来源元数据可能变化，不承诺重新生成文件的二进制字节相同。

旧 v1 JSON 可读入并显式迁移：旧顶层 seed 同时映射为 data/training，旧 evaluation.seed 同时映射为 bp/bootstrap，从而保持旧设定的实际数值。保存时统一写 v2。v2 拒绝继续混入旧 seed 字段。cfg.seed 仅保留只读“数据种子”兼容属性，新代码不使用它。

旧版当前架构的权重/完整检查点仍可用于显式初始化或评价。迁移后的源码和配置不再满足旧训练的原样恢复身份要求；不能把它当作原实现下的精确续训。新版本同环境、epoch 边界恢复继续接受完整状态校验。

CLI 可覆盖相应角色：

```text
python -B -m gib4cl generate --config configs/smoke-pipeline.json --data-seed 100 --data-root data/demo
python -B -m gib4cl train --config configs/smoke-pipeline.json --training-seed 101 --data-root data/demo --output-root outputs/demo --device cpu --threads 1
python -B -m gib4cl evaluate --checkpoint 实际best_checkpoint.pt --data-root data/demo --bp-seed 700 --bootstrap-seed 1700 --output-root outputs/eval --device cpu --threads 1
```

程序 API 使用 `with_seeds(cfg, training=101)` 等显式修改。已有真实评价检查证明：只改 bootstrap 种子，逐条估计/掩码/支持均不变；只改 BP 种子，支持变化而 GNN 选择不变；训练种子变化使模型参数摘要变化，同时数据文件 SHA-256 保持不变。

## 3. 固定数据的实验面板

一个 suite 使用一个明确的基础配置和固定数据集。训练种子与 BP 种子构成面板；多场景/超参数条件使用不同基础配置和 suite_id，分别归档。

示例 [smoke-seeds.json](../configs/suites/smoke-seeds.json) 指定两种训练种子、两种 BP 种子：

```json
{
  "suite_version": "gib4cl.suite.v1",
  "suite_id": "smoke-seeds-v1",
  "base_config": "../smoke-pipeline.json",
  "training_seeds": [101, 102],
  "bp_seeds": [700, 701],
  "split": "test",
  "max_graphs": null,
  "seed_bootstrap_samples": 100
}
```

base_config 相对 suite 文件所在目录解析。种子列表非空且不可重复；规范为升序，使遍历顺序稳定。评价只允许 val/test。正式基础配置仍须填写明确的损失权重，程序不会借用 smoke 系数。

先查看完整解析后的配置和任务表，不生成数据或模型：

```text
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --plan
```

执行：

```text
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --output-root outputs/panel-demo --device cpu --threads 1
```

默认在该 suite 的 data 子目录生成一次 train/val/test；所有训练与评价复用这三个文件。有现成数据时显式传 `--data-root 实际数据根 --reuse-data`。会校验文件与 sidecar、身份、划分和图数，不将不同数据当同一实验。

该 smoke 面板有 2 个训练任务和 4 个评价任务，评价任务分别运行三种拓扑；每次评价沿用基础配置中的 bp_repeats=2。原始记录仍是每图/每 BP 重复/每拓扑，所有检查点与结果校验值可追溯。面板顺序执行，每项在独立 Python 子进程运行，避免多实验争抢资源和全局状态串扰。

每个任务有短路径的 attempt 目录、解析后配置、实际命令、stdout/stderr 日志、退出码、开始/结束时间和产物校验值。内部 worker 直接使用该任务的唯一产物目录，避免在 Windows 上反复嵌套实验 ID/哈希。它仍调用同一 train/evaluate 实现，不包含另一套算法。

## 4. 恢复、失败与重试

新输出目录拒绝覆盖。以下命令在相同 suite/输出根上恢复：

```text
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --output-root outputs/panel-demo --resume --device cpu --threads 1
```

- 完成任务先核对配置、数据、检查点/报告及日志/配置校验，再跳过，不重复训练。
- 未执行的 pending 任务继续执行。
- 失败任务完整保留错误、退出码、日志和尝试历史；依赖失败训练的评价标为 blocked。其他训练种子仍继续处理。
- 失败/中断/running/blocked 状态须显式 `--resume --retry-failed` 才重试；重试创建新 attempt，保留先前记录。未完成的训练任务在新 attempt 从头启动，任务内的 epoch 续训仍使用已有 `train --resume` 接口。
- `--stop-after-jobs 1` 可在完整任务边界停止，用于分段执行和验收；它不修改训练的 epochs 计划。suite 恢复粒度是任务，单次训练恢复粒度是 epoch。
- 配置、源码或执行环境不一致时拒绝恢复。完成产物缺失/校验失败时拒绝继续，不能静默以新文件替代。
- 已完成的整个面板再次恢复时只校验并返回，manifest 不改变。

失败运行以非零退出码结束，不能因生成了部分结果就当成功。只有所有计划任务都完成才写最终 summary.json；否则写带缺失种子列表的 partial_summary.json。当前有效摘要由 manifest 中的路径和校验值指向，历史 partial 文件仍保留。

## 5. 多训练种子统计的两层口径

每个训练模型先合并其固定 BP 种子面板。BP 重复编号重新映射为唯一编号，但独立统计单位仍是 sample_id；不会把多个 BP 种子或节点当作更多独立图。每模型输出与第七步一致的图级均值、区间、曲线和配对差值。

跨训练种子对每个模型的图级均值再次汇总，输出训练种子均值、样本标准差、可用训练种子数和 bootstrap 区间；覆盖/中断曲线及 GNN-minus-Dense 差值也有相同统计。只有一个有效训练种子时 std/CI 为 null。

这层区间**条件于固定的数据集与 BP 种子面板**，用于表征训练随机性。它与每模型的图聚类区间分开，不宣称已给出同时覆盖数据与训练变化的联合区间。

Dense/BFS 在相同数据、相同 BP 面板上必须跨训练模型保持一致。程序核对后只保存一份 fixed_baselines；这些重复副本不增加独立试验数。该 smoke 面板的测试图仍然只有两张，不会变为“2 模型 × 2 BP 种子 × 2 重复 × 2 图”张独立图。

面板根目录输出：

| 文件 | 内容 |
| --- | --- |
| manifest.json | 固定来源/配置/运行身份、数据身份、任务状态、每次尝试和全部产物校验 |
| summary.json / partial_summary.json | 每训练种子完整图级统计、固定基线、跨训练种子 GNN/配对差值/曲线统计、缺失任务 |
| summary.csv / partial_summary.csv | 跨训练种子的标量与配对差值，保留有效数量和空值 |
| report.md / partial_report.md | 可读的完成状态与训练种子变化 |
| jobs/*/attempt-*/ | 子任务配置、日志、检查点、逐次评价 JSON/CSV/报告 |

## 6. 本机严格验收

在仓库根目录、已具备依赖的环境中，使用一个全新的验收目录：

```powershell
.\.venv-data\Scripts\python.exe -B -m tools.accept_cpu --output-root outputs/my-acceptance-01
```

Linux 或新环境将解释器前缀换成该环境的 python。重复验收应换一个输出根，不覆盖既有证据。

该命令真实执行以下检查，任一失败都以非零退出码结束，并将 acceptance.json 标为 failed：

1. 生产区不反向导入历史/参考/测试或旧兼容层；旧路径无第二套算法。三份核心 AST 与不可变快照匹配，NEBP 源码逐字匹配。
2. 全部单元/集成测试，包括生产包脱离旧目录后独立完成流程、四类种子影响、失败注入/显式重试、缺失/损坏产物拒绝。
3. 第六步独立审计：连续训练与 epoch 恢复的模型/优化器/调度/历史/RNG 等完全一致；微调来源及新优化器正确。
4. 第七步独立审计：完整流程与 NumPy/DFS 复算、JSON/CSV/报告、单图/零边/全部无效边界。
5. 真实子进程执行 2 训练种子 × 2 BP 种子面板，中断于首个任务后恢复，校验成功任务未重做、数据字节不变，再检查完成状态下重复调用不改 manifest。
6. 用同一批数据在新目录重新执行整个面板，比较两种训练模型的参数摘要以及四次评价的逐条记录/摘要，要求完全一致；确认两个训练种子确实产生不同参数，独立测试图数量仍为两张。

通过条件：

- acceptance.json 的 status 为 passed；当前集合应为 **273 passed、1 skipped**，唯一允许跳过原因为无 CUDA。未来新增测试后应使用对应候选版本的计数。
- resume-audit.log、metrics-audit.log 无失败，完整审计 JSON 可读取。
- 两个面板全部任务 complete；不存在 failed、blocked、running 或 missing seeds。
- 面板恢复/重放、数据 SHA、模型摘要和逐条数值核对均通过。
- 源码在整个验收期间保持不变，保留 acceptance.json、tests.xml、各阶段日志及面板 manifest/summary。

本次实际通过的证据保存在 `outputs/cpu-accept-01/`，冻结摘要见 [delivery-verification.json](delivery-verification.json)。当前验收是本机已有 Python 环境中的源码/流程验收，生产包隔离复制测试也复用该解释器依赖；不能把它称为全新依赖环境安装验收。外部论文文件若存在会核对；源码单独检出时使用不可变快照中的论文记录，不要求论文位于 checkout 的上级目录。

## 7. 正式实验仍需完成的门槛

本轮没有选定正式损失权重、粒子数、迭代数、训练种子数量或数据规模。两轮 smoke 模型和两张测试图只用于验收。GNN 选空图及稠密 BP 未收敛状态继续如实报告。

进入正式实验前仍应完成：独立环境安装与版本锁定、服务器 GPU 的训练/评价/恢复数值检查、目标规模耗时和峰值资源检查、验证集预实验、正式配置与模型选择规则冻结。正式多训练种子已具备实现入口，但当前小面板不构成正式性能结果。
