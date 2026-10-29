# 第六步⑥：合并生成树、训练及评估入口

**当前交付入口**：生产代码已迁入 `gib4cl/`，主命令为 `python -m gib4cl`，配置升级为 v2 并分离四类种子；多种子编排和一键 CPU 验收见 [delivery-and-seeds.md](delivery-and-seeds.md)。本页保留第六/七步的历史定义与验收记录，旧导入和 CLI 由薄兼容层转发。

主入口为 [experiment.py](../GNN_learning/experiment.py)，统一提供 `generate / train / evaluate / run`。本阶段验证证据见 [stage6-verification.json](stage6-verification.json)，完整本地审计在 `outputs/stage6-delivery/`。论文、第一步快照及 NEBP 源码保持不变，已验证的 GNN、训练目标和 BP 数学实现也保持不变。

**第七步补充（2026-10-23）**：当前评价已升级为 `gib4cl.evaluation.v2`，新增精度尾部/中断曲线、校准曲线/高斯评分/椭圆面积、通信节省/环秩/分量/度数、归一化 ESS 诊断及两个 CSV 导出；精确定义与最新验收见 [evaluation-metrics.md](evaluation-metrics.md)。下文的第六步验收和文件口径保留为历史记录，主入口用法不变。

## 1. 单一实现与兼容入口

| 内容 | 唯一实现 | 说明 |
| --- | --- | --- |
| 多源 BFS 森林 | [BFS_tree.py](../NBP_ST/BFS_tree.py) | 两套旧逻辑合并，无 NetworkX 依赖 |
| 训练与恢复 | [train.py](../GNN_learning/train.py) | 固定新版目标、完整检查点、显式微调 |
| 检查点加载 | [checkpoints.py](../GNN_learning/checkpoints.py) | 实际路径、架构、内容摘要与状态完整性检查 |
| 拓扑选择 | [topology.py](../GNN_learning/topology.py) | Dense / BFS / GNN 共用接口 |
| 指标与统计 | [metrics.py](../GNN_learning/metrics.py) | 明确 NEES/覆盖率、图级统计及无效值 |
| 评估与报告 | [evaluate.py](../GNN_learning/evaluate.py) | 必须指定检查点；共享 BP 支持；严格 JSON |
| 配置、身份、随机状态与原子保存 | [experiment_config.py](../GNN_learning/experiment_config.py)、[runtime.py](../GNN_learning/runtime.py) | 数据身份不变，训练/评价配置可分别记录 |

原 `GNN_learning/train.py` 和 `evaluate.py` 的直接调用及模块调用均转到统一入口。两个历史实验文件 `exp_50agent.py/exp_anneal.py` 现在也是薄启动器，接受相同子命令，不再自行训练、搜索旧 FIM 系数、启动旧评估或硬编码 CUDA；导入时不执行实验。历史代码保存在第一步快照中，后续再做整体目录分类。

## 2. BFS 定义（D12）

名称为**确定性多源 BFS 森林**：

1. 全部锚点按全局节点 ID 升序同时入队，邻居也按节点 ID 升序访问。
2. 仅在首次发现未访问节点时选择相应候选边。多个锚点会得到多棵树，不会强制保留所有锚点边。
3. 仍未访问的分量，从其中最小节点 ID 继续 BFS，形成未锚定的树；孤立节点保留为单节点树。
4. 不添加候选图中不存在的边，不宣称能把原本不可达锚点的分量连接起来。

结果返回物理边掩码、每个节点的森林根和锚点可达标志。改变物理边排列或端点存储方向不改变对应选择。节点 ID 是明确的平局处理依据，因此不声称不同重编号下总得到同一棵树。

旧 `extract_bfs_tree_edges` 与 Data 接口 `extract_bfs_weights` 均调用这一实现；旧局部锚点索引仍明确拒绝。

## 3. 训练、模型选择与检查点

训练继续使用第三、四步目标与 Logistic STE，没有恢复旧 lambda 退火。AdamW 权重衰减、梯度范数上限、最低学习率和确定性开关加入显式训练配置；原值分别为 1e-4、5、1e-5 和 true。温度按绝对 epoch 计算，恢复后不重新从首轮退火。

每个完成的 epoch 原子保存：

| 文件 | 用途 |
| --- | --- |
| `models/last_checkpoint.pt` | 最新完整状态，用于恢复训练 |
| `models/best_checkpoint.pt` | 固定目标下验证集总损失最小的完整状态，用于评价 |
| `models/best_model.pth` | 同一最佳模型的纯 state_dict，供显式初始化或旧验证脚本使用 |
| `models/final_model.pth` | 最近完成轮次的纯 state_dict |
| `config.json / run.json / history.json` | 解析后配置、数据/源码/初始权重身份、状态和分项训练记录 |

完整检查点版本为 `gib4cl.checkpoint.v1`，包含模型、优化器、学习率调度器、epoch、最佳值/轮次、完整历史、Python/NumPy/PyTorch/CUDA 随机状态和两个 DataLoader 的独立随机状态。还记录训练/验证数据 ID 与校验值、运行环境及实际执行源码（包括损失入口）的校验值。

验证在确定性阈值子图上进行，按图数汇总最后一个不足 batch 的批次。最佳模型按同一组固定系数下的验证 total 选择，严格更小时更新；相同值保留较早轮次。不使用测试集选模。

### 恢复与微调是不同操作

- `--resume last_checkpoint.pt`：继续原计划、优化器、学习率和随机流。默认继承保存的设备与线程数；显式改动配置/调度、数据校验值、执行源码或环境会拒绝恢复。验证过的是同一 CPU 环境下的 epoch 边界恢复，不是中途 batch 的逐步恢复或跨硬件逐位一致保证。
- `--init-from PATH`：先加载指定参数，再新建优化器、调度器与随机流，写入新输出目录。可明确改变场景、训练系数和训练时长；模型结构仍须兼容。初始模型摘要和来源文件校验值会记录在 run.json。
- 完整检查点和当前架构的纯 state_dict 均可用于 `init-from`；旧架构缺版本/尺寸不符仍拒绝加载。
- 已有更晚轮次时，不允许用较早的最佳检查点覆盖历史；要分支实验请初始化一个新 run。更改总 epochs 也属于新计划，使用 `init-from`，不静默改变已执行的余弦调度。
- 新训练/评估目录拒绝覆盖。结果与数据通过身份和校验值对应，移动整个 run 后可在满足相同环境/源码和数据身份要求时继续；原绝对路径仅作为来源记录。

当前 CPU 验证中，“连续两轮”与“首轮后恢复第二轮”的模型、优化器、调度器、历史、随机状态及最佳轮次完全一致。微调的初始参数摘要与指定来源完全一致，首轮后参数改变，优化器 step 从新实验开始。

## 4. 必须明确使用被评价的模型

评价要求 `--checkpoint`，读取完整检查点中的模型结构并与评价配置核对；报告记录实际绝对路径、文件 SHA-256、模型参数摘要、epoch、训练配置及训练数据身份。缺失、损坏、旧架构或不匹配的模型明确报错，不再以随机权重替代。

主入口的 `evaluate` 接受版本化完整检查点，不能把纯 `best_model.pth` 当作含完整来源的检查点。旧验证权重可先显式 `init-from` 到新训练，再评价产生的完整检查点。

评价可显式给另一份数据配置，用于跨场景/规模实验；模型结构必须兼容，报告分别保存训练配置和本次评价配置。数据集内部的 split、数据指纹、实际生成范围和文件校验也会核对，不能把改名后的其他 split 静默当作测试集。

GNN 默认严格 `p>0.5`，可在配置中明确改变 `edge_threshold∈(0,1)`。用 logit 比较避免 sigmoid 在阈值附近舍入；不补边或强制保留锚点边。概率、logit、所选物理边 ID 与 BFS 根均保存。

## 5. BP 公平比较与随机重复（D14）

每个独立样本、每个 BP 重复只抽取一套先验支持，Dense/BFS/GNN 共用。种子通过固定命名空间、评价种子、dataset_id、sample_id 和 BP 重复编号的 SHA-256 派生，不依赖样本遍历位置或拓扑名称。

默认 `bp_repeats=1`，可显式增加。每条记录保留种子、支持摘要、位置均值、协方差、ESS、消息残差、收敛轮数、锚点可达性、最小协方差特征值及 BP 配置。低 ESS、未收敛和未锚定节点均不静默删除。

本阶段主验证配置为 `configs/smoke-pipeline.json`：3/2/2 图、16 维隐层、2 层 GNN、2 轮训练，BP 为每节点 64 粒子/6 轮，测试每图 2 个 BP 随机重复。它只验证流程，不是正式模型结构、损失权重或 BP 参数推荐。

## 6. 指标、无效值与置信区间

### 6.1 图级与节点级分开

主表为每图指标先在同图的 BP 重复间平均，再在独立图之间计算均值与样本标准差。另保存 pooled agent-run RMSE、单节点误差分位数，以及独立的图 RMSE 分位数，不再混称“RMSE 分位数”。

中断定义为误差严格大于配置的阈值。通信代价来自初始坐标和数据中的 rho；保留边数、智能体三角形数、度数不足节点数均按唯一物理边计算。

### 6.2 NEES 与覆盖率分开

位置维度为 2，NEES 用 Cholesky 求解，不用伪逆或额外 jitter：

- 椭圆覆盖率：NEES ≤ chi-square_2(1−alpha)，默认约 5.9915。
- 双侧 NEES 区间通过率：NEES 落在 [chi-square_2(alpha/2), chi-square_2(1−alpha/2)]，默认约 [0.05064,7.37776]。
- 两者含义不同，例如误差为零时覆盖椭圆，但不通过双侧 NEES 区间。

协方差要求有限、数值对称、正定，并通过最小/最大特征值比例检查。默认显式 `covariance_rtol=1e-12`，比例不超过该值时记为数值不可用；设 0 可只要求严格正定。对称容差为输入 dtype 的 64 倍机器精度乘矩阵最大绝对值，没有引入固定绝对方差底限。

无效节点计入椭圆/区间未通过，同时保留无效数量；其 NEES 记为 JSON null，不混入有效 NEES 均值。某图或整个评价全部无效时，NEES 均值仍为 null，不写零。奇异协方差不会妨碍误差、边数等其余指标保存。

### 6.3 图为统计重复单位

置信区间采用图聚类 bootstrap：整图的所有节点和所有 BP 重复保持在同一簇，重抽独立 sample_id。默认 2000 次，置信水平 1−alpha，评价配置中显式记录。比较 GNN/BFS 与 Dense 时也按同一图的差值重抽，不把节点当独立 Bernoulli 样本计算 Wilson 区间。

一个独立图无法估计图间标准差和图间置信区间，均存 null；报告显示 n/a。零边图和全部无效 NEES 情形仍可保存/重读严格 JSON，禁止 NaN/Infinity。区间条件于本次指定的训练检查点，不包含不同训练种子的模型变化；正式多训练种子设计仍待后续完成。

### 6.4 FIM 辅助值（D13）

主流程移除旧 `compute_crlb` 的含混输出。可显式开启 `include_information_proxy`，计算“真值几何、已知固定方差、真实先验精度”组成的辅助信息矩阵尺度 `sqrt(trace(J_inverse)/A)`。

它不叫 BP RMSE 的普遍下界，也不当作真实后验协方差；不对已知方差求距离导数、不添加任意对角项、不用伪逆。真值几何只在这一明确标注的评价辅助值及误差统计中使用，不进入 GNN/BP 推断。遇到真值节点重合使距离梯度未定义时，辅助值为空并说明原因。

## 7. 命令与输出

以下在仓库根目录、已安装项目依赖的环境中运行。Windows 当前环境用 `.\.venv-data\Scripts\python.exe`；Linux 已配置的环境可用 `python` 代替解释器前缀。GPU 尚未在本机验证。

先用全新的小规模目录验收：

```powershell
.\.venv-data\Scripts\python.exe -B -m GNN_learning.experiment run --config configs/smoke-pipeline.json --data-root data/pipeline-demo --output-root outputs/pipeline-demo --device cpu --threads 1
```

已有数据复用必须显式 `--reuse-data`，并通过其身份校验；新训练应使用新的输出根或 `--run-name trial02`。生成数据只调用一次，训练/评价变化不改变数据指纹。

分别调用：

```powershell
.\.venv-data\Scripts\python.exe -B -m GNN_learning.experiment generate --config configs/smoke-pipeline.json --data-root data/pipeline-demo
.\.venv-data\Scripts\python.exe -B -m GNN_learning.experiment train --config configs/smoke-pipeline.json --data-root data/pipeline-demo --output-root outputs/train-demo --device cpu --threads 1
```

训练会打印实际 run 目录。把下列占位路径替换为该次真实文件：

```powershell
.\.venv-data\Scripts\python.exe -B -m GNN_learning.experiment train --resume "实际run目录/models/last_checkpoint.pt" --data-root data/pipeline-demo
.\.venv-data\Scripts\python.exe -B -m GNN_learning.experiment train --config configs/smoke-pipeline.json --init-from "实际run目录/models/best_checkpoint.pt" --data-root data/pipeline-demo --output-root outputs/finetune-demo --device cpu --threads 1
.\.venv-data\Scripts\python.exe -B -m GNN_learning.experiment evaluate --checkpoint "实际run目录/models/best_checkpoint.pt" --data-root data/pipeline-demo --output-root outputs/evaluation-demo --device cpu --threads 1
```

`--stop-after-epoch 1` 可在保留总训练计划的情况下，保存首轮后暂停本次命令，便于分段执行/测试恢复。要调整正式 BP 参数、阈值、重复数或统计参数，请写入 JSON，再通过 `--config` 传入评价；不使用隐藏的 quick 模式。

训练输出默认按 `outputs/experiment_id/config_hash[/run_name]/` 隔离；评价按检查点、数据、BP/评价配置及图数上限构成的指纹隔离，输出：

- `results.json`：配置、来源、模型、数据、摘要、逐图/逐重复指标、选择和诊断。
- `report.md`：可阅读的三拓扑比较、图级 RMSE 区间及异常计数。
- `manifest.json`：运行完成状态及报告/结果校验值。

## 8. 本次验收与边界

新增 34 项管线检查，联合既有数据、目标、GNN 和 BP 为 **204 项通过、1 项 CUDA 检查跳过**。证据包括 BFS 断连与多锚点、边序无关、完整恢复、微调来源/优化器重置、检查点缺失/损坏/误用拒绝、数据错配拒绝、共享支持、单图/零边/全无效报告及旧文件无导入副作用。

独立审计完成生成→两轮训练→加载实际最佳模型→Dense/BFS/GNN→报告，并额外验证恢复、微调、单图和全无效场景。两张测试图、每图两次 BP，每种拓扑共 4 次 BP；不是四个独立测试图。

这一验证模型仍在两张测试图上选空图。P=64、6 轮的稠密图均未达到消息容差，报告保留低 ESS 和未收敛状态。训练目标验证、管线可复现与正式性能是不同验收条件；本阶段没有据此宣布模型有效或正式参数已确定。

完整复验：

```powershell
.\.venv-data\Scripts\python.exe -B -m pytest tests/test_experiment_pipeline.py tests/test_particle_bp.py tests/test_model_sampling.py tests/test_objectives.py tests/test_data_contract.py -q -p no:cacheprovider --basetemp=data/stage6-retest
.\.venv-data\Scripts\python.exe -B -m tests.verify_stage6 --output-root outputs/stage6-rerun
```

后续仍需整体目录/依赖整理、GPU 一致性与资源验收，以及正式场景和多训练种子参数设计。
