# 第二步②：统一图数据格式和实验配置

本页记录第二步完成时的数据接口及范围。后续目标进展见 [training-objective.md](training-objective.md)，第六步已完成[统一实验入口](unified-experiment.md)；下文关于第二步当时尚未接入消费者的描述作为阶段记录保留。

状态：已实现数据生成、格式验证、版本化保存、PyG 加载与变规模批处理。验证结果见 [stage2-verification.json](stage2-verification.json)。论文源文件及第一步快照保持不变。

第二步验收范围仅为数据流程，不包含当时的旧 GNN、损失和 BP 修正。后续第三步已完成[新版训练目标](training-objective.md)，第四步已完成[GNN 和采样](gnn-sampling.md)，第五步已完成[粒子 BP 和独立参照](particle-bp.md)。

## 1. 文件入口与配置

| 文件 | 职责 |
| --- | --- |
| [experiment_config.py](../GNN_learning/experiment_config.py) | 配置数据类、参数检查、配置指纹、路径和独立样本种子 |
| [configs/](../configs) | 显式 normal / hard / challenge LOS 配置及小规模数据检查配置 |
| [data_schema.py](../GNN_learning/data_schema.py) | 原始图契约、输入派生量、验证、旧格式显式转换 |
| [generate_network.py](../GNN_learning/generate_network.py) | 一张静态二维定位图生成 |
| [generate_datasets.py](../GNN_learning/generate_datasets.py) | 独立样本生成、数据包与 JSON 清单保存 |
| [dataset.py](../GNN_learning/dataset.py) | 数据包加载、校验和 PyG 自定义批处理 |
| [tests/test_data_contract.py](../tests/test_data_contract.py) | 本阶段验证程序 |
| [requirements-data.txt](../requirements-data.txt) | 本阶段验证使用的直接依赖版本 |

配置文件通过 `load_config(path)` 读取，默认读取 `configs/challenge-los.json`。未知字段、非法数值、`auto` 分布、非法路径名等会报错，不会静默采用拼错参数。JSON 中未写出的字段使用数据类默认值；保存数据时会展开为完整配置。

| 预设 | 锚点布局/数量 | 智能体分布 | 智能体/锚点半径（m） | 用途 |
| --- | --- | --- | --- | --- |
| `normal-los.json` | 4×4 网格 / 16 | uniform | 30 / 30 | 延续旧 normal 设置 |
| `hard-los.json` | 四角加中心 / 5 | uniform | 30 / 30 | 延续旧 hard 设置 |
| `challenge-los.json` | 四角 / 4 | clustered | 30 / 54.044011451988085 | 默认；显式保留旧 challenge 的半径 |
| `smoke-data.json` | 四角 / 4 | uniform | 15 / 30 | 20 m 区域、2—5 智能体、共 7 张图的数据检查 |

场景名称只用于描述。布局、两类半径和分布分别配置，改变 `scenario` 不会暗中改变另外三者。默认参数不代表已经确定论文正式实验设置。

完整配置分为 `data/model/training/objective/bp/evaluation/split_sizes`。新版目标的 `beta/gamma/kappa/eta` 暂为 `null`，留待第三步按新目标确定；旧代码的 lambda、FIM 正则量仍单独标注为 legacy，不能当作新版系数已经确定。

## 2. 原始图记录

图版本：`gib4cl.graph.v1`。所有生成记录在 CPU 上保存。浮点输入为 float32，索引为 int64，标志为 bool。

```text
record
  schema_version
  sample_id
  inputs                      # 推理可用的 C、D
    initial_positions         # (N,2)，智能体初值 + 锚点精确坐标
    initial_covariances       # (A,2,2)，只含智能体，允许相关/异方差先验
    is_anchor                 # (N,)
    edge_index                # (2,E)，唯一物理测量边，全局节点索引
    measurements              # (E,)，z
    measurement_variances     # (E,)，v
    communication_cost_scale # rho，正标量
    measurement_encoding      # 明确的有限数值表示
  targets
    agent_positions           # (A,2)，仅训练标签/评估真值；推理可省略 targets
  metadata                    # 配置、样本种子、split、编号、单位、模型假设
  diagnostics                 # NLOS 标签、候选度数、锚点可达性；不进入默认模型输入
```

其中 `A` 为智能体数，`K` 为锚点数，`N=A+K`。智能体全局编号为 `0..A-1`，锚点编号为 `A..N-1`。每条边必须 `u<v`，且 `u<A`，因此不含自环、重复边或锚点—锚点边。锚点索引不再在不同模块中反复加减 `A`。

允许无锚点、无边、无智能体三角形、孤立节点和无锚点路径的分量。空边必须保持 `(2,0)`；这些图被保留并报告诊断信息，不补边、不丢弃。至少需要一个智能体。相连节点初始坐标重合会明确报错，因为论文在此处要求可定义线性化方向。

单图生成配置支持共享的标量协方差、两项对角方差或对称正定 2×2 矩阵。图记录及适配器支持每智能体独立的 `(A,2,2)` 矩阵。节点 5 维特征仅取对角方差，完整协方差另存，不能因特征只用对角值而丢弃相关项。

## 3. PyG 数据与批处理

`graph_to_data(record)` 输出 `LocalizationData`，原来的 `x/edge_index/edge_attr/y/num_agents/num_anchors` 接口继续存在。新增字段如下：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `(N,5)` | `[x,y,var_x,var_y,is_anchor]` |
| `edge_attr` | `(E,4)` | `[z,v,abs(r),is_anchor_edge]` |
| `agent_prior_cov` / `agent_prior_precision` | `(A,2,2)` | 完整节点先验及其逆；例如 `25I → 0.04I` |
| `agent_node_index` / `anchor_node_index` | `(A,)` / `(K,)` | 显式节点映射 |
| `initial_distances` / `signed_residual` | `(E,)` | 初值距离及 `z-dbar`；带符号残差用于均值更新 |
| `edge_directions` | `(E,2)` | `(rbar_u-rbar_v)/dbar`；仅为端点方向，不是尚未实现的全局 `g` 矩阵 |
| `edge_cost` | `(E,)` | `rho*dbar²`，从初值计算，锚点边也包括 |
| `edge_log_cardinality` | `(E,)` | `log M`，按物理边计算 |
| `message_edge_index` | `(2,2E)` | 前 E 条正向、后 E 条反向 |
| `message_edge_to_measurement` | `(2E,)` | 每条有向消息所共享的物理测量 ID |
| `message_reverse_index` | `(2E,)` | 反向消息索引，应用两次恢复原索引 |
| `triangle_edge_index` | `(3,T)` | 仅智能体三角形的三条物理边 ID，每个三角形一次 |

物理边用于评分、采样、损失和通信计数，只有一次；消息边用于双向传播。不能直接拿双向消息边数当通信边数，也不能在两个方向独立采样。

PyG 的默认规则会把含 `index` 的字段按节点数递增，因此 `LocalizationData.__inc__` 显式区分：节点索引加前图节点数，三角形/消息到测量映射加前图物理边数，反向消息索引加前图消息数。混合空图和不同规模图的 batch 及反拆均已验证。

`include_targets=False` 时不附加 `y`；`include_diagnostics=False` 是默认值，NLOS 诊断标签不附加到模型输入。特征构造函数只接受 `inputs`，无法读取 `targets`。

## 4. 测量模型、表示精度与压缩基数

LOS 按 `z=d+N(0,(sigma0+alpha*d)²)` 生成，不再执行 `clamp(z,min=0.1)`。负测距样本在高噪声下可以存在，符合论文的加性高斯模型。NLOS 可通过配置显式打开，仅给选定智能体边加正偏差，元数据标记为模型失配场景。

方差仍采用论文的理想化“已知、由真实距离生成”假设；当 `alpha>0` 时，这一已知量本身提供距离信息，需在论文实验设置中说明。它与把位置标签直接交给 GNN 是不同问题。此次没有擅自改成估计方差。

数值表示沿用 binary32，不新增人为测距范围或均匀量化器：

- `z` 支持全部有限 float32 数值，包括负值；`-0` 规范为 `+0`。
- `v` 支持全部严格正的有限 float32 数值。
- 拒绝 NaN、Inf、非正方差以及转换后下溢为零的方差。
- `z` 的支持数为 `2*(255*2^23-1)+1 = 4278190079`。
- `v` 的支持数为 `255*2^23-1 = 2139095039`。
- `M = 4278190079 * 2139095039`，`log M` 取自然对数，使用完整乘积支持作为参考编码集合。

这是参考分布的编码支持集，并不声称真实测量在这些数值上均匀分布。有限参考支持可以大于实际场景会产生的数值集合。特征中的残差不是新的保留测量编码，不再重复计入 `M`。

`log M` 为固定数不代表 `sum(p*log M)` 对网络参数是常数。第三步必须保留这一项，并按新尺度确定权重。测量表示、物理噪声方差、局部预测器方差下限及数值 jitter 是不同约定，本阶段没有把方差下限加进数据或用任意常数代替先验。

## 5. 数据身份、随机性与保存

数据包版本：`gib4cl.dataset.v1`，包含 `metadata` 和 `graphs`。加载使用 `weights_only=True`，不需要反序列化自定义 PyG 类。

默认输出：

```text
data/<experiment_id>/<data_fingerprint前16位>/<split>_dataset.pt
data/<experiment_id>/<data_fingerprint前16位>/<split>_dataset.json
```

`data_fingerprint` 使用完整展开的数据配置与主种子；训练学习率、轮数或评价阈值改变不改变同一份数据的身份。JSON 清单保存数据包 SHA-256，加载时若存在清单则核对文件校验值。

每个样本通过 `SHA-256(master_seed, split, seed_offset, sample_number)` 派生 63 位种子，并创建私有的 Python / PyTorch RNG。不同 split 有独立命名空间，不因生成顺序、另一个 split 的大小或前面生成的图改变当前样本。扩展同一数据集的图数保留已有样本前缀，不重置全局 RNG。

`series_id` 包含实验名、数据指纹、split、种子偏移及实际智能体数区间；`dataset_id` 另外包含图数；`sample_id=series_id:sample_number`。覆盖区间或图数会明确保存在 `resolved_split_config`，原始完整配置保存在 `config`，不能只看原配置中的默认区间来判断这次生成范围。

每个数据包保存关键生成/适配源码的 SHA-256、Git HEAD、运行版本、完整配置、数值表示、样本种子和诊断汇总。Git HEAD 不能单独代表有工作区修改的生成代码，因此同时保存源码校验值。

已有 `.pt` 或同名 `.json` 时，默认拒绝覆盖，并在生成之前报错；只有显式 `--overwrite` 才允许重建。参数扫描应使用独立实验名或不同配置指纹。同一 split 的不同临时数量/区间若仍写同一路径也会报错，需主动选择另一个输出根目录或实验名。

## 6. 旧代码与旧数据边界

- 原始列表 `.pt` 不再自动读取。只有 `LocalizationDataset(path,allow_legacy=True)` 才显式转换历史的“锚点局部索引”格式，必须有原始先验协方差。
- 转换会标注旧数据的截断和种子来源未知，不声称它们满足新版 LOS 生成协议。旧数据没有成本尺度元数据，转换中明确记录采用 `rho=1`。
- `generate_localization_network` 名称保留，但返回新版图记录。旧参数覆盖仍可用；`agent_distribution='auto'` 会报错，必须改为显式分布。
- 第二步当时可视化及 `NBP_ST/BFS_tree.py` 输入已改为全局索引；第六步已完成两套 BFS 的语义合并。
- 第二步当时旧训练/评估仅接入新数据路径；后续第三至六步已完成目标、GNN/BP 算法、完整配置 CLI 与模型/报告输出隔离。
- 旧 `config.py` 的数据、模型等常量从默认 JSON 派生。其标为 legacy 的训练目标/BP 标量设置不属于新版算法已经完成的实现。

## 7. 本地运行与验证

本机验证环境为项目内 `.venv-data`，复用已有 Conda 环境的 PyTorch 2.7.1 CPU，只在项目环境补充 PyG 2.6.1。没有修改已有 Conda 环境。该环境及生成数据已加入忽略规则，不能直接复制到服务器使用；服务器应按依赖与所需 CUDA 环境重新配置。

在仓库根目录使用该环境的 Python：

```powershell
.\.venv-data\Scripts\python.exe -B -m GNN_learning.generate_datasets --config configs/smoke-data.json
.\.venv-data\Scripts\python.exe -B -m GNN_learning.generate_network --config configs/challenge-los.json --num-agents 30
```

运行数据验证前先保证 `data/` 存在，并为本次测试选择一个新的临时子目录：

```powershell
New-Item -ItemType Directory -Path data -Force | Out-Null
.\.venv-data\Scripts\python.exe -B -m pytest tests/test_data_contract.py -q -p no:cacheprovider --basetemp=data/test-tmp-local
```

实测使用 `--output-root data/stage2-smoke` 生成 3/2/2 张图，再完成三个 split 的加载、混合规模批处理、反拆、样本身份检查和未训练旧 GNN 的输入形状试运行。输入形状试运行不代表评分方向、采样分布或算法效果正确。

检查覆盖：手算残差/成本/完整先验；标签与诊断隔离；空图、无锚点和断连；双向边及三角形；变规模批处理；配置检查；Gaussian 无截断；NLOS 失配标记；私有随机流；保存/加载及覆盖保护；数据身份不一致拒绝；显式旧格式转换；BFS 的全局索引接口；float32 不受全局默认 dtype 影响。
