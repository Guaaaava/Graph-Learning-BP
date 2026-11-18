# 定位指标、校准与统计口径

当前可交付实现只使用 `gib4cl/`。原有阶段记录与旧源码已集中归档至源仓库的 `legacy/`；发布包通过[固定来源数据](provenance.md)保留校验依据，运行不依赖历史目录。打包与完整验收步骤见[验收说明](acceptance.md)。

**当前路径与种子约定**：实现位于 `gib4cl/evaluation/`，使用 `seeds.bp` 和 `seeds.bootstrap` 分别控制 BP 支持和区间重抽。跨训练种子统计另见 [delivery-and-seeds.md](delivery-and-seeds.md)，不将其与本页的图级区间混为同一统计重复单位。

本阶段补齐评价层，不改第三至五步已经验证的训练目标、GNN/采样或粒子 BP。入口仍为 [统一实验入口](unified-experiment.md)；实现是 [metrics.py](../gib4cl/evaluation/metrics.py)、[topology.py](../gib4cl/evaluation/topology.py) 和 [evaluate.py](../gib4cl/evaluation/evaluate.py)。验收记录见 [来源与历史记录](provenance.md)。

指标协议为 `gib4cl.metrics.v2`，用于区分指标的统计定义。结果 JSON 不再附加独立的格式版本标签。旧结果、基线快照和论文原文未覆盖。结果目录身份保留指标协议、检查点、数据、配置及种子，去掉独立格式标签后目录摘要相应改变，数据指纹不变。

## 1. 位置误差与尾部风险

对图中的 A 个智能体，定义误差向量 e_i = estimate_i - truth_i、欧氏距离 d_i = ||e_i||。每次 BP 运行输出：

| 字段 | 定义与单位 |
| --- | --- |
| `rmse` | sqrt(sum d_i² / A)，m |
| `mae` | sum d_i / A，m |
| `error_p50/p90/p95`、`error_max` | 本图智能体距离误差的分位数/最大值，m；分位数采用 NumPy 默认线性插值 |
| `outage` | mean(d_i > outage_threshold)，严格大于；相等不算中断 |
| `outage_curve` | 预先配置的各阈值下的中断率，以及 error_cdf = mean(d_i ≤ threshold) |
| `agent_errors/agent_error_vectors` | 保留逐节点距离与二维有符号误差，便于复核 |

默认阈值网格为 0.5、1、2、3、5、10 m。主阈值若不在网格内，自动并入，并将实际网格写入结果。有限输入的误差计算如果溢出会明确报错，不静默丢掉节点。

最终摘要还单独输出所有 agent-run 的误差分位数和 pooled RMSE/MAE，以及独立的图 RMSE 分位数。前者按节点运行次数加权，后者按独立图加权；名称和用途不同。

## 2. NEES、覆盖率与高斯评分

设 BP 输出均值 mu_i、协方差 S_i。对有效协方差，NEES q_i = e_i^T S_i^(-1) e_i，以 Cholesky 求解，不用伪逆或 jitter。二维 chi-square 分布的精确分位数为 Q(p) = -2 log(1-p)。

| 字段 | 定义 |
| --- | --- |
| `ellipse_coverage` | q_i ≤ Q(1-alpha) 的节点比例，分母为全部 A 个智能体 |
| `ellipse_coverage_valid` | 同一事件，仅以本次有效协方差节点为分母；全无效时 null |
| `nees_interval_pass` | Q(alpha/2) ≤ q_i ≤ Q(1-alpha/2) 的全节点比例 |
| `nees_below_interval/above_interval` | q_i 严格小于下界/严格大于上界的全节点比例 |
| `invalid_covariance_rate` | 无效节点数 / A |
| `mean_nees_valid` | 有效节点的平均 NEES，全无效时 null |
| `coverage_curve` | 各名义水平 c 下 q_i ≤ Q(c) 的全节点覆盖率、有效节点覆盖率 |
| `gaussian_nll_valid` | 有效节点平均 log(2*pi) + 0.5 log det(S_i) + 0.5 q_i |
| `ellipse_area_valid` | 有效节点平均 pi * Q(1-alpha) * sqrt(det(S_i))，m² |
| `predicted_rmse_valid` | sqrt(有效节点的 mean trace(S_i))，m；为报告的协方差尺度 |

默认 alpha=0.05。默认覆盖曲线水平为 0.5、0.8、0.9、0.95、0.99，并自动并入 1-alpha。主水平的曲线与主覆盖率共用同一个阈值，避免边界上的浮点差异。

区间通过、低于区间、高于区间、无效四项比例之和为 1。椭圆覆盖率与双侧 NEES 区间通过率不同；例如零误差通过椭圆覆盖，却可能低于双侧区间。

NLL 保留二维归一化常数，以米为坐标单位，单位为 nats。NLL 允许为负，不能因此判错。面积和协方差尺度帮助识别“通过扩大协方差提高覆盖率”的情况。面积/评分均保留实际可计算的节点数；极端数值导致某个评分不可表示时该值为 null。

**这些椭圆和 NLL 使用粒子 BP 边缘的一、二阶矩所匹配的高斯分布。** 它们不是粒子分布的最高密度区域，也不是非高斯后验/粒子分布本身的似然。真值只在评价误差和评分中使用，不进入选边/BP。当前不提供额外的信息矩阵代理选项或指标。

NEES 尾部和经验覆盖率提供误差与协方差的一致性证据，不能直接证明论文 eq:directional_overconfidence 中“BP 协方差与真实后验协方差的逐方向大小关系”。第五步小图精确/积分参照仍用于后者的独立核验。三角形更少也不保证定位或校准更好。

### 无效协方差

先保留输入 dtype 的机器精度，转换到 CPU float64 统计。协方差须有限、在 64 倍输入 dtype 机器精度的相对容差内对称、严格正定、最小特征值大于 covariance_rtol * 最大特征值，并能完成 Cholesky/有限 NEES 计算。只有通过对称检查后才平均 S 与 S^T 消除舍入差异。默认 covariance_rtol=1e-12；设为 0 时仍要求严格正定。

逐节点保存 `agent_covariance_status`，区分非有限、不对称、非正定、病态、Cholesky 失败和 NEES 非有限；同时保存分类计数。无效节点在所有全节点覆盖率和区间检验中记为未通过，NEES/NLL/面积等条件评分记为 null。均值不以零替代，不以额外方差下限掩盖退化。

## 3. 通信代价、稀疏度与结构诊断

所有边以数据契约中的唯一物理测量边计数，使用实际二值掩码。拒绝把软概率当作实际选边指标；软概率的期望能量/三角形属于训练日志。

| 字段 | 定义 |
| --- | --- |
| `edges/candidate_edges` | 选中/候选物理边数 |
| `edge_retention/edge_saving` | selected_edges / candidate_edges；saving = 1 - retention |
| `communication_cost` | sum s_ij * edge_cost_ij，edge_cost = rho * 初值距离²，包含锚点边 |
| `candidate_communication_cost` | 候选图全部物理边的成本和 |
| `communication_retention/saving` | selected_cost / candidate_cost；saving = 1 - retention |
| `agent_edges/anchor_edges` | 智能体—智能体边 / 智能体—已知锚点边数 |
| `agent_triangles/candidate_agent_triangles` | 选中/候选的三个顶点均为智能体的三角形数，每个一次 |
| `agent_triangle_retention` | selected_agent_triangles / candidate_agent_triangles |
| `agent_cycle_rank` | E_agent - A + C_agent；C_agent 是智能体诱导图分量数，含孤立智能体 |
| `connected_components/agent_components` | 全部节点图 / 智能体诱导图的连通分量数，均含各自孤立节点 |
| `agent_degree_min/mean/max` | 智能体实际度数，锚点边参与度数 |
| `isolated_agents` | 实际度数为 0 的智能体数 |
| `agents_below_degree_three(_fraction)` | 实际度数小于 3 的智能体数量/比例 |
| `unanchored_agents/anchor_reachable_fraction` | 所选图中无法通过路径到达任何锚点的智能体数量 / 可以到达的比例 |

保留率和节省率先在每张图内计算；候选边数、候选总代价或候选三角形数为零时，相应比率为 null。选空图但候选图非空且代价为正时节省率为 1；候选图本来就空时不能声称“节省 100%”。

rho 的单位由建模选择决定。这里记录的是论文的通信成本代理，不把数值改称实测焦耳、字节数或无线协议通信次数。统计用 float64 对已存 edge_cost 求和，避免 float32 求和造成额外误差。

环秩是独立环空间的维数，不是所有简单环的数量。四边形可以三角形数为 0 而环秩为 1。已知锚点不构成未知变量反馈环，因此不纳入 agent 三角形/环秩。分量、锚点可达性和度数也不是二维刚性或可定位性的充分证明。

## 4. BP 数值诊断

保留原始每节点 ESS、最小协方差特征值、消息残差和收敛状态，并输出：

- `ess_min/mean`、`ess_fraction_min/mean`：ESS 及 ESS/P，P 是本次配置的粒子数。
- `low_ess_agents/fraction`：ESS/P 严格小于 ess_fraction_threshold 的节点数/比例，默认阈值 0.1。恰等于阈值不计低 ESS。
- `bp_converged/iterations/message_residual`：收敛标志、实际迭代数、最后消息残差。没有可用残差时保留 null。
- 摘要保留未收敛 BP 运行次数和低 ESS agent-run 数量，不剔除其定位误差。

这里的 0.1 是明确可改的报告阈值，不宣称是充分粒子数或正确后验的判据。统一图连通性计算还会与 BP 返回的锚点可达性核对，不一致时评价报错。

## 5. 图级聚合、置信区间和配对比较

对于每个标量和每个曲线点：先对同一 sample_id 的 BP 重复取平均，再对独立图取均值、样本标准差（ddof=1），以整图为簇做 percentile bootstrap。默认重抽 2000 次、置信水平 1-alpha。所有节点和 BP 重复留在同一簇，不视为独立观测。

条件评分先对本次有效节点平均，再对有值的 BP 重复平均，最后对有值的独立图平均；没有可用值时为空。它不同于直接汇总所有有效节点。逐节点有效标志、每次评分有效节点数、每图重复数、摘要有效 agent-run 数和每个统计量的 `graphs` 均保留，便于了解实际分母。某指标只有一张有效图时 std/CI 均为 null；bootstrap_samples=0 时明确关闭 CI。

每个拓扑/指标有独立的派生 bootstrap 随机流，重抽顺序按图 ID 排序。改变记录/拓扑遍历顺序或增加其他指标不会改变某个已有统计量的重抽区间。bootstrap 的工作数组分块生成，不一次分配全部“重复数 × 图数”。

配对差值为 BFS-minus-Dense、GNN-minus-Dense；必须具有相同 sample_id 和 BP 重复编号。对每图的重复平均指标先相减，再按独立图重抽。条件评分只使用双方该指标均有值的图，报告其数量。拒绝重复身份、不同阈值/覆盖网格/协方差策略/ESS 阈值混合，以及缺失配对图或重复的记录。

每个覆盖曲线点还保存 coverage_gap = 覆盖率 - 名义水平，CI 同步平移。它是这一水平下的经验差异，不称为逐图后验正确性证明。

区间仅条件于本次训练检查点。BP 随机重复用于表征有限粒子积分的随机性；多训练种子的模型变化仍需在正式实验设计中单独安排。

## 6. 配置与输出

所有现有配置已经显式列出新增字段，数据指纹未改变：

```json
{
  "evaluation": {
    "outage_thresholds": [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
    "coverage_levels": [0.5, 0.8, 0.9, 0.95, 0.99],
    "ess_fraction_threshold": 0.1
  }
}
```

这只是新增字段示意；实际文件仍需保留其他配置。网格必须非空、有限、严格升序且无重复；覆盖水平在 (0,1)，中断阈值 >0，ESS 比例阈值在 (0,1]。JSON 列表读入后规范为 tuple，配置往返一致。

每次评价保存：

| 文件 | 用途 |
| --- | --- |
| `results.json` | 完整配置/协议/数据/检查点/源码身份、逐节点/逐图/逐 BP 重复结果、各指标和曲线 CI、配对差值 |
| `report.md` | 精度、校准/面积/NLL、通信/图结构、BP 诊断、CDF/中断曲线、覆盖曲线、配对比较 |
| `trial_metrics.csv` | 每图/每 BP 重复/每拓扑的标量指标，并带 sample_id、种子、支持摘要和节点数 |
| `summary_metrics.csv` | 全部标量、曲线点和配对差值的 mean/std/graphs/CI；轴值单列 |
| `manifest.json` | 完成状态及四个结果文件的 SHA-256 |

JSON 禁止 NaN/Infinity，null 在 CSV 中为空单元格，Markdown 中显示 n/a。重复调用相同模型/数据/配置可在新输出根或新 evaluation_name 下精确复现数值，不覆盖已有目录。旧版本完整检查点可以继续显式用于新评价；恢复训练仍遵循第六步的原配置/源码/环境身份要求，不能把改变后的运行假称为原样恢复。

统一调用方式未增加隐藏模式，仍可使用：

```text
python -B -m gib4cl.experiments.cli evaluate --checkpoint 实际的best_checkpoint.pt --config configs/smoke-pipeline.json --data-root 实际数据根目录 --output-root outputs/metrics-demo --device cpu --threads 1
```

## 当前验收入口

在项目根目录执行 `python -B -m tools.accept_cpu --output-root outputs/new-cpu-acceptance`，使用一个新目录。该命令重跑当前数学、数据、训练/恢复、指标、编排和打包检查；具体通过条件见[完整验收说明](acceptance.md)。历史验证数字不代表本候选版本测试数量或正式性能，当前数量以 JUnit 报告为准。
