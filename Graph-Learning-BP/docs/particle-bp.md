# 粒子 BP 与独立数学参照

当前可交付实现只使用 `gib4cl/`。原有阶段记录与旧源码已集中归档至源仓库的 `legacy/`；发布包通过[固定来源数据](provenance.md)保留校验依据，运行不依赖历史目录。打包与完整验收步骤见[验收说明](acceptance.md)。

实现见 [particle_bp.py](../gib4cl/inference/particle_bp.py)，独立参照见 [reference.py](../tests/references/particle_posterior.py)，测试见 [test_particle_bp.py](../tests/test_particle_bp.py)。数值参照结果保存在 [来源与历史记录](provenance.md)，完整验收摘要见 [来源与历史记录](provenance.md)。

本阶段修正论文第 2 章的 BP 消息和矩估计，并将“消息算法是否正确”与“有限粒子是否足够准确”分开验证。没有把环图 BP 收敛当成精确后验，也没有给有限粒子树图宣称连续后验精确性。

## 1. 固定支持上的标准 BP

每个智能体维护固定支持 `points[A,P,2]`，锚点直接使用已知坐标。每个节点读取完整的 `agent_prior_cov`，支持相关、异方差先验；不再用全局 `25I` 覆盖数据中的先验。

对于第 i 个节点的支持点，设 `w0_i[p]` 为先验积分质量。定义固定局部项：

```text
local_i[p] = log(w0_i[p])
             + sum_selected_anchor_edges log psi_ia(points_i[p], anchor_a)
```

对每条保留的智能体物理边维护两个有向消息。槽 `2k` 表示 u→v、在 v 的支持上取值；槽 `2k+1` 表示 v→u、在 u 的支持上取值。第 t 轮仅使用第 t−1 轮消息：

```text
cavity_u_to_v[p] ∝ w0_u[p] * anchor_likelihood_u[p]
                         * product_{k in N(u), k != v} message_k_to_u_prev[p]
message_u_to_v[q] ∝ sum_p psi_uv(points_u[p], points_v[q]) * cavity_u_to_v[p]
belief_i[p] ∝ w0_i[p] * anchor_likelihood_i[p] * product_k message_k_to_i[p]
```

这对应 `eq:bp_variable_to_factor`、`eq:bp_factor_to_variable`、`eq:bp_anchor_message` 和 `eq:bp_belief`。直接求和其他入消息，不再将完整 belief 作为发送分布，也不通过相减两个很大的 log 值来构造 cavity。

锚点因子与迭代无关，预先计入局部项；轮次 0 已含锚点测量，智能体消息初始化为常数。每轮同步推进一条智能体边，不在同一轮使用刚更新的消息。

## 2. 先验、提议分布与积分权重（D11）

默认通过 `draw_support` 从各节点的完整高斯先验独立采样，支持在所有迭代中固定：

```text
points_i = initial_mean_i + standard_normal @ chol(Sigma0_i).T
w0_i[p] = 1/P
```

先验提议 q 等于先验，因此重要性积分中的 prior/q 抵消，不能再乘一次先验密度。先验仍通过粒子分布发挥作用，并未从后验模型中移除。

`ParticleSupport` 也接受显式支持和先验 log 积分质量，供网格/求积验证使用。调用方须给出正确的“先验密度 × 积分权重”，支持行顺序须与 `agent_node_index` 一致。显式支持允许零质量（log 质量为负无穷），但每个节点总质量必须为正。

不在迭代中重采样，不添加 roughening，不改变支持而沿用旧消息。这个选择使有限支持上的模型和消息语义清楚，也便于不同拓扑共享同一套积分点。固定先验支持可能漏掉后验模式；增加粒子数和跨种子检查仍然必要。

## 3. 数值与资源

- 默认 float64、全对数域，采用 logsumexp 积分和消息归一化；不再给似然添加任意 eps 底噪。
- 默认直接使用已知正测量方差。BP 的 `measurement_variance_floor=0` 与训练局部预测器的 `prediction_variance_floor=1e-6` 分开。若显式设 BP 下限，则改变了推理模型，结果记录实际受影响边数。
- 每条无向智能体边的一个核块同时用于两个方向；默认块宽 256，单个 float64 标量核块最多约 512 KiB。这不是整个进程的实测峰值内存。
- 成对核的内存从 P×P 改为至多 B×B。默认不保存历史时，总存储量为 O(A·P+E_agent·P+B²)；`return_history=True` 会额外存储各轮快照。
- 时间复杂度仍为 O(T·E_agent·P²)，并未变成线性。正式大粒子实验仍需在服务器上测时间和显存。
- 保留边必须为严格 0/1；不把连续权重静默阈值化。不强制保留锚点边，也不补边连接不连通分量。
- 非有限数值、非法方差/协方差、重复边、缺完整先验或多图误作单图等情形明确报错。

高斯似然中仅依赖已知方差的归一化常数可在每条边上省略，因为它不改变归一化消息和边缘；独立参照采用同一约定，其 `log_partition_relative` 不命名为完整模型证据。

## 4. 矩、停止状态与诊断

最终估计按 `eq:bp_mean`、`eq:bp_covariance` 计算：

```text
mean_i = sum_p w_i[p] * points_i[p]
cov_i = sum_p w_i[p] * (points_i[p]-mean_i)(points_i[p]-mean_i).T
```

这里是加权分布的中心二阶矩，没有 Bessel 修正。结果不添加协方差 jitter；粒子塌缩时可能得到奇异协方差，应在后续统计中显式处理。

`BPResult` 包括最终权重、固定支持、有向消息及其物理边 ID、迭代数、消息残差、收敛标志、各节点 ESS、协方差最小特征值和所选图的锚点可达性。`ESS=1/sum(w²)` 用于检查支持权重退化；高 ESS 本身不证明连续后验准确。

默认固定轮数、不阻尼。可显式设置概率域阻尼 `damping∈[0,1)` 和提前停止。消息先在接收支持上归一化，再对新旧消息作凸组合，使用 logaddexp 计算。收敛判据为归一化 log 消息的最大绝对变化不超过 `message_tolerance`（默认 1e-8）。固定轮数模式仍报告该状态；存在智能体边且运行 0 轮时，残差为 `None`、收敛标志为假。没有智能体边时，局部项即可得到结果，迭代数为 0。

消息收敛、ESS、锚点可达性描述不同性质，不能相互替代。

## 5. 配置及调用

BP 工厂直接读取数值参数，不再设置或检查独立的算法版本标签。预设的 BP 配置为：

```json
{
  "num_particles": 10000,
  "num_iterations": 3,
  "chunk_size": 256,
  "compute_dtype": "float64",
  "measurement_variance_floor": 0.0,
  "damping": 0.0,
  "message_tolerance": 1e-8,
  "early_stopping": false
}
```

粒子数 10000 和轮数 3 是已有配置值的保留，不代表已验证适合正式实验，也不保证消息收敛。具体粒子数、轮数和阻尼须通过后续正式场景检查确定。数据配置及指纹保持不变，原数据可以复用。

```python
from gib4cl.inference import ParticleBP
from gib4cl.config import load_config
import torch

cfg = load_config("configs/experiment.json")
bp = ParticleBP.from_config(cfg.bp)
rng = torch.Generator(device=data.x.device).manual_seed(123)
support = bp.draw_support(data, generator=rng)

dense_result = bp.infer(data, support=support)
selected_result = bp.infer(data, edge_weights=mask, support=support)
# 上面两种拓扑共享同一组支持点；mean/covariance 的行按 agent_node_index。
```

旧 `run` 保留三元组接口：`(mean, covariance, support_points[A,2,P])`。第三项现为固定的带权支持点，**不是旧版重采样粒子**；如需使用它的后验分布，应同时读取 `infer().weights`。不再接受旧 `sigma_meas/init_cov/reg_scale` 参数，避免隐藏替换测量方差或先验。批处理入口 `particle_bp_batch` 可逐图指定共享支持或独立 generator。

第五步当时仅对旧评估脚本做 BP 配置和共享支持的必要衔接；本页参照验收使用独立审计脚本。后续第六步已完成模型路径、BFS、统计与报告的[统一实现](unified-experiment.md)。

## 6. 可信参照的层次

### 6.1 有限支持：穷举完整联合后验

独立 NumPy 程序枚举 P^A 个状态，逐一乘节点先验积分质量及所有所选测量因子，再直接求边缘和矩；没有调用生产 BP 的 cavity、消息、核分块或矩计算函数。枚举分块执行，并有最大状态数保护，避免误将大图当作可穷举问题。

测试覆盖单边、链、树、森林、锚点局部项、全选/全剪/部分掩码及非均匀先验质量。充分同步传播后，树/森林上的 BP 与**相同有限模型**的精确边缘一致。另有独立概率域字典 BP，核对环图的第 0/1/2/8 轮更新。

审计中单边、链、树及掩码案例的最大边缘/均值/协方差分量误差约 8.9e-16。一个三角环案例已收敛，但仍与穷举后验不同：边缘最大分量差约 0.0289、均值约 0.0604、协方差约 0.0582。这展示了“消息达到固定点”与“等于真实后验”的区别，不据此声称所有环图协方差具有同一偏差方向。

### 6.2 连续后验：两种小型定位问题

参照直接计算论文的非线性测距后验。

1. 单智能体、先验与锚点同心的圆对称问题：先验协方差 4I，测量 z=3.2、方差约 0.7（实际读取已存 float32 数值）。通过一维径向积分，包含面积 Jacobian r，得到零均值和约 4.602015841I 的协方差。128/256/512 阶求积并扩大积分半径相互核对；201/401 点每轴的二维网格交叉验证，最后协方差最大分量差约 6.2e-8。该特殊参照只测试非线性 BP，允许先验均值与锚点重合，不用于需要非零线性化距离的 GNN 训练数据。
2. 双智能体、有各自锚点测量的协作链：使用相关先验。每轴 13/21/29/37 阶 Gauss–Hermite 求积对原始后验积分，最后两级的均值和协方差变化小于 3e-13；独立的二维梯形网格按节点组成四维联合积分，协方差最大分量差约 5.2e-9。

这类加密与交叉核对是数值精度证据，不是严格积分误差上界。案例输入、参照均值/协方差、各级差异和种子均保存在结果文件。

### 6.3 有限粒子：多种子与粒子数

同一种子在不同粒子数间复用先验支持的前缀。下表的“均值误差 RMS”是相对于积分参照的整个坐标均值向量误差范数，跨种子取 RMS；“协方差误差 RMS”使用各节点协方差集合的 Frobenius 范数。它们不是对位置真值计算的最终定位 RMSE。

| 案例 | 每节点粒子数 | 种子数 | 均值误差 RMS（m） | 协方差误差 RMS（m²） |
| --- | ---: | ---: | ---: | ---: |
| 单智能体径向 | 64 | 32 | 0.51164 | 1.22813 |
| 单智能体径向 | 256 | 32 | 0.26653 | 0.59610 |
| 单智能体径向 | 1024 | 32 | 0.13408 | 0.27805 |
| 单智能体径向 | 4096 | 32 | 0.06311 | 0.14962 |
| 双智能体协作 | 64 | 16 | 0.15709 | 0.19348 |
| 双智能体协作 | 256 | 16 | 0.07989 | 0.08935 |
| 双智能体协作 | 1024 | 16 | 0.04528 | 0.05603 |

这些结果检查粒子近似随样本数的改善，不要求每个种子的误差单调下降，也不将它推广为任意场景的误差保证。

## 当前验收入口

在项目根目录执行 `python -B -m tools.accept_cpu --output-root outputs/new-cpu-acceptance`，使用一个新目录。该命令重跑当前数学、数据、训练/恢复、指标、编排和打包检查；具体通过条件见[完整验收说明](acceptance.md)。历史验证数字不代表本候选版本测试数量或正式性能，当前数量以 JUnit 报告为准。
