# 参数依据登记表

更新：2026-11-04。配套 [选参协议](selection-protocol.md)。覆盖 configs/experiment.json 的全部字段，并补充关键实现约定。“当前值”是实际文件值，不表示本次已选择该值。

依据分为：论文/数学约束、场景假设、实现/报告约定、已有实验、待训练验证。没有最优性证据的数值明确列为候选或约定，不补造理论理由。

## 场景与输入

| 配置字段 | 当前值 | 依据与边界 | 当前处理 |
| --- | --- | --- | --- |
| experiment_id | five-anchors-los | 实现约定：标识当前场景，无物理最优性 | 保留 |
| data.num_agents_min / num_agents_max | 25 / 35 | 场景假设，控制规模与密度，不是理论最佳节点数 | 正式场景固定 |
| data.area_size | 100 m | 场景假设：二维正方形，和密度/半径共同影响连接 | 固定 |
| data.comm_radius | 30 m | 仿真候选测量边规则，不是设备覆盖的实测值；阶段 13 有单因素诊断 | 固定，不为改善结果改连接条件 |
| 锚点数量与位置（派生） | 5，四角及中心 | 场景假设，提供绝对信息；不是最优布局证明 | 固定，不保证每图可定位 |
| data.initial_covariance | 25I m² | 每轴初始标准差 5 m 的场景假设；协方差正定是数学要求 | 固定，阶段 13 变体仅诊断 |
| data.base_noise_std / distance_noise_scale | 0.5 m / 0.05 | std(d)=0.5+0.05d 的 LOS 仿真假设，不是硬件标定结论 | 固定 |
| data.communication_cost_scale | rho=1 | 相对成本单位，边成本 rho*dbar² 使用初始位置距离，与 gamma 耦合 | 固定，不称为焦耳 |
| data.measurement_encoding | ieee754-binary32-canonical-finite | 有限编码/输入约定，决定 M 及 log(M)；不代表最优传输协议 | 固定，由实现计算保留数据成本 |

保留阶段 10 的建模解释：当前 BP 将已知输入方差当作固定因子参数；真实距离生成的方差、区域与连接条件可能另含几何信息。不能把工作因子后验称为包含全部生成/选择信息的精确条件后验。本次披露这一约定，不修改论文模型。

2026-11-05 [稿件核对](paper-version-review-20261105.md)确认当前第二章已删除距离噪声公式。因此 std=0.5+0.05d 是本实验设定，而不是当前论文的指定公式；当前稿的一般高斯测距模型仍允许该设置，正式场景保持不变。

## 网络与优化

| 配置字段 | 当前值 | 依据与边界 | 当前处理 |
| --- | --- | --- | --- |
| model.node_in_dim / edge_in_dim | 5 / 4 | 论文第 4 章特征结构及接口约束，不是自由宽度 | 固定 |
| model.hidden_dim / num_layers | 64 / 3 | 架构起点，无当前任务最优性证据；层数影响聚合范围 | 首轮控制变量固定，后续按需单因素消融 |
| LayerNorm（实现约定） | 开启 | 现有 MLP 设置，非目标推导所必需 | 披露，收益需消融 |
| training.epochs | 200 | 训练上限候选，不保证收敛 | 保留完整日程；20/60 是暂停点，结合曲线评估 |
| training.batch_size | 32 | 显存与梯度估计的工程起点 | 首轮固定，短试测资源与波动 |
| training.learning_rate / lr_min | 0.001 / 0.00001 | AdamW 与余弦日程的候选，非理论最优 | 首轮固定，后续看轨迹 |
| training.tau_init / tau_min / tau_decay | 5 / 0.1 / 0.98 | 论文支持软梯度温度降低，未确定这些数值；实际 tau=max(0.1,5*0.98^epoch)，约第 194 轮到下限 | 日程候选，20 轮不等于完整低温训练 |
| training.weight_decay | 0.0001 | 工程候选，必要性未判断；不等于论文目标的等价改写 | 暂保留；已接入实际更新和名义衰减日志，等正式短训练/受控对照 |
| training.max_grad_norm | 5 | L2 梯度裁剪候选，不是 AdamW 更新步长上限 | 暂保留；已接入裁剪前后范数、缩放及触发比例 |
| training.deterministic | true | 可复现计算约定，不保证跨硬件/版本逐位相同 | 固定并记录环境 |
| training.validation_repeats | 16 | 2026-11-04 固定的随机选边验证预算，近似论文期望目标；不是充分性的理论保证 | 每轮固定，另记前 8 次均值，后续测成本和敏感性 |
| training.validation_seed | 1700 | 验证专用随机流编号，由 sample_id/重复号派生，与训练/BP 分开 | 固定，改变需新运行；见 [验证选模规则](validation-selection.md) |

硬前向的选边保留概率不随温度变化，温度影响软近似的反向梯度；与修改部署门槛是不同操作。依据为 learning/model.py 及论文第 4.3 节。

## 目标与待选系数

| 配置字段 | 当前值 | 依据与边界 | 当前处理 |
| --- | --- | --- | --- |
| objective.beta | null | 论文 beta>0；阶段 07 梯度比起点约 0.01，无性能选择 | 首轮 0.003/0.01/0.03 |
| objective.gamma | null | 论文通信乘子 gamma>=0；阶段 07 单边成本比起点约 0.1 | 首轮 0/0.03/0.1/0.3 |
| objective.kappa | null | 论文三角形乘子 kappa>=0；阶段 07 p=0.5 参考成本比起点约 150 | 首轮 0/45/150/450 |
| objective.eta | null | 论文 eta>=0；阶段 07 梯度比起点约 6，极小分母不支持极大值推荐 | 首轮 0/1.8/6/18 |
| objective.prediction_variance_floor | 1e-6 | 数值保护；场景测量方差 >=0.25，阶段 07 触发数为 0 | 保留计数，不挪用到 BP |
| objective.compute_dtype | float64 | 已有 CPU/GPU 数值与反向传播验收 | 保留，性能仍需实测 |
| 度数下限（论文/实现约定） | 3 | 包括锚点边的智能体测量度数；不是刚性或精度保证 | 固定，不靠放大 eta 掩盖先天缺口 |
| 三角形范围（论文/实现约定） | 仅三个智能体 | 与论文压缩项对应，不扩为含锚点三角形 | 固定 |

阶段 07 的 32 图有 23 图在候选图本身就存在三度缺口，不应称为训练失败。有效通信/三角形系数为 beta*gamma、beta*kappa。正式 null 仅在验证选择后填写，不用 smoke 数值代替。

## 粒子 BP

| 配置字段 | 当前值 | 依据与边界 | 当前处理 |
| --- | --- | --- | --- |
| bp.num_particles | 10000 | 当前比较预算；阶段 13 实际检查后仍有支持误差与少数严重退化 | 暂留，不称充分/精确，不要求完全修复 |
| bp.num_iterations | 3 | 原始默认，正式场景充分性尚未确认；小图已证实传播轮数可能不足 | 待 3/5/10/20 配对检查 |
| bp.chunk_size | 256 | 资源参数，不改变有限求和目标；阶段 07、13 核对了分块数值一致性 | 生产值未改，下次 GPU 候选 2048 并检查显存 |
| bp.compute_dtype | float64 | 数值验收支持；log 消息降低下溢风险，不消除统计退化 | 固定 |
| bp.measurement_variance_floor | 0 | 保留论文测量因子，不加额外底噪 | 固定 |
| bp.damping | 0 | 普通同步 BP；非零值改变有限轮数算法 | 固定，不新增方案 |
| bp.message_tolerance | 1e-8 | 归一化 log 消息变化诊断门槛，不是后验误差界 | 只作标记，结合矩变化解释 |
| bp.early_stopping | false | 固定轮数比较，容差不控制实际停止 | 固定 |
| 固定先验支持（实现约定） | 独立高斯先验抽样，质量 1/P | prior/q 抵消后的积分权重；阶段 10–11 已核对有限支持计算 | 保留，不加提议或重采样 |

## 评价、种子和样本量

| 配置字段 | 当前值 | 依据与边界 | 当前处理 |
| --- | --- | --- | --- |
| evaluation.edge_threshold | 0.5 | p>0.5 保留边，二元决策的自然默认，不是任务最优保证 | 首轮固定，有模型后在 val 查 0.3–0.7 |
| evaluation.bp_repeats | 1 | 当前默认不足以表征所有粒子随机性；诊断已用 3 种子 | 计划粗筛 1、最终候选 3，正式次数待预算报告 |
| evaluation.outage_threshold | 3 m | 误差超限的仿真报告约定，尚无外部应用要求 | 保留并报告多阈值曲线 |
| evaluation.outage_thresholds | 0.5/1/2/3/5/10 m | 报告约定，避免单点判断 | 固定，test 后不改 |
| evaluation.alpha | 0.05 | 统计报告约定，对应 95% 标称区间 | 保留，不当作正确性保证 |
| evaluation.coverage_levels | 0.5/0.8/0.9/0.95/0.99 | 均值/协方差匹配高斯椭圆的覆盖水平 | 保留，明确不等于粒子精确可信区域 |
| evaluation.bootstrap_samples | 2000 | 图级统计重抽样次数约定，不增加独立图数 | 保留，单图不报图间区间 |
| evaluation.covariance_rtol | 1e-12 | 相对特征值数值有效性门槛，不是物理方差下限 | 保留，报告无效率，不加 jitter 隐藏问题 |
| evaluation.ess_fraction_threshold | 0.1 | 相对 ESS 诊断约定，不是精度合格线 | 保留，另报绝对 ESS/最大权重/ESS<2 计数 |
| seeds.data | 100 | 实验身份约定，按 split/样本派生，无物理最优性 | 固定 |
| seeds.training | 100 | 文件原始默认；计划粗筛覆盖为 101，最终 101/102/103 | 各运行显式记录，不改默认冒充已选 |
| seeds.bp | 700 | 独立粒子流编号，复核计划 700/701/702 | 拓扑共享支持 |
| seeds.bootstrap | 700 | 独立统计重抽样流；通过不同用途/命名与 BP 分开 | 固定，不因编号相同共用状态 |
| split_sizes.train / val / test | 2000 / 500 / 500 | 实验设计起点，不是样本量充分性证明 | 固定，报告图数与区间，封存 test |

新增的 20/60 轮暂停点、64 图轻量验证、16/32 图 BP 复核、36 个系数组合上限和单批 2 小时规划上限，均为资源约定，尚无“必然充分”的依据。执行前按实测估时分批确认；不在本阶段运行。

[训练日志](training-diagnostics.md)另声明概率<0.01/>0.99 的诊断区间和 L2/百分位汇总口径；这些只是观察定义，不是新增模型参数，不参与选边或优化。裁剪的 1e-6 来自现有 PyTorch 裁剪函数，日志沿用其实际缩放，不新增裁剪操作。

## 依据索引

- 论文：工作区 GIB-Chap234-1016.tex 第 2–4 章，重点为 eq:measurement_factor、eq:linear_task_loss、eq:compression_loss、eq:degree_loss、eq:total_loss。
- 实现：gib4cl/config.py、configs/experiment.json、gib4cl/learning/{model,objectives,train}.py、gib4cl/inference/particle_bp.py、gib4cl/evaluation/{topology,metrics}.py。
- 阶段报告位于远程工作目录 /root/autodl-tmp/gib4cl-work/：stage07-resource-and-scale-20261101-155011、stage10-paper-code-audit-20261101-173015、stage11-small-graphs-20261101-174730、stage12-nebp-20261101-180011、stage13-oat-20261101-182429（各 .md/.json 及 records 原始记录）；本地已有报告副本用于本次核对。
- 本次没有新增外部文献或硬件实测依据。已有证据不足的设置如实列为假设、约定或待验证值。
