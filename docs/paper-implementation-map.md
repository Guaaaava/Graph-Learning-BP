# 论文与当前实现对应清单

依据 GIB-Chap234-1016.tex 第 2、3、4 章，使用 TeX 标签定位。下表为当前可交付路径；初始问题和旧行号登记已归档于源仓库 legacy，不再将已删除的旧入口作为当前证据。

| 编号 | 论文要求 | 当前入口 | 实现/解释边界 |
| --- | --- | --- | --- |
| M01 | §2.1，`eq:measurement_model`、`eq:distance_noise_model`：每条无向边一次高斯测量，标准差 `sigma0 + alpha*d` | [generate_graph](../gib4cl/data/network.py) | 五锚点均匀智能体场景，统一通信半径，仅 LOS 加性高斯与已知方差 |
| M02 | §2.2，`eq:posterior_factorization`、`eq:measurement_factor`：先验乘所选边似然，锚点为已知坐标 | [ParticleBP.infer](../gib4cl/inference/particle_bp.py) | 完整先验与所选物理边因子 |
| M03 | `eq:bp_variable_to_factor`：先验乘除接收因子外的上一轮消息 | [ParticleBP.infer](../gib4cl/inference/particle_bp.py) | 有向 cavity 排除反向入消息 |
| M04 | `eq:bp_factor_to_variable`：对发送节点积分，并使用排除接收因子的消息 | [ParticleBP.infer](../gib4cl/inference/particle_bp.py) | 固定支持、分块积分、双向核复用 |
| M05 | `eq:bp_anchor_message`：直接代入已知锚点坐标 | [ParticleBP.infer](../gib4cl/inference/particle_bp.py) | 已知锚点静态因子 |
| M06 | `eq:bp_belief`：固定先验/锚点局部因子乘当前入消息 | [ParticleBP.infer](../gib4cl/inference/particle_bp.py) | 局部项与当前入消息组成 belief |
| M07 | `eq:bp_mean`、`eq:bp_covariance`：belief 的均值和中心二阶矩 | [ParticleBP.infer](../gib4cl/inference/particle_bp.py) | 归一化加权均值与中心协方差 |
| M08 | §2.3，`eq:exact_cavity_marginal`、`eq:tree_branch_factorization`、`eq:tree_exact_bp_marginal`：树上精确 BP 的参照 | [exact_discrete / quadrature helpers](../tests/references/particle_posterior.py) | 有限支持树上精确性与连续积分近似分开 |
| M09 | `eq:directional_overconfidence`：BP 协方差与真实后验边缘协方差的方向性比较 | [calculate_metrics](../gib4cl/evaluation/metrics.py) | 经验 NEES/覆盖不直接证明真实后验逐方向协方差关系 |
| M10 | §2.4，`eq:fim_affine_decomposition`：先验加每条边的信息贡献；未知量只包含智能体 | [gaussian_predictions](../gib4cl/learning/objectives.py) | 先验精度与所选边信息贡献 |
| M11 | §3.1：`C` 为选边后仍可用信息，`D={(z,v)}`，`R=(s, selected A)`；固定数值精度定义有限 `M` | [input_features / encode_measurements](../gib4cl/data/schema.py) | C/D/Y 访问边界与固定表示 |
| M12 | `eq:GIB_objective`：`-I(Y;R\|C)+beta*I(D;R\|C)` | [gib_objective](../gib4cl/learning/objectives.py) | 实现代理目标，不直接估计互信息 |
| M13 | `eq:task_kl_identity`、`eq:task_information_bound`、`eq:task_nll`：使用预测分布 NLL 优化任务信息下界 | [gaussian_task_terms](../gib4cl/learning/objectives.py) | 高斯任务 NLL |
| M14 | §3.2：`J0=blkdiag(Sigma0_i^-1)`，先验均值为初值 | [gaussian_predictions](../gib4cl/learning/objectives.py) | 完整节点先验的块对角精度 |
| M15 | §3.2：在初值处线性化，构造全局梯度 `g`、带符号残差 `r=z-dbar`、`max(v,sigma_min²)` | [gaussian_predictions](../gib4cl/learning/objectives.py) | 初值几何、带符号残差、显式方差下限 |
| M16 | `eq:local_prediction_product`、`eq:prediction_precision`：`Js=J0+Σ s*g*gT/vtilde`，`bs=Σ s*g*r/vtilde` | [gaussian_predictions](../gib4cl/learning/objectives.py) | J 与 b 的可微组装 |
| M17 | `eq:prediction_update`、`eq:prediction_distribution`：解 `Js*delta=bs`，`mu=Ybar+delta`，`S=Js^-1` | [gaussian_predictions](../gib4cl/learning/objectives.py) | Cholesky 解均值与联合协方差 |
| M18 | `eq:linear_task_loss`：`0.5*(-logdet Js+eT Js e)` | [gaussian_task_terms](../gib4cl/learning/objectives.py) | 半系数、对数行列式及误差二次项 |
| M19 | `eq:graph_energy`、`eq:expected_energy`：通信成本由 `C` 定义，例如 `rho*\|\|rbar_i-rbar_j\|\|²`，期望为 `Σ pE` | [compression_terms](../gib4cl/learning/objectives.py) | 初值定义通信成本，包含锚点边 |
| M20 | `eq:triangle_count`、`eq:expected_triangle`：只计算三个顶点全为智能体的三角形；每个三角形一次 | [agent_triangles](../gib4cl/data/schema.py) | 智能体三角形每个一次，压缩项用概率乘积 |
| M21 | `eq:max_entropy_formulation`、`eq:structured_prior`：固定 `gamma,kappa` 的能量参考分布 | [compression_terms](../gib4cl/learning/objectives.py) | 固定 gamma/kappa，移除与模型参数无关的配分常数 |
| M22 | `eq:data_reference`、`eq:complete_reference`：保留测量的均匀有限参考分布，结构与测量共同构成 `q_ref` | [LOG_MEASUREMENT_CARDINALITY](../gib4cl/data/schema.py) | 明确有限 binary32 支持和保留成本 |
| M23 | `eq:compression_kl_identity`、`eq:variational_bound`、`eq:p_factorized`、`eq:entropy_of_p_theta`、`eq:full_kl` | [compression_terms](../gib4cl/learning/objectives.py) | 负 Bernoulli 熵与固定参考展开，非完整 KL 数值 |
| M24 | `eq:compression_loss`：负熵、期望通信成本、期望智能体三角形、期望测量保留成本四项 | [compression_terms](../gib4cl/learning/objectives.py) | 负熵/期望能量/三角形/保留数据四项 |
| M25 | `eq:gib_training_bound`、`eq:gib_train_objective`：代理目标为任务项加 `beta*L_comp` | [gib_objective](../gib4cl/learning/objectives.py) | task + beta × compression |
| M26 | §4.1，`eq:residual_def`：节点 5 维、边 4 维；特征残差为绝对值 | [input_features](../gib4cl/data/schema.py) | 原始节点 5 维、边 4 维 |
| M27 | `eq:message_passing`、`eq:node_update`：邻居聚合，物理边双向传播 | [GNNLayer](../gib4cl/learning/model.py) | 双向聚合与共享原始边特征 |
| M28 | §4.2：两个端点顺序共用评分器，logit 取平均；每条物理边一个概率与决定 | [EdgePredictorGNN.score_edges](../gib4cl/learning/model.py) | 共享双序评分，logit 取平均 |
| M29 | §4.3，`eq:continuous_relaxation`：Logistic 噪声 `log u-log(1-u)` | [bernoulli_sigmoid](../gib4cl/learning/model.py) | Logistic 噪声，不使用单 Gumbel |
| M30 | `eq:forward_hard`、`eq:backward_ste`：前向二值、反向沿软变量 | [bernoulli_sigmoid](../gib4cl/learning/model.py) | 硬前向与软 STE 梯度 |
| M31 | §4.3：任务项经精度矩阵及均值线性求解回传 | [gaussian_predictions](../gib4cl/learning/objectives.py) | 精度与均值求解的梯度通路 |
| M32 | `eq:degree_loss`：智能体度数包括锚点边，使用采样 `s`；每图对智能体求和 | [degree_terms](../gib4cl/learning/objectives.py) | 采样权重、锚点边参与度数、每图求和 |
| M33 | `eq:total_loss`：`L_task+beta*L_comp+eta*L_deg`；压缩项用概率，任务与度数用采样 | [gib_objective](../gib4cl/learning/objectives.py) | task + beta × compression + eta × degree |
| M34 | §4.4：选边和预测分布不直接用 `Y`，真值仅用于任务损失；训练后子图交 BP | [evaluate](../gib4cl/evaluation/evaluate.py) | 实际检查点选边后交 BP，共享支持与完整诊断 |

数据、目标、GNN 和 BP 的核心检查分别位于 test_data_contract、test_objectives、test_model_sampling、test_particle_bp；指标和流程检查位于 test_metrics、test_experiment_pipeline、test_delivery；源代码包检查位于 test_source_package。以上文件均在 tests 下，完整命令见[验收说明](acceptance.md)。

理论恒等式、树上精确性条件和真实后验方向性不等式并非每图训练时直接数值计算的指标。小规模独立参照和经验校准证据保留其适用边界；正式性能和 GPU 预算仍需后续实验。
