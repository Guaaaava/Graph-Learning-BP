整体流程为：

- **公共数据链**：实验配置 → 场景与测量生成 → 数据校验与保存 → 特征构建与图装配。
- **训练链**：图数据 → GNN 打分 → 随机选边 → 可微高斯预测 → 损失 → 参数更新 → 检查点。
- **评估链**：图数据＋检查点 → Dense/BFS/GNN 拓扑 → 粒子 BP → 指标 → 统计与报告。

| 编号 | 阶段 | 模块 | 输入 → 输出 | 主要实现 |
|---|---|---|---|---|
| 1 | 公共 | 实验配置与种子管理 | JSON 配置 → 数据、模型、训练、BP、评估参数及独立种子 | [config.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/config.py) |
| 2 | 公共 | 场景、网络与测量生成 | 场景参数 → 智能体真值、锚点、初始位置、候选边、测距与方差 | [network.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/data/network.py:35) |
| 3 | 公共 | 数据协议与合法性校验 | 原始图记录 → 满足统一索引、形状、测量及先验约束的数据 | [schema.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/data/schema.py:66) |
| 4 | 公共 | 数据集构建与持久化 | 单图生成器＋划分及种子 → train/val/test 数据文件、元数据与校验值 | [generate.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/data/generate.py:43) |
| 5 | 公共 | 特征构建、图装配与批处理 | 图记录 → 5 维节点特征、4 维边特征、双向消息边映射、三角形索引及 PyG 数据 | [dataset.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/data/dataset.py:31)，调用模块 3 的特征派生函数 |
| 6 | 训练／评估共用 | GNN 消息传递与边打分 | 节点、边特征 → 节点表示 → 每条物理边的对称 logit 与保留概率 | [EdgePredictorGNN](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/learning/model.py:109) |
| 7 | 训练 | Bernoulli 采样与 STE | 边 logits＋温度＋随机噪声 → 前向为 0/1、反向可传梯度的边掩码 | [bernoulli_sigmoid](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/learning/model.py:19) |
| 8 | 训练 | 可微高斯预测器 | 初始位置、完整先验、测量与选边掩码 → 联合精度矩阵、信息向量、预测均值 | [gaussian_predictions](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/learning/objectives.py:122) |
| 9 | 训练 | 训练目标与损失汇总 | 高斯预测、位置标签、边概率及掩码 → 任务项、四项压缩项、度数项、总损失 | [gib_objective](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/learning/objectives.py:288) |
| 10 | 训练 | 训练、验证与参数优化 | 批数据＋损失 → 反向传播、梯度裁剪、参数更新、温度和学习率调度、验证日志 | [train.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/learning/train.py:39) |
| 11 | 训练→评估 | 检查点、恢复与模型加载 | 模型及训练状态 ↔ 检查点；检查点 → 评估用选边模型 | [checkpoints.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/learning/checkpoints.py)，保存及完整恢复由模块 10 配合完成 |
| 12 | 评估 | 拓扑选择与基线构建 | 候选图＋已训练 GNN → Dense 全边、BFS 森林、GNN 阈值选边三种掩码 | [topology.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/evaluation/topology.py:8)、[spanning_tree.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/inference/spanning_tree.py:14) |
| 13 | 评估 | 粒子 BP 定位推理 | 先验、测量、拓扑掩码、固定粒子支持 → 后验权重、位置均值、协方差及数值诊断 | [particle_bp.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/inference/particle_bp.py:162) |
| 14 | 评估 | 单次定位与结构指标 | 定位结果＋真值＋所选拓扑 → RMSE、MAE、失效率、覆盖率、NEES、NLL、通信成本及结构指标 | [calculate_metrics](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/evaluation/metrics.py:75)，结构指标由模块 12 计算 |
| 15 | 评估 | 跨图统计与配对比较 | 各图、各 BP 重复的指标记录 → 图级均值、标准差、bootstrap 区间及拓扑配对差异 | [summarize_records](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/evaluation/metrics.py:249) |
| 16 | 评估 | 评估流程与报告输出 | 检查点＋测试集 → 三拓扑配对评估、JSON 结果、CSV 指标表与 Markdown 报告 | [evaluate.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/evaluation/evaluate.py:146) |
| 17 | 可选展示 | 拓扑可视化 | 原始图记录＋选边掩码 → Dense/BFS/GNN 拓扑对比图 | [plotting.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/evaluation/plotting.py:6) |

另外还有贯穿上述流程的工程模块：

| 模块 | 职责 | 实现 |
|---|---|---|
| 统一运行入口 | 分派 generate、train、evaluate、run、suite 命令 | [cli.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/experiments/cli.py:42) |
| 运行环境与可复现支持 | 设备、线程、随机状态、数据身份、源码摘要及原子写入 | [runtime.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/runtime.py) |
| 多种子实验编排 | 固定数据，组织多训练种子与 BP 种子任务，支持恢复及汇总 | [suite.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/experiments/suite.py:301)、[worker.py](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/gib4cl/experiments/worker.py:8) |
| 测试、独立参照与交付验收 | 数据、数学目标、BP、指标、恢复及打包校验 | [tests](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/tests)、[tools](E:/硕士/学术资料/研一/课题1-GIB4CL/GIB4CL-Experiment/Graph-Learning-BP/tools) |

已完成模块和发现的问题：

编号 1：实验配置与种子管理（config.py）
1. `config.py` 四个超参数还是 `None`
2. `config.py` BP 迭代轮数只有 3
3. `config.py` 中 `edge_threshold: float = 0.5` 是否合理
4. 所有参数都应该有依据

编号 2：场景、网络与测量生成
编号 3：数据协议与合法性校验
编号 4：数据集构建与持久化模块
编号 5：特征构建、图装配与批处理模块
（network.py、schema.py、generate.py、dataset.py）
tools/preview_network.py 测试

编号 6：GNN 消息传递与边打分
编号 7：Bernoulli 采样与 STE
（model.py）
1. GNN 架构的设计（每一层是什么，有什么作用等）也是应该有依据的

编号 8：可微高斯预测器
编号 9：训练目标与损失汇总
（objectives.py）

编号 10：训练、验证与参数优化
编号 11：检查点、恢复与模型加载
工程模块：运行环境与可复现支持
（train.py、checkpoints.py、runtime.py）
1. 梯度裁剪和权重衰减是否有必要？（可能需要结合训练日志判断）

编号 12：拓扑选择与基线构建（topology.py、spanning_tree.py）
1. 生成树当前是每个锚点的多源生成森林，是否要改？

编号 13：粒子 BP 定位推理（particle_bp.py）
1. 粒子 BP 在树和有环图上的对比实验，用于支撑第二章的过度自信说明

编号 14：单次定位与结构指标
编号 15：跨图统计与配对比较
（metrics.py）

编号 16：评估流程与报告输出（evaluate.py）
编号 17：拓扑可视化（plotting.py）
工程模块：统一运行入口（cli.py）
工程模块：多种子实验编排（suite.py、worker.py）
工程模块：测试、独立参照与交付验收（tests/、tools/）