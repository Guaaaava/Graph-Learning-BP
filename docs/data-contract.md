# 当前图数据契约与配置

当前生产数据只有一种场景：正方形区域的四角加中心共五个锚点，智能体在整个区域均匀分布，所有测量边使用同一个通信半径，测距采用 LOS 加性高斯模型。

## 配置入口

默认配置为 `configs/experiment.json`。输入不填写配置、模型或 BP 版本，不接受旧种子别名、其他布局/分布选项或历史格式迁移。未知字段和重复 JSON 键均报错。省略的当前字段使用配置类默认值；保存时展开实际生效的完整配置。

`GRAPH_VERSION` 和 `DATASET_VERSION` 是读写文件时检查的格式标识，不是场景选项或多版本实现。它们用于拒绝不兼容的图记录和数据集容器；张量形状、索引、字段及配置身份仍逐项验证。数据与目标函数模块统一使用包导入，重复且相同的 `if __package__` 分支已合并。

| 配置 | 区域边长 | 智能体数 | 统一通信半径 | 用途 |
| --- | ---: | ---: | ---: | --- |
| experiment.json | 100 m | 25–35 | 30 m | 当前正式场景参数模板，损失系数仍须显式选择 |
| smoke-pipeline.json | 20 m | 2–5 | 15 m | 同一种场景规则的小规模流程检查 |

锚点坐标由区域边长 L 确定为 `(0,0)、(L,0)、(0,L)、(L,L)、(L/2,L/2)`。通信半径保持显式给定，不随区域大小、锚点数量或图连通性自动重算。默认值延续当前设定，不代表所有数值已经完成正式性能标定。

生成命令：

```text
python -B -m gib4cl generate --config configs/experiment.json --data-root data/new-five-anchor-data
```

训练前需要在配置或命令行中明确 beta、gamma、kappa、eta。数据生成本身不要求训练系数已确定。

## 图记录与索引

```text
schema_version / sample_id
inputs
  initial_positions         # (N,2)，智能体初值和五个锚点精确坐标
  initial_covariances       # (A,2,2)，完整智能体先验协方差
  is_anchor                 # (N,)
  edge_index                # (2,E)，每条物理测量边一次
  measurements              # (E,)，z
  measurement_variances     # (E,)，v
  communication_cost_scale  # rho
  measurement_encoding      # 固定有限 binary32 表示
targets
  agent_positions           # (A,2)，训练标签或评估真值
metadata                     # 完整数据参数、样本种子、划分、单位和模型假设
diagnostics                  # 候选度数与锚点可达性，不进入模型输入
```

其中 A 为智能体数，N=A+5。智能体索引在前、锚点在后。每条物理边满足全局 `u<v`，不含自环、重复边或锚点—锚点边。候选边由真实距离严格小于 `comm_radius` 决定，智能体边和锚点边使用相同规则。

空图、孤立节点和无锚点路径的分量仍保留并报告，不筛除、不补边。至少需要一个智能体；已连接节点的初始坐标重合会报错，以满足论文的线性化条件。

## 物理测量与先验

测距为 `z=d+N(0,(sigma0+alpha*d)^2)`，不裁剪负测量。方差为根据真实物理距离计算的已知正方差；这是论文的理想化输入假设。

数据生成配置支持标量、两个对角方差或共享对称正定 2×2 先验。图适配器和数学模块保留对每智能体完整 `(A,2,2)` 先验的支持。标量 25 表示 25I，不能误作标准差 25 m。

测量使用规范化有限 float32 表示；排除 NaN/Inf、规范化有符号零并要求方差为正。固定表示的有限支持确定压缩损失中的 log(M)。训练预测器与 BP 可使用 float64 内部计算，测量编码仍保持 float32。

## 特征与批处理

`graph_to_data(record)` 仅从 inputs 派生特征，include_targets=False 时不附加 y。

| 字段 | 形状及作用 |
| --- | --- |
| x | `(N,5)`：初始坐标、两个对角先验方差、锚点标志 |
| edge_attr | `(E,4)`：测距、已知方差、绝对初值残差、锚点边标志 |
| agent_prior_cov | `(A,2,2)`：完整先验，保留相关项 |
| agent_node_index / anchor_node_index | 显式全局节点映射 |
| initial_distances / signed_residual | 初值距离与带符号残差 z-dbar |
| edge_directions | 初值几何的单位方向 |
| edge_cost | rho×初值距离平方，包含锚点边 |
| edge_log_cardinality | 每条物理测量边的 log(M) |
| message_edge_index | `(2,2E)`：每条物理边对应两个方向的消息 |
| message_edge_to_measurement / message_reverse_index | 消息到物理边及反向消息映射 |
| triangle_edge_index | `(3,T)`：每个纯智能体三角形一次 |

物理边用于评分、采样、损失和通信计数；双向消息边只用于 GNN 传播。PyG 批处理分别对节点、物理边和消息边索引递增。数学单元测试可以使用手工小图、空图或无锚点图检查边界，它们不是额外生产场景。

## 种子、身份与持久化

每个样本的 seed 由主 data 种子、split、seed_offset 和编号独立派生；用私有 Python RNG 选节点数、私有 PyTorch Generator 生成观测，不重设全局随机状态。新增后续样本不会改变已有前缀。

数据指纹由当前完整 DataConfig 和 data 种子决定，训练、BP、bootstrap 种子不改变观测身份。series_id 另包含实验名、split、seed_offset 和实际节点数范围，dataset_id 还包含图数。文件身份由 SHA-256 确认。

文件及 sidecar 默认拒绝覆盖。加载器核对配置指纹、样本身份/种子、编码、数量、固定五锚点坐标、真值区域范围和校验值。图记录与数据集保留内部格式标识；旧列表数据和旧配置字段不再自动转换。本次精简产生新数据指纹，应在新目录生成当前数据。

配置对象的 split_sizes 是只读映射，to_dict 导出普通字典供 JSON 和检查点保存。完整运行与恢复要求见 [运行说明](unified-experiment.md) 和 [种子与编排](delivery-and-seeds.md)。
