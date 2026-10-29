# GNN、无向选边与 Logistic STE

当前可交付实现只使用 `gib4cl/`。原有阶段记录与旧源码已集中归档至源仓库的 `legacy/`；发布包通过[固定来源数据](provenance.md)保留校验依据，运行不依赖历史目录。打包与完整验收步骤见[验收说明](acceptance.md)。

本阶段已完成论文第 4 章的双向消息传递、无向对称评分、Logistic 离散采样和 STE 梯度，并接入第三步的训练目标。实现见 [model.py](../gib4cl/learning/model.py)，测试见 [test_model_sampling.py](../tests/test_model_sampling.py)，证据见 [来源与历史记录](provenance.md)。

## 1. 图与消息的语义

原始 `edge_index` 仍是一条无向测量边一次，形状为 `(2,E)`。GNN 只沿 `message_edge_index` 的 `2E` 条有向消息边传播，通过 `message_edge_to_measurement` 复用同一测量特征。两条消息互为反向，模型检查映射、反向配对、重复物理边和跨图连边。

每层按论文 `eq:message_passing`、`eq:node_update` 计算：

```text
m_ij = MLP_msg([h_i, h_j, e_ij])       # i 为接收节点，j 为发送节点
h_i' = MLP_upd([h_i, sum_j m_ij])
```

智能体和锚点都发送、接收消息。孤立节点的邻居和为零，仍经过更新 MLP；不隐式添加自环。消息聚合是求和，不按度数平均。每条物理边在概率、选边、通信成本、三角形和度数计算中仍只出现一次。

## 2. 对 D08 的实现选择

首层节点状态就是论文的 5 维输入；边特征在所有层和评分器中保持原始 4 维。不再添加单独的输入编码，也不再执行旧代码的 `x+x_new` 外加残差。LayerNorm 保留在 MLP 内部，是论文未指定的 MLP 结构细节；它按单个节点/消息的隐通道归一化，不在不同图之间汇总统计。

默认 `hidden_dim=64, num_layers=3`，与现有显式配置一致：

| 模块 | 输入/输出及内部结构 |
| --- | --- |
| 第 1 层消息 MLP | 14 → 64 → 64 |
| 第 1 层更新 MLP | 69 → 64 → 64 |
| 后续层消息 MLP | 132 → 64 → 64 |
| 后续层更新 MLP | 128 → 64 → 64 |
| 消息、更新 MLP 的内部顺序 | Linear → LayerNorm → ReLU → Linear |
| 共享边评分 MLP | 132 → 64 → 32 → 1；前两次 Linear 后接 ReLU |

这些宽度和层数是工程预设，不是正式实验已经确定的最优结构。论文实验设置应交代 MLP 细节。

## 3. 无向边评分

两个端点顺序使用同一个评分网络，使用相同边特征：

```text
a_uv = score([h_u, h_v, e_uv])
a_vu = score([h_v, h_u, e_uv])
logit_uv = (a_uv+a_vu)/2
p_uv = sigmoid(logit_uv)
```

平均的是两个 logit，不是两个概率。模型输出始终按物理边顺序排列，长度为 E。交换端点、重排节点或重排消息边不会改变对应物理边的评分。重排物理边后，概率和选边也按同一顺序对应；随机训练比较需要同时对应每条边的噪声，不能只重置随机种子后忽略边顺序。

模型只读取 `x, edge_attr, edge_index` 及三个消息映射字段；不读取位置标签、真实位置或诊断目标字段。

## 4. Logistic 噪声与 STE

旧代码使用单个 Gumbel 噪声，logit 为零时的硬保留率约 0.6321。新版按论文计算：

```text
u ~ Uniform(0,1)                     # 每条物理边独立一个 u
L = log(u)-log(1-u)                  # Logistic 噪声
soft = sigmoid((logit+L)/tau)
binary = 1{logit+L > 0}
s = binary + (soft-soft.detach())
```

这与 `eq:continuous_relaxation`、`eq:forward_hard`、`eq:backward_ste` 相符。在精确算术下，`soft>0.5` 与 `logit+L>0` 等价；实现直接比较后者，避免 sigmoid 将非常小的正数舍入成 0.5。前向严格为 0/1，反向梯度为 `soft*(1-soft)/tau`，是有偏 STE 近似。

硬训练样本的边缘分布为 Bernoulli(sigmoid(logit))。正温度改变松弛及梯度，不改变这个硬采样分布；不能把较高温度解释为提高硬拓扑的探索概率。

数值约定：

- 随机数使用 logits 的 dtype/device；PyTorch 的均匀随机数可能等于零，因此遇零重新抽取，不对噪声作截断。
- 用 `log1p(-u)` 计算第二项。外部指定的均匀随机数必须严格在 (0,1) 内，形状、dtype 和设备均匹配。
- 温度必须有限、为正，且在当前 dtype 的正常可表示范围内；非有限 logits 明确报错。
- 可传入独立 `torch.Generator`，或每条物理边对应的 `uniform`，两者不可同时提供。后者只作为固定噪声输入，不参与求导。
- `gumbel_sigmoid` 保留为兼容函数名，实际转调正确的 `bernoulli_sigmoid`。

## 5. 验证/推理默认规则（D10）

`model.train()` 时使用上述独立随机采样。`model.eval()` 时不添加噪声，也不消耗随机数：

| 调用 | 返回 |
| --- | --- |
| `model(data, hard=True)` | `logit>0` 的确定性二值选边，数学上对应 `p>0.5`，零 logit 不保留 |
| `model(data, hard=False)` | 原始 Bernoulli 参数 `sigmoid(logit)`，不受温度影响 |
| `model.score_edges(data)` | 对称 logit，不采样 |
| `model.edge_probabilities(data)` | 原始 Bernoulli 参数，不采样 |

论文没有指定部署策略，此处是明确记录的工程默认。主训练的验证损失仍基于确定性子图，不等同于对随机训练目标进行 Monte Carlo 期望估计。后续统一评估如增加阈值扫描或随机部署，应独立标记。当前不强制保留锚点边，不补边修复连通性。

## 6. 新旧权重隔离

模型不再维护独立的架构版本标签。配置记录实际输入维度、隐藏维度和层数；数据指纹不受模型结构影响。

模型 `state_dict` 只保存参数张量，使用 PyTorch 默认的严格参数名和形状检查。完整检查点另核对保存的模型配置，并记录训练源码和数据身份；恢复训练要求源码及运行环境一致。旧权重中带有已移除的附加状态时不自动转换。详见[完整检查点与断点恢复](unified-experiment.md)。

## 当前验收入口

在项目根目录执行 `python -B -m tools.accept_cpu --output-root outputs/new-cpu-acceptance`，使用一个新目录。该命令重跑当前数学、数据、训练/恢复、指标、编排和打包检查；具体通过条件见[完整验收说明](acceptance.md)。历史验证数字不代表本候选版本测试数量或正式性能，当前数量以 JUnit 报告为准。
