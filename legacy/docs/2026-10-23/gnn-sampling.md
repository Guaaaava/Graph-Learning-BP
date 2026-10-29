# 第四步④：修正 GNN 和离散采样

本阶段已完成论文第 4 章的双向消息传递、无向对称评分、Logistic 离散采样和 STE 梯度，并接入第三步的训练目标。实现见 [model.py](../GNN_learning/model.py)，测试见 [test_model_sampling.py](../tests/test_model_sampling.py)，证据见 [stage4-verification.json](stage4-verification.json)。

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

模型只读取 `x, edge_attr, edge_index` 及三个消息映射字段；不读取位置标签、真实位置、NLOS 诊断或其他目标字段。

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

架构标识为 `gib4cl.gnn.symmetric-logistic.v1`。所有预设配置显式记录此版本，完整配置指纹会相应变化；数据指纹不受模型结构影响，第二步数据可直接复用。

模型 `state_dict` 内含架构版本。加载缺少版本或版本不符的权重会明确拒绝，即使调用方设置 `strict=False`。旧编码器、残差和评分结构的权重不能用于新版模型，需要重新训练。第六步已完成[完整检查点与断点恢复](unified-experiment.md)。

## 7. 验证结果与复现

新增 44 项 GNN/采样测试，与已有数据和目标测试联合共 **105 项通过**。重点证据包括：

- 真实消息 MLP/PyG 聚合与独立逐邻居参考计算一致；手工线性层验证锚点/智能体双向传播及孤立节点处理。
- 端点反转、任意节点重编号、物理边和消息边重排、批处理与单图结果对应；含锚点边及无边图。
- 80,000 次五组概率 p=(0.01,0.1,0.5,0.9,0.99)，在温度 (0.1,1,5) 下频率均落入预先指定的六倍标准差容差；120,000 次双边样本的四种联合频率与独立 Bernoulli 乘积一致。
- 固定噪声的 STE 前向值及解析梯度一致；极端 logits、空张量、非法输入及推理不消耗随机数均验证。
- 真实 GNN 与新版总目标联合反传，各消息层和评分器的梯度有限且非零；新权重保存/加载一致，旧版本拒绝加载。

仓库根目录执行：

```powershell
.\.venv-data\Scripts\python.exe -B -m pytest tests/test_model_sampling.py tests/test_objectives.py tests/test_data_contract.py -q -p no:cacheprovider --basetemp=data/test-tmp-stage4-rerun
.\.venv-data\Scripts\python.exe -B -m tests.verify_stage4 --output-root outputs/stage4-smoke-rerun
```

第二条命令需要第二步的 `data/stage2-smoke` 小数据；如需重建，可执行：

```powershell
.\.venv-data\Scripts\python.exe -B -m GNN_learning.generate_datasets --config configs/smoke-data.json --output-root data/stage2-smoke
```

生成/训练均有覆盖保护；若对应输出已存在，复用数据或换一个输出目录。CPU 审计脚本不在 CUDA 环境运行。

本次用 3/2/2 张训练/验证/测试小图运行两次一轮训练。两次日志、最终参数逐项完全一致，42 组参数张量发生更新；最佳/最终权重重新加载后，7 张图的评分、选边、目标及预测均值/协方差均通过检查。完整本地产物在 `outputs/stage4-smoke/`，摘要在 [stage4-verification.json](stage4-verification.json)。

这组系数仅用于验证接口，不代表收敛、定位效果或正式调参结果。CPU 验证不覆盖 GPU。第四步当时保持 BP/NEBP 源码原样；后续第五步已完成[粒子 BP 与独立参照](particle-bp.md)，第六步已完成[统一实验入口](unified-experiment.md)。论文、基线快照和 NEBP 源码继续保持原样，服务器/GPU 环境仍待验收。
