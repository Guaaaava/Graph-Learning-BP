# Code Q&A 速查表

## 数据流  

```
generate_network.py (场景参数)
  → raw dict {true_pos, init_pos, edge_index, measurements, variances, ...}
  → dataset.py (组装 PyG Data, 5维节点+4维边)
  → train.py / evaluate.py
  → model.py (GNN → edge_weights 0/1)
  → particle_bp.py (粒子BP定位)
  → evaluate.py (RMSE, NEES, CRLB, 中断率)
```

## 关键数字

- 节点特征: `[x, y, σ²x, σ²y, is_anchor(0/1)]`  →  `data.x[:, 4]` 判断锚点
- 边特征: `[测距值z, 方差σ², 伪距残差r, is_anchor_edge(0/1)]` → `data.edge_attr[:, 3]`
- FIM: 每条边贡献 `J_e = (w/σ²) * u*u^T` (2×2), 组装为 `(2Na)×(2Na)` 块矩阵
- 先验: `q(d) = 1/(1+exp(γ·d²))`, γ=0.002, d=30m→q≈0.14
- KL: `KL(Bernoulli(p)||Bernoulli(q))` 逐边计算取平均
- 度数: `ReLU(3 - degree_i)` 只惩罚 agent 节点
- FIM归一化: `loss_fim = total_fim / sum(2*Na)`, 行号 `loss.py:139`

## 场景配置

| 场景 | 锚点 | Agent | NLOS | 通信半径 | 边数(30ag) |
|------|------|-------|------|---------|-----------|
| normal | rand | rand | 0% | R | ~160 |
| hard | corners+center | rand | 0% | R | ~150 |
| challenge | corners only | cluster | 15% | 1.8R | ~380 |

## 粒子BP 四步

1. **Anchor权重**: `log_w[i] += -(||p_i - a_k|| - z)² / (2σ²)`
2. **Agent消息**: `msg_v→u = Σ_v exp(-(d(p_u,p_v)-z)²/(2σ²)) · belief_v`
3. **信念更新**: `belief = softmax(log_anchor + Σ log(msg))`
4. **ESS重采样**: if `1/Σw² < P/2` → multinomial resample + roughening noise

## 评估6种拓扑

| 拓扑 | 生成方式 | 文件位置 |
|------|---------|---------|
| Dense | `torch.ones(E)` | evaluate.py:290 |
| BFS | `extract_bfs_weights()` | evaluate.py:129 |
| MST | `baseline_mst()` | evaluate.py:43 |
| k-NN | `baseline_knn(k=3)` | evaluate.py:57 |
| Random | `baseline_random(0.5)` | evaluate.py:73 |
| GNN | `model(data, hard=True)` | evaluate.py:295 |

## 关键Git记录

- `clean` 分支: 从 `devnew` 切出, `.gitignore` 排除 datasets/models/logs/figs
- 本地 3050 4GB → P×P 矩阵 OOM → 加 CPU-BP fallback
- 服务器 3090 24GB → GPU-BP P=10000 每BP 4秒
- `loss.py:139` FIM归一化是关键改动
- `exp_50agent.py` 是当前跑的实验脚本

## 附录

### G. GNN 架构

| 组件       | 规格                                              |
| ---------- | ------------------------------------------------- |
| 节点编码器 | Linear(5→64)                                      |
| 边编码器   | Linear(4→64)                                      |
| 消息函数   | Linear(192→64) + LayerNorm + ReLU + Linear(64→64) |
| 更新函数   | Linear(128→64) + LayerNorm + ReLU + Linear(64→64) |
| 层数       | 3层 MessagePassing + 残差连接                     |
| 边评分器   | Linear(192→64→32→1)                               |
| 输出       | Gumbel-Sigmoid STE ({0,1} 硬掩码)                 |
| 参数量     | 102,721                                           |

### H. 粒子 BP 算法

| 步骤          | 操作                                                 | 位置               |
| ------------- | ---------------------------------------------------- | ------------------ |
| 1. 初始化     | 从 N(init_pos, 25) 采样 P 个 2D 粒子                 | particle_bp.py:106 |
| 2. Anchor权重 | log_w += -(‖p_i - a_k‖-z)² / (2σ²)                   | particle_bp.py:132 |
| 3. Agent消息  | msg_v→u = Σ_v exp(-(d(p_u,p_v)-z)²/(2σ²)) * belief_v | particle_bp.py:157 |
| 4. 信念更新   | belief = softmax(anchor_log + Σ agent_log)           | particle_bp.py:173 |
| 5. 重采样     | ESS < P/2 时 multinomial + roughening noise          | particle_bp.py:186 |

## Q&A

### Q1: 为什么 FIM 是 2N_a × 2N_a 矩阵？

每个 agent 有 **2 个待估计参数**（x 坐标和 y 坐标）。N_a 个 agent 的完整状态向量是：

$$\mathbf{\theta} = [x_1, y_1, x_2, y_2, ..., x_{N_a}, y_{N_a}]^T \quad \in \mathbb{R}^{2N_a}$$

Fisher 信息矩阵定义为 score function 的外积：

$$J = \mathbb{E}\left[ \frac{\partial \log p(\mathbf{z}|\mathbf{\theta})}{\partial \mathbf{\theta}} \cdot \frac{\partial \log p(\mathbf{z}|\mathbf{\theta})}{\partial \mathbf{\theta}}^T \right]$$

$\mathbf{\theta}$ 是 2N_a 维的，外积自然是 **2N_a × 2N_a**。每条测距边贡献一个 2×2 的子块：

```
agent 0 的 x,y: [ J00  J01 ] [ J02  J03 ]   ← agent 0-1 交互
agent 1 的 x,y: [ J10  J11 ] [ J12  J13 ]
                ...
```

代码中 `loss.py:128`: `FIM = FIM_blocks.permute(0,2,1,3).reshape(2*N_a, 2*N_a)`

每条边对 FIM 的贡献 J_e = (w/σ²) · u·u^T 是一个 2×2 的 rank-1 矩阵，u 是单位方向向量。对于 agent-agent 边，J_e 加到四个位置（对角块和反对角块）；对于 anchor 边（anchor 位置已知），只加到 agent 的对角块。

### Q2: 先验分布 q 中的 γ = 0.002 是怎么确定的？

γ 不是调的，是**从通信半径反推**的。先验公式：

$$q(d) = \frac{1}{1 + \exp(\gamma \cdot d^2)}$$

我们希望通信半径边界处（d=30m）的边有较低的保留先验（因为远距离通信代价高、噪声大）。设目标 q(30) ≈ 0.14：

$$q(30) = \frac{1}{1 + \exp(\gamma \cdot 900)} = 0.14 \quad \Rightarrow \quad \gamma \cdot 900 = \ln(1/0.14 - 1) \approx 1.8 \quad \Rightarrow \quad \gamma = 0.002$$

量级规律：$\gamma \approx 1/(R/2)^2 = 1/225 \approx 0.004$，取 0.002 略保守。对应不同距离：

| d | q(d) | 含义 |
|---|------|------|
| 0m | 0.50 | 最近的边 50% 先验保留 |
| 15m | 0.39 | |
| 30m | 0.14 | 通信边界，低概率保留 |
| 50m | 0.007 | 远超通信半径，几乎不可能保留 |

### Q3: 小规模训练/大规模测试的 2000/500/500 到底是什么意思？

三个数据集在实验中**被复用了两次**：

```
Phase 1: 生成三份数据
  train.pt  = 2000 图, 25-35 agents  ← 小规模训练集
  val.pt    =  500 图, 25-35 agents  ← 小规模验证集
  test.pt   = (未生成)

Phase 2-3: 调参 + 训练 (在小规模上)
  用 train(25-35) 训练 200 epoch
  用 val(25-35) 验证调参

Phase 4: 覆盖生成大规模数据
  val.pt    =  500 图, 45-55 agents  ← 覆盖! 变成大规模微调集
  test.pt   =  500 图, 45-55 agents  ← 大规模测试集

Phase 5: 微调 (30 epoch)
  加载 Phase 3 训练好的模型
  在 val(45-55) 上继续训练 30 epoch  ← 这就是微调
  目的: 让模型适应大图的边密度和连通模式

Phase 6: 测试
  在 test(45-55) 上评估, 不训练
```

**本质是三阶段**：小图训练 200ep → 大图微调 30ep → 大图测试。

"500微调"的意思是 val 数据集被覆盖为大图版本后，模型在上面少量 epoch 适应大图，不是从头训练。GNN 的 MessagePassing 天然 perm-invariant，学到的"高质量边 vs 噪声边"判别准则与图大小无关，微调只是让模型适应大图的边密度量级。

**虽然是无监督学习，但是验证集用来防止模型过拟合，验证泛化性。**

### Q4: 粒子数 P=2000 / 10000 是怎么确定的？

三个来源共同决定：

**1. NEBP 论文参考，但场景不同**

原 NEBP 代码 (`bp_test.py:142`) 使用 P=50000，但他们是动态跟踪场景——50个时间步、4D状态空间 `[x, y, vx, vy]`，分布随时间扩散需要更多粒子。我们的场景是静态单快照、2D 位置，分布更集中，所需粒子数自然更少。

**2. 实测对比，有边际递减**

我们在一张测试图上实测了不同 P 值的效果：

| P | RMSE (m) | 单次BP | 评估500图×6 |
|---|---------|--------|------------|
| 500 | 2.20 | 0.3s | 15min |
| 2000 | 1.22 | 0.3s | 15min |
| 5000 | 0.65 | 1.1s | 55min |
| 10000 | 0.22 | 4.0s | 3.3h |

P 从 500→2000 提升最大，之后边际递减。P=10000 的 RMSE（0.22m）已接近此场景的理论下界，再提高意义不大。

**3. 训练和评估分离**

训练阶段 GNN 用 GIB 损失（FIM+KL 公式），不跑粒子 BP，所以 P 不参与训练。粒子 BP 只在评估阶段运行，用来对比不同拓扑的实际定位精度。因此在评估时可以用大 P 追求准确，不影响训练成本。2000 用于中间测试/验证，10000 用于最终评估出表。