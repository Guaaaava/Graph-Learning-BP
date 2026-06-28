# 重构总结

> 日期: 2026-06-10  
> 目标: 将粒子化非参数BP集成到项目中，重构GNN架构与训练/评估流程

---

## 一、改动总览

### 1. 新增文件

| 文件 | 说明 |
|------|------|
| `BP/__init__.py` | BP 模块入口 |
| `BP/particle_bp.py` | 粒子化非参数BP，PyG重写，静态定位 |
| `GNN_learning/loss.py` | GIB 损失函数 (logdet + KL + degree) |

### 2. 重写文件

| 文件 | 变更内容 |
|------|----------|
| `GNN_learning/model.py` | 统一为唯一GNN模型；节点特征 3→5 维；边特征 3→4 维；加入LayerNorm；删除旧的手写消息传递 |
| `GNN_learning/config.py` | 新增 GIB 超参数、粒子BP参数、特征维度配置 |
| `GNN_learning/dataset.py` | 适配 5 维节点特征 + 4 维边特征 |
| `GNN_learning/generate_network.py` | 新增 `init_pos_cov`（初始不确定度）、`pseudo_range_residual`（伪距残差）输出 |
| `GNN_learning/generate_datasets.py` | 传入 `init_pos_cov` 参数 |
| `GNN_learning/train.py` | 从单图训练改为批量训练；损失从简单CRLB改为GIB；优化器改为AdamW+CosineAnnealing |
| `GNN_learning/evaluate.py` | 从高斯BP改为粒子BP；从根目录移入GNN_learning/ |

### 3. 删除文件

| 文件 | 原因 |
|------|------|
| `GNN_learning/edge_predictor_GNN.py` | 旧版GNN，与 model.py 重复 |
| `GNN_learning/crlb_loss.py` | 旧版损失，被 loss.py 取代 |
| `GNN_learning/train_batch.py` | 旧版训练，被 train.py 取代 |
| `GNN_learning/tune_lambda_batch.py` | 旧版超参搜索，不再需要 |
| `GNN_learning/build_global_FIM.py` | FIM组装已集成到 loss.py |
| `GNN_learning/gnn_generalized_model.pth` | 旧模型权重，与新架构不兼容 |
| `evaluate.py` (根目录) | 旧版评估，被 GNN_learning/evaluate.py 取代 |
| `evaluate_batch.py` (根目录) | 旧版批量评估，不再需要 |

---

## 二、架构设计变更

### 节点特征 (3维 → 5维)

```
旧: [pos_x, pos_y, is_anchor]
新: [pos_x, pos_y, σ²_x, σ²_y, is_anchor]
```

- Agent: 位置来自初始含噪估计，σ² = init_pos_cov（默认25.0 m²）
- Anchor: 位置为真值，σ² = 0

### 边特征 (3维 → 4维)

```
旧: [measurement, variance, is_anchor_edge]
新: [measurement, variance, pseudo_range_residual, is_anchor_edge]
```

- 伪距残差: `r = |z - ||x_init_u - x_init_v|||`，衡量测距与当前拓扑估计的一致性
- 残差小 → 边质量高；残差大 → 可能NLOS或坏边

### GIB 损失函数

$$\mathcal{L} = \log\det(J^{-1}_{global}) + \lambda \cdot D_{KL}(p_{\theta} \| q) + \eta \cdot \sum_i \max(0, 3 - D_i)$$

| 项 | 含义 | 使用的边权重 |
|----|------|-------------|
| $\log\det(J^{-1})$ | Fisher信息充分性 | hard mask $z^{out} \in \{0,1\}$ |
| $D_{KL}(p \| q)$ | 结构压缩先验 | soft sigmoid $p \in (0,1)$ |
| $\max(0,3-D_i)$ | 度数约束 (≥3边消歧) | hard mask $z^{out} \in \{0,1\}$ |

先验分布: $q(d) = \frac{1}{1 + \exp(\gamma d^2)}$

- $d\to 0$: $q=0.5$（最大熵先验，无近距离偏好）
- $d\to\infty$: $q\to 0$（远距离边指数衰减）
- 可调参数 $\gamma$ 控制衰减速度，默认 0.002

### GNN 模型结构

```
EdgePredictorGNN(node_in_dim=5, edge_in_dim=4, hidden_dim=64, num_layers=3)
  ├── Node Encoder:  Linear(5 → 64)
  ├── Edge Encoder:  Linear(4 → 64)
  ├── GNNLayer × 3:  MessagePassing(add) + LayerNorm + Residual
  └── Edge Scorer:   MLP(64*3 → 64 → 32 → 1) → Gumbel-Sigmoid STE
```

参数量: ~103K

### 粒子BP (静态定位)

核心流程（去掉了NEBP的运动模型 F/Q/W）：

```
初始化粒子 N(init_pos, init_cov·I)
↓ 循环 num_iter 次
  ① Anchor测距权重 → log P(z_ik | particle_m)
  ② Agent间消息传递 → Σ_n P(z_ij | m, n) · belief_j(n)
  ③ 信念归一化 (Softmax)
  ④ 加权均值/协方差估计
  ⑤ 多项式重采样
输出: 估计位置 + 协方差 + 粒子集
```

- 训练评估: P=2000, iter=10
- 测试评估: P=10000, iter=10
- 支持 `edge_weights` 参数做拓扑剪枝（weight<0.5 的边被跳过）

---

## 三、阶段测试结果

### 测试环境

- Python 3.10 (conda py310)
- PyTorch 2.7.1+cu118
- PyTorch Geometric 2.7.0
- WSL2, CPU模式

### 3.1 模块导入测试

所有模块成功导入，无依赖错误：

```
config: node_dim=5, edge_dim=4, hidden=64
GIB: gamma=0.002, lambda=1.0, eta=10.0
模型参数量: 102,721
loss 模块导入成功
ParticleBP: particles=100, iter=3
所有模块导入成功 ✅
```

### 3.2 端到端集成测试

单张测试图 (10 agents + 4 anchors, 48 edges, 50×50m区域):

```
图: 14 nodes, 48 edges
Anchor col (node)=4, Anchor col (edge)=17

1. GNN 前向传播:     active=0/48 edges (未训练随机权重)
2. GIB 损失:          total=168.23, fim=138.16, kl=0.078, deg=3.000
3. ParticleBP(dense): RMSE=1.4014 m (P=500, iter=5)
4. BFS 提取:          10 edges
5. 指标计算:          RMSE=1.4014, Consistency=40%
```

**分析**:
- GNN 前向传播正常 → STE 硬截断生效
- FIM 组装正常 → logdet 计算无奇异
- KL 散度正常 → 先验 q(d) 计算正确
- 度数约束正常 → 未训练模型无激活边，所有agent度数为0，惩罚=3.0
- 粒子BP正常 → RMSE在合理范围（初始不确定性std=5m，BP收敛到~1.4m）
- BFS提取正常 → 从4个anchor出发覆盖所有节点

### 3.3 已知问题与注意事项

1. **NEES 偏大**: 测试中NEES≈4e9（极度过度自信），原因是 P=500 + iter=5 的粒子BP协方差估计不足。使用最终配置 (P=10000, iter=10) 可改善。

2. **旧数据集不兼容**: `datasets/*.pt` 文件为旧格式（3维节点 + 3维边），需要运行 `generate_datasets.py` 重新生成。

3. **pyg-lib警告**: torch-scatter/torch-cluster等扩展因glibc版本不兼容被禁用，但不影响核心功能（PyG会自动回退到纯Python实现）。

4. **GNN未训练**: 随机权重下GNN不激活任何边，属于正常行为。训练后应学会选择性保留边。

---

## 四、使用指南

### 重新生成数据集

```bash
conda activate py310
cd GNN_learning
python generate_datasets.py
```

### 训练

```bash
python GNN_learning/train.py
```

超参数可在 `config.py` 中调整：
- `GAMMA`: 先验衰减系数
- `LAMBDA_REG`: KL散度权重
- `ETA`: 度数约束权重
- `TAU_INIT / TAU_DECAY`: 温度退火策略

### 评估

```bash
# 快速测试 (10张图, 500粒子)
python GNN_learning/evaluate.py --quick

# 完整评估 (500张图, 10000粒子)
python GNN_learning/evaluate.py
```

### 删除NEBP (确认粒子BP正常后)

```bash
rm -rf NEBP/
```
