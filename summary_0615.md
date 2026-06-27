# 补充报告: 6/10 — 6/15

> 日期: 2026-06-15  
> 范围: 从 `summary.md` 记录的重构完成后至今的增量工作

---

## 一、训练完成 (6/11)

### 超参数调整

从 `summary.md` 中记录的初始参数到最终训练配置有显著调整：

| 参数 | 初始值 (summary.md) | 训练值 (config.py) | 变化原因 |
|------|---------------------|---------------------|----------|
| `LAMBDA_REG` | 1.0 | **20.0** | KL 项权重过低时稀疏性不足 |
| `ETA` | 10.0 | **50.0** | 度数约束过弱时 agent 边数不足 3 |
| `EPOCHS` | ~1000 | **200** | 批量训练收敛速度远快于单图 |
| `LR` | 0.005 | **0.001** + AdamW | 降低学习率配合余弦退火 |
| `FIM_PRIOR` | 未提及 | **0.5** | 防止 FIM 奇异 |
| `SPARSITY_WEIGHT` | 无 | **150.0** | 新增直接稀疏惩罚，作为 KL 项的补充 |
| 优化器 | Adam | **AdamW** + CosineAnnealing | weight decay + 平滑学习率下降 |
| 梯度裁剪 | 2.0 | **5.0** | Gumbel 噪声下需要更宽松的裁剪 |

### 训练输出

```
目录: GNN_learning/models/
  best_model.pth   (427 KB, Jun 11 01:37)
  final_model.pth  (427 KB, Jun 11 01:38)
```

---

## 二、loss.py 损失函数完善

### 新增直接稀疏惩罚

`compute_gib_loss()` 新增 `sparsity_weight` 参数：

```python
if sparsity_weight > 0:
    loss_sparsity = p.mean()  # 均匀压缩: 鼓励所有 p→0
```

总损失公式变为：

$$\mathcal{L} = \underbrace{\log\det(J^{-1})}_{\text{FIM}} + \lambda \cdot KL + \eta_{\text{sparse}} \cdot \bar{p} + \eta \cdot \Sigma \max(0,3-D_i)$$

`SPARSITY_WEIGHT=150` 提供强基线压缩，KL 项负责距离相关的精细化调制。

---

## 三、粒子BP 完善 (6/15)

### 新增 Roughening 噪声

```python
# __init__ 新增参数
reg_scale=0.05  # 相对初始 std 的 roughening 噪声比例

# 每次重采样后
if self.reg_scale > 0:
    noise_std = self.reg_scale * (self.init_cov ** 0.5)
    particles = particles + torch.randn_like(particles) * noise_std
```

**目的**: 防止多项式重采样后粒子多样性坍缩，维持协方差估计的有效性。标准粒子滤波做法。

### 边权重掩码细化

```python
# 分别对 anchor 边和 agent 边应用掩码
anc_active = edge_weights[anc_mask] > 0.5
agt_active = edge_weights[agt_mask] > 0.5
```

支持 GNN 输出的二值掩码在 BP 消息传递中选择性跳过被剪枝的边。

---

## 四、评估结果 (6/15)

### 测试配置

- 图数: **100 张** (test set 子集)
- 粒子数: P=2000, 迭代数: iter=10
- BP 参数: sigma_meas=0.5, init_cov=25.0
- 三种拓扑: Dense (原图) / BFS (生成树) / GNN (剪枝)

### 定量结果

| 指标 | Dense (全图) | BFS (生成树) | GNN (剪枝) |
|------|-------------|-------------|-----------|
| 边数 | 391 | 36 | 147 |
| RMSE (m) | 1.49 | 5.76 | 2.44 |
| CRLB | -108.2 | 34.0 | -79.4 |
| 中断率 (>3m) | 3.8% | 70.4% | 18.0% |
| 一致性 | 33.2% | 62.5% | 21.5% |
| NEES | 19.4 | 10.1 | 35.9 |

**边减少 62%**, RMSE 比 Dense=×1.64

### 分析

1. **Dense 表现最好** (RMSE=1.49m) — 冗余边提供最大 Fisher 信息，但 391 条边通信开销大
2. **BFS 最差** (RMSE=5.76m) — 36 条边的树结构缺乏环闭合约束，长基线误差累积
3. **GNN 折中** (RMSE=2.44m) — 62% 边减少的代价是 1.64× RMSE 退化
4. **NEES 偏高**: 三类拓扑的 NEES 均远大于 2 (理想值)，粒子数 P=2000 可能不足以准确估计协方差
5. **一致性低**: 特别是 GNN (21.5%)，表明协方差估计显著低估了实际误差

---

## 五、代码文件状态

### 未提交的新文件 (untracked)

| 文件 | 状态 |
|------|------|
| `BP/` | BP 模块，含 `__init__.py` + `particle_bp.py` |
| `GNN_learning/loss.py` | GIB 损失函数 |
| `GNN_learning/evaluate.py` | 评估脚本 |
| `GNN_learning/models/` | 训练好的模型权重 |
| `GNN_learning/evaluate_report.txt` | 评估结果报告 |
| `summary.md` | 重构总结 |
| `NEBP/` | 旧 DGL 代码，待删除 |

### 已修改未提交

| 文件 | 变更 |
|------|------|
| `config.py` | 新增所有超参数 + 特征维度 |
| `model.py` | 5维节点 + 4维边 + Gumbel-Sigmoid STE |
| `dataset.py` | 适配新特征维度 |
| `train.py` | 批量训练 + GIB 损失 + 退火 |
| `generate_network.py` | 伪距残差 + init_pos_cov |
| `generate_datasets.py` | 传入新参数 |

### 已删除

8 个旧文件: `build_global_FIM.py`, `crlb_loss.py`, `edge_predictor_GNN.py`, `train_batch.py`, `tune_lambda_batch.py`, `evaluate.py`(root), `evaluate_batch.py`, `gnn_generalized_model.pth`

---

## 六、待完成事项

1. **完整评估**: 当前只测了 100 张图 P=2000，最终需跑 500 张图 P=10000
2. **超参数搜索**: λ/KL、η_degree、sparsity_weight 之间的平衡可进一步调优
3. **NEES 问题**: 粒子 BP 的协方差估计偏低（过度自信），可能需要增大 `reg_scale` 或粒子数
4. **删除 NEBP/**: 确认粒子 BP 稳定后可移除旧 DGL 代码
5. **提交代码**: 所有改动仍在工作树中未提交

---

## 七、运行命令备忘

```bash
# 评估 (快速模式)
cd GNN_learning && python evaluate.py --quick

# 评估 (完整模式)
cd GNN_learning && python evaluate.py

# 重新训练
cd GNN_learning && python train.py
```
