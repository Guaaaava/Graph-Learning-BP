# P0 问题修复报告

> 日期: 2026-06-15

---

## 问题 1：CRLB 指标计算错误

### 问题描述

`evaluate.py` 中报告了负的 CRLB 值（Dense: -108.2, GNN: -79.4），这是物理上不可能的——定位误差下界（单位：米）必须为非负。

**根因**：`evaluate.py:210` 调用 `compute_gib_loss(logits=zeros, gamma=0, lambda_reg=0, eta=0)` 并将返回的 `d['fim']` 直接当作 CRLB 使用。而 `d['fim']` 的实际含义是：

$$\text{loss\_fim} = \log\det(FIM^{-1}) = \sum_i \log(1/\lambda_i)$$

- Dense 图边多，FIM 特征值 $\lambda_i \gg 1$，$\log(1/\lambda_i) < 0$ → 结果为负
- BFS 图边少，FIM 弱，特征值小 → 结果可能为正
- $\log\det$ 和 CRLB 是完全不同的量，数值上根本没有可比性

**真正的 CRLB 定义**：对每个 agent $i$，取其 FIM 逆矩阵的 2×2 对角块：

$$CRLB_i = \sqrt{\text{trace}\left([FIM^{-1}]_{2i:2i+2,\; 2i:2i+2}\right)} \quad \text{（单位：米）}$$

### 解决方案

在 `loss.py` 中新增独立的 `compute_crlb()` 函数：

- 复用 FIM 组装逻辑（方向向量 × 信息系数），构建 $2N_a \times 2N_a$ 的全局 FIM
- 计算 `FIM_inv = torch.linalg.inv(FIM)`（加入 `prior_weight` 对角加载防奇异）
- 遍历每个 agent，提取 2×2 对角块，计算 `sqrt(trace(block))`
- 返回 `(crlb_mean: float, crlb_per_agent: Tensor)`

`evaluate.py` 中将原来的 9 行 `compute_gib_loss(...)` 调用替换为 3 行 `compute_crlb(...)`。

### 效果验证

单张测试图（34 nodes, 252 edges）：

| 拓扑 | 旧"CRLB" | 新 CRLB (m) |
|------|----------|-------------|
| Dense | -108.2 | 0.89 |
| BFS (无edge) | — | 2.00 |
| Random mask | — | 1.20 |

- Dense 拓扑 CRLB = 0.89m，合理（边多信息充分）
- 无边拓扑 CRLB = 2.00m（仅靠 prior_weight=0.5 的 FIM=0.5·I，$CRLB = \sqrt{\text{trace}(2I)} = 2.0$）
- 随机剪枝 CRLB = 1.20m，介于两者之间

**验证通过**：CRLB 恒为非负，单位为米，物理含义明确。

---

## 问题 2：NEES 严重偏高 — 粒子 BP 协方差低估

### 问题描述

评估报告中 NEES 高达 10~36（理想值 ≈ 2），一致性仅 21~34%（理想值 ≈ 95%）。

NEES（Normalized Estimation Error Squared）定义为：

$$\epsilon_i = (\hat{x}_i - x_i^{true})^T \Sigma_i^{-1} (\hat{x}_i - x_i^{true})$$

对一致的 2D 定位估计器，$\mathbb{E}[\epsilon] = 2$。实测 NEES=19.4 意味着 $\Sigma_i$ 约比真实误差协方差小 **10 倍**，说明粒子 BP 的协方差估计极度过度自信。

**根因分析**：

1. **粒子贫化**：旧版每轮无条件执行多项式重采样，即使粒子分布还很均匀。这导致粒子快速坍缩到少数高权重位置
2. **Roughening 噪声过弱**：`reg_scale=0.05` → roughening std = 0.05 × 5m = 0.25m。相比 RMSE≈1.5m，这太弱了，无法在重采样后恢复粒子多样性
3. **恶性循环**：重采样浓缩粒子 → 协方差变小 → 下一轮 BP 消息以"过度自信"的 belief 传递 → 粒子进一步坍缩 → 协方差持续缩小

### 解决方案

修改 `BP/particle_bp.py`，两处改动：

**（1）提高 roughening 噪声**：`reg_scale` 默认值 `0.05 → 0.3`

- 旧：噪声 std = 0.25m（仅为 RMSE 的 17%）
- 新：噪声 std = 1.5m（接近 RMSE 量级）
- roughening 的目的是补偿重采样引起的多样性损失，其强度应匹配定位精度的量级

**（2）ESS 条件重采样**：

```python
ess = 1.0 / (belief.square().sum(dim=1) + 1e-12)    # N_eff = 1/Σw²
ess_thresh = P / 2

for agent_i in range(N):
    if ess[agent_i] < ess_thresh:
        # 粒子退化严重 → 重采样 + roughening
        ...
    # 否则保持当前粒子分布
```

- 有效样本量 $N_{eff} = 1/\sum_i w_i^2$：当 belief 集中在少数粒子时 $N_{eff}$ 很小
- 仅当 $N_{eff} < P/2$ 时才触发重采样，避免不必要的粒子多样性损失
- roughening 仅在重采样时施加（重采样后才需要恢复多样性）

### 效果验证

单张测试图（P=500, iter=5, Dense 拓扑）：

| 指标 | reg_scale=0.05 (旧) | reg_scale=0.3 (新) | 理想值 |
|------|---------------------|---------------------|--------|
| RMSE | 1.654m | **1.288m** | — |
| NEES | 20.31 | **2.65** | ≈ 2 |
| 一致性 | 34% | **97%** | ≈ 95% |

效果显著：

- **NEES 从 20.31 → 2.65**：协方差估计从低估 10× 恢复到接近真实散布
- **一致性从 34% → 97%**：估计的 95% 置信椭圆现在真正覆盖了约 95% 的实际误差
- **RMSE 也略有改善**（1.65 → 1.29m）：ESS 条件重采样避免了过度重采样带来的估计噪声

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `GNN_learning/loss.py` | 新增 `compute_crlb()` 函数 (~90行)，更新模块 docstring |
| `GNN_learning/evaluate.py` | `from loss import compute_crlb`、替换 CRLB 计算逻辑、修正报告格式 |
| `BP/particle_bp.py` | `reg_scale` 默认值 0.05→0.3、重采样改为 ESS 条件触发 |
