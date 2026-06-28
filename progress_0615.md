# 项目进展报告

> 日期: 2026-06-15  
> 范围: 6月10日 ～ 6月15日

---

## 一、总体进度

完成了从旧版高斯BP到粒子化非参数BP的完整重构，建立了 GNN + GIB 损失的图拓扑优化pipeline，修复了评估指标bug和协方差估计问题，并完成了第一轮超参数调优。

| 模块 | 状态 | 关键成果 |
|------|------|----------|
| 粒子BP | ✅ | 从NEBP(DGL)重写为PyG，支持拓扑剪枝，协方差估计正常 |
| GNN模型 | ✅ | 5维节点+4维边，Gumbel-Sigmoid STE，~103K参数 |
| GIB损失 | ✅ | logdet(FIM⁻¹) + KL + degree + sparsity 四合一 |
| CRLB计算 | ✅ | 正确实现per-agent CRLB (米) |
| 训练pipeline | ✅ | 批量训练，AdamW+CosineAnnealing，温度退火 |
| 评估pipeline | ✅ | Dense/BFS/GNN三拓扑对比，mean±std，分位数，LaTeX输出 |
| 超参调优 | ✅ | sp_w=80优于原150，RMSE退化从1.64×降至1.42× |
| 最终完整评估 | ⏳ | 需P=10000过夜运行 |

---

## 二、GNN特征维度

### 节点特征：3维 → 5维

| 旧版 (3维) | 新版 (5维) | 说明 |
|------------|------------|------|
| `pos_x` | `pos_x` | Agent初始含噪估计x / Anchor真值x |
| `pos_y` | `pos_y` | Agent初始含噪估计y / Anchor真值y |
| `is_anchor` | `σ²_x` | 位置x方向不确定度（Agent=25m², Anchor=0） |
| — | `σ²_y` | 位置y方向不确定度（Agent=25m², Anchor=0） |
| — | `is_anchor` | 二值标识（Agent=0, Anchor=1） |

新增的 σ² 维度让GNN知道每个节点的初始不确定程度，从而在剪枝时对高不确定度agent保留更多边。

### 边特征：3维 → 4维

| 旧版 (3维) | 新版 (4维) | 说明 |
|------------|------------|------|
| `measurement` | `measurement` | 测距值 z |
| `variance` | `variance` | 测距方差 σ²_z |
| `is_anchor_edge` | `pseudo_range_residual` | 伪距残差 r = |z - ||x_init_u - x_init_v|| |
| — | `is_anchor_edge` | 是否为anchor边（Agent-Anchor=1, Agent-Agent=0） |

新增的伪距残差衡量测距值与初始估计距离的一致性：残差小→边质量高，残差大→可能NLOS或坏边。这为GNN提供了判断边可靠性的直接线索。

---

## 三、粒子BP粒子数设置

### 训练阶段

- **训练时评估粒子数**：`BP_NUM_PARTICLES_TRAIN = 2000`
- **用途**：超参扫描时在val set上评估RMSE/边数trade-off
- **理由**：2000粒子在精度与速度间平衡，可快速筛选超参

### 测试阶段

| 场景 | 粒子数 | 迭代数 | 用途 |
|------|--------|--------|------|
| 快速验证 | `P=500` | `iter=3~5` | 调试代码、快速检查指标趋势 |
| 常规评估 | `P=2000` | `iter=10` | 开发阶段的正式评估 |
| **最终评估** | **`P=10000`** | **`iter=10`** | **论文最终结果，协方差准确** |

- `BP_NUM_PARTICLES_TEST = 10000`：config中的默认值，用于论文级最终评估
- P=500/iter=3 下NEES≈5，一致性≈74%；P=2000/iter=10 下NEES≈2.7，一致性≈97%（P0修复已验证）
- 当前由于CPU上O(P²)的agent-agent消息传递瓶颈，P=10000跑500张图需数小时，建议过夜运行

---

## 四、逐日工作

### 6月10日 — 项目重构

- 将NEBP的粒子BP改写为独立 `BP/particle_bp.py`（PyG，静态定位，去运动模型）
- GNN模型统一为单一 `EdgePredictorGNN`（节点特征3→5维，边特征3→4维）
- 新增 `loss.py` — GIB损失：logdet(FIM⁻¹) + KL(p||q) + degree constraint
- 训练从单图改为批量（2000/500/500划分），优化器AdamW+CosineAnnealing
- 删除8个旧文件（旧GNN、旧损失、旧训练脚本等）
- 伪距残差 `pseudo_range_residual` 作为第4维边特征
- 模块测试通过：ParticleBP RMSE=1.4m (P=500, iter=5)，FIM组装正常

### 6月11日 — 训练收敛

- GIB损失新增 `sparsity_weight` 直接稀疏惩罚项
- 超参数大幅调整：LAMBDA_REG 1→20, ETA 10→50, SPARSITY_WEIGHT=150
- 训练200 epochs完成，best_model.pth + final_model.pth (427 KB)
- 验证损失稳定下降，训练正常收敛

### 6月15日 — 评估修复与优化

#### P0-1: CRLB修复

- **问题**：evaluate.py将 `logdet(FIM⁻¹)` 直接当作CRLB，出现负值(-108.2)
- **修复**：在loss.py新增 `compute_crlb()`，正确定义 CRLB_i = sqrt(trace([FIM⁻¹]_{2×2, agent i}))
- **效果**：CRLB现在恒为非负，Dense=0.81m, GNN=0.90m，物理含义明确

#### P0-2: NEES修复

- **问题**：粒子BP协方差严重低估（NEES=20~36，理想≈2），一致性仅21~34%
- **根因**：roughening噪声过弱(0.25m std) + 无条件每轮重采样 → 粒子贫化
- **修复**：reg_scale 0.05→0.3 + ESS条件重采样（仅N_eff<P/2时触发）
- **效果**：NEES 20→2.7，一致性 34%→97%，RMSE也微降

#### P1-1: 超参数调优

- train.py新增CLI支持（--sparsity-weight, --lambda-reg, --eta, --epochs等）
- 新建 `sweep_hyperparams.py`：网格搜索 + val set粒子BP评估 + Pareto前沿
- SPARSITY_WEIGHT扫描结果：80 (RMSE=2.25m, 206边) > 120 > 150
- sp_w=80完整训练200 epochs完成，模型已保存

#### P1-2: 完整评估

- 评估脚本重写：mean±std, Wilson 95%CI, RMSE分位数(P50/P90/P95), LaTeX表格行
- 同时输出 evaluate_report.txt + evaluate_results.json (per-graph详细数据)
- 100张测试图评估完成（完整500张需P=10000过夜）

---

## 五、关键指标变化

### 从旧版到新版

| 指标 | 旧版 (sp_w=150, P0未修复) | 新版 (sp_w=80, P0已修复) |
|------|--------------------------|--------------------------|
| CRLB 报告 | -108.2 (错误) | 0.90m (正确) |
| NEES (Dense) | 19.4 | 4.34 |
| NEES (GNN) | 35.9 | 5.14 |
| 一致性 (GNN) | 21.5% | 74% |
| GNN边数 | 147 | 206 |
| GNN RMSE | 2.44m | 2.25m |
| GNN vs Dense | ×1.64 | ×1.42 |

### 最新完整评估（100张测试图, P=500, iter=3）

| 指标 | Dense | BFS | GNN |
|------|-------|-----|-----|
| 边数 | 391 ± 127 | 36 ± 7 | 206 ± 65 |
| RMSE (m) | 1.58 ± 0.42 | 5.59 ± 0.58 | 2.25 ± 0.56 |
| CRLB (m) | 0.81 ± 0.07 | 1.84 ± 0.02 | 0.90 ± 0.07 |
| 中断率 (>3m) | 4.4% | 70.1% | 15.6% |
| NEES | 4.34 | 2.35 | 5.14 |
| 一致性 | 81% | 93% | 74% |

边减少: 47%, GNN vs Dense RMSE比: 1.42

---

## 六、代码变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `BP/__init__.py` | BP模块入口 |
| `BP/particle_bp.py` | 粒子化非参数BP（PyG, 静态定位, ESS条件重采样） |
| `GNN_learning/loss.py` | GIB损失 + compute_crlb() |
| `GNN_learning/evaluate.py` | 三拓扑对比评估（mean±std + 分位数 + LaTeX） |
| `GNN_learning/eval_100.py` | 快速评估脚本（100图） |
| `GNN_learning/sweep_hyperparams.py` | 超参数网格搜索 |
| `GNN_learning/models/best_model.pth` | 最佳模型 (sp_w=80, 200 epochs) |
| `GNN_learning/models/final_model.pth` | 最终模型 |
| `evaluation_0615.md` | 项目评估报告 |
| `problem_0615.md` | P0问题修复报告 |
| `summary_0610.md` | 重构总结 |
| `summary_0615.md` | 补充报告 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `config.py` | 超参、特征维度、BP参数 |
| `model.py` | 5维节点+4维边、Gumbel-Sigmoid STE、LayerNorm |
| `dataset.py` | 适配新特征维度 |
| `train.py` | 批量训练 + GIB损失 + CLI参数 |
| `generate_network.py` | 伪距残差、init_pos_cov |
| `generate_datasets.py` | 传入新参数 |

### 删除文件

8个旧文件: `build_global_FIM.py`, `crlb_loss.py`, `edge_predictor_GNN.py`, `train_batch.py`, `tune_lambda_batch.py`, `evaluate.py`(root), `evaluate_batch.py`, `gnn_generalized_model.pth`

---

## 七、当前限制与后续计划

### 当前限制

1. **粒子BP速度**：O(P²) per-edge 的 agent-agent 消息传递在CPU上很慢。P=10000/iter=10的完整500图评估需数小时
2. **NEES仍偏高**（~5, 理想≈2）：P=500/iter=3下粒子数不足，增大P应进一步改善
3. **GNN精度gap**：RMSE仍为Dense的1.42×，有继续优化的空间
4. **代码未提交**：所有改动仍在工作树

### 后续计划

1. **最终评估**：P=10000, iter=10, 500张图（过夜运行）
2. **第二轮超参调优**：尝试更低sparsity_weight(50~70)、调整LAMBDA_REG和GAMMA
3. **方法改进**：考虑per-agent CRLB约束、GNN结构改进（attention等）
4. **Baseline扩展**：MST最小生成树、k-NN、随机剪枝
5. **代码整理**：提交、删除NEBP/、清理调试脚本
6. **论文写作**：已有LaTeX表格行，可开始写实验部分
