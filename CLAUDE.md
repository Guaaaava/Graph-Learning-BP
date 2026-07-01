# 项目指令

## 沟通
- 中文沟通
- 回复简洁直接，不要过度解释
- 发现更优路径时主动建议，不只按指令做

## 工作习惯
- 复杂改动先讨论方案，简单调整直接做
- 修改超过 3 个文件时先拆成小任务

## 边界
- 不要自动提交或推送代码

## 项目结构


```
BP/particle_bp.py                     # 粒子BP
GNN_learning/config.py                # 配置
GNN_learning/model.py                 # GNN模型
GNN_learning/loss.py                  # GIB损失
GNN_learning/dataset.py               # 数据加载
GNN_learning/generate_network.py      # 图生成（3种场景）
GNN_learning/generate_datasets.py     # 批量数据集
GNN_learning/train.py                 # 训练
GNN_learning/evaluate.py              # 评估（6种拓扑）
GNN_learning/visualize_topology.py    # 可视化
NBP_ST/BFS_tree.py                    # BFS基准
NEBP/                                 # 参考代码
CLAUDE.md                             # 项目指令
```

### 

## 运行环境

运行代码前请先：
```bash
conda activate py310
```

## 特征维度

- 节点特征: `[pos_x, pos_y, σ²_x, σ²_y, is_anchor]` (5维)
- 边特征: `[measurement, variance, pseudo_range_residual, is_anchor_edge]` (4维)

## 工作流

```
generate_network.py (场景参数)
    ↓ raw dict {true_agents_pos, init_agents_pos, anchors_pos, edge_index,
               measurements, edge_variances, pseudo_range_residual, is_anchor_edge}
    ↓
dataset.py (组装 PyG Data)
    ↓ data.x [N,5] + data.edge_attr [E,4] + data.y [N_a,2]
    ↓
model.py (GNN前向)
    ↓ edge_weights ∈ {0,1} (Gumbel-Sigmoid STE)
    ↓
particle_bp.py (粒子BP定位，P=10000)
    ↓ est_pos [N_a,2] + est_cov [N_a,2,2]
    ↓
evaluate.py (6种拓扑对比)
    ↓ RMSE / CRLB / NEES / Outage / Chi2 consistency
```