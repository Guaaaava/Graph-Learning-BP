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
BP/                          # 粒子化非参数BP (PyG)
  particle_bp.py             #   ParticleBP 类

GNN_learning/                # GNN 图拓扑优化
  config.py                  #   全局配置
  model.py                   #   EdgePredictorGNN (5维节点 + 4维边)
  loss.py                    #   GIB损失 (logdet + KL + degree)
  dataset.py                 #   PyG Data 适配器
  generate_network.py        #   单图生成
  generate_datasets.py       #   批量数据集生成
  train.py                   #   训练脚本
  evaluate.py                #   评估脚本 (粒子BP对比)
  visualize_topology.py      #   拓扑可视化

NBP_ST/                      # BFS 基准
  BFS_tree.py

NEBP/                        # 参考代码 (DGL版粒子BP，后续删除)
```

## 运行环境

运行代码前请先：
```bash
conda activate py310
```

## 特征维度

- 节点特征: `[pos_x, pos_y, σ²_x, σ²_y, is_anchor]` (5维)
- 边特征: `[measurement, variance, pseudo_range_residual, is_anchor_edge]` (4维)

## 工作流

1. `generate_datasets.py` → 生成 train/val/test.pt
2. `train.py` → 训练 GNN (GIB 损失)
3. `evaluate.py` → 粒子BP评估三种拓扑 (Dense/BFS/GNN)
