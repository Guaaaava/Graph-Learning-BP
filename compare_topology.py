# %%

from GNN_learning.train import train_gnn_sparsifier
from NBP_ST.BFS_tree import extract_bfs_tree_edges
from GNN_learning.visualize_topology import plot_topology_comparison

# 1. 启动真实训练
print("==================================================")
print("阶段一：GNN 拓扑修剪训练")
print("==================================================")
trained_model, sparse_topology, data, _, _, _ = train_gnn_sparsifier(epochs=300, lr=0.01, lambda_reg=0.2)

# 从数据中动态获取节点数量
num_agents = data['true_agents_pos'].shape[0]
num_anchors = data['anchors_pos'].shape[0]

# 2. 生成传统的 BFS 启发式基准树
print("\n==================================================")
print("阶段二：提取 BFS 基准生成树")
print("==================================================")
bfs_weights = extract_bfs_tree_edges(num_agents, num_anchors, data['edge_index'])

# 3. 画图
print("\n==================================================")
print("阶段三：绘制拓扑对比图")
print("==================================================")
plot_topology_comparison(data, sparse_topology, bfs_weights)
# %%
