import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_topology_comparison(data, gnn_weights, bfs_weights):
    """
    绘制三张对比图：原始稠密图 vs BFS 启发式生成树 vs GNN 智能稀疏图
    """
    agents_pos = data['true_agents_pos'].numpy()
    anchors_pos = data['anchors_pos'].numpy()
    edge_index = data['edge_index'].numpy()
    is_anchor_edge = data['is_anchor_edge'].numpy()
    
    num_agents = agents_pos.shape[0]
    num_anchors = anchors_pos.shape[0]
    
    # 统一合并坐标字典，方便 NetworkX 画图
    pos_dict = {}
    for i in range(num_agents):
        pos_dict[i] = agents_pos[i]
    for k in range(num_anchors):
        pos_dict[num_agents + k] = anchors_pos[k] # Anchor 的索引往后顺延
        
    def draw_single_graph(ax, title, edge_weights):
        G = nx.Graph()
        
        # 1. 添加所有节点
        for i in range(num_agents):
            G.add_node(i, node_type='agent')
        for k in range(num_anchors):
            G.add_node(num_agents + k, node_type='anchor')
            
        # 2. 根据传入的 edge_weights 掩码，添加保留的边
        for e in range(edge_index.shape[1]):
            if edge_weights[e] > 0.5: # 大于 0.5 认为边存在
                u = edge_index[0, e]
                v = edge_index[1, e]
                if is_anchor_edge[e]:
                    v = num_agents + v # 映射到 Anchor 的绝对索引
                G.add_edge(u, v)

        # 3. 开始绘图配置
        # 分离出 Agent 和 Anchor 的节点列表
        agents = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'agent']
        anchors = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'anchor']
        
        # 画节点：Agent 为蓝色圆形，Anchor 为红色显眼的正方形/星形
        nx.draw_networkx_nodes(G, pos_dict, nodelist=agents, node_color='#4A90E2', 
                               node_shape='o', node_size=150, alpha=0.9, ax=ax, label='Agents')
        nx.draw_networkx_nodes(G, pos_dict, nodelist=anchors, node_color='#E94A4A', 
                               node_shape='s', node_size=300, alpha=1.0, ax=ax, label='Anchors')
        
        # 画边：使用灰色，带有一定透明度
        nx.draw_networkx_edges(G, pos_dict, width=1.5, edge_color='#9B9B9B', alpha=0.6, ax=ax)
        
        # 标出图的标题和统计信息
        active_edges = int(edge_weights.sum().item())
        ax.set_title(f"{title}\nEdges: {active_edges}", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off') # 隐藏坐标轴边框
        
        # 只在第一张图加上图例
        if "Dense" in title:
            ax.legend(loc='upper right', fontsize=10)

    # 创建一个 1 行 3 列的宽画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.1)
    
    # 提取全图权重 (全为 1)
    dense_weights = torch.ones(edge_index.shape[1])
    
    # 绘制三联图
    draw_single_graph(axes[0], "1. Original Dense Graph", dense_weights)
    draw_single_graph(axes[1], "2. BFS Spanning Tree", bfs_weights)
    draw_single_graph(axes[2], "3. GNN Pruned Graph", gnn_weights)
    
    plt.tight_layout()
    plt.show()
