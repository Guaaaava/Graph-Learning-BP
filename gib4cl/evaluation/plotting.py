from pathlib import Path

import torch
import networkx as nx
import matplotlib.pyplot as plt

from gib4cl.data.schema import GRAPH_VERSION


def _binary_mask(weights, num_edges, name):
    mask = torch.as_tensor(weights).detach().cpu()
    if mask.ndim != 1 or mask.numel() != num_edges:
        raise ValueError(f'{name} must have shape ({num_edges},)')
    if mask.is_complex() or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f'{name} must contain only finite binary values (0 or 1)')
    return mask.to(torch.bool).numpy()


def plot_topology_comparison(data, gnn_weights, bfs_weights, *, show=True, save_path=None):
    """
    绘制三张对比图：原始稠密图 vs BFS 启发式生成树 vs GNN 智能稀疏图

    两组权重必须是按物理边排列的二值掩码，不能传入保留概率。
    返回 (fig, axes)；show=False 可关闭弹窗，save_path 可指定图片保存路径。
    调用方使用完返回的图对象后，可通过 plt.close(fig) 释放它。
    """
    if data.get('schema_version') != GRAPH_VERSION:
        raise ValueError(f'Plotting requires {GRAPH_VERSION} with global indices')
    inputs = data['inputs']
    agent_mask = ~inputs['is_anchor']
    agents_pos = data['targets']['agent_positions'].cpu().numpy()
    anchors_pos = inputs['initial_positions'][~agent_mask].cpu().numpy()
    edge_index = inputs['edge_index'].cpu().numpy()
    num_edges = edge_index.shape[1]
    gnn_mask = _binary_mask(gnn_weights, num_edges, 'gnn_weights')
    bfs_mask = _binary_mask(bfs_weights, num_edges, 'bfs_weights')
    
    num_agents = agents_pos.shape[0]
    num_anchors = anchors_pos.shape[0]
    
    # 统一合并坐标字典，方便 NetworkX 画图
    pos_dict = {}
    for i in range(num_agents):
        pos_dict[i] = agents_pos[i]
    for k in range(num_anchors):
        pos_dict[num_agents + k] = anchors_pos[k] # Anchor 的索引往后顺延
        
    def draw_single_graph(ax, title, edge_mask):
        G = nx.Graph()
        
        # 1. 添加所有节点
        for i in range(num_agents):
            G.add_node(i, node_type='agent')
        for k in range(num_anchors):
            G.add_node(num_agents + k, node_type='anchor')
            
        # 2. 根据传入的 edge_weights 掩码，添加保留的边
        G.add_edges_from(edge_index[:, edge_mask].T.tolist())

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
        active_edges = G.number_of_edges()
        ax.set_title(f"{title}\nEdges: {active_edges}", fontsize=14, fontweight='bold', pad=10)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off') # 隐藏坐标轴边框
        
        # 只在第一张图加上图例
        if "Dense" in title:
            ax.legend(loc='upper right', fontsize=10)

    # 创建一个 1 行 3 列的宽画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.1)
    
    # 提取全图权重 (全为 1)
    dense_mask = torch.ones(num_edges, dtype=torch.bool).numpy()
    
    # 绘制三联图
    draw_single_graph(axes[0], "1. Original Dense Graph", dense_mask)
    draw_single_graph(axes[1], "2. BFS Forest", bfs_mask)
    draw_single_graph(axes[2], "3. GNN Pruned Graph", gnn_mask)
    
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes
