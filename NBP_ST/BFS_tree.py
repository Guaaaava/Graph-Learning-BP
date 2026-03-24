import torch
import networkx as nx

def extract_bfs_tree_edges(num_agents, num_anchors, edge_index):
    """
    [对照组基准] 提取广度优先搜索 (BFS) 生成树的边。
    模拟传统的 NBP-ST 方法：为了消灭环路，粗暴地提取一棵树。
    """
    N_total = num_agents + num_anchors
    G = nx.Graph()
    G.add_nodes_from(range(N_total))
    
    # 将边加入 networkx 图中
    edges = edge_index.T.numpy()
    for u, v in edges:
        G.add_edge(u, v)
        
    # 找一个度最大的节点作为 BFS 的根节点
    degrees = dict(G.degree())
    root = max(degrees, key=degrees.get)
    
    # 提取 BFS 树的边
    bfs_edges = list(nx.bfs_edges(G, source=root))
    
    # 转换回我们的 edge_weights 掩码 (0-1 向量)
    bfs_weights = torch.zeros(edge_index.shape[1])
    for e in range(edge_index.shape[1]):
        u = edge_index[0, e].item()
        v = edge_index[1, e].item()
        # 判断这条边是否在 BFS 树中（无向图，需判断双向）
        if (u, v) in bfs_edges or (v, u) in bfs_edges:
            bfs_weights[e] = 1.0
            
    return bfs_weights