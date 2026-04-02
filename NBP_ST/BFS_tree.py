import torch
import networkx as nx

def extract_bfs_tree_edges(num_agents, num_anchors, edge_index, is_anchor_edge):
    """
    [对照组基准] 提取广度优先搜索 (BFS) 生成树（森林）的边。
    模拟传统的 NBP-ST 方法：为了消灭环路，粗暴地提取一棵树。
    """
    N_total = num_agents + num_anchors
    G = nx.Graph()
    G.add_nodes_from(range(N_total))
    
    # --- 1. 组装正确的全局图 (修复索引陷阱) ---
    for e in range(edge_index.shape[1]):
        u = edge_index[0, e].item()
        v = edge_index[1, e].item()
        
        # 关键：如果是锚点，把相对索引 (如 0~3) 映射为全局绝对索引 (如 25~28)
        if is_anchor_edge[e]:
            v_absolute = num_agents + v
        else:
            v_absolute = v
            
        G.add_edge(u, v_absolute)
        
    # --- 2. 提取 BFS 森林 (修复多孤岛陷阱) ---
    bfs_edges = []
    # 遍历图中的每一个连通分量，防止某些边缘节点被直接遗弃
    for component in nx.connected_components(G):
        sub_G = G.subgraph(component)
        if len(sub_G.edges()) == 0:
            continue # 忽略没有边的单节点孤岛
            
        degrees = dict(sub_G.degree())
        root = max(degrees, key=degrees.get)
        bfs_edges.extend(list(nx.bfs_edges(sub_G, source=root)))
        
    # --- 3. 转换回 0-1 掩码 ---
    bfs_weights = torch.zeros(edge_index.shape[1])
    for e in range(edge_index.shape[1]):
        u = edge_index[0, e].item()
        v = edge_index[1, e].item()
        
        if is_anchor_edge[e]:
            v_absolute = num_agents + v
        else:
            v_absolute = v
            
        # 判断这条绝对索引的边是否在 BFS 森林的边集中
        if (u, v_absolute) in bfs_edges or (v_absolute, u) in bfs_edges:
            bfs_weights[e] = 1.0
            
    return bfs_weights