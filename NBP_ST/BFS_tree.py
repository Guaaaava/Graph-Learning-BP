import torch
import networkx as nx

def extract_bfs_tree_edges(num_agents, num_anchors, edge_index, is_anchor_edge):
    """
    [对照组基准] 提取以 Anchor 为根的 BFS 生成树。

    策略:
      1. 从所有 Anchor 同时出发做多源 BFS，覆盖能到达锚点的节点
      2. 对孤立组件: 在稠密图中找最短路径连接到已访问集合，
         确保每个 Agent 都有到 Anchor 的路径
    """
    N_total = num_agents + num_anchors
    G = nx.Graph()
    G.add_nodes_from(range(N_total))

    # --- 1. 组装全局图 (锚点局部索引 → 全局绝对索引) ---
    for e in range(edge_index.shape[1]):
        u = edge_index[0, e].item()
        v = edge_index[1, e].item()
        v_absolute = num_agents + v if is_anchor_edge[e] else v
        G.add_edge(u, v_absolute, edge_idx=e)

    # --- 2. 从 Anchors 出发的 BFS ---
    anchor_nodes = set(range(num_agents, N_total))  # anchors are at the end
    bfs_edge_set = set()
    visited = set(anchor_nodes)
    queue = list(anchor_nodes)

    while queue:
        cur = queue.pop(0)
        for nxt in G.neighbors(cur):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                # Record the edge (undirected, store as sorted tuple)
                edge_data = G.get_edge_data(cur, nxt)
                bfs_edge_set.add((min(cur, nxt), max(cur, nxt)))

    # --- 3. 连接孤立组件: 最短路径到已访问集合 ---
    remaining = set(range(N_total)) - visited
    while remaining:
        # BFS from visited set to find closest remaining node
        parent = {}
        local_visited = set(visited)
        q = list(visited)
        found = None
        for node in q:
            parent[node] = None

        while q and found is None:
            cur = q.pop(0)
            for nxt in G.neighbors(cur):
                if nxt not in local_visited:
                    local_visited.add(nxt)
                    parent[nxt] = cur
                    q.append(nxt)
                    if nxt in remaining:
                        found = nxt
                        break

        if found is None:
            break

        # Backtrack: add edges along found → ... → visited path
        node = found
        while parent[node] is not None:
            p = parent[node]
            bfs_edge_set.add((min(node, p), max(node, p)))
            visited.add(node)
            node = p
        visited.add(found)

        remaining = set(range(N_total)) - visited

    # --- 4. 转换为 0-1 掩码 ---
    bfs_weights = torch.zeros(edge_index.shape[1])
    for e in range(edge_index.shape[1]):
        u = edge_index[0, e].item()
        v = edge_index[1, e].item()
        v_absolute = num_agents + v if is_anchor_edge[e] else v
        edge_key = (min(u, v_absolute), max(u, v_absolute))
        if edge_key in bfs_edge_set:
            bfs_weights[e] = 1.0

    return bfs_weights