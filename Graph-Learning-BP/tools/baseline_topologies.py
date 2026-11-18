"""Portable topology helpers extracted from the completed forest/tree experiments.

center_tree follows the original 100 m square's center anchor (50, 50) and
returns None for disconnected graphs. Constant variance makes variance_msf
an equal-weight forest with edge-ID tie breaking.
"""
from collections import deque
import math
from gib4cl.forest_merge import merge_forest_edges
from gib4cl.inference.spanning_tree import extract_bfs_weights

def select_forests(data):
    edges = data.edge_index.T.tolist()
    old = extract_bfs_weights(data).nonzero().flatten().tolist()
    costs = data.edge_cost.tolist()
    selected = {
        'cost_msf': merge_forest_edges(data.num_nodes, edges, [], costs),
        'variance_msf': merge_forest_edges(data.num_nodes, edges, [], data.edge_attr[:, 1].tolist()),
        'merged_bfs': merge_forest_edges(data.num_nodes, edges, old, costs),
    }

    def partition(indices):
        adj = [set() for _ in range(data.num_nodes)]
        for i in indices:
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        seen = set()
        groups = set()
        for node in range(data.num_nodes):
            if node in seen:
                continue
            part = {node}
            todo = [node]
            while todo:
                neighbors = adj[todo.pop()] - part
                part.update(neighbors)
                todo.extend(neighbors)
            groups.add(frozenset(part))
            seen.update(part)
        return groups

    components = partition(range(len(edges)))
    if not set(old) <= set(selected['merged_bfs']):
        raise ValueError(f'Merged BFS lost a parent edge: {data.sample_id}')
    totals = {name: math.fsum(costs[i] for i in indices) for name, indices in selected.items()}
    for name, indices in selected.items():
        if (len(indices) != data.num_nodes - len(components) or partition(indices) != components
                or totals['cost_msf'] > totals[name] + 1e-9):
            raise ValueError(f'Invalid forest or minimum cost: {data.sample_id}, {name}')
    return selected

def center_tree(data):
    anchors = data.anchor_node_index.tolist()
    center, = [node for node in anchors if data.x[node, :2].tolist() == [50., 50.]]
    neighbors = [[] for _ in range(data.num_nodes)]
    for i, (u, v) in enumerate(data.edge_index.T.tolist()):
        neighbors[u].append((v, i))
        neighbors[v].append((u, i))
    seen = {center}
    queue = deque([center])
    selected = []
    while queue:
        for neighbor, edge in sorted(neighbors[queue.popleft()]):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
            selected.append(edge)
    if len(seen) != data.num_nodes:
        return None
    if len(selected) != data.num_nodes - 1:
        raise ValueError(f'Invalid tree: {data.sample_id}')
    return center, sorted(selected)
