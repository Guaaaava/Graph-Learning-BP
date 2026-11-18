"""Join the trees of an existing forest using candidate measurement edges."""

from math import isfinite


def merge_forest_edges(num_nodes, edges, forest_ids, edge_costs):
    """Keep all forest edges and add the cheapest eligible bridges, breaking ties by edge ID.

    Edge IDs index the original candidate graph. Each original component,
    including any isolated vertex, becomes one component of the output forest.
    """
    if len(edges) != len(edge_costs) or any(not isfinite(c) or c < 0 for c in edge_costs):
        raise ValueError("Expected one finite nonnegative cost per candidate edge")
    parent = list(range(num_nodes))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(edge_id):
        u, v = edges[edge_id]
        left, right = root(u), root(v)
        if left == right:
            return False
        parent[right] = left
        return True

    selected = set()
    for edge_id in forest_ids:
        if not 0 <= edge_id < len(edges) or not join(edge_id):
            raise ValueError("Initial edges must form a forest of candidate edges")
        selected.add(edge_id)
    for edge_id in sorted(range(len(edges)), key=lambda i: (edge_costs[i], i)):
        if edge_id not in selected and join(edge_id):
            selected.add(edge_id)
    return sorted(selected)
