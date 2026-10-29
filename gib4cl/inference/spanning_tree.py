"""Deterministic multi-source BFS forest on the candidate physical graph."""
from dataclasses import dataclass
from collections import deque
import torch


@dataclass
class ForestResult:
    edge_weights: torch.Tensor
    forest_root: torch.Tensor
    anchor_reachable: torch.Tensor


def bfs_forest(num_nodes, edge_index, anchor_nodes):
    """Anchors seed simultaneous BFS; remaining components use lowest node ID.

    Neighbor and root order use ascending global node IDs. Every selected edge
    belongs to the candidate graph. Disconnected components are never connected
    by inventing measurements. The result may contain multiple rooted trees.
    """
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError('num_nodes must be a positive integer')
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('Expected long edge_index=(2,E)')
    if anchor_nodes.dtype != torch.long or anchor_nodes.ndim != 1 or anchor_nodes.device != edge_index.device:
        raise ValueError('Expected long anchor node indices on graph device')
    edges, anchors = edge_index.cpu().t().tolist(), anchor_nodes.cpu().tolist()
    if len(set(anchors)) != len(anchors) or any(node < 0 or node >= num_nodes for node in anchors):
        raise ValueError('Invalid anchor nodes')
    adjacency = [[] for _ in range(num_nodes)]
    identities = set()
    for edge, (u,v) in enumerate(edges):
        identity = (min(u,v),max(u,v))
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes or u == v or identity in identities:
            raise ValueError('Require unique valid physical edges without self-loops')
        if u in anchors and v in anchors:
            raise ValueError('Anchor-anchor measurement edges are not supported')
        identities.add(identity)
        adjacency[u].append((v,edge))
        adjacency[v].append((u,edge))
    for neighbors in adjacency:
        neighbors.sort()
    roots = [-1]*num_nodes
    selected = torch.zeros(len(edges),device=edge_index.device)
    def expand(queue):
        while queue:
            node = queue.popleft()
            for neighbor,edge in adjacency[node]:
                if roots[neighbor] == -1:
                    roots[neighbor] = roots[node]
                    selected[edge] = 1
                    queue.append(neighbor)
    for anchor in sorted(anchors):
        roots[anchor] = anchor
    expand(deque(sorted(anchors)))
    for node in range(num_nodes):
        if roots[node] == -1:
            roots[node] = node
            expand(deque([node]))
    root_tensor = torch.tensor(roots,dtype=torch.long,device=edge_index.device)
    return ForestResult(selected,root_tensor,torch.tensor([root in anchors for root in roots],
                         dtype=torch.bool,device=edge_index.device))


def extract_bfs_weights(data):
    """The single Data adapter used by evaluation and compatibility callers."""
    return bfs_forest(data.x.shape[0],data.edge_index,
                      torch.where(data.x[:,4] == 1)[0]).edge_weights


def extract_bfs_tree_edges(num_agents,num_anchors,edge_index,is_anchor_edge):
    """Compatibility API: global indices, agents followed by anchors."""
    if (isinstance(num_agents,bool) or not isinstance(num_agents,int) or num_agents < 1
            or isinstance(num_anchors,bool) or not isinstance(num_anchors,int) or num_anchors < 0):
        raise ValueError('Require positive num_agents and nonnegative num_anchors')
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must have shape (2,E)')
    if is_anchor_edge.shape != (edge_index.shape[1],):
        raise ValueError('is_anchor_edge must have shape (E,)')
    if not ((is_anchor_edge == 0) | (is_anchor_edge == 1)).all():
        raise ValueError('Anchor edge flags must be binary')
    if not torch.equal(is_anchor_edge.bool(),edge_index[1] >= num_agents):
        raise ValueError('Expected global canonical indices; local anchor indices are not accepted')
    anchors = torch.arange(num_agents,num_agents+num_anchors,device=edge_index.device)
    return bfs_forest(num_agents+num_anchors,edge_index,anchors).edge_weights
