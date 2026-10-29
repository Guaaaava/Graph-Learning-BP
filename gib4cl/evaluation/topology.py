"""Shared dense/BFS/GNN topology selection; no hidden connectivity repair."""
import math
import torch
from gib4cl.inference.spanning_tree import bfs_forest,extract_bfs_weights


# s_dense, s_gnn, s_bfs
@torch.no_grad()
def select_topologies(data,model,threshold=.5):
    if not 0 < threshold < 1:
        raise ValueError('threshold must be in (0,1)')
    model.eval()
    logits = model.score_edges(data)
    probabilities = logits.sigmoid()
    # Comparing logits avoids sigmoid rounding near the threshold.
    boundary = math.log(threshold)-math.log1p(-threshold)
    forest = bfs_forest(len(data.x),data.edge_index,torch.where(data.x[:,4] == 1)[0])
    masks = {'dense':torch.ones_like(logits),'bfs':forest.edge_weights.to(logits.dtype),
             'gnn':(logits > boundary).to(logits.dtype)}
    return masks,{'gnn_probabilities':probabilities.tolist(),'gnn_logits':logits.tolist(),
                  'threshold':threshold,'bfs_forest_root':forest.forest_root.tolist(),
                  'candidate_anchor_reachable':forest.anchor_reachable.tolist()}


def selection_metrics(data,mask):
    """Actual selected unique edges, graph costs and agent-only feedback structure.

    Reachability is a structural diagnostic; it is not a rigidity/localizability proof.
    Isolated anchors count as components of the full graph, but not of the agent graph.
    """
    agents = data.agent_node_index.detach().cpu().tolist()
    edges = data.edge_index.detach().cpu()
    active = mask.detach().cpu()
    if active.shape != (edges.shape[1],) or not ((active == 0) | (active == 1)).all():
        raise ValueError('Selection metrics require one binary weight per unique physical edge')
    active = active.bool()
    costs = data.edge_cost.detach().cpu().double()
    if costs.shape != active.shape or not torch.isfinite(costs).all() or (costs < 0).any():
        raise ValueError('Communication costs must be finite and nonnegative')
    agent_set = set(agents)
    n,a = len(data.x),len(agents)
    if not a:
        raise ValueError('At least one agent is required')
    parents = list(range(n))
    agent_parents = list(range(n))
    def root(parent,node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    degrees = [0]*n
    agent_edges = 0
    for u,v in edges[:,active].T.tolist():
        degrees[u] += 1
        degrees[v] += 1
        parents[root(parents,v)] = root(parents,u)
        if u in agent_set and v in agent_set:
            agent_edges += 1
            agent_parents[root(agent_parents,v)] = root(agent_parents,u)
    anchor_roots = {root(parents,i) for i in range(n) if i not in agent_set}
    unanchored = sum(root(parents,i) not in anchor_roots for i in agents)
    components = len({root(parents,i) for i in range(n)})
    agent_components = len({root(agent_parents,i) for i in agents})
    agent_degrees = [degrees[i] for i in agents]
    triangles = data.triangle_edge_index.detach().cpu()
    selected_triangles = int(active[triangles].all(dim=0).sum())
    selected_edges = int(active.sum())
    cost = math.fsum(costs[active].tolist())
    candidate_cost = math.fsum(costs.tolist())
    retention = selected_edges/len(active) if len(active) else None
    cost_retention = cost/candidate_cost if candidate_cost else None
    return {
        'edges':selected_edges,'candidate_edges':len(active),'edge_retention':retention,
        'edge_saving':1-retention if retention is not None else None,
        'communication_cost':cost,'candidate_communication_cost':candidate_cost,
        'communication_retention':cost_retention,
        'communication_saving':1-cost_retention if cost_retention is not None else None,
        'agent_edges':agent_edges,'anchor_edges':selected_edges-agent_edges,
        'agent_triangles':selected_triangles,'candidate_agent_triangles':triangles.shape[1],
        'agent_triangle_retention':selected_triangles/triangles.shape[1] if triangles.shape[1] else None,
        'agent_cycle_rank':agent_edges-a+agent_components,
        'connected_components':components,'agent_components':agent_components,
        'agent_degree_min':min(agent_degrees),'agent_degree_mean':sum(agent_degrees)/a,
        'agent_degree_max':max(agent_degrees),'isolated_agents':agent_degrees.count(0),
        'agents_below_degree_three':sum(d < 3 for d in agent_degrees),
        'agents_below_degree_three_fraction':sum(d < 3 for d in agent_degrees)/a,
        'unanchored_agents':unanchored,'anchor_reachable_fraction':1-unanchored/a,
    }
