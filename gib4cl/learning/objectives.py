"""Paper-aligned Gaussian predictor and conditional GIB objective.

One decision per physical edge. Sum inside each graph, then average graphs.
The predictor never reads position targets. All candidate rows remain in the
linear algebra, including zero-weight edges, so STE derivatives are preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

from gib4cl.config import ObjectiveConfig, _positive


# 3↓ Check functions
def _dtype(name):
    if name not in ('float32', 'float64'):
        raise ValueError('compute_dtype must be float32 or float64')
    return getattr(torch, name)


def _finite(value, name):
    if not torch.isfinite(value.detach()).all():
        raise ValueError(f'{name} contains NaN or Inf')


def _vector(value, size, name, device):
    if not isinstance(value, torch.Tensor) or value.shape != (size,) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating tensor with shape ({size},)')
    if value.device != device:
        raise ValueError(f'{name} must share the graph device')
    _finite(value, name)


@dataclass
class GraphLayout:
    node_batch: torch.Tensor
    edge_batch: torch.Tensor
    agent_nodes: torch.Tensor
    agent_batch: torch.Tensor
    num_graphs: int


def _layout(data):
    """Read only input geometry/indexing; target positions are not accessed."""
    if data.x is None or data.x.ndim != 2 or data.x.shape[1] != 5:
        raise ValueError('Require the v1 five-dimensional node features')
    _finite(data.x, 'node inputs')
    n = data.x.shape[0]
    flags = data.x[:, 4]
    if not ((flags == 0) | (flags == 1)).all():
        raise ValueError('Anchor flags must be binary')
    anchor = flags.bool()
    ei = data.edge_index
    if ei is None or ei.dtype != torch.long or ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError('Require global int64 edge_index with shape (2,E)')
    if ei.device != data.x.device:
        raise ValueError('Edge and node inputs must share a device')
    e = ei.shape[1]
    if data.edge_attr is None or data.edge_attr.shape != (e, 4):
        raise ValueError('Require one four-dimensional feature vector per physical edge')
    _finite(data.edge_attr, 'edge inputs')
    if e and ((ei < 0).any() or (ei >= n).any() or (ei[0] >= ei[1]).any()):
        raise ValueError('Physical edges must use canonical global u<v indices')
    if e and (torch.unique(ei[0] * n + ei[1]).numel() != e or anchor[ei[0]].any()):
        raise ValueError('No duplicate or anchor-anchor physical edges are allowed')
    node_batch = getattr(data, 'batch', None)
    if node_batch is None:
        node_batch = torch.zeros(n, device=data.x.device, dtype=torch.long)
        count = 1
    else:
        count = int(data.num_graphs)
        if node_batch.shape != (n,) or node_batch.dtype != torch.long:
            raise ValueError('Invalid node-to-graph mapping')
    if count < 1 or n == 0:
        raise ValueError('An empty batch is not a training graph')
    if (node_batch < 0).any() or (node_batch >= count).any():
        raise ValueError('Invalid graph IDs')
    edge_batch = node_batch[ei[0]]
    if not torch.equal(edge_batch, node_batch[ei[1]]):
        raise ValueError('A physical measurement cannot join different batch graphs')
    agents = torch.where(~anchor)[0]
    agent_batch = node_batch[agents]
    if (torch.bincount(agent_batch, minlength=count) == 0).any():
        raise ValueError('Each graph needs at least one agent')
    supplied_agents = getattr(data, 'agent_node_index', None)
    if supplied_agents is None or not torch.equal(supplied_agents, agents):
        raise ValueError('Require the v1 agent mapping in target/covariance order')
    cov = getattr(data, 'agent_prior_cov', None)
    if cov is None or cov.shape != (agents.numel(), 2, 2) or cov.device != data.x.device:
        raise ValueError('Require full agent_prior_cov blocks on the graph device')
    _finite(cov, 'agent prior covariance')
    if not torch.equal(cov, cov.transpose(-1, -2)):
        raise ValueError('Prior covariances must be symmetric')
    return GraphLayout(node_batch, edge_batch, agents, agent_batch, count)


@dataclass
class GaussianPrediction:
    graph_id: int
    target_index: torch.Tensor
    edge_ids: torch.Tensor
    gradient: torch.Tensor          # (E_g, 2*A_g), g_ij^T in each row
    prior_precision: torch.Tensor
    precision: torch.Tensor
    information_vector: torch.Tensor
    delta: torch.Tensor
    mean: torch.Tensor              # (A_g,2)
    cholesky: torch.Tensor
    logdet_precision: torch.Tensor

    @property
    def covariance(self):
        # Computed only when requested; training needs no dense inverse.
        return torch.cholesky_inverse(self.cholesky)


def gaussian_predictions(data, edge_weights, *, variance_floor=1e-6,
                         compute_dtype='float64', strict_binary=True):
    """Construct q_pred using initial positions and selected measurements only.

    strict_binary=False is for smooth surrogate/finite-difference diagnostics;
    the training objective below always requires a binary forward mask.
    """
    _positive(variance_floor, 'variance_floor')
    layout = _layout(data)
    _vector(edge_weights, data.edge_index.shape[1], 'edge_weights', data.x.device)
    detached = edge_weights.detach()
    if strict_binary:
        if not ((detached == 0) | (detached == 1)).all():
            raise ValueError('Training requires a binary forward mask (STE is supported)')
    elif ((detached < 0) | (detached > 1)).any():
        raise ValueError('Relaxed diagnostic weights must be in [0,1]')
    dtype = _dtype(compute_dtype)
    pos = data.x[:, :2].to(dtype=dtype)
    cov = data.agent_prior_cov.to(dtype=dtype)
    prior_factor, info = torch.linalg.cholesky_ex(cov)
    if info.ne(0).any():
        raise ValueError('Physical prior covariance must be positive definite; no automatic jitter is applied')
    prior_blocks = torch.cholesky_inverse(prior_factor)
    u, v = data.edge_index
    diff = pos[u] - pos[v]
    distances = diff.norm(dim=1)
    if (distances.detach() <= 0).any():
        raise ValueError('Connected initial coordinates must be distinct')
    _finite(distances, 'initial distances')
    directions = diff / distances[:, None]
    measurements = data.edge_attr[:, 0].to(dtype=dtype)
    raw_variances = data.edge_attr[:, 1].to(dtype=dtype)
    if not (raw_variances.detach() > 0).all():
        raise ValueError('Known measurement variances must be positive')
    variances = raw_variances.clamp_min(variance_floor)
    residual = measurements - distances   # Signed; abs residual is a GNN feature only.
    weights = edge_weights.to(dtype=dtype)
    predictions = []
    for graph_id in range(layout.num_graphs):
        target_rows = torch.where(layout.agent_batch == graph_id)[0]
        agent_nodes = layout.agent_nodes[target_rows]
        edge_ids = torch.where(layout.edge_batch == graph_id)[0]
        local_u, local_v = u[edge_ids], v[edge_ids]
        incidence = ((local_u[:, None] == agent_nodes[None, :]).to(dtype)
                     - (local_v[:, None] == agent_nodes[None, :]).to(dtype))
        gradient = (incidence[:, :, None] * directions[edge_ids, None, :]).reshape(edge_ids.numel(), 2 * target_rows.numel())
        j0 = torch.block_diag(*prior_blocks[target_rows].unbind(0))
        coefficients = weights[edge_ids] / variances[edge_ids]
        precision = j0 + gradient.T @ (coefficients[:, None] * gradient)
        precision = (precision + precision.T) * 0.5
        information = gradient.T @ (coefficients * residual[edge_ids])
        _finite(precision, 'prediction precision')
        _finite(information, 'prediction information vector')
        chol, status = torch.linalg.cholesky_ex(precision)
        if status.ne(0).any():
            raise ValueError(f'Prediction precision failed Cholesky for graph {graph_id}; inspect scale/precision')
        delta = torch.cholesky_solve(information[:, None], chol).squeeze(1)
        mean = pos[agent_nodes] + delta.reshape(-1, 2)
        _finite(mean, 'prediction mean')
        predictions.append(GaussianPrediction(
            graph_id, target_rows, edge_ids, gradient, j0, precision, information,
            delta, mean, chol, 2 * chol.diagonal().log().sum()))
    return predictions


def gaussian_task_terms(predictions, targets):
    """Evaluate target positions after constructing the input-only predictor."""
    total_agents = sum(pred.mean.shape[0] for pred in predictions)
    if targets is None or not isinstance(targets, torch.Tensor) or targets.shape != (total_agents, 2):
        raise ValueError('Task NLL requires agent position targets in v1 order')
    _finite(targets, 'position targets')
    volume, mahalanobis, constants = [], [], []
    for pred in predictions:
        truth = targets.to(device=pred.mean.device, dtype=pred.mean.dtype)[pred.target_index]
        error = (truth - pred.mean).reshape(-1)
        volume.append(-0.5 * pred.logdet_precision)
        mahalanobis.append(0.5 * error @ pred.precision @ error)
        constants.append(pred.mean.new_tensor(pred.mean.numel() * 0.5 * math.log(2 * math.pi)))
    logdet_term = torch.stack(volume)
    mahalanobis_term = torch.stack(mahalanobis)
    task = logdet_term + mahalanobis_term
    return {'task': task, 'task_logdet': logdet_term,
            'task_mahalanobis': mahalanobis_term,
            'task_nll': task + torch.stack(constants)}


def _triangles(data, layout):
    triangles = getattr(data, 'triangle_edge_index', None)
    e = data.edge_index.shape[1]
    if triangles is None or triangles.dtype != torch.long or triangles.ndim != 2 or triangles.shape[0] != 3:
        raise ValueError('Require v1 triangle_edge_index with shape (3,T)')
    if triangles.device != data.x.device:
        raise ValueError('Triangle indices must share the graph device')
    if triangles.numel():
        if (triangles < 0).any() or (triangles >= e).any():
            raise ValueError('Triangle indices must refer to physical measurement edges')
        sorted_ids = triangles.sort(dim=0).values
        if (sorted_ids[1:] <= sorted_ids[:-1]).any() or torch.unique(sorted_ids, dim=1).shape[1] != triangles.shape[1]:
            raise ValueError('Each triangle must have three distinct edges and be counted once')
        endpoints = data.edge_index[:, triangles].permute(2, 1, 0).reshape(-1, 6).sort(dim=1).values
        if (not (endpoints[:, ::2] == endpoints[:, 1::2]).all()
                or not (endpoints[:, 0] < endpoints[:, 2]).all()
                or not (endpoints[:, 2] < endpoints[:, 4]).all()
                or data.x[endpoints, 4].bool().any()):
            raise ValueError('Only closed triangles of three unknown agents may be penalized')
        if not (layout.edge_batch[triangles] == layout.edge_batch[triangles[0]][None, :]).all():
            raise ValueError('A triangle cannot span batch graphs')
    return triangles


def compression_terms(edge_logits, data, *, gamma, kappa, compute_dtype='float64'):
    """Analytic expectation over independent physical-edge Bernoulli variables."""
    _positive(gamma, 'gamma', allow_zero=True)
    _positive(kappa, 'kappa', allow_zero=True)
    layout = _layout(data)
    e = data.edge_index.shape[1]
    _vector(edge_logits, e, 'edge_logits', data.x.device)
    for key in ('edge_cost', 'edge_log_cardinality'):
        value = getattr(data, key, None)
        _vector(value, e, key, data.x.device)
        if (value.detach() < 0).any():
            raise ValueError(f'{key} must be nonnegative')
    triangles = _triangles(data, layout)
    dtype = _dtype(compute_dtype)
    logits = edge_logits.to(dtype=dtype)
    p, complement = logits.sigmoid(), (-logits).sigmoid()
    # Complement via sigmoid(-logit) avoids cancellation at p approximately 1.
    negative_entropy = p * F.logsigmoid(logits) + complement * F.logsigmoid(-logits)
    def sum_edges(values):
        return values.new_zeros(layout.num_graphs).index_add(0, layout.edge_batch, values)
    entropy = sum_edges(negative_entropy)
    energy = sum_edges(p * data.edge_cost.to(dtype=dtype))
    retained_data = sum_edges(p * data.edge_log_cardinality.to(dtype=dtype))
    triangle_values = p[triangles].prod(dim=0)
    triangle_graph = layout.edge_batch[triangles[0]]
    triangle_count = p.new_zeros(layout.num_graphs).index_add(0, triangle_graph, triangle_values)
    compression = entropy + gamma * energy + kappa * triangle_count + retained_data
    return {'compression': compression, 'negative_entropy': entropy,
            'expected_energy': energy, 'expected_triangles': triangle_count,
            'retained_data_cost': retained_data,
            'communication_penalty': gamma * energy, 'triangle_penalty': kappa * triangle_count}


def degree_terms(edge_weights, data, *, compute_dtype='float64'):
    layout = _layout(data)
    _vector(edge_weights, data.edge_index.shape[1], 'edge_weights', data.x.device)
    weights = edge_weights.to(dtype=_dtype(compute_dtype))
    u, v = data.edge_index
    # Do not threshold/index by weights: this sum must retain the STE path.
    degrees = weights.new_zeros(data.x.shape[0]).index_add(0, u, weights).index_add(0, v, weights)
    violations = F.relu(3.0 - degrees[layout.agent_nodes])
    penalty = weights.new_zeros(layout.num_graphs).index_add(0, layout.agent_batch, violations)
    active = weights.new_zeros(layout.num_graphs).index_add(0, layout.edge_batch, weights)
    return {'degree': penalty, 'active_edges': active}


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    per_graph: dict
    predictions: list

    def metrics(self):
        return {key: float(value.detach().mean().cpu()) for key, value in self.per_graph.items()}


def gib_objective(edge_logits, edge_weights, data, *, objective):
    if not isinstance(objective, ObjectiveConfig):
        raise TypeError('objective must be an ObjectiveConfig with explicit coefficients')
    objective.require_ready()
    predictions = gaussian_predictions(
        data, edge_weights, variance_floor=objective.prediction_variance_floor,
        compute_dtype=objective.compute_dtype, strict_binary=True)
    tasks = gaussian_task_terms(predictions, getattr(data, 'y', None))
    compression = compression_terms(edge_logits, data, gamma=objective.gamma, kappa=objective.kappa,
                                    compute_dtype=objective.compute_dtype)
    degrees = degree_terms(edge_weights, data, compute_dtype=objective.compute_dtype)
    total = tasks['task'] + objective.beta * compression['compression'] + objective.eta * degrees['degree']
    _finite(total, 'total objective')
    per_graph = {**tasks, **compression, **degrees, 'total': total,
                 'weighted_compression': objective.beta * compression['compression'],
                 'weighted_degree': objective.eta * degrees['degree']}
    return ObjectiveResult(total.mean(), per_graph, predictions)


def compute_gib_loss(edge_logits, edge_weights, data, *, objective):
    """Training adapter: differentiable scalar plus detached mean-per-graph logs."""
    result = gib_objective(edge_logits, edge_weights, data, objective=objective)
    return result.loss, result.metrics()
