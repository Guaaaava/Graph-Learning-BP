"""Static particle BP with explicit cavity messages on fixed prior supports.

The historical file adapted NEBP's dynamic localization implementation. This
replacement follows the thesis's static sum-product equations: exclude the
receiver factor, integrate on a fixed support, then form weighted moments.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import torch

def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def _dtype(name):
    if name not in ('float32', 'float64'):
        raise ValueError('compute_dtype must be float32 or float64')
    return getattr(torch, name)


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'{name} is not finite; use float64/check input scales; no likelihood floor is applied')


@dataclass
class ParticleSupport:
    """Fixed points (A,P,2) and log prior integration masses (A,P).

    For i.i.d. prior proposals, masses are uniform: prior/proposal cancels.
    For explicit quadrature, caller must supply prior density times integration
    weights, never prior density a second time on prior-sampled particles.
    Rows follow agent_node_index. No resampling or roughening occurs.
    """
    points: torch.Tensor
    log_prior_weights: torch.Tensor
    agent_node_index: torch.Tensor


@dataclass
class BPResult:
    mean: torch.Tensor
    covariance: torch.Tensor
    support: ParticleSupport
    weights: torch.Tensor
    log_messages: torch.Tensor
    directed_edge_index: torch.Tensor
    message_edge_to_measurement: torch.Tensor
    iterations: int
    converged: bool
    message_residual: Optional[float]
    ess: torch.Tensor
    anchor_reachable: torch.Tensor
    covariance_min_eigenvalue: torch.Tensor
    diagnostics: dict
    history: Optional[list] = None

    @property
    def particles(self):
        """Legacy coordinate layout (A,2,P); these are weighted support points."""
        return self.support.points.transpose(1, 2)


@dataclass
class _Graph:
    positions: torch.Tensor
    covariance: torch.Tensor
    agents: torch.Tensor
    node_to_agent: torch.Tensor
    anchor: torch.Tensor
    edges: torch.Tensor
    measurements: torch.Tensor
    variances: torch.Tensor
    active: torch.Tensor


def _graph(data, edge_weights, dtype):
    # This routine intentionally reads no labels, truth, or diagnostic fields.
    if getattr(data, 'num_graphs', 1) != 1:
        raise ValueError('ParticleBP expects one graph; use particle_bp_batch for batches')
    x, ei, ea, cov = (getattr(data, key, None) for key in
                      ('x', 'edge_index', 'edge_attr', 'agent_prior_cov'))
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 5 or not x.is_floating_point():
        raise ValueError('Expected floating x=(N,5)')
    _finite(x, 'x')
    n = x.shape[0]
    if not ((x[:, 4] == 0) | (x[:, 4] == 1)).all():
        raise ValueError('Anchor indicator must be binary')
    anchor = x[:, 4].bool()
    agents = torch.where(~anchor)[0]
    if agents.numel() == 0:
        raise ValueError('At least one agent is required')
    declared_agents = getattr(data, 'agent_node_index', None)
    if declared_agents is None or not torch.equal(declared_agents, agents):
        raise ValueError('agent_node_index must list non-anchor nodes in node order')
    a = agents.numel()
    if not isinstance(cov, torch.Tensor) or cov.shape != (a, 2, 2) or not cov.is_floating_point() or cov.device != x.device:
        raise ValueError('Complete agent_prior_cov=(A,2,2) is required on the graph device')
    cov = cov.to(dtype=dtype)
    _finite(cov, 'prior covariance')
    if not torch.equal(cov, cov.transpose(-1, -2)):
        raise ValueError('Prior covariance must be symmetric')
    if torch.linalg.cholesky_ex(cov).info.any():
        raise ValueError('Prior covariance must be positive definite')
    if not torch.allclose(cov.diagonal(dim1=-2, dim2=-1), x[agents, 2:4].to(dtype), rtol=1e-6, atol=0):
        raise ValueError('Prior covariance diagonal disagrees with node features')
    if not isinstance(ei, torch.Tensor) or ei.ndim != 2 or ei.shape[0] != 2 or ei.dtype != torch.long:
        raise ValueError('Expected long edge_index=(2,E)')
    e = ei.shape[1]
    if (not isinstance(ea, torch.Tensor) or ea.shape != (e, 4)
            or not ea.is_floating_point() or ei.device != x.device or ea.device != x.device):
        raise ValueError('Expected edge_attr=(E,4) on graph device')
    _finite(ea, 'edge_attr')
    if e:
        if ei.min() < 0 or ei.max() >= n:
            raise ValueError('Physical edge index out of bounds (global indices required)')
        lo, hi = ei.min(dim=0).values, ei.max(dim=0).values
        if (lo == hi).any() or torch.unique(lo * n + hi).numel() != e:
            raise ValueError('Physical edges must be unique and have distinct endpoints')
        if (anchor[ei[0]] & anchor[ei[1]]).any():
            raise ValueError('Anchor-anchor factors are not localization measurements')
        if not torch.equal(ea[:, 3], (anchor[ei[0]] | anchor[ei[1]]).to(ea.dtype)):
            raise ValueError('Anchor-edge flags disagree with endpoint types')
    if not (ea[:, 1] > 0).all():
        raise ValueError('Known measurement variances must be positive')
    if edge_weights is None:
        active = torch.ones(e, dtype=torch.bool, device=x.device)
    else:
        if (not isinstance(edge_weights, torch.Tensor) or edge_weights.shape != (e,)
                or edge_weights.device != x.device
                or not ((edge_weights == 0) | (edge_weights == 1)).all()):
            raise ValueError('edge_weights must be a strict binary physical-edge mask on the graph device')
        active = edge_weights.detach().bool()
    mapping = torch.full((n,), -1, dtype=torch.long, device=x.device)
    mapping[agents] = torch.arange(a, device=x.device)
    return _Graph(x[:, :2].to(dtype), cov, agents, mapping, anchor, ei,
                  ea[:, 0].to(dtype), ea[:, 1].to(dtype), active)


def _normalise(log_values):
    return log_values - torch.logsumexp(log_values, dim=-1, keepdim=True)


def _moments(points, weights):
    mean = (points * weights[..., None]).sum(dim=1)
    centered = points - mean[:, None, :]
    covariance = torch.einsum('ap,api,apj->aij', weights, centered, centered)
    return mean, covariance


class ParticleBP:
    """Synchronous sum-product; memory O(A*P + E_agent*P + chunk_size**2).

    num_iter is the maximum number of agent-agent message rounds. Anchors are
    substituted exactly in local factors before round 0. Damping is a convex
    mixture of normalized old/new messages (0 means ordinary BP).
    """

    def __init__(self, num_particles=2000, num_iter=10, *, chunk_size=256,
                 compute_dtype='float64', measurement_variance_floor=0.,
                 damping=0., message_tolerance=1e-8, early_stopping=False,
                 **legacy_options):
        if legacy_options:
            raise TypeError('Obsolete/unknown BP options '+str(sorted(legacy_options))+
                            '; priors/variances come from graph inputs; roughening is unsupported')
        _integer(num_particles, 'num_particles', 2)
        _integer(num_iter, 'num_iter', 0)
        _integer(chunk_size, 'chunk_size', 1)
        _dtype(compute_dtype)
        for value, name in ((measurement_variance_floor, 'measurement_variance_floor'),
                            (damping, 'damping'), (message_tolerance, 'message_tolerance')):
            _nonnegative(value, name)
        if damping >= 1:
            raise ValueError('damping must be < 1')
        if not isinstance(early_stopping, bool):
            raise ValueError('early_stopping must be boolean')
        self.num_particles, self.num_iter, self.chunk_size = num_particles, num_iter, chunk_size
        self.compute_dtype, self.measurement_variance_floor = compute_dtype, measurement_variance_floor
        self.damping, self.message_tolerance, self.early_stopping = damping, message_tolerance, early_stopping

    @classmethod
    def from_config(cls, config):
        return cls(config.num_particles, config.num_iterations, chunk_size=config.chunk_size,
                   compute_dtype=config.compute_dtype,
                   measurement_variance_floor=config.measurement_variance_floor,
                   damping=config.damping, message_tolerance=config.message_tolerance,
                   early_stopping=config.early_stopping)

    @torch.no_grad()
    def draw_support(self, data, *, generator=None):
        graph = _graph(data, None, _dtype(self.compute_dtype))
        a, p = graph.agents.numel(), self.num_particles
        noise = torch.randn((a, p, 2), dtype=graph.positions.dtype,
                            device=graph.positions.device, generator=generator)
        points = graph.positions[graph.agents, None, :] + noise @ torch.linalg.cholesky(graph.covariance).transpose(-1, -2)
        masses = points.new_full((a, p), -math.log(p))
        return ParticleSupport(points, masses, graph.agents.clone())

    def _support(self, graph, support):
        points, masses = support.points, support.log_prior_weights
        a, p = graph.agents.numel(), self.num_particles
        if (points.shape != (a, p, 2) or masses.shape != (a, p)
                or points.device != graph.positions.device or masses.device != points.device
                or points.dtype != graph.positions.dtype or masses.dtype != points.dtype
                or not torch.equal(support.agent_node_index, graph.agents)):
            raise ValueError('Support shape/dtype/device/node ordering must match graph and BP configuration')
        _finite(points, 'support points')
        if torch.isnan(masses).any() or torch.isposinf(masses).any() or not torch.isfinite(masses).any(dim=1).all():
            raise ValueError('Prior log masses may be finite or -inf, with positive total mass for each agent')
        return ParticleSupport(points, _normalise(masses), graph.agents.clone())

    def _pair_messages(self, left, right, left_cavity, right_cavity, z, variance):
        p, chunk = self.num_particles, self.chunk_size
        to_left, to_right = left.new_full((p,), -torch.inf), right.new_full((p,), -torch.inf)
        finite_kernel = torch.ones((), dtype=torch.bool, device=left.device)
        for i in range(0, p, chunk):
            for j in range(0, p, chunk):
                # One kernel block serves both message directions.
                kernel = torch.cdist(left[i:i+chunk], right[j:j+chunk],
                                     compute_mode='donot_use_mm_for_euclid_dist')
                kernel.sub_(z).square_().mul_(-0.5 / variance)
                # Accumulate on device: checking a Python boolean for every
                # block would force a CPU/GPU synchronization per kernel.
                finite_kernel &= torch.isfinite(kernel).all()
                right_block = torch.logsumexp(kernel + left_cavity[i:i+chunk, None], dim=0)
                left_block = torch.logsumexp(kernel + right_cavity[None, j:j+chunk], dim=1)
                to_right[j:j+chunk] = torch.logaddexp(to_right[j:j+chunk], right_block)
                to_left[i:i+chunk] = torch.logaddexp(to_left[i:i+chunk], left_block)
        if not finite_kernel:
            raise FloatingPointError('measurement log likelihood is not finite; use float64/check input scales')
        return _normalise(to_right), _normalise(to_left)

    @torch.no_grad()
    def infer(self, data, edge_weights=None, *, support=None, generator=None, return_history=False):
        if support is not None and generator is not None:
            raise ValueError('Pass fixed support or generator, not both')
        graph = _graph(data, edge_weights, _dtype(self.compute_dtype))
        if support is None:
            support = self.draw_support(data, generator=generator)
        if not isinstance(support, ParticleSupport):
            raise TypeError('support must be a ParticleSupport with explicit prior masses')
        support = self._support(graph, support)
        points, p = support.points, self.num_particles
        variances = graph.variances.clamp_min(self.measurement_variance_floor)
        a = points.shape[0]
        local = support.log_prior_weights.clone()
        pairs, anchor_neighbors = [], []
        adjacency = [[] for _ in range(graph.positions.shape[0])]
        for edge in torch.where(graph.active)[0].tolist():
            u, v = graph.edges[:, edge].tolist()
            adjacency[u].append(v)
            adjacency[v].append(u)
            if graph.anchor[u] or graph.anchor[v]:
                node, anchor = (v, u) if graph.anchor[u] else (u, v)
                agent = int(graph.node_to_agent[node])
                distance = (points[agent] - graph.positions[anchor]).norm(dim=-1)
                likelihood = -(distance - graph.measurements[edge]).square() / (2 * variances[edge])
                _finite(likelihood, 'anchor log likelihood')
                local[agent] += likelihood
                anchor_neighbors.append(edge)
            else:
                pairs.append((int(graph.node_to_agent[u]), int(graph.node_to_agent[v]), edge))
        local = _normalise(local)
        # Slots 2k=u->v (function on v), 2k+1=v->u (function on u).
        incoming = [[] for _ in range(a)]
        directed, physical_ids = [], []
        for k, (u, v, edge) in enumerate(pairs):
            incoming[v].append(2 * k)
            incoming[u].append(2 * k + 1)
            directed.extend([(int(graph.agents[u]), int(graph.agents[v])),
                             (int(graph.agents[v]), int(graph.agents[u]))])
            physical_ids.extend([edge, edge])
        messages = points.new_full((2 * len(pairs), p), -math.log(p))
        def belief(current):
            rows = [local[node] + sum((current[slot] for slot in slots), torch.zeros_like(local[node]))
                    for node, slots in enumerate(incoming)]
            return torch.stack(rows).softmax(dim=-1)
        def snapshot(round_id, current, residual):
            weights = belief(current)
            mean, covariance = _moments(points, weights)
            return {'iteration': round_id, 'mean': mean.clone(), 'covariance': covariance.clone(),
                    'weights': weights.clone(), 'log_messages': current.clone(),
                    'message_residual': residual}
        history = [snapshot(0, messages, None)] if return_history else None
        residual, iterations = (0. if not pairs else None), 0
        for round_id in range(1, self.num_iter + 1) if pairs else ():
            updated = torch.empty_like(messages)
            for k, (u, v, edge) in enumerate(pairs):
                # Sum only the OTHER factors: do not send the receiver's evidence
                # straight back and do not subtract nearly equal large log values.
                cavity_u, cavity_v = local[u].clone(), local[v].clone()
                for slot in incoming[u]:
                    if slot != 2 * k + 1:
                        cavity_u += messages[slot]
                for slot in incoming[v]:
                    if slot != 2 * k:
                        cavity_v += messages[slot]
                uv, vu = self._pair_messages(points[u], points[v], _normalise(cavity_u),
                                              _normalise(cavity_v), graph.measurements[edge], variances[edge])
                updated[2 * k], updated[2 * k + 1] = uv, vu
            if self.damping:
                updated = torch.logaddexp(updated + math.log1p(-self.damping),
                                          messages + math.log(self.damping))
                updated = _normalise(updated)
            residual = float((updated - messages).abs().max())
            messages, iterations = updated, round_id
            if return_history:
                history.append(snapshot(round_id, messages, residual))
            if self.early_stopping and residual <= self.message_tolerance:
                break
        weights = belief(messages)
        mean, covariance = _moments(points, weights)
        _finite(mean, 'posterior mean')
        _finite(covariance, 'posterior covariance')
        ess = weights.square().sum(dim=1).reciprocal()
        reachable = set(torch.where(graph.anchor)[0].tolist())
        queue = list(reachable)
        for node in queue:
            for neighbor in adjacency[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        anchored = torch.tensor([int(node) in reachable for node in graph.agents],
                                dtype=torch.bool, device=points.device)
        directed_tensor = torch.tensor(directed, dtype=torch.long, device=points.device).reshape(-1, 2).t().contiguous()
        ids_tensor = torch.tensor(physical_ids, dtype=torch.long, device=points.device)
        diagnostics = {'compute_dtype': self.compute_dtype,
                       'num_particles': p, 'chunk_size': self.chunk_size,
                       'selected_physical_edges': int(graph.active.sum()), 'agent_agent_edges': len(pairs),
                       'anchor_edges': len(anchor_neighbors), 'measurement_variance_floor': self.measurement_variance_floor,
                       'variance_floor_applied_edges': int(((graph.variances < self.measurement_variance_floor) & graph.active).sum()),
                       'damping': self.damping, 'message_tolerance': self.message_tolerance,
                       'early_stopping': self.early_stopping, 'max_kernel_elements': min(p, self.chunk_size)**2 if pairs and iterations else 0,
                       'pair_distance_evaluations': len(pairs) * iterations * p * p,
                       'resampling': False, 'roughening': False}
        return BPResult(mean, covariance, support, weights, messages, directed_tensor, ids_tensor,
                        iterations, residual is not None and residual <= self.message_tolerance,
                        residual, ess, anchored, torch.linalg.eigvalsh(covariance)[:, 0], diagnostics, history)

    def run(self, data, edge_weights=None, *, support=None, generator=None):
        """Compatibility tuple (mean,covariance,support_points[A,2,P]).

        Third output is NOT resampled. Use infer().weights for its posterior
        masses. Reported moments always use the final weighted distribution.
        """
        result = self.infer(data, edge_weights, support=support, generator=generator)
        return result.mean, result.covariance, result.particles


def particle_bp_batch(data, edge_weights=None, *, supports=None, generators=None,
                      return_results=False, **kwargs):
    """Independent graphs; explicit support/generator per graph supports pairing."""
    count = data.num_graphs
    for values in (supports, generators):
        if values is not None and len(values) != count:
            raise ValueError('One support/generator per graph is required')
    if supports is not None and generators is not None:
        raise ValueError('Pass supports or generators, not both')
    if edge_weights is not None and edge_weights.shape != (data.edge_index.shape[1],):
        raise ValueError('Batch mask must follow physical edges')
    bp, results = ParticleBP(**kwargs), []
    for graph_id in range(count):
        sub = data.get_example(graph_id)
        mask = None if edge_weights is None else edge_weights[data.batch[data.edge_index[0]] == graph_id]
        results.append(bp.infer(sub, mask,
                               support=None if supports is None else supports[graph_id],
                               generator=None if generators is None else generators[graph_id]))
    if return_results:
        return results
    return ([result.mean for result in results], [result.covariance for result in results],
            [result.particles for result in results])
