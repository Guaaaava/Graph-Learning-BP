"""Joint-model data contract and small CPU reference, NOT a GPU BP backend.

One graph/support has shared geometry; masks, unary evidence, cavity messages
and beliefs remain model-specific. NumPy implements synchronous log-domain BP
independently of ParticleBP's pair-message recursion. Its full kernels are
explicitly bounded: this reference must never run at the formal particle budget.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch

from gib4cl.inference.particle_bp import ParticleBP, ParticleSupport, _graph
from gib4cl.runtime import state_digest


@dataclass(frozen=True)
class JointProblem:
    data: object
    support: ParticleSupport
    masks: torch.Tensor                     # (models, physical edges)
    model_ids: tuple
    edge_models: tuple                      # active model rows per physical edge
    identity: dict

    def verify(self):
        if self.identity != _identity(self.data, self.support, self.masks, self.model_ids):
            raise ValueError('Prepared joint inputs were mutated')


def _identity(data, support, masks, model_ids):
    return dict(sample_id=data.sample_id, model_ids=list(model_ids),
                input_digest=state_digest(data.to_dict()),
                support_digest=state_digest({'points': support.points, 'masses': support.log_prior_weights,
                                             'agents': support.agent_node_index}),
                masks_digest=state_digest({'masks': masks}))


def prepare_joint(data, masks, model_ids, support):
    if data.x.device.type != 'cpu' or support.points.device.type != 'cpu':
        raise ValueError('Stage-A reference contract accepts CPU inputs only')
    if not isinstance(masks, torch.Tensor) or masks.device.type != 'cpu':
        raise ValueError('masks must be a CPU tensor')
    model_ids = tuple(model_ids)
    if (not model_ids or any(not isinstance(x, str) or not x for x in model_ids)
            or len(set(model_ids)) != len(model_ids)):
        raise ValueError('Require distinct nonempty model IDs')
    if masks.shape != (len(model_ids), data.num_edges):
        raise ValueError('Expected masks=(models, physical edges)')
    if not ((masks == 0) | (masks == 1)).all():
        raise ValueError('Masks must be strict binary selections')
    graph = _graph(data, None, torch.float64)
    validator = ParticleBP(support.points.shape[1], 0, compute_dtype='float64')
    validator._support(graph, support)        # existing support validation, no inference
    copied = ParticleSupport(support.points.clone(), support.log_prior_weights.clone(),
                             support.agent_node_index.clone())
    clone, selected = data.clone(), masks.clone()
    edge_models = tuple(tuple(torch.where(selected[:, edge] == 1)[0].tolist())
                        for edge in range(data.num_edges))
    return JointProblem(clone, copied, selected, model_ids, edge_models,
                        _identity(clone, copied, selected, model_ids))


def _lse(values, axis=-1, keepdims=False):
    top = np.max(values, axis=axis, keepdims=True)
    # All uses have at least one finite term: positive prior mass and finite
    # log-likelihoods. Reject rather than invent a probability floor.
    if not np.isfinite(top).all():
        raise FloatingPointError('No finite positive mass in reference reduction')
    out = top + np.log(np.exp(values-top).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _norm(values):
    return values - _lse(values, keepdims=True)


def joint_cpu_reference(problem, *, iterations=10, return_history=False, max_kernel_elements=2_000_000):
    """Return independent model states while computing each union-edge kernel once.

    Only the current ordinary BP setting is supported: FP64, no damping,
    early-stop, variance floor, resampling or roughening. No speed claims.
    """
    problem.verify()
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    if (isinstance(max_kernel_elements, bool) or not isinstance(max_kernel_elements, int)
            or max_kernel_elements < 1):
        raise ValueError('max_kernel_elements must be a positive integer')
    data, support = problem.data, problem.support
    graph = _graph(data, None, torch.float64)
    points = support.points.numpy()
    base = _norm(support.log_prior_weights.numpy())
    a, p = base.shape
    if p > 512:
        raise ValueError('Small CPU reference is limited to 512 particles; not a formal backend')
    m = len(problem.model_ids)
    selected = problem.masks.numpy().astype(bool)
    # Each model has its own slots, preserving the original physical-edge order.
    pairs, incoming, directed, locals_, reachables = [], [], [], [], []
    union = {}
    for row in range(m):
        local = base.copy()
        model_pairs, slots = [], [[] for _ in range(a)]
        arcs, adjacency = [], [[] for _ in range(data.num_nodes)]
        for edge in np.flatnonzero(selected[row]):
            u, v = map(int, graph.edges[:, edge])
            adjacency[u].append(v); adjacency[v].append(u)
            z, variance = float(graph.measurements[edge]), float(graph.variances[edge])
            if bool(graph.anchor[u]) or bool(graph.anchor[v]):
                node, anchor = (v, u) if bool(graph.anchor[u]) else (u, v)
                agent = int(graph.node_to_agent[node])
                distance = np.linalg.norm(points[agent]-graph.positions[anchor].numpy(), axis=-1)
                term = -.5*(distance-z)**2/variance
                if not np.isfinite(term).all():
                    raise FloatingPointError('Nonfinite anchor likelihood')
                local[agent] += term
            else:
                left, right = int(graph.node_to_agent[u]), int(graph.node_to_agent[v])
                k = len(model_pairs)
                model_pairs.append((left, right, int(edge)))
                slots[right].append(2*k); slots[left].append(2*k+1)
                arcs.extend([(u, v), (v, u)])
                union[int(edge)] = (left, right, z, variance)
        reachable = set(torch.where(graph.anchor)[0].tolist())
        queue = list(reachable)
        for node in queue:
            for neighbor in adjacency[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor); queue.append(neighbor)
        reachables.append(np.array([int(n) in reachable for n in graph.agents], dtype=bool))
        pairs.append(model_pairs); incoming.append(slots); directed.append(arcs); locals_.append(_norm(local))
    elements = len(union)*p*p if iterations else 0
    if elements > max_kernel_elements:
        raise ValueError('Full reference kernels exceed the explicit element budget')
    kernels = {}
    if iterations:
        for edge, (u, v, z, variance) in union.items():
            distance = np.linalg.norm(points[u, :, None, :]-points[v, None, :, :], axis=-1)
            kernel = -.5*(distance-z)**2/variance
            if not np.isfinite(kernel).all():
                raise FloatingPointError('Nonfinite pair likelihood')
            kernels[edge] = kernel
    messages = [np.full((2*len(edges), p), -math.log(p)) for edges in pairs]
    residuals = [None if edges else 0. for edges in pairs]
    local = np.stack(locals_)

    def states():
        out = {}
        for row, mid in enumerate(problem.model_ids):
            belief = local[row].copy()
            for node, slots in enumerate(incoming[row]):
                for slot in slots:
                    belief[node] += messages[row][slot]
            weights = np.exp(_norm(belief))
            mean = np.einsum('ap,api->ai', weights, points)
            centered = points-mean[:, None, :]
            covariance = np.einsum('ap,api,apj->aij', weights, centered, centered)
            out[mid] = dict(mean=mean, covariance=covariance, weights=weights,
                            log_messages=messages[row].copy(), ess=1/(weights**2).sum(axis=1),
                            anchor_reachable=reachables[row].copy(),
                            directed_edge_index=np.asarray(directed[row], dtype=np.int64).reshape(-1, 2).T,
                            message_edge_to_measurement=np.repeat([e for _, _, e in pairs[row]], 2),
                            message_residual=residuals[row])
        return out

    history = [states()] if return_history else None
    for round_id in range(iterations):
        updated = [x.copy() for x in messages]
        for row, model_pairs in enumerate(pairs):
            for k, (u, v, edge) in enumerate(model_pairs):
                cavity_u, cavity_v = local[row, u].copy(), local[row, v].copy()
                # Explicitly exclude the receiving factor, never subtract two
                # nearly equal accumulated log values; no cross-model state.
                for slot in incoming[row][u]:
                    if slot != 2*k+1: cavity_u += messages[row][slot]
                for slot in incoming[row][v]:
                    if slot != 2*k: cavity_v += messages[row][slot]
                updated[row][2*k] = _norm(_lse(kernels[edge]+_norm(cavity_u)[:, None], axis=0))
                updated[row][2*k+1] = _norm(_lse(kernels[edge]+_norm(cavity_v)[None, :], axis=1))
            if model_pairs:
                residuals[row] = float(np.max(np.abs(updated[row]-messages[row])))
        messages = updated                        # all models advance synchronously
        if return_history:
            history.append(states())
    final = states()
    for row, mid in enumerate(problem.model_ids):
        final[mid]['iterations'] = iterations if pairs[row] else 0
        final[mid]['converged'] = residuals[row] is not None and residuals[row] <= 1e-8
    # No-AA models have no message rounds in the accepted implementation.
    if return_history:
        for row, mid in enumerate(problem.model_ids):
            if not pairs[row]:
                for step in history[1:]: step.pop(mid)
    return dict(models=final, history=history,
                workload=dict(models=m, union_agent_edges=len(union),
                              shared_pair_distances=elements,
                              separate_pair_distances=sum(map(len, pairs))*iterations*p*p,
                              message_directions_integrated=2*sum(map(len, pairs))*iterations,
                              cpu_reference_only=True, gpu_executed=False))
