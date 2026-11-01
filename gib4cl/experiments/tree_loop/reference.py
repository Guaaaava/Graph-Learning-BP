"""Independent NumPy joint marginalization, not BP or a linearized surrogate.

For three variables, the joint mass w0*w1*w2*K01*K12*K02 contracts with two
matrix multiplications. Memory is O(P^2), time O(P^3). All three physical
likelihoods enter exactly once. No production inference helpers are imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import numpy as np


@dataclass
class Posterior:
    mean: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    log_partition_relative: float


def exact_marginals(points, log_prior_mass, measurements, variances, mask, *, max_particles=2048, edges=None):
    if edges is not None:
        return exact_unicyclic(points, log_prior_mass, measurements, variances, edges, mask,
                               max_particles=max_particles)
    points = np.asarray(points, dtype=np.float64)
    logs = np.asarray(log_prior_mass, dtype=np.float64)
    if points.ndim != 3 or points.shape[0] != 3 or points.shape[2] != 2:
        raise ValueError('Expected points=(3,P,2)')
    p = points.shape[1]
    if p < 2 or p > max_particles:
        raise ValueError(f'Exact contraction requires 2 <= P <= {max_particles}')
    if logs.shape != (3, p) or not np.isfinite(points).all() or np.isnan(logs).any() or np.isposinf(logs).any() or not np.isfinite(logs).any(axis=1).all():
        raise ValueError('Invalid prior support/masses')
    z, variance, mask = (np.asarray(v) for v in (measurements, variances, mask))
    if z.shape != (3,) or variance.shape != (3,) or mask.shape != (3,) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected three observations, variances and binary mask values')
    if not np.isfinite(z).all() or not np.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError('Invalid observations/variances')
    shifts = logs.max(axis=1)
    roots = np.exp(.5*(logs-shifts[:, None]))
    scale = float(shifts.sum())
    factors = []
    for edge, (u, v) in enumerate(((0, 1), (1, 2), (0, 2))):
        if mask[edge]:
            difference = points[u, :, None, :]-points[v, None, :, :]
            kernel = -.5*(np.sqrt(np.sum(difference*difference, axis=2))-z[edge])**2/variance[edge]
            shift = float(kernel.max())
            if not math.isfinite(shift):
                raise FloatingPointError('Nonfinite reference likelihood')
            scale += shift
            kernel = np.exp(kernel-shift)
        else:
            kernel = np.ones((p, p), dtype=np.float64)
        factors.append(roots[u, :, None]*kernel*roots[v, None, :])
    left, middle, closure = factors
    joint_02 = (left @ middle)*closure
    m0, m2 = joint_02.sum(axis=1), joint_02.sum(axis=0)
    m1 = ((left.T @ closure)*middle).sum(axis=1)
    mass = float(m0.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise FloatingPointError('Reference joint mass underflow; no floor/jitter is applied')
    weights = np.stack((m0, m1, m2))/mass
    if not np.allclose(weights.sum(axis=1), 1., rtol=2e-12, atol=2e-14):
        raise FloatingPointError('Inconsistent exact marginal normalizers')
    weights /= weights.sum(axis=1, keepdims=True)
    mean = np.einsum('ap,api->ai', weights, points)
    centered = points-mean[:, None, :]
    covariance = np.einsum('ap,api,apj->aij', weights, centered, centered)
    return Posterior(mean, covariance, weights, math.log(mass)+scale)


def exact_unicyclic(points, log_prior_mass, measurements, variances, edges, mask, *, max_particles=2048):
    """Exact finite-support marginals of a forest with at most one cycle/component.

    Sum out leaves, contract the remaining cycle using scaled prefix/suffix
    products, then recover branch marginals by separator conditioning. The
    independent NumPy solver costs O(N*P^3), never P**N, and applies each
    physical likelihood exactly once. It does not call production BP.
    """
    points, logs = np.asarray(points, dtype=np.float64), np.asarray(log_prior_mass, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2 or points.shape[0] < 1:
        raise ValueError('Expected points=(N,P,2)')
    n, p = points.shape[:2]
    if p < 2 or p > max_particles or logs.shape != (n, p):
        raise ValueError('Reference support shape/particle budget mismatch')
    if (not np.isfinite(points).all() or np.isnan(logs).any() or np.isposinf(logs).any()
            or not np.isfinite(logs).any(axis=1).all()):
        raise ValueError('Invalid finite prior support/masses')
    edges, z, variance, mask = (np.asarray(v) for v in (edges, measurements, variances, mask))
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or z.shape != (len(edges),) or variance.shape != z.shape or mask.shape != z.shape):
        raise ValueError('Expected edges=(E,2), observations/variances/mask=(E,)')
    if (not np.isfinite(z).all() or not np.isfinite(variance).all() or not (variance > 0).all()
            or not np.isin(mask, [0, 1]).all()):
        raise ValueError('Invalid observations/variances/binary mask')
    canonical = [tuple(sorted(map(int, edge))) for edge in edges]
    if any(u < 0 or v >= n or u == v for u, v in canonical) or len(set(canonical)) != len(edges):
        raise ValueError('Physical edges must be unique, distinct and in range')
    adjacency = [set() for _ in range(n)]
    for active, (u, v) in zip(mask, canonical):
        if active:
            adjacency[u].add(v)
            adjacency[v].add(u)
    unseen = set(range(n))
    while unseen:
        component, queue = set(), [min(unseen)]
        for u in queue:
            if u in component:
                continue
            component.add(u)
            queue.extend(v for v in adjacency[u] if v not in component)
        unseen -= component
        if sum(len(adjacency[u]) for u in component)//2 > len(component):
            raise ValueError('Exact diagnostic reference supports trees and single-cycle components only')
    shifts = logs.max(axis=1)
    unary = np.exp(logs-shifts[:, None])
    scale = float(shifts.sum())
    kernels = {}
    for edge, (u, v) in enumerate(canonical):
        if not mask[edge]:
            continue
        difference = points[u, :, None]-points[v, None, :]
        log_kernel = -.5*(np.linalg.norm(difference, axis=2)-z[edge])**2/variance[edge]
        shift = float(log_kernel.max())
        if not math.isfinite(shift):
            raise FloatingPointError('Nonfinite reference likelihood')
        kernels[u, v] = np.exp(log_kernel-shift)
        scale += shift

    def kernel(u, v):
        return kernels[u, v] if u < v else kernels[v, u].T

    def normalize(value, total=False):
        constant = float(value.sum() if total else value.max())
        if not math.isfinite(constant) or constant <= 0:
            raise FloatingPointError('Reference mass underflow; no floor or jitter is applied')
        return value/constant, math.log(constant)

    weights = np.zeros((n, p), dtype=np.float64)
    remaining = set(range(n))
    leaves = deque(u for u in range(n) if len(adjacency[u]) <= 1)
    removed = []
    while leaves:
        u = leaves.popleft()
        if u not in remaining:
            continue
        if not adjacency[u]:
            weights[u], log_z = normalize(unary[u], total=True)
            scale += log_z
        else:
            v = next(iter(adjacency[u]))
            message, log_z = normalize(unary[u] @ kernel(u, v))
            scale += log_z
            removed.append((u, v, unary[u].copy(), message))
            unary[v], log_z = normalize(unary[v]*message)
            scale += log_z
            adjacency[v].remove(u)
            if len(adjacency[v]) <= 1:
                leaves.append(v)
        remaining.remove(u)
        adjacency[u].clear()
    while remaining:
        start = min(remaining)
        cycle, previous, current = [], None, start
        while True:
            cycle.append(current)
            if len(adjacency[current]) != 2:
                raise ValueError('Reference core is not a simple cycle')
            following = min(v for v in adjacency[current] if v != previous)
            previous, current = current, following
            if current == start:
                break
            if current in cycle:
                raise ValueError('Invalid cycle traversal')
        factors = []
        for i, u in enumerate(cycle):
            v = cycle[(i+1) % len(cycle)]
            factors.append(np.sqrt(unary[u])[:, None]*kernel(u, v)*np.sqrt(unary[v])[None, :])
        prefix = [np.eye(p)]
        log_prefix = 0.
        for index, factor in enumerate(factors):
            value, log_z = normalize(factor if index == 0 else prefix[-1] @ factor)
            prefix.append(value)
            log_prefix += log_z
        suffix = [None]*len(cycle)+[np.eye(p)]
        for i in range(len(cycle)-1, -1, -1):
            suffix[i], _ = normalize(factors[i] if i == len(cycle)-1 else factors[i] @ suffix[i+1])
        trace = float(np.trace(prefix[-1]))
        if not math.isfinite(trace) or trace <= 0:
            raise FloatingPointError('Reference cycle partition underflow')
        scale += log_prefix+math.log(trace)
        for i, u in enumerate(cycle):
            weights[u], _ = normalize(np.sum(suffix[i]*prefix[i].T, axis=1), total=True)
        remaining -= set(cycle)
    for u, v, subtree, message in reversed(removed):
        positive = (weights[v] > 0) & (message > 0)
        if ((weights[v] > 0) & (message == 0)).any() or not positive.any():
            raise FloatingPointError('Inconsistent separator mass in exact reference')
        cavity = np.zeros(p)
        log_cavity = np.log(weights[v, positive])-np.log(message[positive])
        cavity[positive] = np.exp(log_cavity-log_cavity.max())
        weights[u], _ = normalize(subtree*(kernel(u, v) @ cavity), total=True)
    mean = np.einsum('np,npi->ni', weights, points)
    centered = points-mean[:, None]
    covariance = np.einsum('np,npi,npj->nij', weights, centered, centered)
    return Posterior(mean, covariance, weights, scale)


def hermite_support(centers, prior_std, order):
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError('Quadrature order must be an integer >= 2')
    nodes, masses = np.polynomial.hermite.hermgauss(order)
    xx, yy = np.meshgrid(nodes, nodes, indexing='ij')
    offsets = math.sqrt(2)*prior_std*np.stack((xx.ravel(), yy.ravel()), axis=1)
    points = np.asarray(centers, dtype=np.float64)[:, None, :]+offsets
    logs = np.log(np.outer(masses, masses).ravel()/math.pi)
    return points, np.broadcast_to(logs, points.shape[:2]).copy()


def continuous_reference(condition, trial, topologies, orders, tolerance, max_particles=2048, edges=None):
    """Require TWO consecutive refinements of means and directional variances.

    This is numerical stability evidence, not a rigorous integration error bound.
    Nonconvergence is retained, never relabelled an exact continuous posterior.
    """
    previous, stable, records, final = {}, 0, [], {}
    for order in orders:
        points, logs = hermite_support(condition.centers, condition.prior_std, order)
        current = {name: exact_marginals(points, logs, trial.measurements, trial.variances, mask,
                                         max_particles=max_particles, edges=edges) for name, mask in topologies.items()}
        changes = {}
        if previous:
            for name, value in current.items():
                old = previous[name]
                # Generalized covariance difference checks all directions, not
                # only a Frobenius norm that can hide a small eigen-direction.
                relative = []
                for new_cov, old_cov in zip(value.covariance, old.covariance):
                    chol = np.linalg.cholesky(new_cov)
                    scaled = np.linalg.solve(chol, old_cov-new_cov)
                    scaled = np.linalg.solve(chol, scaled.T).T
                    relative.append(float(np.abs(np.linalg.eigvalsh((scaled+scaled.T)/2)).max()))
                changes[name] = {'covariance_relative_max': max(relative),
                                 'mean_change_in_prior_std': float(np.abs(value.mean-old.mean).max()/condition.prior_std)}
            passed = all(max(v.values()) <= tolerance for v in changes.values())
            stable = stable+1 if passed else 0
        records.append({'order': order, 'points_per_node': order**2, 'changes': changes})
        final = current
        if stable >= 2:
            break
        previous = current
    return {'converged': stable >= 2, 'consecutive_stable_refinements': stable,
            'tolerance': tolerance, 'refinements': records, 'final_order': records[-1]['order']}, points, final
