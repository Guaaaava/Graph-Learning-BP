"""Independent CPU references for tiny localization problems.

No production BP helpers are imported. NumPy joint enumeration sums the entire
finite posterior, including loops. Quadrature and radial integration separately
check the continuous nonlinear range model, not a linearized/FIM surrogate.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ReferenceResult:
    mean: np.ndarray
    covariance: np.ndarray
    marginals: np.ndarray
    log_partition_relative: float
    joint_states: int


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def exact_discrete(data, points, log_prior_weights, edge_weights=None, *,
                   max_states=2000000, chunk_size=65536):
    """Enumerate all P**A configurations, with bounded working memory.

    log_prior_weights are already prior integration masses (not densities unless
    accompanied by integration weights). Gaussian likelihood constants depending
    only on known edge variance are omitted; log_partition_relative is therefore
    not a normalized Gaussian model's evidence. Marginals/moments are unchanged.
    """
    x, edges, features = (_numpy(value) for value in (data.x, data.edge_index, data.edge_attr))
    points, base = _numpy(points).astype(np.float64), _numpy(log_prior_weights).astype(np.float64)
    agents = np.flatnonzero(x[:, 4] == 0)
    if points.ndim != 3 or points.shape[0] != len(agents) or points.shape[2] != 2 or base.shape != points.shape[:2]:
        raise ValueError('Expected points=(A,P,2) and log_prior_weights=(A,P)')
    if not np.isfinite(points).all() or np.isnan(base).any() or np.isposinf(base).any() or not np.isfinite(base).any(axis=1).all():
        raise ValueError('Invalid reference support or prior masses')
    a, p = base.shape
    states = p ** a
    if states > max_states:
        raise ValueError(f'Exact reference needs {states} states, above budget {max_states}')
    if chunk_size < 1 or max_states < 1:
        raise ValueError('Reference budgets must be positive')
    active = np.ones(edges.shape[1], dtype=bool) if edge_weights is None else _numpy(edge_weights)
    if active.shape != (edges.shape[1],) or not np.isin(active, [0, 1]).all():
        raise ValueError('Reference mask must be binary and follow physical edges')
    node_to_agent = {int(node): row for row, node in enumerate(agents)}
    marginals = np.zeros((a, p), dtype=np.float64)
    accumulated_mass, scale = 0., -np.inf
    for start in range(0, states, chunk_size):
        numbers = np.arange(start, min(start + chunk_size, states), dtype=np.int64)
        assignment = np.empty((len(numbers), a), dtype=np.int64)
        for node in range(a):
            assignment[:, node] = numbers % p
            numbers = numbers // p
        log_joint = sum(base[node, assignment[:, node]] for node in range(a))
        for edge in np.flatnonzero(active):
            u, v = map(int, edges[:, edge])
            z, variance = map(float, features[edge, :2])
            if not np.isfinite(variance) or variance <= 0:
                raise ValueError('Measurement variance must be positive')
            left = points[node_to_agent[u], assignment[:, node_to_agent[u]]] if u in node_to_agent else x[u, :2].astype(np.float64)
            right = points[node_to_agent[v], assignment[:, node_to_agent[v]]] if v in node_to_agent else x[v, :2].astype(np.float64)
            distance = np.linalg.norm(left - right, axis=-1)
            log_joint = log_joint - .5 * (distance - z)**2 / variance
        block_max = np.max(log_joint)
        if not np.isfinite(block_max):
            continue
        new_scale = max(scale, block_max)
        old_multiplier = math.exp(scale - new_scale)
        block_weights = np.exp(log_joint - new_scale)
        marginals *= old_multiplier
        accumulated_mass *= old_multiplier
        for node in range(a):
            marginals[node] += np.bincount(assignment[:, node], weights=block_weights, minlength=p)
        accumulated_mass += block_weights.sum()
        scale = new_scale
    if not accumulated_mass > 0:
        raise FloatingPointError('No finite positive joint mass in reference')
    marginals /= accumulated_mass
    mean = np.einsum('ap,api->ai', marginals, points)
    centered = points - mean[:, None, :]
    covariance = np.einsum('ap,api,apj->aij', marginals, centered, centered)
    return ReferenceResult(mean, covariance, marginals, scale + math.log(accumulated_mass), states)


def gaussian_hermite_support(data, order):
    """2D tensor Gauss-Hermite quadrature of each full Gaussian prior."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError('Quadrature order must be >= 2')
    x, covariance = _numpy(data.x).astype(np.float64), _numpy(data.agent_prior_cov).astype(np.float64)
    agents = np.flatnonzero(x[:, 4] == 0)
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    u, v = np.meshgrid(nodes, nodes, indexing='ij')
    normal = math.sqrt(2) * np.stack([u.ravel(), v.ravel()], axis=1)
    masses = (weights[:, None] * weights[None, :] / math.pi).reshape(-1)
    points = np.stack([x[node, :2] + normal @ np.linalg.cholesky(covariance[row]).T
                       for row, node in enumerate(agents)])
    return points, np.broadcast_to(np.log(masses), points.shape[:2]).copy()


def grid_support(data, lower, upper, points_per_axis):
    """Trapezoidal 2D grids: Gaussian prior density * area quadrature weights."""
    x, covariance = _numpy(data.x).astype(np.float64), _numpy(data.agent_prior_cov).astype(np.float64)
    agents = np.flatnonzero(x[:, 4] == 0)
    lower, upper = np.broadcast_to(lower, (len(agents), 2)), np.broadcast_to(upper, (len(agents), 2))
    if not (upper > lower).all() or points_per_axis < 2:
        raise ValueError('Require ordered bounds and >=2 grid points per axis')
    points, masses = [], []
    for row, node in enumerate(agents):
        axes = [np.linspace(lower[row, axis], upper[row, axis], points_per_axis) for axis in range(2)]
        xx, yy = np.meshgrid(*axes, indexing='ij')
        coordinates = np.stack([xx.ravel(), yy.ravel()], axis=1)
        differences = coordinates - x[node, :2]
        prior_log = -.5 * np.einsum('pi,ij,pj->p', differences, np.linalg.inv(covariance[row]), differences)
        prior_log -= math.log(2 * math.pi) + .5 * np.linalg.slogdet(covariance[row])[1]
        one_dimensional = np.ones(points_per_axis)
        one_dimensional[[0, -1]] = .5
        area_weights = (one_dimensional[:, None] * one_dimensional[None, :]).ravel()
        area_weights *= np.prod((upper[row] - lower[row]) / (points_per_axis - 1))
        points.append(coordinates)
        masses.append(prior_log + np.log(area_weights))
    return np.stack(points), np.stack(masses)


def radial_single_anchor(prior_variance, measurement, measurement_variance, *,
                         order=256, radius=None):
    """Independent 1D radial integral: anchor and isotropic prior centered at 0.

    E[y]=0; Cov[y]=I*E[r**2]/2. The radial Jacobian r is essential.
    Finite-radius/order accuracy must be checked by refinement.
    """
    if prior_variance <= 0 or measurement_variance <= 0 or order < 2:
        raise ValueError('Require positive variances and order >= 2')
    if radius is None:
        radius = max(12 * math.sqrt(prior_variance),
                     abs(measurement) + 12 * math.sqrt(measurement_variance))
    if radius <= 0:
        raise ValueError('radius must be positive')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    r = (nodes + 1) * radius / 2
    log_mass = np.log(weights * radius / 2) + np.log(r)
    log_mass -= r*r / (2 * prior_variance) + (r - measurement)**2 / (2 * measurement_variance)
    relative = np.exp(log_mass - log_mass.max())
    second_radial_moment = np.sum(relative * r*r) / relative.sum()
    return {'mean': np.zeros(2), 'covariance': np.eye(2) * second_radial_moment / 2,
            'radius': radius, 'order': order}
