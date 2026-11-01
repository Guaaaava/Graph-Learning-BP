"""Prior-predictive trials; truth never enters the BP input graph."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import SimpleNamespace
import numpy as np
import torch

from gib4cl.inference import ParticleSupport

EDGES = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
MASKS = {'tree': (1, 1, 0), 'loop': (1, 1, 1),
         'tree_01_02': (1, 0, 1), 'tree_12_02': (0, 1, 1)}


@dataclass
class GraphSpec:
    edges: np.ndarray
    masks: dict
    labels: tuple
    focus: int
    cycle_lengths: dict


def graph_spec(condition):
    if condition.graph_family == 'triangle':
        return GraphSpec(EDGES.copy(), MASKS.copy(), ('1', '2', '3'), 1, {'loop': 3})
    edges = np.array([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6),
                      (1, 2), (2, 3), (3, 5)], dtype=np.int64)
    masks = {'tree': (1,)*6+(0,)*3}
    for edge, length in enumerate((3, 4, 5), start=6):
        mask = list(masks['tree'])
        mask[edge] = 1
        masks['loop'+str(length)] = tuple(mask)
    return GraphSpec(edges, masks, ('i', 'j', 'k', 'a', 'b', 'c', 'd'), 0,
                     {'loop3': 3, 'loop4': 4, 'loop5': 5})


def seed_for(base, *keys):
    data = json.dumps([int(base), *keys], separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') % 2**63


@dataclass
class Trial:
    graph: SimpleNamespace
    truth: np.ndarray
    measurements: np.ndarray
    variances: np.ndarray
    data_seed: int


def make_trial(condition, base_seed, case_index, phase='study'):
    # Common standard normal draws across conditions; independent across cases.
    seed = seed_for(base_seed, 'tree-loop-data', phase, case_index)
    rng = np.random.default_rng(seed)
    centers = np.asarray(condition.centers, dtype=np.float64)
    edges = graph_spec(condition).edges
    n, e = len(centers), len(edges)
    truth = centers + condition.prior_std*rng.standard_normal((n, 2))
    ranges = np.linalg.norm(truth[edges[:, 0]]-truth[edges[:, 1]], axis=1)
    measurements = ranges + condition.range_std*rng.standard_normal(e)
    variances = np.full(e, condition.range_std**2)
    x = torch.zeros((n, 5), dtype=torch.float64)
    x[:, :2] = torch.from_numpy(centers.copy())
    x[:, 2:4] = condition.prior_std**2
    attributes = torch.zeros((e, 4), dtype=torch.float64)
    attributes[:, 0] = torch.from_numpy(measurements.copy())
    attributes[:, 1] = torch.from_numpy(variances.copy())
    graph = SimpleNamespace(x=x, edge_index=torch.from_numpy(edges.T.copy()), edge_attr=attributes,
                            agent_node_index=torch.arange(n), num_graphs=1,
                            agent_prior_cov=torch.eye(2, dtype=torch.float64).repeat(n, 1, 1)*condition.prior_std**2)
    return Trial(graph, truth, measurements, variances, seed)


def prior_points(condition, base_seed, case_index, repeat, budget, phase='study'):
    seed = seed_for(base_seed, 'tree-loop-particles', phase, case_index, repeat)
    normal = np.random.default_rng(seed).standard_normal((len(condition.centers), budget, 2))
    return np.asarray(condition.centers)[:, None, :] + condition.prior_std*normal, seed


def as_support(points):
    points = np.ascontiguousarray(points, dtype=np.float64)
    p = points.shape[1]
    n = points.shape[0]
    return ParticleSupport(torch.from_numpy(points.copy()),
                           torch.full((n, p), -math.log(p), dtype=torch.float64), torch.arange(n))
