"""The graph contract and input-only derived quantities."""

from __future__ import annotations

import math
import torch

from gib4cl.config import MEASUREMENT_ENCODING, config_from_dict, fingerprint, sample_seed, _integer

# Persisted-file format identifiers checked by readers, not experiment settings.
GRAPH_VERSION = 'gib4cl.graph.v1'
DATASET_VERSION = 'gib4cl.dataset.v1'
POSITIVE_FINITE_FLOAT32 = 255 * (1 << 23) - 1
SIGNED_FINITE_FLOAT32 = 2 * POSITIVE_FINITE_FLOAT32 + 1
LOG_MEASUREMENT_CARDINALITY = math.log(SIGNED_FINITE_FLOAT32) + math.log(POSITIVE_FINITE_FLOAT32)
INPUT_FIELDS = {'initial_positions', 'initial_covariances', 'is_anchor', 'edge_index',
                'measurements', 'measurement_variances', 'communication_cost_scale',
                'measurement_encoding'}


# DATA FLOW
# 1. `generate_graph()` → a record
# 2. pack up many records → bundle (dataset)
# 3. `validate_bundle()` → check whether figerprint, configuration are consistent
# 4. when training: get graph (record) from bundle; use `input_features()` to get feature `x`, `edge_attr`; input GNN for training

# Data contract: ranging values, variance allowed value space (float32 finite numbers, variance must be positive)
def representation_metadata():
    """Numerical values, not NaN/Inf encodings; signed zero is canonicalized."""
    return {'name': MEASUREMENT_ENCODING, 'range_value_count': SIGNED_FINITE_FLOAT32,
            'variance_value_count': POSITIVE_FINITE_FLOAT32,
            'joint_value_count': SIGNED_FINITE_FLOAT32 * POSITIVE_FINITE_FLOAT32,
            'log_joint_value_count': LOG_MEASUREMENT_CARDINALITY,
            'log_base': 'e', 'signed_zero': 'canonical positive zero',
            'range_clipping': False, 'nonfinite_or_nonpositive_variance': 'reject'}


# Measurement check: must be finite binary32; variances must also be strictly positive
def encode_measurements(values, *, positive=False):
    values = torch.as_tensor(values, dtype=torch.float32).clone()
    values[values == 0] = 0.0
    if not torch.isfinite(values).all() or (positive and not (values > 0).all()):
        raise ValueError('Measurements must be finite binary32; variances must also be strictly positive')
    return values


# Convert covariance into standard (A,2,2) tensor, validate finite, symmetric and positive definite (via Cholesky)
def normalize_covariances(value, num_agents):
    """Accept scalar, length-2 diagonal, shared 2x2, or per-agent (A,2,2)."""
    cov = torch.as_tensor(value, dtype=torch.float32)
    if cov.ndim == 0:
        cov = torch.eye(2, dtype=torch.float32) * cov
    elif cov.shape == (2,):
        cov = torch.diag(cov)
    if cov.shape == (2, 2):
        cov = cov.unsqueeze(0).repeat(num_agents, 1, 1)
    if cov.shape != (num_agents, 2, 2):
        raise ValueError('Covariance must be scalar, length-2, (2,2), or (A,2,2)')
    if not torch.isfinite(cov).all() or not torch.equal(cov, cov.transpose(-1, -2)):
        raise ValueError('Initial covariances must be finite and symmetric')
    if num_agents and torch.linalg.cholesky_ex(cov).info.ne(0).any():
        raise ValueError('Initial covariances must be positive definite')
    return cov.clone()


# Tensor verification: type, shape, no NaN/Inf
def _tensor(value, name, shape, dtype):
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must be {dtype} with shape {tuple(shape)}')
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f'{name} contains NaN or Inf')


# Check record, record = single sample output by `generate_graph()`
def validate_record(record, *, require_targets=False):
    if not isinstance(record, dict) or record.get('schema_version') != GRAPH_VERSION:
        raise ValueError(f'Expected {GRAPH_VERSION}; legacy input requires explicit migration')
    if not isinstance(record.get('sample_id'), str) or not record['sample_id']:
        raise ValueError('sample_id must be a nonempty string')
    inputs = record.get('inputs')
    if not isinstance(inputs, dict) or set(inputs) != INPUT_FIELDS:
        raise ValueError(f'inputs must have exactly these fields: {sorted(INPUT_FIELDS)}')
    pos = inputs['initial_positions']
    if not isinstance(pos, torch.Tensor) or pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError('initial_positions must have shape (N,2)')
    n = pos.shape[0]
    _tensor(pos, 'initial_positions', (n, 2), torch.float32)
    anchor = inputs['is_anchor']
    _tensor(anchor, 'is_anchor', (n,), torch.bool)
    a = int((~anchor).sum())
    if a < 1 or anchor[:a].any() or not anchor[a:].all():
        raise ValueError('Require at least one agent, followed by zero or more anchors')
    cov = inputs['initial_covariances']
    _tensor(cov, 'initial_covariances', (a, 2, 2), torch.float32)
    normalize_covariances(cov, a)
    ei = inputs['edge_index']
    if not isinstance(ei, torch.Tensor) or ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError('edge_index must have shape (2,E), including (2,0) for an empty graph')
    e = ei.shape[1]
    _tensor(ei, 'edge_index', (2, e), torch.long)
    tensors = (pos, anchor, cov, ei, inputs['measurements'], inputs['measurement_variances'])
    if any(not isinstance(t, torch.Tensor) or t.device != pos.device for t in tensors):
        raise ValueError('All input tensors must share a device')
    _tensor(inputs['measurements'], 'measurements', (e,), torch.float32)
    _tensor(inputs['measurement_variances'], 'measurement_variances', (e,), torch.float32)
    if not (inputs['measurement_variances'] > 0).all():
        raise ValueError('Measurement variances must be strictly positive')
    z = inputs['measurements']
    if ((z == 0) & torch.signbit(z)).any():
        raise ValueError('Negative zero must be canonicalized before saving')
    if inputs['measurement_encoding'] != MEASUREMENT_ENCODING:
        raise ValueError('Unsupported measurement encoding')
    rho = inputs['communication_cost_scale']
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0:
        raise ValueError('communication_cost_scale must be finite and positive')
    if e:
        u, v = ei
        if (u < 0).any() or (v >= n).any() or (u >= v).any() or (u >= a).any():
            raise ValueError('Require canonical global u<v, valid node indices, and no anchor-anchor edges')
        keys = u * n + v
        if torch.unique(keys).numel() != e:
            raise ValueError('Each physical measurement edge must appear once')
        dbar = (pos[u] - pos[v]).norm(dim=1)
        if not torch.isfinite(dbar).all() or (dbar <= 0).any():
            raise ValueError('Connected initial coordinates must be distinct with finite distance')
        cost = rho * dbar.square()
        residual = z - dbar
        if not torch.isfinite(cost).all() or not torch.isfinite(residual).all():
            raise ValueError('Derived cost/residual is not representable as finite binary32')
    targets = record.get('targets')
    if require_targets and targets is None:
        raise ValueError('Position targets are required for this dataset')
    if targets is not None:
        if not isinstance(targets, dict) or set(targets) != {'agent_positions'}:
            raise ValueError('targets must contain only agent_positions')
        _tensor(targets['agent_positions'], 'targets.agent_positions', (a, 2), torch.float32)
        if targets['agent_positions'].device != pos.device:
            raise ValueError('Targets and inputs must share a device')
    metadata = record.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be an object')
    if metadata.get('measurement_model', 'los_gaussian') != 'los_gaussian':
        raise ValueError('Only LOS graphs are supported; measurement_model must be los_gaussian')
    diagnostics = record.get('diagnostics', {})
    if not isinstance(diagnostics, dict):
        raise ValueError('diagnostics must be an object')
    return record


# Construct all features from C, D. This function cannot access targets
def input_features(inputs):
    pos, cov, anchor = (inputs[k] for k in ('initial_positions', 'initial_covariances', 'is_anchor'))
    a = cov.shape[0]
    variances = pos.new_zeros((pos.shape[0], 2))
    variances[:a] = cov.diagonal(dim1=-2, dim2=-1)
    x = torch.cat((pos, variances, anchor[:, None].to(pos.dtype)), dim=1)
    u, v = inputs['edge_index']
    diff = pos[u] - pos[v]
    dbar = diff.norm(dim=1)
    residual = inputs['measurements'] - dbar
    anchor_edge = anchor[v]
    edge_attr = torch.stack((inputs['measurements'], inputs['measurement_variances'],
                             residual.abs(), anchor_edge.to(pos.dtype)), dim=1)
    return {'x': x, 'edge_attr': edge_attr, 'initial_distances': dbar,
            'signed_residual': residual, 'edge_directions': diff / dbar[:, None],
            'edge_cost': inputs['communication_cost_scale'] * dbar.square(),
            'edge_log_cardinality': pos.new_full((u.numel(),), LOG_MEASUREMENT_CARDINALITY)}


# Check bundle, bundle = dataset container (dict)
def validate_bundle(bundle, *, require_targets=True):
    """Check configuration, split/sample identities, cardinality and graph contents."""
    if not isinstance(bundle, dict) or bundle.get('schema_version') != DATASET_VERSION:
        raise ValueError(f'Expected {DATASET_VERSION}')
    metadata, graphs = bundle.get('metadata'), bundle.get('graphs')
    if not isinstance(metadata, dict) or not isinstance(graphs, list):
        raise ValueError('Dataset requires a metadata object and graph list')
    required = {'config', 'data_fingerprint', 'split', 'seed_offset', 'agents_min',
                'agents_max', 'experiment_id', 'series_id', 'dataset_id', 'num_graphs',
                'representation'}
    if not required.issubset(metadata):
        raise ValueError('Incomplete dataset metadata')
    cfg = config_from_dict(metadata['config'])
    if cfg.data_fingerprint != metadata['data_fingerprint'] or cfg.experiment_id != metadata['experiment_id']:
        raise ValueError('Dataset configuration fingerprint/experiment ID mismatch')
    if metadata['representation'] != representation_metadata():
        raise ValueError('Dataset measurement representation mismatch')
    _integer(metadata['num_graphs'], 'num_graphs', 0)
    _integer(metadata['agents_min'], 'agents_min')
    _integer(metadata['agents_max'], 'agents_max')
    _integer(metadata['seed_offset'], 'seed_offset', 0)
    if len(graphs) != metadata['num_graphs'] or metadata['agents_min'] > metadata['agents_max']:
        raise ValueError('Dataset graph count or agent interval mismatch')
    identity = {key: metadata[key] for key in ('experiment_id', 'data_fingerprint', 'split',
                                              'seed_offset', 'agents_min', 'agents_max')}
    series = fingerprint(identity)
    if series != metadata['series_id'] or metadata['dataset_id'] != fingerprint({'series_id': series, 'num_graphs': len(graphs)}):
        raise ValueError('Dataset identity mismatch')
    for i, graph in enumerate(graphs):
        validate_record(graph, require_targets=require_targets)
        info = graph.get('metadata', {})
        if graph['sample_id'] != f'{series}:{i}':
            raise ValueError('Sample identity/order mismatch')
        expected_seed = sample_seed(cfg.seeds.data, metadata['split'], metadata['seed_offset'], i)
        if (info.get('sample_seed') != expected_seed or info.get('sample_number') != i
                or info.get('split') != metadata['split'] or info.get('series_id') != series
                or info.get('data_fingerprint') != cfg.data_fingerprint):
            raise ValueError('Sample generation metadata mismatch')
        if fingerprint(info.get('data_config')) != fingerprint(metadata['config']['data']):
            raise ValueError('Sample data configuration mismatch')
        a = graph['inputs']['initial_covariances'].shape[0]
        if not metadata['agents_min'] <= a <= metadata['agents_max']:
            raise ValueError('Sample agent count outside resolved split interval')
        positions = graph['inputs']['initial_positions']
        expected_anchors = positions.new_tensor(cfg.data.anchor_coordinates)
        if positions[a:].shape != expected_anchors.shape or not torch.equal(positions[a:], expected_anchors):
            raise ValueError('Dataset requires the fixed five-anchor geometry')
        targets = graph.get('targets')
        if targets is not None:
            truth = targets['agent_positions']
            if ((truth < 0) | (truth > cfg.data.area_size)).any():
                raise ValueError('Dataset agent positions must be inside the configured square')
    return bundle


# For the same ranging observation, bidirectional messages share the same observation, and the ranging will not be calculated twice
def message_edges(edge_index):
    """First E forward arcs, then E reverse arcs; one shared measurement ID."""
    e = edge_index.shape[1]
    ids = torch.arange(e, device=edge_index.device)
    return (torch.cat((edge_index, edge_index.flip(0)), dim=1),
            torch.cat((ids, ids)), torch.cat((ids + e, ids)))


# Find agent triangles (i,j,k), and output the IDs of the three sides of the triangle
def agent_triangles(edge_index, num_agents):
    edge_lookup = {tuple(edge): i for i, edge in enumerate(edge_index.t().tolist())
                   if edge[1] < num_agents}
    neighbors = [set() for _ in range(num_agents)]
    for u, v in edge_lookup:
        neighbors[u].add(v)
        neighbors[v].add(u)
    triples = []
    for (i, j), edge_id in edge_lookup.items():
        for k in sorted(neighbors[i] & neighbors[j]):
            if k > j:
                triples.append((edge_id, edge_lookup[(j, k)], edge_lookup[(i, k)]))
    return torch.tensor(triples, dtype=torch.long, device=edge_index.device).reshape(-1, 3).t().contiguous()


# Count connectivity and positioning feasibility indicators of each node in the graph
def graph_diagnostics(inputs):
    """Topology diagnostics; never reject or repair unanchored components."""
    anchor, ei = inputs['is_anchor'], inputs['edge_index']
    n, a = anchor.numel(), inputs['initial_covariances'].shape[0]
    adjacency = [set() for _ in range(n)]
    for u, v in ei.t().tolist():
        adjacency[u].add(v)
        adjacency[v].add(u)
    visited = set(range(a, n))
    queue = list(visited)
    for u in queue:
        for v in sorted(adjacency[u]):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    reachable = torch.tensor([i in visited for i in range(a)], dtype=torch.bool, device=ei.device)
    degree = torch.bincount(ei.flatten(), minlength=n)[:a]
    return {'agent_anchor_reachable': reachable, 'candidate_agent_degree': degree,
            'num_unanchored_agents': int((~reachable).sum()),
            'num_agents_below_degree_three': int((degree < 3).sum())}


# prior: Initial position prior
def prior_precision_blocks(inputs):
    """Physical prior precision, never an arbitrary diagonal regularizer."""
    return torch.linalg.inv(inputs['initial_covariances'])


