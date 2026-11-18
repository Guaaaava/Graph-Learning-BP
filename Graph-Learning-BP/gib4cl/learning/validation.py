"""Fixed per-graph Bernoulli validation, separate from deployment and BP draws."""
from __future__ import annotations

import hashlib
import math
import torch

from gib4cl.config import canonical_json
from .model import bernoulli_sigmoid
from .objectives import gib_objective


VALIDATION_VERSION = 'gib4cl.validation.v1'


def stream_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical_json(parts).encode()).digest()[:8], 'big') & ((1 << 63)-1)


def uniform_noise(shape, dtype, seed):
    """CPU draws with an independent generator; reject zero rather than clip."""
    generator = torch.Generator().manual_seed(seed)
    value = torch.rand(shape, generator=generator, dtype=dtype)
    zero = value == 0
    while zero.any():
        value[zero] = torch.rand((int(zero.sum()),), generator=generator, dtype=dtype)
        zero = value == 0
    return value


def validation_protocol(config, num_graphs):
    return {'version': VALIDATION_VERSION,
            'selection': 'minimum_mc_mean_total', 'tie_break': 'earliest_epoch',
            'split': 'val', 'num_graphs': num_graphs, 'full_loaded_split': True,
            'repeats': config.training.validation_repeats,
            'seed': config.training.validation_seed,
            'sampling': 'cpu_uniforms_sha256_validation_seed_sample_id_repeat',
            'aggregation': 'mean_repeats_within_graph_then_equal_mean_graphs',
            'deterministic_monitor_threshold': config.evaluation.edge_threshold,
            'deterministic_monitor_selects_checkpoint': False}


@torch.no_grad()
def validation_probe(model, data, objective, *, repeats, seed, threshold=.5):
    """Score one graph once, average hard masks, and separately monitor deployment.

    The namespace and uniforms match the original standalone diagnostic. Fixed
    uniforms across epochs reduce comparison noise; they are not fresh draws
    independent of a model selected using this validation sample.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('validation repeats must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('validation seed must be an integer in [0,2**63)')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError('validation threshold must be in (0,1)')
    sample_id = getattr(data, 'sample_id', None)
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError('validation requires one graph with a nonempty sample_id')
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        logits = model.score_edges(data)
        rows = []
        for repeat in range(repeats):
            u = uniform_noise(logits.shape, logits.dtype,
                              stream_seed('validation', seed, sample_id, repeat)).to(logits.device)
            mask = bernoulli_sigmoid(logits, tau=1., training=True, uniform=u)
            rows.append(gib_objective(logits, mask, data, objective=objective).metrics())
        mask = (logits > math.log(threshold/(1-threshold))).to(logits.dtype)
        deterministic = gib_objective(logits, mask, data, objective=objective).metrics()
        totals = [row['total'] for row in rows]
        return {'random_totals': totals, 'random_mean': math.fsum(totals)/repeats,
                'random_metrics': rows, 'deterministic_total': deterministic['total'],
                'deterministic_metrics': deterministic, 'threshold': threshold}
    finally:
        for module, training in modes.items():
            module.training = training


@torch.no_grad()
def validation_epoch(model, loader, objective, device, *, repeats, seed, threshold=.5):
    """Average graphs equally, regardless of node counts or loader batch sizes."""
    random_rows, deterministic_rows, prefixes = [], [], []
    prefix_repeats = max(1, repeats//2)
    for batch in loader:
        # Evaluate each graph independently so batch boundaries do not affect
        # GNN arithmetic or the per-sample random stream.
        for graph in batch.to_data_list():
            graph = graph.to(device)
            result = validation_probe(model, graph, objective, repeats=repeats,
                                      seed=seed, threshold=threshold)
            rows = result['random_metrics']
            random_rows.append({key: math.fsum(row[key] for row in rows)/repeats
                                for key in rows[0]})
            deterministic_rows.append(result['deterministic_metrics'])
            prefixes.append(math.fsum(result['random_totals'][:prefix_repeats])/prefix_repeats)
    if not random_rows:
        raise ValueError('A validation split must contain at least one graph')
    def average(rows):
        return {key: math.fsum(row[key] for row in rows)/len(rows) for key in rows[0]}
    random, deterministic = average(random_rows), average(deterministic_rows)
    prefix_total = math.fsum(prefixes)/len(prefixes)
    return random, deterministic, {'repeats': repeats, 'prefix_repeats': prefix_repeats,
                                   'prefix_total': prefix_total,
                                   'prefix_minus_selected_total': prefix_total-random['total']}
