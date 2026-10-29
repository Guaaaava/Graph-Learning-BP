"""Static localization graph generation with the versioned global-index contract.

LOS follows the paper's Gaussian model without range clipping.
"""
from __future__ import annotations
from dataclasses import asdict
import torch

from gib4cl.config import DataConfig
from gib4cl.data.schema import (GRAPH_VERSION, encode_measurements, graph_diagnostics,
                               normalize_covariances, validate_record)

def anchor_positions(settings):
    return torch.tensor(settings.anchor_coordinates, dtype=torch.float32)


# generate_graph: generate one cooperative localization graph sample
# Agents at unknown location + known coordinate anchor, randomly placed within a square area, generates ranging edges and distance-related Gaussian ranging noise within the communication radius.
# Return packaged graph input + ground truth + metadata for GNN training/evaluation
def generate_graph(settings, num_agents, *, generator=None, sample_id='standalone', metadata=None):
    if not isinstance(settings, DataConfig):
        raise TypeError('settings must be a DataConfig')
    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents < 1:
        raise ValueError('num_agents must be a positive integer')
    anchors = anchor_positions(settings)
    truth = torch.rand((num_agents, 2), generator=generator, dtype=torch.float32) * settings.area_size
    cov = normalize_covariances(settings.initial_covariance, num_agents)
    noise = torch.randn((num_agents, 2, 1), generator=generator, dtype=torch.float32)
    initial = truth + (torch.linalg.cholesky(cov) @ noise).squeeze(-1) # By default σ_{0,i}=25I, so the initial standard deviation of error for each coordinate axis is 5m
    positions = torch.cat((initial, anchors))
    true_positions = torch.cat((truth, anchors))
    n = positions.shape[0]
    candidate = torch.triu_indices(n, n, offset=1)
    u, v = candidate
    distances = (true_positions[u] - true_positions[v]).norm(dim=1)
    keep = (u < num_agents) & (distances < settings.comm_radius)
    ei = candidate[:, keep].contiguous()
    d = distances[keep]
    std = settings.base_noise_std + settings.distance_noise_scale * d
    measurements = d + torch.randn(d.shape, generator=generator, dtype=torch.float32) * std
    inputs = {
        'initial_positions': positions,
        'initial_covariances': cov,
        'is_anchor': torch.arange(n) >= num_agents,
        'edge_index': ei,
        'measurements': encode_measurements(measurements),
        'measurement_variances': encode_measurements(std.square(), positive=True),
        'communication_cost_scale': settings.communication_cost_scale,
        'measurement_encoding': settings.measurement_encoding,
    }
    record = {
        'schema_version': GRAPH_VERSION, 'sample_id': sample_id, 'inputs': inputs,
        'targets': {'agent_positions': truth},
        'metadata': {'data_config': asdict(settings), 'units': {'position': 'm', 'variance': 'm^2'},
                     'measurement_model': 'los_gaussian',
                     'geometry': 'five_anchors_uniform_agents',
                     'variance_assumption': 'known variance from physical true distance',
                     'range_clipped': False},
        'diagnostics': graph_diagnostics(inputs),
    }
    if metadata:
        record['metadata'].update(metadata)
    return validate_record(record, require_targets=True) # record = single sample output by `generate_graph()`
