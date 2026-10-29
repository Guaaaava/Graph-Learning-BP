"""Versioned dataset loader and PyG adapter; physical edges remain unique."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from gib4cl.data.schema import (DATASET_VERSION, agent_triangles, input_features,
                               message_edges, prior_precision_blocks,
                               validate_record, validate_bundle)

# Convert the previously generated and saved `bundle` into a PyG standard `Data` object for direct training of GNN; `LocalizationData` is a custom PyG Data class, and `LocalizationDataset` is a PyG Dataset
class LocalizationData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # These fields index edges, not nodes.
        if key in ('message_edge_to_measurement', 'triangle_edge_index'):
            return self.edge_index.shape[1]
        if key == 'message_reverse_index':
            return self.message_edge_index.shape[1]
        return super().__inc__(key, value, *args, **kwargs)


def graph_to_data(record, *, include_targets=True):
    validate_record(record, require_targets=include_targets)
    inputs = record['inputs']
    features = input_features(inputs)
    a, n = inputs['initial_covariances'].shape[0], inputs['initial_positions'].shape[0]
    ei = inputs['edge_index']
    arcs, arc_to_edge, reverse = message_edges(ei)
    data = LocalizationData(
        **features, edge_index=ei.clone(),
        agent_prior_cov=inputs['initial_covariances'].clone(),
        agent_prior_precision=prior_precision_blocks(inputs),
        agent_node_index=torch.arange(a, device=ei.device),
        anchor_node_index=torch.arange(a, n, device=ei.device),
        message_edge_index=arcs, message_edge_to_measurement=arc_to_edge,
        message_reverse_index=reverse, triangle_edge_index=agent_triangles(ei, a),
        num_agents=a, num_anchors=n - a, sample_id=record['sample_id'],
    )
    if include_targets:
        data.y = record['targets']['agent_positions'].clone()
    return data


class LocalizationDataset(Dataset):
    def __init__(self, dataset_path, *, include_targets=True):
        self.path = Path(dataset_path)
        self.include_targets = include_targets
        sidecar = self.path.with_suffix('.json')
        if sidecar.exists():
            manifest = json.loads(sidecar.read_text(encoding='utf-8'))
            actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if manifest.get('sha256') != actual:
                raise ValueError('Dataset SHA-256 does not match its sidecar')
        bundle = torch.load(self.path, map_location='cpu', weights_only=True)
        if isinstance(bundle, dict) and bundle.get('schema_version') == DATASET_VERSION:
            validate_bundle(bundle, require_targets=include_targets)
            self.metadata = bundle['metadata']
            self.raw_data_list = bundle['graphs']
            if not isinstance(self.raw_data_list, list):
                raise ValueError('Dataset graphs must be a list')
            if self.metadata.get('num_graphs') != len(self.raw_data_list):
                raise ValueError('Dataset graph count does not match metadata')
        else:
            raise ValueError(f'Expected {DATASET_VERSION}')
        sample_ids = []
        for raw in self.raw_data_list:
            validate_record(raw, require_targets=include_targets)
            sample_ids.append(raw['sample_id'])
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError('Duplicate sample IDs in dataset')

    def __len__(self):
        return len(self.raw_data_list)

    def __getitem__(self, idx):
        return graph_to_data(self.raw_data_list[idx], include_targets=self.include_targets)
