"""Freeze structural validation samples locally, without loading model weights.

A CPU-regenerated prefix is checked against available historical input digests.
Unseen indices still require server-original verification before GPU execution;
this diagnostic regeneration must never replace the original 500-graph dataset.
"""
from __future__ import annotations
import math
import torch
from gib4cl.config import config_from_dict
from gib4cl.data.generate import build_dataset
from gib4cl.data.dataset import graph_to_data
from gib4cl.evaluation.topology import selection_metrics
from gib4cl.inference.spanning_tree import bfs_forest
from gib4cl.runtime import state_digest, dataset_identity


def choose_samples(rows, *, mandatory=(0, 6, 5), count=8):
    indexed = {r['graph_index']:r for r in rows}
    if len(indexed) != len(rows) or not set(mandatory) <= set(indexed):
        raise ValueError('Duplicate or missing graph indices')
    if not len(set(mandatory)) <= count <= len(rows):
        raise ValueError('Invalid sample count')
    ordered = sorted(indexed, key=lambda i:(indexed[i]['dense_agent_edges'], i))
    ranks = {i:k for k, i in enumerate(ordered)}
    selected = set(mandatory)
    reasons = {i:['initial pilot' if i in (0, 6) else 'previously documented difficult graph'] for i in mandatory}
    for q in (0., .25, .5, .75, 1.):
        i = ordered[math.floor(q*(len(ordered)-1)+.5)]
        if len(selected) < count or i in selected:
            selected.add(i); reasons.setdefault(i, []).append(f'dense-edge rank quantile {q}')
    while len(selected) < count:
        remaining = [i for i in ordered if i not in selected]
        i = min(remaining, key=lambda x:(-min(abs(ranks[x]-ranks[s]) for s in selected), x))
        selected.add(i); reasons[i] = ['fill largest uncovered dense-edge rank interval']
    return [dict(index=i, reasons=reasons[i]) for i in sorted(selected)]


def prepare_samples(reference_panel, *, prefix=32, diagnostic_regeneration=False):
    context = reference_panel['context']
    if context['dataset']['split'] != 'val' or context['method'] != 'particle_bp':
        raise ValueError('Only original validation particle-BP references are accepted')
    cfg = config_from_dict(context['base_config'])
    bundle = build_dataset(cfg, 'val', prefix)
    expected = context['input_digests']
    rows = []
    for i, raw in enumerate(bundle['graphs']):
        data = graph_to_data(raw)
        digest = state_digest(data.to_dict())
        sid = data.sample_id
        verified = sid in expected and digest == expected[sid]
        if sid in expected and not verified and not diagnostic_regeneration:
            raise ValueError('Regenerated validation inputs differ from historical record: '+sid)
        dense = selection_metrics(data, torch.ones(data.num_edges))
        bfs = bfs_forest(data.num_nodes, data.edge_index, data.anchor_node_index)
        forest = selection_metrics(data, bfs.edge_weights)
        rows.append(dict(graph_index=i, sample_id=sid, input_digest=digest,
                         historical_input_verified=verified, historical_input_digest=expected.get(sid),
                         agents=int(data.num_agents),
                         dense_agent_edges=dense['agent_edges'], dense_physical_edges=dense['edges'],
                         bfs_agent_edges=forest['agent_edges'], bfs_physical_edges=forest['edges']))
    if bundle['metadata']['series_id'] != context['dataset']['series_id']:
        raise ValueError('Validation series identity differs')
    selected = choose_samples(rows)
    return dict(schema='gib4cl.bp-fixed-samples.v1', split='val', prefix=prefix,
                original_dataset=context['dataset'], initial=[0, 6],
                expanded=selected, all_graphs=rows,
                historical_inputs_verified=sum(r['historical_input_verified'] for r in rows),
                pending_server_verification=[r['graph_index'] for r in rows if not r['historical_input_verified']],
                execution_ready=False, diagnostic_regeneration=diagnostic_regeneration,
                selection_uses_speed_or_new_performance=False, learned_masks_loaded=False,
                note='Indices fixed from diagnostic structure. Use and verify original server dataset/hash; '
                     'local rebuilt tensors are not substitutes and may differ by environment'), bundle


def verify_original_val(plan, dataset, identity):
    """Stage-C gate, after load_split verified the original file and sidecar.

    Known historical input hashes must match. For other indices the immutable
    original dataset hash binds the data; record actual input hashes freshly.
    Diagnostic rebuilt-input hashes are deliberately NOT accepted as originals.
    """
    if dataset_identity(identity) != dataset_identity(plan['original_dataset']):
        raise ValueError('Original server validation dataset identity differs')
    indices=sorted(set(plan['initial']) | {r['index'] for r in plan['expanded']})
    local={r['graph_index']:r for r in plan['all_graphs']}
    actual=[]
    for index in indices:
        data=dataset[index]
        if data.x.device.type!='cpu': raise ValueError('Verify original inputs before GPU transfer')
        row=local[index]; checksum=state_digest(data.to_dict())
        if data.sample_id!=row['sample_id']: raise ValueError('Original sample identity differs')
        expected=row['historical_input_digest']
        if expected is not None and expected!=checksum:
            raise ValueError('Historical original graph input digest differs')
        metrics=selection_metrics(data,torch.ones(data.num_edges))
        if (metrics['agent_edges']!=row['dense_agent_edges'] or metrics['edges']!=row['dense_physical_edges']
                or int(data.num_agents)!=row['agents']):
            raise ValueError('Original structure differs from fixed sample planning')
        actual.append(dict(index=index,sample_id=data.sample_id,input_digest=checksum,
                           historical_hash_checked=expected is not None))
    return dict(status='verified_original_val',dataset=dataset_identity(identity),inputs=actual,
                local_rebuilt_tensors_used=False)
