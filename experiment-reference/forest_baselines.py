"""Evaluate three spanning forest baselines on the full frozen formal test."""

from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT.parent / 'formal-test-20261111'
original = runpy.run_path(str(FORMAL / 'run.py'))
read = original['read']
atomic_json = original['atomic_json']
sha256 = original['sha256']
checked_graph = original['checked_graph']
fingerprint = original['fingerprint']

import numpy as np
import torch
from gib4cl.inference.spanning_tree import extract_bfs_weights
from forest_merge import merge_forest_edges

BASELINES = ('cost_msf', 'variance_msf', 'merged_bfs')


def progress(status, **fields):
    atomic_json(ROOT / 'progress.json', dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields))


def bind(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen input changed: {path}')
    if not path.exists():
        atomic_json(path, value)


def select_forests(data):
    edges = data.edge_index.T.tolist()
    old = extract_bfs_weights(data).nonzero().flatten().tolist()
    costs = data.edge_cost.tolist()
    selected = {
        'cost_msf': merge_forest_edges(data.num_nodes, edges, [], costs),
        'variance_msf': merge_forest_edges(data.num_nodes, edges, [], data.edge_attr[:, 1].tolist()),
        'merged_bfs': merge_forest_edges(data.num_nodes, edges, old, costs),
    }

    def partition(indices):
        adj = [set() for _ in range(data.num_nodes)]
        for i in indices:
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        seen = set()
        groups = set()
        for node in range(data.num_nodes):
            if node in seen:
                continue
            part = {node}
            todo = [node]
            while todo:
                neighbors = adj[todo.pop()] - part
                part.update(neighbors)
                todo.extend(neighbors)
            groups.add(frozenset(part))
            seen.update(part)
        return groups

    components = partition(range(len(edges)))
    if not set(old) <= set(selected['merged_bfs']):
        raise ValueError(f'Merged BFS lost a parent edge: {data.sample_id}')
    totals = {name: math.fsum(costs[i] for i in indices) for name, indices in selected.items()}
    for name, indices in selected.items():
        if (len(indices) != data.num_nodes - len(components) or partition(indices) != components
                or totals['cost_msf'] > totals[name] + 1e-9):
            raise ValueError(f'Invalid forest or minimum cost: {data.sample_id}, {name}')
    return selected


def summary_row(row):
    keys = ('mse', 'coverage95', 'communication_cost', 'model_id')
    m = row['metrics']
    return {**{key: row[key] for key in keys},
            'metrics': {key: m[key] for key in ('ellipse_area_valid', 'edges', 'anchor_edges')}}


def summarize(panel, protocol):
    groups = {'dense': ['dense'], **{name: [name] for name in BASELINES}, 'gib4cl': protocol['methods'][2:]}
    fields = ('rmse_m', 'coverage95', 'ellipse_area_m2', 'communication_cost', 'cost_pct_dense', 'edges', 'anchor_edges')
    agg = protocol['aggregation']
    draws = np.random.default_rng(agg['bootstrap_seed']).integers(len(panel), size=(agg['bootstrap_draws'], len(panel)))

    def row_values(row):
        m = row['metrics']
        return row['mse'], row['coverage95'], m['ellipse_area_valid'], row['communication_cost'], 0., m['edges'], m['anchor_edges']

    values = {name: np.array([np.mean([row_values(r) for r in rows if r['model_id'] in ids], axis=0)
                              for rows in panel]) for name, ids in groups.items()}
    if not all(np.isfinite(a).all() for a in values.values()):
        raise ValueError('Nonfinite summary input')
    means = {name: a.mean(axis=0) for name, a in values.items()}
    samples = {name: a[draws].mean(axis=1) for name, a in values.items()}
    for table in (means, samples):
        dense = table['dense'][..., 3]
        for a in table.values():
            a[..., 0] = np.sqrt(a[..., 0])
            a[..., 4] = 100 * a[..., 3] / dense

    def stats(point, draws):
        interval = np.percentile(draws, [2.5, 97.5], axis=0).T
        return {k: dict(value=float(v), ci95=ci.tolist()) for k, v, ci in zip(fields, point, interval)}

    return {
        'methods': {name: stats(means[name], samples[name]) for name in groups},
        'paired': {f'{a}_minus_{b}': stats(means[a] - means[b], samples[a] - samples[b])
                   for a, b in [('gib4cl', name) for name in list(groups)[:-1]]
                   + [('variance_msf', 'cost_msf'), ('merged_bfs', 'cost_msf'), ('merged_bfs', 'variance_msf')]},
        'bootstrap_unit': 'whole graph, conditional on the frozen checkpoints',
    }


@torch.no_grad()
def run():
    progress('preflight')
    protocol_path = FORMAL / 'formal-test-protocol.json'
    formal_identity = read(FORMAL / 'identity.json')
    if (sha256(protocol_path) != original['PROTOCOL_SHA'] or formal_identity['protocol_sha'] != original['PROTOCOL_SHA']
            or sha256(FORMAL / 'run.py') != formal_identity['runner_sha']):
        raise ValueError('Frozen formal protocol or runner changed')
    protocol = read(protocol_path)
    artifacts = protocol['artifacts']
    repo = original['REPO']
    cfg_path = repo / 'configs/experiment.json'
    if sha256(cfg_path) != artifacts['config_sha256'] or original['source_manifest']() != artifacts['production_sources']:
        raise ValueError('Frozen production sources or configuration changed')
    for name, digest in artifacts['evaluation_tool_sources'].items():
        if sha256(repo / 'tools' / name) != digest:
            raise ValueError(f'Frozen evaluation tool changed: {name}')
    phase_d = ROOT.parent / 'bp-stage-d-20261111/full-bp/progress.json'
    if sha256(phase_d) != artifacts['phase_d_progress_sha256']:
        raise ValueError('Frozen backend acceptance record changed')
    original['require_idle_cuda']()
    device = original['execution']('cuda', 1, True, None)
    accepted = read(phase_d)['identity']
    if original['runtime_info'](device) != accepted['runtime'] or torch.cuda.get_device_name() != accepted['gpu_name']:
        raise ValueError(f'Runtime differs: {original["runtime_info"](device)}, {torch.cuda.get_device_name()}')

    cfg = original['load_config'](cfg_path)
    dataset, info = original['load_split'](cfg, 'test', FORMAL / 'data')
    if info['sha256'] != formal_identity['test_sha'] or len(dataset) != protocol['scope']['graphs']:
        raise ValueError('Original dataset does not match the formal test')
    budget = protocol['inference']
    seeds = budget['bp_seeds']
    methods = protocol['methods']
    subset, references, supports = [], {}, {}
    for index in range(len(dataset)):
        data = dataset[index]
        selected = select_forests(data)
        rows = checked_graph(FORMAL / 'graphs' / f'{index:04d}.json', formal_identity, data, methods, seeds)
        references[index] = [summary_row(r) for r in rows if r['model_id'] != 'bfs']
        supports[index] = {r['bp_seed']: r['support_digest'] for r in rows if r['model_id'] == 'dense'}
        workload = sum(int((data.edge_index[:, edges] < data.num_agents).all(dim=0).sum())
                       for edges in selected.values())
        subset.append(dict(index=index, sample_id=data.sample_id, edges=selected, agent_edges=workload))
    if subset[133]['edges']['merged_bfs'] != read(ROOT / 'scene-0133.json'):
        raise ValueError('Graph 133 merged BFS differs from Fig. 1')
    bind(ROOT / 'selection.json', dict(source=formal_identity, selected=subset))
    identity = dict(formal_identity=formal_identity, selection_sha=sha256(ROOT / 'selection.json'),
                    selector_sha=sha256(ROOT / 'forest_merge.py'), runner_sha=sha256(Path(__file__)))
    bind(ROOT / 'identity.json', identity)
    graph_dir = ROOT / 'graphs'
    graph_dir.mkdir(exist_ok=True)
    solver = original['CachedParticleBP'].from_config(replace(
        cfg.bp, num_particles=budget['particles'], num_iterations=budget['iterations'], chunk_size=budget['chunk_size']))
    if solver.cache_bytes != 0 or solver.compute_dtype != 'float64':
        raise ValueError('Unexpected BP settings')

    def evaluate(item):
        idx = item['index']
        data = dataset[idx]
        path = graph_dir / f'{idx:04d}.json'
        expected = supports[idx]
        if not path.exists():
            start = time.perf_counter()
            data = data.to(device)
            masks = {}
            for name, edges in item['edges'].items():
                mask = torch.zeros(data.num_edges, device=device)
                mask[edges] = 1
                masks[name] = mask
            rows = []
            for seed in seeds:
                support, digest = original['support_for'](data, cfg, seed, budget['particles'], device)
                if digest != expected[seed]:
                    raise ValueError(f'Particle support differs: {idx}, {seed}')
                for name, mask in masks.items():
                    torch.cuda.synchronize()
                    ts = time.perf_counter()
                    pred = solver.infer(data, mask, support=support)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - ts
                    row = original['metric_record'](data, mask, pred.mean, pred.covariance, cfg, prediction=pred)
                    if not all(math.isfinite(row[k]) for k in ('mse', 'coverage95', 'communication_cost')):
                        raise ValueError(f'Nonfinite metrics: {idx}, {seed}')
                    row.update(model_id=name, bp_seed=seed, support_digest=digest, inference_seconds=elapsed,
                               actual_iterations=pred.iterations, converged=pred.converged, message_residual=pred.message_residual)
                    rows.append(row)
            atomic_json(path, dict(identity=identity, rows=rows, rows_digest=fingerprint(rows),
                                   seconds=time.perf_counter() - start))
        rows = checked_graph(path, identity, data.cpu(), BASELINES, seeds)
        if any(r['support_digest'] != expected[r['bp_seed']] or r['selected_edge_ids'] != item['edges'][r['model_id']]
               or r['actual_iterations'] != budget['iterations'] for r in rows):
            raise ValueError(f'Saved tree results do not match frozen settings: {idx}')
        return rows, read(path)['seconds']

    ordered = sorted(subset, key=lambda item: (item['agent_edges'], item['index']))
    pilot = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    progress('pilot', target=len(subset), indices=[item['index'] for item in pilot])
    pilot_seconds = [evaluate(item)[1] for item in pilot]
    for item in pilot:
        evaluate(item)
    atomic_json(ROOT / 'pilot.json', dict(indices=[item['index'] for item in pilot], seconds=pilot_seconds,
                agent_edges=[item['agent_edges'] for item in pilot],
                projected_seconds=sum(pilot_seconds) / sum(item['agent_edges'] for item in pilot)
                * sum(item['agent_edges'] for item in subset)))
    panel, seconds = [], 0.
    for item in subset:
        progress('running', completed=len(panel), target=len(subset), graph=item['index'], measured_seconds=seconds)
        rows, elapsed = evaluate(item)
        panel.append(references[item['index']] + [summary_row(r) for r in rows])
        seconds += elapsed
    summary = summarize(panel, protocol)
    summary.update(graphs=len(panel), new_records=len(panel) * len(seeds) * len(BASELINES), measured_seconds=seconds, identity=identity)
    atomic_json(ROOT / 'summary.json', summary)
    lines = ['# Three spanning forest baselines on the 500 formal test graphs', '',
             '| Method | RMSE (m) | 95% coverage (%) | Ellipse area (m²) | Cost / Dense (%) | Edges | Anchor edges |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, row in summary['methods'].items():
        keys = ('rmse_m', 'coverage95', 'ellipse_area_m2', 'cost_pct_dense', 'edges', 'anchor_edges')
        vals = [row[k]['value'] * (100 if k == 'coverage95' else 1) for k in keys]
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    progress('complete', completed=len(panel), target=len(subset), new_records=summary['new_records'], measured_seconds=seconds)


if __name__ == '__main__':
    with original['lock'](ROOT):
        try:
            run()
        except Exception:
            progress('failed', error=traceback.format_exc())
            raise
