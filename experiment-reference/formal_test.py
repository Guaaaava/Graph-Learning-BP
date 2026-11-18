"""Execute the frozen independent test with graph-level recovery."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REPO = ROOT/'Graph-Learning-BP'
CAND = WORK/'night-candidates-screen-20261106/workspaces/candidates'
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from gib4cl.config import fingerprint, load_config
from gib4cl.data.generate import create_dataset
from gib4cl.learning.checkpoints import load_selector
from gib4cl.runtime import atomic_json, execution, load_split, runtime_info, sha256, source_manifest, state_digest
from tools.bp_benchmark import require_idle_cuda
from tools.bp_cached import CachedParticleBP
from tools.night_runner import lock
from tools.selection_evaluate import masks, metric_record, support_for

PROTOCOL_SHA = 'f88d751be85b0739c0e3c546fa02917911f7b171bb8c13954601f375afd5bf08'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def progress(status, **info):
    atomic_json(ROOT/'progress.json', dict(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **info))


def checked_graph(path, identity, data, methods, seeds):
    record = read(path)
    rows = record['rows']
    pairs = {(row['bp_seed'], row['model_id']) for row in rows}
    expected = {(seed, method) for seed in seeds for method in methods}
    digest = state_digest(data.to_dict())
    if (record['identity'] != identity or record['rows_digest'] != fingerprint(rows)
            or len(rows) != len(expected) or pairs != expected):
        raise ValueError(f'Incomplete or altered graph record: {path}')
    if any(r['status'] != 'complete' or r['sample_id'] != data.sample_id or r['input_digest'] != digest for r in rows):
        raise ValueError(f'Graph input differs: {path}')
    if any(len({r['support_digest'] for r in rows if r['bp_seed']==s}) != 1 for s in seeds):
        raise ValueError(f'Particle supports differ: {path}')
    return rows


def summarize(graphs, protocol):
    models = protocol['methods'][2:]
    groups = {'dense':['dense'], 'bfs':['bfs'], 'communication':models}
    groups.update({mid:[mid] for mid in models})
    fields = ('mse','coverage95','communication_cost','invalid_covariance_fraction')
    stats = {}
    draws = np.random.default_rng(protocol['aggregation']['bootstrap_seed']).integers(
        0, len(graphs), size=(protocol['aggregation']['bootstrap_draws'], len(graphs)))
    values = {}
    for group, mids in groups.items():
        arrays = [[r for r in rows if r['model_id'] in mids] for rows in graphs]
        a = np.asarray([[np.mean([r[k] for r in rows]) for k in fields] for rows in arrays])
        values[group] = a
        mean = a.mean(axis=0)
        sample = a[draws].mean(axis=1)
        mean[0] = np.sqrt(mean[0])
        sample[:,0] = np.sqrt(sample[:,0])
        names = ('rmse_m',*fields[1:])
        stats[group] = dict(zip(names, mean.tolist()))
        stats[group]['intervals'] = {key:np.percentile(sample[:,i],[2.5,97.5]).tolist() for i,key in enumerate(names)}
        for k in ('mean_nees_valid','ellipse_area_valid','unanchored_agents','low_ess_fraction'):
            per_graph = [[r['metrics'][k] for r in rows if r['metrics'].get(k) is not None] for rows in arrays]
            observed = [float(np.mean(v)) for v in per_graph if v]
            stats[group][k] = float(np.mean(observed)) if observed else None
        stats[group]['ess_below_2'] = sum(sum(e<2 for e in r['ess']) for rows in arrays for r in rows)
        stats[group]['ess_node_observations'] = sum(len(r['ess']) for rows in arrays for r in rows)
    paired = {}
    candidate = values['communication']
    for base in ('dense','bfs'):
        a,b = candidate[draws].mean(axis=1),values[base][draws].mean(axis=1)
        paired[base] = dict(
            rmse_difference=stats['communication']['rmse_m']-stats[base]['rmse_m'],
            rmse_difference_interval=np.percentile(np.sqrt(a[:,0])-np.sqrt(b[:,0]),[2.5,97.5]).tolist(),
            coverage_difference=stats['communication']['coverage95']-stats[base]['coverage95'],
            coverage_difference_interval=np.percentile(a[:,1]-b[:,1],[2.5,97.5]).tolist(),
            communication_difference=stats['communication']['communication_cost']-stats[base]['communication_cost'],
            communication_difference_interval=np.percentile(a[:,2]-b[:,2],[2.5,97.5]).tolist())
    return dict(rows=stats, paired_vs=paired, bootstrap_scope='whole test graphs, conditional on the trained checkpoints')


@torch.no_grad()
def run(pilot):
    require_idle_cuda()
    device = execution('cuda',1,True,None)
    protocol_path = ROOT/'formal-test-protocol.json'
    if sha256(protocol_path) != PROTOCOL_SHA:
        raise ValueError('Frozen protocol hash differs')
    protocol = read(protocol_path)
    artifacts = protocol['artifacts']
    cfg_path = REPO/'configs/experiment.json'
    if sha256(cfg_path) != artifacts['config_sha256'] or source_manifest() != artifacts['production_sources']:
        raise ValueError('Frozen source or configuration differs')
    for name, digest in artifacts['evaluation_tool_sources'].items():
        if sha256(REPO/'tools'/name) != digest:
            raise ValueError(f'Frozen evaluation tool differs: {name}')
    phase_d = WORK/'bp-stage-d-20261111/full-bp/progress.json'
    if sha256(phase_d) != artifacts['phase_d_progress_sha256']:
        raise ValueError('Accepted runtime evidence differs')
    accepted = read(phase_d)['identity']
    if runtime_info(device) != accepted['runtime'] or torch.cuda.get_device_name() != accepted['gpu_name']:
        raise ValueError('Accepted execution environment differs')
    cfg = load_config(cfg_path)
    scope = protocol['scope']
    methods = protocol['methods']
    seeds = protocol['inference']['bp_seeds']
    threshold = protocol['edge_rule']['threshold']
    if len(methods)*len(seeds)*scope['graphs'] != protocol['expected_records'] or cfg.seeds.data != scope['data_seed']:
        raise ValueError('Frozen scope differs')
    models = {}
    for model in protocol['checkpoints']:
        path = CAND/model['path']
        if sha256(path) != model['sha256']:
            raise ValueError('Frozen checkpoint differs')
        models[model['model_id']] = load_selector(path,cfg.model,device)[0]
    data_root = ROOT/'data'
    path = cfg.dataset_path('test',data_root)
    if not path.exists():
        if not pilot:
            raise ValueError('Pilot dataset is missing')
        progress('generating-test', completed=0, target=scope['graphs'])
        create_dataset('test',scope['graphs'],seed_offset=scope['seed_offset'],settings=cfg,output_root=data_root)
    dataset, info = load_split(cfg,'test',data_root)
    if (len(dataset) != scope['graphs'] or info['resolved_split_config']['seed_offset'] != scope['seed_offset']
            or read(path.with_suffix('.json'))['sha256'] != info['sha256']):
        raise ValueError('Test dataset identity differs')
    identity = dict(protocol_sha=PROTOCOL_SHA, test_sha=info['sha256'], runner_sha=sha256(Path(__file__)))
    identity_path = ROOT/'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Cannot resume under a different identity')
    atomic_json(identity_path, identity)
    atomic_json(ROOT/'test-dataset.json',info)
    graph_root = ROOT/'graphs'
    graph_root.mkdir(exist_ok=True)
    budget = protocol['inference']
    bp_cfg = replace(cfg.bp, num_particles=budget['particles'],num_iterations=budget['iterations'],
                     chunk_size=budget['chunk_size'])
    solver = CachedParticleBP.from_config(bp_cfg)
    if solver.cache_bytes != 0 or solver.compute_dtype != 'float64':
        raise ValueError('Unexpected backend configuration')
    graphs, graph_seconds = [], []
    for index in range(1 if pilot else len(dataset)):
        data = dataset[index]
        graph_path = graph_root/f'{index:04d}.json'
        if graph_path.exists():
            graphs.append(checked_graph(graph_path,identity,data,methods,seeds))
            graph_seconds.append(read(graph_path)['seconds'])
            continue
        if not pilot and index==0:
            raise ValueError('Pilot record missing on restart')
        progress('running', phase='pilot' if pilot else 'full', completed=index,
                 target=len(dataset), graph=index, measured_graph_seconds=sum(graph_seconds))
        start = time.perf_counter()
        data = data.to(device)
        selected = masks(data)
        for name, net in models.items():
            logits = net.score_edges(data)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite edge scores')
            selected[name] = (logits > math.log(threshold/(1-threshold))).to(logits.dtype)
        rows = []
        for seed in seeds:
            support, digest = support_for(data,cfg,seed,budget['particles'],device)
            for name in methods:
                torch.cuda.synchronize()
                ts = time.perf_counter()
                pred = solver.infer(data,selected[name],support=support)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-ts
                row = metric_record(data,selected[name],pred.mean,pred.covariance,cfg,prediction=pred)
                row.update(model_id=name,bp_seed=seed,support_digest=digest,
                           inference_seconds=elapsed,actual_iterations=pred.iterations,
                           converged=pred.converged,message_residual=pred.message_residual,
                           ess=pred.ess.cpu().tolist(),maximum_weight=pred.weights.max(dim=1).values.cpu().tolist())
                if not all(math.isfinite(row[k]) for k in ('mse','coverage95','communication_cost','invalid_covariance_fraction')):
                    raise ValueError('Nonfinite primary test metric')
                rows.append(row)
                del pred
        seconds = time.perf_counter()-start
        atomic_json(graph_path,dict(identity=identity,rows=rows,rows_digest=fingerprint(rows),seconds=seconds))
        graphs.append(checked_graph(graph_path,identity,data.cpu(),methods,seeds))
        graph_seconds.append(seconds)
    if pilot:
        progress('pilot-complete', completed=1, target=len(dataset), measured_graph_seconds=graph_seconds[0])
        return
    if sum(map(len,graphs)) != protocol['expected_records']:
        raise ValueError('Incomplete test panel')
    summary = summarize(graphs,protocol)
    summary.update(identity=identity,status='complete',graphs=len(graphs),records=sum(map(len,graphs)))
    atomic_json(ROOT/'summary.json',summary)
    lines = ['# Independent 500-graph evaluation','','| Method | RMSE (m) | Coverage at 95% | Communication cost |',
             '| --- | ---: | ---: | ---: |']
    for name,r in summary['rows'].items():
        lines.append(f"| {name} | {r['rmse_m']:.4f} | {100*r['coverage95']:.2f}% | {r['communication_cost']:.2f} |")
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    progress('complete',completed=len(graphs),target=len(dataset),records=summary['records'],
             measured_graph_seconds=sum(graph_seconds))


if __name__ == '__main__':
    try:
        if len(sys.argv)==1:
            with lock(ROOT):
                for phase in ('--pilot','--full'):
                    subprocess.run([sys.executable,'-B',__file__,phase],cwd=REPO,check=True)
        elif sys.argv[1:] in (['--pilot'],['--full']):
            run(sys.argv[1]=='--pilot')
        else:
            raise ValueError('Unexpected invocation')
    except Exception:
        if len(sys.argv)>1 or not (ROOT/'progress.json').exists():
            progress('failed',error=traceback.format_exc())
        raise
