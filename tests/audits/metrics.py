"""CPU metric audit: independent NumPy/DFS checks, replay and invalid-data evidence."""
from dataclasses import replace
import argparse
import csv
from itertools import combinations
import json
import math
import re
from pathlib import Path
import numpy as np
import torch

from gib4cl.experiments.cli import main
from gib4cl.config import REPO_ROOT,load_config
from gib4cl.evaluation.evaluate import evaluate
from gib4cl.data.generate import create_dataset
from gib4cl.evaluation.metrics import METRICS_VERSION,SCALAR_METRICS
from gib4cl.runtime import atomic_json,load_split,runtime_info,sha256,source_manifest


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=reject)


def close(actual,expected):
    assert math.isclose(actual,expected,rel_tol=2e-12,abs_tol=2e-12),(actual,expected)


def components(nodes,edges):
    """Independent set/DFS reference; no production union-find or BFS implementation."""
    adjacent = {i:set() for i in nodes}
    for u,v in edges:
        adjacent[u].add(v)
        adjacent[v].add(u)
    groups,remaining = [],set(nodes)
    while remaining:
        pending = [min(remaining)]
        group = set()
        while pending:
            node = pending.pop()
            if node not in group:
                group.add(node)
                pending.extend(adjacent[node]-group)
        remaining -= group
        groups.append(group)
    return groups


def independent_metrics(result,cfg,data_root):
    dataset,_ = load_split(cfg,'test',data_root)
    graphs = {data.sample_id:data for data in dataset}
    selections = {graph['sample_id']:graph['selection']['selected_edge_ids'] for graph in result['graphs']}
    for row in result['records']:
        data = graphs[row['sample_id']]
        value = row['metrics']
        errors = np.asarray(row['estimates'])-data.y.numpy()
        distances = np.linalg.norm(errors,axis=1)
        close(value['rmse'],np.sqrt(np.mean(np.sum(errors**2,axis=1))))
        close(value['mae'],np.mean(distances))
        for name,q in [('error_p50',50),('error_p90',90),('error_p95',95)]:
            close(value[name],np.percentile(distances,q))
        close(value['outage'],np.mean(distances > cfg.evaluation.outage_threshold))
        for i,cov in enumerate(row['covariances']):
            if value['agent_covariance_valid'][i]:
                q = float(errors[i]@np.linalg.solve(cov,errors[i]))
                close(value['agent_nees'][i],q)
                logdet = np.linalg.slogdet(cov)[1]
                close(value['agent_gaussian_nll'][i],np.log(2*np.pi)+(logdet+q)/2)
                close(value['agent_ellipse_area'][i],np.pi*value['ellipse_threshold']*np.sqrt(np.linalg.det(cov)))
            else:
                assert value['agent_nees'][i] is None
        selected = selections[data.sample_id][row['topology']]
        pairs = [tuple(sorted(pair)) for pair in data.edge_index[:,selected].T.tolist()]
        agent_set = set(data.agent_node_index.tolist())
        agent_pairs = [pair for pair in pairs if set(pair) <= agent_set]
        pair_set = set(agent_pairs)
        triangles = sum(all(tuple(sorted(pair)) in pair_set for pair in combinations(triple,2))
                        for triple in combinations(agent_set,3))
        agent_groups = components(agent_set,agent_pairs)
        full_groups = components(range(len(data.x)),pairs)
        assert value['agent_triangles'] == triangles
        assert value['agent_cycle_rank'] == len(agent_pairs)-len(agent_set)+len(agent_groups)
        assert value['connected_components'] == len(full_groups)
        assert value['agent_components'] == len(agent_groups)
        unanchored = sum(len(group) for group in full_groups if group <= agent_set)
        assert value['unanchored_agents'] == unanchored
        candidate_cost = data.edge_cost.double().numpy()
        close(value['communication_cost'],candidate_cost[selected].sum())
        if candidate_cost.sum() > 0:
            close(value['communication_saving'],1-candidate_cost[selected].sum()/candidate_cost.sum())
        else:
            assert value['communication_saving'] is None
        partition = sum(value[name] for name in (
            'nees_below_interval','nees_interval_pass','nees_above_interval','invalid_covariance_rate'))
        close(partition,1)
        assert set(SCALAR_METRICS) <= set(value)
    for sample in graphs:
        for repeat in range(cfg.evaluation.bp_repeats):
            rows = [r for r in result['records'] if r['sample_id']==sample and r['bp_repeat']==repeat]
            assert len(rows)==3 and len({r['support_digest'] for r in rows})==1
    return len(result['records'])


def check_tables(destination,result):
    manifest = read_json(destination/'manifest.json')
    for file,key in [('results.json','results_sha256'),('report.md','report_sha256'),
                     ('trial_metrics.csv','trial_metrics_sha256'),('summary_metrics.csv','summary_metrics_sha256')]:
        assert sha256(destination/file)==manifest[key]
    assert manifest['status']=='complete'
    with (destination/'trial_metrics.csv').open(encoding='utf-8',newline='') as handle:
        trials = list(csv.DictReader(handle))
    assert len(trials)==len(result['records'])
    for record,export in zip(result['records'],trials):
        for name in SCALAR_METRICS:
            value = record['metrics'].get(name)
            assert export[name]==('' if value is None else str(value))
    with (destination/'summary_metrics.csv').open(encoding='utf-8',newline='') as handle:
        stats = list(csv.DictReader(handle))
    expected = 0
    for topology,values in result['summary']['topologies'].items():
        for metric,stat in values['graph_metrics'].items():
            saved = next(r for r in stats if r['comparison']=='absolute' and r['topology']==topology and r['metric']==metric)
            for key in ('mean','std','graphs','ci_low','ci_high'):
                assert saved[key]==('' if stat[key] is None else str(stat[key]))
            expected += 1
    assert len(stats)>expected  # curves and paired differences are also retained
    assert not re.search(r'\|\s*[+-]?(?:nan|inf(?:inity)?)\s*(?:\||$)',
                         (destination/'report.md').read_text(encoding='utf-8'),re.I|re.M)


def verify(root,previous_evaluation=None):
    root.mkdir(parents=True,exist_ok=False)
    cfg_path = REPO_ROOT/'configs/smoke-pipeline.json'
    cfg = load_config(cfg_path)
    pipeline = main(['run','--config',str(cfg_path),'--data-root',str(root/'data'),
                     '--output-root',str(root/'pipeline'),'--device','cpu','--threads','1'])
    result = read_json(pipeline['evaluation']/'results.json')
    audited = independent_metrics(result,cfg,root/'data')
    check_tables(pipeline['evaluation'],result)
    replay = evaluate(checkpoint=pipeline['models']/'best_checkpoint.pt',settings=cfg,data_root=root/'data',
                      output_root=root/'replay',device='cpu',threads=1)
    again = read_json(replay/'results.json')
    assert result['records']==again['records']
    assert result['summary']==again['summary'] and result['graphs']==again['graphs']
    check_tables(replay,again)
    one = evaluate(checkpoint=pipeline['models']/'best_checkpoint.pt',settings=cfg,data_root=root/'data',
                   output_root=root/'single',max_graphs=1,device='cpu',threads=1)
    single = read_json(one/'results.json')
    for topology in single['summary']['topologies'].values():
        assert all(stat['std'] is None and stat['ci_low'] is None and stat['ci_high'] is None
                   for stat in topology['graph_metrics'].values())
    check_tables(one,single)
    empty = replace(cfg,data=replace(cfg.data,comm_radius=0.),
                    bp=replace(cfg.bp,num_particles=2,num_iterations=0),
                    evaluation=replace(cfg.evaluation,bp_repeats=1))
    create_dataset('test',1,settings=empty,output_root=root/'empty-data')
    zero = evaluate(checkpoint=pipeline['models']/'best_checkpoint.pt',settings=empty,data_root=root/'empty-data',
                    output_root=root/'empty-evaluation',device='cpu',threads=1)
    invalid = read_json(zero/'results.json')
    for row in invalid['records']:
        m = row['metrics']
        assert m['edge_saving'] is m['communication_saving'] is m['agent_triangle_retention'] is None
        assert m['mean_nees_valid'] is m['gaussian_nll_valid'] is m['ellipse_area_valid'] is None
        assert m['nees_invalid']==m['num_agents'] and m['ellipse_coverage']==0
    check_tables(zero,invalid)
    from tools.provenance import preserved_core,preserved_references,paper_sha256,record_path,fixture,provenance_scope
    preserved = preserved_core()
    reference_hashes = preserved_references()
    prior_comparison = None
    if previous_evaluation is not None:
        prior = read_json(previous_evaluation/'results.json')
        assert result['checkpoint']['model_digest']==prior['checkpoint']['model_digest']
        old_rows = {(r['sample_id'],r['bp_repeat'],r['topology']):r for r in prior['records']}
        for row in result['records']:
            previous = old_rows[(row['sample_id'],row['bp_repeat'],row['topology'])]
            assert row['estimates']==previous['estimates'] and row['covariances']==previous['covariances']
            assert row['support_digest']==previous['support_digest']
        prior_comparison = {'path':str(previous_evaluation),'all_estimates_covariances_supports_exact':True,
                            'model_digest_equal':True,'records':len(old_rows)}
    report = {
        'stage':'cpu-metric-audit','date':'2026-10-26','metrics_version':METRICS_VERSION,
        'runtime':runtime_info(torch.device('cpu')),'config':cfg.to_dict(),'config_sha256':sha256(cfg_path),
        'pipeline':{k:record_path(v) for k,v in pipeline.items()},
        'audited_bp_records':audited,'evaluation_graphs':result['num_graphs'],
        'bp_trials_per_topology':result['num_graphs']*cfg.evaluation.bp_repeats,
        'independent_numpy_metric_and_dfs_topology_references_passed':True,
        'shared_supports_across_topologies':True,'replay_records_and_summary_exact':True,
        'json_csv_report_consistent':True,'single_graph_uncertainty_null':True,
        'zero_candidate_and_all_invalid_null_policy_passed':True,
        'zero_edge_evaluation':record_path(zero),
        'previous_stage6_comparison':prior_comparison,
        'main_summary':result['summary'],'preserved_core_algorithms':preserved,'reference_provenance':provenance_scope(),'reference_hashes':reference_hashes,
        'source_sha256':source_manifest(),'results_sha256':sha256(pipeline['evaluation']/'results.json'),
        'paper_sha256':paper_sha256(),'baseline_sha256':fixture()['snapshots']['initial']['sha256'],
        'limitations':['CPU only; no GPU runtime or peak-memory benchmark',
                       'Two-epoch smoke verification is not a formal performance experiment',
                       'Gaussian ellipses and NLL use BP mean/covariance, not true posterior density',
                       'Bootstrap conditions on one training checkpoint; no between-training-seed uncertainty'],
    }
    atomic_json(root/'verification.json',report)
    print(json.dumps({'audit':str(root/'verification.json'),'independent_records':audited,
                      'replay_exact':True,'csv_json_consistent':True,'all_invalid_saved':True},ensure_ascii=False))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=REPO_ROOT/'outputs/audit-metrics')
    parser.add_argument('--previous-evaluation',type=Path)
    args = parser.parse_args()
    verify(args.output_root.resolve(),args.previous_evaluation)
