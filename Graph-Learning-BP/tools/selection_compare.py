"""Common graph-equal metrics; deliberately never ranks weighted losses."""
from __future__ import annotations
import csv
import io
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

from gib4cl.config import fingerprint
from gib4cl.learning.validation import stream_seed
from gib4cl.runtime import atomic_json, _io_path

COEFFICIENTS = ('beta','gamma','kappa','eta')
DIMENSIONS = ('rmse_graph_equal','absolute_95pct_ellipse_coverage_gap',
              'communication_cost_graph_mean','invalid_covariance_fraction')
ORDERS = {'accuracy':DIMENSIONS,
          'calibration':(DIMENSIONS[1],DIMENSIONS[3],DIMENSIONS[0],DIMENSIONS[2]),
          'cost':(DIMENSIONS[2],DIMENSIONS[0],DIMENSIONS[1],DIMENSIONS[3])}


def coefficients(value):
    if set(value) != set(COEFFICIENTS):
        raise ValueError('Require exactly beta/gamma/kappa/eta')
    result = {k:float(value[k]) for k in COEFFICIENTS}
    if any(not math.isfinite(v) or v < 0 for v in result.values()) or result['beta'] <= 0:
        raise ValueError('Invalid objective coefficients')
    return result


def candidate_id(value):
    return fingerprint(coefficients(value))[:12]


def ordering(row,role='accuracy'):
    return tuple(row[k] for k in ORDERS[role])+tuple(row['coefficients'][k] for k in COEFFICIENTS)


def choose(rows):
    rows = [r for r in rows if not r['baseline']]
    if not rows:
        raise ValueError('No complete candidates to compare')
    frontier = [row for row in rows if not any(
        all(other[k] <= row[k] for k in DIMENSIONS) and any(other[k] < row[k] for k in DIMENSIONS)
        for other in rows)]
    roles = {role:min(frontier,key=lambda r:ordering(r,role))['candidate_id'] for role in ORDERS}
    representatives = list(dict.fromkeys(roles.values()))
    return {'frontier':[r['candidate_id'] for r in sorted(frontier,key=ordering)],
            'roles':roles,'representatives':representatives,
            'primary':roles['accuracy'],'rule':'graph-equal four-dimension Pareto; fixed role lexicographic order'}


def summarize(report,*,bootstrap_samples=2000,bootstrap_seed=700):
    if report.get('schema') != 'gib4cl.selection-records.v1' or report.get('status') != 'complete':
        raise ValueError('Ranking requires a complete selection-records report')
    context,records = report['context'],report['records']
    if context.get('method') not in ('gaussian_proxy','particle_bp') or context.get('dataset',{}).get('split') != 'val':
        raise ValueError('Only a single declared val evaluator may be ranked; test is sealed')
    graph_ids = context['sample_ids']
    if not graph_ids or len(set(graph_ids)) != len(graph_ids):
        raise ValueError('Graph identities must be nonempty and unique')
    specs = {m['model_id']:m for m in context['models']}
    if len(specs) != len(context['models']):
        raise ValueError('Duplicate model identity')
    repeats = context['bp_seeds'] if context['method'] == 'particle_bp' else [None]
    if not repeats or len(set(repeats)) != len(repeats):
        raise ValueError('Particle repeat identities must be nonempty and unique')
    if not context['training_seeds'] or len(set(context['training_seeds'])) != len(context['training_seeds']):
        raise ValueError('Training seed identities must be nonempty and unique')
    lookup = {}
    support = {}
    for row in records:
        if row.get('status') != 'complete' or row['model_id'] not in specs:
            raise ValueError('Failed or unplanned prediction cannot be silently excluded')
        key = (row['model_id'],row['sample_id'],row['bp_seed'])
        if key in lookup:
            raise ValueError('Duplicate prediction identity')
        if row['sample_id'] not in graph_ids or row['bp_seed'] not in repeats:
            raise ValueError('Unexpected graph or particle repeat')
        if row['input_digest'] != context['input_digests'][row['sample_id']]:
            raise ValueError('Input digest mismatch')
        if context['method'] == 'particle_bp':
            pair = (row['sample_id'],row['bp_seed'])
            if pair in support and support[pair] != row['support_digest']:
                raise ValueError('BP comparisons require identical support on each graph/repeat')
            support[pair] = row['support_digest']
        lookup[key] = row
    expected = {(m,g,r) for m in specs for g in graph_ids for r in repeats}
    if set(lookup) != expected:
        raise ValueError('Incomplete model/graph/repeat panel; no partial ranking')
    groups = defaultdict(list)
    for spec in specs.values():
        groups[spec['candidate_id']].append(spec)
    expected_seeds = sorted(context['training_seeds'])
    values, rows = {},[]
    indices = (np.random.default_rng(stream_seed('selection-bootstrap',bootstrap_seed)).integers(
        0,len(graph_ids),size=(bootstrap_samples,len(graph_ids))) if len(graph_ids)>1 and bootstrap_samples else None)
    for group,models in sorted(groups.items()):
        baseline = models[0]['baseline']
        if any(m['baseline'] != baseline or m.get('coefficients') != models[0].get('coefficients') for m in models):
            raise ValueError('Candidate group has inconsistent coefficients or baseline identity')
        if not baseline and sorted(m['training_seed'] for m in models) != expected_seeds:
            raise ValueError('All candidates require the same complete training-seed panel')
        if baseline and len(models) != 1:
            raise ValueError('Dense/BFS baselines must be shared, not duplicated per training seed')
        per_graph = []
        for graph in graph_ids:
            trials = [lookup[(m['model_id'],graph,r)] for m in models for r in repeats]
            # Average both kinds of repeats within a graph, then average graphs.
            per_graph.append([math.fsum(t[key] for t in trials)/len(trials) for key in
                ('mse','coverage95','communication_cost','invalid_covariance_fraction')])
        array = np.asarray(per_graph,dtype=float)
        if not np.isfinite(array).all() or (array[:,0]<0).any() or (array[:,2]<0).any() or ((array[:,[1,3]]<0)|(array[:,[1,3]]>1)).any():
            raise ValueError('Primary metrics must be finite and in range')
        values[group] = array
        means = array.mean(axis=0)
        row = {'candidate_id':group,'baseline':baseline,'coefficients':models[0].get('coefficients'),
               'graphs':len(graph_ids),'models':len(models),'bp_repeats':len(repeats),
               DIMENSIONS[0]:math.sqrt(means[0]),DIMENSIONS[1]:abs(means[1]-.95),
               DIMENSIONS[2]:float(means[2]),DIMENSIONS[3]:float(means[3]),
               'coverage95':float(means[1]),'intervals':{},'per_training_seed':{}}
        if not baseline:
            row['effective_communication_weight'] = row['coefficients']['beta']*row['coefficients']['gamma']
            row['effective_triangle_weight'] = row['coefficients']['beta']*row['coefficients']['kappa']
        for model in models:
            if baseline:
                continue
            data = [lookup[(model['model_id'],g,r)] for g in graph_ids for r in repeats]
            row['per_training_seed'][str(model['training_seed'])] = {
                'rmse_graph_equal':math.sqrt(math.fsum(t['mse'] for t in data)/len(data)),
                'coverage95':math.fsum(t['coverage95'] for t in data)/len(data),
                'selected_epoch':model.get('selected_epoch'),'checkpoint':model.get('checkpoint')}
        if indices is not None:
            boot = array[indices].mean(axis=1)
            transformed = [np.sqrt(boot[:,0]),np.abs(boot[:,1]-.95),boot[:,2],boot[:,3]]
            row['intervals'] = {k:dict(zip(('low','high'),map(float,np.percentile(v,[2.5,97.5]))))
                                for k,v in zip(DIMENSIONS,transformed)}
        else:
            row['intervals'] = {k:None for k in DIMENSIONS}
        # Secondary scores stay conditional when covariances are invalid.
        secondary = ('mae','outage','ellipse_area_valid','mean_nees_valid','gaussian_nll_valid',
                     'edges','agent_triangles','agents_below_degree_three','unanchored_agents',
                     'ess_min','ess_mean','low_ess_fraction','bp_converged')
        row['secondary'] = {}
        data = [lookup[(m['model_id'],g,r)] for m in models for g in graph_ids for r in repeats]
        for key in secondary:
            graph_means = []
            for g in graph_ids:
                available = [t['metrics'].get(key) for t in data if t['sample_id']==g and t['metrics'].get(key) is not None]
                if available:
                    graph_means.append(math.fsum(available)/len(available))
            row['secondary'][key] = {'mean':math.fsum(graph_means)/len(graph_means) if graph_means else None,
                                     'available_graphs':len(graph_means)}
        rows.append(row)
    for row in rows:
        row['cost_relative_to_dense'] = (row[DIMENSIONS[2]]/float(values['dense'][:,2].mean())
            if 'dense' in values and values['dense'][:,2].mean()>0 else None)
        row['paired_differences'] = {}
        for baseline in ('dense','bfs'):
            if baseline not in values:
                continue
            own,ref = values[row['candidate_id']],values[baseline]
            point = math.sqrt(own[:,0].mean())-math.sqrt(ref[:,0].mean())
            interval = None
            if indices is not None:
                delta = np.sqrt(own[indices,0].mean(axis=1))-np.sqrt(ref[indices,0].mean(axis=1))
                interval = list(map(float,np.percentile(delta,[2.5,97.5])))
            row['paired_differences'][baseline] = {'rmse_delta_m':point,'graph_bootstrap_95_interval':interval}
    return {'schema':'gib4cl.selection-comparison.v1','status':'complete','context':context,
            'rows':rows,'selection':choose(rows),'bootstrap':{'samples':bootstrap_samples,'seed':bootstrap_seed,
            'unit':'graph; all nodes, particle repeats and training seeds stay in the same graph cluster',
            'scope':'conditional on selected models; not independent test performance'},
            'weighted_total_used_for_ranking':False}


def write_comparison(path,report):
    path = Path(path)
    atomic_json(path,report)
    text = io.StringIO()
    writer = csv.writer(text)
    fields = ['candidate_id','baseline',*COEFFICIENTS,'effective_communication_weight',
              'effective_triangle_weight',*DIMENSIONS,'coverage95','cost_relative_to_dense']
    writer.writerow(fields)
    for row in report['rows']:
        writer.writerow([(row.get('coefficients') or {}).get(k) if k in COEFFICIENTS else row.get(k) for k in fields])
    _io_path(path.with_suffix('.csv')).write_text(text.getvalue(),encoding='utf-8')
    lines = ['# 候选共同指标比较','','定位/校准/成本分开比较；不使用加权总损失。',
             '高斯代理结果与粒子 BP 结果分别汇总，不能混排。','',
             '|候选|RMSE (m)|95% 覆盖率|通信成本|无效协方差比例|',
             '|---|---:|---:|---:|---:|']
    for row in report['rows']:
        lines.append('|{}|{:.6g}|{:.6g}|{:.6g}|{:.6g}|'.format(
            row['candidate_id'],row[DIMENSIONS[0]],row['coverage95'],row[DIMENSIONS[2]],row[DIMENSIONS[3]]))
    lines += ['','选择角色：'+str(report['selection']['roles']),
              '小均值差异不自动代表显著优越；成本代表可能精度很差。']
    _io_path(path.with_suffix('.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
