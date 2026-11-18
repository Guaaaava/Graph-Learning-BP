"""One explicit checkpoint -> paired Dense/BFS/GNN evaluation and strict reports."""
from __future__ import annotations
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import torch

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gib4cl.inference import ParticleBP
from gib4cl.config import REPO_ROOT,load_config,config_from_dict,fingerprint,validate_name,with_seeds
from gib4cl.learning.checkpoints import read_checkpoint,load_selector
from gib4cl.evaluation.metrics import (METRICS_VERSION,SCALAR_METRICS,metric_protocol,calculate_metrics,
                                  bp_diagnostic_metrics,summarize_records)
from gib4cl.evaluation.topology import select_topologies,selection_metrics
from gib4cl.runtime import (atomic_json,execution,runtime_info,load_split,
                                  dataset_identity,sha256,state_digest,source_manifest)


# checkpoint + evaluation configuration + test dataset
#                       ↓
#               Load trained GNN
#                       ↓
#       Generate Dense／BFS／GNN for each graph
#                       ↓
#           Get a set of shared prior BP particle
#                       ↓
#       3 topology run BP and calculate metrics
#                       ↓
# Summarize across different graphs、bootstrap、comparison
#                       ↓
#           JSON + CSV + Markdown report

def bp_seed(master_seed,dataset_id,sample_id,repeat):
    value = json.dumps(['bp-evaluation-v1',master_seed,dataset_id,sample_id,repeat],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'big') & ((1<<63)-1)


def render_report(result):
    summary = result['summary']
    def number(value):
        return 'n/a' if value is None else f'{value:.5g}'
    def confidence(stat):
        return number(stat['mean'])+' ['+number(stat['ci_low'])+', '+number(stat['ci_high'])+']'
    def table(columns,rows):
        return ['| '+' | '.join(columns)+' |','| '+' | '.join(['---']+['---:']*(len(columns)-1))+' |',
                *['| '+' | '.join(map(str,row))+' |' for row in rows],'']
    names = list(summary['topologies'])
    def mean(name,metric):
        return number(summary['topologies'][name]['graph_metrics'][metric]['mean'])
    cfg = result['config']['evaluation']
    lines = ['# Localization evaluation','',
             f"Graphs: {result['num_graphs']}; BP repeats per graph: {cfg['bp_repeats']}.",
             f"Metrics protocol: {result['metric_protocol']['version']}",
             f"Checkpoint: {result['checkpoint']['path']}",
             f"Checkpoint SHA-256: {result['checkpoint']['sha256']}",'',
             '## Accuracy','']
    lines += table(
        ['Topology','Graph RMSE and CI (m)','Graph RMSE std','MAE (m)','Outage','Pooled error p95 (m)','Pooled RMSE (m)'],
        [[name,confidence(summary['topologies'][name]['graph_metrics']['rmse']),
          number(summary['topologies'][name]['graph_metrics']['rmse']['std']),
          mean(name,'mae'),mean(name,'outage'),number(summary['topologies'][name]['agent_error_quantiles']['p95']),
          number(summary['topologies'][name]['pooled_agent_run_rmse'])] for name in names])
    lines += [f"Outage is error > {cfg['outage_threshold']} m; equality is successful localization.",
              'Pooled agent-run values and quantiles are descriptive; the main columns give equal weight to independent graphs.','',
              '## Covariance calibration and sharpness','']
    lines += table(
        ['Topology','Ellipse coverage','Valid-only coverage','NEES interval pass','Below interval','Above interval',
         'Valid mean NEES','Gaussian NLL (nats)','Ellipse area (m^2)','Invalid agent-runs'],
        [[name,*[mean(name,k) for k in ('ellipse_coverage','ellipse_coverage_valid','nees_interval_pass',
                                       'nees_below_interval','nees_above_interval','mean_nees_valid',
                                       'gaussian_nll_valid','ellipse_area_valid')],
          summary['topologies'][name]['invalid_covariance_agent_runs']] for name in names])
    lines += ['Ellipse coverage and the two-sided NEES interval pass rate are separate metrics.',
              f"Ellipse nominal coverage: {1-cfg['alpha']:.5g}; two-sided NEES tail mass: {cfg['alpha']:.5g}.",
              'Invalid covariance fails all-agent coverage/interval checks. Valid-only columns exclude it and retain denominator counts.',
              'Gaussian NLL includes the 2D normalizing constant. Ellipse areas and NLL describe moment-matched Gaussian marginals, not particle HPD regions or the true BP density.',
              'NEES tails/coverage are empirical consistency diagnostics; they do not prove directional ordering against the exact posterior.','',
              '## Selected topology and communication cost','']
    lines += table(
        ['Topology','Mean edges','Edge saving','Model cost','Cost saving','Agent triangles','Agent cycle rank',
         'Agent degree','Unanchored agents'],
        [[name,*[mean(name,k) for k in ('edges','edge_saving','communication_cost','communication_saving',
                                       'agent_triangles','agent_cycle_rank','agent_degree_mean','unanchored_agents')]]
         for name in names])
    lines += ['Costs use rho * initial-distance^2 and include anchor edges; they are not measured joules or packet counts.',
              'Cycle rank = agent edges - agents + agent components. Known anchors do not form unknown-variable feedback cycles.',
              'Saving = 1 - selected/candidate, computed within each graph; a zero denominator is n/a.',
              'Anchor reachability and degree are structural diagnostics, not proofs of localization rigidity.','',
              '## BP numerical diagnostics','']
    lines += table(
        ['Topology','BP trials','Unconverged trials','Min ESS','Min ESS / P','Low-ESS agent-runs','Iterations','Message residual'],
        [[name,summary['topologies'][name]['bp_trials'],summary['topologies'][name]['bp_unconverged_trials'],
          mean(name,'ess_min'),mean(name,'ess_fraction_min'),summary['topologies'][name]['low_ess_agent_runs'],
          mean(name,'bp_iterations'),mean(name,'bp_message_residual')] for name in names])
    lines += [f"Low ESS means ESS/P < {cfg['ess_fraction_threshold']}; invalid covariance, low ESS and unconverged trials remain in the evaluation. ",'',
              '## Error CDF and outage curves','']
    axis = summary['topologies'][names[0]]['outage_curve']
    lines += table(['Threshold (m)',*[name+' CDF' for name in names],*[name+' outage' for name in names]],
                   [[number(point['threshold']),
                     *[number(summary['topologies'][name]['outage_curve'][i]['error_cdf']['mean']) for name in names],
                     *[number(summary['topologies'][name]['outage_curve'][i]['outage']['mean']) for name in names]]
                    for i,point in enumerate(axis)])
    lines += ['## Gaussian ellipse calibration curve','']
    axis = summary['topologies'][names[0]]['coverage_curve']
    lines += table(['Nominal',*[name+' coverage and CI' for name in names]],
                   [[number(point['nominal']),
                     *[confidence(summary['topologies'][name]['coverage_curve'][i]['coverage']) for name in names]]
                    for i,point in enumerate(axis)])
    lines += ['## Paired differences from Dense','']
    lines += table(['Comparison','Metric','Mean difference and CI','Available graphs'],
                   [[name,metric,confidence(stat),stat['graphs']]
                    for name,values in summary['paired_graph_differences'].items()
                    for metric,stat in values.items()
                    if metric in ('rmse','outage','ellipse_coverage','communication_cost','agent_triangles')])
    lines += [f"Rates are fractions in [0,1]. Confidence level: {summary['confidence_level']:.5g}.",
              'Confidence intervals use independent graphs as clusters; all BP repeats and agents remain together.',
              'Single-graph standard deviations and confidence intervals are n/a. Conditional metrics include only available trials/graphs; their counts are saved.',
              'Intervals condition on this training checkpoint and do not include training-seed variation.','']
    lines += ['results.json retains all graph identities, masks, seeds, estimates, covariances, per-agent metrics and BP diagnostics.',
              'trial_metrics.csv contains scalar values per graph/repeat/topology; summary_metrics.csv contains graph means, std, counts and confidence intervals, including curves.',
              'Runtime/peak memory benchmarking and multiple training seeds are separate checks; this report does not estimate them.',
              'A completed run is a workflow check, not evidence of trained-model quality or sufficient particle count.','']
    return '\n'.join(lines)


def write_metric_tables(destination,result):
    """Flat analysis exports. JSON retains the full definitions and nested records."""
    with (destination/'trial_metrics.csv').open('w',encoding='utf-8',newline='') as handle:
        fields = ['sample_id','bp_repeat','bp_seed','topology','support_digest','num_agents',*SCALAR_METRICS]
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for record in result['records']:
            values = {key:record[key] for key in fields[:5]}
            values.update({key:record['metrics'].get(key) for key in fields[5:]})
            writer.writerow(values)
    with (destination/'summary_metrics.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['comparison','topology','metric','axis_value','mean','std','graphs','ci_low','ci_high'])
        writer.writeheader()
        for name,values in result['summary']['topologies'].items():
            for metric,stat in values['graph_metrics'].items():
                writer.writerow(dict(comparison='absolute',topology=name,metric=metric,**stat))
            for curve,xkey,fields in [('outage_curve','threshold',('outage','error_cdf')),
                                      ('coverage_curve','nominal',('coverage','coverage_valid'))]:
                for point in values[curve]:
                    for field in fields:
                        writer.writerow(dict(comparison='absolute',topology=name,metric=curve+'.'+field,
                                             axis_value=point[xkey],**point[field]))
        for name,values in result['summary']['paired_graph_differences'].items():
            for metric,stat in values.items():
                writer.writerow(dict(comparison=name,topology=name.split('_minus_')[0],metric=metric,**stat))


def evaluate(*,checkpoint,settings=None,config_path=None,data_root=None,output_root=None,
             split='test',max_graphs=None,evaluation_name=None,device='auto',threads=None,
             data_seed=None,bp_master_seed=None,bootstrap_seed=None,evaluation_directory=None):
    if settings is not None and config_path is not None:
        raise ValueError('Pass settings or config_path, not both')
    saved = read_checkpoint(checkpoint)
    cfg = settings if settings is not None else load_config(config_path) if config_path is not None else config_from_dict(saved['config'])
    cfg = with_seeds(cfg,data=data_seed,bp=bp_master_seed,bootstrap=bootstrap_seed)
    validate_name(split,'split')
    if max_graphs is not None and (isinstance(max_graphs,bool) or not isinstance(max_graphs,int) or max_graphs < 1):
        raise ValueError('max_graphs must be positive; an empty evaluation is not a result')
    if evaluation_name is not None:
        validate_name(evaluation_name,'evaluation_name')
    if evaluation_directory is not None and (output_root is not None or evaluation_name is not None):
        raise ValueError('Explicit evaluation_directory and output_root/evaluation_name are mutually exclusive')
    target = execution(device,threads,cfg.training.deterministic)
    model,_,model_info = load_selector(checkpoint,cfg.model,target)
    dataset,data_info = load_split(cfg,split,data_root)
    count = len(dataset) if max_graphs is None else min(max_graphs,len(dataset))
    identity = {'metrics_version':METRICS_VERSION,
                'checkpoint_sha256':model_info['sha256'],'dataset':dataset_identity(data_info),
                'evaluation':asdict(cfg.evaluation),'bp':asdict(cfg.bp),'max_graphs':max_graphs,
                'seeds':{'bp':cfg.seeds.bp,'bootstrap':cfg.seeds.bootstrap}}
    root = Path(output_root) if output_root is not None else REPO_ROOT/'outputs'
    destination = Path(evaluation_directory) if evaluation_directory is not None else root/'evaluations'/cfg.experiment_id/fingerprint(identity)[:16]
    if evaluation_name is not None:
        destination = destination/evaluation_name
    destination.mkdir(parents=True,exist_ok=False)
    manifest = {'status':'running','identity':identity}
    atomic_json(destination/'manifest.json',manifest)
    bp = ParticleBP.from_config(cfg.bp)
    records,graphs = [],[]
    with torch.no_grad():
        for index in range(count):
            data = dataset[index].to(target)
            masks,selection = select_topologies(data,model,cfg.evaluation.edge_threshold)
            topology_metrics = {name:selection_metrics(data,mask) for name,mask in masks.items()}
            selection['selected_edge_ids'] = {name:torch.where(mask==1)[0].tolist() for name,mask in masks.items()}
            graphs.append({'sample_id':data.sample_id,'selection':selection,
                           'num_agents':len(data.agent_node_index),'physical_edges':data.num_edges})
            for repeat in range(cfg.evaluation.bp_repeats):
                seed = bp_seed(cfg.seeds.bp,data_info['dataset_id'],data.sample_id,repeat)
                generator = torch.Generator(device=target).manual_seed(seed)
                support = bp.draw_support(data,generator=generator)
                support_digest = state_digest({'points':support.points,'prior_log_weights':support.log_prior_weights,
                                                'agent_node_index':support.agent_node_index})
                for name,mask in masks.items():
                    prediction = bp.infer(data,mask,support=support)
                    metrics = calculate_metrics(prediction.mean,data.y,prediction.covariance,
                                                 cfg.evaluation.outage_threshold,cfg.evaluation.alpha,
                                                 covariance_rtol=cfg.evaluation.covariance_rtol,
                                                 outage_thresholds=cfg.evaluation.outage_thresholds,
                                                 coverage_levels=cfg.evaluation.coverage_levels)
                    metrics.update(topology_metrics[name])
                    diagnostics_metrics = bp_diagnostic_metrics(prediction,cfg.bp.num_particles,
                                                               cfg.evaluation.ess_fraction_threshold)
                    if diagnostics_metrics['unanchored_agents'] != metrics['unanchored_agents']:
                        raise ValueError('BP anchor reachability disagrees with the selected physical graph')
                    metrics.update(diagnostics_metrics)
                    diagnostics = dict(prediction.diagnostics,converged=prediction.converged,
                        message_residual=prediction.message_residual,iterations=prediction.iterations,
                        ess=prediction.ess.tolist(),anchor_reachable=prediction.anchor_reachable.tolist(),
                        covariance_min_eigenvalue=prediction.covariance_min_eigenvalue.tolist())
                    records.append({'sample_id':data.sample_id,'bp_repeat':repeat,'bp_seed':seed,
                        'topology':name,'support_digest':support_digest,'metrics':metrics,'diagnostics':diagnostics,
                        'estimates':prediction.mean.tolist(),'covariances':prediction.covariance.tolist()})
    summary = summarize_records(records,alpha=cfg.evaluation.alpha,
                                 bootstrap_samples=cfg.evaluation.bootstrap_samples,seed=cfg.seeds.bootstrap)
    result = {'config':cfg.to_dict(),'runtime':runtime_info(target),
              'metric_protocol':metric_protocol(),
              'checkpoint':model_info,'dataset':data_info,'source_sha256':source_manifest(),
              'num_graphs':count,'seed_rule':'sha256(namespace,seeds.bp,dataset_id,sample_id,bp_repeat); independent of graph order; seeds.bootstrap only controls CI resampling',
              'topology_rule':'dense, deterministic multi-source BFS forest, symmetric GNN logit threshold; no repair',
              'summary':summary,'graphs':graphs,'records':records}
    atomic_json(destination/'results.json',result)
    write_metric_tables(destination,result)
    (destination/'report.md').write_text(render_report(result),encoding='utf-8')
    manifest.update({'status':'complete','results_sha256':sha256(destination/'results.json'),
                     'trial_metrics_sha256':sha256(destination/'trial_metrics.csv'),
                     'summary_metrics_sha256':sha256(destination/'summary_metrics.csv'),
                     'report_sha256':sha256(destination/'report.md'),'checkpoint':model_info,
                     'num_graphs':count,'runtime':result['runtime']})
    atomic_json(destination/'manifest.json',manifest)
    print(f'Evaluated {count} graphs; checkpoint={Path(checkpoint).resolve()}; results={destination}')
    return destination


def main():
    from gib4cl.experiments.cli import main as unified_main
    return unified_main(['evaluate',*sys.argv[1:]])


if __name__ == '__main__':
    main()
