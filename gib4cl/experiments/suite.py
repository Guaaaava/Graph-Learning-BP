"""Sequential, auditable fixed-data experiment panels with independent seed roles."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from gib4cl.config import REPO_ROOT,load_config,config_from_dict,with_seeds,fingerprint,validate_name,SeedConfig,_unique_object
from gib4cl.data.generate import create_dataset
from gib4cl.evaluation.metrics import summarize_records,_summary,_rng
from gib4cl.learning.checkpoints import read_checkpoint
from gib4cl.runtime import atomic_json,sha256,source_manifest,execution,runtime_info,load_split,dataset_identity

def _canonical(value):
    return json.loads(json.dumps(value,allow_nan=False))


class SuiteRunError(RuntimeError):
    pass


def _panel(values,name):
    if not isinstance(values,list) or not values:
        raise ValueError(name+' must be a nonempty list')
    for value in values:
        SeedConfig(data=value)  # same bounded non-boolean integer contract for every role
    if len(set(values)) != len(values):
        raise ValueError(name+' must not contain duplicates')
    return sorted(values)


def load_suite(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=_unique_object)
    if not isinstance(value,dict):
        raise ValueError('Suite configuration must be an object')
    allowed = {'suite_id','base_config','training_seeds','bp_seeds',
               'split','max_graphs','seed_bootstrap_samples'}
    if set(value)-allowed:
        raise ValueError('Unknown suite settings')
    validate_name(value['suite_id'],'suite_id')
    if not isinstance(value.get('base_config'),str):
        raise ValueError('base_config must name a saved experiment JSON')
    cfg = load_config(path.parent/value['base_config'])
    split = value.get('split','test')
    if split not in ('val','test'):
        raise ValueError('Suite evaluation split must be val or test')
    maximum = value.get('max_graphs')
    samples = value.get('seed_bootstrap_samples',2000)
    if maximum is not None and (isinstance(maximum,bool) or not isinstance(maximum,int) or maximum < 1):
        raise ValueError('max_graphs must be positive or null')
    if isinstance(samples,bool) or not isinstance(samples,int) or samples < 0:
        raise ValueError('seed_bootstrap_samples must be a nonnegative integer')
    return _canonical({'suite_id':value['suite_id'],'base_config':cfg.to_dict(),
            'training_seeds':_panel(value['training_seeds'],'training_seeds'),
            'bp_seeds':_panel(value['bp_seeds'],'bp_seeds'),'split':split,
            'max_graphs':maximum,'seed_bootstrap_samples':samples})


def plan_suite(spec):
    cfg = config_from_dict(spec['base_config'])
    jobs = {}
    for training_seed in spec['training_seeds']:
        training = with_seeds(cfg,training=training_seed)
        parent = 'train-'+str(training_seed)
        jobs[parent] = {'kind':'train','training_seed':training_seed,'config':_canonical(training.to_dict()),
                        'status':'pending','attempts':[]}
        for bp_seed in spec['bp_seeds']:
            evaluation = with_seeds(training,bp=bp_seed)
            jobs[f'evaluate-{training_seed}-{bp_seed}'] = {
                'kind':'evaluate','training_seed':training_seed,'bp_seed':bp_seed,'parent':parent,
                'config':_canonical(evaluation.to_dict()),'status':'pending','attempts':[]}
    return jobs


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def _artifact(root,path):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Artifact must be an existing file inside the suite directory')
    return {'path':path.relative_to(root.resolve()).as_posix(),'sha256':sha256(path)}


def _checked(root,artifact):
    path = (root/artifact['path']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path)!=artifact['sha256']:
        raise ValueError('Completed artifact missing or checksum mismatch: '+artifact['path'])
    return path


def _verify_job(root,job,jobs,data_info,split):
    for attempt in job['attempts']:
        for key in ('config_artifact','log_artifact'):
            if key in attempt:
                _checked(root,attempt[key])
    artifacts = job['artifacts']
    paths = {name:_checked(root,value) for name,value in artifacts.items()}
    cfg = config_from_dict(job['config'])
    if job['kind']=='train':
        last,best = read_checkpoint(paths['last_checkpoint']),read_checkpoint(paths['best_checkpoint'])
        if last['epoch']!=cfg.training.epochs or last['config']!=cfg.to_dict() or best['config']!=cfg.to_dict():
            raise ValueError('Completed training job configuration or epoch differs')
        if any(dataset_identity(last['datasets'][key])!=dataset_identity(data_info[key]) for key in ('train','val')):
            raise ValueError('Completed training job used another dataset')
    else:
        result,manifest = _read(paths['results']),_read(paths['manifest'])
        parent = jobs[job['parent']]
        if manifest['status']!='complete' or result['config']!=_canonical(cfg.to_dict()):
            raise ValueError('Completed evaluation job configuration differs')
        if result['checkpoint']['sha256']!=parent['artifacts']['best_checkpoint']['sha256']:
            raise ValueError('Evaluation checkpoint differs from its training parent')
        if dataset_identity(result['dataset'])!=dataset_identity(data_info[split]):
            raise ValueError('Evaluation job used another dataset')
        for name,key in [('results','results_sha256'),('report','report_sha256'),
                         ('trials','trial_metrics_sha256'),('summary_csv','summary_metrics_sha256')]:
            if manifest[key]!=artifacts[name]['sha256']:
                raise ValueError('Evaluation output/manifest checksums disagree')


def _prepare_data(cfg,root,reuse):
    info = {}
    for split in ('train','val','test'):
        if cfg.split_sizes.get(split,0)<1:
            raise ValueError('Suite requires nonempty train/val/test splits')
    existing = any(cfg.dataset_path(s,root).exists() or cfg.dataset_path(s,root).with_suffix('.json').exists()
                   for s in ('train','val','test'))
    if existing and not reuse:
        raise FileExistsError('Suite data exists; use --reuse-data to validate fixed data explicitly')
    for split in ('train','val','test'):
        path = cfg.dataset_path(split,root)
        if not path.exists():
            if path.with_suffix('.json').exists():
                raise ValueError('Incomplete dataset pair: '+str(path))
            create_dataset(split,cfg.split_sizes[split],settings=cfg,output_root=root)
        if not path.with_suffix('.json').is_file():
            raise ValueError('Suite requires a checksum sidecar: '+str(path))
        _,info[split] = load_split(cfg,split,root)
        if info[split]['num_graphs']!=cfg.split_sizes[split]:
            raise ValueError('Suite split size differs from the fixed dataset')
    return info


def _run_child(command,log_path,training_seed):
    env = dict(os.environ,PYTHONHASHSEED=str(training_seed%(2**32)),PYTHONUNBUFFERED='1')
    with log_path.open('w',encoding='utf-8') as handle:
        handle.write('command: '+json.dumps(command,ensure_ascii=False)+'\n')
        handle.flush()
        return subprocess.run(command,cwd=REPO_ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode


def _execute(root,job_id,job,jobs,data_root,spec,device,threads):
    attempt_dir = root/'jobs'/job_id/f"attempt-{len(job['attempts'])+1:03d}"
    attempt_dir.mkdir(parents=True,exist_ok=False)
    cfg = config_from_dict(job['config'])
    config_path = attempt_dir/'config.json'
    atomic_json(config_path,cfg.to_dict())
    output = attempt_dir/'artifacts'
    command = [sys.executable,'-B','-m','gib4cl.experiments.worker',job['kind'],'--config',str(config_path),
               '--data-root',str(data_root),'--destination',str(output),
               '--device',device,'--threads',str(threads)]
    if job['kind']=='evaluate':
        parent = jobs[job['parent']]
        command += ['--checkpoint',str(_checked(root,parent['artifacts']['best_checkpoint'])),
                    '--split',spec['split']]
        if spec['max_graphs'] is not None:
            command += ['--max-graphs',str(spec['max_graphs'])]
    attempt = {'status':'running','started_at':_now(),'directory':attempt_dir.relative_to(root).as_posix(),
               'log':(attempt_dir/'run.log').relative_to(root).as_posix(),'command':command}
    attempt['config_artifact'] = _artifact(root,config_path)
    job['attempts'].append(attempt)
    job['status'] = 'running'
    return attempt,command,output,cfg


def _collect(root,job,output,cfg):
    if job['kind']=='train':
        models = output/'models'
        return {name:_artifact(root,path) for name,path in {
            'best_checkpoint':models/'best_checkpoint.pt','last_checkpoint':models/'last_checkpoint.pt',
            'run_manifest':models.parent/'run.json','history':models.parent/'history.json'}.items()}
    results = list(output.rglob('results.json'))
    if len(results)!=1:
        raise ValueError('Expected exactly one evaluation result in the fresh attempt directory')
    directory = results[0].parent
    return {name:_artifact(root,directory/file) for name,file in {
        'results':'results.json','manifest':'manifest.json','report':'report.md',
        'trials':'trial_metrics.csv','summary_csv':'summary_metrics.csv'}.items()}


def _training_stat(values,cfg,spec,domain):
    stat = _summary(values,cfg.evaluation.alpha,spec['seed_bootstrap_samples'],
                    _rng(cfg.seeds.bootstrap,['training-panel',domain]))
    stat['training_seeds'] = stat.pop('graphs')
    return stat


def summarize_suite(root,manifest):
    spec,jobs = manifest['spec'],manifest['jobs']
    cfg = config_from_dict(spec['base_config'])
    per_seed = {}
    for seed in spec['training_seeds']:
        panel = [jobs[f'evaluate-{seed}-{bp}'] for bp in spec['bp_seeds']]
        if not all(job['status']=='complete' for job in panel):
            continue
        records = []
        for panel_index,job in enumerate(panel):
            result = _read(_checked(root,job['artifacts']['results']))
            for original in result['records']:
                # Preserve graph clusters, make BP repeat IDs unique across the fixed BP-seed panel.
                records.append(dict(original,bp_repeat=panel_index*cfg.evaluation.bp_repeats+original['bp_repeat']))
        per_seed[str(seed)] = summarize_records(records,alpha=cfg.evaluation.alpha,
            bootstrap_samples=cfg.evaluation.bootstrap_samples,seed=cfg.seeds.bootstrap)
    status = 'complete' if all(job['status']=='complete' for job in jobs.values()) else 'partial'
    result = {'status':status,'suite_id':spec['suite_id'],
              'planned_training_seeds':spec['training_seeds'],'used_training_seeds':list(map(int,per_seed)),
              'missing_training_seeds':[s for s in spec['training_seeds'] if str(s) not in per_seed],
              'bp_seed_panel':spec['bp_seeds'],'datasets':manifest.get('datasets',{}),'per_training_seed':per_seed,
              'graph_uncertainty_scope':'Within each model: independent sample_id clusters, BP seeds/repeats kept together.',
              'training_uncertainty_scope':'Across independently initialized training seeds, conditional on the fixed dataset and fixed BP seed panel.',
              'joint_uncertainty_scope':'No joint data-and-training confidence interval is claimed.',
              'fixed_baseline_scope':'Dense/BFS reuse the same data and BP supports for every model; repeated copies are not additional independent experiments.'}
    if not per_seed:
        return result
    first = next(iter(per_seed.values()))
    result['unique_evaluation_graphs'] = first['topologies']['gnn']['graphs']
    result['fixed_baselines'] = {name:first['topologies'][name] for name in ('dense','bfs')}
    # A baseline must be identical across models on the same panel, not counted again as new data.
    for summary in per_seed.values():
        for name in ('dense','bfs'):
            if summary['topologies'][name]!=result['fixed_baselines'][name]:
                raise ValueError('Fixed-data/fixed-BP baseline changed across training seeds')
    learned = {}
    for metric in first['topologies']['gnn']['graph_metrics']:
        values = [s['topologies']['gnn']['graph_metrics'][metric]['mean'] for s in per_seed.values()]
        learned[metric] = _training_stat(values,cfg,spec,['gnn',metric])
    result['gnn_across_training_seeds'] = learned
    result['paired_gnn_minus_dense_across_training_seeds'] = {
        metric:_training_stat([s['paired_graph_differences']['gnn_minus_dense'][metric]['mean'] for s in per_seed.values()],
                             cfg,spec,['paired',metric])
        for metric in first['paired_graph_differences']['gnn_minus_dense']}
    for curve,xkey,fields in [('outage_curve','threshold',('outage','error_cdf')),
                              ('coverage_curve','nominal',('coverage','coverage_valid'))]:
        result['gnn_'+curve+'_across_training_seeds'] = [
            {xkey:point[xkey],**{field:_training_stat(
                [s['topologies']['gnn'][curve][i][field]['mean'] for s in per_seed.values()],
                cfg,spec,[curve,point[xkey],field]) for field in fields}}
            for i,point in enumerate(first['topologies']['gnn'][curve])]
    return result


def _write_summary(root,manifest):
    summary = summarize_suite(root,manifest)
    path = root/('summary.json' if summary['status']=='complete' else 'partial_summary.json')
    atomic_json(path,summary)
    manifest['summary'] = _artifact(root,path)
    csv_path = path.with_suffix('.csv')
    with csv_path.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['scope','metric','mean','std','ci_low','ci_high','training_seeds'])
        writer.writeheader()
        for scope,key in [('gnn','gnn_across_training_seeds'),('gnn_minus_dense','paired_gnn_minus_dense_across_training_seeds')]:
            for metric,stat in summary.get(key,{}).items():
                writer.writerow(dict(scope=scope,metric=metric,**stat))
    manifest['summary_csv'] = _artifact(root,csv_path)
    lines = ['# Multi-seed experiment panel','',f"Status: {summary['status']}",
             f"Completed training seeds: {summary['used_training_seeds']}",
             f"Missing training seeds: {summary['missing_training_seeds']}",
             'The test dataset is fixed. BP seeds and repeats are clustered within each graph.',
             'Across-training intervals condition on this fixed dataset and BP panel; baseline copies add no independent observations.',
             'No joint data-and-training confidence interval is claimed.','']
    if summary.get('gnn_across_training_seeds'):
        lines += ['| Metric | GNN mean | Training-seed std | Training-seed CI | Seeds |',
                  '| --- | ---: | ---: | ---: | ---: |']
        fmt = lambda v:'n/a' if v is None else f'{v:.6g}'
        for metric in ('rmse','outage','ellipse_coverage','communication_cost','agent_triangles','unanchored_agents'):
            stat = summary['gnn_across_training_seeds'][metric]
            lines.append(f"| {metric} | {fmt(stat['mean'])} | {fmt(stat['std'])} | "+
                         f"[{fmt(stat['ci_low'])}, {fmt(stat['ci_high'])}] | {stat['training_seeds']} |")
    report = root/('report.md' if summary['status']=='complete' else 'partial_report.md')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['report'] = _artifact(root,report)


def run_suite(*,suite_config,data_root=None,output_root=None,resume=False,reuse_data=False,
              retry_failed=False,stop_after_jobs=None,device='auto',threads=1,plan_only=False):
    spec = load_suite(suite_config)
    cfg = config_from_dict(spec['base_config'])
    if plan_only:
        return {'spec':spec,'jobs':plan_suite(spec)}
    cfg.objective.require_ready()
    if retry_failed and not resume:
        raise ValueError('--retry-failed requires --resume')
    if stop_after_jobs is not None and (isinstance(stop_after_jobs,bool) or not isinstance(stop_after_jobs,int) or stop_after_jobs<1):
        raise ValueError('stop_after_jobs must be a positive integer')
    target = execution(device,1 if threads is None else threads,cfg.training.deterministic)
    runtime = runtime_info(target)
    root = (Path(output_root) if output_root is not None else REPO_ROOT/'outputs/suites')/spec['suite_id']/fingerprint(spec)[:16]
    root = root.resolve()
    data_root = Path(data_root).resolve() if data_root is not None else root/'data'
    identity = {'spec':spec,'source_sha256':source_manifest(),'runtime':runtime}
    path = root/'manifest.json'
    if resume:
        manifest = _read(path)
        if manifest['identity']!=identity:
            raise ValueError('Suite resume configuration/source/runtime differs; choose a new output root')
        if {k:{n:j[n] for n in plan_suite(spec)[k] if n not in ('status','attempts')} for k,j in manifest['jobs'].items()} != {
                k:{n:v for n,v in j.items() if n not in ('status','attempts')} for k,j in plan_suite(spec).items()}:
            raise ValueError('Suite saved job plan differs')
    else:
        root.mkdir(parents=True,exist_ok=False)
        manifest = {'spec':spec,'identity':identity,'created_at':_now(),
                    'status':'pending','jobs':plan_suite(spec),'data_root':str(data_root)}
        atomic_json(path,manifest)
    try:
        info = _prepare_data(cfg,data_root,reuse_data or resume)
        if manifest.get('datasets') and any(dataset_identity(info[k])!=dataset_identity(manifest['datasets'][k]) for k in info):
            raise ValueError('Suite fixed dataset identity/checksum differs')
        for job in manifest['jobs'].values():
            if job['status']=='complete':
                _verify_job(root,job,manifest['jobs'],info,spec['split'])
        if manifest['status']=='complete':
            for key in ('summary','summary_csv','report'):
                _checked(root,manifest[key])
            print('Suite already complete; all artifacts verified: '+str(root))
            return root
        manifest.update(status='running',datasets=info,data_root=str(data_root))
        atomic_json(path,manifest)
        executed = 0
        for job_id,job in manifest['jobs'].items():
            if source_manifest()!=identity['source_sha256']:
                raise ValueError('Production source changed during suite execution')
            if job['status']=='complete':
                continue
            if stop_after_jobs is not None and executed>=stop_after_jobs:
                break
            if job['status'] in ('failed','running','interrupted','blocked') and not retry_failed:
                continue
            if job['kind']=='evaluate' and manifest['jobs'][job['parent']]['status']!='complete':
                job.update(status='blocked',error='Training parent has not completed')
                atomic_json(path,manifest)
                continue
            attempt,command,output,job_cfg = _execute(root,job_id,job,manifest['jobs'],data_root,spec,str(target),runtime['num_threads'])
            atomic_json(path,manifest)  # persist running state before spawning
            started = time.perf_counter()
            try:
                code = _run_child(command,root/attempt['log'],job['training_seed'])
                attempt['exit_code'] = code
                if code!=0:
                    raise RuntimeError('Child exited with code '+str(code)+'; see '+attempt['log'])
                job['artifacts'] = _collect(root,job,output,job_cfg)
                _verify_job(root,job,manifest['jobs'],info,spec['split'])
                attempt['status'] = job['status'] = 'complete'
                job.pop('error',None)
            except Exception as error:
                attempt['status'] = job['status'] = 'failed'
                attempt['error'] = job['error'] = str(error)
                attempt['traceback'] = traceback.format_exc()
            except KeyboardInterrupt:
                attempt['status'] = job['status'] = 'interrupted'
                raise
            finally:
                attempt.update(finished_at=_now(),elapsed_seconds=time.perf_counter()-started)
                if (root/attempt['log']).is_file():
                    attempt['log_artifact'] = _artifact(root,root/attempt['log'])
                atomic_json(path,manifest)
            executed += 1
            print(f"Suite {job_id}: {job['status']}; attempt={len(job['attempts'])}",flush=True)
        failed = any(job['status'] in ('failed','blocked','running','interrupted') for job in manifest['jobs'].values())
        complete = all(job['status']=='complete' for job in manifest['jobs'].values())
        manifest['status'] = 'complete' if complete else 'failed' if failed else 'partial'
        _write_summary(root,manifest)
        manifest['updated_at'] = _now()
        atomic_json(path,manifest)
        if failed:
            raise SuiteRunError('Suite contains failed/blocked jobs; completed jobs preserved. Review logs and use --resume --retry-failed: '+str(root))
        return root
    except (Exception,KeyboardInterrupt) as error:
        if not isinstance(error,SuiteRunError):
            manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                            error=str(error),updated_at=_now())
            atomic_json(path,manifest)
        raise
