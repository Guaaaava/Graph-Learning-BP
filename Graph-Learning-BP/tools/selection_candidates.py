"""Explicit A/B/C/D candidate stages and same-schedule promotions."""
from dataclasses import replace
from itertools import product

from gib4cl.config import fingerprint, with_seeds
from gib4cl.learning.train import train_gnn
from gib4cl.learning.checkpoints import read_checkpoint
from gib4cl.runtime import sha256
from tools.selection_state import WorkLimit, read_json, freeze_copy
from tools.selection_compare import coefficients, candidate_id, ordering, summarize, write_comparison
from tools.selection_evaluate import prepare_data, evaluate_panel


def complete(workspace,stage):
    report = workspace.report('stage-'+stage)
    if report['status'] != 'complete':
        raise ValueError('Preceding stage is not complete: '+stage)
    return report


def accuracy_coefficients(workspace,stage):
    complete(workspace,stage)
    comparison = workspace.report('comparison-'+stage)
    primary = comparison['selection']['primary']
    return next(r['coefficients'] for r in comparison['rows'] if r['candidate_id']==primary)


def grid(workspace,protocol,stage):
    search = protocol['coefficient_search']
    first = search['first_pass']
    if stage=='A':
        values = [dict(beta=b,gamma=0,kappa=0,eta=e) for b,e in product(first['beta'],first['eta'])]
    elif stage=='B':
        base = accuracy_coefficients(workspace,'A')
        values = [base]+[{**base,'gamma':g,'kappa':0} for g in first['gamma'] if g>0]
        values += [{**base,'gamma':0,'kappa':k} for k in first['kappa'] if k>0]
    elif stage=='C':
        base = accuracy_coefficients(workspace,'A')
        complete(workspace,'B')
        rows = workspace.report('comparison-B')['rows']
        def best_values(name,other):
            branch = [r for r in rows if not r['baseline'] and r['coefficients'][name]>0 and r['coefficients'][other]==0]
            return [0]+[r['coefficients'][name] for r in sorted(branch,key=ordering)[:2]]
        values = [{**base,'gamma':g,'kappa':k} for g,k in product(best_values('gamma','kappa'),best_values('kappa','gamma'))]
    elif stage=='D':
        base = accuracy_coefficients(workspace,'C')
        neighbors = next(v for k,v in search['eta_neighborhoods'].items() if float(k)==base['eta'])
        values = [{**base,'beta':b,'eta':e} for b,e in product(first['beta'],neighbors)]
    else:
        raise ValueError('Unknown screening stage')
    unique = {candidate_id(v):coefficients(v) for v in values}
    if any(v[k] not in first[k] for v in unique.values() for k in first):
        raise ValueError('Stage attempted unapproved coefficient expansion')
    known = set()
    for previous in ('A','B','C','D'):
        if 'stage-'+previous in workspace.manifest['reports']:
            known.update(m['candidate_id'] for m in workspace.report('stage-'+previous).get('models',[]))
    if len(known|set(unique)) > search['max_unique_combinations']:
        raise ValueError('Search exceeds declared unique coefficient budget')
    return list(unique.values())


def train_candidate(workspace,cfg,coef,seed,target,device,threads):
    coef = coefficients(coef)
    cid = candidate_id(coef)
    model_id = cid+'-s'+str(seed)
    config = with_seeds(replace(cfg,objective=replace(cfg.objective,**coef)),training=seed)
    key = {'type':'candidate-training','coefficients':coef,'training_seed':seed,'target_epoch':target}
    def run():
        directory = workspace.root/'runs'/model_id
        attempts = [(0,directory)] if directory.exists() else []
        for path in (workspace.root/'runs').glob(model_id+'-retry*'):
            suffix = path.name.split('-retry')[-1]
            if suffix.isdigit():
                attempts.append((int(suffix),path))
        if attempts:
            attempt,directory = max(attempts,key=lambda row:row[0])
            if not (directory/'models/last_checkpoint.pt').exists():
                if not workspace.retry:
                    raise ValueError('No complete epoch checkpoint; retain partial run and explicitly retry_failed to restart its seed')
                directory = workspace.root/'runs'/(model_id+'-retry'+str(attempt+1))
        last = directory/'models/last_checkpoint.pt'
        data_root = workspace.root/'data'
        stop = target
        if last.exists():
            saved = read_checkpoint(last)
            info = read_json(directory/'run.json')
            best_state = read_checkpoint(directory/'models/best_checkpoint.pt')
            if (info.get('last_epoch')!=saved['epoch'] or info.get('best_epoch')!=saved['best_epoch'] or
                    info.get('best_metric')!=saved['best_metric'] or info.get('last_model_digest')!=saved['model_digest'] or
                    best_state['epoch']!=saved['best_epoch'] or best_state['best_metric']!=saved['best_metric'] or
                    read_json(directory/'history.json')!=saved['history']):
                raise ValueError('Training epoch files do not form a committed boundary; inspect retained run before recovery')
            if saved['epoch']>target:
                raise ValueError('Later model exists without this frozen milestone; refuse look-ahead snapshot')
            if workspace.max_training_epochs is not None:
                stop = min(target,saved['epoch']+workspace.max_training_epochs)
            models = train_gnn(resume=last,settings=config,data_root=data_root,device=str(device),threads=threads,
                               stop_after_epoch=stop,stop_requested=workspace.should_stop)
        else:
            if workspace.max_training_epochs is not None:
                stop = min(target,workspace.max_training_epochs)
            models = train_gnn(settings=config,data_root=data_root,run_directory=directory,device=str(device),
                               threads=threads,stop_after_epoch=stop,stop_requested=workspace.should_stop)
        state = read_checkpoint(models/'last_checkpoint.pt')
        if state['epoch'] < target:
            raise WorkLimit()
        if state['epoch'] != target:
            raise ValueError('Candidate did not reach the requested epoch')
        best = freeze_copy(models/'best_checkpoint.pt',workspace.root/'checkpoints'/f'{model_id}-e{target}-best.pt')
        final = freeze_copy(models/'last_checkpoint.pt',workspace.root/'checkpoints'/f'{model_id}-e{target}-last.pt')
        info = read_json(directory/'run.json')
        value = {'model_id':model_id,'candidate_id':cid,'baseline':False,'coefficients':coef,
                 'training_seed':seed,'target_epoch':target,'selected_epoch':read_checkpoint(best)['epoch'],
                 'checkpoint':workspace.artifact(best),'last_snapshot':workspace.artifact(final),
                 'initial_model_digest':info['initial_model_digest'],
                 'training_run_directory':directory.relative_to(workspace.root).as_posix(),
                 'effective_communication_weight':coef['beta']*coef['gamma'],
                 'effective_triangle_weight':coef['beta']*coef['kappa']}
        return value,[best,final]
    return workspace.job(key,run)


def compare_panel(workspace,cfg,specs,*,graphs,method,device,name,bp_settings=None):
    records = evaluate_panel(workspace,cfg,specs,graphs=graphs,method=method,device=device,bp_settings=bp_settings)
    workspace.publish('records-'+name,records)
    report = summarize(records,bootstrap_samples=cfg.evaluation.bootstrap_samples,bootstrap_seed=cfg.seeds.bootstrap)
    workspace.publish('comparison-'+name,report)
    write_comparison(workspace.root/'tables'/('comparison-'+name+'.json'),report)
    return report


def screening_pool(workspace):
    models = {}
    for stage in ('A','B','C','D'):
        for spec in complete(workspace,stage)['models']:
            if spec['model_id'] in models and models[spec['model_id']] != spec:
                raise ValueError('Conflicting frozen screen milestone')
            models[spec['model_id']] = spec
    return list(models.values())


def run_candidates(workspace,cfg,protocol,*,stage,device,threads,bp_settings=None,data_root=None):
    search = protocol['coefficient_search']
    if cfg.training.epochs != search['configured_epochs']:
        raise ValueError('Training schedule must match protocol configured_epochs')
    screen,promote,final = search['screening_stop_epoch'],search['promoted_stop_epoch'],search['configured_epochs']
    if not 1 <= screen < promote <= final:
        raise ValueError('Require screen < promotion <= full configured schedule')
    if stage in ('A','B','C','D'):
        values = grid(workspace,protocol,stage)
        seeds,target,method = [search['screening_training_seed']],screen,'gaussian_proxy'
        graphs = protocol['data_policy']['light_validation_prefix_graphs']
    else:
        if bp_settings is None:
            raise ValueError('Promoted/final comparisons require an explicit common BP budget')
        previous = 'screen' if stage=='promote' else 'promote'
        if stage=='promote':
            complete(workspace,'D')
        else:
            complete(workspace,'promote')
        comparison = workspace.report('comparison-'+previous)
        ids = comparison['selection']['representatives']
        maximum = search['max_promoted_candidates'] if stage=='promote' else search['max_final_candidates']
        if len(ids)>maximum:
            raise ValueError('Representative count exceeds declared promotion limit')
        values = [r['coefficients'] for r in comparison['rows'] if r['candidate_id'] in ids]
        seeds = [search['screening_training_seed']] if stage=='promote' else search['final_training_seeds']
        target,method = (promote if stage=='promote' else final),'particle_bp'
        key = 'initial' if stage=='promote' else 'final'
        graphs = protocol['candidate_bp_review'][key]['validation_prefix_graphs']
        if bp_settings['seeds'] != protocol['candidate_bp_review'][key]['bp_seeds']:
            raise ValueError('Particle repeat panel differs from declared review phase')
    workspace.bind('candidate-stage-'+stage,{'coefficients':values,'seeds':seeds,'target':target,
                   'graphs':graphs,'method':method,'bp_settings':bp_settings})
    if not 1 <= graphs <= cfg.split_sizes['val']:
        raise ValueError('Comparison graph prefix exceeds val')
    prepare_data(workspace,cfg,source_root=data_root)
    specs = []
    try:
        for coef in values:
            for seed in seeds:
                specs.append(train_candidate(workspace,cfg,coef,seed,target,device,threads))
        # Same-seed candidate starts must be paired, including after stage reuse.
        for seed in seeds:
            if len({s['initial_model_digest'] for s in specs if s['training_seed']==seed}) != 1:
                raise ValueError('Candidate initializations are not paired by training seed')
        comparison = compare_panel(workspace,cfg,specs,graphs=graphs,method=method,device=device,
                                   name=stage,bp_settings=bp_settings)
        result = workspace.publish('stage-'+stage,{'status':'complete','models':specs,
            'target_epoch':target,'comparison_report':'comparison-'+stage,'selection':comparison['selection']})
        if stage=='D':
            compare_panel(workspace,cfg,screening_pool(workspace),graphs=graphs,method='gaussian_proxy',
                          device=device,name='screen')
        return result
    except WorkLimit:
        return workspace.publish('stage-'+stage,{'status':'partial','models':specs,'target_epoch':target,
                                   'note':'No ranking or automatic next stage until all planned jobs complete'})
