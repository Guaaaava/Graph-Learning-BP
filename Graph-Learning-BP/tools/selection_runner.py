"""Plan by default; explicitly execute one BP/selection stage. Never reads test."""
from __future__ import annotations
import argparse
import json
from itertools import product
from pathlib import Path

from gib4cl.config import REPO_ROOT, load_config
from gib4cl.runtime import execution, sha256
from gib4cl.learning.checkpoints import read_checkpoint
from tools.selection_state import Workspace, WorkLimit, read_json, freeze_copy
from tools.selection_candidates import run_candidates, compare_panel
from tools.selection_compare import candidate_id, coefficients, summarize, write_comparison
from tools.selection_evaluate import prepare_data
from tools.selection_iterations import run_iterations


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command',required=True)
    for name in ('iterations','candidates','review','compare'):
        sub = commands.add_parser(name)
        sub.add_argument('--config',default=str(REPO_ROOT/'configs/experiment.json'))
        sub.add_argument('--protocol',default=str(REPO_ROOT/'configs/selection/protocol.json'))
        sub.add_argument('--output-root',type=Path)
        group = sub.add_mutually_exclusive_group()
        group.add_argument('--execute',action='store_true')
        group.add_argument('--plan',action='store_true')
        sub.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        sub.add_argument('--threads',type=int,default=1)
        sub.add_argument('--resume',action='store_true')
        sub.add_argument('--retry-failed',action='store_true')
        sub.add_argument('--max-jobs',type=int)
        if name in ('candidates','review'):
            sub.add_argument('--data-root',type=Path)
            sub.add_argument('--bp-particles',type=int)
            sub.add_argument('--bp-iterations',type=int)
            sub.add_argument('--bp-chunk-size',type=int)
            sub.add_argument('--bp-seeds',type=int,nargs='+')
        if name=='candidates':
            sub.add_argument('--stage',choices=('A','B','C','D','promote','final'),required=True)
        elif name=='iterations':
            sub.add_argument('--graph-indices',type=int,nargs='+')
            sub.add_argument('--seeds',type=int,nargs='+')
            sub.add_argument('--particles',type=int)
            sub.add_argument('--iterations',type=int,nargs='+')
            sub.add_argument('--chunk-size',type=int)
            sub.add_argument('--support-reference',type=Path)
        elif name=='review':
            sub.add_argument('--checkpoint',type=Path,nargs='+',required=True)
            sub.add_argument('--method',choices=('gaussian_proxy','particle_bp'),default='particle_bp')
            sub.add_argument('--graphs',type=int)
        else:
            sub.add_argument('--input',type=Path,required=True)
    return result


def bp_settings(args,protocol):
    final = getattr(args,'stage',None)=='final'
    defaults = protocol['candidate_bp_review']['final' if final else 'initial']
    settings = {'particles':args.bp_particles if args.bp_particles is not None else protocol['candidate_bp_review']['particles'],
                'iterations':args.bp_iterations,'chunk_size':args.bp_chunk_size,
                'seeds':args.bp_seeds if args.bp_seeds is not None else defaults['bp_seeds']}
    if settings['particles']<2 or (settings['iterations'] is not None and settings['iterations']<0) or (settings['chunk_size'] is not None and settings['chunk_size']<1):
        raise ValueError('Invalid common BP budget')
    if not settings['seeds'] or len(set(settings['seeds']))!=len(settings['seeds']) or any(not 0<=s<2**63 for s in settings['seeds']):
        raise ValueError('Invalid BP repeat panel')
    if settings['iterations'] is None or settings['chunk_size'] is None:
        if args.execute:
            raise ValueError('Explicit --bp-iterations and --bp-chunk-size are required; no silent T=3 fallback')
        return None
    return settings


def main(argv=None,*,control=None,max_training_epochs=None):
    args = parser().parse_args(argv)
    cfg,protocol = load_config(args.config),read_json(args.protocol)
    if args.threads<1 or (args.max_jobs is not None and args.max_jobs<1):
        raise ValueError('threads/max-jobs must be positive')
    plan = {'plan_only':not args.execute,'command':args.command,'test_access':False,
            'automatic_next_stage':False,'config':str(Path(args.config).resolve())}
    settings = None
    if args.command=='iterations':
        section = protocol['iteration_check']
        request = {'indices':args.graph_indices if args.graph_indices is not None else section['validation_graph_indices'],
                   'seeds':args.seeds if args.seeds is not None else section['first_batch_bp_seeds'],
                   'particles':args.particles if args.particles is not None else section['particles'],
                   'iterations':args.iterations if args.iterations is not None else section['iterations'],
                   'chunk':args.chunk_size if args.chunk_size is not None else section['chunk_candidate']}
        if not request['indices'] or len(set(request['indices']))!=len(request['indices']) or min(request['indices'])<0 or max(request['indices'])>=cfg.split_sizes['val']:
            raise ValueError('Graph indices must fit val')
        if not request['seeds'] or len(set(request['seeds']))!=len(request['seeds']) or any(not isinstance(s,int) or isinstance(s,bool) or not 0<=s<2**63 for s in request['seeds']):
            raise ValueError('Invalid BP seeds')
        if request['particles']<2 or request['chunk']<1 or not request['iterations'] or min(request['iterations'])<0 or len(set(request['iterations']))!=len(request['iterations']):
            raise ValueError('Invalid iteration grid')
        plan.update(request=request,trajectories=len(request['indices'])*len(request['seeds'])*2,
                    comparison_points=len(request['indices'])*len(request['seeds'])*2*len(request['iterations']))
    elif args.command=='candidates':
        search = protocol['coefficient_search']
        if cfg.training.epochs != search['configured_epochs']:
            raise ValueError('Config schedule must equal protocol configured_epochs')
        stage = args.stage
        plan['stage'] = stage
        if stage in ('A','B','C','D'):
            plan['new_coefficient_combinations_upper_bound'] = next(r['max_new_combinations'] for r in search['stages'] if r['id']==stage)
            plan['target_epoch'] = search['screening_stop_epoch']
            plan['schedule_epochs'] = search['configured_epochs']
            plan['method'] = 'gaussian_proxy'
            if stage=='A':
                plan['coefficients'] = [coefficients(dict(beta=b,gamma=0,kappa=0,eta=e))
                    for b,e in product(search['first_pass']['beta'],search['first_pass']['eta'])]
        else:
            settings = bp_settings(args,protocol)
            phase = 'initial' if stage=='promote' else 'final'
            plan.update(bp_budget=settings,target_epoch=search['promoted_stop_epoch'] if stage=='promote' else search['configured_epochs'],
                        schedule_epochs=search['configured_epochs'],method='particle_bp',
                        graph_inferences_upper_bound=protocol['candidate_bp_review'][phase]['max_graph_inferences'])
        plan['requires_completed_stage'] = {'A':None,'B':'A','C':'B','D':'C','promote':'D','final':'promote'}[stage]
    elif args.command=='review':
        settings = bp_settings(args,protocol) if args.method=='particle_bp' else None
        plan.update(method=args.method,checkpoints=[str(p) for p in args.checkpoint],bp_budget=settings,
                    graphs=args.graphs if args.graphs is not None else protocol['candidate_bp_review']['initial']['validation_prefix_graphs'])
    else:
        plan['input'] = str(args.input)
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return plan
    if args.output_root is None:
        raise ValueError('Execution requires --output-root')
    if args.command=='compare':
        data = read_json(args.input)
        result = summarize(data,bootstrap_samples=cfg.evaluation.bootstrap_samples,bootstrap_seed=cfg.seeds.bootstrap)
        args.output_root.mkdir(parents=True,exist_ok=False)
        result['input_sha256'] = sha256(args.input)
        write_comparison(args.output_root/'comparison.json',result)
        return result
    device = execution(args.device,args.threads,cfg.training.deterministic,0)
    workspace = Workspace(args.output_root,kind=args.command,config=cfg,protocol=protocol,device=device,
                          resume=args.resume,retry_failed=args.retry_failed,max_jobs=args.max_jobs,
                          control=control,max_training_epochs=max_training_epochs)
    if args.command=='iterations':
        result = run_iterations(workspace,cfg,device=device,reference=args.support_reference,**request)
    elif args.command=='candidates':
        result = run_candidates(workspace,cfg,protocol,stage=args.stage,device=device,threads=args.threads,
                                bp_settings=settings,data_root=args.data_root)
    else:
        graphs = plan['graphs']
        if not 1 <= graphs <= cfg.split_sizes['val']:
            raise ValueError('Review prefix must fit val')
        request = {'checkpoints':[{'path':str(p.resolve()),'sha256':sha256(p)} for p in args.checkpoint],
                   'method':args.method,'graphs':graphs,'bp_budget':settings,
                   'data_root':str(args.data_root.resolve()) if args.data_root else None}
        workspace.bind('review',request)
        source_checkpoint = read_checkpoint(args.checkpoint[0])
        source_paths = {split:source_checkpoint['datasets'][split]['path'] for split in ('train','val')}
        prepare_data(workspace,cfg,source_root=args.data_root,source_paths=source_paths)
        models = []
        for path in args.checkpoint:
            saved = read_checkpoint(path)
            obj = coefficients({k:saved['config']['objective'][k] for k in ('beta','gamma','kappa','eta')})
            cid,seed = candidate_id(obj),saved['config']['seeds']['training']
            frozen = freeze_copy(path,workspace.root/'imports'/(sha256(path)[:16]+'.pt'))
            models.append({'model_id':cid+'-s'+str(seed),'candidate_id':cid,'coefficients':obj,
                           'training_seed':seed,'baseline':False,'checkpoint':workspace.artifact(frozen),
                           'selected_epoch':saved['epoch']})
        if len({m['model_id'] for m in models}) != len(models):
            raise ValueError('Duplicate coefficient/seed model; use one preselected checkpoint per seed')
        try:
            result = compare_panel(workspace,cfg,models,graphs=graphs,method=args.method,device=device,
                                   name='review',bp_settings=settings)
        except WorkLimit:
            result = workspace.publish('review-progress',{'status':'partial','note':'No partial ranking'})
    print(json.dumps({'output':str(workspace.root),'status':result['status'],'new_jobs':workspace.executed},ensure_ascii=False))
    return result


if __name__=='__main__':
    main()
