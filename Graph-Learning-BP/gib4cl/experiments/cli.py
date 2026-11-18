"""Unified experiment CLI: generate / train / evaluate / run.

Use python -m gib4cl.experiments.cli --help from the repository root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gib4cl.config import load_config,with_seeds
from gib4cl.data.generate import create_dataset
from gib4cl.learning.train import train_gnn,resolve_training_config
from gib4cl.evaluation.evaluate import evaluate
from gib4cl.runtime import load_split


def _common(parser,execution_options=True):
    parser.add_argument('--config')
    parser.add_argument('--data-seed',type=int)
    parser.add_argument('--data-root')
    if execution_options:
        parser.add_argument('--output-root')
        parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
        parser.add_argument('--threads',type=int)


def _training(parser):
    for name in ('beta','gamma','kappa','eta','lr','tau-init','tau-decay'):
        parser.add_argument('--'+name,type=float)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--training-seed',type=int)
    parser.add_argument('--run-name')


def _overrides(args):
    return {name:getattr(args,name) for name in ('beta','gamma','kappa','eta','lr','tau_init','tau_decay','epochs','data_seed','training_seed')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    generate = commands.add_parser('generate',help='Generate versioned datasets without overwriting existing files')
    _common(generate,False)
    generate.add_argument('--split',nargs='+',default=['train','val','test'])
    generate.add_argument('--num-graphs',type=int)
    generate.add_argument('--seed-offset',type=int,default=0)
    train = commands.add_parser('train',help='Train, resume an existing schedule, or initialize a new fine-tuning run')
    _common(train)
    _training(train)
    group = train.add_mutually_exclusive_group()
    group.add_argument('--resume')
    group.add_argument('--init-from')
    train.add_argument('--stop-after-epoch',type=int)
    evaluation = commands.add_parser('evaluate',help='Evaluate a specific versioned checkpoint')
    _common(evaluation)
    evaluation.add_argument('--checkpoint',required=True)
    evaluation.add_argument('--bp-seed',type=int)
    evaluation.add_argument('--bootstrap-seed',type=int)
    evaluation.add_argument('--split',default='test')
    evaluation.add_argument('--max-graphs',type=int)
    evaluation.add_argument('--evaluation-name')
    run = commands.add_parser('run',help='Generate -> train -> load best checkpoint -> Dense/BFS/GNN -> reports')
    _common(run)
    _training(run)
    run.add_argument('--reuse-data',action='store_true')
    run.add_argument('--bp-seed',type=int)
    run.add_argument('--bootstrap-seed',type=int)
    run.add_argument('--max-graphs',type=int)
    run.add_argument('--evaluation-name')
    suite = commands.add_parser('suite',help='Run or resume a fixed-data multi-training-seed and BP-seed panel')
    suite.add_argument('--suite-config',required=True)
    suite.add_argument('--data-root')
    suite.add_argument('--output-root')
    suite.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    suite.add_argument('--threads',type=int,default=1)
    suite.add_argument('--resume',action='store_true')
    suite.add_argument('--reuse-data',action='store_true')
    suite.add_argument('--retry-failed',action='store_true')
    suite.add_argument('--stop-after-jobs',type=int)
    suite.add_argument('--plan',action='store_true',help='Print resolved configurations and job identities without running')
    args = parser.parse_args(argv)
    if args.command == 'suite':
        from gib4cl.experiments.suite import run_suite,SuiteRunError
        try:
            result = run_suite(suite_config=args.suite_config,data_root=args.data_root,output_root=args.output_root,
                               resume=args.resume,reuse_data=args.reuse_data,retry_failed=args.retry_failed,
                               stop_after_jobs=args.stop_after_jobs,device=args.device,threads=args.threads,
                               plan_only=args.plan)
        except SuiteRunError as error:
            parser.exit(1,str(error)+'\n')
        if args.plan:
            print(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            print('Suite output: '+str(result))
        return result
    if args.command == 'generate':
        cfg = with_seeds(load_config(args.config),data=args.data_seed)
        paths = []
        for split in args.split:
            count = args.num_graphs if args.num_graphs is not None else cfg.split_sizes.get(split)
            if count is None:
                parser.error(f'No configured graph count for {split}')
            paths.append(create_dataset(split,count,seed_offset=args.seed_offset,settings=cfg,output_root=args.data_root))
        return paths
    if args.command == 'train':
        return train_gnn(config_path=args.config,data_root=args.data_root,output_root=args.output_root,
                         resume=args.resume,init_from=args.init_from,run_name=args.run_name,
                         device=args.device,threads=args.threads,stop_after_epoch=args.stop_after_epoch,
                         **_overrides(args))
    if args.command == 'evaluate':
        return evaluate(checkpoint=args.checkpoint,config_path=args.config,data_root=args.data_root,
                        output_root=args.output_root,device=args.device,threads=args.threads,
                        split=args.split,max_graphs=args.max_graphs,evaluation_name=args.evaluation_name,
                        data_seed=args.data_seed,bp_master_seed=args.bp_seed,bootstrap_seed=args.bootstrap_seed)
    cfg = with_seeds(resolve_training_config(args.config,**_overrides(args)),
                     bp=args.bp_seed,bootstrap=args.bootstrap_seed)
    for split in ('train','val','test'):
        if cfg.split_sizes.get(split,0) < 1:
            raise ValueError('run requires nonempty train, val and test splits')
    if args.reuse_data:
        for split in ('train','val','test'):
            load_split(cfg,split,args.data_root)
    else:
        if any(cfg.dataset_path(split,args.data_root).exists() or cfg.dataset_path(split,args.data_root).with_suffix('.json').exists()
               for split in ('train','val','test')):
            raise FileExistsError('Pipeline data already exists; explicitly pass --reuse-data after verifying its identity')
        for split in ('train','val','test'):
            create_dataset(split,cfg.split_sizes[split],settings=cfg,output_root=args.data_root)
    models = train_gnn(settings=cfg,data_root=args.data_root,output_root=args.output_root,
                       run_name=args.run_name,device=args.device,threads=args.threads)
    report = evaluate(checkpoint=models/'best_checkpoint.pt',settings=cfg,data_root=args.data_root,
                      output_root=args.output_root,device=args.device,threads=args.threads,
                      max_graphs=args.max_graphs,evaluation_name=args.evaluation_name)
    return {'models':models,'evaluation':report}


if __name__ == '__main__':
    main()
