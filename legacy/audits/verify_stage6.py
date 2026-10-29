"""Reproducible stage-6 audit: pipeline, exact resume, fine-tune and invalid reports."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import zipfile
import torch

from gib4cl.experiments.cli import main
from gib4cl.config import REPO_ROOT,load_config,fingerprint
from gib4cl.learning.train import train_gnn
from gib4cl.evaluation.evaluate import evaluate
from gib4cl.learning.checkpoints import read_checkpoint
from gib4cl.data.generate import create_dataset
from gib4cl.runtime import sha256,atomic_json,source_manifest


def identical(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(identical(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(identical(a,b) for a,b in zip(left,right))
    return left==right


def verify(root):
    root.mkdir(parents=True,exist_ok=False)
    cfg_path = REPO_ROOT/'configs/smoke-pipeline.json'
    cfg = load_config(cfg_path)
    result = main(['run','--config',str(cfg_path),'--data-root',str(root/'data'),
                   '--output-root',str(root/'continuous'),'--device','cpu','--threads','1'])
    full = read_checkpoint(result['models']/'last_checkpoint.pt')
    interrupted = train_gnn(settings=cfg,data_root=root/'data',output_root=root/'interrupted',
                             stop_after_epoch=1,device='cpu',threads=1)
    first = read_checkpoint(interrupted/'last_checkpoint.pt')
    assert first['epoch']==1 and first['scheduler_state']['T_max']==2
    train_gnn(resume=interrupted/'last_checkpoint.pt',data_root=root/'data',device='cpu',threads=1)
    resumed = read_checkpoint(interrupted/'last_checkpoint.pt')
    fields = ['model_state','optimizer_state','scheduler_state','history','rng_state','best_metric','best_epoch']
    exact = {key:identical(full[key],resumed[key]) for key in fields}
    assert all(exact.values()),exact
    fine = train_gnn(settings=cfg,init_from=result['models']/'last_checkpoint.pt',
                     data_root=root/'data',output_root=root/'fine',stop_after_epoch=1,device='cpu',threads=1)
    fine_state = read_checkpoint(fine/'last_checkpoint.pt')
    fine_manifest = json.loads((fine.parent/'run.json').read_text(encoding='utf-8'))
    assert fine_manifest['initial_model_digest']==full['model_digest']
    assert fine_state['model_digest']!=full['model_digest']
    fine_report = evaluate(checkpoint=fine/'best_checkpoint.pt',settings=cfg,data_root=root/'data',
                           output_root=root/'fine-evaluation',max_graphs=1,device='cpu',threads=1)
    fine_result = json.loads((fine_report/'results.json').read_text(encoding='utf-8'))
    assert fine_result['checkpoint']['sha256']==sha256(fine/'best_checkpoint.pt')
    assert all(value['graph_metrics']['rmse']['ci_low'] is None for value in fine_result['summary']['topologies'].values())
    empty = replace(cfg,data=replace(cfg.data,agent_comm_radius=0.,anchor_comm_radius=0.),
                    bp=replace(cfg.bp,num_particles=2,num_iterations=0),
                    evaluation=replace(cfg.evaluation,bp_repeats=1))
    create_dataset('test',1,settings=empty,output_root=root/'empty-data')
    empty_report = evaluate(checkpoint=result['models']/'best_checkpoint.pt',settings=empty,
                            data_root=root/'empty-data',output_root=root/'empty-evaluation',device='cpu',threads=1)
    empty_result = json.loads((empty_report/'results.json').read_text(encoding='utf-8'))
    assert all(value['pooled_mean_nees_valid'] is None for value in empty_result['summary']['topologies'].values())
    assert all(value['invalid_covariance_agent_runs']==value['agent_runs'] for value in empty_result['summary']['topologies'].values())
    main_result = json.loads((result['evaluation']/'results.json').read_text(encoding='utf-8'))
    for graph in main_result['graphs']:
        for repeat in range(cfg.evaluation.bp_repeats):
            rows = [row for row in main_result['records'] if row['sample_id']==graph['sample_id'] and row['bp_repeat']==repeat]
            assert len(set(row['support_digest'] for row in rows))==1
            assert len(set(row['bp_seed'] for row in rows))==1
            assert next(row for row in rows if row['topology']=='bfs')['metrics']['agent_triangles']==0
    from tools.provenance import preserved_core,source_path,paper_sha256,record_path
    preserved = preserved_core()
    baseline = REPO_ROOT/'docs/baselines/2026-10-22/source-snapshot.zip'
    with zipfile.ZipFile(baseline) as archive:
        if (REPO_ROOT.parent/'GIB-Chap234-1016.tex').is_file():
            assert archive.read('paper/GIB-Chap234-1016.tex')==(REPO_ROOT.parent/'GIB-Chap234-1016.tex').read_bytes()
        for name in archive.namelist():
            if name.startswith('repository/NEBP/') and name.endswith('.py'):
                assert archive.read(name)==source_path(name[len('repository/'):]).read_bytes()
    report = {'stage':'06-unified-forest-training-evaluation','date':'2026-10-23',
              'runtime':full['runtime'],'config':cfg.to_dict(),'config_sha256':sha256(cfg_path),
              'pipeline':{key:record_path(value) for key,value in result.items()},
              'continuous_model_digest':full['model_digest'],'resume_exact_matches':exact,
              'fine_tuning':{'models':record_path(fine),
                              'initial_digest':fine_manifest['initial_model_digest'],
                              'source_digest':full['model_digest'],'final_digest':fine_state['model_digest'],
                              'optimizer_reset':True,'evaluation':record_path(fine_report),
                              'checkpoint_sha256':fine_result['checkpoint']['sha256']},
              'main_evaluation_summary':main_result['summary'],
              'shared_supports_and_seeds_across_topologies':True,
              'single_graph_ci_is_null':True,'zero_edge_all_invalid_saved':True,
              'zero_edge_report':record_path(empty_report),
              'evaluation_graphs':main_result['num_graphs'],'bp_trials_per_topology':main_result['num_graphs']*cfg.evaluation.bp_repeats,
              'results_sha256':sha256(result['evaluation']/'results.json'),
              'preserved_core_algorithms':preserved,'source_sha256':source_manifest(),
              'paper_sha256':paper_sha256(),
              'baseline_sha256':sha256(baseline),
              'limitations':['CPU workflow verification only; CUDA not available in this environment',
                             'Two-epoch smoke weights and BP settings are verification choices, not formal experiments',
                             'Exact resume verified only at epoch boundaries on the same source/runtime/device family',
                             'Graph bootstrap is conditional on the chosen trained checkpoint; it does not cover training-seed variation']}
    atomic_json(root/'verification.json',report)
    print(json.dumps({'audit':str(root/'verification.json'),'resume_exact':all(exact.values()),
                      'evaluation_graphs':main_result['num_graphs'],'bp_trials_per_topology':report['bp_trials_per_topology'],
                      'fine_tuning_source_verified':True,'zero_edge_all_invalid_saved':True},ensure_ascii=False))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=REPO_ROOT/'outputs/stage6-verified')
    args = parser.parse_args()
    verify(args.output_root.resolve())
