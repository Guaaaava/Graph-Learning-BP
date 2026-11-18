"""Executable selection contracts with independent arithmetic and tiny real runs."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import pytest
import torch

from gib4cl.config import load_config
from gib4cl.data.dataset import graph_to_data
from gib4cl.data.generate import build_dataset
from gib4cl.inference import ParticleBP
from gib4cl.runtime import sha256
from tools.selection_runner import main
from tools.selection_compare import summarize
from tools.selection_evaluate import gaussian_moments
from tools.selection_state import read_json, Workspace
from tests.test_objectives import make_graph


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def inputs(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    cfg = replace(cfg,training=replace(cfg.training,validation_repeats=2),
                  evaluation=replace(cfg.evaluation,bootstrap_samples=20))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg.to_dict()),encoding='utf-8')
    return cfg,config,Path('configs/selection/smoke-protocol.json').resolve()


def test_default_plan_never_generates_data_loads_models_or_initializes_cuda(tmp_path,monkeypatch):
    import tools.selection_runner as runner
    def forbidden(*args,**kwargs):
        raise AssertionError('Plan attempted execution')
    monkeypatch.setattr(runner,'execution',forbidden)
    monkeypatch.setattr(runner,'Workspace',forbidden)
    result = main(['iterations','--device','cuda','--output-root',str(tmp_path/'not-created')])
    assert result['comparison_points']==32 and result['trajectories']==8
    assert result['plan_only'] and not (tmp_path/'not-created').exists()
    result = main(['candidates','--stage','A'])
    assert result['target_epoch']==20 and result['schedule_epochs']==200
    assert result['new_coefficient_combinations_upper_bound']==12
    assert not result['test_access'] and not result['automatic_next_stage']


def synthetic_report():
    specs = [{'model_id':b,'candidate_id':b,'baseline':True,'training_seed':None,'coefficients':None} for b in ('dense','bfs')]
    for group,beta in [('A',.01),('B',.03)]:
        for seed in (101,102):
            specs.append({'model_id':group+str(seed),'candidate_id':group,'baseline':False,'training_seed':seed,
                          'coefficients':{'beta':beta,'gamma':0.,'kappa':0.,'eta':0.}})
    context = {'method':'particle_bp','dataset':{'split':'val'},'sample_ids':['small','large'],
               'input_digests':{'small':'input1','large':'input2'},'training_seeds':[101,102],
               'bp_seeds':[700,701],'models':specs}
    records = []
    for spec in specs:
        for graph,mse in [('small',1.),('large',9.)]:
            for repeat in (700,701):
                group = spec['candidate_id']
                records.append({'status':'complete','sample_id':graph,'model_id':spec['model_id'],'bp_seed':repeat,
                    'input_digest':context['input_digests'][graph],'support_digest':graph+str(repeat),
                    'mse':4. if group=='B' else mse,'coverage95':.9 if group=='B' else .8,
                    'communication_cost':2. if group=='B' else 1.,'invalid_covariance_fraction':0.,
                    'metrics':{'total':-99999. if group=='A' else 99999.,'num_agents':1 if graph=='small' else 9}})
    return {'schema':'gib4cl.selection-records.v1','status':'complete','context':context,'records':records}


def test_comparison_uses_sqrt_mean_mse_graph_weights_and_ignores_weighted_loss():
    report = synthetic_report()
    result = summarize(report,bootstrap_samples=50)
    table = {r['candidate_id']:r for r in result['rows']}
    assert table['A']['rmse_graph_equal']==pytest.approx(math.sqrt(5))
    assert table['A']['rmse_graph_equal'] != pytest.approx(2.)  # mean of per-graph RMSE
    assert table['A']['rmse_graph_equal'] != pytest.approx(math.sqrt(8.2))  # pooled nodes
    assert result['selection']['roles']=={'accuracy':'B','calibration':'B','cost':'A'}
    assert result['selection']['representatives']==['B','A']
    assert not result['weighted_total_used_for_ranking']
    swapped = deepcopy(report)
    for row in swapped['records']: row['metrics']['total'] *= -1
    assert summarize(swapped,bootstrap_samples=50)['selection']==result['selection']


@pytest.mark.parametrize('change',['missing','duplicate','support','input','nonfinite','test','seed','failed'])
def test_incomplete_or_mismatched_panels_cannot_be_ranked(change):
    report = synthetic_report()
    if change=='missing': report['records'].pop()
    elif change=='duplicate': report['records'].append(deepcopy(report['records'][0]))
    elif change=='support': report['records'][-1]['support_digest']='wrong'
    elif change=='input': report['records'][-1]['input_digest']='wrong'
    elif change=='nonfinite': report['records'][-1]['mse']=float('nan')
    elif change=='test': report['context']['dataset']['split']='test'
    elif change=='seed': report['context']['training_seeds']=[101]
    else: report['records'][-1]['status']='failed'
    with pytest.raises(ValueError): summarize(report,bootstrap_samples=0)


def test_proxy_covariance_is_joint_inverse_marginal_not_inverse_diagonal_precision():
    cfg = load_config('configs/smoke-pipeline.json')
    data = make_graph([[0.,0.],[3.,0.]],[],[[0,1]],[3.5],[1.])
    _,covariance = gaussian_moments(data,torch.ones(1),cfg)
    assert covariance[0,0,0].item()==pytest.approx(20/9)
    assert covariance[1,0,0].item()==pytest.approx(20/9)
    assert covariance[0,1,1].item()==pytest.approx(4.)
    assert covariance[0,0,0].item()!=pytest.approx(1/1.25)


def test_single_graph_has_no_graph_interval_and_zero_dense_cost_has_no_ratio():
    report = synthetic_report()
    report['context']['sample_ids']=['small']
    report['context']['input_digests']={'small':'input1'}
    report['records']=[r for r in report['records'] if r['sample_id']=='small']
    for row in report['records']: row['communication_cost']=0.
    result = summarize(report,bootstrap_samples=50)
    assert all(all(v is None for v in r['intervals'].values()) for r in result['rows'])
    assert all(r['cost_relative_to_dense'] is None for r in result['rows'])


def test_iterations_resume_keeps_exact_support_and_complete_cache_never_infers(inputs,tmp_path,monkeypatch):
    cfg,config,protocol = inputs
    output = tmp_path/'iterations'
    args = ['iterations','--config',str(config),'--protocol',str(protocol),'--output-root',str(output),'--execute']
    partial = main(args+['--max-jobs','1'])
    assert partial['status']=='partial' and partial['comparison_points']==3
    report = main(args+['--resume'])
    assert report['status']=='complete' and report['comparison_points']==12
    first = report['trajectories'][0]
    graph = graph_to_data(build_dataset(cfg,'val',2)['graphs'][0])
    # Every saved round is independently verified by the runner at its first T.
    assert first['independent_round_checked']==0
    assert [r['requested_iterations'] for r in first['rows']]==[0,1,2]
    assert len({r['support_digest'] for r in first['rows']})==1
    assert first['rows'][0]['sample_id']==graph.sample_id
    def forbidden(*args,**kwargs): raise AssertionError('Cached trajectory reran BP')
    monkeypatch.setattr(ParticleBP,'infer',forbidden)
    assert main(args+['--resume'])['trajectories']==report['trajectories']
    assert report['historical_reuse'].startswith('unverified_new_run')


def test_workspace_failure_retry_and_corruption_guards(inputs,tmp_path):
    cfg,_,protocol = inputs
    root = tmp_path/'cache'
    p = read_json(protocol)
    ws = Workspace(root,kind='test',config=cfg,protocol=p,device=torch.device('cpu'))
    key = {'work':'one'}
    def bad(): raise RuntimeError('injected failure')
    with pytest.raises(RuntimeError): ws.job(key,bad)
    ws = Workspace(root,kind='test',config=cfg,protocol=p,device=torch.device('cpu'),resume=True)
    with pytest.raises(ValueError,match='Failed job'): ws.job(key,lambda:({'ok':True},[]))
    ws = Workspace(root,kind='test',config=cfg,protocol=p,device=torch.device('cpu'),resume=True,retry_failed=True)
    assert ws.job(key,lambda:({'ok':True},[]))=={'ok':True}
    record = next((root/'jobs').glob('*.json'))
    value = json.loads(record.read_text());value['value']['ok']=False;record.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='record changed'): ws.job(key,lambda:({},[]))
    altered = deepcopy(p);altered['revision']='different'
    with pytest.raises(ValueError,match='identical'): Workspace(root,kind='test',config=cfg,protocol=altered,device=torch.device('cpu'),resume=True)


def test_historical_stage13_reference_checks_inputs_support_and_T3(inputs,tmp_path):
    cfg,config,protocol = inputs
    common = ['iterations','--config',str(config),'--protocol',str(protocol),'--graph-indices','0',
              '--iterations','1','3','--execute']
    first = main(common+['--output-root',str(tmp_path/'first')])
    graph = graph_to_data(build_dataset(cfg,'val',1)['graphs'][0])
    from gib4cl.runtime import state_digest
    historical = {'baseline_input_digests':[state_digest(graph.to_dict())],
                  'graphs':{'0':{'baseline':{'sample_id':graph.sample_id+':baseline'}}},'trials':[]}
    for trajectory in first['trajectories']:
        row = trajectory['rows'][-1]
        historical['trials'].append({'sample_number':0,'seed':700,'condition':'baseline','particles':16,
            'topology':trajectory['topology'],'support_digest':row['support_digest'],
            'mean':row['mean'],'covariance':row['covariance']})
    reference = tmp_path/'historical.json';reference.write_text(json.dumps(historical))
    second = main(common+['--output-root',str(tmp_path/'matched'),'--support-reference',str(reference)])
    assert all(t['historical_T3_checked'] for t in second['trajectories'])
    assert second['historical_reuse']=='verified_against_supplied_reference'
    historical['baseline_input_digests'][0]='wrong-input'
    reference.write_text(json.dumps(historical))
    with pytest.raises(ValueError,match='reference does not match'):
        main(common+['--output-root',str(tmp_path/'bad-reference'),'--support-reference',str(reference)])


def test_completed_job_rejects_modified_snapshot(inputs,tmp_path):
    cfg,_,protocol = inputs
    ws = Workspace(tmp_path/'artifacts',kind='test',config=cfg,protocol=read_json(protocol),device=torch.device('cpu'))
    artifact = ws.root/'weights.txt';artifact.write_text('verified')
    ws.job({'one':'artifact'},lambda:({'ok':True},[artifact]))
    artifact.write_text('changed')
    with pytest.raises(ValueError,match='artifact missing or changed'):
        ws.job({'one':'artifact'},lambda:({'ok':False},[]))


def test_explicit_retry_before_first_epoch_preserves_failed_run(inputs,tmp_path,monkeypatch):
    import tools.selection_candidates as candidates
    _,config,protocol = inputs
    root = tmp_path/'retry-search'
    args = ['candidates','--config',str(config),'--protocol',str(protocol),'--stage','A',
            '--output-root',str(root),'--execute','--max-jobs','1']
    original = candidates.train_gnn
    def interrupted(**kwargs):
        directory = kwargs['run_directory'];directory.mkdir(parents=True)
        (directory/'interrupted-marker.txt').write_text('first epoch not committed')
        raise RuntimeError('injected pre-epoch failure')
    monkeypatch.setattr(candidates,'train_gnn',interrupted)
    with pytest.raises(RuntimeError,match='pre-epoch'): main(args)
    monkeypatch.setattr(candidates,'train_gnn',original)
    result = main(args+['--resume','--retry-failed'])
    assert result['status']=='partial' and len(result['models'])==1
    assert result['models'][0]['training_run_directory'].endswith('-retry1')
    assert list((root/'runs').glob('*/interrupted-marker.txt'))
    assert list((root/'failures').glob('*.json'))


def test_real_candidate_stages_resume_promote_and_final_seed_panel(inputs,tmp_path,monkeypatch):
    cfg,config,protocol = inputs
    output = tmp_path/'search'
    common = ['candidates','--config',str(config),'--protocol',str(protocol),'--output-root',str(output),'--execute']
    partial = main(common+['--stage','A','--max-jobs','1'])
    assert partial['status']=='partial'
    with pytest.raises(ValueError,match='not complete'): main(common+['--stage','B','--resume'])
    a = main(common+['--stage','A','--resume'])
    assert a['status']=='complete' and len(a['models'])==4
    assert not cfg.dataset_path('test',output/'data').exists()
    original = {m['checkpoint']['path']:m['checkpoint']['sha256'] for m in a['models']}
    for stage in ('B','C','D'):
        assert main(common+['--stage',stage,'--resume'])['status']=='complete'
    budget = ['--bp-iterations','1','--bp-chunk-size','8']
    with pytest.raises(ValueError,match='Explicit'): main(common+['--stage','promote','--resume'])
    promoted = main(common+['--stage','promote','--resume']+budget)
    assert promoted['status']=='complete'
    assert all(sha256(output/p)==digest for p,digest in original.items())
    final = main(common+['--stage','final','--resume']+budget)
    assert final['status']=='complete'
    groups = {}
    for model in final['models']: groups.setdefault(model['candidate_id'],set()).add(model['training_seed'])
    assert all(seeds=={101,102} for seeds in groups.values())
    comparison = read_json(output/'reports/comparison-final.json')
    assert comparison['context']['method']=='particle_bp'
    assert comparison['context']['training_seeds']==[101,102]
    # Dense is computed once per graph/BP seed, not once per candidate or stage.
    jobs = [read_json(p) for p in (output/'jobs').glob('*.json')]
    dense = [j for j in jobs if j['key'].get('type')=='prediction' and j['key'].get('method')=='particle_bp' and j['key'].get('model_id')=='dense']
    assert len(dense)==4
    for spec in final['models']:
        from gib4cl.learning.checkpoints import read_checkpoint
        saved = read_checkpoint(output/spec['last_snapshot']['path'])
        assert saved['config']['training']['epochs']==2 and saved['epoch']==2
    exported = main(['compare','--config',str(config),'--protocol',str(protocol),'--input',str(output/'reports/records-final.json'),
                     '--output-root',str(tmp_path/'export'),'--execute'])
    assert exported['selection']==comparison['selection']
    checkpoints = [str(output/m['checkpoint']['path']) for m in final['models']]
    review_args = ['review','--config',str(config),'--protocol',str(protocol),'--checkpoint',*checkpoints,
                   '--output-root',str(tmp_path/'review'),'--execute',*budget]
    assert main(review_args+['--max-jobs','1'])['status']=='partial'
    review = main(review_args+['--resume'])
    assert review['status']=='complete' and review['context']['method']=='particle_bp'
    def forbidden(*args,**kwargs): raise AssertionError('Completed review reran BP')
    monkeypatch.setattr(ParticleBP,'infer',forbidden)
    assert main(review_args+['--resume'])['selection']==review['selection']
