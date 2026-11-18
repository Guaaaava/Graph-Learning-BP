"""Stage 6: forest semantics, checkpoint continuity, paired evaluation and reporting."""
from copy import deepcopy
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from gib4cl.inference.spanning_tree import bfs_forest,extract_bfs_tree_edges,extract_bfs_weights
from gib4cl.learning.checkpoints import make_model,read_checkpoint,load_selector
from gib4cl.evaluation.evaluate import evaluate,bp_seed,render_report
from gib4cl.config import load_config,fingerprint,config_from_dict
from gib4cl.experiments.cli import main
from gib4cl.data.generate import create_dataset
from gib4cl.evaluation.metrics import calculate_metrics,summarize_records
from gib4cl.runtime import sha256,state_digest
from gib4cl.evaluation.topology import select_topologies
from gib4cl.learning.train import train_gnn
from tests.test_objectives import make_graph


@pytest.fixture(autouse=True,scope='module')
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('unified')
    cfg = load_config('configs/smoke-pipeline.json')
    cfg = replace(cfg,evaluation=replace(cfg.evaluation,bootstrap_samples=40))
    for split in ('train','val','test'):
        create_dataset(split,cfg.split_sizes[split],settings=cfg,output_root=root/'data')
    return root,cfg


@pytest.fixture(scope='module')
def runs(prepared):
    root,cfg = prepared
    continuous = train_gnn(settings=cfg,data_root=root/'data',output_root=root/'continuous',device='cpu')
    split = train_gnn(settings=cfg,data_root=root/'data',output_root=root/'split',device='cpu',stop_after_epoch=1)
    first = read_checkpoint(split/'last_checkpoint.pt')
    torch.save(first,split/'epoch_one.pt')
    resumed = train_gnn(resume=split/'last_checkpoint.pt',data_root=root/'data',device='cpu')
    fine = train_gnn(settings=cfg,init_from=continuous/'last_checkpoint.pt',data_root=root/'data',
                     output_root=root/'fine',device='cpu',stop_after_epoch=1)
    return root,cfg,continuous,resumed,fine


def test_bfs_multi_anchor_and_unanchored_components_have_explicit_roots():
    edges = torch.tensor([[0,0,0,1,3],[5,6,1,2,4]])
    forest = bfs_forest(7,edges,torch.tensor([5,6]))
    assert forest.edge_weights.tolist() == [1.,0.,1.,1.,1.]
    assert forest.forest_root.tolist() == [5,5,5,3,3,5,6]
    assert forest.anchor_reachable.tolist() == [True,True,True,False,False,True,True]
    assert int(forest.edge_weights.sum()) == 7-len(set(forest.forest_root.tolist()))


def test_bfs_edge_order_orientation_and_wrapper_use_one_algorithm():
    data = make_graph([[0.,0.],[1.,0.],[0.,1.]],[[2.,2.],[3.,3.]],
                       [[0,1],[0,2],[1,2],[0,3],[0,4]],[1.]*5,[1.]*5)
    original = extract_bfs_weights(data)
    wrapped = extract_bfs_tree_edges(3,2,data.edge_index,data.edge_attr[:,3])
    torch.testing.assert_close(original,wrapped,rtol=0,atol=0)
    order = torch.tensor([4,2,0,3,1])
    changed = bfs_forest(5,data.edge_index[:,order].flip(0),torch.tensor([4,3]))
    torch.testing.assert_close(changed.edge_weights,original[order],rtol=0,atol=0)
    assert original[4] == 0  # all anchor edges are not forcibly retained


def test_bfs_without_anchors_and_empty_graph():
    forest = bfs_forest(3,torch.empty(2,0,dtype=torch.long),torch.empty(0,dtype=torch.long))
    assert forest.forest_root.tolist() == [0,1,2] and not forest.anchor_reachable.any()
    cycle = bfs_forest(3,torch.tensor([[0,1,0],[1,2,2]]),torch.empty(0,dtype=torch.long))
    assert cycle.edge_weights.tolist() == [1.,0.,1.]


@pytest.mark.parametrize('edges,anchors',[([[0,0],[1,1]],[2]),([[0],[0]],[2]),([[0],[3]],[2]),
                                         ([[0],[1]],[0,1]),([[0],[1]],[2,2])])
def test_bfs_invalid_candidate_inputs_rejected(edges,anchors):
    with pytest.raises(ValueError):
        bfs_forest(3,torch.tensor(edges),torch.tensor(anchors))


def test_rates_nees_ellipse_and_population_units_are_distinct():
    estimate = torch.zeros(4,2)
    truth = torch.tensor([[0.,0.],[1.,0.],[3.,0.],[2.,0.]])
    cov = torch.eye(2).repeat(4,1,1)
    cov[-1] = 0
    result = calculate_metrics(estimate,truth,cov,e_th=2.)
    assert result['rmse'] == pytest.approx(math.sqrt(3.5))
    assert result['outage'] == .25
    assert result['ellipse_coverage'] == .5 and result['nees_interval_pass'] == .25
    assert result['mean_nees_valid'] == pytest.approx(10/3)
    assert result['agent_nees'] == [0.,1.,9.,None]
    assert result['nees_invalid'] == 1
    assert 1-math.exp(-result['ellipse_threshold']/2) == pytest.approx(.95)
    low,high = result['nees_interval']
    assert math.exp(-low/2)-math.exp(-high/2) == pytest.approx(.95)


@pytest.mark.parametrize('covariance',[[[0.,0.],[0.,0.]],[[-1.,0.],[0.,-1.]],
                                       [[1.,2.],[0.,1.]],[[float('nan'),0.],[0.,1.]]])
def test_invalid_covariance_is_counted_and_not_replaced_by_zero_nees(covariance):
    result = calculate_metrics(torch.zeros(1,2),torch.ones(1,2),torch.tensor([covariance]))
    assert result['mean_nees_valid'] is None and result['nees_invalid'] == 1
    assert result['ellipse_coverage'] == result['nees_interval_pass'] == 0
    json.dumps(result,allow_nan=False)


def synthetic(sample,error,agents=1,repeat=0,topology='dense',invalid=False):
    result = calculate_metrics(torch.zeros(agents,2),torch.tensor([[error,0.]]).repeat(agents,1),
                               torch.zeros(agents,2,2) if invalid else torch.eye(2).repeat(agents,1,1))
    result.update(edges=0,communication_cost=0.,agent_triangles=0,agents_below_degree_three=agents,
                  unanchored_agents=agents,ess_min=10.,ess_mean=10.,bp_converged=1,bp_iterations=0)
    return {'sample_id':sample,'topology':topology,'bp_repeat':repeat,'metrics':result}


def test_cluster_bootstrap_keeps_nodes_and_repeats_together():
    records = [synthetic('a',0.,1),synthetic('b',10.,9)]
    one = summarize_records(records,bootstrap_samples=400,seed=7)
    repeated = summarize_records(records+[dict(row,bp_repeat=1) for row in records],bootstrap_samples=400,seed=7)
    a,b = (value['topologies']['dense'] for value in (one,repeated))
    assert a['graph_metrics'] == b['graph_metrics']
    assert a['graph_metrics']['rmse']['mean'] == 5.
    assert a['graph_metrics']['rmse']['std'] == pytest.approx(math.sqrt(50))
    assert a['graph_metrics']['rmse']['ci_low'] == 0 and a['graph_metrics']['rmse']['ci_high'] == 10.
    assert a['pooled_agent_run_rmse'] == pytest.approx(math.sqrt(90))
    assert a['graphs'] == b['graphs'] == 2 and b['bp_trials'] == 4
    assert a['agent_error_quantiles']['p50'] == 10.
    assert a['graph_rmse_quantiles']['p50'] == 5.


def test_single_graph_all_invalid_report_has_null_ci_and_nees():
    rows = [synthetic('one',1.,2,invalid=True)]
    summary = summarize_records(rows,bootstrap_samples=50)
    stat = summary['topologies']['dense']
    assert stat['graph_metrics']['rmse']['std'] is None
    assert stat['graph_metrics']['rmse']['ci_low'] is None
    assert stat['graph_metrics']['mean_nees_valid']['mean'] is None
    assert stat['pooled_mean_nees_valid'] is None and stat['invalid_covariance_agent_runs'] == 2
    json.dumps(summary,allow_nan=False)
    with pytest.raises(ValueError):
        summarize_records([])


def test_paired_differences_bootstrap_graph_units():
    rows = [synthetic('a',1.),synthetic('b',3.),synthetic('a',2.,topology='gnn'),
            synthetic('b',4.,topology='gnn')]
    result = summarize_records(rows,bootstrap_samples=50)
    delta = result['paired_graph_differences']['gnn_minus_dense']['rmse']
    assert delta['mean'] == delta['ci_low'] == delta['ci_high'] == 1.


def test_checkpoint_resume_exactly_reproduces_uninterrupted_training(runs):
    root,cfg,continuous,resumed,fine = runs
    full,split = (read_checkpoint(path/'last_checkpoint.pt') for path in (continuous,resumed))
    assert full['history'] == split['history']
    assert full['model_digest'] == split['model_digest']
    for key,value in full['model_state'].items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(value,split['model_state'][key],rtol=0,atol=0)
    assert full['scheduler_state'] == split['scheduler_state']
    for parameter,state in full['optimizer_state']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,split['optimizer_state']['state'][parameter][key],rtol=0,atol=0)
    assert full['best_epoch'] == min(full['history'],key=lambda row:row['val']['total'])['epoch']


def test_fine_tuning_loads_parameters_before_first_step_and_resets_optimizer(runs):
    root,cfg,parent,resumed,fine = runs
    initial = read_checkpoint(parent/'last_checkpoint.pt')
    result = read_checkpoint(fine/'last_checkpoint.pt')
    manifest = json.loads((fine.parent/'run.json').read_text(encoding='utf-8'))
    assert manifest['initial_model_digest'] == initial['model_digest']
    assert manifest['init_from']['model_digest'] == initial['model_digest']
    assert result['model_digest'] != initial['model_digest']
    assert result['epoch'] == 1
    steps = [float(row['step']) for row in result['optimizer_state']['state'].values()]
    assert steps and max(steps) == 2  # 3 training graphs at batch size 2


def test_resume_stale_checkpoint_and_changed_schedule_are_rejected(runs):
    root,cfg,parent,resumed,fine = runs
    with pytest.raises(ValueError,match='newer checkpoint'):
        train_gnn(resume=resumed/'epoch_one.pt',data_root=root/'data',device='cpu')
    with pytest.raises(ValueError,match='original full config'):
        train_gnn(resume=resumed/'last_checkpoint.pt',settings=replace(cfg,training=replace(cfg.training,epochs=3)),
                  data_root=root/'data',device='cpu')
    with pytest.raises(ValueError,match='mutually exclusive'):
        train_gnn(resume=resumed/'last_checkpoint.pt',init_from=parent/'best_checkpoint.pt')


def test_resume_missing_rng_and_corrupt_model_are_rejected(runs,tmp_path):
    *_,parent,resumed,fine = runs
    original = read_checkpoint(parent/'last_checkpoint.pt')
    missing = deepcopy(original)
    del missing['rng_state']
    path = tmp_path/'missing.pt'
    torch.save(missing,path)
    with pytest.raises(ValueError,match='Incomplete'):
        read_checkpoint(path)
    damaged = deepcopy(original)
    key = next(name for name,value in damaged['model_state'].items() if isinstance(value,torch.Tensor))
    damaged['model_state'][key] += 1
    torch.save(damaged,path)
    with pytest.raises(ValueError,match='digest mismatch'):
        read_checkpoint(path)
    damaged = deepcopy(original)
    damaged['epoch'] += 1
    torch.save(damaged,path)
    with pytest.raises(ValueError,match='epoch/history'):
        read_checkpoint(path)


def test_resume_rejects_different_dataset_bytes(runs,tmp_path):
    root,cfg,parent,resumed,fine = runs
    for split in ('train','val'):
        create_dataset(split,cfg.split_sizes[split],seed_offset=1,settings=cfg,output_root=tmp_path/'changed')
    with pytest.raises(ValueError,match='dataset identity'):
        train_gnn(resume=resumed/'last_checkpoint.pt',data_root=tmp_path/'changed',device='cpu')


def test_eval_uses_exact_checkpoint_and_pairs_supports_and_reports(runs):
    root,cfg,parent,resumed,fine = runs
    path = evaluate(checkpoint=fine/'best_checkpoint.pt',settings=cfg,data_root=root/'data',
                    output_root=root/'evaluate',device='cpu')
    result = json.loads((path/'results.json').read_text(encoding='utf-8'))
    assert result['checkpoint']['sha256'] == sha256(fine/'best_checkpoint.pt')
    assert result['checkpoint']['model_digest'] == read_checkpoint(fine/'best_checkpoint.pt')['model_digest']
    assert result['num_graphs'] == 2
    assert len(result['records']) == 2*cfg.evaluation.bp_repeats*3
    for sample in result['graphs']:
        for repeat in range(cfg.evaluation.bp_repeats):
            rows = [row for row in result['records'] if row['sample_id']==sample['sample_id'] and row['bp_repeat']==repeat]
            assert len(set(row['support_digest'] for row in rows)) == 1
            assert len(set(row['bp_seed'] for row in rows)) == 1
            assert set(row['topology'] for row in rows) == {'dense','bfs','gnn'}
        assert 'selected_edge_ids' in sample['selection']
    assert result['summary']['topologies']['gnn']['graphs'] == 2
    assert result['summary']['topologies']['gnn']['bp_trials'] == 4
    assert all('information_proxy' not in row for row in result['records'])
    assert 'information_radius_proxy' not in result['metric_protocol']['scalar_metrics']
    assert 'information radius' not in (path/'report.md').read_text(encoding='utf-8')
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and manifest['results_sha256'] == sha256(path/'results.json')
    with pytest.raises(FileExistsError):
        evaluate(checkpoint=fine/'best_checkpoint.pt',settings=cfg,data_root=root/'data',
                 output_root=root/'evaluate',device='cpu')


def test_missing_raw_or_wrong_model_never_falls_back_to_random(runs,tmp_path):
    root,cfg,parent,resumed,fine = runs
    with pytest.raises(FileNotFoundError):
        evaluate(checkpoint=tmp_path/'absent.pt',settings=cfg,output_root=tmp_path/'outputs')
    with pytest.raises(ValueError,match='versioned full checkpoint'):
        evaluate(checkpoint=parent/'best_model.pth',settings=cfg,output_root=tmp_path/'outputs')
    wrong = replace(cfg,model=replace(cfg.model,hidden_dim=20))
    with pytest.raises(ValueError,match='model configuration'):
        evaluate(checkpoint=parent/'best_checkpoint.pt',settings=wrong,output_root=tmp_path/'outputs')


def test_single_graph_evaluation_and_zero_edge_ratios_are_serializable(runs):
    root,cfg,parent,resumed,fine = runs
    path = evaluate(checkpoint=parent/'best_checkpoint.pt',settings=cfg,data_root=root/'data',
                    output_root=root/'one_graph',max_graphs=1,device='cpu')
    result = json.loads((path/'results.json').read_text(encoding='utf-8'))
    assert all(v['graph_metrics']['rmse']['std'] is None for v in result['summary']['topologies'].values())
    assert all(v['graph_metrics']['rmse']['ci_low'] is None for v in result['summary']['topologies'].values())
    assert 'n/a' in (path/'report.md').read_text(encoding='utf-8')
    with pytest.raises(ValueError,match='max_graphs'):
        evaluate(checkpoint=parent/'best_checkpoint.pt',settings=cfg,max_graphs=0)


def test_bp_seeds_tied_to_sample_identity_not_position():
    a = bp_seed(700,'dataset','sample-a',0)
    assert a == bp_seed(700,'dataset','sample-a',0)
    assert len({a,bp_seed(700,'dataset','sample-b',0),bp_seed(700,'dataset','sample-a',1)}) == 3


def test_numerically_singular_covariance_cutoff_is_relative_and_explicit():
    positions,truth = torch.zeros(1,2),torch.ones(1,2)
    ill_conditioned = torch.diag(torch.tensor([1.,1e-16],dtype=torch.float64)).unsqueeze(0)
    assert calculate_metrics(positions,truth,ill_conditioned)['mean_nees_valid'] is None
    allowed = calculate_metrics(positions,truth,ill_conditioned,covariance_rtol=0.)
    assert allowed['mean_nees_valid'] is not None
    tiny_isotropic = torch.eye(2,dtype=torch.float64).unsqueeze(0)*1e-20
    assert calculate_metrics(positions,positions,tiny_isotropic)['mean_nees_valid'] == 0.


def test_training_output_and_completed_resume_do_not_overwrite_history(runs):
    root,cfg,parent,resumed,fine = runs
    before = sha256(resumed/'last_checkpoint.pt')
    train_gnn(resume=resumed/'last_checkpoint.pt',data_root=root/'data',device='cpu')
    assert sha256(resumed/'last_checkpoint.pt') == before
    with pytest.raises(FileExistsError):
        train_gnn(settings=cfg,data_root=root/'data',output_root=root/'continuous',device='cpu')


def test_resume_cli_inherits_saved_device_threads_and_configuration(runs):
    root,cfg,parent,resumed,fine = runs
    before = sha256(resumed/'last_checkpoint.pt')
    process = subprocess.run([sys.executable,'-B','-m','gib4cl','train',
                              '--resume',str(resumed/'last_checkpoint.pt'),'--data-root',str(root/'data')],
                              cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert process.returncode == 0,process.stderr
    assert sha256(resumed/'last_checkpoint.pt') == before


def test_configuration_and_split_identity_guards(prepared,tmp_path):
    root,cfg = prepared
    assert replace(cfg,evaluation=replace(cfg.evaluation,bp_repeats=3)).data_fingerprint == cfg.data_fingerprint
    with pytest.raises(ValueError):
        replace(cfg.evaluation,edge_threshold=1.)
    with pytest.raises(ValueError):
        replace(cfg.evaluation,bp_repeats=0)
    with pytest.raises(ValueError):
        replace(cfg.training,lr_min=1.)
    wrong = replace(cfg,data=replace(cfg.data,area_size=21.))
    # An explicitly supplied dataset at the expected path still must match its declared config.
    target = wrong.dataset_path('train',tmp_path)
    target.parent.mkdir(parents=True)
    source = cfg.dataset_path('train',root/'data')
    target.write_bytes(source.read_bytes())
    from gib4cl.runtime import load_split
    with pytest.raises(ValueError,match='identity mismatch'):
        load_split(wrong,'train',tmp_path)


def test_zero_edge_single_graph_all_invalid_evaluation_saves_report(runs,tmp_path):
    root,cfg,parent,resumed,fine = runs
    empty = replace(cfg,data=replace(cfg.data,comm_radius=0.),
                    bp=replace(cfg.bp,num_particles=2,num_iterations=0),
                    evaluation=replace(cfg.evaluation,bp_repeats=1))
    create_dataset('test',1,settings=empty,output_root=tmp_path/'data')
    path = evaluate(checkpoint=parent/'best_checkpoint.pt',settings=empty,data_root=tmp_path/'data',
                    output_root=tmp_path/'outputs',device='cpu')
    result = json.loads((path/'results.json').read_text(encoding='utf-8'))
    for values in result['summary']['topologies'].values():
        assert values['graph_metrics']['edges']['mean'] == 0
        assert values['pooled_mean_nees_valid'] is None
        assert values['invalid_covariance_agent_runs'] == values['agent_runs']
    assert (path/'report.md').is_file()


def test_production_entry_modules_import_without_running():
    for name in ('gib4cl.experiments.cli','gib4cl.learning.train','gib4cl.evaluation.evaluate','gib4cl.experiments.worker'):
        module = importlib.import_module(name)
        assert callable(module.main)


def test_direct_and_module_clis_share_help_and_require_checkpoint():
    repo = Path(__file__).resolve().parents[1]
    for command in ([sys.executable,'-B','-m','gib4cl','--help'],
                    [sys.executable,'-B',str(repo/'gib4cl/learning/train.py'),'--help'],
                    [sys.executable,'-B',str(repo/'gib4cl/evaluation/evaluate.py'),'--help'],
                    [sys.executable,'-B','-m','gib4cl.experiments.cli','--help']):
        result = subprocess.run(command,cwd=repo,capture_output=True,text=True)
        assert result.returncode == 0,result.stderr
        assert '--help' in result.stdout
    result = subprocess.run([sys.executable,'-B','-m','gib4cl','evaluate'],
                             cwd=repo,capture_output=True,text=True)
    assert result.returncode != 0 and '--checkpoint' in result.stderr


def test_pipeline_cli_runs_generate_train_restore_evaluate(tmp_path,prepared):
    _,cfg = prepared
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg.to_dict()),encoding='utf-8')
    result = main(['run','--config',str(path),'--data-root',str(tmp_path/'data'),
                   '--output-root',str(tmp_path/'outputs'),'--device','cpu','--threads','1'])
    assert (result['models']/'best_checkpoint.pt').is_file()
    evaluation = json.loads((result['evaluation']/'results.json').read_text(encoding='utf-8'))
    assert evaluation['checkpoint']['sha256'] == sha256(result['models']/'best_checkpoint.pt')
    with pytest.raises(FileExistsError,match='reuse-data'):
        main(['run','--config',str(path),'--data-root',str(tmp_path/'data'),
              '--output-root',str(tmp_path/'other')])
