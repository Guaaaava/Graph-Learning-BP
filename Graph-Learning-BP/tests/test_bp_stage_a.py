"""Independent numerical and accounting checks for local stage A."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from gib4cl.config import load_config
from gib4cl.data.dataset import graph_to_data
from gib4cl.data.generate import build_dataset
from gib4cl.inference import ParticleBP, ParticleSupport
from gib4cl.runtime import state_digest
from tests.test_particle_bp import graph_case, explicit_support
from tests.test_objectives import make_graph
from tests.references.particle_posterior import exact_discrete
from tools.bp_joint_reference import prepare_joint, joint_cpu_reference
from tools.bp_timing import PhaseTimer, benchmark_pairs, panel_ledger
from tools.bp_cost import workload, estimate, validate_merge
from tools.bp_sample_plan import choose_samples, prepare_samples, verify_original_val


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('kind', ['single', 'chain', 'tree', 'forest', 'cycle'])
@pytest.mark.parametrize('rounds', [0, 1, 4])
def test_joint_reference_each_round_matches_independent_original_runs(kind, rounds):
    data = graph_case(kind); support = explicit_support(data, 4)
    masks = torch.stack([torch.ones(data.num_edges), torch.arange(data.num_edges)%2, torch.zeros(data.num_edges)])
    problem = prepare_joint(data, masks, ('dense', 'partial', 'empty'), support)
    actual = joint_cpu_reference(problem, iterations=rounds, return_history=True)
    for row, mid in enumerate(problem.model_ids):
        expected = ParticleBP(4, rounds, chunk_size=3).infer(data, masks[row], support=support, return_history=True)
        for field in ('mean', 'covariance', 'weights', 'log_messages', 'ess', 'anchor_reachable', 'directed_edge_index',
                      'message_edge_to_measurement'):
            np.testing.assert_allclose(actual['models'][mid][field], getattr(expected, field).numpy(), rtol=2e-12, atol=2e-13)
        assert actual['models'][mid]['iterations'] == expected.iterations
        histories = [step[mid] for step in actual['history'] if mid in step]
        assert len(histories) == len(expected.history)
        for first, second in zip(histories, expected.history):
            np.testing.assert_allclose(first['weights'], second['weights'].numpy(), rtol=2e-12, atol=2e-13)
    assert actual['workload']['shared_pair_distances'] <= actual['workload']['separate_pair_distances']


def test_joint_tree_matches_joint_enumeration_and_other_models_do_not_leak():
    data = graph_case('tree'); support = explicit_support(data, 4)
    masks = torch.stack([torch.ones(data.num_edges), torch.zeros(data.num_edges)])
    original = joint_cpu_reference(prepare_joint(data, masks, ('A', 'B'), support))
    masks[1] = 1
    changed = joint_cpu_reference(prepare_joint(data, masks, ('A', 'B'), support))
    np.testing.assert_array_equal(original['models']['A']['weights'], changed['models']['A']['weights'])
    exact = exact_discrete(data, support.points, support.log_prior_weights)
    np.testing.assert_allclose(original['models']['A']['weights'], exact.marginals, rtol=2e-12, atol=2e-13)
    assert changed['workload']['shared_pair_distances']*20 == changed['workload']['separate_pair_distances']


def test_joint_reference_copies_inputs_detects_mutation_and_rejects_unbounded_work():
    data = graph_case('cycle'); support = explicit_support(data, 4)
    masks = torch.ones((1, data.num_edges))
    problem = prepare_joint(data, masks, ('A',), support)
    data.edge_attr[0, 0] += 1; masks.zero_(); support.points.add_(1)
    problem.verify()
    with pytest.raises(ValueError, match='budget'):
        joint_cpu_reference(problem, max_kernel_elements=1)
    problem.support.points.add_(1)
    with pytest.raises(ValueError, match='mutated'):
        joint_cpu_reference(problem)
    with pytest.raises(ValueError, match='distinct'):
        prepare_joint(data, torch.ones((2, data.num_edges)), ('A', 'A'), support)


def test_joint_reference_preserves_underflow_and_zero_prior_masses():
    data = make_graph([[0., 0.], [5., 0.]], [], [[0, 1]], [-100.], [.01])
    support = ParticleSupport(torch.tensor([[[0.,0.],[1.,0.],[2.,0.]], [[5.,0.],[6.,0.],[7.,0.]]], dtype=torch.float64),
                              torch.zeros(2,3,dtype=torch.float64), torch.tensor([0,1]))
    support.log_prior_weights[:,0] = -torch.inf
    masks = torch.tensor([[1.], [0.]])
    actual = joint_cpu_reference(prepare_joint(data, masks, ('full', 'none'), support), iterations=3)
    for row, mid in enumerate(('full', 'none')):
        expected = ParticleBP(3,3).infer(data,masks[row],support=support)
        np.testing.assert_allclose(actual['models'][mid]['weights'], expected.weights, rtol=1e-12, atol=1e-13)


def test_phase_timer_does_not_overlap_or_invent_missing_times():
    clock = [0.]
    timer = PhaseTimer(clock=lambda:clock[0])
    with timer.phase('load'): clock[0] += 2.
    clock[0] += 1.
    with timer.phase('inference'):
        with pytest.raises(ValueError):
            with timer.phase('save'): pass
        clock[0] += 5.
    result = timer.snapshot()
    assert result['wall_seconds'] == 8 and result['uninstrumented_seconds'] == 1
    assert result['phase_seconds']['metrics'] is None
    with pytest.raises(RuntimeError):
        with timer.phase('save'): raise RuntimeError('disk')
    assert timer.snapshot()['failed_phases'] == ['save']


def methods():
    return [dict(method_id='dense',role='baseline',training_seeds=[],seconds_per_prediction=20.,
                 overhead_seconds=2.,inference_speedup=2.,timing_source='synthetic'),
            dict(method_id='learned',role='learned',training_seeds=[101,102,103],seconds_per_prediction=10.,
                 overhead_seconds=1.,inference_speedup=2.,timing_source='synthetic')]


def test_workload_shares_baselines_deduplicates_extra_repeats_and_keeps_overhead():
    work = workload(graphs=[0,1],bp_seeds=[700],methods=methods(),inference_identity={'test':'synthetic'},
                    extra_graphs=[0],extra_bp_seeds=[700,701])
    result = estimate(work,devices=2,margin=1.)
    assert result['units'] == 12 and result['baseline_units'] == 3
    assert result['inference_hours']*3600 == 75.
    assert result['estimated_wall_hours']*3600 == 60.
    assert result['planned_billed_device_hours']*3600 == 120.
    assert all(len(s['graphs']) == 1 for s in result['shards'])
    with pytest.raises(ValueError,match='shared'):
        bad = methods(); bad[0]['training_seeds'] = [101]
        workload(graphs=[0],bp_seeds=[700],methods=bad,inference_identity={'x':1})


def synthetic_outputs(work, shards):
    byid = {r['unit_id']:r for r in work['units']}
    return [dict(device=s['device'],workload_identity=work['identity'],records=[
        dict(unit_id=u,status='complete',input_digest='input'+str(byid[u]['graph']),
             support_digest='support'+str(byid[u]['graph'])+'-'+str(byid[u]['bp_seed'])) for u in s['unit_ids']]) for s in shards]


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'identity', 'support', 'input', 'shard'])
def test_merge_rejects_partial_duplicated_or_unpaired_results(bad):
    work = workload(graphs=[0,1,2],bp_seeds=[700,701],methods=methods(),inference_identity={'x':1})
    shards = estimate(work,devices=2)['shards']; outputs = synthetic_outputs(work,shards)
    assert validate_merge(work,shards,outputs)['validated_units'] == 24
    if bad=='missing': outputs[0]['records'].pop()
    elif bad=='duplicate': outputs[0]['records'].append(outputs[0]['records'][0])
    elif bad=='identity': outputs[0]['workload_identity']='other'
    elif bad=='support': outputs[0]['records'][1]['support_digest']='other'
    elif bad=='input': outputs[0]['records'][1]['input_digest']='other'
    else: outputs[1]['records'].append(outputs[0]['records'].pop())
    with pytest.raises(ValueError): validate_merge(work,shards,outputs)


def test_warm_speed_ratio_uses_total_work_not_mean_ratios():
    state=dict(identity=dict(plan=dict(profile=False,repeats=2,variants=[dict(name='new')])),records={})
    for case,ref,new in [('big',100.,50.),('small',1.,1.)]:
        for name,seconds in [('reference',ref),('new',new)]:
            state['records'][case+name]=dict(case=case,variant=name,repeat=1,seconds=seconds)
    assert benchmark_pairs(state)['rows'][0]['speedup'] == 101/51
    state['identity']['plan']['profile']=True
    assert not benchmark_pairs(state)['comparable']


def test_sample_selection_is_deterministic_and_does_not_use_speed_or_outcomes():
    rows=[dict(graph_index=i,dense_agent_edges=(i*17)%71) for i in range(32)]
    chosen=choose_samples(rows)
    assert chosen == choose_samples(list(reversed(rows)))
    assert len(chosen)==8 and {0,5,6} <= {r['index'] for r in chosen}


def test_sample_regeneration_mismatch_is_never_accepted_as_original_data():
    cfg=load_config('configs/smoke-pipeline.json')
    bundle=build_dataset(cfg,'val',8)
    inputs={r['sample_id']:state_digest(graph_to_data(r).to_dict()) for r in bundle['graphs']}
    panel=dict(context=dict(method='particle_bp',dataset=dict(split='val',series_id=bundle['metadata']['series_id']),
                            base_config=cfg.to_dict(),input_digests=inputs))
    good,_=prepare_samples(panel,prefix=8)
    assert good['historical_inputs_verified']==8 and not good['execution_ready']
    first=next(iter(inputs));inputs[first]='different'
    with pytest.raises(ValueError,match='differ'):prepare_samples(panel,prefix=8)
    diagnostic,_=prepare_samples(panel,prefix=8,diagnostic_regeneration=True)
    assert diagnostic['historical_inputs_verified']==7 and diagnostic['pending_server_verification']==[0]
    assert not diagnostic['execution_ready']


def test_original_dataset_gate_uses_original_hashes_not_rebuilt_substitutes():
    cfg=load_config('configs/smoke-pipeline.json');bundle=build_dataset(cfg,'val',8)
    dataset=[graph_to_data(r) for r in bundle['graphs']]
    identity=dict(sha256='original-file-hash',dataset_id=bundle['metadata']['dataset_id'],
        series_id=bundle['metadata']['series_id'],split='val',num_graphs=8,data_fingerprint=cfg.data_fingerprint)
    panel=dict(context=dict(method='particle_bp',dataset=identity,base_config=cfg.to_dict(),
        input_digests={d.sample_id:state_digest(d.to_dict()) for d in dataset}))
    plan,_=prepare_samples(panel,prefix=8)
    assert verify_original_val(plan,dataset,identity)['status']=='verified_original_val'
    with pytest.raises(ValueError,match='dataset identity'):
        verify_original_val(plan,dataset,dict(identity,sha256='other'))
    dataset[0].x[0,0]+=1.
    with pytest.raises(ValueError,match='input digest'):
        verify_original_val(plan,dataset,identity)


def test_duplicate_shard_assignment_is_rejected_before_merge():
    work=workload(graphs=[0,1],bp_seeds=[700],methods=methods(),inference_identity={'x':1})
    shards=estimate(work,devices=2)['shards'];outputs=synthetic_outputs(work,shards)
    shards[0]['unit_ids'].append(shards[0]['unit_ids'][0])
    with pytest.raises(ValueError,match='Duplicate'):validate_merge(work,shards,outputs)


def test_panel_ledger_deduplicates_cached_evidence_but_rejects_test_and_conflicts(tmp_path):
    context=dict(method='particle_bp',dataset=dict(split='val'),bp_settings=dict(particles=3,iterations=2,seeds=[700]),
        base_config=dict(bp=dict(compute_dtype='float64')),evaluation=dict(threshold=.5),sources={},
        models=[dict(model_id='dense',candidate_id='dense',training_seed=None)])
    row=dict(status='complete',sample_id='series:0',input_digest='i',support_digest='s',bp_seed=700,
        selected_edge_ids=[0],model_id='dense',inference_seconds=5.,mean=[[0,0]],covariance=[[[1,0],[0,1]]],
        mse=0.,coverage95=1.,communication_cost=2.,metrics=dict(agent_edges=1,edges=1))
    panel=dict(schema='gib4cl.selection-records.v1',status='complete',context=context,records=[row])
    first=tmp_path/'a.json'; second=tmp_path/'b.json'
    first.write_text(json.dumps(panel),encoding='utf-8'); second.write_text(json.dumps(panel),encoding='utf-8')
    ledger=panel_ledger([first,second])
    assert ledger['unique_predictions']==1 and ledger['unique_timing_observations']==1
    assert ledger['inference_seconds']==5 and ledger['records'][0]['phase_seconds']['load'] is None
    panel['records'][0]['mean']=[[1,0]];second.write_text(json.dumps(panel),encoding='utf-8')
    with pytest.raises(ValueError,match='Conflicting'): panel_ledger([first,second])
    panel['context']['dataset']['split']='test';second.write_text(json.dumps(panel),encoding='utf-8')
    with pytest.raises(ValueError,match='validation'): panel_ledger([second])
