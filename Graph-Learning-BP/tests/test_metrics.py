"""Stage 7: independently specified metrics, aggregation and output semantics."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from gib4cl.config import EvaluationConfig,config_from_dict,load_config
from gib4cl.evaluation.evaluate import write_metric_tables
from gib4cl.evaluation.metrics import (METRICS_VERSION,calculate_metrics,bp_diagnostic_metrics,
                                  summarize_records,metric_protocol)
from gib4cl.evaluation.topology import selection_metrics
from tests.test_objectives import make_graph


@pytest.fixture(autouse=True,scope='module')
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def metrics(errors=(0.,1.,2.,4.),covariance=None,**kwargs):
    estimate = torch.tensor([[v,0.] for v in errors],dtype=torch.float64)
    cov = torch.eye(2,dtype=torch.float64).repeat(len(errors),1,1) if covariance is None else covariance
    return calculate_metrics(estimate,torch.zeros_like(estimate),cov,**kwargs)


def row(sample,errors,topology='dense',repeat=0,**kwargs):
    values = metrics(errors,**kwargs)
    values.update(edges=0,communication_cost=0.,agent_triangles=0,bp_converged=1)
    return dict(sample_id=sample,topology=topology,bp_repeat=repeat,metrics=values,bp_seed=1,support_digest='fixed')


def test_accuracy_curve_and_quantiles_use_distances_and_strict_outage():
    result = metrics(e_th=2.,outage_thresholds=(1.,2.,4.))
    assert result['rmse'] == pytest.approx(math.sqrt(21/4))
    assert result['mae'] == 1.75
    assert result['error_p50'] == 1.5
    assert result['error_p90'] == pytest.approx(3.4)
    assert result['error_p95'] == pytest.approx(3.7)
    assert result['error_max'] == 4.
    assert result['outage'] == .25
    assert result['outage_curve'] == [
        dict(threshold=1.,outage=.5,error_cdf=.5),
        dict(threshold=2.,outage=.25,error_cdf=.75),
        dict(threshold=4.,outage=0.,error_cdf=1.)]
    assert result['agent_error_vectors'] == [[0.,0.],[1.,0.],[2.,0.],[4.,0.]]


def test_correlated_covariance_nees_area_nll_and_radius_have_hand_reference():
    estimate = torch.tensor([[3.,2.]],dtype=torch.float64)
    truth = torch.tensor([[1.,1.]],dtype=torch.float64)
    covariance = torch.tensor([[[4.,1.],[1.,2.]]],dtype=torch.float64)
    value = calculate_metrics(estimate,truth,covariance)
    assert value['agent_nees'] == pytest.approx([8/7])
    assert value['ellipse_area_valid'] == pytest.approx(math.pi*(-2*math.log(.05))*math.sqrt(7))
    assert value['gaussian_nll_valid'] == pytest.approx(math.log(2*math.pi)+math.log(7)/2+4/7)
    assert value['predicted_rmse_valid'] == pytest.approx(math.sqrt(6))
    normal = torch.distributions.MultivariateNormal(estimate[0],covariance_matrix=covariance[0])
    assert value['gaussian_nll_valid'] == pytest.approx(float(-normal.log_prob(truth[0])))
    rotation = torch.tensor([[.6,-.8],[.8,.6]],dtype=torch.float64)
    changed = calculate_metrics(estimate@rotation.T,truth@rotation.T,rotation@covariance@rotation.T)
    for key in ('rmse','mae','mean_nees_valid','ellipse_area_valid','gaussian_nll_valid','predicted_rmse_valid'):
        assert changed[key] == pytest.approx(value[key])


def test_covariance_scaling_separates_accuracy_calibration_and_sharpness():
    base = metrics(errors=(2.,))
    wide = metrics(errors=(2.,),covariance=torch.eye(2,dtype=torch.float64)[None]*4)
    assert base['rmse'] == wide['rmse'] == 2.
    assert wide['mean_nees_valid'] == base['mean_nees_valid']/4
    assert wide['ellipse_area_valid'] == pytest.approx(base['ellipse_area_valid']*4)
    assert wide['gaussian_nll_valid']-base['gaussian_nll_valid'] == pytest.approx(math.log(4)-1.5)


def test_nees_tails_and_invalid_categories_form_a_partition():
    cov = torch.eye(2,dtype=torch.float64).repeat(4,1,1)
    cov[-1] = 0
    result = metrics(covariance=cov)
    assert result['nees_below_interval'] == .25
    assert result['nees_interval_pass'] == .5
    assert result['nees_above_interval'] == 0
    assert result['invalid_covariance_rate'] == .25
    assert sum(result[k] for k in ('nees_below_interval','nees_interval_pass','nees_above_interval','invalid_covariance_rate')) == 1.
    high = metrics(errors=(10.,))
    assert high['nees_above_interval'] == 1.
    assert result['agent_covariance_status'] == ['valid']*3+['nonpositive_covariance']


def test_invalid_covariance_retained_in_coverage_but_not_conditional_scores():
    cov = torch.eye(2,dtype=torch.float64).repeat(2,1,1)
    cov[1] = 0
    result = metrics(errors=(0.,0.),covariance=cov,coverage_levels=(.5,.9))
    assert result['ellipse_coverage'] == .5
    assert result['ellipse_coverage_valid'] == 1.
    assert result['gaussian_nll_valid_agents'] == result['ellipse_area_valid_agents'] == 1
    assert result['agent_gaussian_nll'][1] is result['agent_ellipse_area'][1] is None
    assert all(point['coverage'] == .5 and point['coverage_valid'] == 1. for point in result['coverage_curve'])
    primary = next(p for p in result['coverage_curve'] if p['nominal'] == .95)
    assert primary['coverage'] == result['ellipse_coverage']
    assert primary['chi_square_threshold'] == result['ellipse_threshold']


@pytest.mark.parametrize('cov,reason',[
    ([[0.,0.],[0.,0.]],'nonpositive_covariance'),
    ([[-1.,0.],[0.,1.]],'nonpositive_covariance'),
    ([[1.,2.],[0.,1.]],'asymmetric_covariance'),
    ([[float('inf'),0.],[0.,1.]],'nonfinite_covariance'),
    ([[1.,0.],[0.,1e-15]],'ill_conditioned_covariance'),
    ([[1e-310,0.],[0.,1e-310]],'nonfinite_nees'),
])
def test_invalid_reason_is_explicit_and_all_scores_remain_json_safe(cov,reason):
    result = metrics(errors=(2.,),covariance=torch.tensor([cov],dtype=torch.float64))
    assert result['covariance_status_counts'] == {reason:1}
    assert result['gaussian_nll_valid'] is result['ellipse_area_valid'] is result['predicted_rmse_valid'] is None
    assert result['mean_nees_valid'] is None
    json.dumps(result,allow_nan=False)


def test_nees_and_coverage_curves_match_constructed_chi_square_population():
    # Use exact inverse-CDF positions rather than a noisy Monte Carlo acceptance test.
    probabilities = (np.arange(1000)+.5)/1000
    q = -2*np.log1p(-probabilities)
    value = metrics(errors=np.sqrt(q),coverage_levels=(.5,.8,.9,.95,.99))
    for point in value['coverage_curve']:
        assert point['coverage'] == pytest.approx(point['nominal'],abs=1e-12)
    assert value['mean_nees_valid'] == pytest.approx(2.,abs=.002)
    assert value['nees_interval_pass'] == .95


def graph_with_anchor_triangle():
    data = make_graph([[0.,0.],[1.,0.],[0.,1.],[3.,3.]],[[2.,2.],[9.,9.]],
                      [[0,1],[0,2],[1,2],[0,4],[1,4]],[1.]*5,[1.]*5)
    data.edge_cost = torch.arange(1,6,dtype=torch.float32)
    return data


def test_topology_counts_unknown_cycles_and_candidate_relative_costs():
    data = graph_with_anchor_triangle()
    dense = selection_metrics(data,torch.ones(5))
    assert dense['edges'] == 5 and dense['agent_edges'] == 3 and dense['anchor_edges'] == 2
    assert dense['candidate_agent_triangles'] == dense['agent_triangles'] == 1  # anchor triangle excluded
    assert dense['agent_cycle_rank'] == 1
    assert dense['connected_components'] == 3 and dense['agent_components'] == 2
    assert dense['unanchored_agents'] == dense['isolated_agents'] == 1
    assert dense['anchor_reachable_fraction'] == .75
    assert dense['agent_degree_min'] == 0 and dense['agent_degree_mean'] == 2. and dense['agent_degree_max'] == 3
    assert dense['agents_below_degree_three_fraction'] == .5
    assert dense['edge_saving'] == dense['communication_saving'] == 0
    sparse = selection_metrics(data,torch.tensor([1,0,1,1,0]))
    assert sparse['edges'] == 3 and sparse['edge_retention'] == .6 and sparse['edge_saving'] == .4
    assert sparse['communication_cost'] == 8. and sparse['candidate_communication_cost'] == 15.
    assert sparse['communication_saving'] == pytest.approx(7/15)
    assert sparse['agent_triangles'] == sparse['agent_cycle_rank'] == 0
    assert sparse['agent_triangle_retention'] == 0.


def test_cycle_rank_detects_four_cycle_without_any_triangles():
    data = make_graph([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],[],
                      [[0,1],[1,2],[2,3],[0,3]],[1.]*4,[1.]*4)
    value = selection_metrics(data,torch.ones(4))
    assert value['agent_cycle_rank'] == 1 and value['agent_triangles'] == 0
    assert value['agent_triangle_retention'] is None
    assert value['unanchored_agents'] == 4


def test_empty_candidates_and_zero_cost_have_null_ratios_without_fabricated_savings():
    data = make_graph([[0.,0.],[1.,0.]],[[0.,2.]],[],[],[])
    value = selection_metrics(data,torch.empty(0))
    assert value['edges'] == value['communication_cost'] == 0
    assert value['edge_retention'] is value['edge_saving'] is None
    assert value['communication_retention'] is value['communication_saving'] is None
    assert value['agent_triangle_retention'] is None
    assert value['connected_components'] == 3 and value['agent_components'] == 2
    data = graph_with_anchor_triangle()
    data.edge_cost.zero_()
    value = selection_metrics(data,torch.ones(5))
    assert value['edge_saving'] == 0 and value['communication_saving'] is None
    json.dumps(value,allow_nan=False)


def test_topology_metrics_ignore_edge_order_endpoint_orientation_and_labels():
    data = graph_with_anchor_triangle()
    mask = torch.tensor([1.,0.,1.,1.,0.])
    expected = selection_metrics(data,mask)
    order = torch.tensor([4,0,3,1,2])
    changed = data.clone()
    changed.edge_index = changed.edge_index.flip(0)[:,order]
    changed.edge_cost = changed.edge_cost[order]
    changed.triangle_edge_index = torch.argsort(order)[changed.triangle_edge_index]
    changed.y[:] = 1000
    assert selection_metrics(changed,mask[order]) == expected
    with pytest.raises(ValueError,match='binary'):
        selection_metrics(data,mask*.5)


def test_ess_fraction_threshold_is_strict_and_particle_normalized():
    prediction = SimpleNamespace(ess=torch.tensor([1.,2.,20.]),anchor_reachable=torch.tensor([False,True,True]),
                                 converged=False,iterations=6,message_residual=.25)
    value = bp_diagnostic_metrics(prediction,20,.1)
    assert value['ess_fraction_min'] == .05 and value['ess_fraction_mean'] == pytest.approx(23/60)
    assert value['low_ess_agents'] == 1 and value['low_ess_fraction'] == pytest.approx(1/3)
    assert value['bp_converged'] == 0 and value['bp_message_residual'] == .25
    assert value['unanchored_agents'] == 1
    prediction.ess[0] = 21
    with pytest.raises(ValueError,match='exceeds'):
        bp_diagnostic_metrics(prediction,20)


def test_curve_and_scalar_bootstrap_do_not_treat_agents_or_bp_repeats_as_independent():
    records = [row('a',(0.,)),row('b',(4.,)*9)]
    single = summarize_records(records,bootstrap_samples=300,seed=3)
    repeated = summarize_records(records+[dict(r,bp_repeat=1) for r in records],bootstrap_samples=300,seed=3)
    left,right = (s['topologies']['dense'] for s in (single,repeated))
    assert left['graph_metrics'] == right['graph_metrics']
    assert left['coverage_curve'] == right['coverage_curve']
    assert left['outage_curve'] == right['outage_curve']
    at_three = next(p for p in left['outage_curve'] if p['threshold'] == 3.)
    assert at_three['outage']['mean'] == .5
    assert at_three['outage']['ci_low'] == 0. and at_three['outage']['ci_high'] == 1.
    assert left['pooled_agent_run_mae'] == 3.6
    assert left['graph_metrics']['mae']['mean'] == 2.


def test_bootstrap_is_order_invariant_and_not_affected_by_added_metrics():
    records = [row(sample,(float(i),)) for i,sample in enumerate('abcde')]
    records += [dict(r,topology='gnn',metrics=dict(r['metrics'],rmse=r['metrics']['rmse']+1)) for r in records]
    first = summarize_records(records,bootstrap_samples=100,seed=7)
    assert first == summarize_records(list(reversed(records)),bootstrap_samples=100,seed=7)
    changed = deepcopy(records)
    for r in changed:
        r['metrics']['edges'] = 7.
    second = summarize_records(changed,bootstrap_samples=100,seed=7)
    assert first['topologies']['gnn']['coverage_curve'] == second['topologies']['gnn']['coverage_curve']
    assert first['topologies']['gnn']['graph_metrics']['rmse'] == second['topologies']['gnn']['graph_metrics']['rmse']
    delta = first['paired_graph_differences']['gnn_minus_dense']['rmse']
    assert delta['mean'] == delta['ci_low'] == delta['ci_high'] == 1.


def test_conditional_statistics_record_available_graphs_and_agent_counts():
    invalid = torch.zeros(2,2,2,dtype=torch.float64)
    summary = summarize_records([row('a',(1.,1.),covariance=invalid),row('b',(1.,))],bootstrap_samples=100)
    value = summary['topologies']['dense']
    assert value['graph_metrics']['mean_nees_valid']['graphs'] == 1
    assert value['graph_metrics']['mean_nees_valid']['ci_low'] is None
    assert value['all_invalid_nees_graphs'] == 1 and value['invalid_covariance_agent_runs'] == 2
    assert value['gaussian_nll_valid_agent_runs'] == value['ellipse_area_valid_agent_runs'] == 1
    assert value['graph_metrics']['invalid_covariance_rate']['mean'] == .5
    json.dumps(summary,allow_nan=False)


@pytest.mark.parametrize('change',('duplicate','missing_repeat','different_threshold','different_grid','different_ess'))
def test_mixed_or_unpaired_metric_records_are_rejected(change):
    records = [row('a',(1.,)),row('a',(1.,),topology='gnn')]
    if change == 'duplicate':
        records.append(records[0])
    elif change == 'missing_repeat':
        records[1]['bp_repeat'] = 1
    elif change == 'different_threshold':
        records[1]['metrics']['outage_threshold'] = 4.
    elif change == 'different_grid':
        records[1]['metrics']['coverage_curve'].pop()
    else:
        records[1]['metrics']['ess_fraction_threshold'] = .2
    with pytest.raises(ValueError):
        summarize_records(records)


@pytest.mark.parametrize('field,values',[
    ('coverage_levels',[]),('coverage_levels',[.9,.5]),('coverage_levels',[.9,.9]),
    ('coverage_levels',[1.]),('coverage_levels',[True]),('coverage_levels',[float('nan')]),
    ('outage_thresholds',[0.]),('outage_thresholds',[3.,2.]),
    ('ess_fraction_threshold',0.),('ess_fraction_threshold',1.1),
])
def test_evaluation_configuration_rejects_ambiguous_grids_and_thresholds(field,values):
    with pytest.raises(ValueError):
        replace(EvaluationConfig(),**{field:values})


def test_configuration_roundtrip_keeps_data_identity_and_includes_primary_thresholds():
    cfg = load_config('configs/smoke-pipeline.json')
    changed = replace(cfg,evaluation=replace(cfg.evaluation,outage_threshold=2.5,alpha=.02,
                                            outage_thresholds=[1.,2.],coverage_levels=[.5,.9]))
    restored = config_from_dict(json.loads(json.dumps(changed.to_dict())))
    assert restored == changed and restored.data_fingerprint == cfg.data_fingerprint
    values = metrics(e_th=2.5,alpha=.02,outage_thresholds=restored.evaluation.outage_thresholds,
                     coverage_levels=restored.evaluation.coverage_levels)
    assert 2.5 in [p['threshold'] for p in values['outage_curve']]
    assert .98 in [p['nominal'] for p in values['coverage_curve']]


def test_flat_exports_include_counts_cis_curves_and_blank_nulls(tmp_path):
    records = [row('only',(1.,))]
    summary = summarize_records(records,bootstrap_samples=10)
    write_metric_tables(tmp_path,dict(records=records,summary=summary))
    with (tmp_path/'trial_metrics.csv').open(encoding='utf-8',newline='') as handle:
        trials = list(csv.DictReader(handle))
    with (tmp_path/'summary_metrics.csv').open(encoding='utf-8',newline='') as handle:
        stats = list(csv.DictReader(handle))
    assert trials[0]['rmse'] == '1.0' and trials[0]['edge_retention'] == ''
    stat = next(r for r in stats if r['metric'] == 'rmse')
    assert stat['std'] == stat['ci_low'] == stat['ci_high'] == '' and stat['graphs'] == '1'
    assert any(r['metric'] == 'coverage_curve.coverage' and r['axis_value'] == '0.95' for r in stats)
    assert metric_protocol()['version'] == METRICS_VERSION
