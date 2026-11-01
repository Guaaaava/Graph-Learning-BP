"""Independent exactness and interpretation checks for branched single cycles."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch

from gib4cl.experiments.tree_loop.config import Condition, Settings
from gib4cl.experiments.tree_loop.model import EDGES, MASKS, graph_spec, make_trial, prior_points, as_support
from gib4cl.experiments.tree_loop.reference import exact_marginals
from gib4cl.experiments.tree_loop.metrics import nees_chi_square_summary
from gib4cl.experiments.tree_loop.gaussian_control import controlled_rings
from gib4cl.experiments.tree_loop.experiment import run_experiment
from gib4cl.experiments.tree_loop.calibration import summarize_focus_coverage
from gib4cl.experiments.tree_loop.figures import _paper_layout
from gib4cl.inference import ParticleBP
from tests.references.particle_posterior import exact_discrete


def branched(cases=2):
    return Condition('branches', ((0,0),(-10,0),(10,0),(-20,1),(-10,-10),(20,1),(10,-10)),
                     cases, graph_family='branched7')


@pytest.fixture(autouse=True, scope='module')
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('name', ['tree', 'loop3', 'loop4', 'loop5', 'forest'])
def test_seven_node_reference_matches_full_joint_enumeration(name):
    c = branched()
    trial, spec = make_trial(c, 12, 0), graph_spec(c)
    points, _ = prior_points(c, 42, 0, 0, 4)
    logs = np.random.default_rng(8).normal(size=(7,4))
    logs[:,0] = -np.inf
    mask = spec.masks[name] if name != 'forest' else (1,0,0,1,0,0,0,0,0)
    actual = exact_marginals(points, logs, trial.measurements, trial.variances, mask, edges=spec.edges)
    expected = exact_discrete(trial.graph, points, logs, np.array(mask))
    np.testing.assert_allclose(actual.weights, expected.marginals, atol=1e-13, rtol=2e-11)
    np.testing.assert_allclose(actual.mean, expected.mean, atol=2e-12, rtol=2e-11)
    np.testing.assert_allclose(actual.covariance, expected.covariance, atol=2e-12, rtol=2e-11)
    assert actual.log_partition_relative == pytest.approx(expected.log_partition_relative, abs=2e-11)


@pytest.mark.parametrize('mask', [*MASKS.values(), (0,0,0)])
def test_general_solver_matches_existing_three_node_contraction(mask):
    c = Condition('three', ((0,0),(10,1),(20,0)), 1)
    trial = make_trial(c, 9, 0)
    points, _ = prior_points(c, 30, 0, 0, 7)
    logs = np.random.default_rng(5).normal(size=(3,7))
    a = exact_marginals(points, logs, trial.measurements, trial.variances, mask)
    b = exact_marginals(points, logs, trial.measurements, trial.variances, mask, edges=EDGES)
    np.testing.assert_allclose(a.weights, b.weights, atol=2e-14, rtol=2e-12)
    assert a.log_partition_relative == pytest.approx(b.log_partition_relative, abs=2e-12)


def test_seven_node_tree_bp_and_topology_invariants():
    c = branched()
    trial, spec = make_trial(c, 12, 0), graph_spec(c)
    points, _ = prior_points(c, 42, 0, 0, 16)
    support = as_support(points)
    result = ParticleBP(16, 8).infer(trial.graph, torch.tensor(spec.masks['tree']), support=support)
    expected = exact_marginals(points, support.log_prior_weights.numpy(), trial.measurements,
                               trial.variances, spec.masks['tree'], edges=spec.edges)
    np.testing.assert_allclose(result.weights.numpy(), expected.weights, atol=1e-13)
    assert result.converged and len(spec.labels)==7 and spec.focus==0
    for name in ('loop3','loop4','loop5'):
        assert sum(spec.masks[name])==7
    xy,paper_labels=_paper_layout(spec,c.graph_family)
    closure=spec.edges[np.array(spec.masks['loop5'])-np.array(spec.masks['tree'])==1][0]
    assert {paper_labels[node] for node in closure}=={'b','c'}
    assert len(set(paper_labels))==7 and paper_labels[spec.focus]=='i'
    assert xy[spec.focus,1]>max(xy[1:,1])
    with pytest.raises(ValueError, match='single-cycle'):
        exact_marginals(points, support.log_prior_weights.numpy(), trial.measurements,
                        trial.variances, np.ones(9), edges=spec.edges)


def test_graph_family_configuration_requires_matching_geometry_and_primary_loop():
    settings = Settings((branched(),), topologies=('tree','loop3','loop4','loop5'), primary_loop='loop5')
    assert settings.primary_loop == 'loop5'
    with pytest.raises(ValueError, match='primary_loop'):
        replace(settings, primary_loop='loop')
    with pytest.raises(ValueError, match='shape'):
        replace(branched(), centers=((0,0),(1,1),(2,0)))


def test_auxiliary_gaussian_control_matches_closed_form_and_known_triangle():
    value=controlled_rings()
    for row in value['rows']:
        assert row['variance_ratio']==pytest.approx(row['closed_form_ratio'],abs=2e-12)
    for b in (1.,4.):
        ratios=[r['variance_ratio'] for r in value['rows'] if r['edge_precision']==b]
        assert np.all(np.diff(ratios)>0) and max(ratios)<1
    triangle=next(r for r in value['rows'] if r['ring_length']==3 and r['edge_precision']==4.)
    assert triangle['exact_variance']==pytest.approx(5/13)
    assert triangle['gaussian_bp_variance']==pytest.approx(1/math.sqrt(17))


def test_nees_two_sided_rejection_separates_both_tails_and_invalid_values():
    c=Condition('three',((0,0),(10,1),(20,0)),2)
    settings=Settings((c,),num_particles=8,particle_grid=(8,),bootstrap_samples=30)
    case={'condition':'three','runs':[{'particles':8,'topologies':{
        name:{'agents':{'nees':[0.,100.,None]}} for name in ('tree','loop')}}]}
    rows=nees_chi_square_summary([case,case],settings,8)['groups']['three']['loop']
    at95=next(r for r in rows if r['nominal']==.95)
    assert at95['expected_tail']==pytest.approx(.025)
    assert at95['below']['mean']==at95['above']['mean']==at95['invalid']['mean']==pytest.approx(1/3)
    assert at95['inside']['mean']==0


def test_focus_calibration_uses_only_target_node_and_averages_repeats_within_trials(tmp_path):
    settings=Settings((branched(2),),num_particles=4,particle_grid=(4,),repeats=2,
                      continuous_cases=0,bootstrap_samples=40,topologies=('tree','loop5'),primary_loop='loop5')
    points=np.zeros((7,4,2))
    points[:,:,0]=np.arange(4)
    exact=np.zeros((7,4)); exact[:,0]=1.
    cases=[]
    for case_index in range(2):
        runs=[]
        for repeat in range(2):
            weights=exact.copy()
            if case_index or repeat:
                weights[0]=[0.,0.,0.,1.]
            path=tmp_path/f'{case_index}-{repeat}.npz'
            np.savez(path,points=points,tree_weights=exact,tree_reference_weights=exact,
                     loop5_weights=weights,loop5_reference_weights=exact)
            runs.append({'particles':4,'repeat':repeat,'artifact':{'path':path.name}})
        cases.append({'condition':'branches','case_index':case_index,'truth':np.zeros((7,2)).tolist(),'runs':runs})
    result=summarize_focus_coverage(tmp_path,cases,settings,[4])
    group=result['groups']['branches']['4']
    at95=result['coverage_levels'].index(.95)
    value=group['topologies']['loop5']['coverage_x'][at95]
    assert value['mean']==pytest.approx(.25) and value['valid_cases']==2
    assert group['topologies']['loop5']['finite_reference_coverage_x'][at95]['mean']==1
    for case in cases:
        for node in range(1,7):case['truth'][node][0]=1e6
    assert summarize_focus_coverage(tmp_path,cases,settings,[4])==result


def test_seven_node_pipeline_and_reader_figures(tmp_path):
    settings=Settings((branched(2),),purpose='smoke',num_particles=20,particle_grid=(12,20),
                      repeats=1,iterations=10,bootstrap_samples=30,continuous_cases=0,
                      topologies=('tree','loop3','loop4','loop5'),primary_loop='loop5')
    root=tmp_path/'seven'
    result=run_experiment(settings,root,progress=None)
    group=result['groups']['branches']['20']
    assert result['status']=='complete' and set(group['topologies'])==set(settings.topologies)
    assert result['numerical_checks']['tree_max_marginal_error']<1e-12
    for name in settings.topologies:
        assert group['topologies'][name]['statistics']['focus_variance_ratio_x']['valid_cases']==2
    for name in ('branches-mechanism','branches-calibration','branches-cycle-length-comparison',
                 'branches-calibration-all-agents','branches-covariance-iterations-full','gaussian-ring-control'):
        assert (root/'figures'/f'{name}.pdf').read_bytes().startswith(b'%PDF')
    assert not (root/'figures/branches-nees-consistency-supplement.pdf').exists()
    assert (root/'figures/captions.md').is_file()
    focus=result['focus_calibration']
    assert focus['groups']['branches']['20']['node_label']=='i' and len(focus['coverage_levels'])>=50
    assert (root/'focus-calibration.json').is_file()
