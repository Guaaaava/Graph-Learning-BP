"""Scientific checks for the Chapter 2 tree/loop diagnostic experiment."""
from dataclasses import replace
import json
import math
import numpy as np
import pytest
import torch

from gib4cl.experiments.tree_loop.config import Condition, Settings, from_dict
from gib4cl.experiments.tree_loop.experiment import render_outputs, run_experiment
from gib4cl.experiments.tree_loop.metrics import bootstrap_stat, covariance_comparison, summarize_cases, weighted_quantile
from gib4cl.experiments.tree_loop.model import MASKS, as_support, make_trial, prior_points
from gib4cl.experiments.tree_loop.reference import exact_marginals, hermite_support
from gib4cl.inference import ParticleBP, ParticleSupport
from gib4cl.runtime import sha256
from tests.references.particle_posterior import exact_discrete


@pytest.fixture(autouse=True, scope='module')
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def condition(cases=2):
    return Condition('corridor', ((0, 0), (10, 1), (20, 0)), cases)


def tiny_settings(**kwargs):
    values = dict(conditions=(condition(),), purpose='smoke', num_particles=12,
                  particle_grid=(8, 12), repeats=1, iterations=8, chunk_size=12,
                  bootstrap_samples=40, pilot_cases=2, continuous_cases=0,
                  quadrature_orders=(3, 5, 7))
    values.update(kwargs)
    return Settings(**values)


@pytest.mark.parametrize('mask', [(0, 0, 0), (1, 0, 0), *MASKS.values()])
def test_contraction_matches_independent_full_joint_enumeration(mask):
    trial = make_trial(condition(), 12, 0)
    points, _ = prior_points(condition(), 42, 0, 0, 7)
    logs = np.random.default_rng(7).normal(size=(3, 7))
    logs[:, 0] = -np.inf  # include genuinely zero prior mass
    actual = exact_marginals(points, logs, trial.measurements, trial.variances, mask)
    expected = exact_discrete(trial.graph, points, logs, np.array(mask))
    np.testing.assert_allclose(actual.weights, expected.marginals, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(actual.mean, expected.mean, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(actual.covariance, expected.covariance, atol=2e-13, rtol=2e-12)
    assert actual.log_partition_relative == pytest.approx(expected.log_partition_relative, abs=2e-12)


def test_reference_budget_and_invalid_variance_are_rejected():
    trial = make_trial(condition(), 12, 0)
    points, _ = prior_points(condition(), 42, 0, 0, 7)
    logs = np.zeros((3, 7))
    with pytest.raises(ValueError, match='P <='):
        exact_marginals(points, logs, trial.measurements, trial.variances, MASKS['loop'], max_particles=6)
    with pytest.raises(ValueError, match='variances'):
        exact_marginals(points, logs, trial.measurements, [0, 1, 1], MASKS['loop'])


def test_all_spanning_trees_match_reference_on_shared_prior_support():
    trial = make_trial(condition(), 12, 0)
    points, _ = prior_points(condition(), 42, 0, 0, 17)
    support = as_support(points)
    for name in ('tree', 'tree_01_02', 'tree_12_02'):
        mask = MASKS[name]
        actual = ParticleBP(17, 8).infer(trial.graph, torch.tensor(mask), support=support)
        expected = exact_marginals(points, support.log_prior_weights.numpy(), trial.measurements, trial.variances, mask)
        np.testing.assert_allclose(actual.weights.numpy(), expected.weights, atol=2e-13, rtol=2e-12)
        assert actual.converged


def test_scalar_gaussian_limit_recovers_analytic_overconfidence_without_linearizing_bp():
    # Ordered collinear supports at widely separated means make Euclidean
    # ranges exactly signed differences; high-order 1D quadrature has a known
    # Gaussian analytic answer. y is fixed only in this auxiliary unit test.
    c = Condition('line', ((-1000, 0), (0, 0), (1000, 0)), 1)
    trial = make_trial(c, 12, 0)
    trial.measurements[:] = [1000., 1000., 2000.]
    trial.graph.edge_attr[:, 0] = torch.from_numpy(trial.measurements)
    nodes, weights = np.polynomial.hermite.hermgauss(81)
    points = np.zeros((3, 81, 2))
    points[:, :, 0] = np.array([-1000., 0., 1000.])[:, None]+math.sqrt(2)*nodes
    logs = np.broadcast_to(np.log(weights/math.sqrt(math.pi)), (3, 81)).copy()
    support = ParticleSupport(torch.from_numpy(points), torch.from_numpy(logs), torch.arange(3))
    reference = exact_marginals(points, logs, trial.measurements, trial.variances, MASKS['loop'])
    result = ParticleBP(81, 50).infer(trial.graph, support=support)
    np.testing.assert_allclose(reference.covariance[:, 0, 0], 5/13, rtol=2e-6)
    np.testing.assert_allclose(result.covariance[:, 0, 0].numpy(), 1/math.sqrt(17), rtol=2e-6)
    assert result.converged


def test_prefix_supports_and_paired_conditions_share_standard_normal_draws():
    c = condition()
    other = replace(c, name='other', centers=((0, 0), (10, 17), (20, 0)), range_std=2.)
    points, seed = prior_points(c, 42, 4, 2, 32, 'pilot')
    other_points, other_seed = prior_points(other, 42, 4, 2, 32, 'pilot')
    np.testing.assert_allclose(points-np.array(c.centers)[:, None],
                               other_points-np.array(other.centers)[:, None])
    assert seed == other_seed
    assert as_support(points[:, :8]).points.shape == (3, 8, 2)
    trial = make_trial(c, 12, 4)
    repeated = make_trial(c, 12, 4)
    np.testing.assert_array_equal(trial.truth, repeated.truth)
    assert not hasattr(trial.graph, 'truth') and not hasattr(trial.graph, 'y')
    assert not np.array_equal(trial.truth, make_trial(c, 12, 5).truth)
    assert not np.array_equal(trial.truth, make_trial(c, 12, 4, 'pilot').truth)


def test_weighted_intervals_respect_mass_and_exclude_zero_weight_atoms():
    actual = weighted_quantile([100, 2, 0, 1], [0, .8, .1, .1], [0, .1, .2, .5, 1])
    np.testing.assert_array_equal(actual, [0, 0, 1, 2, 2])
    with pytest.raises(ValueError, match='weighted'):
        weighted_quantile([0, 1], [1, -1], [.5])


def test_generalized_ratios_detect_opposite_directional_biases():
    ref = np.array([[3., 1.], [1., 2.]])
    chol = np.linalg.cholesky(ref)
    covariance = chol @ np.diag([.25, 2.]) @ chol.T
    values = covariance_comparison(covariance[None], ref[None])
    assert values['lambda_min'] == pytest.approx([.25])
    assert values['lambda_max'] == pytest.approx([2.])
    assert values['area_ratio'] == pytest.approx([math.sqrt(.5)])


def test_quadrature_integrates_unobserved_prior_moments():
    c = condition()
    trial = make_trial(c, 12, 0)
    points, logs = hermite_support(c.centers, c.prior_std, 5)
    value = exact_marginals(points, logs, trial.measurements, trial.variances, (0, 0, 0))
    np.testing.assert_allclose(value.mean, c.centers, atol=2e-14)
    np.testing.assert_allclose(value.covariance, np.tile(np.eye(2), (3, 1, 1)), atol=2e-14)


def test_statistics_bootstrap_independent_trials_not_particle_repeats():
    cases = []
    for case_index, numbers in enumerate(((1., 3.), (5., 7.))):
        runs = []
        for repeat, value in enumerate(numbers):
            topologies = {}
            for name, offset in (('tree', 0), ('loop', 2)):
                topologies[name] = {'scalar': {'mse': value+offset, 'converged': 1, 'variance_ratio_x': 1.,
                                               'lambda_min': 1., 'lambda_max': 1., 'nees_normalized': value}, 'coverage_x': [.5]*5,
                                    'exact_coverage_x': [.5]*5}
            runs.append({'particles': 12, 'repeat': repeat, 'topologies': topologies})
        cases.append({'condition': 'corridor', 'case_index': case_index, 'runs': runs})
    group = summarize_cases(cases, tiny_settings(repeats=2), [12])['corridor']['12']
    stat = group['topologies']['tree']['statistics']['mse']
    assert group['independent_cases'] == stat['valid_cases'] == 2
    assert stat['mean'] == 4.
    assert stat['std'] == pytest.approx(math.sqrt(8))
    delta = group['paired_loop_minus_tree']['mse']
    assert delta['mean'] == delta['ci_low'] == delta['ci_high'] == 2.


def test_single_valid_trial_does_not_get_a_zero_width_confidence_interval():
    stat = bootstrap_stat([None, 3., None], np.array([[0, 1, 2], [1, 1, 1]]))
    assert stat['mean'] == 3. and stat['valid_cases'] == 1
    assert stat['ci_low'] is None and stat['ci_high'] is None


@pytest.mark.parametrize('change', [{'num_particles': True}, {'particle_grid': [8, 8]},
                                   {'topologies': ['loop']}, {'damping': 1}, {'unknown': 1}])
def test_invalid_settings_rejected(change):
    config = tiny_settings().to_dict()
    config.update(change)
    with pytest.raises(ValueError):
        from_dict(config)


def test_run_resume_replay_and_raw_artifact_integrity(tmp_path, monkeypatch):
    settings = tiny_settings()
    root = tmp_path/'experiment'
    partial = run_experiment(settings, root, stop_after_cases=1, make_plots=False, progress=None)
    assert partial['status'] == 'partial' and partial['completed_cases'] == 1
    first = root/'cases/corridor/000000.json'
    first_hash = sha256(first)
    completed = run_experiment(settings, root, resume=True, make_plots=False, progress=None)
    assert completed['status'] == 'complete' and completed['completed_cases'] == 2
    assert sha256(first) == first_hash
    assert completed['numerical_checks']['tree_max_marginal_error'] < 1e-12
    with pytest.raises(ValueError, match='identical settings'):
        run_experiment(replace(settings, iterations=9), root, resume=True, progress=None)
    with pytest.raises(FileExistsError):
        run_experiment(settings, root, progress=None)
    monkeypatch.setattr(ParticleBP, 'infer', lambda *a, **k: pytest.fail('plot must not run BP'))
    rendered = render_outputs(root)
    assert rendered == completed
    assert (root/'figures/corridor-mechanism.png').is_file()
    assert (root/'figures/corridor-calibration.pdf').read_bytes().startswith(b'%PDF')
    assert '<svg' in (root/'figures/corridor-mechanism.svg').read_text(encoding='utf-8')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    case = json.loads(first.read_text(encoding='utf-8'))
    artifact = root/case['runs'][0]['artifact']['path']
    artifact.write_bytes(artifact.read_bytes()+b'corrupted')
    with pytest.raises(ValueError, match='Raw artifact checksum'):
        render_outputs(root, manifest, make_plots=False)
