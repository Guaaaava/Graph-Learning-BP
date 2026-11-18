"""Numerical equivalence, bounded memory, isolation and benchmark contracts."""
from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from gib4cl.config import load_config
from gib4cl.data.generate import create_dataset
from gib4cl.inference import ParticleBP, ParticleSupport
from tests.references.particle_posterior import exact_discrete
from tests.test_objectives import make_graph
from tests.test_particle_bp import graph_case, explicit_support
from tools.bp_cached import CachedParticleBP
from tools import bp_benchmark as bench


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def compare_results(x, y, *, exact=True):
    for name in bench.FIELDS:
        torch.testing.assert_close(getattr(x, name), getattr(y, name),
                                   rtol=0 if exact else 1e-12, atol=0 if exact else 1e-13)
    assert x.iterations == y.iterations and x.converged == y.converged
    assert x.message_residual == y.message_residual


@pytest.mark.parametrize('kind', ['anchors', 'single', 'chain', 'cycle', 'forest'])
@pytest.mark.parametrize('cap', [0, 1, 73, 100000])
def test_cached_trajectory_matches_original_at_every_round(kind, cap):
    data = graph_case(kind)
    support = explicit_support(data, 7)
    expected = ParticleBP(7, 4, chunk_size=3).infer(data, support=support, return_history=True)
    bp = CachedParticleBP(7, 4, chunk_size=3, cache_bytes=cap)
    actual = bp.infer(data, support=support, return_history=True)
    compare_results(actual, expected)
    for first, second in zip(actual.history, expected.history):
        for name in ('mean', 'covariance', 'weights', 'log_messages'):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    stats = actual.diagnostics['experimental_likelihood_cache']
    assert stats['retained_payload_peak_bytes'] <= cap
    assert stats['logical_pair_distances'] == actual.diagnostics['pair_distance_evaluations']
    assert bp._kernel_cache is None and bp.last_cache_stats['released_after_infer']
    if kind != 'anchors' and cap == 100000:
        assert stats['computed_pair_distances']*4 == stats['logical_pair_distances']


def test_forest_also_matches_independent_full_joint_enumeration():
    data = graph_case('tree')
    support = explicit_support(data, 4)
    expected = exact_discrete(data, support.points, support.log_prior_weights)
    actual = CachedParticleBP(4, 10, chunk_size=3, cache_bytes=100000).infer(data, support=support)
    np.testing.assert_allclose(actual.weights.numpy(), expected.marginals, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(actual.covariance.numpy(), expected.covariance, rtol=3e-13, atol=2e-14)


def test_cache_does_not_reuse_changed_measurements_supports_or_masks():
    data = graph_case('cycle')
    support = explicit_support(data, 7)
    bp = CachedParticleBP(7, 5, chunk_size=3, cache_bytes=100000)
    for mode in ('original', 'measurements', 'support', 'mask', 'original'):
        changed, points = deepcopy(data), deepcopy(support)
        mask = torch.ones(data.num_edges)
        if mode == 'measurements':
            changed.edge_attr[:, 0] += .75
        elif mode == 'support':
            points.points[0, :, 0] += .25
        elif mode == 'mask':
            mask[0] = 0
        actual = bp.infer(changed, mask, support=points)
        expected = ParticleBP(7, 5, chunk_size=3).infer(changed, mask, support=points)
        compare_results(actual, expected)


def test_expanded_aliased_rows_do_not_confuse_edges_with_different_likelihoods():
    data = graph_case('cycle')
    points = explicit_support(data, 7)
    support = ParticleSupport(points.points[:1].expand(3, -1, -1),
                              points.log_prior_weights, points.agent_node_index)
    actual = CachedParticleBP(7, 4, chunk_size=3, cache_bytes=100000).infer(data, support=support)
    expected = ParticleBP(7, 4, chunk_size=3).infer(data, support=support)
    compare_results(actual, expected)


def test_log_domain_underflow_and_zero_prior_mass_remain_supported():
    data = make_graph([[0., 0.], [5., 0.]], [], [[0, 1]], [-100.], [.01])
    support = ParticleSupport(torch.tensor([[[0., 0.], [1., 0.], [2., 0.]],
                                            [[5., 0.], [6., 0.], [7., 0.]]], dtype=torch.float64),
                              torch.zeros(2, 3, dtype=torch.float64), torch.tensor([0, 1]))
    support.log_prior_weights[:, 0] = -torch.inf
    actual = CachedParticleBP(3, 4, chunk_size=2, cache_bytes=1000).infer(data, support=support)
    compare_results(actual, ParticleBP(3, 4, chunk_size=2).infer(data, support=support))
    assert torch.isfinite(actual.log_messages).all()
    assert (actual.weights[:, 0] == 0).all() and (actual.ess == 1).all()


def test_failure_releases_cache_and_instance_can_be_reused(monkeypatch):
    data = graph_case('chain')
    bp = CachedParticleBP(5, 3, chunk_size=2, cache_bytes=10000)
    original = torch.cdist
    calls = []
    def fail_after_one_block(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            assert bp.last_cache_stats['retained_payload_peak_bytes'] > 0
            raise RuntimeError('injected mid-inference failure')
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'cdist', fail_after_one_block)
        with pytest.raises(RuntimeError, match='mid-inference'):
            bp.infer(data, support=explicit_support(data))
    assert bp._kernel_cache is None and bp.last_cache_stats['status'] == 'failed'
    compare_results(bp.infer(data, support=explicit_support(data)),
                    ParticleBP(5, 3, chunk_size=2).infer(data, support=explicit_support(data)))


@pytest.mark.parametrize('options', [dict(damping=.3), dict(early_stopping=True)])
def test_existing_damping_and_early_stop_semantics_are_inherited(options):
    data = graph_case('chain')
    support = explicit_support(data)
    expected = ParticleBP(5, 30, chunk_size=3, **options).infer(data, support=support)
    actual = CachedParticleBP(5, 30, chunk_size=3, cache_bytes=10000, **options).infer(data, support=support)
    compare_results(actual, expected)


@pytest.mark.parametrize('cache', [-1, True, 3.5])
def test_invalid_cache_cap_is_rejected(cache):
    with pytest.raises(ValueError):
        CachedParticleBP(cache_bytes=cache)


def test_plan_has_no_data_or_device_side_effects(monkeypatch, tmp_path):
    def forbidden(*a, **k):
        raise AssertionError('Planning attempted execution')
    monkeypatch.setattr(bench, 'execution', forbidden)
    monkeypatch.setattr(bench, 'load_split', forbidden)
    result = bench.main(['--output', str(tmp_path/'no-output')])
    assert result['split'] == 'val' and not result['test_access']
    assert result['inference_units'] == 16
    assert not (tmp_path/'no-output').exists()


def test_cuda_benchmark_refuses_active_compute_process(monkeypatch):
    class Busy:
        stdout = str(123456789)+'\n'
    monkeypatch.setattr(bench.subprocess, 'run', lambda *a, **k: Busy())
    with pytest.raises(RuntimeError, match='active'):
        bench.require_idle_cuda()


def test_small_benchmark_budget_resume_and_identity_rejection(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg.to_dict()), encoding='utf-8')
    create_dataset('val', 2, settings=cfg, output_root=tmp_path/'data')
    args = ['--execute', '--device', 'cpu', '--config', str(config), '--data-root', str(tmp_path/'data'),
            '--graphs', '0', '--particles', '13', '--iterations', '3', '--reference-chunk', '5',
            '--variant', '5:1', '--variant', '7:0', '--output', str(tmp_path/'bench'), '--max-inferences', '3']
    first = bench.main(args)
    assert first['status'] == 'budget-reached' and len(first['records']) == 3
    original = deepcopy(first['records'])
    while first['status'] == 'budget-reached':
        first = bench.main([*args, '--resume'])
    assert first['status'] == 'complete' and len(first['records']) == 12
    assert first['identity']['dataset']['split'] == 'val'
    assert all(first['records'][key] == value for key, value in original.items())
    assert all(row['paired_cases'] == 2 for row in first['speed_summary']['rows'])
    repeated = bench.main([*args, '--resume'])
    assert repeated['records'] == first['records']
    with pytest.raises(ValueError, match='identical'):
        bench.main([*args, '--resume', '--bp-seed', '701'])
    bad = tmp_path/'bench'/first['references']['graph0-dense']['path']
    bad.write_bytes(b'changed')
    # Completed records are not re-executed, but a future comparison must verify
    # the stored reference before trusting it.
    state_path = tmp_path/'bench/progress.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    key = next(k for k in state['records'] if 'chunk7' in k)
    del state['records'][key]
    state_path.write_text(json.dumps(state), encoding='utf-8')
    with pytest.raises(ValueError, match='Reference'):
        bench.main([*args, '--resume'])
    assert json.loads(state_path.read_text(encoding='utf-8'))['status'] == 'failed'


def test_profile_run_exports_traces_but_never_claims_speedup(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    create_dataset('val', 1, settings=cfg, output_root=tmp_path/'data')
    result = bench.main(['--execute', '--profile', '--device', 'cpu',
                         '--config', 'configs/smoke-pipeline.json', '--data-root', str(tmp_path/'data'),
                         '--graphs', '0', '--particles', '7', '--iterations', '2', '--reference-chunk', '4',
                         '--variant', '4:1', '--repeats', '1', '--output', str(tmp_path/'profile')])
    assert result['status'] == 'complete'
    assert not result['speed_summary']['speed_claims']
    assert len(list((tmp_path/'profile').glob('*.trace.json'))) == 4
    assert len(list((tmp_path/'profile').glob('*.operators.txt'))) == 4
    assert all(row['profiling'] for row in result['records'].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime unavailable')
def test_cached_cuda_matches_original_and_cpu():
    data = graph_case('cycle')
    support = explicit_support(data, 31)
    cpu = ParticleBP(31, 5, chunk_size=11).infer(data, support=support)
    cuda_support = ParticleSupport(support.points.cuda(), support.log_prior_weights.cuda(),
                                   support.agent_node_index.cuda())
    original = ParticleBP(31, 5, chunk_size=11).infer(data.cuda(), support=cuda_support)
    actual = CachedParticleBP(31, 5, chunk_size=11, cache_bytes=100000).infer(data, support=cuda_support)
    compare_results(actual, original)
    for name in ('mean', 'covariance', 'weights', 'ess'):
        torch.testing.assert_close(getattr(actual, name).cpu(), getattr(cpu, name), rtol=1e-10, atol=1e-11)
