"""Selection-tool contracts: no optimization, common noise, nested supports and resume."""
import json
import math
import pytest
import torch
from gib4cl.config import REPO_ROOT, load_config
from gib4cl.data.generate import build_dataset
from gib4cl.data.dataset import graph_to_data
from gib4cl.inference import ParticleBP, ParticleSupport
from gib4cl.learning.checkpoints import make_model
from tools import selection_diagnostics as diagnostic


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample():
    cfg = load_config(REPO_ROOT/'configs/smoke-pipeline.json')
    graph = graph_to_data(build_dataset(cfg, 'train', 1)['graphs'][0])
    return cfg, graph


def test_nested_support_is_per_node_prefix_and_reweights_prior_mass():
    _, data = sample()
    support = ParticleBP(16, 0).draw_support(data, generator=torch.Generator().manual_seed(99))
    prefix = diagnostic.nested_support(support, 8)
    assert torch.equal(prefix.points, support.points[:, :8])
    torch.testing.assert_close(prefix.log_prior_weights.exp().sum(1), torch.ones(len(prefix.points), dtype=torch.float64))
    assert torch.equal(prefix.log_prior_weights, torch.full_like(prefix.log_prior_weights, -math.log(8)))
    wrong = ParticleSupport(support.points, support.log_prior_weights + torch.arange(16), support.agent_node_index)
    with pytest.raises(ValueError, match='uniform'):
        diagnostic.nested_support(wrong, 8)


def test_zero_gradient_ratios_remain_unidentified():
    assert diagnostic.safe_ratio(1., 0.) is None
    assert diagnostic.safe_ratio(0., 0.) is None
    assert diagnostic.safe_ratio(6., 2.) == 3.


def test_scale_probe_does_not_update_parameters_or_accumulate_gradients():
    cfg, data = sample()
    torch.manual_seed(22)
    model = make_model(cfg.model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    row = diagnostic.scale_probe(model, data, tau=1., offset=0., seed=77)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert all(math.isfinite(v) for v in row['gradient_norms'].values())
    assert row['unavoidable_degree_deficit'] >= 0


def test_validation_common_noise_prefix_eval_mode_and_rng_isolation(monkeypatch):
    cfg, data = sample()
    model = make_model(cfg.model).train()
    original = model.score_edges
    modes = []
    def score(graph):
        modes.append(model.training)
        return original(graph)
    monkeypatch.setattr(model, 'score_edges', score)
    rng = torch.random.get_rng_state().clone()
    a = diagnostic.validation_probe(model, data, cfg.objective, repeats=4, seed=80)
    b = diagnostic.validation_probe(model, data, cfg.objective, repeats=8, seed=80)
    assert a['random_totals'] == b['random_totals'][:4]
    assert a['deterministic_total'] == b['deterministic_total']
    assert model.training and modes == [False, False]
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_sparse_masks_are_nested_and_repeatable():
    _, data = sample()
    a, b = diagnostic.fixed_masks(data), diagnostic.fixed_masks(data)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert (a['sparse25'] <= a['sparse50']).all()
    assert (a['sparse50'] <= a['sparse75']).all()


def test_plan_does_not_generate_data_and_never_uses_test(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A plan must not generate graphs')
    monkeypatch.setattr(diagnostic, 'build_dataset', forbidden)
    scale = diagnostic.main(['scale', '--plan'])
    bp = diagnostic.main(['bp', '--plan'])
    assert scale['split'] == 'train' and scale['trials'] == 864
    assert bp['split'] == 'val' and bp['trials'] == 120
    assert not scale['test_access'] and not bp['test_access']


def test_bp_completed_trials_resume_without_inference(tmp_path, monkeypatch):
    output = tmp_path/'bp'
    args = ['bp', '--config', str(REPO_ROOT/'configs/smoke-pipeline.json'),
            '--graphs', '1', '--seeds', '700', '--particles', '8', '16',
            '--iterations', '0', '1', '--chunk-sizes', '4', '--output-root', str(output)]
    first = diagnostic.main(args)
    assert first['trials'] == first['ok'] == 20
    records = [json.loads(p.read_text()) for p in (output/'trials').glob('*.json')]
    for p in (8, 16):
        assert len({r['support_digest'] for r in records if r['key']['particles'] == p}) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed inference must not be repeated')
    monkeypatch.setattr(ParticleBP, 'infer', forbidden)
    # A truncated append log must not damage the authoritative atomic trial records.
    with (output/'rows.jsonl').open('a') as f:
        f.write('{unfinished')
    second = diagnostic.main([*args, '--resume'])
    assert second == first
    assert json.loads((output/'manifest.json').read_text())['status'] == 'complete'


def test_resume_rejects_changed_grid(tmp_path):
    output = tmp_path/'scale'
    args = ['scale', '--config', str(REPO_ROOT/'configs/smoke-pipeline.json'), '--graphs', '1',
            '--seeds', '101', '--temperatures', '1', '--offsets', '0', '--output-root', str(output)]
    diagnostic.main(args)
    with pytest.raises(ValueError, match='identical'):
        diagnostic.main([*args, '--resume', '--offsets', '1'])
