"""Stage 4: message semantics, equivariance, Bernoulli law and STE gradients."""
from copy import deepcopy
from dataclasses import replace
import io
import math

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from gib4cl.data.schema import message_edges
from gib4cl.config import ModelConfig, load_config, fingerprint
from gib4cl.learning.model import EdgePredictorGNN, GNNLayer, bernoulli_sigmoid, gumbel_sigmoid
from gib4cl.learning.objectives import gib_objective
from tests.test_objectives import make_graph, settings


@pytest.fixture(autouse=True, scope='module')
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def graph():
    return make_graph([[0., 0.], [3., 0.], [0., 4.]], [[8., 2.], [2., 9.]],
                      [[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 3]],
                      [3.4, 3.7, 8.1, 5.2, 8.9, 8.4], [1., 2., .5, 1., 3., .7])


@pytest.fixture
def model():
    torch.manual_seed(415)
    return EdgePredictorGNN(hidden_dim=12, num_layers=3)


def rebuild_messages(data):
    data.message_edge_index, data.message_edge_to_measurement, data.message_reverse_index = message_edges(data.edge_index)
    return data


def test_layer_matches_independent_neighbor_sum_and_update(graph):
    torch.manual_seed(7)
    layer = GNNLayer(5, 4, 8).double()
    x, e, edges = graph.x.double(), graph.edge_attr.double(), graph.edge_index
    # Reference uses unique physical edges, never the PyG propagation machinery.
    sums = [torch.zeros(8, dtype=torch.float64) for _ in range(x.shape[0])]
    for edge, (u, v) in enumerate(edges.t().tolist()):
        sums[u] = sums[u] + layer.msg_mlp(torch.cat([x[u], x[v], e[edge]]))
        sums[v] = sums[v] + layer.msg_mlp(torch.cat([x[v], x[u], e[edge]]))
    expected = torch.stack([layer.upd_mlp(torch.cat([x[i], total])) for i, total in enumerate(sums)])
    actual = layer(x, graph.message_edge_index, e[graph.message_edge_to_measurement])
    torch.testing.assert_close(actual, expected)


def test_anchor_and_agent_send_receive_and_isolated_node_stays_local():
    x = torch.tensor([[1., 2., 0., 0., 0.], [3., 4., 0., 0., 1.], [8., 9., 0., 0., 0.]])
    layer = GNNLayer(5, 4, 2)
    layer.msg_mlp = nn.Linear(14, 2, bias=False)
    layer.upd_mlp = nn.Linear(7, 2, bias=False)
    with torch.no_grad():
        layer.msg_mlp.weight.zero_()
        layer.msg_mlp.weight[:, 5:7] = torch.eye(2)  # sender coordinate
        layer.upd_mlp.weight.zero_()
        layer.upd_mlp.weight[:, 5:] = torch.eye(2)  # incoming sum only
    actual = layer(x, torch.tensor([[0, 1], [1, 0]]), torch.zeros(2, 4))
    torch.testing.assert_close(actual, torch.tensor([[3., 4.], [1., 2.], [0., 0.]]))


def test_architecture_uses_raw_features_and_no_external_residual(model, graph):
    assert not hasattr(model, 'node_encoder') and not hasattr(model, 'edge_encoder')
    assert model.layers[0].msg_mlp[0].in_features == 2 * 5 + 4
    assert model.layers[1].msg_mlp[0].in_features == 2 * 12 + 4
    assert model.edge_scorer[0].in_features == 2 * 12 + 4
    for layer in model.layers:
        for parameter in layer.parameters():
            nn.init.zeros_(parameter)
    torch.testing.assert_close(model.node_embeddings(graph), torch.zeros(graph.num_nodes, 12))


def test_score_is_average_of_two_shared_logits(model, graph):
    class OrderedScore(nn.Module):
        def forward(self, values):
            return (.7 * values[:, 0] + .2 * values[:, 12] + .1 * values[:, -4]).unsqueeze(1)
    model.edge_scorer = OrderedScore()
    hidden = model.node_embeddings(graph)
    u, v = graph.edge_index
    uv = model.edge_scorer(torch.cat([hidden[u], hidden[v], graph.edge_attr], dim=1)).flatten()
    vu = model.edge_scorer(torch.cat([hidden[v], hidden[u], graph.edge_attr], dim=1)).flatten()
    logits = model.score_edges(graph)
    torch.testing.assert_close(logits, .5 * (uv + vu))
    assert not torch.allclose(torch.sigmoid(logits), .5 * (torch.sigmoid(uv) + torch.sigmoid(vu)), atol=1e-8, rtol=1e-8)


def test_endpoint_reversal_invariance_including_anchor_edges(model, graph):
    changed = deepcopy(graph)
    changed.edge_index = changed.edge_index.flip(0)
    rebuild_messages(changed)
    torch.testing.assert_close(model.score_edges(changed), model.score_edges(graph))


def test_arbitrary_node_permutation_equivariance(model, graph):
    order = torch.tensor([4, 2, 0, 3, 1])  # includes anchors and agents
    inverse = torch.argsort(order)
    changed = deepcopy(graph)
    changed.x = graph.x[order]
    changed.edge_index = inverse[graph.edge_index]
    rebuild_messages(changed)
    torch.testing.assert_close(model.node_embeddings(changed), model.node_embeddings(graph)[order])
    torch.testing.assert_close(model.score_edges(changed), model.score_edges(graph))


def test_message_arc_order_invariance(model, graph):
    changed = deepcopy(graph)
    order = torch.tensor([10, 4, 2, 9, 0, 7, 1, 11, 6, 8, 5, 3])
    inverse = torch.argsort(order)
    changed.message_edge_index = graph.message_edge_index[:, order]
    changed.message_edge_to_measurement = graph.message_edge_to_measurement[order]
    changed.message_reverse_index = inverse[graph.message_reverse_index[order]]
    torch.testing.assert_close(model.score_edges(changed), model.score_edges(graph))


def test_physical_edge_order_and_paired_sampling_equivariance(model, graph):
    order = torch.tensor([3, 0, 5, 2, 1, 4])
    changed = deepcopy(graph)
    changed.edge_index = graph.edge_index[:, order]
    changed.edge_attr = graph.edge_attr[order]
    rebuild_messages(changed)
    uniform = torch.tensor([.1, .2, .4, .6, .8, .9])
    weights, logits = model(graph, uniform=uniform)
    other_weights, other_logits = model(changed, uniform=uniform[order])
    torch.testing.assert_close(other_logits, logits[order])
    torch.testing.assert_close(other_weights, weights[order], rtol=0, atol=0)
    assert weights.shape == (graph.num_edges,)  # not 2E


def test_batch_matches_individual_graphs_with_empty_edges(model, graph):
    empty = make_graph([[2., 3.], [6., 7.]], [], [], [], [])
    batch = Batch.from_data_list([graph, empty, graph])
    model.eval()
    actual, logits = model(batch)
    expected = torch.cat([model.score_edges(item) for item in (graph, empty, graph)])
    torch.testing.assert_close(logits, expected)
    assert actual.numel() == 2 * graph.num_edges
    assert model(empty)[0].shape == (0,)
    assert model.node_embeddings(empty).shape == (2, 12)


def test_labels_and_diagnostics_never_affect_scoring(model, graph):
    reference = model.score_edges(graph)
    changed = deepcopy(graph)
    changed.y = torch.full_like(graph.y, float('nan'))
    changed.diagnostic_truth = torch.ones(graph.num_edges, dtype=torch.bool)
    changed.true_positions = torch.randn(graph.num_nodes, 2) * 999
    torch.testing.assert_close(model.score_edges(changed), reference)
    del changed.y
    torch.testing.assert_close(model.score_edges(changed), reference)


@pytest.mark.parametrize('corruption', ['missing', 'one_way', 'wrong_map', 'wrong_reverse', 'duplicate', 'nonfinite'])
def test_invalid_graph_inputs_fail_clearly(model, graph, corruption):
    changed = deepcopy(graph)
    if corruption == 'missing':
        del changed.message_edge_index
    elif corruption == 'one_way':
        changed.message_edge_index = graph.edge_index.clone()
    elif corruption == 'wrong_map':
        changed.message_edge_to_measurement[0] = 1
    elif corruption == 'wrong_reverse':
        changed.message_reverse_index[0] = 0
    elif corruption == 'duplicate':
        changed.edge_index[:, 1] = changed.edge_index[:, 0]
    else:
        changed.edge_attr[0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(changed)


@pytest.mark.parametrize('tau', [.1, 1., 5.])
def test_empirical_bernoulli_frequencies_at_multiple_temperatures(tau):
    n = 80000
    probabilities = torch.tensor([.01, .1, .5, .9, .99], dtype=torch.float64)
    logits = torch.logit(probabilities).repeat(n, 1)
    sample = bernoulli_sigmoid(logits, tau=tau, generator=torch.Generator().manual_seed(1001))
    frequencies = sample.mean(0)
    tolerance = 6 * torch.sqrt(probabilities * (1 - probabilities) / n) + 1 / n
    assert (torch.abs(frequencies - probabilities) < tolerance).all()
    assert ((sample == 0) | (sample == 1)).all()


def test_independent_edge_draws_match_joint_bernoulli_law():
    n = 120000
    p = torch.tensor([.3, .7], dtype=torch.float64)
    sample = bernoulli_sigmoid(torch.logit(p).repeat(n, 1),
                               generator=torch.Generator().manual_seed(203))
    for left in (0, 1):
        for right in (0, 1):
            expected = (p[0] if left else 1 - p[0]) * (p[1] if right else 1 - p[1])
            observed = ((sample[:, 0] == left) & (sample[:, 1] == right)).double().mean()
            assert abs(observed - expected) < 6 * math.sqrt(expected * (1 - expected) / n)


@pytest.mark.parametrize('tau', [.1, 1., 5.])
def test_ste_forward_binary_and_backward_matches_analytic_soft_derivative(tau):
    logits = torch.tensor([-1.3, -.2, .4, 2.], dtype=torch.float64, requires_grad=True)
    uniform = torch.tensor([.2, .7, .45, .9], dtype=torch.float64)
    perturbed = logits.detach() + torch.logit(uniform)
    soft = torch.sigmoid(perturbed / tau)
    weights = bernoulli_sigmoid(logits, tau=tau, uniform=uniform)
    torch.testing.assert_close(weights, (perturbed > 0).double(), rtol=0, atol=0)
    weights.sum().backward()
    torch.testing.assert_close(logits.grad, soft * (1 - soft) / tau)
    torch.testing.assert_close(bernoulli_sigmoid(logits, tau=tau, uniform=uniform, hard=False), soft)


def test_hard_samples_do_not_depend_on_temperature_but_gradients_do():
    logits = torch.tensor([-.3, .1, 1.], dtype=torch.float64, requires_grad=True)
    uniform = torch.tensor([.4, .6, .3], dtype=torch.float64)
    low = bernoulli_sigmoid(logits, tau=.1, uniform=uniform)
    high = bernoulli_sigmoid(logits, tau=5., uniform=uniform)
    torch.testing.assert_close(low, high, rtol=0, atol=0)
    assert not torch.allclose(torch.autograd.grad(low.sum(), logits)[0],
                              torch.autograd.grad(high.sum(), logits)[0])


def test_eval_probabilities_threshold_ties_and_no_rng_consumption(model, graph):
    logits = torch.tensor([-1., 0., 1e-8, 2.])
    rng = torch.random.get_rng_state().clone()
    for tau in (.1, 1., 5.):
        torch.testing.assert_close(bernoulli_sigmoid(logits, tau, training=False, hard=False), logits.sigmoid())
        torch.testing.assert_close(bernoulli_sigmoid(logits, tau, training=False), torch.tensor([0., 0., 1., 1.]))
    model.eval()
    probabilities, scores = model(graph, hard=False)
    torch.testing.assert_close(probabilities, scores.sigmoid())
    torch.testing.assert_close(model.edge_probabilities(graph), probabilities)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_generator_reproducibility_isolated_from_global_rng_and_alias():
    logits = torch.linspace(-3, 3, 1000)
    rng = torch.random.get_rng_state().clone()
    a = bernoulli_sigmoid(logits, generator=torch.Generator().manual_seed(43))
    b = gumbel_sigmoid(logits, generator=torch.Generator().manual_seed(43))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_uniform_zero_draw_is_resampled_not_logged(monkeypatch):
    calls = []
    def draws(shape, **kwargs):
        calls.append(shape)
        return torch.tensor([0., .25]) if len(calls) == 1 else torch.tensor([.75])
    monkeypatch.setattr(torch, 'rand', draws)
    actual = bernoulli_sigmoid(torch.zeros(2))
    torch.testing.assert_close(actual, torch.tensor([1., 0.]))
    assert len(calls) == 2


@pytest.mark.parametrize('tau', [0., -1., float('nan'), float('inf'), True, 1e-50])
def test_invalid_temperatures_rejected(tau):
    with pytest.raises(ValueError):
        bernoulli_sigmoid(torch.zeros(1), tau=tau)


@pytest.mark.parametrize('uniform', [torch.tensor([0.]), torch.tensor([1.]),
                                   torch.tensor([float('nan')]), torch.tensor([.2, .3]),
                                   torch.tensor([.5], dtype=torch.float64)])
def test_invalid_supplied_noise_rejected(uniform):
    with pytest.raises(ValueError):
        bernoulli_sigmoid(torch.zeros(1), uniform=uniform)


def test_ambiguous_noise_and_nonfinite_logits_rejected():
    with pytest.raises(ValueError):
        bernoulli_sigmoid(torch.zeros(1), uniform=torch.tensor([.5]), generator=torch.Generator())
    with pytest.raises(ValueError):
        bernoulli_sigmoid(torch.zeros(1), training=False, uniform=torch.tensor([.5]))
    with pytest.raises(ValueError):
        bernoulli_sigmoid(torch.tensor([float('inf')]))
    with pytest.raises(TypeError):
        bernoulli_sigmoid(torch.tensor([0]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_extreme_logits_and_empty_tensors_have_finite_gradients(dtype):
    values = torch.tensor([-1e30, -1000., 0., 1000., 1e30], dtype=dtype, requires_grad=True)
    actual = bernoulli_sigmoid(values, tau=1e-6, uniform=torch.full_like(values, .5))
    actual.sum().backward()
    assert torch.isfinite(actual).all() and torch.isfinite(values.grad).all()
    assert values.grad[2] > 0
    empty = torch.empty(0, dtype=dtype, requires_grad=True)
    bernoulli_sigmoid(empty).sum().backward()
    assert empty.grad.shape == (0,)


def test_actual_model_and_objective_propagate_to_all_layers(model, graph):
    data = Batch.from_data_list([graph])
    weights, logits = model(data, tau=2., uniform=torch.linspace(.15, .85, graph.num_edges))
    result = gib_objective(logits, weights, data, objective=settings())
    result.loss.backward()
    assert torch.isfinite(result.loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for layer in model.layers:
        assert sum(parameter.grad.abs().sum() for parameter in layer.parameters()) > 0
    assert sum(parameter.grad.abs().sum() for parameter in model.edge_scorer.parameters()) > 0


def test_parameter_state_roundtrip_and_structure_checks(model, graph):
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    restored = EdgePredictorGNN(hidden_dim=12, num_layers=3)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.score_edges(graph), model.score_edges(graph), rtol=0, atol=0)
    assert '_extra_state' not in state
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    first = next(iter(state))
    missing = {name: value for name, value in state.items() if name != first}
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(missing)
    wrong_shape = dict(state)
    wrong_shape[first] = state[first][:-1]
    with pytest.raises(RuntimeError, match='size mismatch'):
        restored.load_state_dict(wrong_shape)


def test_model_config_has_no_version_and_dataset_identity_is_unchanged():
    cfg = load_config('configs/smoke-pipeline.json')
    assert 'architecture_version' not in cfg.to_dict()['model']
    changed = replace(cfg, model=replace(cfg.model, hidden_dim=32))
    assert cfg.data_fingerprint == changed.data_fingerprint
    assert fingerprint(cfg.to_dict()) != fingerprint(changed.to_dict())
    with pytest.raises(TypeError):
        ModelConfig(architecture_version='old')
