"""Independent algebra, finite-state KL, and differentiability checks for stage 3."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import math

import pytest
import torch
from torch_geometric.data import Batch

from gib4cl.data.schema import GRAPH_VERSION
from gib4cl.data.dataset import graph_to_data
from gib4cl.config import ObjectiveConfig, load_config
from gib4cl.learning.objectives import compute_gib_loss
from gib4cl.learning.objectives import (
    compression_terms, degree_terms, gaussian_predictions, gaussian_task_terms, gib_objective,
)


@pytest.fixture(autouse=True, scope='module')
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_graph(agents, anchors, edges, measurements, variances, covariances=None, targets=None):
    a, k = len(agents), len(anchors)
    if covariances is None:
        covariances = torch.eye(2).repeat(a, 1, 1) * 4
    inputs = {
        'initial_positions': torch.tensor(agents + anchors, dtype=torch.float32),
        'initial_covariances': torch.as_tensor(covariances, dtype=torch.float32),
        'is_anchor': torch.tensor([False] * a + [True] * k),
        'edge_index': torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
        'measurements': torch.tensor(measurements, dtype=torch.float32),
        'measurement_variances': torch.tensor(variances, dtype=torch.float32),
        'communication_cost_scale': 1.,
        'measurement_encoding': 'ieee754-binary32-canonical-finite',
    }
    record = {'schema_version': GRAPH_VERSION, 'sample_id': 'objective-hand-graph', 'inputs': inputs,
              'targets': {'agent_positions': torch.tensor(targets if targets is not None else agents, dtype=torch.float32)}}
    return graph_to_data(record)


@pytest.fixture
def anchor_graph():
    return make_graph([[3., 4.]], [[0., 0.]], [[0, 1]], [7.], [2.],
                      [[[4., 0.], [0., 9.]]], [[3.2, 4.9]])


@pytest.fixture
def triangle_graph():
    return make_graph([[0., 0.], [3., 0.], [0., 4.]], [],
                      [[0, 1], [0, 2], [1, 2]], [3.4, 3.7, 5.2], [1., 2., .5],
                      targets=[[.2, -.1], [3.3, .2], [-.1, 4.1]])


def settings(**overrides):
    return replace(ObjectiveConfig(beta=.02, gamma=.002, kappa=.2, eta=1.), **overrides)


def test_single_anchor_hand_precision_information_mean_covariance(anchor_graph):
    pred = gaussian_predictions(anchor_graph, torch.ones(1))[0]
    expected_j = torch.tensor([[.25 + .18, .24], [.24, 1 / 9 + .32]], dtype=torch.float64)
    expected_b = torch.tensor([.6, .8], dtype=torch.float64)
    expected_delta = torch.tensor([12 / 23, 36 / 23], dtype=torch.float64)
    sigma = torch.diag(torch.tensor([4., 9.], dtype=torch.float64))
    su = torch.tensor([2.4, 7.2], dtype=torch.float64)
    expected_cov = sigma - torch.outer(su, su) / 9.2
    torch.testing.assert_close(pred.gradient, torch.tensor([[.6, .8]], dtype=torch.float64))
    torch.testing.assert_close(pred.precision, expected_j)
    torch.testing.assert_close(pred.information_vector, expected_b)
    torch.testing.assert_close(pred.delta, expected_delta)
    torch.testing.assert_close(pred.mean, torch.tensor([[3., 4.]], dtype=torch.float64) + expected_delta)
    torch.testing.assert_close(pred.covariance, expected_cov)
    torch.testing.assert_close(pred.logdet_precision, torch.linalg.slogdet(expected_j).logabsdet)
    assert pred.precision.dtype == torch.float64


def test_full_nll_matches_independent_gaussian_distribution(anchor_graph):
    predictions = gaussian_predictions(anchor_graph, torch.ones(1))
    pred = predictions[0]
    terms = gaussian_task_terms(predictions, anchor_graph.y)
    reference = torch.distributions.MultivariateNormal(pred.mean.reshape(-1), covariance_matrix=pred.covariance)
    expected = -reference.log_prob(anchor_graph.y.double().reshape(-1))
    torch.testing.assert_close(terms['task_nll'][0], expected)
    torch.testing.assert_close(terms['task_nll'] - terms['task'], torch.tensor([math.log(2 * math.pi)], dtype=torch.float64))
    torch.testing.assert_close(terms['task'], terms['task_logdet'] + terms['task_mahalanobis'])


def test_agent_agent_gradient_and_negative_cross_blocks():
    data = make_graph([[0., 0.], [3., 0.]], [], [[0, 1]], [4.], [2.],
                      [[[4., 0.], [0., 9.]], [[16., 0.], [0., 25.]]])
    pred = gaussian_predictions(data, torch.ones(1))[0]
    h = torch.tensor([[-1., 0., 1., 0.]], dtype=torch.float64)
    prior = torch.diag(torch.tensor([.25, 1 / 9, 1 / 16, 1 / 25], dtype=torch.float64))
    torch.testing.assert_close(pred.gradient, h)
    torch.testing.assert_close(pred.precision, prior + .5 * h.T @ h)
    assert pred.precision[0, 2] == -.5
    assert pred.information_vector[0] < 0 and pred.information_vector[2] > 0


def test_signed_residual_moves_mean_towards_the_correct_direction():
    data = make_graph([[3., 4.]], [[10., 4.]], [[0, 1]], [9.], [1.])
    pred = gaussian_predictions(data, torch.ones(1))[0]
    assert pred.mean[0, 0] < 3. and pred.mean[0, 1] == 4.


def test_full_correlated_prior_and_no_selected_edges():
    cov = torch.tensor([[[4., 1.], [1., 3.]], [[9., -2.], [-2., 16.]]])
    data = make_graph([[0., 0.], [3., 1.]], [[0., 5.]], [[0, 1], [0, 2]], [3., 5.], [1., 2.], cov)
    # A stale single-precision cache must not replace the physical covariance.
    data.agent_prior_precision.fill_(999.)
    pred = gaussian_predictions(data, torch.zeros(2))[0]
    torch.testing.assert_close(pred.mean, data.x[:2, :2].double())
    torch.testing.assert_close(pred.delta, torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(pred.covariance, torch.block_diag(*cov.double().unbind(0)))


def test_prediction_is_label_free_and_unselected_measurements_do_not_change_forward(anchor_graph):
    inference = anchor_graph.clone()
    del inference.y
    before = gaussian_predictions(inference, torch.zeros(1))[0]
    inference.edge_attr[:, :2] = torch.tensor([[-100., 300.]])
    inference.signed_residual.fill_(999.)
    after = gaussian_predictions(inference, torch.zeros(1))[0]
    torch.testing.assert_close(before.mean, after.mean)
    torch.testing.assert_close(before.precision, after.precision)
    with pytest.raises(ValueError, match='targets'):
        gib_objective(torch.zeros(1), torch.zeros(1), inference, objective=settings())
    first = gib_objective(torch.zeros(1), torch.ones(1), anchor_graph, objective=settings())
    changed = anchor_graph.clone()
    changed.y += 100.
    second = gib_objective(torch.zeros(1), torch.ones(1), changed, objective=settings())
    torch.testing.assert_close(first.predictions[0].mean, second.predictions[0].mean)
    torch.testing.assert_close(first.per_graph['compression'], second.per_graph['compression'])
    assert first.loss != second.loss


def test_variance_floor_is_max_not_additive_jitter(anchor_graph):
    anchor_graph.edge_attr[:, 1] = 1e-9
    pred = gaussian_predictions(anchor_graph, torch.ones(1), variance_floor=.25)[0]
    h = torch.tensor([[.6, .8]], dtype=torch.float64)
    expected = torch.diag(torch.tensor([.25, 1 / 9], dtype=torch.float64)) + h.T @ h / .25
    torch.testing.assert_close(pred.precision, expected)
    assert anchor_graph.edge_attr[0, 1] < 1e-8


def test_small_graph_precision_matches_finite_difference_range_jacobian(triangle_graph):
    pred = gaussian_predictions(triangle_graph, torch.tensor([1., 0., 1.]))[0]
    origin = triangle_graph.x[:, :2].double().reshape(-1)
    edges = triangle_graph.edge_index
    eps = 1e-6
    columns = []
    for coordinate in range(origin.numel()):
        plus, minus = origin.clone(), origin.clone()
        plus[coordinate] += eps
        minus[coordinate] -= eps
        def ranges(y):
            points = y.reshape(-1, 2)
            return (points[edges[0]] - points[edges[1]]).norm(dim=1)
        columns.append((ranges(plus) - ranges(minus)) / (2 * eps))
    torch.testing.assert_close(pred.gradient, torch.stack(columns, dim=1), rtol=1e-8, atol=1e-8)


def test_compression_matches_exact_eight_mask_kl_and_gradient(triangle_graph):
    triangle_graph.edge_cost = torch.tensor([.4, .7, 1.1], dtype=torch.float64)
    triangle_graph.edge_log_cardinality = torch.tensor([math.log(2), math.log(3), math.log(5)], dtype=torch.float64)
    logits = torch.logit(torch.tensor([.2, .6, .8], dtype=torch.float64)).requires_grad_()
    gamma, kappa = .3, .7
    terms = compression_terms(logits, triangle_graph, gamma=gamma, kappa=kappa)
    masks = torch.tensor(list(product((0., 1.), repeat=3)), dtype=torch.float64)
    log_p = (masks * torch.nn.functional.logsigmoid(logits)
             + (1 - masks) * torch.nn.functional.logsigmoid(-logits)).sum(dim=1)
    energy = masks @ triangle_graph.edge_cost
    cycles = masks.prod(dim=1)
    log_z = torch.logsumexp(-gamma * energy - kappa * cycles, dim=0)
    log_q_r = -gamma * energy - kappa * cycles - log_z - masks @ triangle_graph.edge_log_cardinality
    exact_kl = (log_p.exp() * (log_p - log_q_r)).sum()
    torch.testing.assert_close(terms['compression'][0] + log_z, exact_kl)
    analytic_grad = torch.autograd.grad(terms['compression'].sum(), logits, retain_graph=True)[0]
    exact_grad = torch.autograd.grad(exact_kl, logits)[0]
    torch.testing.assert_close(analytic_grad, exact_grad)
    torch.testing.assert_close(terms['expected_triangles'], torch.tensor([.096], dtype=torch.float64))
    torch.testing.assert_close(terms['compression'], terms['negative_entropy'] + gamma * terms['expected_energy']
                               + kappa * terms['expected_triangles'] + terms['retained_data_cost'])


def test_removed_partition_constant_does_not_make_compression_nonnegative(triangle_graph):
    triangle_graph.edge_cost.zero_()
    triangle_graph.edge_log_cardinality.zero_()  # Toy alphabets with M=1.
    terms = compression_terms(torch.zeros(3), triangle_graph, gamma=0., kappa=0.)
    torch.testing.assert_close(terms['compression'], torch.tensor([-3 * math.log(2)], dtype=torch.float64))


def test_entropy_extreme_logits_are_finite(triangle_graph):
    logits = torch.tensor([-1000., 0., 1000.], dtype=torch.float64, requires_grad=True)
    terms = compression_terms(logits, triangle_graph, gamma=.1, kappa=.2)
    torch.testing.assert_close(terms['negative_entropy'], torch.tensor([-math.log(2)], dtype=torch.float64))
    assert torch.isfinite(torch.autograd.grad(terms['compression'].sum(), logits)[0]).all()


def test_degree_counts_anchor_edges_and_preserves_ste_gradient(anchor_graph):
    logit = torch.tensor([-1.], dtype=torch.float64, requires_grad=True)
    soft = logit.sigmoid()
    mask = torch.zeros_like(soft) - soft.detach() + soft
    terms = degree_terms(mask, anchor_graph)
    assert terms['degree'][0] == 3.
    gradient = torch.autograd.grad(terms['degree'].sum(), logit)[0]
    torch.testing.assert_close(gradient, -soft.detach() * (1 - soft.detach()))
    assert degree_terms(torch.ones(1), anchor_graph)['degree'][0] == 2.


def test_zero_forward_edge_still_has_task_surrogate_gradient(anchor_graph):
    weights = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    pred = gaussian_predictions(anchor_graph, weights)[0]
    derivative = torch.autograd.grad(pred.mean[0, 0], weights)[0]
    assert derivative.abs().item() > .1


def test_smooth_task_and_covariance_pass_gradcheck(anchor_graph):
    weights = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
    def outputs(w):
        predictions = gaussian_predictions(anchor_graph, w, strict_binary=False)
        p = predictions[0]
        return (p.precision, p.information_vector, p.mean, p.covariance,
                gaussian_task_terms(predictions, anchor_graph.y)['task'])
    assert torch.autograd.gradcheck(outputs, (weights,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_compression_passes_gradcheck(triangle_graph):
    logits = torch.tensor([-.7, .2, 1.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda l: compression_terms(l, triangle_graph, gamma=.03, kappa=.4)['compression'],
        (logits,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_mean_solve_gradient_is_not_detached_from_task_error(anchor_graph):
    w = torch.tensor([.5], dtype=torch.float64, requires_grad=True)
    predictions = gaussian_predictions(anchor_graph, w, strict_binary=False)
    p = predictions[0]
    task = gaussian_task_terms(predictions, anchor_graph.y)['task'].sum()
    actual = torch.autograd.grad(task, w, retain_graph=True)[0]
    error_without_mean_gradient = (anchor_graph.y.double() - p.mean.detach()).reshape(-1)
    incomplete = -.5 * p.logdet_precision + .5 * error_without_mean_gradient @ p.precision @ error_without_mean_gradient
    wrong = torch.autograd.grad(incomplete, w)[0]
    assert (actual - wrong).abs().item() > .01


def test_variable_size_batch_is_equal_graph_average(anchor_graph, triangle_graph):
    empty = make_graph([[0., 0.], [2., 2.]], [], [], [], [], targets=[[1., 0.], [2., 1.]])
    graphs = [anchor_graph, empty, triangle_graph]
    logits = [torch.tensor([.2]), torch.empty(0), torch.tensor([-.4, .8, 1.2])]
    masks = [torch.ones(1), torch.empty(0), torch.tensor([1., 0., 1.])]
    singles = [gib_objective(l, w, g, objective=settings()) for l, w, g in zip(logits, masks, graphs)]
    combined = gib_objective(torch.cat(logits), torch.cat(masks), Batch.from_data_list(graphs), objective=settings())
    for key in combined.per_graph:
        torch.testing.assert_close(combined.per_graph[key], torch.cat([r.per_graph[key] for r in singles]))
    torch.testing.assert_close(combined.loss, torch.stack([r.loss for r in singles]).mean())
    torch.testing.assert_close(combined.loss, (combined.per_graph['task'] + .02 * combined.per_graph['compression']
                                             + combined.per_graph['degree']).mean())


def test_empty_graph_has_prior_task_zero_compression_and_degree_cost():
    data = make_graph([[0., 0.], [2., 1.]], [], [], [], [], targets=[[1., 1.], [3., 2.]])
    result = gib_objective(torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True), data, objective=settings())
    assert result.per_graph['compression'][0] == 0 and result.per_graph['degree'][0] == 6.
    assert torch.isfinite(result.loss)
    result.loss.backward()


def test_total_ste_chain_rule_and_logging(anchor_graph):
    logit = torch.tensor([-.3], dtype=torch.float64, requires_grad=True)
    soft = ((logit + .7) / .8).sigmoid()
    mask = (soft > .5).double() - soft.detach() + soft
    result = gib_objective(logit, mask, anchor_graph, objective=settings())
    actual = torch.autograd.grad(result.loss, logit)[0]
    independent_logit = logit.detach().clone().requires_grad_()
    independent_mask = mask.detach().clone().requires_grad_()
    independent = gib_objective(independent_logit, independent_mask, anchor_graph, objective=settings())
    direct, through_mask = torch.autograd.grad(independent.loss, (independent_logit, independent_mask))
    expected = direct + through_mask * soft.detach() * (1 - soft.detach()) / .8
    torch.testing.assert_close(actual, expected)
    loss, metrics = compute_gib_loss(logit, mask, anchor_graph, objective=settings())
    assert loss.requires_grad and all(isinstance(v, float) for v in metrics.values())
    assert 'fim' not in metrics and 'kl' not in metrics


@pytest.mark.parametrize('kind', ['soft_mask', 'missing_weights', 'nan_logits', 'negative_variance', 'non_spd_prior', 'duplicate_triangle'])
def test_invalid_objective_inputs_are_rejected(triangle_graph, kind):
    logits, weights, cfg = torch.zeros(3), torch.ones(3), settings()
    if kind == 'soft_mask':
        weights[0] = .5
    elif kind == 'missing_weights':
        cfg = ObjectiveConfig()
    elif kind == 'nan_logits':
        logits[0] = float('nan')
    elif kind == 'negative_variance':
        triangle_graph.edge_attr[0, 1] = -1.
    elif kind == 'non_spd_prior':
        triangle_graph.agent_prior_cov[0, 0, 0] = -1.
    else:
        triangle_graph.triangle_edge_index = triangle_graph.triangle_edge_index.repeat(1, 2)
    with pytest.raises(ValueError):
        gib_objective(logits, weights, triangle_graph, objective=cfg)


def test_anchor_triangle_is_not_a_structural_feedback_triangle():
    data = make_graph([[0., 0.], [3., 0.]], [[0., 4.]],
                      [[0, 1], [0, 2], [1, 2]], [3., 4., 5.], [1., 1., 1.])
    terms = compression_terms(torch.zeros(3), data, gamma=.1, kappa=.3)
    assert terms['expected_triangles'][0] == 0
    data.triangle_edge_index = torch.tensor([[0], [1], [2]])
    with pytest.raises(ValueError, match='unknown agents'):
        compression_terms(torch.zeros(3), data, gamma=.1, kappa=.3)


def test_legacy_keywords_are_not_silently_reinterpreted(anchor_graph):
    with pytest.raises(TypeError):
        compute_gib_loss(torch.zeros(1), torch.ones(1), anchor_graph, lambda_reg=1., prior_weight=.5)
    assert load_config('configs/smoke-pipeline.json').objective.require_ready().beta == .02
    with pytest.raises(ValueError, match='explicit'):
        load_config().objective.require_ready()


def test_node_permutation_preserves_objective_and_permuted_covariance(triangle_graph):
    original = gib_objective(torch.tensor([-.3, .6, .4]), torch.tensor([1., 0., 1.]),
                             triangle_graph, objective=settings())
    perm = torch.tensor([2, 0, 1])
    inverse = torch.argsort(perm)
    changed = triangle_graph.clone()
    changed.x = changed.x[perm]
    changed.y = changed.y[perm]
    changed.agent_prior_cov = changed.agent_prior_cov[perm]
    changed.edge_index = inverse[changed.edge_index].sort(dim=0).values
    changed_result = gib_objective(torch.tensor([-.3, .6, .4]), torch.tensor([1., 0., 1.]),
                                   changed, objective=settings())
    torch.testing.assert_close(changed_result.loss, original.loss)
    torch.testing.assert_close(changed_result.predictions[0].mean, original.predictions[0].mean[perm])
    coordinates = (perm[:, None] * 2 + torch.arange(2)).reshape(-1)
    torch.testing.assert_close(changed_result.predictions[0].covariance,
                               original.predictions[0].covariance[coordinates][:, coordinates])


def test_main_training_config_requires_explicit_fixed_weights():
    from gib4cl.learning.train import resolve_training_config
    base = load_config('configs/smoke-pipeline.json')
    resolved = resolve_training_config(settings=base, beta=.04, epochs=2)
    assert resolved.objective.beta == .04 and resolved.training.epochs == 2
    assert resolved.data_fingerprint == base.data_fingerprint
    with pytest.raises(ValueError, match='explicit'):
        resolve_training_config()
    with pytest.raises(ValueError):
        resolve_training_config(settings=base, kappa=-1.)


def test_main_epoch_weights_batches_by_graph_count_and_updates_parameters(anchor_graph, triangle_graph):
    from gib4cl.learning.train import objective_epoch
    from torch_geometric.loader import DataLoader
    class DummySelector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score = torch.nn.Parameter(torch.tensor(.2))

        def forward(self, data, tau=1., hard=True):
            logits = self.score.expand(data.edge_index.shape[1])
            soft = (logits / tau).sigmoid()
            return (soft > .5).float() - soft.detach() + soft, logits

    empty = make_graph([[0., 0.], [2., 2.]], [], [], [], [], targets=[[1., 0.], [3., 1.]])
    graphs = [anchor_graph, triangle_graph, empty]
    model = DummySelector()
    reference = []
    for graph in graphs:
        weights, logits = model(graph)
        reference.append(gib_objective(logits, weights, graph, objective=settings()).metrics())
    observed = objective_epoch(model, DataLoader(graphs, batch_size=2), settings(), 'cpu', tau=1.)
    for key in observed:
        assert observed[key] == pytest.approx(sum(r[key] for r in reference) / 3, abs=1e-10)
    before = model.score.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    trained = objective_epoch(model, DataLoader(graphs, batch_size=3), settings(), 'cpu', tau=1., optimizer=optimizer)
    assert torch.isfinite(model.score) and model.score.detach() != before
    assert math.isfinite(trained['total'])


def test_single_precision_is_explicit_optional_mode(anchor_graph):
    result = gib_objective(torch.zeros(1), torch.ones(1), anchor_graph,
                           objective=settings(compute_dtype='float32'))
    assert result.loss.dtype == torch.float32
    double = gib_objective(torch.zeros(1), torch.ones(1), anchor_graph, objective=settings())
    torch.testing.assert_close(result.loss.double(), double.loss, atol=2e-6, rtol=2e-6)
