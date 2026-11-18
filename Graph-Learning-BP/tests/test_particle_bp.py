"""Stage 5: exact finite-state posterior, cavity semantics and particle numerics."""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gib4cl.inference import ParticleBP, ParticleSupport, particle_bp_batch
from tests.references.particle_posterior import exact_discrete, gaussian_hermite_support, grid_support, radial_single_anchor
from gib4cl.config import BPConfig
from tests.test_objectives import make_graph


@pytest.fixture(autouse=True, scope='module')
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph_case(kind):
    a = 1 if kind == 'anchors' else 2 if kind == 'single' else 3 if kind in ('chain', 'cycle') else 4
    agents = [[0., 0.], [2., 0.], [4., 1.], [2., 3.]][:a]
    anchors = [[-2., 1.], [6., 2.]]
    aa = {'anchors': [], 'single': [[0, 1]], 'chain': [[0, 1], [1, 2]],
          'cycle': [[0, 1], [0, 2], [1, 2]], 'tree': [[0, 1], [1, 2], [1, 3]],
          'forest': [[0, 1]]}[kind]
    edges = sorted(aa + [[0, a], [a-1, a+1]])
    coordinates = torch.tensor(agents + anchors)
    measurements = [float((coordinates[u]-coordinates[v]).norm()) + .4 for u, v in edges]
    covariance = torch.tensor([[[1.2, .2], [.2, 2.]]]).repeat(a, 1, 1)
    return make_graph(agents, anchors, edges, measurements, [1.1]*len(edges), covariance)


def explicit_support(data, p=5):
    generator = torch.Generator().manual_seed(55)
    agents = data.agent_node_index
    points = data.x[agents, :2].double()[:, None, :] + torch.randn(len(agents), p, 2, generator=generator, dtype=torch.float64)
    log_mass = torch.randn(len(agents), p, generator=generator, dtype=torch.float64)
    return ParticleSupport(points, log_mass.log_softmax(dim=1), agents.clone())


def as_support(data, points, masses):
    return ParticleSupport(torch.tensor(points, dtype=torch.float64),
                           torch.tensor(masses, dtype=torch.float64), data.agent_node_index.clone())


@pytest.mark.parametrize('kind', ['anchors', 'single', 'chain', 'tree', 'forest'])
@pytest.mark.parametrize('mask_mode', ['all', 'none', 'partial'])
def test_forest_bp_matches_full_joint_enumeration(kind, mask_mode):
    data = graph_case(kind)
    support = explicit_support(data, 4)
    mask = torch.ones(data.num_edges)
    if mask_mode == 'none':
        mask.zero_()
    elif mask_mode == 'partial':
        mask[::2] = 0
    actual = ParticleBP(4, 10, chunk_size=2).infer(data, mask, support=support)
    reference = exact_discrete(data, support.points, support.log_prior_weights, mask)
    np.testing.assert_allclose(actual.weights.numpy(), reference.marginals, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(actual.mean.numpy(), reference.mean, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(actual.covariance.numpy(), reference.covariance, rtol=3e-13, atol=2e-14)
    assert actual.converged


def test_single_edge_has_no_immediate_return_or_repeated_evidence():
    data, bp = graph_case('single'), ParticleBP(5, 12, chunk_size=3)
    result = bp.infer(data, support=explicit_support(data), return_history=True)
    for item in result.history[2:]:
        torch.testing.assert_close(item['weights'], result.history[1]['weights'], rtol=0, atol=0)
    assert not torch.equal(result.history[0]['weights'], result.history[1]['weights'])


def test_chain_information_advances_one_hop_per_synchronous_round():
    data, other = graph_case('chain'), graph_case('chain')
    anchor_edge = torch.where((data.edge_index[0] == 0) & (data.edge_index[1] == 3))[0]
    other.edge_attr[anchor_edge, 0] += 2
    bp, support = ParticleBP(5, 3), explicit_support(data)
    first, second = (bp.infer(item, support=support, return_history=True) for item in (data, other))
    torch.testing.assert_close(first.history[1]['weights'][2], second.history[1]['weights'][2], rtol=0, atol=0)
    assert not torch.allclose(first.history[2]['weights'][2], second.history[2]['weights'][2])


def independent_probability_bp(data, support, rounds):
    # Tiny moderate-valued problem: probability-domain dictionary recursion,
    # implemented independently of log-domain/chunked production code.
    points = support.points.numpy()
    masses = support.log_prior_weights.exp().numpy()
    a, p = masses.shape
    local = masses.copy()
    neighbors, kernels = [[] for _ in range(a)], {}
    for edge, (u, v) in enumerate(data.edge_index.t().tolist()):
        z, variance = data.edge_attr[edge, :2].double().numpy()
        if v >= a:
            local[u] *= np.exp(-.5*(np.linalg.norm(points[u]-data.x[v, :2].numpy(), axis=1)-z)**2/variance)
        else:
            kernel = np.exp(-.5*(np.linalg.norm(points[u, :, None]-points[v, None, :], axis=-1)-z)**2/variance)
            kernels[u, v], kernels[v, u] = kernel, kernel.T
            neighbors[u].append(v)
            neighbors[v].append(u)
    messages = {pair: np.ones(p)/p for pair in kernels}
    for _ in range(rounds):
        updated = {}
        for (u, v), kernel in kernels.items():
            cavity = local[u].copy()
            for neighbor in neighbors[u]:
                if neighbor != v:
                    cavity *= messages[neighbor, u]
            cavity /= cavity.sum()
            message = cavity @ kernel
            updated[u, v] = message/message.sum()
        messages = updated
    weights = local.copy()
    for node in range(a):
        for neighbor in neighbors[node]:
            weights[node] *= messages[neighbor, node]
        weights[node] /= weights[node].sum()
    return weights


@pytest.mark.parametrize('rounds', [0, 1, 2, 8])
def test_cyclic_iterations_match_independent_probability_bp(rounds):
    data, support = graph_case('cycle'), explicit_support(graph_case('cycle'))
    actual = ParticleBP(5, rounds, chunk_size=2).infer(data, support=support)
    np.testing.assert_allclose(actual.weights.numpy(), independent_probability_bp(data, support, rounds),
                               rtol=4e-13, atol=2e-14)


def test_loop_is_not_required_to_equal_exact_posterior():
    data = graph_case('cycle')
    support = explicit_support(data, 5)
    actual = ParticleBP(5, 40).infer(data, support=support)
    reference = exact_discrete(data, support.points, support.log_prior_weights)
    # A controlled case shows the distinction; no universal sign is claimed.
    assert np.max(np.abs(actual.weights.numpy()-reference.marginals)) > 1e-5


@pytest.mark.parametrize('chunk', [1, 2, 4, 99])
def test_chunking_matches_dense_and_reuses_one_kernel_for_both_directions(chunk):
    data, support = graph_case('cycle'), explicit_support(graph_case('cycle'), 7)
    dense = ParticleBP(7, 4, chunk_size=99).infer(data, support=support)
    actual = ParticleBP(7, 4, chunk_size=chunk).infer(data, support=support)
    torch.testing.assert_close(actual.mean, dense.mean, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(actual.log_messages, dense.log_messages, rtol=1e-12, atol=1e-13)
    assert actual.diagnostics['max_kernel_elements'] == min(chunk, 7)**2
    assert actual.diagnostics['pair_distance_evaluations'] == 3*4*7*7


def test_damping_is_probability_mixture_and_converges_to_tree_reference():
    data, support = graph_case('chain'), explicit_support(graph_case('chain'))
    undamped = ParticleBP(5, 1).infer(data, support=support)
    damped = ParticleBP(5, 1, damping=.7).infer(data, support=support)
    torch.testing.assert_close(damped.log_messages.exp(), .7/5 + .3*undamped.log_messages.exp())
    result = ParticleBP(5, 300, damping=.3, early_stopping=True, message_tolerance=1e-12).infer(data, support=support)
    reference = exact_discrete(data, support.points, support.log_prior_weights)
    assert result.converged and result.iterations < 300
    np.testing.assert_allclose(result.weights.numpy(), reference.marginals, atol=2e-12, rtol=1e-11)


def test_early_stop_fixed_rounds_and_zero_round_status():
    data, support = graph_case('chain'), explicit_support(graph_case('chain'))
    early = ParticleBP(5, 20, early_stopping=True).infer(data, support=support)
    fixed = ParticleBP(5, 20).infer(data, support=support)
    zero = ParticleBP(5, 0).infer(data, support=support)
    assert early.iterations <= 4 and early.converged
    assert fixed.iterations == 20
    assert zero.iterations == 0 and not zero.converged and zero.message_residual is None
    torch.testing.assert_close(early.weights, fixed.weights)


def test_full_covariance_prior_proposal_and_uniform_masses():
    data = make_graph([[1., -2.]], [], [], [], [], [[[4., 1.5], [1.5, 9.]]])
    p = 50000
    bp = ParticleBP(p, 0)
    support = bp.draw_support(data, generator=torch.Generator().manual_seed(876))
    result = bp.infer(data, support=support)
    covariance = data.agent_prior_cov[0].double()
    mean_tolerance = 6*torch.sqrt(covariance.diag()/p)
    assert ((result.mean[0]-data.x[0, :2].double()).abs() < mean_tolerance).all()
    covariance_tolerance = 6*torch.sqrt((covariance.square()+covariance.diag()[:, None]*covariance.diag()[None, :])/p)
    assert ((result.covariance[0]-covariance).abs() < covariance_tolerance).all()
    torch.testing.assert_close(result.weights, torch.full((1,p), 1/p, dtype=torch.float64))
    assert abs(float(result.ess)-p) < 1e-7


def test_prior_not_multiplied_twice_and_weighted_population_covariance():
    data = make_graph([[0., 0.]], [], [], [], [], [torch.eye(2).tolist()])
    support = ParticleSupport(torch.tensor([[[0.,0.],[1.,0.],[2.,0.]]], dtype=torch.float64),
                              torch.zeros(1,3,dtype=torch.float64), torch.tensor([0]))
    result = ParticleBP(3, 5).infer(data, support=support)
    torch.testing.assert_close(result.mean, torch.tensor([[1.,0.]],dtype=torch.float64))
    torch.testing.assert_close(result.covariance, torch.tensor([[[2/3,0.],[0.,0.]]],dtype=torch.float64))
    assert result.covariance_min_eigenvalue[0] == 0  # no artificial jitter


def test_known_small_variance_not_replaced_by_old_baseline_floor():
    data = make_graph([[1.,0.]], [[0.,0.]], [[0,1]], [1.1], [.0001])
    support = explicit_support(data)
    actual = ParticleBP(5).infer(data, support=support)
    reference = exact_discrete(data, support.points, support.log_prior_weights)
    np.testing.assert_allclose(actual.weights.numpy(), reference.marginals, rtol=1e-10, atol=1e-14)
    altered = ParticleBP(5, measurement_variance_floor=.25).infer(data, support=support)
    assert not torch.allclose(actual.weights, altered.weights)
    assert altered.diagnostics['variance_floor_applied_edges'] == 1


def test_tiny_likelihoods_do_not_get_epsilon_floor_and_ess_reports_collapse():
    data = make_graph([[1.,0.]], [[0.,0.]], [[0,1]], [-1000.], [.0001])
    support = ParticleSupport(torch.tensor([[[1.,0.],[2.,0.],[3.,0.]]],dtype=torch.float64),
                              torch.zeros(1,3,dtype=torch.float64),torch.tensor([0]))
    result = ParticleBP(3).infer(data,support=support)
    torch.testing.assert_close(result.weights,torch.tensor([[1.,0.,0.]],dtype=torch.float64))
    assert result.ess[0] == 1 and result.covariance.count_nonzero() == 0


def test_zero_prior_masses_are_supported_without_nan():
    data, support = graph_case('chain'), explicit_support(graph_case('chain'))
    support.log_prior_weights[:,0] = -torch.inf
    result = ParticleBP(5).infer(data,support=support)
    assert torch.isfinite(result.mean).all() and (result.weights[:,0] == 0).all()


def test_all_pair_likelihoods_underflow_in_probability_domain_but_bp_remains_informative():
    data = make_graph([[0.,0.],[5.,0.]],[],[[0,1]],[-100.],[.01])
    support = ParticleSupport(torch.tensor([[[0.,0.],[1.,0.],[2.,0.]],
                                            [[5.,0.],[6.,0.],[7.,0.]]],dtype=torch.float64),
                              torch.zeros(2,3,dtype=torch.float64),torch.tensor([0,1]))
    result = ParticleBP(3,2,chunk_size=2).infer(data,support=support)
    exact = exact_discrete(data,support.points,support.log_prior_weights)
    np.testing.assert_allclose(result.weights.numpy(),exact.marginals,atol=1e-14,rtol=0)
    torch.testing.assert_close(result.mean,torch.tensor([[2.,0.],[5.,0.]],dtype=torch.float64))
    assert torch.isfinite(result.log_messages).all() and (result.ess == 1).all()


def test_float32_overflow_is_reported_without_hidden_regularization():
    data = make_graph([[0.,0.],[1.,0.]],[],[[0,1]],[1.],[1e-20])
    support = ParticleSupport(torch.tensor([[[1e30,0.],[2e30,0.]],
                                            [[-1e30,0.],[-2e30,0.]]]),
                              torch.zeros(2,2),torch.tensor([0,1]))
    with pytest.raises(FloatingPointError,match='float64'):
        ParticleBP(2,1,compute_dtype='float32').infer(data,support=support)
    precise = ParticleSupport(support.points.double(),support.log_prior_weights.double(),support.agent_node_index)
    assert torch.isfinite(ParticleBP(2,1).infer(data,support=precise).mean).all()


def test_label_isolation_reproducibility_and_no_resampling_or_roughening():
    data, bp = graph_case('chain'), ParticleBP(5,4)
    rng = torch.random.get_rng_state().clone()
    support = bp.draw_support(data,generator=torch.Generator().manual_seed(9))
    copied = bp.draw_support(data,generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(support.points,copied.points,rtol=0,atol=0)
    first = bp.infer(data,support=support)
    changed = deepcopy(data)
    changed.y.fill_(float('nan'))
    changed.diagnostic_truth = torch.ones(data.num_edges,dtype=torch.bool)
    second = bp.infer(changed,support=support)
    torch.testing.assert_close(first.mean,second.mean,rtol=0,atol=0)
    mean,covariance,points = bp.run(data,support=support)
    torch.testing.assert_close(mean,first.mean,rtol=0,atol=0)
    torch.testing.assert_close(covariance,first.covariance,rtol=0,atol=0)
    torch.testing.assert_close(points,support.points.transpose(1,2),rtol=0,atol=0)
    assert torch.equal(rng,torch.random.get_rng_state())


def test_endpoint_and_edge_order_invariance():
    data, support = graph_case('tree'), explicit_support(graph_case('tree'))
    changed = deepcopy(data)
    changed.edge_index = data.edge_index.flip(0).flip(1)
    changed.edge_attr = data.edge_attr.flip(0)
    expected = ParticleBP(5).infer(data,support=support)
    actual = ParticleBP(5).infer(changed,support=support)
    torch.testing.assert_close(actual.mean,expected.mean)
    torch.testing.assert_close(actual.covariance,expected.covariance)


def test_arbitrary_global_node_order_including_anchor_first():
    data, support = graph_case('chain'), explicit_support(graph_case('chain'))
    order = torch.tensor([3,2,0,4,1])
    inverse = order.argsort()
    changed = deepcopy(data)
    changed.x = data.x[order]
    changed.edge_index = inverse[data.edge_index]
    changed.agent_node_index = torch.where(changed.x[:,4] == 0)[0]
    old_agent_row = order[changed.agent_node_index]  # original agents were first
    changed.agent_prior_cov = data.agent_prior_cov[old_agent_row]
    changed_support = ParticleSupport(support.points[old_agent_row], support.log_prior_weights[old_agent_row], changed.agent_node_index)
    actual = ParticleBP(5).infer(changed,support=changed_support)
    expected = ParticleBP(5).infer(data,support=support)
    torch.testing.assert_close(actual.mean,expected.mean[old_agent_row])
    torch.testing.assert_close(actual.covariance,expected.covariance[old_agent_row])


def test_anchor_reachability_respects_pruning_and_disconnected_components():
    data = graph_case('forest')
    result = ParticleBP(5).infer(data,support=explicit_support(data))
    assert result.anchor_reachable.tolist() == [True,True,False,True]
    pruned = ParticleBP(5).infer(data,torch.zeros(data.num_edges),support=explicit_support(data))
    assert not pruned.anchor_reachable.any()


def test_batch_matches_individual_with_shared_supports_and_masks():
    graphs = [graph_case('single'),make_graph([[1.,2.]],[],[],[],[]),graph_case('tree')]
    supports = [explicit_support(data) for data in graphs]
    masks = [torch.arange(data.num_edges)%2 for data in graphs]
    batch = Batch.from_data_list(graphs)
    results = particle_bp_batch(batch,torch.cat(masks),supports=supports,return_results=True,num_particles=5,num_iter=5)
    for data,support,mask,result in zip(graphs,supports,masks,results):
        expected = ParticleBP(5,5).infer(data,mask,support=support)
        torch.testing.assert_close(result.mean,expected.mean,rtol=0,atol=0)
    with pytest.raises(ValueError,match='one graph'):
        ParticleBP(5).infer(batch)
    with pytest.raises(ValueError):
        particle_bp_batch(batch,supports=supports[:1],num_particles=5)


@pytest.mark.parametrize('corruption',['prior_missing','prior_not_spd','prior_not_symmetric','diagonal_mismatch',
                                      'duplicate','out_of_bounds','anchor_flag','variance','nonfinite','mask','support'])
def test_invalid_inputs_fail_instead_of_silent_reinterpretation(corruption):
    data, support = graph_case('chain'), explicit_support(graph_case('chain'))
    mask = torch.ones(data.num_edges)
    if corruption == 'prior_missing':
        del data.agent_prior_cov
    elif corruption == 'prior_not_spd':
        data.agent_prior_cov[0,0,0] = -1
    elif corruption == 'prior_not_symmetric':
        data.agent_prior_cov[0,0,1] = .4
    elif corruption == 'diagonal_mismatch':
        data.x[0,2] = 99
    elif corruption == 'duplicate':
        data.edge_index[:,1] = data.edge_index[:,0]
    elif corruption == 'out_of_bounds':
        data.edge_index[1,0] = 99
    elif corruption == 'anchor_flag':
        data.edge_attr[0,3] = 1-data.edge_attr[0,3]
    elif corruption == 'variance':
        data.edge_attr[0,1] = 0
    elif corruption == 'nonfinite':
        data.edge_attr[0,0] = float('nan')
    elif corruption == 'mask':
        mask[0] = .4
    else:
        support.log_prior_weights[0] = -torch.inf
    with pytest.raises((ValueError,FloatingPointError)):
        ParticleBP(5).infer(data,mask,support=support)


@pytest.mark.parametrize('options',[{'num_particles':1},{'num_iter':-1},{'chunk_size':0},
                                    {'damping':1},{'message_tolerance':-1},{'compute_dtype':'float16'},
                                    {'sigma_meas':.5},{'init_cov':25.},{'reg_scale':.01}])
def test_invalid_or_obsolete_options_rejected(options):
    with pytest.raises((ValueError,TypeError)):
        ParticleBP(**options)


def test_config_factory_and_data_identity():
    config = BPConfig(num_particles=5,num_iterations=2,chunk_size=3)
    solver = ParticleBP.from_config(config)
    assert solver.chunk_size == 3 and solver.num_iter == 2
    data = graph_case('chain')
    result = solver.infer(data, support=explicit_support(data))
    assert 'algorithm_version' not in result.diagnostics
    with pytest.raises(TypeError):
        replace(config,algorithm_version='old')


def test_float32_agrees_on_moderate_problem():
    data,support = graph_case('chain'),explicit_support(graph_case('chain'))
    actual = ParticleBP(5,5,compute_dtype='float32').infer(data,support=ParticleSupport(support.points.float(),support.log_prior_weights.float(),support.agent_node_index))
    expected = ParticleBP(5,5).infer(data,support=support)
    torch.testing.assert_close(actual.mean.double(),expected.mean,rtol=2e-5,atol=2e-6)


def test_quadrature_gaussian_moments_and_reference_budget():
    data = make_graph([[1.,-2.]],[],[],[],[],[[[4.,1.],[1.,9.]]])
    points,masses = gaussian_hermite_support(data,3)
    reference = exact_discrete(data,points,masses)
    np.testing.assert_allclose(reference.mean,data.x[:,:2].numpy(),atol=1e-14)
    np.testing.assert_allclose(reference.covariance,data.agent_prior_cov.numpy(),atol=1e-14)
    with pytest.raises(ValueError,match='above budget'):
        exact_discrete(graph_case('tree'),explicit_support(graph_case('tree')).points,
                       explicit_support(graph_case('tree')).log_prior_weights,max_states=10)


def test_continuous_single_anchor_grid_matches_independent_radial_integral():
    data = make_graph([[.1,0.]],[[0.,0.]],[[0,1]],[3.2],[.7],[[[4.,0.],[0.,4.]]])
    data.x[0,:2] = 0  # nonlinear BP permits coincident prior mean and anchor
    radial = radial_single_anchor(4.,float(data.edge_attr[0,0]),float(data.edge_attr[0,1]),order=256)
    refined = radial_single_anchor(4.,float(data.edge_attr[0,0]),float(data.edge_attr[0,1]),order=512,radius=30.)
    np.testing.assert_allclose(radial['covariance'],refined['covariance'],atol=1e-12)
    points,masses = grid_support(data,[-12.,-12.],[12.,12.],401)
    reference = exact_discrete(data,points,masses)
    np.testing.assert_allclose(reference.mean[0],np.zeros(2),atol=1e-13)
    np.testing.assert_allclose(reference.covariance[0],radial['covariance'],atol=2e-6)
    actual = ParticleBP(len(points[0]),1).infer(data,support=as_support(data,points,masses))
    np.testing.assert_allclose(actual.covariance.numpy(),reference.covariance,atol=1e-12)


def test_two_agent_nonlinear_reference_agrees_across_quadrature_methods():
    data = make_graph([[-3.,0.],[3.,0.]], [[-7.,0.],[7.,1.]], [[0,1],[0,2],[1,3]],
                      [6.2,4.3,3.9], [1.,1.,1.],
                      [[[.5,.1],[.1,.7]],[[.6,-.1],[-.1,.5]]])
    points,masses = gaussian_hermite_support(data,29)
    reference = exact_discrete(data,points,masses)
    finer_points,finer_masses = gaussian_hermite_support(data,37)
    refined = exact_discrete(data,finer_points,finer_masses)
    np.testing.assert_allclose(reference.mean,refined.mean,atol=1e-11,rtol=0)
    np.testing.assert_allclose(reference.covariance,refined.covariance,atol=1e-11,rtol=0)
    mean = data.x[data.agent_node_index,:2].double().numpy()
    sd = np.sqrt(data.agent_prior_cov.diagonal(dim1=-2,dim2=-1).double().numpy())
    grid_points,grid_masses = grid_support(data,mean-6*sd,mean+6*sd,25)
    grid = exact_discrete(data,grid_points,grid_masses)
    np.testing.assert_allclose(grid.mean,refined.mean,atol=1e-8,rtol=0)
    np.testing.assert_allclose(grid.covariance,refined.covariance,atol=1e-8,rtol=0)
    actual = ParticleBP(len(points[0]),2,chunk_size=128).infer(data,support=as_support(data,points,masses))
    np.testing.assert_allclose(actual.mean.numpy(),reference.mean,atol=2e-12,rtol=0)
    np.testing.assert_allclose(actual.covariance.numpy(),reference.covariance,atol=2e-12,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime unavailable')
def test_cuda_matches_cpu_on_fixed_support():
    data,support = graph_case('chain'),explicit_support(graph_case('chain'))
    expected = ParticleBP(5,5).infer(data,support=support)
    gpu = data.clone().to('cuda')
    actual = ParticleBP(5,5).infer(gpu,support=ParticleSupport(support.points.cuda(),support.log_prior_weights.cuda(),support.agent_node_index.cuda()))
    torch.testing.assert_close(actual.mean.cpu(),expected.mean,rtol=1e-10,atol=1e-11)
