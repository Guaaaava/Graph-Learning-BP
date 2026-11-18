"""Check extracted experiment functionality without historical directories."""
import pytest
import torch
from gib4cl.config import DataConfig
from gib4cl.data.network import generate_graph
from gib4cl.data.dataset import graph_to_data
from gib4cl.inference import ParticleBP
from gib4cl.forest_merge import merge_forest_edges
from tests.test_particle_bp import graph_case, explicit_support
from tools.bp_cached_fast import CachedParticleBP
from tools.baseline_topologies import select_forests, center_tree


@pytest.mark.parametrize('kind', ['chain', 'cycle', 'forest'])
def test_formal_distance_backend_preserves_bp_results(kind):
    data = graph_case(kind)
    support = explicit_support(data, 7)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = ParticleBP(7, 4, chunk_size=3).infer(data, support=support)
        solver = CachedParticleBP(7, 4, chunk_size=3, cache_bytes=0)
        actual = solver.infer(data, support=support)
        for field in ('mean', 'covariance', 'weights', 'log_messages', 'ess'):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=1e-12, atol=1e-13)
        assert solver.last_cache_stats['released_after_infer']
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('radius', [0., 200.])
def test_extracted_forests_and_center_tree_on_connected_and_isolated_nodes(radius):
    settings = DataConfig(comm_radius=radius, base_noise_std=1., distance_noise_scale=0.)
    data = graph_to_data(generate_graph(settings, 6, generator=torch.Generator().manual_seed(1800)))
    forests = select_forests(data)
    equal = merge_forest_edges(data.num_nodes, data.edge_index.T.tolist(), [], [1.] * data.num_edges)
    assert forests['variance_msf'] == equal
    center = center_tree(data)
    if radius == 0:
        assert all(ids == [] for ids in forests.values())
        assert center is None
    else:
        assert all(len(ids) == data.num_nodes - 1 for ids in forests.values())
        assert center[0] in data.anchor_node_index.tolist()
        assert len(center[1]) == data.num_nodes - 1
