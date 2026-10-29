"""Independent small-graph and persistence checks for stage 2 only."""
from copy import deepcopy
from dataclasses import replace
import json
import random

import pytest
import torch
from torch_geometric.data import Batch

from gib4cl.data.schema import (
    GRAPH_VERSION, LOG_MEASUREMENT_CARDINALITY, agent_triangles, encode_measurements,
    input_features, normalize_covariances, prior_precision_blocks, representation_metadata,
    validate_record, validate_bundle,
)
from gib4cl.data.dataset import LocalizationDataset, graph_to_data
from gib4cl.config import (
    DataConfig, ExperimentConfig, config_from_dict, load_config, sample_seed,
)
from gib4cl.data.generate import build_dataset, create_dataset, save_dataset
from gib4cl.data.network import generate_graph


@pytest.fixture(autouse=True, scope='module')
def small_tensor_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def hand_graph():
    return {
        'schema_version': GRAPH_VERSION, 'sample_id': 'hand-graph',
        'inputs': {
            'initial_positions': torch.tensor([[0., 0.], [3., 0.], [0., 4.], [3., 4.]]),
            'initial_covariances': torch.tensor([[[4., 0.], [0., 9.]],
                                                 [[1., .25], [.25, 2.]],
                                                 [[25., 0.], [0., 25.]]]),
            'is_anchor': torch.tensor([False, False, False, True]),
            'edge_index': torch.tensor([[0, 0, 1, 0, 1, 2], [1, 2, 2, 3, 3, 3]]),
            'measurements': torch.tensor([2., 6., 4., 5., -2., 3.]),
            'measurement_variances': torch.tensor([1., 2., 3., 4., 5., 6.]),
            'communication_cost_scale': 2.,
            'measurement_encoding': 'ieee754-binary32-canonical-finite',
        },
        'targets': {'agent_positions': torch.tensor([[100., 100.], [101., 101.], [102., 102.]])},
        'diagnostics': {},
    }


def test_hand_features_and_physical_prior(hand_graph):
    validate_record(hand_graph)
    data = graph_to_data(hand_graph)
    torch.testing.assert_close(data.x[0], torch.tensor([0., 0., 4., 9., 0.]))
    torch.testing.assert_close(data.x[3], torch.tensor([3., 4., 0., 0., 1.]))
    torch.testing.assert_close(data.initial_distances, torch.tensor([3., 4., 5., 5., 4., 3.]))
    torch.testing.assert_close(data.signed_residual, torch.tensor([-1., 2., -1., 0., -6., 0.]))
    torch.testing.assert_close(data.edge_attr[:, 2], data.signed_residual.abs())
    torch.testing.assert_close(data.edge_attr[:, 3], torch.tensor([0., 0., 0., 1., 1., 1.]))
    torch.testing.assert_close(data.edge_cost, torch.tensor([18., 32., 50., 50., 32., 18.]))
    precision = prior_precision_blocks(hand_graph['inputs'])
    torch.testing.assert_close(precision[0], torch.diag(torch.tensor([.25, 1 / 9])))
    torch.testing.assert_close(precision[1], torch.tensor([[2., -.25], [-.25, 1.]]) / 1.9375)
    torch.testing.assert_close(precision[2], .04 * torch.eye(2))


def test_features_do_not_access_labels_or_diagnostic_flags(hand_graph):
    before = graph_to_data(hand_graph, include_targets=False)
    modified = deepcopy(hand_graph)
    modified.pop('targets')
    modified['diagnostics'] = {'untrusted_truth': 'must never enter features'}
    after = graph_to_data(modified, include_targets=False)
    assert 'y' not in after and 'untrusted_truth' not in after
    assert set(before.keys()) == set(after.keys())
    for key in before.keys():
        if isinstance(before[key], torch.Tensor):
            torch.testing.assert_close(before[key], after[key])
    with pytest.raises(ValueError, match='targets'):
        graph_to_data(modified)


def test_triangles_and_two_arcs_per_unique_measurement(hand_graph):
    data = graph_to_data(hand_graph)
    assert data.triangle_edge_index.tolist() == [[0], [2], [1]]
    assert data.message_edge_index.shape == (2, 12)
    torch.testing.assert_close(data.message_edge_index[:, 6:], data.edge_index.flip(0))
    torch.testing.assert_close(data.message_edge_to_measurement, torch.arange(6).repeat(2))
    torch.testing.assert_close(data.message_reverse_index[data.message_reverse_index], torch.arange(12))


def test_empty_graph_and_unanchored_agents_are_retained():
    cfg = replace(DataConfig(), comm_radius=0.)
    record = generate_graph(cfg, 2, generator=torch.Generator().manual_seed(10))
    data = graph_to_data(record)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, 4)
    assert data.message_edge_index.shape == (2, 0)
    assert data.triangle_edge_index.shape == (3, 0)
    assert record['diagnostics']['num_unanchored_agents'] == 2
    assert record['diagnostics']['num_agents_below_degree_three'] == 2
    assert data.agent_prior_cov.shape == (2, 2, 2)


def test_graph_without_anchors(hand_graph):
    record = deepcopy(hand_graph)
    record['inputs']['initial_positions'] = record['inputs']['initial_positions'][:3]
    record['inputs']['is_anchor'] = torch.zeros(3, dtype=torch.bool)
    for key in ('edge_index',):
        record['inputs'][key] = record['inputs'][key][:, :3]
    for key in ('measurements', 'measurement_variances'):
        record['inputs'][key] = record['inputs'][key][:3]
    record['diagnostics'] = {}
    data = graph_to_data(record)
    assert data.num_anchors == 0 and data.anchor_node_index.numel() == 0
    assert not data.edge_attr[:, 3].any()


def test_mixed_size_batch_offsets_and_roundtrip(hand_graph):
    empty = generate_graph(replace(DataConfig(), comm_radius=0.),
                           2, generator=torch.Generator().manual_seed(1), sample_id='empty')
    larger = generate_graph(replace(DataConfig(), comm_radius=1000.),
                            5, generator=torch.Generator().manual_seed(2), sample_id='larger')
    singles = [graph_to_data(hand_graph), graph_to_data(empty), graph_to_data(larger)]
    batch = Batch.from_data_list(singles)
    offset_nodes = offset_edges = offset_arcs = 0
    for i, original in enumerate(singles):
        e, arcs = original.edge_index.shape[1], original.message_edge_index.shape[1]
        torch.testing.assert_close(batch.message_edge_to_measurement[offset_arcs:offset_arcs + arcs],
                                   original.message_edge_to_measurement + offset_edges)
        torch.testing.assert_close(batch.message_reverse_index[offset_arcs:offset_arcs + arcs],
                                   original.message_reverse_index + offset_arcs)
        for key in original.keys():
            restored = batch.get_example(i)[key]
            if isinstance(original[key], torch.Tensor):
                torch.testing.assert_close(restored, original[key])
        assert torch.equal(batch.batch[batch.message_edge_index[0]],
                           batch.batch[batch.message_edge_index[1]])
        offset_nodes += original.num_nodes
        offset_edges += e
        offset_arcs += arcs
    # A real PyG batch must align triangle measurement IDs with the correct graph.
    assert batch.triangle_edge_index[:, 1:].min() >= singles[0].num_edges
    assert batch.x.shape[0] == offset_nodes


@pytest.mark.parametrize('failure', ['duplicate_edge', 'self_edge', 'local_anchor',
                                    'zero_variance', 'nan_range', 'same_initial_position',
                                    'non_spd', 'label_in_inputs'])
def test_reject_invalid_records(hand_graph, failure):
    inputs = hand_graph['inputs']
    if failure == 'duplicate_edge':
        inputs['edge_index'][:, 1] = inputs['edge_index'][:, 0]
    elif failure == 'self_edge':
        inputs['edge_index'][1, 0] = 0
    elif failure == 'local_anchor':
        inputs['edge_index'][1, 3] = 0
    elif failure == 'zero_variance':
        inputs['measurement_variances'][0] = 0
    elif failure == 'nan_range':
        inputs['measurements'][0] = float('nan')
    elif failure == 'same_initial_position':
        inputs['initial_positions'][1] = inputs['initial_positions'][0]
    elif failure == 'non_spd':
        inputs['initial_covariances'][0, 0, 0] = -1
    else:
        inputs['true_positions'] = hand_graph['targets']['agent_positions']
    with pytest.raises(ValueError):
        validate_record(hand_graph)


def test_encoding_support_and_no_range_clipping():
    meta = representation_metadata()
    assert meta['name'] == 'ieee754-binary32-canonical-finite'
    assert meta['range_value_count'] == 4278190079
    assert meta['variance_value_count'] == 2139095039
    assert meta['joint_value_count'] == 4278190079 * 2139095039
    assert LOG_MEASUREMENT_CARDINALITY > 40
    encoded = encode_measurements([-5., -0., 0., 2.])
    assert encoded[0] == -5. and not torch.signbit(encoded[1])
    with pytest.raises(ValueError):
        encode_measurements([float('inf')])
    with pytest.raises(ValueError):
        encode_measurements([1e-100], positive=True)


def test_gaussian_measurements_are_not_truncated():
    cfg = replace(DataConfig(), area_size=1., comm_radius=2.,
                  base_noise_std=100., distance_noise_scale=0.)
    errors, observations = [], []
    for seed in range(60):
        graph = generate_graph(cfg, 4, generator=torch.Generator().manual_seed(seed))
        truth = torch.cat((graph['targets']['agent_positions'],
                           graph['inputs']['initial_positions'][4:]))
        u, v = graph['inputs']['edge_index']
        z = graph['inputs']['measurements']
        errors.append((z - (truth[u] - truth[v]).norm(dim=1)) / 100.)
        observations.append(z)
    samples = torch.cat(errors)
    assert abs(samples.mean()) < .12
    assert .88 < samples.std() < 1.12
    assert (torch.cat(observations) < 0).any()


def test_single_scene_geometry_uniform_positions_and_shared_radius():
    cfg = DataConfig()
    record = generate_graph(cfg, 5, generator=torch.Generator().manual_seed(13))
    assert record['metadata']['measurement_model'] == 'los_gaussian'
    expected = torch.rand((5, 2), generator=torch.Generator().manual_seed(13)) * cfg.area_size
    torch.testing.assert_close(record['targets']['agent_positions'], expected, rtol=0, atol=0)
    anchors = torch.tensor([[0., 0.], [100., 0.], [0., 100.], [100., 100.], [50., 50.]])
    torch.testing.assert_close(record['inputs']['initial_positions'][5:], anchors)
    truth = torch.cat((expected, anchors))
    candidates = torch.triu_indices(10, 10, offset=1)
    distances = (truth[candidates[0]] - truth[candidates[1]]).norm(dim=1)
    kept = (candidates[0] < 5) & (distances < cfg.comm_radius)
    torch.testing.assert_close(record['inputs']['edge_index'], candidates[:, kept])


def test_dataset_loader_rejects_an_unsupported_measurement_model(tmp_path):
    bundle = build_dataset(load_config(), 'train', 1, agents_min=3, agents_max=3)
    bundle['graphs'][0]['metadata']['measurement_model'] = 'unsupported'
    path = tmp_path / 'unsupported.pt'
    torch.save(bundle, path)
    with pytest.raises(ValueError, match='Only LOS'):
        LocalizationDataset(path)


def test_configuration_is_explicit_validated_and_roundtrips():
    cfg = load_config()
    assert config_from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert cfg.data.num_anchors == 5
    assert cfg.data.comm_radius == 30.
    assert cfg.objective.beta is None and cfg.objective.kappa is None
    assert replace(cfg, training=replace(cfg.training, epochs=1)).data_fingerprint == cfg.data_fingerprint
    assert replace(cfg, data=replace(cfg.data, comm_radius=40.)).dataset_path('train') != cfg.dataset_path('train')
    assert sample_seed(100, 'train', 0, 0) != sample_seed(100, 'test', 0, 0)
    for invalid in ({'typo': 1}, {'data': {'comm_raduis': 10}},
                    {'data': {'agent_distribution': 'auto'}},
                    {'data': {'num_agents_min': True}}, {'seed': -1},
                    {'experiment_id': '../escape'}, {'data': {'initial_covariance': [[1, 2], [2, 1]]}}):
        with pytest.raises(ValueError):
            config_from_dict(invalid)


def test_per_sample_streams_prefix_and_split_identity():
    cfg = replace(load_config(), data=replace(DataConfig(), num_agents_min=2, num_agents_max=4))
    state_torch, state_python = torch.get_rng_state().clone(), random.getstate()
    first = build_dataset(cfg, 'train', 2)
    extended = build_dataset(cfg, 'train', 3)
    test = build_dataset(cfg, 'test', 2)
    assert torch.equal(torch.get_rng_state(), state_torch) and random.getstate() == state_python
    for i in range(2):
        assert first['graphs'][i]['sample_id'] == extended['graphs'][i]['sample_id']
        for key, value in first['graphs'][i]['inputs'].items():
            other = extended['graphs'][i]['inputs'][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other)
            else:
                assert value == other
    assert not ({g['metadata']['sample_seed'] for g in first['graphs']}
                & {g['metadata']['sample_seed'] for g in test['graphs']})


def test_serialization_checksum_and_overwrite_protection(tmp_path):
    cfg = load_config()
    path = create_dataset('train', 2, agents_min=2, agents_max=4,
                          settings=cfg, output_root=tmp_path)
    ds = LocalizationDataset(path)
    assert len(ds) == 2 and ds[0].x.shape[1] == 5
    assert ds.metadata['config']['data']['comm_radius'] == cfg.data.comm_radius
    assert ds.metadata['representation']['log_joint_value_count'] == LOG_MEASUREMENT_CARDINALITY
    with pytest.raises(FileExistsError):
        create_dataset('train', 2, settings=cfg, output_root=tmp_path)
    sidecar = path.with_suffix('.json')
    report = json.loads(sidecar.read_text(encoding='utf-8'))
    report['sha256'] = 'bad'
    sidecar.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        LocalizationDataset(path)


def test_legacy_list_datasets_are_not_migrated(tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save([{'init_agents_pos': torch.zeros((2, 2))}], path)
    with pytest.raises(ValueError, match='Expected'):
        LocalizationDataset(path)


@pytest.mark.parametrize('failure', ['config', 'sample_id', 'seed', 'representation', 'count', 'agent_interval', 'anchor_geometry'])
def test_bundle_rejects_inconsistent_provenance(failure):
    bundle = build_dataset(load_config(), 'train', 1, agents_min=2, agents_max=2)
    if failure == 'config':
        bundle['metadata']['config']['data']['area_size'] = 1.
    elif failure == 'sample_id':
        bundle['graphs'][0]['sample_id'] = 'wrong'
    elif failure == 'seed':
        bundle['graphs'][0]['metadata']['sample_seed'] += 1
    elif failure == 'representation':
        bundle['metadata']['representation']['joint_value_count'] = 3
    elif failure == 'count':
        bundle['metadata']['num_graphs'] = 2
    elif failure == 'agent_interval':
        bundle['graphs'][0]['inputs']['initial_covariances'] = torch.eye(2).repeat(1, 1, 1)
    else:
        bundle['graphs'][0]['inputs']['initial_positions'][-1] += 1.
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_correlated_prior_sampling_is_reproducible():
    cfg = replace(DataConfig(), initial_covariance=((4., 1.), (1., 9.)))
    first = generate_graph(cfg, 4, generator=torch.Generator().manual_seed(12))
    repeated = generate_graph(cfg, 4,
                              generator=torch.Generator().manual_seed(12))
    for name, value in first['inputs'].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, repeated['inputs'][name])
    torch.testing.assert_close(first['inputs']['initial_covariances'][0], torch.tensor([[4., 1.], [1., 9.]]))


def test_bfs_adapter_accepts_global_indices_and_rejects_local_anchors(hand_graph):
    from gib4cl.inference.spanning_tree import extract_bfs_tree_edges
    inputs = hand_graph['inputs']
    flags = inputs['is_anchor'][inputs['edge_index'][1]]
    weights = extract_bfs_tree_edges(3, 1, inputs['edge_index'], flags)
    assert weights.shape == (6,) and weights[3:].all()
    local = inputs['edge_index'].clone()
    local[1, flags] -= 3
    with pytest.raises(ValueError):
        extract_bfs_tree_edges(3, 1, local, flags)


def test_binary32_generation_does_not_depend_on_global_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        record = generate_graph(DataConfig(),
                                3, generator=torch.Generator().manual_seed(19))
        validate_record(record)
        assert record['inputs']['initial_positions'].dtype == torch.float32
        assert record['inputs']['measurements'].dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)
