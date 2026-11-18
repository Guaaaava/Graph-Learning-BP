"""Current configuration contract and rejection of removed experiment options."""
from dataclasses import replace
import json
import pytest

from gib4cl.config import (DataConfig, ExperimentConfig, ModelConfig,
                          config_from_dict, load_config)


@pytest.mark.parametrize('value', [
    {'config_version': 'gib4cl.config.v1'},
    {'seed': 100},
    {'evaluation': {'seed': 700}},
    {'model': {'architecture_version': 'old'}},
    {'bp': {'algorithm_version': 'old'}},
    {'evaluation': {'include_information_proxy': True}},
    *[{'data': {name: value}} for name, value in (
        ('scenario', 'challenge'), ('anchor_layout', 'corners'),
        ('agent_distribution', 'clustered'), ('agent_comm_radius', 30.),
        ('anchor_comm_radius', 54.), ('cluster_std_ratio', 1/6),
        ('cluster_margin_ratio', .1), ('enable_nlos', False),
        ('nlos_ratio', 0.), ('nlos_bias_min', 2.), ('nlos_bias_max', 10.))],
])
def test_removed_inputs_fail_instead_of_being_ignored(value):
    with pytest.raises(ValueError, match='Unknown'):
        config_from_dict(value)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"seeds":{"training":100,"training":101}}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate configuration key: training'):
        load_config(path)


def test_split_sizes_are_immutable_and_serialization_is_detached():
    cfg = ExperimentConfig()
    with pytest.raises(TypeError):
        cfg.split_sizes['train'] = -1
    exported = cfg.to_dict()
    exported['split_sizes']['train'] = -1
    assert cfg.split_sizes['train'] == 2000
    copied = replace(cfg, split_sizes={'train': 2, 'val': 1, 'test': 1})
    assert copied.split_sizes['train'] == 2 and cfg.split_sizes['train'] == 2000
    assert config_from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


@pytest.mark.parametrize('section', ['seeds','data','model','training','objective','bp','evaluation'])
def test_python_construction_checks_every_nested_config(section):
    with pytest.raises(ValueError, match=section + ' must be a '):
        ExperimentConfig(**{section: {}})


@pytest.mark.parametrize('options', [{'node_in_dim': 5.0}, {'edge_in_dim': 4.0}])
def test_feature_dimensions_must_be_integers(options):
    with pytest.raises(ValueError, match='integer'):
        ModelConfig(**options)


def test_five_anchor_coordinates_follow_area_size_without_radius_changes():
    cfg = DataConfig(area_size=20., comm_radius=7.)
    assert cfg.anchor_coordinates == ((0.,0.), (20.,0.), (0.,20.), (20.,20.), (10.,10.))
    assert cfg.num_anchors == 5 and cfg.comm_radius == 7.
