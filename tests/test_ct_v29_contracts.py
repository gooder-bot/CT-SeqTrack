"""新方法身份不能被旧checkpoint/resume/策略身份覆盖。"""
import copy
from pathlib import Path

import pytest

from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.online_contract import (build_online_resume_contract,
                                   validate_scratch_training_contract,
                                   validate_v28_evaluation_checkpoint)
from utils.action_calibration_v27 import action_calibration_config_identity
from utils.v29_contracts import V29_CONTRACTS


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arm,backend,variant', [('b0','gru','b0'),
    ('full_cfc','cfc','full'), ('full_gru','gru','full')])
def test_formal_three_arm_protocol(arm, backend, variant):
    config = load_yaml_config(ROOT / f'cfgs/ct_seqtrack/29_{arm}_nuscenes_full.yaml')
    configure_ct_variant(config)
    validate_scratch_training_contract(config)
    assert (config['ct_variant'], config['motion_v3_temporal_backend']) == (variant, backend)
    assert config['train_split'] == 'train_track'
    assert config['val_split'] == config['test_split'] == 'val'
    assert config['epoch'] == 60 and config['batch_size'] == 16
    assert config['workers'] == 4
    assert config['preloading'] is False
    identity = action_calibration_config_identity(config)
    resume = build_online_resume_contract(config)['fields']
    for key, expected in V29_CONTRACTS.items():
        assert config[key] == resume[key] == identity[key] == expected


@pytest.mark.parametrize('field', list(V29_CONTRACTS))
def test_v29_does_not_silently_normalize_a_different_method(field):
    config = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/29_b0_nuscenes_full.yaml')
    config[field] = 'wrong-contract'
    with pytest.raises(ValueError, match='v29 contract mismatch'):
        configure_ct_variant(config)


def test_v28_checkpoint_cannot_be_relabelled_as_v29():
    config = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/29_b0_nuscenes_full.yaml')
    old = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/28_b0_nuscenes_full.yaml')
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_v28_evaluation_checkpoint({'hyper_parameters': {'config': old}}, config)
    new = copy.deepcopy(config)
    validate_v28_evaluation_checkpoint({'hyper_parameters': {'config': new}}, config)


@pytest.mark.parametrize('arm', ['b0', 'full_cfc', 'full_gru'])
@pytest.mark.parametrize('workers', [4, 12])
def test_v29_registered_worker_counts_preserve_resume_identity(arm, workers):
    config = configure_ct_variant(load_yaml_config(
        ROOT / f'cfgs/ct_seqtrack/29_{arm}_nuscenes_full.yaml'))
    other = dict(config, workers=12 if workers == 4 else 4)
    config['workers'] = workers
    validate_scratch_training_contract(config)
    assert build_online_resume_contract(config) != build_online_resume_contract(other)


@pytest.mark.parametrize('workers', [0, -1, 8, 4.0, '4', True])
def test_v29_rejects_unregistered_formal_worker_counts(workers):
    config = configure_ct_variant(load_yaml_config(
        ROOT / 'cfgs/ct_seqtrack/29_b0_nuscenes_full.yaml'))
    config['workers'] = workers
    with pytest.raises(ValueError, match='formal v29 workers'):
        validate_scratch_training_contract(config)
