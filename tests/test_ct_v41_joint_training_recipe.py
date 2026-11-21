"""单组联合训练与历史配方的入口、回放来源及模型身份。"""
from pathlib import Path

import pytest

from models.ct_v41.config import config_identity, load_config, recipe_identity
from models.ct_v41.entry import parse_config


CONFIGS = Path(__file__).resolve().parents[1] / 'cfgs/ct_seqtrack'


def test_joint_training_uses_one_model_and_one_updating_prefix_source():
    cfg = parse_config(['--cfg', str(CONFIGS / '41_full_kitti_joint_training.yaml')])
    assert cfg.v41_joint_search and cfg.v41_q0_lateral and not cfg.v41_q0_lateral_replay
    assert cfg.v41_history_training == 'window_1448' and cfg.v41_joint_replay == 'previous_epoch'
    assert cfg.v41_fixed_replay_bank is None and cfg.v41_q0_replay_manifest is None
    assert cfg.eval_split == 'test' and cfg.full_diagnostic_epochs == [58, 59, 60]
    identity = recipe_identity(cfg)
    assert len(identity['candidates']) == 6 and identity['lateral']['warmup_epochs'] == 10
    assert identity['replay'] == 'causal_auxiliary_bank_every_fourth_singleton'
    flexible = parse_config(['--cfg', cfg.cfg, '--v41_q0_lateral_warmup_epochs', '0',
                             '--v41_patrol_seconds', '.6', '--v41_weak_patrol_seconds', '.15'])
    assert recipe_identity(flexible)['patrol_seconds'] == .6
    assert recipe_identity(flexible)['weak_patrol_seconds'] == .15
    assert recipe_identity(flexible)['lateral']['warmup_epochs'] == 0
    assert config_identity(flexible) != config_identity(cfg)
    assert config_identity(dict(cfg, v41_history_bank='elsewhere.json')) == config_identity(cfg)


def test_history_comparisons_and_uniform_donor_recipe_keep_their_own_semantics():
    pair = load_config(CONFIGS / '41_history_window_1448.yaml')
    assert pair.v41_joint_replay == 'fixed_e2' and not pair.v41_q0_lateral
    assert recipe_identity(pair)['replay'] == 'shared_e2_natural_same_endpoint_branch0_v1'
    rolling = dict(pair, v41_joint_replay='previous_epoch', v41_fixed_replay_bank=None)
    assert config_identity(pair) != config_identity(rolling)
    old = load_config(CONFIGS / '41_full_kitti_q0_lateral_replay.yaml')
    assert old.v41_q0_lateral and old.v41_q0_lateral_replay and not old.v41_joint_search
    with pytest.raises(ValueError, match='prefixes require'):
        load_config(CONFIGS / '41_full_kitti_joint_training.yaml', {'v41_q0_lateral_replay': True})
