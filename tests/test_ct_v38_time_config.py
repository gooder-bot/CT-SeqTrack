"""工程时间 manifest 的真实 role 选择、路径迁移身份及完整 epoch 边界。"""
from pathlib import Path

import pytest

from datasets.protocol_utils import resolve_dynamics_time_kwargs
from models.ct_v38.config import normalize_config, config_identity
from models.ct_v38.entry import parse_config
from models.seqtrack_reference.config import normalize_config as normalize_reference
from utils.time_config import TIME_MANIFEST_KEYS


def manifest(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding='utf-8')
    return str(path)


def test_shuffled_train_and_eval_paths_resolve_and_hash_contents(tmp_path):
    train = manifest(tmp_path, 'train.json', 'train manifest bytes')
    evaluation = manifest(tmp_path, 'test.json', 'test manifest bytes')
    values = dict(ct_engineering_check=True, dynamics_time_mode='shuffled',
        train_dynamics_time_manifest=train, eval_dynamics_time_manifest=evaluation)
    cfg = normalize_config(values)
    assert resolve_dynamics_time_kwargs(cfg, 'train')['dynamics_time_manifest'] == train
    assert resolve_dynamics_time_kwargs(cfg, 'test')['dynamics_time_manifest'] == evaluation
    identity = config_identity(cfg)
    moved = dict(values, train_dynamics_time_manifest=manifest(tmp_path, 'moved_train.json', 'train manifest bytes'),
                 eval_dynamics_time_manifest=manifest(tmp_path, 'moved_test.json', 'test manifest bytes'))
    assert config_identity(moved) == identity
    Path(train).write_text('different training assignment', encoding='utf-8')
    assert config_identity(cfg) != identity


@pytest.mark.parametrize('key', TIME_MANIFEST_KEYS)
def test_all_declared_manifest_aliases_bind_bytes_without_binding_path(tmp_path, key):
    original = manifest(tmp_path, 'base.json', 'base')
    specific = manifest(tmp_path, 'specific.json', 'role')
    cfg = dict(ct_engineering_check=True, dynamics_time_mode='shuffled', dynamics_time_manifest=original)
    cfg[key] = specific
    identity = config_identity(cfg)
    moved = dict(cfg)
    moved[key] = manifest(tmp_path, 'moved.json', 'role')
    assert config_identity(moved) == identity
    Path(specific).write_text('role changed', encoding='utf-8')
    assert config_identity(cfg) != identity


def test_actual_dataset_roles_and_existing_resolver_precedence(tmp_path):
    train = manifest(tmp_path, 'train.json', 'train')
    evaluation = manifest(tmp_path, 'evaluation.json', 'eval')
    preferred = manifest(tmp_path, 'preferred.json', 'role')
    cfg = normalize_config(dict(ct_engineering_check=True, dynamics_time_mode='shuffled',
        train_dynamics_time_manifest=train, dynamics_time_manifest_train=preferred,
        eval_dynamics_time_manifest=evaluation))
    assert resolve_dynamics_time_kwargs(cfg, 'train')['dynamics_time_manifest'] == preferred
    with pytest.raises(ValueError, match='test'):
        normalize_config(dict(ct_engineering_check=True, dynamics_time_mode='shuffled',
            train_dynamics_time_manifest=train, val_dynamics_time_manifest=evaluation))
    kitti = normalize_config(dict(ct_engineering_check=True, dataset='kitti_mf',
        version='kitti_tracking', ct_coordinate_mode='sensor_relative', dynamics_time_mode='shuffled',
        train_dynamics_time_manifest=train, eval_dynamics_time_manifest=evaluation))
    for role in ('dev', 'test'):
        assert resolve_dynamics_time_kwargs(kitti, role)['dynamics_time_manifest'] == evaluation


def test_fixed_time_engineering_override_and_formal_guard():
    cfg = normalize_config(dict(ct_engineering_check=True, dynamics_time_mode='fixed',
        dynamics_fixed_delta_t=.1))
    assert resolve_dynamics_time_kwargs(cfg, 'test')['dynamics_fixed_delta_t'] == .1
    assert config_identity(cfg) != config_identity(dict(cfg, dynamics_fixed_delta_t=.5))
    for invalid in (0, -1, float('inf'), True, '0.1'):
        with pytest.raises(ValueError, match='positive finite'):
            normalize_config(dict(cfg, dynamics_fixed_delta_t=invalid))
    for change in (dict(train_dynamics_time_manifest='x'), dict(dynamics_fixed_delta_t=.1)):
        with pytest.raises(ValueError, match='ct_engineering_check'):
            normalize_config(change)


@pytest.mark.parametrize('normalizer,late_key', [(normalize_config, 'v31_evaluate_late3'),
                                               (normalize_reference, 'evaluate_late3')])
@pytest.mark.parametrize('limit', [1, .5])
def test_partial_training_must_disable_automatic_complete_epoch_evaluation(normalizer, late_key, limit):
    values = dict(ct_engineering_check=True, limit_train_batches=limit, **{late_key: True})
    with pytest.raises(ValueError, match='--no_late3'):
        normalizer(values)
    assert normalizer(dict(values, **{late_key: False})).limit_train_batches == limit
    assert normalizer(dict(values, limit_train_batches=1.)).limit_train_batches == 1.
    assert normalizer(dict(values, test=True, checkpoint='same_run.ckpt')).test


def test_reference_keeps_original_pseudo_time_protocol():
    for change in (dict(dynamics_time_mode='shuffled'), dict(dynamics_time_mode='fixed'),
                   dict(train_dynamics_time_manifest='train.json'), dict(dynamics_fixed_delta_t=.1)):
        with pytest.raises(ValueError, match='pseudo-time'):
            normalize_reference(change)


def test_disabled_validation_and_no_late3_need_only_training_manifest(tmp_path):
    path = manifest(tmp_path, 'train.json', 'train')
    cfg = normalize_config(dict(ct_engineering_check=True, dynamics_time_mode='shuffled',
        train_dynamics_time_manifest=path, limit_val_batches=0, v31_evaluate_late3=False))
    assert cfg.limit_val_batches == 0
    assert normalize_reference(dict(limit_val_batches=0)).limit_val_batches == 0


def test_engineering_cli_exposes_role_manifests_and_fixed_interval(tmp_path):
    train = manifest(tmp_path, 'train.json', 'train')
    evaluation = manifest(tmp_path, 'evaluation.json', 'eval')
    common = ['--cfg', 'cfgs/ct_seqtrack/38_full_standard_mini.yaml', '--ct_engineering_check']
    cfg = parse_config(common + ['--dynamics_time_mode', 'shuffled',
        '--train_dynamics_time_manifest', train, '--eval_dynamics_time_manifest', evaluation])
    assert resolve_dynamics_time_kwargs(cfg, 'test')['dynamics_time_manifest'] == evaluation
    fixed = parse_config(common + ['--dynamics_time_mode', 'fixed', '--dynamics_fixed_delta_t', '.1'])
    assert fixed.dynamics_fixed_delta_t == .1
