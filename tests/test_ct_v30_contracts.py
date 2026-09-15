"""v30 方法、配置和恢复身份；CPU-only，不运行训练或连接服务器。"""

import ast
import copy
from pathlib import Path

import pytest

from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.online_contract import (
    build_online_resume_contract, validate_online_resume_contract,
    validate_scratch_training_contract, validate_v28_evaluation_checkpoint,
    validate_v28_observation_updates)
from utils.v29_diagnostics import assert_h3_diagnostic_contract, H1OnlyLabels
from utils.v30_contracts import V30_CONTRACTS, V30_IDENTITY_FIELDS, v30_dataloader_kwargs


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIGS = sorted(path for path in (ROOT / 'cfgs/ct_seqtrack').glob('30_*.yaml')
                     if not path.name.endswith('base.yaml'))


def config(name='30_full_cfc_mini.yaml'):
    return configure_ct_variant(load_yaml_config(ROOT / 'cfgs/ct_seqtrack' / name))


@pytest.mark.parametrize('path', RUN_CONFIGS, ids=lambda path: path.stem)
def test_registered_v30_configs_are_complete_and_keep_budget(path):
    cfg = configure_ct_variant(load_yaml_config(path))
    validate_scratch_training_contract(cfg)
    assert (cfg['epoch'], cfg['batch_size'], cfg['workers'], cfg['seed']) == (60, 16, 4, 42)
    assert cfg['ct_expansion_point_count'] == 768
    assert cfg['ct_relation_topk'] + cfg['ct_relation_coverage_count'] + cfg['ct_relation_exploration_count'] == 256
    assert cfg['ct_training_state_policy'] == 'mixed_accepted_v30'
    assert not cfg['preloading'] and not cfg['ct_separate_optimizers']
    if cfg['ct_variant'] == 'b0':
        assert not any(cfg['ct_enable_' + part] for part in ('b1', 'b2', 'b3'))
    elif cfg['ct_variant'] == 'full_minus_b3':
        assert cfg['ct_enable_b1'] and cfg['ct_enable_b2'] and not cfg['ct_enable_b3']
        assert cfg['proposal_inference_mode'] == 'bounded_always'
    else:
        assert all(cfg['ct_enable_' + part] for part in ('b1', 'b2', 'b3'))
    identity = build_online_resume_contract(cfg)
    assert identity['schema'] == 'ct_seqtrack.online_resume_contract.v30'
    for field in V30_IDENTITY_FIELDS:
        assert identity['fields'][field] == cfg.get(field)


@pytest.mark.parametrize('field', list(V30_CONTRACTS))
def test_different_method_strings_fail_instead_of_silent_normalization(field):
    cfg = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/30_full_cfc_mini.yaml')
    cfg[field] = 'incorrect-contract'
    with pytest.raises(ValueError, match='v30 contract mismatch'):
        configure_ct_variant(cfg)


def test_v30_bounds_and_state_are_not_rewritten_by_legacy_variant_configuration():
    cfg = config()
    assert [cfg['search_v3_fixed_margin_parallel'], cfg['search_v3_fixed_margin_perpendicular']] == [.75, .5]
    old = configure_ct_variant(load_yaml_config(ROOT / 'cfgs/ct_seqtrack/29_full_cfc_nuscenes_full.yaml'))
    assert [old['search_v3_fixed_margin_parallel'], old['search_v3_fixed_margin_perpendicular']] == [2., 1.]
    assert old['ct_training_state_policy'] == 'mixed_accepted_v1'
    validate_scratch_training_contract(old)
    legacy = config('30_ablate_legacy_acquisition_mini.yaml')
    assert legacy['ct_acquisition_margin_min'] == [2., 1.]
    assert legacy['ct_acquisition_margin_max'] == [6., 3.]
    assert legacy['ct_acquisition_margin_initial'] == pytest.approx([2.0398072074676173, 1.0199036037338087])


def test_checkpoint_cannot_cross_v29_v30_or_mode_and_data_identity():
    cfg = config()
    old = configure_ct_variant(load_yaml_config(ROOT / 'cfgs/ct_seqtrack/29_full_cfc_nuscenes_full.yaml'))
    checkpoint = dict(hyper_parameters={'config': copy.deepcopy(cfg)},
                      ct_online_resume_contract=build_online_resume_contract(cfg),
                      ct_epoch_boundary_complete=True,
                      ct_global_rng_state={'schema': 'ct_seqtrack.global_rng.v1'})
    validate_online_resume_contract(checkpoint, cfg)
    for key, value in [('ct_mode_count', 1), ('ct_frame_stride', 2),
                       ('ct_b0_long_rollin_enabled', False), ('ct_mode_quality_weight', 0.),
                       ('ct_dataset_manifest_sha256', 'changed')]:
        different = dict(cfg, **{key: value})
        with pytest.raises(ValueError, match='online resume contract mismatch'):
            validate_online_resume_contract(checkpoint, different)
    with pytest.raises(ValueError, match='does not carry'):
        validate_online_resume_contract(checkpoint, old)
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_v28_evaluation_checkpoint({'hyper_parameters': {'config': old}}, cfg)


def test_diagnostics_accept_v30_h1_aux_but_cannot_read_h3_labels():
    assert_h3_diagnostic_contract(config())
    labels = H1OnlyLabels({'target': 1, 'ct_h3_valid': 1})
    assert labels['target'] == 1
    with pytest.raises(RuntimeError, match='diagnostic H3'):
        _ = labels['ct_h3_valid']


def test_natural_loader_length_and_prefetch_do_not_create_extra_updates():
    cfg = config()
    # New contract follows complete selected rows, including sparse/empty rows.
    audit = validate_v28_observation_updates(cfg, sample_count=101, batch_size=16,
                                              drop_last=True, observed_updates=6)
    assert audit['observed_updates_per_epoch'] == 6
    with pytest.raises(ValueError, match='loader length'):
        validate_v28_observation_updates(cfg, sample_count=101, batch_size=16,
                                          drop_last=True, observed_updates=7)
    assert v30_dataloader_kwargs(cfg) == {'prefetch_factor': 2, 'persistent_workers': False}
    assert v30_dataloader_kwargs(dict(cfg, workers=0)) == {}
    assert v30_dataloader_kwargs({'ct_enable_v29': True, 'workers': 4}) == {}


def test_main_loaders_all_dispatch_registered_v30_execution_options():
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8-sig'))
    stride = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and any(isinstance(arg, ast.Constant) and arg.value == '--ct_frame_stride' for arg in node.args)]
    assert len(stride) == 1
    assert any(item.arg == 'type' and isinstance(item.value, ast.Name) and item.value.id == 'int'
               for item in stride[0].keywords)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'DataLoader']
    assert len(calls) == 6
    for call in calls:
        assert any(keyword.arg is None and isinstance(keyword.value, ast.Call)
                   and isinstance(keyword.value.func, ast.Name)
                   and keyword.value.func.id == 'v30_dataloader_kwargs' for keyword in call.keywords)


def test_engineering_override_is_bounded_and_keeps_formal_method_contract():
    cfg = dict(config(), ct_engineering_check=True, epoch=1, workers=0,
               limit_train_batches=8, limit_val_batches=1, check_val_every_n_epoch=1,
               log_dir=str(ROOT / 'artifacts/ct_checks/v30_bounded_test'))
    validate_scratch_training_contract(cfg)
    for key, value in [('ct_engineering_check', False), ('epoch', 4),
                       ('workers', -1), ('limit_train_batches', 101),
                       ('limit_train_batches', 1.), ('limit_val_batches', 0),
                       ('check_val_every_n_epoch', 5), ('batch_size', 1),
                       ('ct_relation_topk', 256), ('init_checkpoint', 'other-run.ckpt'),
                       ('log_dir', str(ROOT / 'output/v30_smoke')),
                       ('log_dir', str(ROOT / 'artifacts/ct_checks')),
                       ('log_dir', str(ROOT / 'artifacts/ct_checks/../escaped'))]:
        with pytest.raises(ValueError, match='invalid v30 scratch contract'):
            validate_scratch_training_contract(dict(cfg, **{key: value}))
