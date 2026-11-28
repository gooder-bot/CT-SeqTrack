"""V41 v4 新运行配置与恢复身份边界。"""
import json
from pathlib import Path

import pytest

from models.ct_v41.config import config_identity, load_config, normalize_config, recipe_identity


def test_new_training_recipe_does_not_require_engineering_mode():
    cfg = normalize_config(dict(epoch=12, batch_size=8, seed=7, workers=0,
        lr=8e-5, wd=.001, lr_milestones=[4, 9], lr_decay_rate=.5,
        lr_warmup_steps=20, check_val_every_n_epoch=2, ct_pointcloud_cache_bytes=0))
    assert not cfg.ct_engineering_check
    assert (cfg.epoch, cfg.batch_size, cfg.seed, cfg.workers) == (12, 8, 7, 0)
    assert cfg.lr == 8e-5 and cfg.wd == .001
    assert cfg.check_val_every_n_epoch == 2 and cfg.ct_pointcloud_cache_bytes == 0
    identity = recipe_identity(cfg)
    assert identity['lr_milestones'] == [4, 9]
    assert identity['optimization']['lr'] == 8e-5
    assert identity['optimization']['lr_warmup_steps'] == 20
    assert cfg.v41_execution_revision == identity['execution_revision'] == 'v4'
    assert identity['cpu_geometry']


def test_yaml_inheritance_and_explicit_overrides_keep_unsupplied_values(tmp_path):
    (tmp_path / 'base.yaml').write_text(
        'batch_size: 8\nseed: 7\nlr: 0.00008\nconsole_mode: bar\n', encoding='utf-8')
    child = tmp_path / 'run.yaml'
    child.write_text('_base_: base.yaml\nworkers: 0\nepoch: 12\n', encoding='utf-8')
    cfg = load_config(child, dict(seed=9))
    assert (cfg.batch_size, cfg.seed, cfg.workers, cfg.epoch) == (8, 9, 0, 12)
    assert cfg.lr == 8e-5 and cfg.console_mode == 'bar'
    assert cfg.diagnostics_level == 'summary' and not cfg.v41_memory_logging
    assert cfg.full_diagnostic_epochs == []


def test_existing_step_and_piecewise_schedules_remain_usable():
    step = normalize_config(dict(lr_schedule='step', lr_decay_step=7))
    assert step.lr_milestones == []
    assert normalize_config(step) == step
    piecewise = normalize_config(dict(lr=.0001, lr_schedule='piecewise',
        lr_milestones=[5, 9], lr_stage_values=[.0001, .00005, .00001]))
    assert recipe_identity(piecewise)['optimization']['lr_stage_values'] == [.0001, .00005, .00001]
    with pytest.raises(ValueError, match='step uses no lr_milestones'):
        normalize_config(dict(lr_schedule='step', lr_milestones=[20]))


def test_inverse_lwh_recipe_changes_only_q0_center_weighting():
    directory = Path(__file__).resolve().parents[1] / 'cfgs' / 'ct_seqtrack'
    uniform = load_config(directory / '41_full_kitti.yaml')
    inverse = load_config(directory / '41_full_kitti_q0_inverse_lwh.yaml')
    assert uniform.v41_q0_center_weighting == 'uniform'
    assert inverse == dict(uniform, cfg=inverse.cfg, v41_q0_center_weighting='inverse_lwh')
    assert recipe_identity(uniform)['q0_center_weighting'] == 'uniform'
    assert recipe_identity(inverse) == dict(recipe_identity(uniform), q0_center_weighting='inverse_lwh')
    assert config_identity(uniform) != config_identity(inverse)


def test_recovery_execution_record_distinguishes_training_and_inference():
    for recipe, joint, train_r, infer_r in (
        ('b0', False, False, False), ('full', False, True, False), ('full', True, True, True),
    ):
        cfg = normalize_config(dict(v41_training_recipe=recipe, v41_joint_search=joint,
                                    ct_engineering_check=recipe == 'b0'))
        recovery = recipe_identity(cfg)['recovery']
        assert (recovery['training_enabled'], recovery['inference_enabled']) == (train_r, infer_r)


def test_support_center_has_one_four_slot_recipe_with_the_november_27_schedule():
    directory = Path(__file__).resolve().parents[1] / 'cfgs' / 'ct_seqtrack'
    cfg = load_config(directory / '41_support_center_kitti.yaml')
    assert cfg.v41_support_center and cfg.v41_local_vote
    assert (cfg.seed, cfg.batch_size, cfg.epoch) == (42, 16, 60)
    assert (cfg.lr, cfg.lr_schedule, cfg.lr_decay_step, cfg.lr_decay_rate, cfg.lr_milestones) == (
        .0001, 'step', 20, .1, [])
    legacy = normalize_config(dict(cfg, v41_support_center=False))
    assert config_identity(cfg) != config_identity(legacy)
    assert 'q0_center' not in recipe_identity(legacy)
    assert recipe_identity(cfg)['candidates'] == recipe_identity(legacy)['candidates']
    with pytest.raises(ValueError, match='v41_support_center'):
        normalize_config(dict(v41_support_center=True))
    with pytest.raises(ValueError, match='v41_support_center'):
        normalize_config(dict(cfg, v41_support_center='false'))


def test_resume_records_current_numeric_environment_and_preserves_initial_manifest(tmp_path):
    from models.ct_v41.entry import write_resume_event, write_run_metadata

    config = normalize_config()
    initial = dict(training_execution_recipe=recipe_identity(config), run_id='q0-run', source={'sha256': 'same-source'}, torch='2.0.1+cu118',
                   lightning='2.0.2', precision_policy='FP32 except BF16 QKV inside FA2 kernel',
                   flash_attention=dict(backend='fa2_varlen', flash_attn='2.5.8', device=0))
    write_run_metadata(config, tmp_path, initial, preserve_existing=False)
    config.checkpoint = str(tmp_path / 'epoch=020.ckpt')
    resumed = dict(initial, flash_attention=dict(initial['flash_attention'], device=1))
    event = json.loads(write_resume_event(config, tmp_path, resumed).read_text(encoding='utf-8'))
    for key in ('torch', 'lightning', 'precision_policy', 'flash_attention', 'training_execution_recipe'):
        assert event[key] == resumed[key]
    assert json.loads((tmp_path / 'run_manifest.json').read_text(encoding='utf-8')) == initial


@pytest.mark.parametrize('value', ['inverse', None, False])
def test_invalid_q0_center_weighting_is_rejected(value):
    with pytest.raises(ValueError, match='v41_q0_center_weighting'):
        normalize_config(dict(v41_q0_center_weighting=value))


def test_reporting_overrides_do_not_invalidate_independent_checkpoint_evaluation():
    training = normalize_config()
    evaluation = normalize_config(dict(training, test=True, checkpoint='same-run.ckpt',
        eval_split='dev', diagnostics_level='full', full_diagnostic_epochs=[9, 3, 9],
        console_mode='plain', progress_refresh_seconds=2., plain_log_interval_seconds=60.,
        log_every_n_steps=100, tensorboard=True, v41_memory_logging=True,
        v35_train_diagnostics=False, v41_diagnostics=False, v31_evaluate_late3=False,
        checkpoint_every_n_epochs=2, report_checkpoint='last', report_last_k=5))
    assert evaluation.full_diagnostic_epochs == [3, 9]
    assert config_identity(evaluation) == config_identity(training)
    assert config_identity(normalize_config(evaluation)) == config_identity(evaluation)
    assert config_identity(normalize_config(training)) == config_identity(training)


@pytest.mark.parametrize('change', [dict(lr=1e-4), dict(epoch=59), dict(batch_size=8),
    dict(seed=52), dict(lr_milestones=[15, 30]), dict(wd=.01)])
def test_training_changes_still_change_resume_identity(change):
    assert config_identity(normalize_config(change)) != config_identity(normalize_config())


@pytest.mark.parametrize('change', [dict(v41_trust_threshold=.7), dict(v41_m_weight=.1),
    dict(v41_aux_loss_weight=.5), dict(v31_long_window=16), dict(point_sample_size=2048),
    dict(v41_execution_revision='v1'), dict(v41_execution_revision='v2'), dict(v41_execution_revision='v3')])
def test_algorithm_and_old_execution_revision_are_not_silently_opened(change):
    with pytest.raises(ValueError):
        normalize_config(change)


@pytest.mark.parametrize('change', [dict(console_mode='progress'), dict(diagnostics_level='off'),
    dict(report_checkpoint='best_success'), dict(full_diagnostic_epochs=[0]),
    dict(progress_refresh_seconds=0), dict(report_last_k=True), dict(tensorboard='false')])
def test_invalid_reporting_inputs_are_rejected(change):
    with pytest.raises(ValueError):
        normalize_config(change)


def test_legacy_late3_flag_no_longer_blocks_partial_engineering_runs():
    cfg = normalize_config(dict(ct_engineering_check=True, limit_train_batches=2))
    assert cfg.limit_train_batches == 2 and cfg.v31_evaluate_late3
