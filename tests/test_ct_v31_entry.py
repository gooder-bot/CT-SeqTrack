"""v33 入口、三配方与历史实验身份隔离。"""
import subprocess
import sys

import pytest

from models.ct_v31.config import config_identity
from models.ct_v31.entry import parse_config, resolve_run_directory


@pytest.mark.parametrize('name,arm,backend', [
    ('seqtrack_ref', 'b0', 'cfc'), ('b0', 'b0', 'cfc'),
    ('b0_late_decay', 'b0', 'cfc'), ('b0_half_lr', 'b0', 'cfc')])
def test_registered_mini_arms_have_joint_budget(name, arm, backend):
    args = ['--cfg', f'cfgs/ct_seqtrack/33_{name}_mini.yaml', '--batch_size', '16',
            '--epoch', '60', '--workers', '4', '--seed', '42',
            '--check_val_every_n_epoch', '5', '--tag', 'mini_car_seed42_60ep_bs16',
            '--accelerator', 'gpu', '--trainer_devices', '1']
    config = parse_config(args)
    assert (config.v31_arm, config.v31_temporal_backend) == (arm, backend)
    assert config.path == '/home/lishengjie/data/nuscenes-mini'
    assert config.trainer_devices == 1 and config.precision == 32
    assert config.accelerator == 'gpu'
    assert config.v31_evaluate_late3
    run_arm = 'seqtrack_ref' if name == 'seqtrack_ref' else 'b0'
    assert f'-33_{run_arm}-mini_car_seed42_60ep_bs16' in str(resolve_run_directory(config))
    assert config.experiment_family == 'ct_seqtrack_v33'


def test_backends_have_separate_resume_identity():
    from models.ct_v31.config import normalize_config
    base = parse_config(['--cfg', 'cfgs/ct_seqtrack/33_b0_mini.yaml'])
    values = [base, normalize_config(dict(base, v31_arm='full', v31_temporal_backend='cfc')),
              normalize_config(dict(base, v31_arm='full', v31_temporal_backend='gru'))]
    assert len({config_identity(value) for value in values}) == 3


def test_cli_rejects_removed_flags_and_keeps_batch_limit_types():
    base = ['--cfg', 'cfgs/ct_seqtrack/33_b0_mini.yaml']
    with pytest.raises(SystemExit):
        parse_config(base + ['--preloading'])
    with pytest.raises(ValueError, match='scratch_only'):
        parse_config(base + ['--init_checkpoint', 'old.ckpt'])
    config = parse_config(base + ['--ct_engineering_check', '--limit_train_batches', '2',
                                  '--limit_val_batches', '1.0'])
    assert type(config.limit_train_batches) is int
    assert type(config.limit_val_batches) is float


@pytest.mark.parametrize('args', [[], ['--cfg', 'cfgs/ct_seqtrack/33_b0_mini.yaml']])
def test_main_v33_help_does_not_import_legacy_dataset_sdk(args):
    result = subprocess.run([sys.executable, 'main.py', *args, '--help'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'CT-SeqTrack v33 joint' in result.stdout
    assert '--preloading' not in result.stdout


def test_entry_rejects_legacy_model_identity(tmp_path):
    config_path = tmp_path / 'legacy.yaml'
    config_path.write_text('net_model: seqtrack3d\n', encoding='utf-8')
    with pytest.raises(ValueError, match='model and experiment identity'):
        parse_config(['--cfg', str(config_path)])


@pytest.mark.parametrize('version', [31, 32])
@pytest.mark.parametrize('name', ['b0', 'full_cfc', 'full_gru'])
def test_frozen_configs_cannot_silently_run_v33(version, name):
    with pytest.raises(ValueError, match=f'reproduce frozen v{version}'):
        parse_config(['--cfg', f'cfgs/ct_seqtrack/{version}_{name}_mini.yaml'])


def test_recipe_is_part_of_resume_identity():
    base = parse_config(['--cfg', 'cfgs/ct_seqtrack/33_b0_mini.yaml', '--ct_engineering_check'])
    for key, value in (('v32_reserve_windows', 0), ('v32_seed_translation', .2),
                       ('v32_seed_yaw_degrees', 1.), ('seed', 52)):
        changed = dict(base, **{key: value})
        assert config_identity(changed) != config_identity(base)


def test_v33_formal_recipe_rejects_unregistered_perturbation():
    from models.ct_v31.config import normalize_config
    with pytest.raises(ValueError, match='formal v33 budget mismatch'):
        normalize_config({'v32_seed_translation': .5})


def test_reference_has_own_model_source_and_seed_identity():
    b0 = parse_config(['--cfg', 'cfgs/ct_seqtrack/33_b0_mini.yaml'])
    reference = parse_config(['--cfg', 'cfgs/ct_seqtrack/33_seqtrack_ref_mini.yaml'])
    assert reference.net_model == 'seqtrack_reference' and reference.v31_arm == 'b0'
    assert '-33_seqtrack_ref-' in str(resolve_run_directory(reference))
    assert config_identity(reference) != config_identity(b0)
    assert config_identity(reference) != config_identity(dict(reference, seed=52))


def test_old_checkpoint_schema_is_rejected_before_weight_loading():
    from models.ct_v31.runtime import validate_resume_payload
    with pytest.raises(ValueError, match='runtime schema'):
        validate_resume_payload({'schema': 'ct_seqtrack.v31.epoch_boundary.v1'}, {}, training=False)
