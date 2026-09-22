"""唯一入口运行 v31，禁止旧标定/插件配置混入。"""
import subprocess
import sys

import pytest

from models.ct_v31.config import config_identity
from models.ct_v31.entry import parse_config, resolve_run_directory


@pytest.mark.parametrize('name,arm,backend', [
    ('b0', 'b0', 'cfc'), ('full_cfc', 'full', 'cfc'), ('full_gru', 'full', 'gru')])
def test_registered_mini_arms_have_joint_budget(name, arm, backend):
    args = ['--cfg', f'cfgs/ct_seqtrack/31_{name}_mini.yaml', '--batch_size', '16',
            '--epoch', '60', '--workers', '4', '--seed', '42',
            '--check_val_every_n_epoch', '5', '--tag', 'mini_car_seed42_60ep_bs16']
    config = parse_config(args)
    assert (config.v31_arm, config.v31_temporal_backend) == (arm, backend)
    assert config.path == '/home/lishengjie/data/nuscenes-mini'
    assert config.trainer_devices == 1 and config.precision == 32
    assert config.v31_evaluate_late3
    assert f'-31_{name}-mini_car_seed42_60ep_bs16' in str(resolve_run_directory(config))


def test_backends_have_separate_resume_identity():
    values = [parse_config(['--cfg', f'cfgs/ct_seqtrack/31_{name}_mini.yaml'])
              for name in ('b0', 'full_cfc', 'full_gru')]
    assert len({config_identity(value) for value in values}) == 3


def test_cli_rejects_removed_flags_and_keeps_batch_limit_types():
    base = ['--cfg', 'cfgs/ct_seqtrack/31_b0_mini.yaml']
    with pytest.raises(SystemExit):
        parse_config(base + ['--preloading'])
    with pytest.raises(ValueError, match='scratch_only'):
        parse_config(base + ['--init_checkpoint', 'old.ckpt'])
    config = parse_config(base + ['--ct_engineering_check', '--limit_train_batches', '2',
                                  '--limit_val_batches', '1.0'])
    assert type(config.limit_train_batches) is int
    assert type(config.limit_val_batches) is float


@pytest.mark.parametrize('args', [[], ['--cfg', 'cfgs/ct_seqtrack/31_full_gru_mini.yaml']])
def test_main_v31_help_does_not_import_legacy_dataset_sdk(args):
    result = subprocess.run([sys.executable, 'main.py', *args, '--help'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'CT-SeqTrack v31 joint' in result.stdout
    assert '--preloading' not in result.stdout


def test_entry_rejects_legacy_model_identity(tmp_path):
    config_path = tmp_path / 'legacy.yaml'
    config_path.write_text('net_model: seqtrack3d\n', encoding='utf-8')
    with pytest.raises(ValueError, match='model and experiment identity'):
        parse_config(['--cfg', str(config_path)])
