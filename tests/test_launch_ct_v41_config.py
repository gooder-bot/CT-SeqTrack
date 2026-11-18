"""V41 运行工具读取所选配方；启动器只打印命令。"""
from datetime import datetime
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from tools.launch_ct_v41 import (_ct_arguments, baseline_evaluation_commands,
                                launch_commands, parse_args, tail_commands)


def test_smoke_uses_selected_full_config_and_keeps_engineering_overrides(monkeypatch):
    from tools import check_v41_server as smoke
    directory = Path(__file__).resolve().parents[1] / 'cfgs/ct_seqtrack'
    monkeypatch.chdir(directory)
    for filename, options, weighting in (
            ('41_full_kitti.yaml', [], 'uniform'),
            ('41_full_kitti_q0_inverse_lwh.yaml', ['--cfg', '41_full_kitti_q0_inverse_lwh.yaml'], 'inverse_lwh')):
        args = smoke.parse_args(['--path', 'kitti', '--devices', '1', *options])
        assert Path(args.cfg) == directory / filename
        config = smoke.smoke_config(args)
        assert config.cfg == args.cfg and config.v41_q0_center_weighting == weighting
        assert config.path == str(directory / 'kitti') and config.devices == 1
        assert config.v41_arm == 'full' and config.v41_attention_backend == 'fa2_varlen'
        assert config.ct_engineering_check and config.workers == 0 and config.log_dir is None


def test_default_q0_arms_have_distinct_launch_and_tail_labels():
    b_yaml = 'cfgs/ct_seqtrack/41_full_kitti_q0_inverse_lwh.yaml'
    a = parse_args(['--arm', 'full'])
    b = parse_args(['--arm', 'full', '--cfg', b_yaml])
    tagged = parse_args(['--arm', 'full', '--cfg', b_yaml, '--tag', 'my_run'])
    a_label = 'v41_full_kitti_car_seed42-seed42-60ep-bs16'
    b_label = 'v41_full_kitti_car_seed42-q0-inverse-lwh-seed42-60ep-bs16'
    for commands in (launch_commands, tail_commands):
        assert a_label in commands(a, 'full') and b_label not in commands(a, 'full')
        assert b_label in commands(b, 'full') and a_label not in commands(b, 'full')
        assert 'my_run-seed42-60ep-bs16' in commands(tagged, 'full')


def test_direct_run_directories_distinguish_q0_arms_at_same_timestamp(tmp_path, monkeypatch):
    from models.ct_v41 import entry
    directory = Path(__file__).resolve().parents[1] / 'cfgs/ct_seqtrack'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(entry, 'datetime', SimpleNamespace(now=lambda: datetime(2026, 1, 2, 3, 4, 5)))

    def destination(filename, *options):
        argv = ['--cfg', str(directory / filename), *options]
        return entry.resolve_run_directory(entry.parse_config(argv), command_line=argv)

    a_yaml, b_yaml = '41_full_kitti.yaml', '41_full_kitti_q0_inverse_lwh.yaml'
    stem = '20260102-030405-41_v3_full-'
    default_tag = 'v41_full_kitti_car_seed42'
    a = destination(a_yaml)
    assert a == tmp_path / 'output' / (stem + default_tag)
    assert destination(b_yaml) == tmp_path / 'output' / (a.name + '-q0-inverse-lwh')
    assert destination(b_yaml, '--log_dir', 'custom') == tmp_path / 'custom'
    assert destination(b_yaml, '--tag', 'mine') == tmp_path / 'output' / (stem + 'mine')
    assert destination(b_yaml, '--tag=' + default_tag) == a


def test_launcher_uses_yaml_recipe_in_label_without_overwriting_it(tmp_path):
    path = tmp_path / 'run.yaml'
    path.write_text('tag: custom_run\nseed: 7\nepoch: 12\nbatch_size: 8\nworkers: 0\n',
                    encoding='utf-8')
    args = parse_args(['--arm', 'full', '--cfg', str(path)])
    command = launch_commands(args, 'full')
    assert 'custom_run-seed7-12ep-bs8' in command
    assert 'custom_run-seed7-12ep-bs8' in tail_commands(args, 'full')
    assert 'PYTHONHASHSEED=7' in command
    assert '--batch_size' not in command and '--workers' not in command
    assert '--epoch' not in command and '--check_val_every_n_epoch' not in command


def test_launcher_forwards_named_overrides_booleans_and_empty_lists(tmp_path):
    path = tmp_path / 'run.yaml'
    path.write_text('tag: yaml_tag\nseed: 7\nlr_milestones: [20, 40]\n', encoding='utf-8')
    args = parse_args(['--arm', 'full', '--cfg', str(path), '--seed', '9',
        '--tag', 'cli tag', '--batch_size', '4', '--epoch', '15', '--workers', '0',
        '--lr_schedule', 'step', '--lr_milestones', '--lr_decay_step', '5',
        '--no-tensorboard', '--v41_memory_logging', '--full_diagnostic_epochs',
        '--console_mode', 'plain'])
    tokens = shlex.split(_ct_arguments(args))
    assert tokens[tokens.index('--tag') + 1] == 'cli tag'
    assert '--no-tensorboard' in tokens and '--v41_memory_logging' in tokens
    assert tokens[tokens.index('--lr_milestones') + 1] == '--lr_decay_step'
    assert tokens[tokens.index('--full_diagnostic_epochs') + 1] == '--no-tensorboard'
    command = launch_commands(args, 'full')
    assert 'cli_tag-seed9-15ep-bs4' in command and 'PYTHONHASHSEED=9' in command
    assert '--workers 0' in command and '--console_mode plain' in command


def test_baseline_command_keeps_its_existing_recipe_and_default_path():
    args = parse_args(['--arm', 'seqtrack'])
    command = launch_commands(args, 'seqtrack')
    assert '--batch_size 16 --epoch 60 --workers 10 --seed 42' in command
    assert 'CUDA_VISIBLE_DEVICES=1' in command
    assert 'PYTORCH_CUDA_ALLOC_CONF=backend:native ' in command
    assert '${PYTORCH_CUDA_ALLOC_CONF' not in command
    assert '--diagnostics_level' not in command
    assert '--path /home/lishengjie/data/cxtrack/training' in baseline_evaluation_commands(args)


def test_ct_command_preserves_ambient_allocator_with_shell_default():
    args = parse_args(['--arm', 'full'])
    command = launch_commands(args, 'full')
    assert ('PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-'
            'backend:native,roundup_power2_divisions:2}" ') in command


@pytest.mark.parametrize('option', ['--set', '--batch-size', '--bat'])
def test_launcher_does_not_add_generic_or_abbreviated_interfaces(option):
    with pytest.raises(SystemExit):
        parse_args([option, '8'])
