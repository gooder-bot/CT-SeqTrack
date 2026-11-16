"""启动器只打印命令，并保持具名 CLI > YAML > 默认值。"""
import shlex

import pytest

from tools.launch_ct_v41 import (_ct_arguments, baseline_evaluation_commands,
                                launch_commands, parse_args, tail_commands)


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
