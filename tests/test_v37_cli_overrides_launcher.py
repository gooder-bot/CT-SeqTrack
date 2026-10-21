"""启动器参数只经 argv 传递；所有 shell 测试在临时目录使用假的训练解释器。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / 'tools' / 'launch_v37_centroid.sh'
RECIPE = 'full_mass_detach'
CFG = f'cfgs/ct_seqtrack/37_{RECIPE}_mini.yaml'


@pytest.fixture
def bash():
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    executable = str(git_bash) if git_bash.is_file() else shutil.which('bash')
    if executable is None:
        pytest.skip('launcher shell tests require Bash')
    return executable


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    path.chmod(0o755)


@pytest.fixture
def workspace(tmp_path, bash):
    # 不导入项目生产模型，不允许真正执行 main.py。
    _write(tmp_path / 'main.py', 'raise AssertionError("real training must never run")\n')
    _write(tmp_path / CFG, json.dumps({'v37_centroid_recipe': RECIPE,
                                      'path': 'default YAML data'}))
    _write(tmp_path / 'models/__init__.py', '')
    _write(tmp_path / 'models/ct_v37/__init__.py', '')
    _write(tmp_path / 'models/ct_v37/entry.py', '''import argparse
import json
import os
from pathlib import Path

def parse_config(argv):
    cfg_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    cfg_parser.add_argument('--cfg', required=True)
    cfg_args, _ = cfg_parser.parse_known_args(argv)
    defaults = json.loads(Path(cfg_args.cfg).read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--path', default=defaults.get('path'))
    parser.add_argument('--devices', type=int)
    parser.add_argument('--accelerator')
    parser.add_argument('--trainer_devices', type=int)
    for key, default in [('batch_size', 16), ('epoch', 60), ('seed', 42), ('workers', 4),
                         ('check_val_every_n_epoch', 1)]:
        parser.add_argument('--' + key, type=int, default=default)
    parser.add_argument('--lr', type=float, default=.00015)
    for key in ('tag', 'log_dir', 'checkpoint', 'init_checkpoint'):
        parser.add_argument('--' + key)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ct_engineering_check', action='store_true')
    config = parser.parse_args(argv)
    config.v37_centroid_recipe = os.environ.get('FAKE_RECIPE',
        defaults['v37_centroid_recipe'])
    Path(os.environ['CAPTURE_PARSED']).write_text(json.dumps(vars(config)), encoding='utf-8')
    return config

def validate_formal_registration(config):
    if os.environ.get('FAIL_PREFLIGHT'):
        raise ValueError('injected registration/source failure')
    print('registration override notice')
    return {'label': config.v37_centroid_recipe, 'effective_batch_size': config.batch_size}
''')
    bin_dir = tmp_path / 'fake bin'
    fake_python = bin_dir / 'fake-python'
    _write(fake_python, '''#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -c ]]; then
  printf '%s\\0' "$@" >"$CAPTURE_PREFLIGHT"
  exec "$REAL_PYTHON" "$@"
fi
if [[ "${1:-}" == -u && "${2:-}" == main.py ]]; then
  printf '%s\\0' "$@" >"$CAPTURE_MAIN"
  printf 'fake training invocation only\\n'
  exit 0
fi
echo 'unexpected fake Python invocation' >&2
exit 98
''')
    _write(bin_dir / 'nohup', '''#!/usr/bin/env bash
set -euo pipefail
printf '%s\\0' "$@" >"$CAPTURE_NOHUP"
exec "$@"
''')
    # Git Bash 会将自身工具目录置于 Windows PATH 前面；在 shell 初始化后放回假的 nohup。
    shell_env = tmp_path / 'fake shell environment'
    _write(shell_env, 'export PATH="$CT_FAKE_BIN:$PATH"\n')
    shell_bin = bin_dir.as_posix()
    if os.name == 'nt':
        shell_bin = '/' + shell_bin[0].lower() + shell_bin[2:]
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ.get('PATH', ''),
        CT_PYTHON=fake_python.as_posix(),
        CT_FAKE_BIN=shell_bin, BASH_ENV=shell_env.as_posix(),
        REAL_PYTHON=Path(sys.executable).as_posix(), MSYS2_ARG_CONV_EXCL='*')
    env.pop('CT_DATA_ROOT', None)
    for key in ('PREFLIGHT', 'MAIN', 'NOHUP', 'PARSED'):
        env['CAPTURE_' + key] = (tmp_path / ('capture_' + key.lower())).as_posix()
    return tmp_path, env


def _run(bash, workspace, extras=(), **environment):
    directory, env = workspace
    return subprocess.run([bash, LAUNCHER.as_posix(), RECIPE, '1', *extras],
        cwd=directory, env=dict(env, **environment), capture_output=True, text=True,
        encoding='utf-8', timeout=15)


def _arguments(path):
    return [part.decode('utf-8') for part in path.read_bytes().split(b'\0')[:-1]]


def _wait_for(path):
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), 'fake background command did not record argv'


def test_launcher_has_valid_bash_syntax(bash):
    result = subprocess.run([bash, '-n', LAUNCHER.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_overrides_reach_preflight_and_main_literally_and_name_actual_run(bash, workspace):
    directory, env = workspace
    overridden_path = (directory / 'data with spaces $(touch path_injected)').as_posix()
    literal_tag = 'literal $(touch shell_injected); `touch backtick_injected`'
    extras = ['--batch_size', '8', '--epoch=12', '--seed', '52', '--workers', '0',
              '--check_val_every_n_epoch', '3', '--lr', '.0002',
              '--devices=2', '--path', overridden_path, '--tag', literal_tag]
    result = _run(bash, workspace, extras)
    assert result.returncode == 0, result.stderr
    lines = dict(line.split('=', 1) for line in result.stdout.splitlines())
    run = lines['RUN']
    assert f'-37_{RECIPE}-mini_car_seed52_12ep_bs8-' in run
    _wait_for(directory / 'capture_main')
    common = ['--cfg', CFG, '--devices', '1',
              '--accelerator', 'gpu', '--trainer_devices', '1']
    preflight = _arguments(directory / 'capture_preflight')
    assert preflight[0] == '-c' and preflight[2:] == [RECIPE, *common, *extras]
    assert _arguments(directory / 'capture_main') == [
        '-u', 'main.py', *common, *extras, '--log_dir', run]
    parsed = json.loads((directory / 'capture_parsed').read_text(encoding='utf-8'))
    assert parsed['path'] == overridden_path and parsed['tag'] == literal_tag
    assert (parsed['workers'], parsed['check_val_every_n_epoch'], parsed['lr']) == (0, 3, .0002)
    assert 'registration override notice' in result.stderr
    assert 'registration override notice' not in result.stdout
    assert (directory / run / 'train.pid').read_text().strip() == lines['PID']
    assert (directory / run / 'train.log').is_file()
    nohup = _arguments(directory / 'capture_nohup')
    assert nohup[:7] == ['env', 'CUDA_VISIBLE_DEVICES=2', 'OMP_NUM_THREADS=1',
                        'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                        'PYTORCH_CUDA_ALLOC_CONF=backend:native',
                        'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    assert not any((directory / name).exists() for name in (
        'path_injected', 'shell_injected', 'backtick_injected'))


def test_omitted_overrides_keep_default_suffix_and_distinct_directories(bash, workspace):
    directory, _ = workspace
    runs = []
    for _ in range(2):
        result = _run(bash, workspace)
        assert result.returncode == 0, result.stderr
        run = dict(line.split('=', 1) for line in result.stdout.splitlines())['RUN']
        assert f'-37_{RECIPE}-mini_car_seed42_60ep_bs16-' in run
        runs.append(run)
    assert runs[0] != runs[1]
    _wait_for(directory / 'capture_main')
    parsed = json.loads((directory / 'capture_parsed').read_text(encoding='utf-8'))
    assert parsed['path'] == 'default YAML data'
    assert parsed['devices'] == 1


@pytest.mark.parametrize('option', ['--checkpoint', '--init_checkpoint', '--test'])
@pytest.mark.parametrize('equal_form', [False, True])
def test_managed_options_fail_before_preflight_or_output(bash, workspace, option, equal_form):
    directory, _ = workspace
    extra = [option + '=other'] if equal_form else [option]
    result = _run(bash, workspace, extra)
    assert result.returncode == 2 and 'launcher manages' in result.stderr
    assert 'main.py directly' in result.stderr
    assert not (directory / 'capture_preflight').exists()
    assert not (directory / 'output').exists()


@pytest.mark.parametrize('extras,environment', [
    ([], {'FAIL_PREFLIGHT': '1'}),
    (['--unknown_argument', 'x'], {}),
    (['--checkp', 'foreign-run'], {}),
    (['--ct_engineering_check'], {}),
    ([], {'FAKE_RECIPE': 'full_mass_detach/unsafe'}),
])
def test_failed_preflight_never_creates_output_or_starts_fake_training(bash, workspace, extras, environment):
    directory, _ = workspace
    result = _run(bash, workspace, extras, **environment)
    assert result.returncode != 0
    assert not (directory / 'output').exists()
    assert not (directory / 'capture_main').exists()


@pytest.mark.parametrize('environment_path,cli_path,expected', [
    (None, None, 'custom YAML data with spaces'),
    ('environment data', None, 'environment data'),
    ('environment data', 'explicit CLI data', 'explicit CLI data'),
    (None, 'explicit CLI data', 'explicit CLI data'),
])
def test_custom_yaml_same_recipe_preserves_path_precedence(
        bash, workspace, environment_path, cli_path, expected):
    directory, _ = workspace
    custom_cfg = directory / 'custom config.yaml'
    _write(custom_cfg, json.dumps({'v37_centroid_recipe': RECIPE,
                                  'path': 'custom YAML data with spaces'}))
    extras = ['--cfg=' + custom_cfg.as_posix()]
    if cli_path is not None:
        extras.extend(['--path', cli_path])
    environment = {} if environment_path is None else {'CT_DATA_ROOT': environment_path}
    result = _run(bash, workspace, extras, **environment)
    assert result.returncode == 0, result.stderr
    parsed = json.loads((directory / 'capture_parsed').read_text(encoding='utf-8'))
    assert parsed['cfg'] == custom_cfg.as_posix()
    assert parsed['path'] == expected
    _wait_for(directory / 'capture_main')
    passed = _arguments(directory / 'capture_main')
    assert passed.count('--path') == int(environment_path is not None) + int(cli_path is not None)


def test_custom_yaml_other_recipe_rejected_before_output(bash, workspace):
    directory, _ = workspace
    custom_cfg = directory / 'other.yaml'
    _write(custom_cfg, json.dumps({'v37_centroid_recipe': 'b1_b2_mass_detach'}))
    result = _run(bash, workspace, ['--cfg', custom_cfg.as_posix()])
    assert result.returncode != 0
    assert 'change the first recipe argument' in result.stderr
    assert not (directory / 'output').exists()
    assert not (directory / 'capture_main').exists()


@pytest.mark.parametrize('existing', [False, True])
def test_explicit_new_or_empty_log_directory_is_used(bash, workspace, existing):
    directory, _ = workspace
    run = directory / 'custom run with spaces'
    if existing:
        run.mkdir()
    extras = ['--log_dir=' + run.as_posix()]
    result = _run(bash, workspace, extras)
    assert result.returncode == 0, result.stderr
    assert dict(line.split('=', 1) for line in result.stdout.splitlines())['RUN'] == run.as_posix()
    _wait_for(directory / 'capture_main')
    assert _arguments(directory / 'capture_main')[-1] == extras[-1]
    assert (run / 'train.log').is_file() and (run / 'train.pid').is_file()
    assert not (directory / 'output').exists()


@pytest.mark.parametrize('content_name', ['train.log', '.hidden'])
def test_existing_nonempty_log_directory_is_rejected_without_truncation(bash, workspace, content_name):
    directory, _ = workspace
    run = directory / 'existing run'
    _write(run / content_name, 'protected previous contents\n')
    result = _run(bash, workspace, ['--log_dir', run.as_posix()])
    assert result.returncode != 0
    assert 'new or empty directory' in result.stderr
    assert (run / content_name).read_text(encoding='utf-8') == 'protected previous contents\n'
    assert sorted(path.name for path in run.iterdir()) == [content_name]
    assert not (directory / 'output').exists()
    assert not (directory / 'capture_main').exists()
