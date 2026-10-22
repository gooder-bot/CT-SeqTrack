"""只测 shell 预检与字面 argv；假解释器退出，绝不启动训练。"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('tool', ['launch', 'resume'])
def test_failed_preflight_preserves_quoted_paths_and_never_creates_run(tool, tmp_path):
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    bash = str(git_bash) if git_bash.is_file() else shutil.which('bash')
    if bash is None:
        pytest.skip('Bash unavailable')
    (tmp_path / 'main.py').write_text('raise AssertionError("must not execute")', encoding='utf-8')
    cfg = tmp_path / 'cfgs/ct_seqtrack/38_full_standard_mini.yaml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text('v38_arm: full', encoding='utf-8')
    fake = tmp_path / 'fake python'
    with fake.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CT_CAPTURE"\nexit 7\n')
    fake.chmod(0o755)
    capture = tmp_path / 'capture'
    env = dict(os.environ, CT_PYTHON=fake.as_posix(), CT_CAPTURE=capture.as_posix(), MSYS2_ARG_CONV_EXCL='*')
    argv = [bash, (ROOT / f'tools/{tool}_v38.sh').as_posix(), 'full_standard', '1']
    if tool == 'resume':
        run = tmp_path / 'old run'
        run.mkdir()
        (run / 'run_manifest.json').write_text('{}', encoding='utf-8')
        checkpoint = run / 'epoch=010.ckpt'
        checkpoint.write_bytes(b'fixture')
        argv += [run.as_posix(), checkpoint.as_posix()]
    literal = (tmp_path / 'data with spaces $(touch injected)').as_posix()
    result = subprocess.run(argv + ['--path', literal], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 7, result.stderr
    args = [part.decode('utf-8') for part in capture.read_bytes().split(b'\0')[:-1]]
    assert args[args.index('--path') + 1] == literal
    assert not (tmp_path / 'output').exists() and not (tmp_path / 'injected').exists()
    assert not list(tmp_path.rglob('*.log')) and not list(tmp_path.rglob('*.pid'))


@pytest.mark.parametrize('tool', ['launch', 'resume'])
def test_launcher_bash_syntax(tool):
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    bash = str(git_bash) if git_bash.is_file() else shutil.which('bash')
    if bash is None:
        pytest.skip('Bash unavailable')
    result = subprocess.run([bash, '-n', (ROOT / f'tools/{tool}_v38.sh').as_posix()],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
