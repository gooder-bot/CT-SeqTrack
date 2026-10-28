"""启动器用伪 Python/nvidia-smi 验证拒绝占卡、顺序和失败阻断；不运行模型。"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def bash_path(path):
    value = str(Path(path).resolve()).replace('\\', '/')
    return '/' + value[0].lower() + value[2:] if os.name == 'nt' else value


def shell():
    if os.name == 'nt':
        candidate = Path('C:/Program Files/Git/bin/bash.exe')
        if candidate.exists():
            return str(candidate)
    candidate = shutil.which('bash')
    if candidate is None:
        pytest.skip('bash unavailable')
    return candidate


def run_fixture(tmp_path, *, busy=False, fail_a=False, fail_cuda_tests=False, fail_cuda_gate=False):
    repo = tmp_path / 'repo'
    (repo / 'tools').mkdir(parents=True)
    (repo / 'data' / 'v1.0-mini').mkdir(parents=True)
    (repo / 'data' / 'samples').mkdir()
    launcher = repo / 'tools' / 'launch_v39_sequential.sh'
    launcher.write_bytes(Path('tools/launch_v39_sequential.sh').read_bytes())
    bin_dir = repo / 'mock_bin'
    bin_dir.mkdir()
    python = bin_dir / 'fake_python'
    python.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_RECORD"
if [[ "$*" == *"main.py"* && "$*" == *"39_full_a_mini.yaml"* && "$FAIL_A" == "1" ]]; then
  exit 17
fi
if [[ "$*" == *"-m pytest"* && "$FAIL_CUDA_TESTS" == "1" ]]; then
  exit 23
fi
if [[ "$1" == "-" && "$*" == *"cuda_tests.xml"* && "$FAIL_CUDA_GATE" == "1" ]]; then
  exit 24
fi
exit 0
''', encoding='utf-8')
    nvidia = bin_dir / 'nvidia-smi'
    nvidia.write_text('''#!/usr/bin/env bash
if [[ "$*" == *"--query-gpu=uuid"* ]]; then
  printf '%s\\n' 'GPU-TARGET'
elif [[ "$GPU_BUSY" == "1" ]]; then
  printf '%s\\n' 'GPU-TARGET, 1234, someone_else'
else
  printf '%s\\n' 'GPU-OTHER, 4567, unrelated_gpu_job'
fi
''', encoding='utf-8')
    for path in (launcher, python, nvidia):
        path.chmod(0o755)
    record = repo / 'calls.txt'
    env = dict(os.environ, MOCK_BIN=bash_path(bin_dir), FAKE_RECORD=bash_path(record),
               GPU_BUSY='1' if busy else '0', FAIL_A='1' if fail_a else '0',
               FAIL_CUDA_TESTS='1' if fail_cuda_tests else '0', FAIL_CUDA_GATE='1' if fail_cuda_gate else '0')
    result = subprocess.run([shell(), '-c', 'export PATH="$MOCK_BIN:$PATH"; exec bash "$@"',
        'launcher-test', bash_path(launcher), '--python', bash_path(python),
        '--path', bash_path(repo / 'data'), '--run-root', bash_path(repo / 'new_runs')],
        env=env, text=True, capture_output=True, timeout=30)
    calls = record.read_text(encoding='utf-8').splitlines() if record.exists() else []
    return result, calls, repo


def test_busy_gpu_refuses_probe_and_training_without_killing(tmp_path):
    result, calls, repo = run_fixture(tmp_path, busy=True)
    assert result.returncode != 0
    assert 'refusing to start' in result.stderr
    assert not any('check_v39_full_batch' in call or 'main.py' in call for call in calls)
    assert not any('-m pytest' in call for call in calls)
    assert not (repo / 'new_runs').exists()


def test_runs_two_fresh_arms_in_order_after_probe_and_registration(tmp_path):
    result, calls, repo = run_fixture(tmp_path)
    assert result.returncode == 0, result.stderr
    training = [call for call in calls if 'main.py' in call]
    assert len(training) == 2
    assert '39_full_a_mini.yaml' in training[0] and '39_full_b_mini.yaml' in training[1]
    assert all('--devices 0' in call and '--checkpoint' not in call for call in training)
    tests = next(i for i,call in enumerate(calls) if '-m pytest' in call)
    assert all(name in calls[tests] for name in ('test_ct_v39_efficiency.py','test_ct_v39_g.py',
        'test_ct_v39_data_contracts.py','test_ct_v39_integration.py'))
    gate = next(i for i,call in enumerate(calls) if call.startswith('- ') and 'cuda_tests.xml' in call)
    probe = next(i for i,call in enumerate(calls) if 'check_v39_full_batch' in call)
    assert tests < gate < probe < calls.index(training[0])
    profile = next(call for call in calls if 'benchmark_v39.py' in call)
    assert '--training --steps 12 --warmup 2' in profile
    assert calls.index(profile) < calls.index(training[0])
    assert (repo / 'new_runs' / 'full_a' / 'train.log').exists()
    assert (repo / 'new_runs' / 'full_b' / 'train.log').exists()


def test_a_failure_prevents_b(tmp_path):
    result, calls, repo = run_fixture(tmp_path, fail_a=True)
    assert result.returncode == 17
    training = [call for call in calls if 'main.py' in call]
    assert len(training) == 1 and '39_full_a_mini.yaml' in training[0]
    assert not (repo / 'new_runs' / 'full_b').exists()


@pytest.mark.parametrize('failure,code', [('fail_cuda_tests',23),('fail_cuda_gate',24)])
def test_cuda_failure_or_skipped_required_test_prevents_real_probe_and_training(tmp_path, failure, code):
    result,calls,repo=run_fixture(tmp_path,**{failure:True})
    assert result.returncode==code
    assert not any('check_v39_full_batch' in call or 'benchmark_v39' in call or 'main.py' in call for call in calls)
    assert not (repo/'new_runs').exists()
