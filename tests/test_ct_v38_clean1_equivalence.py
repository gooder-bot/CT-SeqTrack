"""冻结14c3b2f独立进程对照；旧源码只作只读实验基准。"""
from pathlib import Path
import json
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'artifacts/ct_checks/20261023_v38_clean1'
REMOVED = ('decoder.extension_projection.', 'decoder.memory_projection.')
RETIRED_LOSSES = {'loss_identity', 'loss_vote', 'loss_reliability', 'loss_modes',
                 'loss_modes_center', 'loss_modes_angle', 'mode_positive_fraction'}


def run_worker(root, destination, *, mode, arm='full', seed=42, common_state=None):
    command = [sys.executable, '-B', str(ROOT / 'tests/helpers/clean1_worker.py'),
        '--root', str(root), '--out', str(destination), '--mode', mode, '--arm', arm, '--seed', str(seed)]
    if common_state is not None:
        command += ['--common-state', str(common_state)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return torch.load(destination, map_location='cpu', weights_only=False)


def compare_values(old, new, *, rtol, atol, path='', errors=None):
    if isinstance(old, torch.Tensor):
        assert isinstance(new, torch.Tensor), path
        torch.testing.assert_close(old, new, rtol=rtol, atol=atol, msg=lambda msg: path + ': ' + msg)
        if errors is not None and old.numel() and old.is_floating_point():
            delta = (old-new).abs()
            errors.append((path, float(delta.max()), float((delta / old.abs().clamp_min(1e-12)).max())))
    elif isinstance(old, dict):
        assert all(float(old[key]) == 0. for key in RETIRED_LOSSES.intersection(old)), path
        expected = {key for key in old if not key.startswith(REMOVED) and key not in RETIRED_LOSSES}
        assert expected == set(new), (path, expected ^ set(new))
        for key in expected:
            compare_values(old[key], new[key], rtol=rtol, atol=atol, path=path+'/'+key, errors=errors)
    elif isinstance(old, (tuple, list)):
        assert len(old) == len(new), path
        for index, (a, b) in enumerate(zip(old, new)):
            compare_values(a, b, rtol=rtol, atol=atol, path=path+'/'+str(index), errors=errors)
    else:
        assert old == new, (path, old, new)


@pytest.mark.parametrize('seed', [42, 52])
@pytest.mark.parametrize('arm', ['b1', 'full'])
def test_retained_initialization_and_cpu_rng_are_exact(comparison_root, seed, arm):
    root, frozen = comparison_root
    old = run_worker(frozen, root/f'old_init_{arm}_{seed}.pt', mode='init', arm=arm, seed=seed)
    new = run_worker(ROOT, root/f'new_init_{arm}_{seed}.pt', mode='init', arm=arm, seed=seed)
    compare_values(old, new, rtol=0, atol=0)
    assert sum(value.numel() for key,value in old['initial'].items() if key.startswith(REMOVED)) == 8320


@pytest.mark.parametrize('arm,mode', [('b1','fp64'), ('full','fp64'), ('full','masked_fp64')])
def test_nonzero_outputs_losses_gradients_bn_adam_match_frozen(comparison_root, arm, mode):
    root, frozen = comparison_root
    old = run_worker(frozen, root/f'old_{mode}_{arm}.pt', mode=mode, arm=arm)
    new = run_worker(ROOT, root/f'new_{mode}_{arm}.pt', mode=mode, arm=arm)
    errors = []
    compare_values(old['eval'], new['eval'], rtol=1e-5, atol=1e-6, errors=errors)
    for index, (left, right) in enumerate(zip(old['steps'], new['steps'])):
        for key in left:
            numerical = key in ('gradients','state','optimizer')
            compare_values(left[key], right[key], rtol=1e-4 if numerical else 1e-5,
                atol=1e-5 if numerical else 1e-6, path=f'step{index}/{key}', errors=errors)
    (root/f'errors_{arm}_{mode}.json').write_text(json.dumps(sorted(errors,key=lambda row:row[1],reverse=True),indent=2),encoding='utf-8')
    if mode == 'masked_fp64':
        assert old['controlled_dropout_shapes'].keys() == new['controlled_dropout_shapes'].keys()
        assert len(new['controlled_dropout_shapes']) > 5


@pytest.mark.parametrize('arm', ['b1','full'])
def test_fp32_same_weight_outputs_and_gradients(comparison_root, arm):
    """用户确认：不把不同FP32归约后的Adam累积轨迹当作等价门。"""
    root, frozen = comparison_root
    old_path = root/f'old_fp32_common_{arm}.pt'
    old = run_worker(frozen, old_path, mode='deterministic', arm=arm)
    new = run_worker(ROOT, root/f'new_fp32_common_{arm}.pt', mode='deterministic', arm=arm, common_state=old_path)
    errors = []
    compare_values(old['eval'], new['eval'], rtol=1e-5, atol=1e-6, errors=errors)
    for index,(left,right) in enumerate(zip(old['steps'],new['steps'])):
        for key in ('output','losses','gradients'):
            gradient = key == 'gradients'
            compare_values(left[key],right[key],rtol=1e-4 if gradient else 1e-5,
                atol=1e-5 if gradient else 1e-6,path=f'step{index}/{key}',errors=errors)
    (root/f'errors_{arm}_fp32_common.json').write_text(json.dumps(sorted(errors,key=lambda row:row[1],reverse=True),indent=2),encoding='utf-8')


def test_new_natural_dropout_training_is_reproducible(comparison_root):
    root, _ = comparison_root
    a = run_worker(ROOT, root/'repeat_a.pt', mode='natural')
    b = run_worker(ROOT, root/'repeat_b.pt', mode='natural')
    compare_values(a, b, rtol=0, atol=0)
