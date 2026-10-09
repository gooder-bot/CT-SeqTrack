"""完整事务的优化前后数值与工程工具边界。"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from tests.test_ct_v31_runtime import TinySource
from tools.check_execution_performance import (assert_identical, freeze, load_engineering_config,
                                               parse_args, run_equivalence, encode_report, _numerical_delta)
from utils.feature_precision import feature_autocast, feature_fp32


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('path', [
    'cfgs/ct_seqtrack/35_b1_w_scaled_lr_mini.yaml',
    'cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml',
    'cfgs/ct_seqtrack/36_a_full_mini.yaml',
])
def test_short_training_and_evaluation_closed_loop_is_bitwise_identical(path):
    config = load_engineering_config(path)
    config.update(point_sample_size=8, batch_size=2, workers=0, v32_reserve_windows=0)
    report = run_equivalence(config, 'cpu', steps=3,
        sources={'train': TinySource((5, 4)), 'val': TinySource((4, 3))})
    assert report['status'] == 'passed' and report['bitwise_equal']
    assert report['training_steps'] == 6 and report['evaluation_steps'] == 6
    assert [phase['epoch_index'] for phase in report['phases']] == [0, 10]
    assert report['phases'][1]['training_coverage']['nonempty_postcommit_state_steps'] > 0
    assert report['evaluation']['legacy'] == report['evaluation']['candidate']
    assert not report['saved_training_weights']


def test_bf16_cpu_is_rejected_and_fp32_is_a_noop():
    tensor = torch.ones(2, requires_grad=True)
    with feature_autocast(tensor, enabled=False):
        assert feature_fp32(tensor) is tensor
    with pytest.raises(RuntimeError, match='CUDA'):
        with feature_autocast(tensor, enabled=True):
            pass
    config = load_engineering_config('cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml')
    with pytest.raises(RuntimeError, match='CUDA'):
        run_equivalence(config, 'cpu', candidate_profile='optimized_bf16')


def test_comparison_keeps_dtype_order_and_invalid_nan_semantics():
    original = {'x': torch.tensor([float('nan'), 1.]), 'ids': np.array([2], dtype=np.int64)}
    assert_identical(freeze(original), freeze(original))
    assert_identical(torch.tensor(1.).expand(1), torch.ones(1))
    changed = deepcopy(original)
    changed['ids'] = changed['ids'].astype(np.int32)
    with pytest.raises(AssertionError, match='dtype'):
        assert_identical(original, changed)
    with pytest.raises(AssertionError, match='order'):
        assert_identical(original, dict(reversed(list(original.items()))))
    # 数值相等仍可能位模式不同；工程验收要求真正逐位比较。
    for positive, negative in ((torch.tensor(0.), torch.tensor(-0.)),
                               (np.array([0.]), np.array([-0.])), (0., -0.)):
        with pytest.raises(AssertionError):
            assert_identical(positive, negative)


def test_tool_cli_requires_paired_manifest_and_bounded_windows():
    cfg = ['--cfg', 'cfgs/ct_seqtrack/36_a_full_mini.yaml']
    args = parse_args(cfg)
    assert (args.warmup, args.steps, args.repeats, args.check_steps) == (20, 100, 3, 8)
    for extra in (['--steps', '0'], ['--weights', 'example.ckpt'], ['--warmup', '-1']):
        with pytest.raises(SystemExit):
            parse_args(cfg + extra)


def test_report_serialization_failure_is_valid_failed_json():
    content, status = encode_report({'status': 'passed', 'delta': float('nan')})
    assert status == 'failed'
    assert json.loads(content)['status'] == 'failed'
    assert 'NaN' not in content


def test_bf16_difference_report_excludes_invalid_infinite_logits():
    packet = dict(output=dict(accepted_box=torch.zeros(1, 4), selected_index=torch.zeros(1, dtype=torch.long),
        decoder=dict(quality_logits=torch.tensor([[0., float('-inf')]]),
                     hypothesis_valid=torch.tensor([[True, False]])),
        evidence={name: torch.zeros(1, 2, dtype=torch.long) for name in ('point_indices', 'point_valid', 'members')}),
        losses={'loss_total': torch.tensor(1.)}, rng={'torch': torch.zeros(1, dtype=torch.uint8)},
        host_steps=1, optimizer={'state': {0: {'step': torch.tensor(1.)}}}, state={})
    report = _numerical_delta(packet, deepcopy(packet))
    assert report['max_abs_quality_logit'] == 0.
    assert report['rng_equal'] and report['per_parameter_adam_steps_equal']
    assert report['host_step_counts_equal'] and report['memory_state_equal']
    json.dumps(report, allow_nan=False)
