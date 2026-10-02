"""新完整链的真实模型 CPU 生命周期；数据为合成点云，不代表 CUDA 已检查。"""
import math

import pytest
import torch

from models.ct_v31.config import load_config
from models.ct_v31.data import build_loaders
from tests.test_ct_v31_runtime import TinySource
from tools import check_v35_full_batch as check


def test_full_batch_actual_model_forward_backward_adam_and_commit():
    from models import ctseqtrackv31 as host
    diagnostics_enabled = host.pl is not None
    config = load_config('cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml', dict(
        ct_engineering_check=True, workers=0, v35_train_diagnostics=diagnostics_enabled))
    loaders = build_loaders(config, roles=('train',), sources={'train': TinySource((3,) * 16)})
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    report = {}
    try:
        check.run_one_batch(config, 'cpu', report, loaders=loaders)
    finally:
        torch.set_num_threads(threads)
    assert report['status'] == 'passed'
    assert report['optimizer']['state_step_values'] == [1]
    assert report['candidate_records']['rows'] == 16
    assert report['candidate_records']['example']['schema'] == 'ct_seqtrack.v35.candidate_records.v1'
    assert report['batch']['shape'] == [16, 4, 1024, 5]
    assert report['commit']['epoch_rows'] == 16 and report['commit']['epoch_steps'] == 1
    assert not report['commit']['pending_host'] and not report['commit']['pending_builder']
    assert all(math.isfinite(value) for value in report['timing_seconds'].values())
    assert all(value['tensors_with_gradient'] > 0 for value in report['module_gradients'].values())
    assert ('training_diagnostics' in report) is diagnostics_enabled


def test_full_check_does_not_accept_b0_or_reduced_batch():
    b0 = load_config('cfgs/ct_seqtrack/35_b0_w_scaled_lr_mini.yaml', dict(ct_engineering_check=True))
    reduced = load_config('cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml', dict(ct_engineering_check=True, batch_size=2))
    for config in (b0, reduced):
        with pytest.raises(ValueError, match='requires v35 Full'):
            check.run_one_batch(config, 'cpu', {}, loaders={})
