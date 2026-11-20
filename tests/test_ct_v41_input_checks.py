"""主辅前向复用同一尺寸/时间校验，保留独立接口和失效后的检查。"""
from collections import Counter

import pytest
import torch

from models.ct_v41 import input_checks
from models.ct_v41.decoder import SharedHypothesisDecoder
from models.ct_v41.history_context import history_descriptor
from models.ct_v41.model import JointTracker
from tests.test_ct_v41_recovery import config, values


def count_checks(monkeypatch, **fields):
    pointers = {tensor.data_ptr(): name for name, tensor in fields.items()}
    calls = Counter()
    original = torch.isfinite

    def counted(tensor):
        name = pointers.get(tensor.data_ptr())
        if name is not None:
            calls[name] += 1
        return original(tensor)

    monkeypatch.setattr(torch, 'isfinite', counted)
    return calls


def test_main_and_aux_each_validate_once(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        net = JointTracker(config()).train()
        data = values(2, 11)
        counts = count_checks(monkeypatch, size=data['box_size'], times=data['frame_times'])
        net(data, include_local_diagnostics=False)
        assert counts == {'size': 1, 'times': 1}
        net.forward_aux(data, include_local_diagnostics=False)
        assert counts == {'size': 2, 'times': 2}


def test_detached_alias_replacement_and_scope_lifetime(monkeypatch):
    size = torch.ones(2, 3)
    counts = count_checks(monkeypatch, size=size)
    with input_checks.input_check_scope():
        assert input_checks.positive_finite_size(size)
        alias = input_checks.detached_size(size)
        assert input_checks.positive_finite_size(alias) and counts['size'] == 1
        with input_checks.input_check_scope():
            assert input_checks.positive_finite_size(size) and counts['size'] == 2
        assert input_checks.positive_finite_size(size) and counts['size'] == 2
        assert not input_checks.positive_finite_size(torch.zeros_like(size))
        alias[0, 0] = 0.
        assert not input_checks.positive_finite_size(size)
        assert not input_checks.positive_finite_size(alias)
        size.fill_(1.)
        assert input_checks.positive_finite_size(size) and counts['size'] == 5
    # 独立组件不能继承上一次 forward 的证明。
    assert input_checks.positive_finite_size(size)
    assert input_checks.positive_finite_size(size) and counts['size'] == 7


def test_inference_tensors_have_no_reusable_version(monkeypatch):
    with torch.inference_mode(), input_checks.input_check_scope():
        size = torch.ones(2, 3)
        counts = count_checks(monkeypatch, size=size)
        assert input_checks.positive_finite_size(size)
        assert input_checks.positive_finite_size(size) and counts['size'] == 2
        size.zero_()
        assert not input_checks.positive_finite_size(size)


def test_casts_recheck_values_in_decoder_dtype():
    size = torch.full((1, 3), 1e-50, dtype=torch.float64)
    times = torch.tensor([[-1e100, -2., -1., 0.]], dtype=torch.float64, requires_grad=True)
    batch = {'frame_times': times}
    with input_checks.input_check_scope():
        assert input_checks.positive_finite_size(size)
        assert not input_checks.positive_finite_size(input_checks.detached_size(size).float())
        descriptor = history_descriptor(batch, torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.bool),
                                        size, torch.zeros(1, 3, 3), .5)
        assert not descriptor.requires_grad
        with pytest.raises(ValueError, match='frame_times must be finite'):
            SharedHypothesisDecoder._frame_times(batch, size.float())
        with torch.no_grad():
            times[0, 0] = torch.nan
        with pytest.raises(ValueError, match='frame_times must be finite'):
            SharedHypothesisDecoder._frame_times(batch, size)
