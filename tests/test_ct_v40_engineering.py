"""CPU 工程门的拒绝测试；真实 CUDA/数据验收只由上传后的服务器工具执行。"""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from models.ct_v40.engineering import (CPUReplaySnapshot, GIB, MIB, analyze_memory_cycles,
    capacity_gate, digest_tree, tensor_addresses, timing_summary)


def cycles(offsets=(0, 0, 0), *, reserved=False):
    result = []
    for offset in offsets:
        samples = []
        for role in ('train', 'val', 'train_resume'):
            for index in range(2):
                for phase in ('before_prepare', 'after_forward', 'after_cleanup'):
                    samples.append(dict(role=role, index=index, phase=phase,
                        allocated=100 * MIB + (0 if reserved else offset),
                        reserved=500 * MIB + (offset if reserved else 0), oom_count=0))
        result.append(dict(workload_sha256='same-input-mask-cu-branch', samples=samples, allocation_retries_delta=0))
    return result


def test_memory_bounds_do_not_hide_ongoing_growth():
    assert analyze_memory_cycles(cycles())['passed']
    assert not analyze_memory_cycles(cycles((0, 4 * MIB, 8 * MIB)))['passed']
    assert not analyze_memory_cycles(cycles((0, 512, 1024)))['passed']
    assert not analyze_memory_cycles(cycles((0, 16 * MIB, 32 * MIB), reserved=True))['passed']
    assert not analyze_memory_cycles(cycles((0, 70 * MIB, 70 * MIB)))['passed']
    assert not analyze_memory_cycles(cycles((0, 300 * MIB, 300 * MIB), reserved=True))['passed']
    retried = cycles()
    retried[1]['allocation_retries_delta'] = 1
    assert not analyze_memory_cycles(retried)['passed']


def test_memory_mismatched_workload_or_phase_is_invalid():
    altered = cycles()
    altered[1]['workload_sha256'] = 'different-mask'
    with pytest.raises(RuntimeError, match='inputs/masks'):
        analyze_memory_cycles(altered)
    altered = cycles()
    altered[2]['samples'].pop()
    with pytest.raises(RuntimeError, match='coverage'):
        analyze_memory_cycles(altered)


def test_fingerprint_covers_bf16_mask_and_branch():
    original = dict(q=torch.ones(2, 4, dtype=torch.bfloat16), mask=torch.tensor([True, False]), branch=2)
    assert digest_tree(original) == digest_tree(copy.deepcopy(original))
    for changed in (dict(original, branch=3), dict(original, mask=~original['mask']),
                    dict(original, q=original['q'].float())):
        assert digest_tree(original) != digest_tree(changed)


class Builder:
    def __init__(self):
        self.states = {'track': dict(box=np.array([1., 2., 3., 0.]))}
        self._pending = None

    def reset(self):
        self.states.clear()
        self._pending = None


class ReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        self.train_builder, self.evaluation_builder = Builder(), Builder()
        self._pending_train = None
        self._epoch_rows = self._epoch_steps = self._completed_epoch = 0
        self._epoch_complete = False
        self.training_diagnostics = None


def test_cpu_restore_reproduces_adam_bn_rng_and_keeps_storage(monkeypatch):
    # 即使服务器有 CUDA，这个 unit test 也不创建 CUDA context。
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model = ReplayModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, foreach=False)
    inputs = torch.tensor([[1., 2., 3.], [2., 1., 0.], [0., 2., 4.]])
    snapshot = CPUReplaySnapshot(model, optimizer)
    addresses = tensor_addresses(model, optimizer)
    def update():
        loss = model.layer(inputs).square().sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.train_builder.states['track']['box'] += 4
        return digest_tree((model.state_dict(), [optimizer.state[p] for p in model.parameters()])), (
            random.random(), float(np.random.random()), float(torch.rand(())))
    first = update()
    snapshot.restore(model, optimizer)
    assert np.array_equal(model.train_builder.states['track']['box'], [1., 2., 3., 0.])
    assert tensor_addresses(model, optimizer) == addresses
    assert update() == first
    snapshot.restore(model, optimizer)
    assert tensor_addresses(model, optimizer) == addresses


def test_restore_refuses_reallocated_parameters(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model = ReplayModel()
    optimizer = torch.optim.Adam(model.parameters(), foreach=False)
    snapshot = CPUReplaySnapshot(model, optimizer)
    model.layer[0].weight = nn.Parameter(model.layer[0].weight.detach().clone())
    with pytest.raises(RuntimeError, match='storage was replaced'):
        snapshot.restore(model, optimizer)


def test_capacity_uses_total_free_including_external_usage():
    sample = dict(total=24 * GIB, free=3 * GIB, used=21 * GIB, external_and_driver_bytes=9 * GIB,
                  processes={'101': 6 * GIB, '102': 6 * GIB, '999': 8 * GIB}, external_pids=[999])
    assert capacity_gate([sample], [101, 102])['passed']
    assert not capacity_gate([dict(sample, free=2 * GIB)], [101, 102])['passed']
    with pytest.raises(RuntimeError, match='simultaneously'):
        capacity_gate([dict(sample, processes={'101': GIB}), dict(sample, processes={'102': GIB})], [101, 102])


def test_timing_requires_real_finite_positive_samples():
    assert timing_summary([1., 2., 3.])['median_seconds'] == 2.
    with pytest.raises(RuntimeError, match='invalid timings'):
        timing_summary([0.])
    with pytest.raises(RuntimeError, match='invalid timings'):
        timing_summary([float('nan')])


def test_launcher_can_select_second_pair_without_reordering_default_or_registration():
    from tools.launch_ct_v40 import selected_waves
    from tools.check_v40_server import WAVES
    assert selected_waves() == [(1, ('b0', 'b1')), (2, ('b1_mtp', 'full'))]
    assert selected_waves('2') == [(2, ('b1_mtp', 'full'))]
    assert selected_waves('1') == [(1, ('b0', 'b1'))]
    assert WAVES == (('b0', 'b1'), ('b1_mtp', 'full'))
    with pytest.raises(RuntimeError, match='wave must be'):
        selected_waves('3')


def test_compare_reconstructs_actual_sampler_budget_and_accepts_training_bool():
    from models.ct_v40.config import normalize_config
    from models.ct_v40.data import ReadyQueueBatchSampler
    from tools.compare_ct_v40 import validate_sampler
    from tools._tracking_audit import digest, EvidenceError
    config = normalize_config(dict(ct_engineering_check=True))
    sampler = ReadyQueueBatchSampler([10, 6], batch_size=16, seed=42, training=True,
        short_window=4, long_window=8, curriculum_epochs=10, source_sha256='a' * 64,
        reserve_windows=112, seed_translation=.3, seed_yaw_degrees=1.5,
        initial_seed_policy=ReadyQueueBatchSampler.INITIAL_SEED_POLICY)
    sampler.set_epoch(10)
    actual = sampler.state_dict()
    assert validate_sampler(actual, config, 11, 'test') == 'a' * 64
    altered = dict(actual, rows=actual['rows'] + 1)
    altered['manifest_sha256'] = digest({key: value for key, value in altered.items() if key != 'manifest_sha256'})
    with pytest.raises(EvidenceError, match='reconstructed'):
        validate_sampler(altered, config, 11, 'test')
