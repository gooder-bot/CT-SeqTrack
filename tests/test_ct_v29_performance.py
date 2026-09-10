"""性能执行开关、批量读回和隔离恢复；不更换网络数值路径。"""

import copy
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.online_contract import build_online_resume_contract, validate_scratch_training_contract
from utils.action_calibration_v27 import action_calibration_config_identity
from utils.training_isolation import capture_global_rng_state
from utils.v28_numerical_audit import compare_values
from utils.v29_performance import (
    PERFORMANCE_DEFAULTS, performance_enabled, diagnostics_sampled,
    scalar_items_to_python, restore_changed_buffers, preserved_model_state,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arm', ['b0', 'full_cfc', 'full_gru'])
def test_perf_configs_only_change_execution_diagnostics_and_experiment_identity(arm):
    base = configure_ct_variant(load_yaml_config(ROOT / f'cfgs/ct_seqtrack/29_{arm}_nuscenes_full.yaml'))
    perf = configure_ct_variant(load_yaml_config(ROOT / f'cfgs/ct_seqtrack/29_{arm}_nuscenes_full_perf.yaml'))
    validate_scratch_training_contract(perf)
    assert not performance_enabled(base) and performance_enabled(perf)
    assert diagnostics_sampled(perf)
    difference = {k for k in base.keys() | perf.keys() if base.get(k) != perf.get(k)}
    assert difference == set(PERFORMANCE_DEFAULTS) | {'experiment_name'}
    assert perf['workers'] == 4 and perf['ct_checkpoint_every_n_epochs'] == 2
    before = build_online_resume_contract(base)['fields']
    after = build_online_resume_contract(perf)['fields']
    identity = action_calibration_config_identity(perf)
    for key in PERFORMANCE_DEFAULTS:
        assert key not in before
        assert after[key] == identity[key] == perf[key]


@pytest.mark.parametrize('key,value', [
    ('ct_runtime_optimization', 'fast_typo'), ('ct_diagnostic_policy', 'sample'),
    ('ct_scalar_log_every_n_steps', 0), ('ct_diagnostic_every_n_steps', True),
    ('ct_h3_diagnostic_keep_ratio', 0), ('ct_h3_diagnostic_keep_ratio', float('nan')),
])
def test_perf_contract_rejects_invalid_configuration(key, value):
    cfg = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/29_b0_nuscenes_full_perf.yaml')
    cfg[key] = value
    with pytest.raises(ValueError):
        configure_ct_variant(cfg)


def test_scalar_pack_preserves_dtype_values_order_and_autograd():
    values = {'double': torch.tensor(1.00000000000001, dtype=torch.float64, requires_grad=True),
              'large_int': torch.tensor(2**60 + 1), 'boolean': torch.tensor(True),
              'single': torch.tensor(1.25, requires_grad=True), 'python': 7,
              'nan': torch.tensor(float('nan'))}
    packed = scalar_items_to_python(values)
    assert list(packed) == list(values)
    for key, value in values.items():
        if key == 'nan':
            assert math.isnan(packed[key])
        else:
            expected = value.item() if torch.is_tensor(value) else value
            assert type(packed[key]) is type(expected) and packed[key] == expected
    values['single'].backward()
    assert values['single'].grad == 1
    with pytest.raises(ValueError, match='not a scalar'):
        scalar_items_to_python({'wrong': torch.zeros(2)})


def _buffer_module(device='cpu'):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3)).to(device)
    model.register_buffer('empty', torch.empty(0, device=device))
    model.register_buffer('counter', torch.tensor(3, device=device))
    return model


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA requires server'))])
def test_batched_restore_matches_equal_loop_and_preserves_unmodified_versions(device):
    model = _buffer_module(device)
    snapshots = {name: value.clone() for name, value in model.named_buffers()}
    original_versions = {name: value._version for name, value in model.named_buffers()}
    model[0].running_mean.add_(2)
    model.counter.add_(1)
    old = copy.deepcopy(model)
    versions = {name: value._version for name, value in model.named_buffers()}
    with torch.no_grad():
        for name, value in old.named_buffers():
            if not torch.equal(value, snapshots[name]):
                value.copy_(snapshots[name])
    restore_changed_buffers(model, snapshots)
    for name, value in model.named_buffers():
        assert torch.equal(value, dict(old.named_buffers())[name])
        changed = name in ('0.running_mean', 'counter')
        assert value._version == versions[name] + int(changed)
        if not changed:
            assert value._version == original_versions[name]


def test_diagnostic_state_guard_restores_rng_buffers_flags_even_on_error():
    host = _buffer_module()
    host[0].eval()
    buffers = {k: v.clone() for k, v in host.named_buffers()}
    flags = [m.training for m in host.modules()]
    before = capture_global_rng_state()
    with pytest.raises(RuntimeError, match='diagnostic failure'):
        with preserved_model_state(host):
            host.train()
            host(torch.randn(4, 3))
            random.random()
            np.random.rand()
            host.counter.add_(12)
            raise RuntimeError('diagnostic failure')
    assert compare_values(before, capture_global_rng_state()) is None
    assert flags == [m.training for m in host.modules()]
    assert all(torch.equal(buffers[k], v) for k, v in host.named_buffers())


def test_buffer_fallback_matches_equal_nan_dtype_shape_and_noncontiguous():
    model = torch.nn.Module()
    model.register_buffer('noncontiguous', torch.arange(12.).reshape(3, 4).t())
    model.register_buffer('nan', torch.tensor([float('nan')]))
    model.register_buffer('dtype_changed', torch.tensor([1., 2.]))
    snapshots = {name: value.clone() for name, value in model.named_buffers()}
    snapshots['dtype_changed'] = torch.tensor([1, 2], dtype=torch.int64)
    versions = {name: value._version for name, value in model.named_buffers()}
    assert torch.equal(model.dtype_changed, snapshots['dtype_changed'])
    restore_changed_buffers(model, snapshots)
    assert model.noncontiguous._version == versions['noncontiguous']
    assert not model.noncontiguous.is_contiguous()
    assert model.dtype_changed._version == versions['dtype_changed']
    assert model.nan._version == versions['nan'] + 1  # torch.equal(NaN, NaN) is False.
    assert torch.isnan(model.nan).all()
    snapshots['noncontiguous'] = torch.ones(2, 2)
    with pytest.raises(RuntimeError):
        restore_changed_buffers(model, snapshots)


def test_quantized_cpu_buffer_uses_original_equal_fallback():
    value = torch.quantize_per_tensor(torch.tensor([1., 2.]), .1, 0, torch.qint8)
    model = torch.nn.Module()
    model.register_buffer('quantized', value)
    version = value._version
    restore_changed_buffers(model, {'quantized': value.clone()})
    assert value._version == version


def test_aliased_buffers_keep_original_sequential_restore_versions():
    model = torch.nn.Module()
    value = torch.arange(6.)
    model.register_buffer('first', value)
    model.register_buffer('alias', value[1:])
    snapshots = {name: tensor.clone() for name, tensor in model.named_buffers()}
    model.first.add_(10)
    version = value._version
    restore_changed_buffers(model, snapshots)
    assert torch.equal(value, snapshots['first'])
    assert value._version == version + 1  # 恢复 first 后 alias 已相等，不能再 copy_。


def test_sparse_layout_uses_original_equal_result_or_error():
    value = torch.tensor([[1., 0.], [0., 2.]]).to_sparse()
    model = torch.nn.Module()
    model.register_buffer('sparse', value)
    before = value.clone()
    try:
        equal = torch.equal(value, before)
    except NotImplementedError:
        with pytest.raises(NotImplementedError):
            restore_changed_buffers(model, {'sparse': before})
    else:
        assert equal
        version = value._version
        restore_changed_buffers(model, {'sparse': before})
        assert value._version == version


@pytest.mark.parametrize('monitor', ['checkpoint_monitor', 'scheduler_monitor', 'model_selection_metric'])
def test_sampled_metrics_cannot_control_model_selection(monitor):
    cfg = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/29_full_gru_nuscenes_full_perf.yaml')
    cfg[monitor] = 'sampled_ct_relation_ap'
    with pytest.raises(ValueError, match='sampled training diagnostics'):
        configure_ct_variant(cfg)


def test_sampling_rejects_pre_v29_and_delayed_b3_target():
    from utils.v29_performance import validate_performance_contract
    cfg = load_yaml_config(ROOT / 'cfgs/ct_seqtrack/29_full_gru_nuscenes_full_perf.yaml')
    cfg['ct_b3_target_contract'] = 'h1_h3_mixture'
    with pytest.raises(ValueError, match='H1-only'):
        validate_performance_contract(cfg)
    cfg['ct_enable_v29'] = False
    with pytest.raises(ValueError, match='requires v29'):
        validate_performance_contract(cfg)
