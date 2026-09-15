"""真实 B0 验证显存优化前后输出、训练梯度、BN/RNG 与 Adam 更新逐位一致。"""
import copy
import importlib

import pytest
import torch
from torch import nn

from tests.test_ct_v30_host import (construct30, batch30, full_model_runtime,
                                   sampler_runtime)  # noqa: F401
from tests.test_ct_v30_masked_memory import original_partial_bn
from utils.deterministic_pooling import DeterministicMaxPool1d
import utils.masked_observation as masked


def _original_mask_values(value, valid):
    mask = valid[:, None] if value.ndim == 3 else valid.reshape(-1, 1)
    return torch.where(mask, value, torch.zeros_like(value))


def _original_sequence(module, value, valid):
    """首次 v30 每层后的 mask 均保留，作为优化前的独立执行对照。"""
    value = _original_mask_values(value, valid)
    for layer in module:
        if isinstance(layer, nn.BatchNorm1d):
            value = masked.masked_batch_norm(value, valid, layer)
        elif isinstance(layer, (nn.AdaptiveMaxPool1d, DeterministicMaxPool1d)):
            value, valid = masked.masked_max_pool(value, valid, layer.output_size)
        elif isinstance(layer, nn.Flatten):
            value = layer(value)
            valid = valid.any(-1)
        else:
            value = _original_mask_values(layer(value), valid)
    return value, valid


def _snapshot(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _assert_equal(first, second, path='snapshot'):
    if torch.is_tensor(first):
        assert torch.equal(first, second), path
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), path
        for key in first:
            _assert_equal(first[key], second[key], f'{path}.{key}')
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second), path
        for index, (left, right) in enumerate(zip(first, second)):
            _assert_equal(left, right, f'{path}[{index}]')
    else:
        assert first == second, path


@pytest.mark.parametrize('recompute', [False, True])
def test_real_b0_original_and_execution_modes_preserve_training_update(full_model_runtime, monkeypatch, recompute):
    baseline = construct30(full_model_runtime, 'b0').train()
    config = copy.deepcopy(baseline.config)
    config.ct_b0_masked_bn_recompute = recompute
    baseline = type(baseline)(config).train()
    for pointnet in (baseline.seg_pointnet, baseline.mini_pointnet, baseline.feature_pointnet):
        assert all(layer.ct_b0_masked_bn_recompute is recompute for layer in pointnet.modules()
                   if isinstance(layer, nn.BatchNorm1d))
    data, _, _ = batch30(full_model_runtime, baseline, batch_size=2)
    # 行 0 当前空、缺一个历史；行 1 留当前测量、缺不同历史及部分 token。
    data['b0_point_valid_mask'][0, -1] = False
    data['b0_point_valid_mask'][0, 0] = False
    data['valid_mask'][0, 0] = 0
    data['b0_point_valid_mask'][1, 1] = False
    data['valid_mask'][1, 1] = 0
    data['b0_point_valid_mask'][1, -1, 2:] = False
    flat_valid = data['b0_point_valid_mask'].flatten(1).bool()
    data['points'][~flat_valid] = 10000.
    data['candidate_bc'][~flat_valid] = -10000.
    assert not data['b0_point_valid_mask'][0, -1].any()
    assert data['b0_point_valid_mask'][1, -1].sum() == 2

    pointnet = importlib.import_module('models.backbone.pointnet')
    current_bn = masked.masked_batch_norm
    partial_calls = []

    def original_bn(value, valid, module):
        count = int(valid.sum())
        if not module.training or count < 2 or count == valid.numel():
            return current_bn(value, valid, module)
        partial_calls.append((tuple(value.shape), count))
        return original_partial_bn(value, valid, module)

    snapshots = []
    for original in (True, False):
        model = copy.deepcopy(baseline)
        optimizer = model.configure_optimizers()['optimizer']
        batch = copy.deepcopy(data)
        torch.manual_seed(123)
        with monkeypatch.context() as patch:
            if original:
                patch.setattr(masked, 'mask_values', _original_mask_values)
                patch.setattr(masked, 'masked_batch_norm', original_bn)
                patch.setattr(masked, 'masked_sequence', _original_sequence)
                # PointNet 在 import 时绑定了 helper，需要一并替换真实调用点。
                patch.setattr(pointnet, 'mask_values', _original_mask_values)
                patch.setattr(pointnet, 'masked_sequence', _original_sequence)
            output = model(batch)
            losses = model.compute_loss(batch, output)
            assert torch.isfinite(losses['loss_total'])
            forward = _snapshot(output)
            loss_values = _snapshot(losses)
            losses['loss_total'].backward()
        gradients = {name: _snapshot(param.grad) for name, param in model.named_parameters()}
        assert any(grad is not None and grad.count_nonzero() for grad in gradients.values())
        assert all(grad is None or torch.isfinite(grad).all() for grad in gradients.values())
        bn = {name: _snapshot(buffer) for name, buffer in model.named_buffers()
              if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        assert bn
        optimizer.step()
        snapshots.append(dict(output=forward, losses=loss_values, gradients=gradients,
            bn=bn, parameters=_snapshot(dict(model.named_parameters())),
            adam=_snapshot(optimizer.state_dict()), rng=torch.get_rng_state().clone()))
    assert partial_calls, 'the real host must execute the changed partial-validity BN path'
    assert any(len(shape) == 3 for shape, _ in partial_calls)
    differences = []
    for group in ('gradients', 'parameters'):
        for name, value in snapshots[0][group].items():
            other = snapshots[1][group][name]
            if torch.is_tensor(value) and not torch.equal(value, other):
                differences.append((group, name, float((value - other).abs().max()),
                                    int(torch.count_nonzero(value - other))))
    assert not differences, differences
    _assert_equal(snapshots[0], snapshots[1])
