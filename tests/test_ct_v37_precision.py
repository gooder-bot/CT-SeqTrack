"""v37 实际精度边界：仅特征算子可用 BF16，其余训练链保持 FP32。"""
from dataclasses import fields, is_dataclass

import pytest
import torch
from torch import nn

from models.ct_v37.model import JointTracker
from tests.test_ct_v37_network import batch, config, threads
import utils.masked_observation as masked_observation


def _floating_tensors(value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _floating_tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floating_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _floating_tensors(item)


def _feature_scope(name):
    # 列出真实局部 autocast 范围；BN 的计算精度另在函数入口观测。
    scopes = ('observation.seg_pointnet.seq_per_point',
        'observation.seg_pointnet.seq_per_point2', 'observation.mini_pointnet.per_point',
        'observation.mini_pointnet.hidden', 'observation.feature_pointnet.seq_per_point',
        'observation.feature_pointnet.seq_per_point2', 'observation.feature_pointnet.fc',
        'decoder.source_projection', 'decoder.encoder', 'decoder.encoder_global', 'decoder.decoder')
    return any(name == prefix or name.startswith(prefix + '.') for prefix in scopes)


@pytest.mark.parametrize('device,profile', [
    pytest.param('cpu', 'optimized_fp32', id='cpu-fp32'),
    pytest.param('cuda', 'optimized_fp32', id='cuda-fp32', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires real CUDA execution')),
    pytest.param('cuda', 'optimized_bf16', id='cuda-bf16', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires real CUDA BF16 execution')),
])
def test_v37_feature_scope_and_sensitive_training_dtypes(device, profile, monkeypatch):
    if profile == 'optimized_bf16' and not torch.cuda.is_bf16_supported():
        pytest.skip('native CUDA BF16 unavailable')
    cfg = dict(config(), execution_profile=profile, point_sample_size=8)
    model = JointTracker(cfg).to(device).train()
    values = {key: value.to(device) for key, value in batch(2, 8).items()}
    feature_dtype = torch.bfloat16 if profile == 'optimized_bf16' else torch.float32
    modules = dict(model.named_modules())
    names_by_id = {id(module): name for name, module in modules.items()}
    observed, bn_calls, handles = set(), set(), []

    def record(name, module, inputs, output):
        floating = list(_floating_tensors(output))
        if not floating:
            return
        observed.add(name)
        if _feature_scope(name) and not isinstance(module, nn.BatchNorm1d):
            allowed = {torch.float32, feature_dtype}
            for value in floating:
                assert value.dtype in allowed, f'{name}: unexpected feature dtype {value.dtype}'
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                for value in floating:
                    assert value.dtype == feature_dtype, f'{name}: feature operator did not use {feature_dtype}'
        else:
            for value in _floating_tensors(inputs):
                assert value.dtype == torch.float32, f'{name}: sensitive input is {value.dtype}'
            for value in floating:
                assert value.dtype == torch.float32, f'{name}: sensitive output is {value.dtype}'

    # masked BN 的稀疏路径直接使用函数，单纯 module hook 会漏掉实际统计。
    original_bn = masked_observation.masked_batch_norm

    def observe_bn(value, valid, module, *, valid_count=None, force_fp32=False):
        name = names_by_id[id(module)]
        if not force_fp32:
            assert value.dtype == torch.float32, f'{name}: BN statistics input is {value.dtype}'
            bn_calls.add(name)
        output = original_bn(value, valid, module, valid_count=valid_count, force_fp32=force_fp32)
        assert output.dtype == torch.float32, f'{name}: BN output is {output.dtype}'
        for key in ('running_mean', 'running_var'):
            state = getattr(module, key)
            assert state.dtype == torch.float32, f'{name}.{key}: BN state is {state.dtype}'
        return output

    monkeypatch.setattr(masked_observation, 'masked_batch_norm', observe_bn)
    for name, module in modules.items():
        handles.append(module.register_forward_hook(
            lambda current, inputs, output, name=name: record(name, current, inputs, output)))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    try:
        output = model(values)
        losses = model.compute_losses(values, output)
        for name, loss in losses.items():
            assert loss.dtype == torch.float32, f'{name}: loss/log dtype is {loss.dtype}'
            assert torch.isfinite(loss).all(), f'{name}: nonfinite loss/log'
        losses['loss_total'].backward()
        for name, parameter in model.named_parameters():
            assert parameter.dtype == torch.float32, f'{name}: parameter is {parameter.dtype}'
            if parameter.grad is not None:
                assert parameter.grad.dtype == torch.float32, f'{name}: gradient is {parameter.grad.dtype}'
                assert torch.isfinite(parameter.grad).all(), f'{name}: nonfinite gradient'
        optimizer.step()
    finally:
        for handle in handles:
            handle.remove()

    required = {'observation.seg_pointnet.fc', 'observation.coarse_box_head.0',
        'observation.coarse_box_head.6',
        'observation.seg_pointnet.seq_per_point.0.0', 'decoder.source_projection',
        'decoder.decoder.layer_stack.0.enc_attn.w_qs', 'decoder.pose_head',
        'decoder.mode_pose_head', 'decoder.quality_head', 'decoder.local_reader.output',
        'prior.temporal_cell', 'prior.mean_head', 'prior.sigma_head', 'prior.acquisition_head',
        'evidence.local_geometry', 'evidence.vote_head', 'evidence.reliability_head',
        'evidence.identity_head', 'verifier.local_reader', 'verifier.output', ''}
    assert required <= observed, f'precision hooks did not execute: {sorted(required - observed)}'
    assert bn_calls == {name for name, module in modules.items() if isinstance(module, nn.BatchNorm1d)}, \
        'some BN calculations were not observed'
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32, f'{name}: Adam changed parameter dtype'
        assert torch.isfinite(parameter).all(), f'{name}: nonfinite parameter after Adam'
        state = optimizer.state.get(parameter, {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state:
                assert state[key].dtype == torch.float32, f'{name}.{key}: Adam state is {state[key].dtype}'
                assert torch.isfinite(state[key]).all(), f'{name}.{key}: nonfinite Adam state'
