"""PointNet 计数复用保持 FP32 更新逐位一致；BF16 的 BN/输出边界。"""
from copy import deepcopy
from dataclasses import fields

import pytest
import torch
from torch import nn

from models.ct_v38.observation import B0Observation
from tests.helpers.synthetic_batch import make_batch
from utils.masked_observation import masked_batch_norm, masked_sequence


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('count', (0, 1, 5, 8))
@pytest.mark.parametrize('momentum', (.1, None))
def test_cached_bn_count_preserves_gradients_buffers_and_adam(count, momentum):
    torch.manual_seed(73)
    a = nn.BatchNorm1d(3, momentum=momentum).train()
    b = deepcopy(a)
    x = torch.randn(2, 3, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    valid = torch.arange(8).reshape(2, 4) < count
    gradient = torch.randn_like(x)
    first = masked_batch_norm(x, valid, a)
    second = masked_batch_norm(y, valid, b, valid_count=count)
    assert torch.equal(first, second)
    first.backward(gradient); second.backward(gradient)
    assert torch.equal(x.grad, y.grad)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p.grad, q.grad)
    for module in (a, b):
        torch.optim.Adam(module.parameters(), lr=1e-4, foreach=False).step()
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_sequence_invalidates_count_after_pooling_and_flatten():
    torch.manual_seed(71)
    a = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.BatchNorm1d(3))
    b = deepcopy(a)
    x = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    first, _ = masked_sequence(a, x, valid)
    second, _ = masked_sequence(b, x, valid, valid_count=6)
    assert torch.equal(first, second)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


@pytest.mark.parametrize('case', ('full', 'partial', 'empty', 'singleton'))
def test_optimized_observation_preserves_outputs_gradients_bn_adam_and_rng(case):
    torch.manual_seed(72)
    batch = make_batch(2, 8)
    batch['history_is_initial'] = torch.zeros(2, 3, dtype=torch.bool)
    if case == 'partial':
        batch['point_valid'][0, :, ::2] = False
        batch['history_valid'][1, :2] = False
        batch['point_ids'][0, -1, 1] = batch['point_ids'][0, -1, 3]
    elif case in ('empty', 'singleton'):
        batch['point_valid'].zero_()
        if case == 'singleton':
            batch['point_valid'][0, -1, 0] = True
    first = B0Observation(4, query_context=True, coarse_condition=True).train()
    second = deepcopy(first)
    second.execution_profile = 'optimized_fp32'
    initial_rng = torch.get_rng_state().clone()
    left, right = first(batch), second(batch)
    assert torch.equal(initial_rng, torch.get_rng_state())
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if a is None:
            assert b is None
        else:
            assert torch.equal(a, b), field.name
    for output in (left, right):
        (output.coarse_box.square().sum() + output.source_tokens.square().sum()
            + output.segmentation_logits.square().sum() + output.bc_prediction.square().sum()).backward()
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), name
    for model in (first, second):
        torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False).step()
    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key
    assert torch.equal(initial_rng, torch.get_rng_state())


@pytest.mark.parametrize('count', (0, 1, 5, 8))
def test_bf16_bn_boundary_uses_fp32_stats_and_output(count):
    torch.manual_seed(74)
    first = nn.BatchNorm1d(3).train()
    second = deepcopy(first)
    value = torch.randn(2, 3, 4).bfloat16()
    valid = torch.arange(8).reshape(2, 4) < count
    expected = masked_batch_norm(value.float(), valid, first, valid_count=count)
    actual = masked_batch_norm(value, valid, second, valid_count=count, force_fp32=True)
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    for key, item in first.state_dict().items():
        assert torch.equal(item, second.state_dict()[key]), key


def test_bf16_observation_rejects_cpu_without_updating_bn():
    model = B0Observation(4).train()
    model.execution_profile = 'optimized_bf16'
    before = {key: item.clone() for key, item in model.named_buffers()}
    with pytest.raises(RuntimeError, match='CUDA'):
        model(make_batch(2, 8))
    for key, item in model.named_buffers():
        assert torch.equal(item, before[key]), key
