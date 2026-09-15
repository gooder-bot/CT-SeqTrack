"""v30 显存回归：与修复前 BN 的数值、状态、Adam 及保存量直接对照。"""
import copy

import pytest
import torch
from torch import nn

from utils.masked_observation import masked_batch_norm


def original_partial_bn(value, valid, module):
    """首次 v30 的部分有效分支；独立保留用于数值对照。"""
    count = int(valid.sum())
    mask = valid[:, None] if value.ndim == 3 else valid.reshape(-1, 1)
    clean = torch.where(mask, value, torch.zeros_like(value))
    rows = clean.movedim(1, -1).reshape(-1, clean.shape[1]) if clean.ndim == 3 else clean
    weights = valid.reshape(-1, 1).to(clean.dtype)
    mean = rows.sum(0) / count
    variance = ((rows - mean).square() * weights).sum(0) / count
    if module.track_running_stats:
        with torch.no_grad():
            module.num_batches_tracked.add_(1)
            factor = (1. / float(module.num_batches_tracked)
                      if module.momentum is None else module.momentum)
            module.running_mean.lerp_(mean.detach(), factor)
            module.running_var.lerp_(variance.detach() * count / (count - 1), factor)
    shape = (1, -1, 1) if value.ndim == 3 else (1, -1)
    result = (clean - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + module.eps)
    if module.affine:
        result = result * module.weight.reshape(shape) + module.bias.reshape(shape)
    return torch.where(mask, result, torch.zeros_like(result))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(4, 5), (3, 5, 8)])
@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('momentum', [.1, None])
@pytest.mark.parametrize('recompute', [False, True])
def test_bn_execution_modes_match_original_forward_backward_and_single_state_update(dtype, shape, affine, momentum, recompute):
    torch.manual_seed(37)
    x = torch.randn(*shape, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = torch.ones(shape[:1] + shape[2:], dtype=torch.bool)
    mask.reshape(-1)[::3] = False
    first = nn.BatchNorm1d(shape[1], affine=affine, momentum=momentum).to(dtype).train()
    second = copy.deepcopy(first)
    second.ct_b0_masked_bn_recompute = recompute
    grad = torch.randn_like(x)
    rng = torch.get_rng_state().clone()
    a = original_partial_bn(x, mask, first)
    b = masked_batch_norm(y, mask, second)
    assert torch.equal(a, b)
    a.backward(grad)
    b.backward(grad)
    assert torch.equal(x.grad, y.grad)
    assert torch.count_nonzero(y.grad.masked_select(~(mask[:, None] if len(shape) == 3 else mask[:, None]))) == 0
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left.grad, right.grad)
    for name, value in first.named_buffers():
        assert torch.equal(value, dict(second.named_buffers())[name]), name
    assert second.num_batches_tracked.item() == 1
    assert torch.equal(rng, torch.get_rng_state())
    if affine:
        for module in (first, second):
            torch.optim.Adam(module.parameters(), lr=1e-4, betas=(.5, .999),
                             eps=1e-6, foreach=False).step()
        assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def saved_storage_bytes(call):
    storages = {}
    def pack(tensor):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = call()
    # Keeping result alive while counting prevents allocator address reuse during forward.
    assert result.requires_grad
    return sum(storages.values())


def test_partial_bn_does_not_retain_three_full_size_activation_copies():
    value = torch.randn(2, 128, 1024, requires_grad=True)
    valid = torch.ones(2, 1024, dtype=torch.bool)
    valid[0, :256] = False
    original = nn.BatchNorm1d(128)
    revised = copy.deepcopy(original)
    revised.ct_b0_masked_bn_recompute = True
    old_bytes = saved_storage_bytes(lambda: original_partial_bn(value, valid, original))
    new_bytes = saved_storage_bytes(lambda: masked_batch_norm(value, valid, revised))
    assert new_bytes < .4 * old_bytes


@pytest.mark.parametrize('recompute', [False, True])
def test_bn_execution_modes_support_affine_only_autograd_grad(recompute):
    x = torch.randn(2, 5, 8)  # 机制隔离或固定输入不要求输入梯度，参数仍应训练。
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, :3] = False
    first, second = nn.BatchNorm1d(5), nn.BatchNorm1d(5)
    second.ct_b0_masked_bn_recompute = recompute
    a = original_partial_bn(x, valid, first)
    b = masked_batch_norm(x, valid, second)
    grad = torch.randn_like(x)
    ga = torch.autograd.grad(a, tuple(first.parameters()), grad)
    gb = torch.autograd.grad(b, tuple(second.parameters()), grad)
    assert all(torch.equal(a, b) for a, b in zip(ga, gb))


def test_default_direct_bn_never_enters_recomputation(monkeypatch):
    from utils.masked_observation import _RecomputedMaskedBatchNorm
    def forbidden(*args):
        raise AssertionError('default direct BN entered recomputation')
    monkeypatch.setattr(_RecomputedMaskedBatchNorm, 'apply', forbidden)
    x = torch.randn(2, 5, 8, requires_grad=True)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, :3] = False
    module = nn.BatchNorm1d(5).train()
    result = masked_batch_norm(x, valid, module)
    result.square().sum().backward()
    assert module.num_batches_tracked.item() == 1
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_bn_execution_modes_cuda_strict_forward_backward_adam(recompute):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(42)
        x = torch.randn(4, 64, 1024, device='cuda', requires_grad=True)
        y = x.detach().clone().requires_grad_()
        valid = torch.ones(4, 1024, device='cuda', dtype=torch.bool)
        valid[0, :256] = False
        first = nn.BatchNorm1d(64).cuda().train()
        second = copy.deepcopy(first)
        second.ct_b0_masked_bn_recompute = recompute
        a = original_partial_bn(x, valid, first)
        b = masked_batch_norm(y, valid, second)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        grad = torch.randn_like(a)
        a.backward(grad)
        b.backward(grad)
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        for model in (first, second):
            torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False).step()
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)
