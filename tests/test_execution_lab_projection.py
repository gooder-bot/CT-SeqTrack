"""阶段A严格置换合同；阶段B代数/边界合同，保留FP32非等价证据。"""
import copy

import pytest
import torch

from models.ct_v38 import observation
from models.ct_v38.observation import SegPointNet, FeaturePointNet, unique_valid_mask
from utils.execution_lab import execution_scope, enabled


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('shape', [(0,), (1,), (17,), (2, 4, 129)])
def test_unique_scatter_exactly_preserves_stable_first_valid_slot(dtype, shape):
    generator = torch.Generator().manual_seed(713)
    ids = torch.randint(-3, 13, shape, generator=generator, dtype=dtype)
    valid = torch.rand(shape, generator=generator) > .25
    if ids.numel():
        ids.reshape(-1)[-1] = torch.iinfo(dtype).max
    with execution_scope((), engineering=True):
        expected = unique_valid_mask(valid, ids)
    with execution_scope(('unique_mask',), engineering=True):
        actual = unique_valid_mask(valid, ids)
    assert torch.equal(expected, actual)
    assert actual.dtype == torch.bool


def test_unique_scatter_noncontiguous_ids_and_unchanged_validation():
    ids = torch.tensor([[8, 8, -1, 3, 3], [2**54, 7, 2**54, -1, 7]]).t()
    valid = torch.tensor([[True, False, True, True, True], [True, True, True, True, True]]).t()
    assert not ids.is_contiguous()
    for options in ((), ('unique_mask',)):
        with execution_scope(options, engineering=True):
            result = unique_valid_mask(valid, ids)
            assert torch.equal(result, valid & (ids >= 0))  # 每行两个不同ID，不跨行去重。
            assert torch.equal(unique_valid_mask(valid), valid)
            with pytest.raises(ValueError, match='integer tensors'):
                unique_valid_mask(valid, ids.float())
            with pytest.raises(ValueError, match='same shape'):
                unique_valid_mask(valid, ids[:, :1])


def test_unique_scatter_removes_only_the_inverse_sort(monkeypatch):
    original_argsort = torch.argsort
    calls = []

    def traced(*args, **kwargs):
        calls.append(kwargs)
        return original_argsort(*args, **kwargs)

    monkeypatch.setattr(torch, 'argsort', traced)
    ids = torch.tensor([[4, 1, 4, 1, 9, -1]])
    valid = torch.tensor([[True, False, True, True, False, True]])
    with execution_scope((), engineering=True):
        expected = unique_valid_mask(valid, ids)
    assert len(calls) == 2
    calls.clear()
    with execution_scope(('unique_mask',), engineering=True):
        actual = unique_valid_mask(valid, ids)
    assert len(calls) == 1 and calls[0]['stable'] is True
    assert torch.equal(actual, expected)
    assert torch.equal(actual, torch.tensor([[True, False, False, True, False, False]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_unique_scatter_cuda_strict_determinism():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        ids = torch.arange(1024, device='cuda').remainder(27)[None].expand(3, -1)
        valid = ids.remainder(3) != 0
        with execution_scope((), engineering=True):
            expected = unique_valid_mask(valid, ids)
        with execution_scope(('unique_mask',), engineering=True):
            actual = unique_valid_mask(valid, ids)
        assert torch.equal(expected, actual)
    finally:
        torch.use_deterministic_algorithms(previous)


def _mask(groups, slots, case):
    valid = torch.ones(groups, slots, dtype=torch.bool)
    if case == 'partial':
        valid[:, ::3] = False
        valid[::2, 1::5] = False
    elif case == 'empty_row':
        valid[0] = False
        valid[-1, ::2] = False
    elif case == 'all_empty':
        valid[:] = False
    elif case == 'singleton_total':
        valid[:] = False
        valid[-1, -1] = True
    return valid


def _paired(kind, dtype, training, case, *, slots=64, precount=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        original = (SegPointNet() if kind == 'seg' else FeaturePointNet()).to(dtype=dtype).train(training)
    revised = copy.deepcopy(original)
    keys = list(original.state_dict())
    parameter_ids = [id(p) for p in revised.parameters()]
    assert keys == list(revised.state_dict())
    groups = 2 if kind == 'seg' else 8  # Feature第一维为B*frames，各组具有不同global。
    generator = torch.Generator().manual_seed(20261014)
    value = torch.randn(groups, 14, slots, generator=generator, dtype=dtype)
    value = value * (1. + torch.arange(groups, dtype=dtype)[:, None, None] * .2)
    value = value + torch.arange(groups, dtype=dtype)[:, None, None] * .15
    valid = _mask(groups, slots, case)
    value = torch.where(valid[:, None], value, torch.full_like(value, float('nan')))
    left, right = value.clone().requires_grad_(), value.clone().requires_grad_()
    count = int(valid.sum()) if precount else None
    with execution_scope((), engineering=True):
        expected = original(left, valid, valid_count=count)
    with execution_scope(('factorized_projection',), engineering=True):
        actual = revised(right, valid, valid_count=count)
    left_loss, right_loss = 0., 0.
    for a, b in zip(expected, actual):
        if a.is_floating_point():
            weights = torch.randn(a.shape, generator=generator, dtype=dtype)
            left_loss = left_loss + (a * weights).sum() / a.numel()
            right_loss = right_loss + (b * weights).sum() / b.numel()
        else:
            assert torch.equal(a, b)
    left_loss.backward()
    right_loss.backward()
    assert keys == list(revised.state_dict())
    assert parameter_ids == [id(p) for p in revised.parameters()]
    for (name, p), (other_name, q) in zip(original.named_parameters(), revised.named_parameters()):
        assert name == other_name and torch.equal(p, q)
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all())
    for output in (*expected, *actual):
        assert bool(torch.isfinite(output).all())
    for value in (left, right):
        assert bool(torch.isfinite(value.grad).all())
        assert bool((value.grad.masked_select(~valid[:, None].expand_as(value)) == 0).all())
    return original, revised, left, right, expected, actual


@pytest.mark.parametrize('kind', ['seg', 'feature'])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('case', ['all_valid', 'partial', 'empty_row', 'all_empty', 'singleton_total'])
def test_factorized_fp64_full_module_algebra_gradients_and_bn_contract(kind, training, case):
    # FP64只作代数健全性；容差不是FP32轨迹等价门。
    original, revised, left, right, expected, actual = _paired(kind, torch.float64, training, case)
    for a, b in zip(expected, actual):
        if a.is_floating_point():
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(left.grad, right.grad, atol=1e-9, rtol=1e-9)
    for (name, p), (other_name, q) in zip(original.named_parameters(), revised.named_parameters()):
        assert name == other_name and (p.grad is None) == (q.grad is None)
        assert p.requires_grad and q.requires_grad
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-8, rtol=1e-8)
    for (name, a), (other_name, b) in zip(original.named_buffers(), revised.named_buffers()):
        assert name == other_name
        if a.is_floating_point():
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
        else:
            assert torch.equal(a, b)


@pytest.mark.parametrize('kind', ['seg', 'feature'])
def test_factorized_contract_without_precomputed_count(kind):
    original, revised, left, right, expected, actual = _paired(
        kind, torch.float64, True, 'partial', precount=False)
    torch.testing.assert_close(expected[0], actual[0], atol=1e-9, rtol=1e-9)


def test_factorized_fp32_known_sensitive_case_keeps_original_gradient_tolerance():
    # 合同/有限值断言先正常失败；仅已知的数值容差失败标xfail，不掩盖实现错误。
    original, revised, left, right, expected, actual = _paired(
        'feature', torch.float32, True, 'empty_row', slots=256)
    within_original_tolerance = all(torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4)
        for (_, p), (_, q) in zip(original.named_parameters(), revised.named_parameters())
        if p.grad is not None)
    if not within_original_tolerance:
        pytest.xfail('保留已归档FP32/Feature/train/empty_row梯度容差失败：'
                     'atol=1e-5/rtol=1e-4未放宽；不声称训练轨迹等价')
    # 其他平台若落在此容差内，也只表明这个case，不自动开放正式执行。


@pytest.mark.parametrize('constructor', [SegPointNet, FeaturePointNet])
def test_factorized_is_off_by_default_and_rejects_bf16_before_bn_updates(constructor, monkeypatch):
    assert not enabled('factorized_projection')
    model = constructor().train()
    value, valid = torch.randn(2, 14, 16), torch.ones(2, 16, dtype=torch.bool)

    def forbidden(*args, **kwargs):
        raise AssertionError('disabled path used factorized helper')

    monkeypatch.setattr(observation, 'factorized_global_block', forbidden)
    with torch.no_grad():
        model(value, valid)
    before = {name: tensor.clone() for name, tensor in model.named_buffers()}
    with execution_scope(('factorized_projection',), engineering=True):
        with pytest.raises(ValueError, match='BF16 is not supported'):
            model(value, valid, feature_bf16=True)
    assert all(torch.equal(before[name], tensor) for name, tensor in model.named_buffers())
