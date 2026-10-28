"""v39 E 的统计/梯度/空观测合同；CUDA FA2 对照仅在兼容设备可用时运行。"""
from copy import deepcopy
from dataclasses import fields
from types import MethodType, ModuleType
import sys

import pytest
import torch
from torch import nn

from models.attn.Modules import ScaledDotProductAttention
from models.ct_v38.decoder import SharedHypothesisDecoder
from models.ct_v38.observation import B0Observation, SegPointNet, MiniPointNet, FeaturePointNet
from models.ct_v39.attention import EfficientAttention, ValidityMask, _flash_varlen
from models.ct_v39.efficient_observation import (
    PackedPoints, configure_efficient_modules, grouped_first_max, packed_batch_norm,
    packed_seg_forward, packed_mini_forward, packed_feature_forward)
from tests.helpers.synthetic_batch import make_batch
from utils.masked_observation import masked_batch_norm


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize('count', (0, 1, 5, 8))
@pytest.mark.parametrize('momentum', (.1, None))
def test_packed_bn_matches_valid_population_gradients_and_running_state(count, momentum):
    torch.manual_seed(3901)
    dense_bn = nn.BatchNorm1d(3, momentum=momentum)
    compact_bn = deepcopy(dense_bn)
    valid = (torch.arange(8).reshape(2, 4) < count).roll(1, dims=1)
    left = torch.randn(2, 3, 4, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    expected = masked_batch_norm(left, valid, dense_bn)
    packed = PackedPoints.from_dense(right, valid)
    actual = packed.scatter(packed_batch_norm(packed.values, compact_bn, population_size=valid.numel()))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(right.grad, left.grad, rtol=3e-5, atol=3e-6)
    for (name, a), (_, b) in zip(dense_bn.named_parameters(), compact_bn.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=3e-6, msg=name)
    for name, a in dense_bn.named_buffers():
        torch.testing.assert_close(a, dict(compact_bn.named_buffers())[name], rtol=2e-5, atol=2e-6)
    assert int(compact_bn.num_batches_tracked) == int(count >= 2)


def test_packed_bn_bfloat16_input_keeps_fp32_statistics():
    torch.manual_seed(3902)
    left, right = nn.BatchNorm1d(4), nn.BatchNorm1d(4)
    right.load_state_dict(left.state_dict())
    value = torch.randn(7, 4).bfloat16()
    expected = left(value.float())
    actual = packed_batch_norm(value, right)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(expected, actual, rtol=0., atol=0.)
    assert right.running_mean.dtype == right.running_var.dtype == torch.float32


def test_grouped_max_retains_first_tie_gradient_and_empty_segments():
    value = torch.tensor([[2., 1.], [2., 3.], [1., 3.], [5., 5.]], requires_grad=True)
    result = grouped_first_max(value, torch.tensor([3, 0, 1, 0]))
    torch.testing.assert_close(result, torch.tensor([[2., 3.], [0., 0.], [5., 5.], [0., 0.]]))
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([[1., 0.], [0., 1.], [0., 0.], [1., 1.]]))


@pytest.mark.parametrize('kind', ('seg', 'mini', 'feature'))
@pytest.mark.parametrize('case', ('partial', 'singleton', 'empty'))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_packed_pointnets_preserve_outputs_gradients_and_bn(kind, case, dtype):
    torch.manual_seed(3903)
    options = {'seg': (SegPointNet, packed_seg_forward, 14),
               'mini': (MiniPointNet, packed_mini_forward, 13),
               'feature': (lambda: FeaturePointNet(4), packed_feature_forward, 14)}
    constructor, forward, width = options[kind]
    left = constructor().train().to(dtype)
    right = deepcopy(left)
    right.forward = MethodType(forward, right)
    # FP64 严格核验压缩语义；FP32 的预算相对于同输入/参数的 FP64 参考。
    # Conv1d 与 packed GEMM 的归约顺序不同，dense FP32 本身也有误差。
    oracle = deepcopy(left).double() if dtype == torch.float32 else None
    valid = torch.tensor([[True, False, True, True, False, True, True],
                          [False, True, True, False, True, False, False],
                          [True, False, True, False, True, True, False]])
    if case != 'partial':
        valid.zero_()
        if case == 'singleton':
            valid[1, 3] = True
    x = torch.randn(3, width, 7).to(dtype)
    x.transpose(1, 2)[~valid] = float('nan')
    x.requires_grad_()
    y = x.detach().clone().requires_grad_()
    kwargs = dict(valid_count=int(valid.sum()))
    probability = torch.rand(3, 7, requires_grad=True)
    if kind == 'mini':
        kwargs.update(foreground=probability, row_count=int(valid.any(-1).sum()))
    expected, actual = left(x, valid, **kwargs), right(y, valid, **kwargs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = actual if isinstance(actual, tuple) else (actual,)
    if oracle is not None:
        oracle_kwargs = {key: value.double() if isinstance(value, torch.Tensor) else value
                         for key, value in kwargs.items()}
        oracle_outputs = oracle(x.detach().double(), valid, **oracle_kwargs)
        oracle_outputs = oracle_outputs if isinstance(oracle_outputs, tuple) else (oracle_outputs,)
        loss_oracle = torch.zeros((), dtype=torch.float64)
    loss_a, loss_b = x.new_zeros(()), y.new_zeros(())
    for index, (a, b) in enumerate(zip(expected, actual)):
        if a.dtype == torch.bool:
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=8e-4 if oracle is not None else 1e-8,
                                       atol=1e-4 if oracle is not None else 1e-9)
            weights = torch.randn_like(a)
            loss_a = loss_a + (a * weights).sum()
            loss_b = loss_b + (b * weights).sum()
            if oracle is not None:
                loss_oracle = loss_oracle + (oracle_outputs[index] * weights.double()).sum()
    loss_a.backward()
    loss_b.backward()
    if oracle is not None:
        loss_oracle.backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-3 if oracle is not None else 1e-8,
                               atol=4e-4 if oracle is not None else 1e-9)
    dense_grads, packed_grads, reference_grads = [], [], []
    oracle_parameters = dict(oracle.named_parameters()) if oracle is not None else {}
    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            if oracle is None:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-8, atol=1e-9,
                                           msg=lambda message: name + ': ' + message)
            else:
                dense_grads.append(a.grad.reshape(-1).double())
                packed_grads.append(b.grad.reshape(-1).double())
                reference_grads.append(oracle_parameters[name].grad.reshape(-1))
    if oracle is not None:
        _assert_fp32_reduction_budget(torch.cat(dense_grads), torch.cat(packed_grads),
                                     torch.cat(reference_grads))
    for name, a in left.named_buffers():
        torch.testing.assert_close(a, dict(right.named_buffers())[name],
                                   rtol=1e-4 if oracle is not None else 1e-10,
                                   atol=1e-5 if oracle is not None else 1e-11, msg=name)
    assert probability.grad is None
    assert torch.equal(y.grad.transpose(1, 2)[~valid], torch.zeros_like(y.grad.transpose(1, 2)[~valid]))


def _assert_fp32_reduction_budget(dense, packed, reference):
    """有据可查的数值预算，不声称所有输入的理论误差上界。

    FP64 独立测试核验语义/反传。两条 FP32 路径分别要求相对 FP64 的
    L2 与 L∞ 参数梯度误差 <= 1e-4，即整体向量保留四位相对精度；不要求
    近零单元素相对等价，也不按某个后端 dense 恰好更小的舍入误差定预算。
    CPU torch 2.0.1 的 MiniPointNet 实测 packed 比值为 2.77e-5 / 4.17e-5；
    多层 BN 会放大 GEMM 舍入。正式 FP32/BF16 计算未因此改为 double。
    """
    assert torch.isfinite(dense).all() and torch.isfinite(packed).all()
    for norm in (2, float('inf')):
        scale = torch.linalg.vector_norm(reference, ord=norm)
        for name, value in (('dense', dense), ('packed', packed)):
            observed = torch.linalg.vector_norm(value - reference, ord=norm)
            assert observed <= 1e-4 * scale, (
                f'{name} FP32 L{norm} gradient error {float(observed):.9g} exceeds '
                f'1e-4 * FP64 reference scale {float(scale):.9g}')


def test_configuration_preserves_parameters_state_keys_and_full_observation():
    torch.manual_seed(3904)
    observation = B0Observation(4, query_context=True, coarse_condition=True).eval()
    reference = deepcopy(observation)
    decoder = SharedHypothesisDecoder().eval()
    keys = list(observation.state_dict()), list(decoder.state_dict())
    params = [id(p) for p in observation.parameters()], [id(p) for p in decoder.parameters()]
    rng = torch.get_rng_state().clone()
    configure_efficient_modules(observation, decoder, dict(v39_feature_bf16=False,
                                v39_attention_backend='math_reference'))
    assert keys == (list(observation.state_dict()), list(decoder.state_dict()))
    assert params == ([id(p) for p in observation.parameters()], [id(p) for p in decoder.parameters()])
    assert torch.equal(rng, torch.get_rng_state())
    batch = make_batch(2, 8)
    batch['point_valid'][0, :, 1::2] = False
    batch['point_ids'][1, -1, 2] = batch['point_ids'][1, -1, 0]
    batch['history_is_initial'] = torch.zeros(2, 3, dtype=torch.bool)
    expected, actual = reference(batch), observation(batch)
    for field in fields(expected):
        a, b = getattr(expected, field.name), getattr(actual, field.name)
        if a is None:
            assert b is None
        elif a.dtype == torch.bool:
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5, msg=field.name)
    source, source_valid = decoder.encode_sources(actual, batch['history_valid'])
    assert torch.isfinite(source).all()
    assert (source[~source_valid] == 0).all()
    assert all(m.last_backend == 'math_reference' for m in
               list(decoder.encoder.modules()) + list(decoder.encoder_global.modules())
               if isinstance(m, EfficientAttention))


@pytest.mark.parametrize('case', ('partial', 'empty'))
def test_math_attention_matches_original_zero_row_and_gradients(case):
    torch.manual_seed(3905)
    tensors = [torch.randn(2, 4, length, 64, requires_grad=True) for length in (5, 7, 7)]
    copies = [x.detach().clone().requires_grad_() for x in tensors]
    query = torch.tensor([[True, False, True, False, True], [False]*5])
    keys = torch.tensor([[True, False, True, True, False, False, False], [True]*7])
    mask = query[:, None, :, None] & keys[:, None, None, :]
    if case == 'empty':
        mask.zero_()
    reference = ScaledDotProductAttention(8., attn_dropout=0.).eval()
    efficient = EfficientAttention(8., attn_dropout=0.).eval()
    expected, _ = reference(*tensors, mask=mask, zero_invalid_rows=True)
    actual, weights = efficient(*copies, mask=mask, zero_invalid_rows=True)
    assert weights is None
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    for a, b in zip(tensors, copies):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=3e-6)
    assert torch.isfinite(actual).all()
    assert (actual[~query[:, None, :, None].expand_as(actual)] == 0).all()


def test_validity_mask_preserves_sequence_boundaries_and_rejects_pairwise_masks():
    query = torch.tensor([[True, False, True], [False]*3, [True, True, False]])
    key = torch.tensor([[True, False, True, True], [True]*4, [False, True, False, False]])
    mask = query[:, :, None] & key[:, None, :]
    packed = ValidityMask.from_allowed(mask)
    assert packed.query_indices.tolist() == [0, 2, 6, 7]
    assert packed.key_indices.tolist() == [0, 2, 3, 9]
    assert packed.cu_query.tolist() == [0, 2, 4]
    assert packed.cu_key.tolist() == [0, 3, 4]
    assert torch.equal(packed.dense()[:, 0], mask)
    nonseparable = torch.eye(3, dtype=torch.bool)[None]
    with pytest.raises(ValueError, match='separable'):
        ValidityMask.from_allowed(nonseparable)


def test_formal_flash_path_refuses_cpu_and_fallback_even_for_empty_observations():
    attention = EfficientAttention(8., backend='fa2_varlen')
    value = torch.zeros(1, 4, 2, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='CUDA.*fallback is forbidden'):
        attention(value, value, value, mask=torch.zeros(1, 1, 2, 2, dtype=torch.bool),
                  zero_invalid_rows=True)
    with pytest.raises(ValueError, match='backend'):
        EfficientAttention(8., backend='auto')
    with pytest.raises(ValueError, match='deterministic'):
        EfficientAttention(8., deterministic=False)


def test_attention_eval_disables_dropout_and_train_uses_configured_probability():
    torch.manual_seed(3906)
    q = torch.randn(2, 4, 5, 64)
    attention = EfficientAttention(8., attn_dropout=.75).eval()
    first, _ = attention(q, q, q)
    second, _ = attention(q, q, q)
    assert torch.equal(first, second)
    attention.train()
    first, _ = attention(q, q, q)
    second, _ = attention(q, q, q)
    assert not torch.equal(first, second)


def test_reference_retains_legacy_uniform_empty_row_when_zero_flag_is_off():
    torch.manual_seed(3908)
    q, k, v = (torch.randn(1, 4, n, 64) for n in (3, 5, 5))
    mask = torch.zeros(1, 1, 3, 5, dtype=torch.bool)
    expected, _ = ScaledDotProductAttention(8., 0.).eval()(q, k, v, mask=mask)
    actual, _ = EfficientAttention(8., 0.).eval()(q, k, v, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


@pytest.mark.parametrize('temperature', (8., 5.))
def test_math_attention_uses_torch20_sdpa_signature_and_temperature(monkeypatch, temperature):
    from torch.nn import functional as functional
    original = functional.scaled_dot_product_attention
    calls = []

    # torch 2.0 的关键字签名没有 scale；新 torch 下也用相同限制回归。
    def torch20_sdpa(q, k, v, *, attn_mask=None, dropout_p=0., is_causal=False):
        calls.append(q)
        return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    monkeypatch.setattr(functional, 'scaled_dot_product_attention', torch20_sdpa)
    torch.manual_seed(3910)
    q, k, v = (torch.randn(2, 4, n, 64) for n in (3, 5, 5))
    actual, _ = EfficientAttention(temperature, 0.).eval()(q, k, v)
    expected, _ = ScaledDotProductAttention(temperature, 0.).eval()(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert len(calls) == 1
    if temperature == 8.:
        assert calls[0] is q


@pytest.mark.parametrize('version,accepted', (('2.5.8', True), ('2.5.8+cu118torch2.0cxx11abiFALSE', True),
                                             ('2.5.7', False), ('2.6.0', False), ('', False)))
def test_flash_extension_enforces_registered_version_with_local_suffix(monkeypatch, version, accepted):
    module = ModuleType('flash_attn')
    module.__version__ = version
    module.flash_attn_varlen_func = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'flash_attn', module)
    _flash_varlen.cache_clear()
    try:
        if accepted:
            assert _flash_varlen() is module.flash_attn_varlen_func
        else:
            with pytest.raises(RuntimeError, match='registered flash-attn==2.5.8'):
                _flash_varlen()
    finally:
        _flash_varlen.cache_clear()


def test_bf16_packed_rejects_cpu_before_any_bn_update():
    model = SegPointNet().train()
    model.forward = MethodType(packed_seg_forward, model)
    before = {key: value.clone() for key, value in model.named_buffers()}
    with pytest.raises(RuntimeError, match='CUDA'):
        model(torch.randn(2, 14, 7), torch.ones(2, 7, dtype=torch.bool), feature_bf16=True)
    for key, value in model.named_buffers():
        assert torch.equal(value, before[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/FA2 validation runs in the registered GPU environment')
def test_cuda_fa2_varlen_forward_backward_matches_math():
    pytest.importorskip('flash_attn')
    if not torch.cuda.is_bf16_supported():
        pytest.skip('native BF16 required')
    torch.manual_seed(3907)
    tensors = [torch.randn(3, 4, length, 64, device='cuda', dtype=torch.bfloat16,
                           requires_grad=True) for length in (5, 7, 7)]
    copies = [x.detach().clone().requires_grad_() for x in tensors]
    query = torch.tensor([[True, False, True, True, False], [False]*5, [True]*5], device='cuda')
    key = torch.tensor([[True, False, True, True, False, False, False], [False]*7,
                        [True, False, True, False, True, False, True]], device='cuda')
    mask = query[:, None, :, None] & key[:, None, None, :]
    reference = EfficientAttention(8.).eval()
    flash = EfficientAttention(8., backend='fa2_varlen').eval()
    expected, _ = reference(*tensors, mask=mask, zero_invalid_rows=True)
    actual, _ = flash(*copies, mask=mask, zero_invalid_rows=True)
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.02)
    weights = torch.randn_like(actual)
    expected.backward(weights)
    actual.backward(weights)
    for a, b in zip(tensors, copies):
        torch.testing.assert_close(a.grad, b.grad, rtol=.05, atol=.03)
    assert flash.last_backend == 'fa2_varlen'
    assert flash.kernel_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='packed CUDA/BF16 preflight requires the registered GPU environment')
def test_cuda_packed_bf16_training_keeps_fp32_outputs_stats_and_finite_gradients():
    if not torch.cuda.is_bf16_supported():
        pytest.skip('native BF16 required')
    torch.manual_seed(3909)
    model = SegPointNet().cuda().train()
    model.forward = MethodType(packed_seg_forward, model)
    x = torch.randn(3, 14, 31, device='cuda', requires_grad=True)
    valid = torch.arange(31, device='cuda')[None] < torch.tensor([19, 0, 7], device='cuda')[:, None]
    output, features = model(x, valid, feature_bf16=True)
    assert output.dtype == features.dtype == torch.float32
    assert torch.isfinite(output).all() and torch.isfinite(features).all()
    (output.square().mean() + features.square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            assert module.running_mean.dtype == module.running_var.dtype == torch.float32
            assert int(module.num_batches_tracked) == 1
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
