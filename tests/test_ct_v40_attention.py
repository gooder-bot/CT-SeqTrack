"""窄 FA2：mask/空行、主参数精度、实际 CUDA kernel；写入后不在本机执行。"""
import pytest
import torch

from models.attn.Modules import ScaledDotProductAttention
from models.ct_v40.attention import (EfficientAttention, ValidityMask,
    configure_attention_modules, validate_fa_runtime)
from models.ct_v40.decoder import SharedHypothesisDecoder


def test_math_matches_original_attention_and_empty_row_gradients():
    generator = torch.Generator().manual_seed(40)
    inputs = [torch.randn(shape, generator=generator, requires_grad=True) for shape in
              ((2, 4, 3, 64), (2, 4, 5, 64), (2, 4, 5, 64))]
    copies = [value.detach().clone().requires_grad_() for value in inputs]
    query = torch.tensor([[True, False, True], [False, False, False]])
    key = torch.tensor([[True, False, True, True, False], [False] * 5])
    mask = query[:, None, :, None] & key[:, None, None, :]
    expected, _ = ScaledDotProductAttention(8., 0.).eval()(*inputs, mask, zero_invalid_rows=True)
    actual, _ = EfficientAttention(8., 0., backend='math').eval()(*copies, mask, zero_invalid_rows=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for reference, value in zip(inputs, copies):
        torch.testing.assert_close(value.grad, reference.grad, rtol=3e-5, atol=3e-6)
    assert actual[1].eq(0).all() and copies[0].grad[1].eq(0).all()


def test_validity_rejects_nonseparable_masks_and_preserves_sparse_positions():
    with pytest.raises(ValueError, match='separable'):
        ValidityMask.from_allowed(torch.eye(3, dtype=torch.bool)[None])
    mask = torch.tensor([[[True, False, True], [False, False, False]]])
    validity = ValidityMask.from_allowed(mask)
    assert validity.query_indices.tolist() == [0]
    assert validity.key_indices.tolist() == [0, 2]
    assert validity.cu_query.tolist() == [0, 1]
    assert validity.cu_key.tolist() == [0, 2]
    assert torch.equal(validity.dense()[:, 0], mask)


def test_installer_only_changes_nine_parameterless_cores():
    decoder = SharedHypothesisDecoder()
    before = {name: parameter for name, parameter in decoder.named_parameters()}
    rng = torch.get_rng_state().clone()
    configure_attention_modules(decoder, backend='math')
    after = dict(decoder.named_parameters())
    assert before.keys() == after.keys()
    assert all(before[name] is after[name] for name in before)
    assert torch.equal(rng, torch.get_rng_state())
    cores = [module for module in decoder.modules() if isinstance(module, EfficientAttention)]
    assert len(cores) == 9
    assert all(not list(core.parameters()) for core in cores)
    assert all(parameter.dtype == torch.float32 for parameter in decoder.parameters())


def test_fa_cpu_fails_without_silent_fallback_and_runtime_reports_cuda(monkeypatch):
    q = torch.zeros(1, 4, 2, 64)
    with pytest.raises(RuntimeError, match='CUDA'):
        EfficientAttention(8., backend='fa2_varlen')(q, q, q, zero_invalid_rows=True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        validate_fa_runtime()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires real CUDA and FA2 2.5.8')
@pytest.mark.parametrize('empty', (False, True))
def test_cuda_fa_core_casts_only_kernel_inputs_and_returns_fp32(empty):
    validate_fa_runtime()
    generator = torch.Generator(device='cuda').manual_seed(40)
    inputs = [torch.randn(shape, generator=generator, device='cuda', requires_grad=True) for shape in
              ((2, 4, 8, 64), (2, 4, 16, 64), (2, 4, 16, 64))]
    query = torch.ones(2, 8, dtype=torch.bool, device='cuda')
    key = torch.ones(2, 16, dtype=torch.bool, device='cuda')
    query[0, 2:5] = False
    key[0, 3:8] = False
    if empty:
        key.zero_()
    mask = query[:, None, :, None] & key[:, None, None, :]
    flash = EfficientAttention(8., 0., backend='fa2_varlen').eval()
    result, _ = flash(*inputs, mask, zero_invalid_rows=True)
    quantized = [value.detach().to(torch.bfloat16).requires_grad_() for value in inputs]
    reference, _ = EfficientAttention(8., 0., backend='math').eval()(
        *quantized, mask, zero_invalid_rows=True)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, reference.float(), rtol=.03, atol=.02)
    result.square().sum().backward()
    reference.float().square().sum().backward()
    for source, baseline in zip(inputs, quantized):
        assert source.grad.dtype == torch.float32 and torch.isfinite(source.grad).all()
        torch.testing.assert_close(source.grad, baseline.grad.float(), rtol=.05, atol=.03)
    assert flash.last_backend == 'fa2_varlen'
    assert flash.kernel_calls == (0 if empty else 1)
    if not empty:
        assert flash.last_kernel_dtype == 'torch.bfloat16'
