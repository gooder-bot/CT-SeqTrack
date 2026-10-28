"""G 混合精度审计实际 Linear，允许 LayerNorm 保留 FP32 出口。"""
from contextlib import nullcontext
from copy import deepcopy
import os

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.ct_v39.g_observation import GObservation
from tests.test_ct_v39_g import batch_fixture
from tools.check_v39_full_batch import GPrecisionAudit


def emulate_cuda_autocast_on_cpu(monkeypatch, model):
    """只模拟 CUDA 的 dtype 边界；不是实际 CUDA 数值/性能证据。"""
    import models.ct_v39.g_observation as implementation
    monkeypatch.setattr(implementation, 'feature_autocast',
        lambda value, enabled=False: torch.autocast('cpu', dtype=torch.bfloat16) if enabled else nullcontext())
    for layer in model.modules():
        if isinstance(layer, nn.LayerNorm):
            def fp32_norm(value, layer=layer):
                with torch.autocast('cpu', enabled=False):
                    return F.layer_norm(value.float(), layer.normalized_shape,
                                        layer.weight.float(), layer.bias.float(), layer.eps)
            monkeypatch.setattr(layer, 'forward', fp32_norm)


def encode_and_heads(model):
    pool, _, _, size, _ = batch_fixture(batch=1)
    encoded = model.encode(pool, size)
    targetness = model._logit_head(model.targetness, torch.ones(2, 132))
    logits = model._logit_head(model.head, torch.ones(2, 149))
    return encoded, targetness, logits


def test_bf16_linears_with_fp32_norm_output_pass_audit(monkeypatch):
    model = GObservation(feature_bf16=True)
    emulate_cuda_autocast_on_cpu(monkeypatch, model)
    with GPrecisionAudit(model) as audit:
        encoded, targetness, logits = encode_and_heads(model)
        # 原报错恰是把这个合法 FP32 出口当作 BF16 没有执行。
        assert encoded['features'].dtype == torch.float32
        assert targetness.dtype == logits.dtype == torch.float32
        audit.assert_bf16()
        rows = audit.summary()
        for row in rows.values():
            assert row['calls'] > 0
            expected = 'torch.bfloat16' if row['role'] == 'feature_linear' else 'torch.float32'
            assert row['dtypes'] == [expected]


def test_fp32_linears_cannot_pass_by_casting_only_final_features():
    model = GObservation(feature_bf16=False)
    with GPrecisionAudit(model) as audit:
        encoded, _, _ = encode_and_heads(model)
        cast_output = encoded['features'].to(torch.bfloat16)
        assert cast_output.dtype == torch.bfloat16
        with pytest.raises(RuntimeError, match='precision mismatch.*edge_mlp'):
            audit.assert_bf16()


@pytest.mark.parametrize('group,width', [('targetness', 132), ('head', 149)])
def test_output_head_bf16_is_rejected(monkeypatch, group, width):
    model = GObservation(feature_bf16=True)
    emulate_cuda_autocast_on_cpu(monkeypatch, model)
    pool, _, _, size, _ = batch_fixture(batch=1)
    with GPrecisionAudit(model) as audit:
        model.encode(pool, size)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            wrong = getattr(model, group)(torch.ones(2, width))
        assert wrong.dtype == torch.bfloat16
        with pytest.raises(RuntimeError, match='precision mismatch.*' + group + '.*expected torch.float32'):
            audit.assert_bf16()


@pytest.mark.parametrize('partial', [False, True])
def test_missing_encoder_execution_cannot_pass(partial):
    model = GObservation()
    with GPrecisionAudit(model) as audit:
        if partial:
            model.edge_mlp(torch.ones(2, 4))
        with pytest.raises(RuntimeError, match='every encoder Linear'):
            audit.assert_bf16()


def test_audit_preserves_forward_gradients_and_removes_hooks():
    torch.manual_seed(71)
    plain = GObservation()
    checked = deepcopy(plain)
    outputs = []
    for model, auditing in ((plain, False), (checked, True)):
        with GPrecisionAudit(model) if auditing else nullcontext():
            encoded, targetness, logits = encode_and_heads(model)
            features = encoded['features']
            (features.square().mean() + targetness.square().mean() + logits.square().mean()).backward()
            outputs.append((features.detach(), targetness.detach(), logits.detach()))
        assert all(not module._forward_hooks for module in model.modules())
    for expected, actual in zip(*outputs):
        assert torch.equal(expected, actual)
    for expected, actual in zip(plain.parameters(), checked.parameters()):
        assert (expected.grad is None) == (actual.grad is None)
        if expected.grad is not None:
            assert torch.equal(expected.grad, actual.grad)


def test_audit_removes_hooks_when_body_raises():
    model = GObservation()
    with pytest.raises(RuntimeError, match='fixture failure'):
        with GPrecisionAudit(model) as audit:
            encode_and_heads(model)
            raise RuntimeError('fixture failure')
    assert not audit.handles
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires real CUDA BF16 G execution')
def test_cuda_g_linears_bf16_geometry_and_logits_fp32():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_bf16_supported():
        pytest.skip('native CUDA BF16 unavailable')
    model = GObservation(feature_bf16=True).cuda().eval()
    pool, templates, boxes, size, _ = batch_fixture(batch=1)
    pool = {key: value.cuda() for key, value in pool.items()}
    templates = {key: value.cuda() for key, value in templates.items()}
    with torch.no_grad(), GPrecisionAudit(model) as audit:
        result = model.verify(pool, templates, boxes.cuda(), size.cuda())
        audit.assert_bf16()
        assert result['logits'].dtype == result['q_logits'].dtype == torch.float32
        assert result['encoded_pool']['xyz'].dtype == torch.float32
        assert all(torch.isfinite(result[key]).all() for key in ('logits', 'q_logits', 'region_no_match_logits'))
