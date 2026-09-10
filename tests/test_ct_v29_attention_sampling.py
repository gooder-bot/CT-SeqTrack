"""v29 物理帧、缺失帧注意力和真实稀疏槽；旧路径保留精确行为。"""
import copy

import numpy as np
import pytest
import torch

from models.attn.Models import Seq2SeqFormer
from models.attn.Modules import ScaledDotProductAttention
from utils.b0_sampling import regularize_b0_seqtrack_compat, regularize_b0_sparse_v29
from utils.v29_observation import frame_aligned_pointnet_input
from tests.test_ct_v27_input_flow import sampler_runtime  # noqa: F401


def test_v29_frame_layout_preserves_each_batch_frame_channel_point_and_gradient():
    # B=2,C=14,L=4 是正式 box-aware 布局，多个 batch 排除仅单样本正确的误修。
    original = torch.arange(2 * 4 * 14 * 7, dtype=torch.float64).reshape(2, 4, 14, 7)
    x = original.permute(0, 2, 1, 3).reshape(2, 14, 28).requires_grad_()
    aligned = frame_aligned_pointnet_input(x, 4)
    assert torch.equal(aligned, original.reshape(8, 14, 7))
    assert not torch.equal(aligned, x.reshape(8, 14, 7))
    weights = torch.arange(aligned.numel(), dtype=x.dtype).reshape_as(aligned)
    (aligned * weights).sum().backward()
    expected = weights.reshape(2, 4, 14, 7).permute(0, 2, 1, 3).reshape_as(x)
    assert torch.equal(x.grad, expected)


@pytest.mark.parametrize('shape,frames', [((2, 14, 31), 4), ((2, 14, 0), 4),
                                        ((2, 14, 32), 0), ((14, 32), 4)])
def test_v29_layout_rejects_ambiguous_frame_slots(shape, frames):
    with pytest.raises(ValueError):
        frame_aligned_pointnet_input(torch.zeros(shape), frames)


def _former():
    torch.manual_seed(17)
    return Seq2SeqFormer(d_word_vec=8, d_model=8, d_inner=16, n_layers=2,
                        n_head=2, d_k=4, d_v=4, dropout=0.0).double().eval()


def _inputs():
    generator = torch.Generator().manual_seed(42)
    return (torch.randn(2, 32, 4, generator=generator, dtype=torch.float64),
            torch.randn(2, 4 * 3, 128, generator=generator, dtype=torch.float64))


def test_v28_default_still_ignores_history_mask_and_keeps_decoder_export_identical():
    model = _former()
    corners, source = _inputs()
    one = torch.ones(2, 3)
    zero = torch.zeros_like(one)
    expected = model(corners, source, one)
    output, state = model(corners, source, zero, return_decoder_state=True,
                          enable_v29=False, frame_measurement_valid=torch.zeros(2, 4))
    assert torch.equal(output, expected)
    assert state.shape == (2, 4, 8)
    assert torch.equal(model.l2(state), output)


def test_v29_all_valid_is_bitwise_equal_to_existing_attention_outputs_and_gradients():
    old = _former()
    new = copy.deepcopy(old)
    corners, source = _inputs()
    corners_old, source_old = corners.clone().requires_grad_(), source.clone().requires_grad_()
    corners_new, source_new = corners.clone().requires_grad_(), source.clone().requires_grad_()
    valid = torch.ones(2, 3, dtype=torch.bool)
    output_old, state_old = old(corners_old, source_old, valid, return_decoder_state=True)
    output_new, state_new = new(corners_new, source_new, valid, return_decoder_state=True,
                                enable_v29=True, frame_measurement_valid=torch.ones(2, 4, dtype=torch.bool))
    assert torch.equal(output_old, output_new) and torch.equal(state_old, state_new)
    (output_old.square().sum() + state_old.square().sum()).backward()
    (output_new.square().sum() + state_new.square().sum()).backward()
    assert torch.equal(corners_old.grad, corners_new.grad)
    assert torch.equal(source_old.grad, source_new.grad)
    for (name_old, parameter_old), (name_new, parameter_new) in zip(old.named_parameters(), new.named_parameters()):
        assert name_old == name_new
        assert torch.equal(parameter_old.grad, parameter_new.grad), name_old


def test_v29_missing_history_and_empty_measurements_cannot_leak_through_attention():
    model = _former()
    corners, source = _inputs()
    exists = torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.bool)
    measured = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]], dtype=torch.bool)
    changed_corners = corners.clone()
    changed_corners[:, 8:16] = 1e6  # 不存在历史，角点 query 无效。
    changed_source = source.clone()
    changed_source[:, 3:9] = -1e6  # 第1帧不存在；第2帧存在但无测量。
    args = dict(enable_v29=True, frame_measurement_valid=measured, return_decoder_state=True)
    before = model(corners, source, exists, **args)
    after = model(changed_corners, changed_source, exists, **args)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert torch.count_nonzero(before[0][:, 1]) == 0
    assert torch.count_nonzero(before[1][:, 1]) == 0
    assert torch.count_nonzero(before[1][:, -1]) > 0  # 当前query保留。

    corners.requires_grad_()
    source.requires_grad_()
    boxes, state = model(corners, source, exists, **args)
    (boxes.square().sum() + state.square().sum()).backward()
    assert torch.count_nonzero(corners.grad[:, 8:16]) == 0
    assert torch.count_nonzero(source.grad[:, 3:9]) == 0
    assert torch.isfinite(corners.grad).all() and torch.isfinite(source.grad).all()


def test_v29_all_empty_sources_zero_attention_and_encoder_outputs_after_ffn_bias():
    model = _former()
    corners, source = _inputs()
    observed_attention = []
    observed_sources = []
    handles = []
    for module in model.modules():
        if isinstance(module, ScaledDotProductAttention):
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, output: observed_attention.append(output)))
    for module in (model.encoder, model.encoder_global):
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output: observed_sources.append(output[0])))
    try:
        boxes, state = model(corners, source, torch.zeros(2, 3), enable_v29=True,
                             frame_measurement_valid=torch.zeros(2, 4), return_decoder_state=True)
    finally:
        for handle in handles:
            handle.remove()
    assert observed_attention and observed_sources
    for values, weights in observed_attention:
        assert torch.count_nonzero(values) == torch.count_nonzero(weights) == 0
    assert all(torch.count_nonzero(value) == 0 for value in observed_sources)
    assert torch.isfinite(boxes).all() and torch.isfinite(state).all()
    assert torch.count_nonzero(boxes[:, :3]) == torch.count_nonzero(state[:, :3]) == 0
    # 当前corner query仍存在；整个B0的reference-box hold由host而非attention层执行。


@pytest.mark.parametrize('measurement', [None, torch.ones(2, 3)])
def test_v29_rejects_missing_or_misaligned_measurement_mask(measurement):
    corners, source = _inputs()
    with pytest.raises(ValueError, match='measurement validity'):
        _former()(corners, source, torch.ones(2, 3), enable_v29=True,
                  frame_measurement_valid=measurement)


@pytest.mark.parametrize('count', [0, 1, 2, 3, 16, 21])
def test_v29_sparse_sampler_preserves_real_ids_and_v28_zero_or_dense_path(sampler_runtime, count):
    points = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    sampled, indices = regularize_b0_sparse_v29(points, 16, seed=42)
    legacy, legacy_indices = regularize_b0_seqtrack_compat(points, 16, seed=42)
    assert sampled.shape == (16, 3)
    if count == 0:
        assert indices is None and legacy_indices is None
        np.testing.assert_array_equal(sampled, legacy)
    elif count <= 2:
        assert legacy_indices is None and np.count_nonzero(legacy) == 0
        assert indices is not None and set(indices) == set(range(count))
        np.testing.assert_array_equal(sampled, points[indices])
        # 重复采样槽始终能追溯到最多count个物理点，不能变成16个新增测量。
        assert len(np.unique(indices)) == count
    else:
        np.testing.assert_array_equal(sampled, legacy)
        np.testing.assert_array_equal(indices, legacy_indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual CUDA strict-attention backward')
def test_v29_cuda_strict_masked_attention_backward_repeats_bitwise(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    snapshots = []
    try:
        for _ in range(2):
            torch.manual_seed(42)
            module = ScaledDotProductAttention(temperature=2.0, attn_dropout=0.0).cuda()
            tensors = [torch.randn(2, 2, 4, 8, device='cuda', requires_grad=True) for _ in range(3)]
            mask = torch.tensor([[[[1, 0, 1, 0]]], [[[0, 0, 0, 0]]]], device='cuda', dtype=torch.bool)
            output, attention = module(*tensors, mask=mask, zero_invalid_rows=True)
            output.square().sum().backward()
            snapshots.append([output.detach().cpu(), attention.detach().cpu()]
                             + [tensor.grad.cpu() for tensor in tensors])
            assert torch.count_nonzero(output[1]) == torch.count_nonzero(attention[1]) == 0
        assert all(torch.equal(a, b) for a, b in zip(*snapshots))
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
