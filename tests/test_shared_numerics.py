"""当前模型共享数值原语：保留历史修复的独立数值与边界回归。"""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.ct_v31.cfc import FullGatedCfCCell
from models.ct_v31.losses import seqtrack_segmentation_cross_entropy
from models.ct_v31.motion import motion_aligned_axes, motion_aligned_covariance, physical_motion_uncertainty_loss
from models.attn.Modules import ScaledDotProductAttention
from utils.masked_observation import masked_batch_norm, masked_max_pool
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('case', ('background', 'foreground', 'imbalanced', 'ignored'))
def test_segmentation_ce_preserves_weighted_batch_loss_and_gradient(dtype, case):
    generator = torch.Generator().manual_seed(42)
    # 实际分割输出是 11 通道中的前 2 个，保留其非连续布局。
    source = torch.randn(4, 11, 4096, generator=generator, dtype=dtype,
                         requires_grad=True)
    logits = source[:, :2]
    labels = torch.zeros(4, 4096, dtype=torch.long)
    if case == 'foreground':
        labels.fill_(1)
    elif case in ('imbalanced', 'ignored'):
        labels[0] = 1
        labels[1, :13] = 1
        labels[3, :3] = 1
        if case == 'ignored':
            labels[2, :7] = -100
    expected = F.cross_entropy(logits, labels, weight=logits.new_tensor([.5, 2.]))
    actual = seqtrack_segmentation_cross_entropy(logits, labels)
    expected_gradient = torch.autograd.grad(expected, source, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, source)[0]
    assert torch.equal(actual, expected)
    assert torch.equal(actual_gradient, expected_gradient)


def test_segmentation_ce_uses_original_class_axis_and_only_two_dimensional_nll(monkeypatch):
    log_softmax, nll_loss = F.log_softmax, F.nll_loss
    calls = []

    def record_softmax(value, dim):
        calls.append(('softmax', tuple(value.shape), dim))
        return log_softmax(value, dim=dim)

    def record_nll(value, target, **kwargs):
        calls.append(('nll', tuple(value.shape), tuple(target.shape), kwargs['reduction']))
        return nll_loss(value, target, **kwargs)

    monkeypatch.setattr(F, 'log_softmax', record_softmax)
    monkeypatch.setattr(F, 'nll_loss', record_nll)
    seqtrack_segmentation_cross_entropy(torch.zeros(16, 2, 4096),
                                       torch.zeros(16, 4096, dtype=torch.long))
    assert calls == [('softmax', (16, 2, 4096), 1),
                     ('nll', (65536, 2), (65536,), 'mean')]


@pytest.mark.parametrize('device', (
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires actual CUDA runtime'))))
def test_strict_segmentation_ce_repeats_loss_gradient_and_adam_update(device):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    generator = torch.Generator().manual_seed(42)
    initial = torch.randn(16, 11, 4096, generator=generator)
    labels = torch.randint(0, 2, (16, 4096), generator=generator).to(device)
    snapshots = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        for _ in range(2):
            parameter = torch.nn.Parameter(initial.to(device).clone())
            optimizer = torch.optim.Adam([parameter], lr=1e-4, betas=(.5, .999),
                                         eps=1e-6, foreach=False, fused=False)
            loss = seqtrack_segmentation_cross_entropy(parameter[:, :2], labels)
            loss.backward()
            gradient = parameter.grad.detach().clone()
            optimizer.step()
            state = optimizer.state[parameter]
            snapshots.append((loss.detach(), gradient, parameter.detach().clone(),
                              state['exp_avg'].clone(), state['exp_avg_sq'].clone()))
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        for first, second in zip(*snapshots):
            assert torch.equal(first, second)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


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
def test_masked_bn_matches_reference_forward_backward_adam_and_single_state_update(dtype, shape, affine, momentum):
    torch.manual_seed(37)
    x = torch.randn(*shape, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = torch.ones(shape[:1] + shape[2:], dtype=torch.bool)
    mask.reshape(-1)[::3] = False
    first = nn.BatchNorm1d(shape[1], affine=affine, momentum=momentum).to(dtype).train()
    second = copy.deepcopy(first)
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


def test_masked_ce_ignores_padding_and_preserves_weighted_denominator():
    logits = torch.tensor([[[1., 2., 999.], [2., 1., -999.]]], requires_grad=True)
    labels = torch.tensor([[1, 0, -1]])
    valid = torch.tensor([[True, True, False]])
    actual = seqtrack_segmentation_cross_entropy(logits, labels, valid)
    expected = F.cross_entropy(logits[:, :, :2], labels[:, :2], weight=torch.tensor([.5, 2.]))
    assert torch.equal(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad[:, :, 2]) == 0
    empty = seqtrack_segmentation_cross_entropy(logits, labels, torch.zeros_like(valid))
    assert empty.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(empty, logits)[0]) == 0


@pytest.mark.parametrize('count', [0, 1, 8])
def test_masked_bn_empty_singleton_and_all_valid_preserve_correct_state(count):
    torch.manual_seed(12)
    value = torch.randn(2, 3, 4, requires_grad=True)
    valid = torch.arange(8).reshape(2, 4) < count
    module = nn.BatchNorm1d(3).train()
    original = copy.deepcopy(module)
    before = {name: state.clone() for name, state in module.named_buffers()}
    actual = masked_batch_norm(value, valid, module)
    if count == 8:
        expected = original(value)
        assert torch.equal(actual, expected)
        assert all(torch.equal(value, dict(original.named_buffers())[name])
                   for name, value in module.named_buffers())
    else:
        expected = F.batch_norm(value, original.running_mean, original.running_var,
                                original.weight, original.bias, False, 0., original.eps)
        expected = torch.where(valid[:, None], expected, 0.)
        assert torch.equal(actual, expected)
        assert all(torch.equal(value, before[name]) for name, value in module.named_buffers())
    actual.square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(value.grad.masked_select(~valid[:, None])) == 0


def test_masked_pool_empty_buckets_and_first_max_gradient():
    value = torch.tensor([[[3., 3., 9., 2., 8., 7., 6., 5.]]], requires_grad=True)
    valid = torch.tensor([[True, True, False, False, False, False, True, True]])
    output, token_valid = masked_max_pool(value, valid, 4)
    assert torch.equal(output, torch.tensor([[[3., 0., 0., 6.]]]))
    assert torch.equal(token_valid, torch.tensor([[True, False, False, True]]))
    output.sum().backward()
    assert torch.equal(value.grad, torch.tensor([[[1., 0., 0., 0., 0., 0., 1., 0.]]]))


def test_cfc_parameter_layout_and_real_time_sensitivity():
    torch.manual_seed(7)
    cell = FullGatedCfCCell(64, 128, 105).eval()
    assert sum(parameter.numel() for parameter in cell.parameters()) == 74537
    assert list(cell.state_dict()) == [
        'backbone.0.weight', 'backbone.0.bias', 'first_state.weight', 'first_state.bias',
        'second_state.weight', 'second_state.bias', 'time_a.weight', 'time_a.bias',
        'time_b.weight', 'time_b.bias']
    inputs, hidden = torch.randn(3, 64), torch.randn(3, 128)
    short = cell(inputs, hidden, torch.full((3,), .5))
    long = cell(inputs, hidden, torch.full((3,), 2.))
    assert not torch.allclose(short, long)
    separate = torch.cat([cell(inputs[i:i + 1], hidden[i:i + 1], .5) for i in range(3)])
    torch.testing.assert_close(short, separate, atol=1e-6, rtol=1e-6)
    (short.square().mean() + long.square().mean()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in cell.parameters())
    assert any(torch.count_nonzero(parameter.grad) for parameter in cell.time_a.parameters())
    for elapsed in (-.1, float('nan'), torch.zeros(4)):
        with pytest.raises(ValueError):
            cell(inputs, hidden, elapsed)


def test_motion_frame_stationary_fallback_and_rotated_covariance():
    velocity = torch.tensor([[0., 0.], [0., 2.]])
    direction, perpendicular, speed = motion_aligned_axes(velocity)
    assert torch.equal(direction, torch.tensor([[1., 0.], [0., 1.]]))
    assert torch.equal(perpendicular, torch.tensor([[0., 1.], [-1., 0.]]))
    assert torch.equal(speed, torch.tensor([0., 2.]))
    output = motion_aligned_covariance(velocity, torch.tensor([[2., 8.], [2., 8.]]).sqrt().log())
    torch.testing.assert_close(output['covariance_xy'], torch.tensor([[[4., 0.], [0., 4.]],
                                                                    [[8., 0.], [0., 2.]]]))


def test_physical_sigma_loss_cannot_train_mean_or_residual():
    mean = torch.tensor([[.1, .2], [.3, -.1]], requires_grad=True)
    sigma = torch.zeros(2, 2, requires_grad=True)
    residual = torch.zeros(2, 2, requires_grad=True)
    terms = physical_motion_uncertainty_loss(mean, torch.tensor([[.5, .3], [70., -70.]]),
        sigma, torch.tensor([[1., 0.], [1., 0.]]), torch.ones(2),
        kinematic_xy=torch.zeros(2, 2), envelope_parallel_perp=torch.ones(2, 2),
        residual_unit_parallel_perp=residual)
    assert torch.isfinite(terms['nll_per_sample']).all()
    assert terms['tail_axis_fraction_per_sample'][1].item() == 1.
    terms['nll_per_sample'].sum().backward(retain_graph=True)
    assert mean.grad is None and residual.grad is None
    assert torch.isfinite(sigma.grad).all() and torch.count_nonzero(sigma.grad)
    sigma.grad = None
    terms['mean_per_sample'].sum().backward()
    assert sigma.grad is None
    assert torch.isfinite(residual.grad).all() and torch.count_nonzero(residual.grad)


def test_attention_empty_rows_and_forbidden_keys_have_zero_gradient():
    torch.manual_seed(4)
    attention = ScaledDotProductAttention(temperature=2., attn_dropout=0.)
    query = torch.randn(1, 1, 2, 4, requires_grad=True)
    key = torch.randn(1, 1, 3, 4, requires_grad=True)
    value = torch.randn(1, 1, 3, 4, requires_grad=True)
    mask = torch.tensor([[[[False, False, False], [True, False, True]]]])
    output, weights = attention(query, key, value, mask, zero_invalid_rows=True)
    assert torch.count_nonzero(output[:, :, 0]) == torch.count_nonzero(weights[:, :, 0]) == 0
    assert weights[0, 0, 1, 1].item() == 0.
    changed = value.detach().clone()
    changed[:, :, 1] = 1e9
    again, _ = attention(query, key, changed, mask, zero_invalid_rows=True)
    assert torch.equal(output, again)
    output.square().sum().backward()
    for tensor in (query, key, value):
        assert torch.isfinite(tensor.grad).all()
    assert torch.count_nonzero(query.grad[:, :, 0]) == 0
    assert torch.count_nonzero(key.grad[:, :, 1]) == torch.count_nonzero(value.grad[:, :, 1]) == 0


def test_metric_geometry_and_benchmark_compat_remain_distinct():
    first = LocalYawBox([0, 0, 0, 0], [2, 2, 2])
    taller = LocalYawBox([0, 0, -1, 0], [2, 2, 4])
    assert box_metrics(first, taller, up_axis=(0, 0, 1), mode='benchmark_compat')[0] == pytest.approx(.2)
    assert box_metrics(first, taller, up_axis=(0, 0, 1), mode='geometry_exact')[0] == pytest.approx(.5)
    shifted = LocalYawBox([3, 4, 0, 0], [2, 2, 2])
    assert box_metrics(first, shifted, up_axis=(0, 0, 1), dim=2)[1] == 0.
    assert box_metrics(first, shifted, up_axis=(0, 0, 1), dim=2, mode='geometry_exact')[1] == 5.


def test_metric_threshold_rounding_and_endpoint_denominator():
    success_thresholds = torch.linspace(0, 1, 21).numpy()
    precision_thresholds = torch.linspace(0, 2, 21).numpy()
    for iou, distance in zip(success_thresholds, precision_thresholds):
        success, precision = metric_contributions(iou, distance)
        success_curve = (iou >= success_thresholds).astype(np.float64)
        precision_curve = (distance <= precision_thresholds).astype(np.float64)
        expected_s = np.trapz(success_curve, success_thresholds.astype(np.float64))
        expected_p = np.trapz(precision_curve, precision_thresholds.astype(np.float64)) / 2
        assert success == expected_s and precision == expected_p
    assert metric_contributions(1., 0.) == (1., 1.)
    assert metric_contributions(0., 3.)[0] == pytest.approx(.025)


@pytest.mark.parametrize('period', [0, 2])
def test_checkpoint_keeps_marker_identity_and_final_epoch_schedule(tmp_path, period):
    pytest.importorskip('pytorch_lightning')
    from pytorch_lightning.callbacks import Checkpoint
    from utils.lightning_runtime import FinalWindowCheckpoint
    saved = []
    trainer = SimpleNamespace(max_epochs=60, default_root_dir=str(tmp_path),
                              current_epoch=0, save_checkpoint=lambda path: saved.append(Path(path)))
    callback = FinalWindowCheckpoint(keep=3, every_n_epochs=period)
    assert isinstance(callback, Checkpoint)
    expected_key = 'ct_seqtrack.FinalWindowCheckpoint.keep=3.dir=formal_checkpoints'
    assert callback.state_key == expected_key + ('.every=2' if period else '')
    for epoch in range(60):
        trainer.current_epoch = epoch
        callback.on_train_epoch_end(trainer, None)
    expected = [epoch for epoch in range(1, 61) if epoch >= 58 or (period and epoch % period == 0)]
    assert [path.name for path in saved] == [f'epoch={epoch:03d}.ckpt' for epoch in expected]
    assert all(path.parent.name == 'formal_checkpoints' for path in saved)
