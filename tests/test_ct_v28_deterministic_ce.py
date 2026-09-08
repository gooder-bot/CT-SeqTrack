"""v28 CUDA 空间 NLL 修复：整批目标不变，严格确定性保持开启。"""

import pytest
import torch
import torch.nn.functional as F

from models.ct_v2.observation_reference import seqtrack_segmentation_cross_entropy


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
