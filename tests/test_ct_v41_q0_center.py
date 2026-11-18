"""q0 局部轴损失、端点统计和 Full 主辅梯度的配对检查。"""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from models.ct_v41.base_losses import compute_losses, observation_loss_statistics
from models.ct_v41.config import normalize_config
from models.ct_v41.model import JointTracker
from tests.helpers.synthetic_batch import batch


@pytest.mark.parametrize('mode', ['uniform', 'inverse_lwh'])
def test_metre_huber_in_anchor_axes_and_endpoint_statistics(mode):
    data = batch(b=4)
    data['box_size'] = torch.tensor([[4., 2., 1.], [2., 1., 4.], [3., 3., 3.], [4., 2., 1.]])
    # 当前 GT 尺寸/航向故意与首框尺寸/锚点航向不同。
    data['target_box_size'] = torch.ones(4, 3)
    data['target_box'][:] = torch.tensor([0., 0., 0., -.6])
    data['anchor_box'][1, 3] = torch.pi / 2
    data['point_valid'][1, -1] = False  # 仅历史有点的端点仍有主框监督
    data['point_valid'][3] = False
    valid = data['point_valid'].any(dim=-1).any(dim=-1)
    q0 = torch.tensor([[.5, 2., -.25, .3], [-2., .5, -.25, .3],
                       [.5, 2., -.25, .3], [5., 4., 1., .3]], requires_grad=True)
    observation = SimpleNamespace(coarse_box=q0.detach() + .2, sequence_valid=valid,
        segmentation_logits=torch.zeros(4, 4, 16, 2), bc_prediction=torch.zeros_like(data['bc_targets']))
    decoder = SimpleNamespace(q0_box=q0, q0_quality_logits=torch.zeros(4),
                              history_boxes=data['history_boxes'] + .1)
    output = SimpleNamespace(observation=observation, prior=None, decoder=decoder)
    policy = dict(q0_center_weighting=mode)
    losses = compute_losses(data, output, enable_b1=False, q0_exact_iou=torch.zeros(4, 1), **policy)
    stats = observation_loss_statistics(data, output, **policy)

    # 前三个端点具有相同局部误差 (.5, 2, -.25)，同时覆盖 Huber 两侧。
    weights = torch.ones(3, 3)
    if mode == 'inverse_lwh':
        weights = torch.tensor([[3/7, 6/7, 12/7], [6/7, 12/7, 3/7], [1., 1., 1.]])
    huber = torch.tensor([.125, 1.5, .03125])
    expected = 2 * (weights * huber).mean(dim=-1)
    torch.testing.assert_close(losses['loss_main_center'], expected.mean())
    main_stats = stats['loss_main_center']
    torch.testing.assert_close(main_stats['numerator'], torch.cat((expected, torch.zeros(1))))
    torch.testing.assert_close(main_stats['denominator'], valid.float())
    for name in ('loss_main_center', 'loss_main_angle', 'loss_main',
                 'loss_coarse_center', 'loss_coarse', 'loss_history_center', 'loss_history'):
        torch.testing.assert_close(stats[name]['batch_contribution'].sum(), losses[name])

    gradient, = torch.autograd.grad(losses['loss_main'], q0)
    local_grad = weights * torch.tensor([.5, 1., -.25]) * (2/9)
    expected_grad = local_grad.clone()
    expected_grad[1, 0], expected_grad[1, 1] = -local_grad[1, 1], local_grad[1, 0]
    torch.testing.assert_close(gradient[:3, :3], expected_grad)
    assert (gradient[3] == 0).all()

    data['point_valid'].zero_()
    observation.sequence_valid = torch.zeros_like(valid)
    empty = compute_losses(data, output, enable_b1=False, q0_exact_iou=torch.zeros(4, 1), **policy)
    empty_grad, = torch.autograd.grad(empty['loss_main'], q0)
    assert empty['loss_main_center'].item() == 0 and (empty_grad == 0).all()


def test_full_tracker_changes_only_the_direct_q0_center_objective():
    data = batch()
    data['recovery_seed_box'] = data['fallback_box'].clone()
    data['recovery_seed_valid'] = torch.ones(2, dtype=torch.bool)
    data.update(future_valid=torch.ones(2, 2, dtype=torch.bool),
                future_dt=torch.tensor([[1., 2.], [1., 2.]]), future_displacement=torch.ones(2, 2, 2))
    cfg = normalize_config(dict(ct_engineering_check=True, v41_attention_backend='math',
                                point_sample_size=16, batch_size=2, workers=0))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2741)
        model = JointTracker(cfg).train()
        output = model(data)
        a = model.compute_losses(data, output)
        aux = model.forward_aux(data)
        aux_a = model.compute_aux_losses(data, aux)
        model.config.v41_q0_center_weighting = 'inverse_lwh'
        b = model.compute_losses(data, output)
        aux_b = model.compute_aux_losses(data, aux)

    # 此 batch 的航向锚点为 0，原 v3 直接平均米制 XYZ Smooth L1。
    q0 = output.decoder.q0_box
    legacy = 2 * F.smooth_l1_loss(q0[:, :3], data['target_box'][:, :3], reduction='none').mean(-1).mean()
    torch.testing.assert_close(a['loss_main_center'], legacy, atol=0, rtol=0)
    legacy_grad, = torch.autograd.grad(legacy, q0, retain_graph=True)
    a_grad, = torch.autograd.grad(a['loss_main'], q0, retain_graph=True)
    torch.testing.assert_close(a_grad[:, :3], legacy_grad[:, :3], atol=0, rtol=0)
    delta = b['loss_main_center'] - a['loss_main_center']
    assert delta.abs().item() > 1e-6
    for name in a:
        if name in ('loss_main_center', 'loss_main', 'loss_total'):
            torch.testing.assert_close(b[name] - a[name], delta, atol=2e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(b[name], a[name], atol=0, rtol=0, equal_nan=True)
    for name in aux_a:
        torch.testing.assert_close(aux_b[name], aux_a[name], atol=0, rtol=0, equal_nan=True)
    # 共享解码器仍接收 q0 梯度；B 相对 A 的总梯度差等于主框项梯度差。
    parameter = list(model.decoder.pose_head.parameters())[-1]
    total_delta, = torch.autograd.grad(b['loss_total'] - a['loss_total'], parameter, retain_graph=True)
    main_delta, = torch.autograd.grad(b['loss_main'] - a['loss_main'], parameter)
    assert main_delta.abs().sum() > 0
    torch.testing.assert_close(total_delta, main_delta, atol=2e-5, rtol=1e-5)
