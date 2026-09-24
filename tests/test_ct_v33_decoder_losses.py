"""v33 输出参考中心、无 FG 的 BC 与端点权重的 CPU 行为检查。"""
from dataclasses import replace
import math

import pytest
import torch

from models.ct_v31.decoder import SharedHypothesisDecoder
from models.ct_v31.geometry import rotate_xyz
from models.ct_v31.losses import local_observation_box_loss, observation_bc_loss
from models.ct_v31.model import JointTracker
from tests.test_ct_v31_joint import make_batch, model_config
from tests.test_ct_v31_observation_decoder import _decoder_inputs


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(33)
    yield
    torch.set_num_threads(previous)


def test_same_corner_and_evidence_queries_can_distinguish_absolute_and_residual_targets():
    observation, evidence, batch, prior = _decoder_inputs()
    # 一整个证据簇恰好与 coarse 同中心：q0/mode 的角点、时间和可读 KV 相同。
    evidence.centers_xyz[:, 0] = observation.coarse_box[:, :3]
    evidence.members[:, 0] = evidence.point_valid
    decoder = SharedHypothesisDecoder().eval()
    inputs = []
    hook = decoder.corner_projection.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone()))
    output = decoder(observation, evidence, batch, prior)
    hook.remove()
    queries = inputs[0].reshape(1, 7, 8, -1)
    torch.testing.assert_close(queries[:, 3, :, :4], queries[:, 4, :, :4])
    # 不同输出参考中心必须是网络可见输入，不能只藏在最终加法中。
    assert not torch.allclose(queries[:, 3], queries[:, 4])
    assert not torch.allclose(output.decoder_features[:, 0], output.decoder_features[:, 1])
    torch.testing.assert_close(output.hypothesis_boxes[:, 0, :3], torch.zeros(1, 3))
    torch.testing.assert_close(output.hypothesis_boxes[:, 1, :3], evidence.centers_xyz[:, 0])


def test_direct_q0_and_seed_residuals_restore_world_axes_and_keep_yaw_residuals():
    observation, evidence, batch, prior = _decoder_inputs()
    batch['anchor_box'][:, 3] = math.pi / 2
    decoder = SharedHypothesisDecoder().eval()
    delta_yaw = .25
    with torch.no_grad():
        decoder.pose_head.bias.copy_(torch.tensor([1., 0., .2, math.sin(delta_yaw), math.cos(delta_yaw)]))
    output = decoder(observation, evidence, batch, prior)
    torch.testing.assert_close(output.hypothesis_boxes[:, 0, :3], torch.tensor([[0., 1., .2]]), atol=1e-6, rtol=0)
    delta_world = rotate_xyz(torch.tensor([[1., 0., .2]]), batch['anchor_box'][:, 3], to_world=True)
    torch.testing.assert_close(output.history_boxes[..., :3], batch['history_boxes'][..., :3] + delta_world[:, None])
    torch.testing.assert_close(output.hypothesis_boxes[:, 1:, :3], evidence.centers_xyz + delta_world[:, None])
    torch.testing.assert_close(output.history_boxes[..., 3], batch['history_boxes'][..., 3] + delta_yaw)
    torch.testing.assert_close(output.hypothesis_boxes[..., 3],
                               observation.coarse_box[:, 3:4].expand(-1, 4) + delta_yaw)


def test_center_gradient_has_no_coarse_addition_but_keeps_query_and_live_mode_paths():
    observation, evidence, batch, prior = _decoder_inputs()
    observation.coarse_box.requires_grad_()
    evidence.centers_xyz.requires_grad_()
    batch['history_boxes'].requires_grad_()
    decoder = SharedHypothesisDecoder().eval()
    initial = decoder(observation, evidence, batch, prior)
    grad = torch.autograd.grad(initial.hypothesis_boxes[:, 0, :3].sum(), observation.coarse_box,
                               retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad))
    # 学到非零 head 后，main 中心仍可沿 coarse corner query 训练粗定位。
    with torch.no_grad():
        torch.nn.init.normal_(decoder.pose_head.weight[:3], std=.03)
    output = decoder(observation, evidence, batch, prior)
    coarse_grad, history_grad = torch.autograd.grad(output.hypothesis_boxes[:, 0, :3].sum(),
        (observation.coarse_box, batch['history_boxes']), retain_graph=True, allow_unused=True)
    assert coarse_grad[:, :3].abs().sum() > 0
    assert history_grad is None
    mode_query_grad = torch.autograd.grad(output.decoder_features.square().sum(), evidence.centers_xyz,
                                         retain_graph=True, allow_unused=True)[0]
    assert mode_query_grad is None
    mode_grad = torch.autograd.grad(output.hypothesis_boxes[:, 1:, :3].sum(), evidence.centers_xyz)[0]
    torch.testing.assert_close(mode_grad, torch.ones_like(mode_grad))


def test_bc_group_balance_handles_current_only_history_only_and_empty_endpoints():
    counts = torch.tensor([[1, 3, 2, 4], [0, 0, 0, 1], [0, 0, 2, 0], [0, 0, 0, 0]])
    valid = torch.arange(4)[None, None] < counts[..., None]
    errors = torch.tensor([[1., 2., 3., 4.], [0., 0., 0., 2.], [0., 0., 3., 0.], [0., 0., 0., 0.]])
    prediction = errors[..., None, None].expand(4, 4, 4, 9).clone()
    prediction[~valid] = float('nan')
    prediction.requires_grad_()
    target = torch.zeros_like(prediction).masked_fill(~valid[..., None], float('nan'))
    loss, current, history = observation_bc_loss(prediction, target, valid)
    # 端点损失依次 2.5、1.5、2.5；第四个完全缺测端点不进分母。
    torch.testing.assert_close(loss, torch.tensor(6.5 / 3))
    torch.testing.assert_close(current, torch.tensor(2.5))
    torch.testing.assert_close(history, torch.tensor(2.))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[valid] > 0).all()
    assert torch.count_nonzero(prediction.grad[~valid]) == 0


def test_all_empty_bc_has_finite_zero_loss_and_gradients():
    prediction = torch.full((2, 4, 3, 9), float('nan'), requires_grad=True)
    valid = torch.zeros(2, 4, 3, dtype=torch.bool)
    losses = observation_bc_loss(prediction, prediction.detach(), valid)
    assert all(value == 0 and torch.isfinite(value) for value in losses)
    losses[0].backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_history_localization_weights_endpoints_independently_of_history_length():
    predicted = torch.zeros(2, 3, 4)
    predicted[0, :, 0], predicted[1, :, 0] = 2., 4.
    predicted[0, :, 3], predicted[1, :, 3] = .3, .6
    predicted.requires_grad_()
    valid = torch.tensor([[False, False, True], [True, True, True]])
    target = torch.zeros_like(predicted)
    value = local_observation_box_loss(predicted, target, valid, torch.zeros(2), .2, 1., endpoint_mean=True)
    expected = .2 * ((1.5 + 3.5) / 3) / 2 + ((1 - math.cos(.3)) + (1 - math.cos(.6))) / 2
    torch.testing.assert_close(value, torch.tensor(expected))
    # 复制同一端点已有历史监督，不应提升该端点的总权重。
    repeated = local_observation_box_loss(predicted, target, torch.ones_like(valid),
                                          torch.zeros(2), .2, 1., endpoint_mean=True)
    torch.testing.assert_close(value, repeated)
    value.backward()
    torch.testing.assert_close(predicted.grad[0, :, 0].sum(), predicted.grad[1, :, 0].sum())
    assert torch.count_nonzero(predicted.grad[~valid]) == 0


def test_no_fg_bc_keeps_geometry_gradients_and_logged_components_do_not_add_objectives():
    batch = make_batch(1)
    batch['segmentation_labels'].zero_()
    model = JointTracker(model_config('b0')).eval()
    output = model(batch)
    output.observation.bc_prediction.retain_grad()
    losses = model.compute_losses(batch, output)
    assert losses['loss_bc'] > 0
    losses['loss_bc'].backward(retain_graph=True)
    assert output.observation.bc_prediction.grad[:, -1].abs().sum() > 0
    for name in ('coarse', 'main', 'history', 'modes'):
        center, angle = losses['loss_' + name + '_center'], losses['loss_' + name + '_angle']
        assert not center.requires_grad and not angle.requires_grad
        torch.testing.assert_close(losses['loss_' + name], center + angle)
    assert not losses['loss_bc_current'].requires_grad and not losses['loss_bc_history'].requires_grad
    for name, weight in (('seg', .1), ('quality', .5)):
        contribution = losses['loss_contribution_' + name]
        assert not contribution.requires_grad
        torch.testing.assert_close(contribution, weight * losses['loss_' + name])
    # B0 无额外模块任务，分项日志不能使定位/BC 被重复累加。
    expected = (losses['loss_coarse'] + losses['loss_main'] + losses['loss_history']
                + .1 * losses['loss_seg'] + losses['loss_bc'] + .5 * losses['loss_quality'])
    torch.testing.assert_close(losses['loss_total'], expected)


def test_no_observation_still_falls_back_to_prior_with_or_without_mode_evidence():
    observation, evidence, batch, prior = _decoder_inputs()
    observation = replace(observation, sequence_valid=torch.zeros(1, dtype=torch.bool),
        current_valid=torch.zeros(1, dtype=torch.bool), source_valid=torch.zeros_like(observation.source_valid))
    decoder = SharedHypothesisDecoder().eval()
    output = decoder(observation, evidence, batch, prior)
    torch.testing.assert_close(output.hypothesis_boxes[:, 0], prior.box)
    torch.testing.assert_close(output.hypothesis_boxes[:, 1:, :3], evidence.centers_xyz)
    empty = replace(evidence, point_valid=torch.zeros_like(evidence.point_valid))
    output = decoder(observation, empty, batch, prior)
    torch.testing.assert_close(output.hypothesis_boxes, prior.box[:, None].expand(-1, 4, -1))
    assert output.hypothesis_valid.tolist() == [[True, False, False, False]]
