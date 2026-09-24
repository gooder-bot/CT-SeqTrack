"""v32 局部观测、真实点 token 和逐端点监督的 CPU 回归检查。"""
import math

import pytest
import torch
from torch.nn import functional as F

from models.ct_v31.contracts import EvidenceHypotheses, TrackOutput
from models.ct_v31.decoder import SharedHypothesisDecoder
from models.ct_v31.geometry import anchor_yaw, rotate_xyz, transform_boxes
from models.ct_v31.losses import (compute_losses, local_observation_box_loss,
                                 observation_bc_loss, observation_segmentation_loss)
from models.ct_v31.observation import B0Observation, _pooled_tokens
from models.ct_v31.prior import empty_prior


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(32)
    yield
    torch.set_num_threads(previous)


def _batch(batch_size=2, count=16, yaw=.7):
    points = torch.randn(batch_size, 4, count, 5) * .2
    points[..., 3] = torch.tensor([-1.5, -1., -.5, 0.])[None, :, None]
    points[..., 4] = .5
    history = torch.tensor([[-.6, 0., 0., yaw - .1], [-.3, 0., 0., yaw],
                            [0., 0., 0., yaw]]).repeat(batch_size, 1, 1)
    size = torch.tensor([4., 2., 1.5]).repeat(batch_size, 1)
    target = torch.tensor([.4, .1, .1, yaw + .1]).repeat(batch_size, 1)
    return dict(points=points,
        point_valid=torch.ones(batch_size, 4, count, dtype=torch.bool),
        point_ids=torch.arange(count).expand(batch_size, 4, count).clone(),
        history_boxes=history, history_valid=torch.ones(batch_size, 3, dtype=torch.bool),
        history_times=torch.tensor([-1.5, -1., -.5]).repeat(batch_size, 1),
        anchor_box=torch.tensor([7., -3., 1., yaw]).repeat(batch_size, 1),
        fallback_box=torch.tensor([0., 0., 0., yaw]).repeat(batch_size, 1),
        target_box=target, history_target_boxes=history.clone(),
        box_size=size, target_box_size=size.clone(),
        segmentation_labels=torch.ones(batch_size, 4, count, dtype=torch.long),
        bc_targets=torch.zeros(batch_size, 4, count, 9))


@pytest.mark.parametrize('measured', [0, 1, 2, 40, 128, 129, 1024])
def test_ragged_tokens_preserve_sparse_points_and_dense_seqtrack_bins(measured):
    value = torch.arange(1024., requires_grad=True).reshape(1, 1, 1024)
    valid = torch.arange(1024)[None] < measured
    pooled, present = _pooled_tokens(value, valid, 128)
    tokens = min(measured, 128)
    assert present.sum().item() == tokens
    if tokens:
        boundaries = torch.div(torch.arange(tokens + 1) * measured, tokens, rounding_mode='floor')
        expected = torch.stack([value[0, 0, boundaries[i]:boundaries[i + 1]].max()
                                for i in range(tokens)])
        torch.testing.assert_close(pooled[0, 0, :tokens], expected, rtol=0, atol=0)
    assert torch.count_nonzero(pooled[..., tokens:]) == 0
    if measured == 1024:
        torch.testing.assert_close(pooled, F.adaptive_max_pool1d(value, 128), rtol=0, atol=0)
    pooled.sum().backward()


def test_ragged_bins_do_not_overlap_and_padding_is_inert():
    value = torch.ones(1, 2, 17, requires_grad=True)
    pooled, valid = _pooled_tokens(value, torch.ones(1, 17, dtype=torch.bool), 4)
    pooled.sum().backward()
    assert valid.all()
    expected = torch.zeros_like(value)
    expected[..., [0, 4, 8, 12]] = 1.
    torch.testing.assert_close(value.grad, expected, rtol=0, atol=0)
    scattered = torch.full((1, 2, 35), float('nan'))
    scattered[..., ::2] = 0.
    scattered[..., 1::2] = value.detach()
    scattered_valid = torch.zeros(1, 35, dtype=torch.bool)
    scattered_valid[..., 1::2] = True
    compact_result, compact_valid = _pooled_tokens(scattered, scattered_valid, 4)
    torch.testing.assert_close(compact_result, pooled, rtol=0, atol=0)
    assert torch.equal(compact_valid, valid)


def test_geometry_round_trip_preserves_world_relative_contract_and_gradients():
    yaw = torch.tensor([math.pi / 2, -math.pi + 1e-4])
    boxes = torch.tensor([[[1., 2., 3., -math.pi + .1]], [[-2., 4., -1., math.pi - .1]]],
                         requires_grad=True)
    local = transform_boxes(boxes, yaw)
    restored = transform_boxes(local, yaw, to_world=True)
    torch.testing.assert_close(restored[..., :3], boxes[..., :3], atol=1e-6, rtol=0)
    torch.testing.assert_close((restored[..., 3] - boxes[..., 3]).sin(), torch.zeros(2, 1), atol=1e-6, rtol=0)
    restored[..., :3].sum().backward()
    torch.testing.assert_close(boxes.grad[..., :3], torch.ones(2, 1, 3))


def test_b0_local_features_and_world_box_are_consistent_under_world_rotation():
    batch = _batch(1)
    model = B0Observation(token_count=8).eval()
    before = model(batch)
    delta = torch.tensor([1.2])
    changed = dict(batch)
    changed['points'] = torch.cat((rotate_xyz(batch['points'][..., :3], delta, to_world=True),
                                   batch['points'][..., 3:]), -1)
    changed['history_boxes'] = transform_boxes(batch['history_boxes'], delta, to_world=True)
    changed['anchor_box'] = transform_boxes(batch['anchor_box'], delta, to_world=True)
    after = model(changed)
    for field in ('point_features', 'source_tokens', 'segmentation_logits', 'bc_prediction', 'quality'):
        torch.testing.assert_close(getattr(after, field), getattr(before, field), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(after.coarse_box,
                               transform_boxes(before.coarse_box, delta, to_world=True), atol=2e-6, rtol=2e-5)


def test_decoder_rotates_geometry_and_residuals_without_touching_b2_features():
    batch = _batch(1, yaw=math.pi / 2)
    observation = B0Observation(token_count=8).eval()(batch)
    evidence = EvidenceHypotheses.empty(observation.coarse_box)
    evidence.centers_xyz[0, 0] = torch.tensor([2., 3., .1])
    evidence.centers_xyz.requires_grad_()
    evidence.point_valid[:, 0] = True
    evidence.members[:, 0, 0] = True
    evidence.mode_valid[:, 0] = True
    evidence.point_features[:, 0] = torch.arange(64.) / 64.
    evidence.point_features.requires_grad_()
    before_center = evidence.centers_xyz.detach().clone()
    before_covariance = evidence.covariance_xy.clone()
    prior = empty_prior(batch)
    decoder = SharedHypothesisDecoder().eval()
    with torch.no_grad():
        decoder.pose_head.bias[:3] = torch.tensor([1., 0., .2])
    captured = []
    hook = decoder.extension_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    result = decoder(observation, evidence, batch, prior)
    hook.remove()
    torch.testing.assert_close(captured[0], evidence.point_features)
    expected_delta = torch.tensor([[0., 1., .2]])
    torch.testing.assert_close(result.hypothesis_boxes[:, 0, :3], expected_delta,
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(result.hypothesis_boxes[:, 1, :3], evidence.centers_xyz[:, 0] + expected_delta,
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(evidence.centers_xyz.detach(), before_center, rtol=0, atol=0)
    torch.testing.assert_close(evidence.covariance_xy, before_covariance, rtol=0, atol=0)
    feature_grad = torch.autograd.grad(result.decoder_features.sum(), evidence.centers_xyz,
                                       allow_unused=True, retain_graph=True)[0]
    assert feature_grad is None
    center_grad = torch.autograd.grad(result.hypothesis_boxes[:, 1, :3].sum(), evidence.centers_xyz)[0]
    torch.testing.assert_close(center_grad[:, 0], torch.ones(1, 3))


def test_history_only_predicts_and_learns_without_claiming_current_measurements():
    batch = _batch(1)
    batch['point_valid'][:, -1] = False
    batch['segmentation_labels'][:, -1] = -1
    observation_model = B0Observation(token_count=8).eval()
    observation = observation_model(batch)
    assert not observation.current_valid.any()
    assert observation.sequence_valid.all()
    assert not observation.source_valid[:, -1].any()
    assert not observation.foreground_probability[:, -1].any()
    decoder = SharedHypothesisDecoder().eval()
    with torch.no_grad():
        decoder.pose_head.bias[:3] = torch.tensor([.1, -.2, .3])
    prior = empty_prior(batch)
    evidence = EvidenceHypotheses.empty(observation.coarse_box)
    decoded = decoder(observation, evidence, batch, prior)
    expected_delta = rotate_xyz(decoder.pose_head.bias[None, :3], anchor_yaw(batch, observation.coarse_box), to_world=True)
    torch.testing.assert_close(decoded.hypothesis_boxes[:, 0, :3], expected_delta)
    output = TrackOutput(decoded.hypothesis_boxes[:, 0], torch.zeros(1, dtype=torch.long),
                         decoded.quality_logits[:, 0].sigmoid(), prior, observation, evidence, decoded)
    losses = compute_losses(batch, output, enable_b1=False, enable_b2=False, enable_b3=False)
    assert losses['loss_coarse'] > 0 and losses['loss_main'] > 0
    (losses['loss_coarse'] + losses['loss_main']).backward()
    assert observation_model.coarse_box_head[-1].weight.grad.abs().sum() > 0


def test_segmentation_weights_frames_and_endpoints_not_dense_point_counts():
    logits = torch.zeros(2, 4, 12, 2, requires_grad=True)
    with torch.no_grad():
        logits[0, :, :, 1] = torch.tensor([.1, .6, 1.2, -.3])[:, None]
        logits[1, :, :, 1] = torch.tensor([.2, -.6, .3, .7])[:, None]
    labels = torch.ones(2, 4, 12, dtype=torch.long)
    valid = torch.arange(12)[None, None] < torch.tensor([[1, 4, 12, 2], [12, 1, 3, 11]])[..., None]
    actual, current, history = observation_segmentation_loss(logits, labels, valid)
    per_frame = F.softplus(-logits[..., 0, 1])
    expected = .5 * (per_frame[:, -1] + per_frame[:, :3].mean(-1)).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(current, per_frame[:, -1].mean())
    torch.testing.assert_close(history, per_frame[:, :3].mean())
    actual.backward()
    assert logits.grad.abs().sum() > 0
    # 当前帧缺失时仅存在的历史任务仍有完整端点权重。
    valid[:, -1] = False
    actual, _, _ = observation_segmentation_loss(logits.detach(), labels, valid)
    torch.testing.assert_close(actual, per_frame[:, :3].mean())


def test_bc_balances_current_history_groups_and_unique_point_counts():
    target = torch.zeros(2, 4, 12, 9)
    prediction = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])[..., None, None].expand_as(target).clone()
    prediction.requires_grad_()
    valid = torch.arange(12)[None, None] < torch.tensor([[1, 4, 12, 2], [12, 1, 3, 11]])[..., None]
    value, current, history = observation_bc_loss(prediction, target, valid)
    per_frame = F.smooth_l1_loss(prediction[..., 0, 0], torch.zeros(2, 4), reduction='none')
    expected = .5 * (per_frame[:, -1] + per_frame[:, :3].mean(-1)).mean()
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(current, per_frame[:, -1].mean())
    torch.testing.assert_close(history, per_frame[:, :3].mean())
    value.backward()
    assert torch.count_nonzero(prediction.grad[valid]) > 0
    assert torch.count_nonzero(prediction.grad[~valid]) == 0


def test_local_b0_loss_is_invariant_to_world_orientation_but_world_b2_objective_is_preserved():
    predicted = torch.tensor([[2., .2, .1, .4]])
    target = torch.tensor([[.1, -.2, .0, .3]])
    valid = torch.ones(1, dtype=torch.bool)
    yaw = torch.tensor([.2])
    delta = torch.tensor([.9])
    original = local_observation_box_loss(predicted, target, valid, yaw)
    changed = local_observation_box_loss(transform_boxes(predicted, delta, to_world=True),
        transform_boxes(target, delta, to_world=True), valid, yaw + delta)
    torch.testing.assert_close(original, changed)
    # 原 B2 的世界轴/LWH 逐轴目标有方向依赖，本轮不把它悄悄换成 local。
    size = torch.tensor([[4., 2., 1.5]])
    error = predicted[:, :3] - target[:, :3]
    old_vote = F.smooth_l1_loss(error / size, torch.zeros_like(error))
    rotated_vote = F.smooth_l1_loss(rotate_xyz(error, delta, to_world=True) / size, torch.zeros_like(error))
    assert not torch.allclose(old_vote, rotated_vote)


def test_empty_supervision_is_finite_and_keeps_a_zero_gradient_path():
    logits = torch.randn(1, 4, 3, 2, requires_grad=True)
    valid = torch.zeros(1, 4, 3, dtype=torch.bool)
    labels = torch.full((1, 4, 3), -1)
    loss, current, history = observation_segmentation_loss(logits, labels, valid)
    assert loss == current == history == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad) == 0


def test_compute_losses_keeps_b2_vote_and_b3_mode_targets_in_world_axes():
    batch = _batch(1)
    batch['extension_valid'] = torch.arange(768)[None] < 2
    batch['extension_labels'] = torch.zeros(1, 768)
    batch['extension_labels'][:, 0] = 1.
    observation = B0Observation(token_count=8).eval()(batch)
    prior = empty_prior(batch)
    evidence = EvidenceHypotheses.empty(observation.coarse_box)
    evidence.point_valid[:, :2] = True
    evidence.point_indices[:, :2] = torch.tensor([0, 1])
    evidence.point_ids[:, :2] = torch.tensor([0, 1])
    evidence.vote_xyz[:, 0] = torch.tensor([2.4, .3, .1])
    evidence.centers_xyz[:, 0] = torch.tensor([2.4, .3, .1])
    evidence.mode_valid[:, 0] = True
    evidence.members[:, 0, 0] = True
    decoded = SharedHypothesisDecoder().eval()(observation, evidence, batch, prior)
    output = TrackOutput(decoded.hypothesis_boxes[:, 0], torch.zeros(1, dtype=torch.long),
                         decoded.quality_logits[:, 0].sigmoid(), prior, observation, evidence, decoded)
    first = compute_losses(batch, output, enable_b1=False, enable_b2=True, enable_b3=True)
    changed = dict(batch)
    changed['anchor_box'] = batch['anchor_box'].clone()
    changed['anchor_box'][:, 3] += .8
    second = compute_losses(changed, output, enable_b1=False, enable_b2=True, enable_b3=True)
    for field in ('loss_identity', 'loss_vote', 'loss_reliability', 'loss_modes', 'loss_quality'):
        torch.testing.assert_close(first[field], second[field], rtol=0, atol=0)
    error = (evidence.vote_xyz[:, 0] - batch['target_box'][:, :3]) / batch['box_size']
    torch.testing.assert_close(first['loss_vote'], F.smooth_l1_loss(error, torch.zeros_like(error)))
