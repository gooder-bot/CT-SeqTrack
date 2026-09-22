"""v31 测量、梯度所有权及四假设解码的 CPU 契约。"""
from dataclasses import replace
import math

import pytest
import torch
from torch import nn

from models.ct_v31.contracts import EvidenceHypotheses, ObservationFeatures, PriorContext
from models.ct_v31.decoder import SharedHypothesisDecoder
from models.ct_v31.observation import B0Observation, box_corners_xyz, unique_valid_mask


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    yield
    torch.set_num_threads(previous)


def _batch(batch_size=2, count=8):
    return dict(
        points=torch.randn(batch_size, 4, count, 5),
        point_valid=torch.ones(batch_size, 4, count, dtype=torch.bool),
        point_ids=torch.arange(count).expand(batch_size, 4, count).clone(),
        history_boxes=torch.tensor([[-1., .1, .2, .3], [-.5, .2, .3, .4],
                                    [0., 0., 0., .5]]).expand(batch_size, -1, -1).clone(),
        history_valid=torch.ones(batch_size, 3, dtype=torch.bool),
        history_times=torch.tensor([-1.5, -1., -.5]).expand(batch_size, -1).clone(),
        anchor_box=torch.zeros(batch_size, 4),
        box_size=torch.tensor([4., 2., 1.5]).expand(batch_size, -1).clone())


def _decoder_inputs(batch_size=1):
    batch = _batch(batch_size)
    observation = ObservationFeatures(
        coarse_box=torch.tensor([.8, -.2, .1, .7]).expand(batch_size, -1).clone(),
        point_features=torch.randn(batch_size, 4, 8, 64),
        source_tokens=torch.randn(batch_size, 4, 2, 128),
        source_valid=torch.ones(batch_size, 4, 2, dtype=torch.bool),
        segmentation_logits=torch.randn(batch_size, 4, 8, 2),
        bc_prediction=torch.randn(batch_size, 4, 8, 9),
        foreground_probability=torch.rand(batch_size, 4, 8),
        quality=torch.rand(batch_size, 4),
        current_valid=torch.ones(batch_size, dtype=torch.bool),
        sequence_valid=torch.ones(batch_size, dtype=torch.bool))
    z = torch.zeros(batch_size, 2)
    prior = PriorContext(feature=torch.randn(batch_size, 128), mean_xy=z.clone(),
        log_sigma=z.clone(), valid=torch.ones(batch_size, dtype=torch.bool),
        acquisition_fraction=z.clone(), direction_xy=torch.tensor([1., 0.]).expand(batch_size, -1),
        kinematic_xy=z.clone(), envelope=z.clone(), unit_residual=z.clone(),
        box=torch.tensor([.3, .2, .1, .5]).expand(batch_size, -1).clone())
    evidence = EvidenceHypotheses.empty(observation.coarse_box)
    evidence.centers_xyz = torch.tensor([[1., .1, .2], [2., .2, .3], [3., .3, .4]]).expand(batch_size, -1, -1).clone()
    evidence.mode_valid[:] = True
    evidence.point_features = torch.randn(batch_size, 6, 64)
    evidence.point_xyz = torch.randn(batch_size, 6, 3)
    evidence.point_valid = torch.ones(batch_size, 6, dtype=torch.bool)
    evidence.members = torch.zeros(batch_size, 3, 6, dtype=torch.bool)
    for mode in range(3):
        evidence.members[:, mode, 2 * mode:2 * mode + 2] = True
    evidence.memory_features = torch.randn(batch_size, 3, 64)
    evidence.memory_valid = torch.ones(batch_size, 3, dtype=torch.bool)
    return observation, evidence, batch, prior


def _assert_finite_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)


def test_observation_soft_foreground_protects_logits_but_keeps_shared_learning():
    batch = _batch()
    model = B0Observation(token_count=4).eval()
    captured = []
    hook = model.mini_pointnet.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    output = model(batch)
    hook.remove()
    output.segmentation_logits.retain_grad()
    output.coarse_box.square().sum().backward()
    assert output.coarse_box.shape == (2, 4)
    assert output.source_tokens.shape == (2, 4, 4, 128)
    assert output.point_features.shape == (2, 4, 8, 64)
    assert output.bc_prediction.shape == (2, 4, 8, 9)
    assert torch.equal(captured[0][:, :3].transpose(1, 2), batch['points'][..., :3].reshape(2, 32, 3))
    assert output.segmentation_logits.grad is None
    assert torch.count_nonzero(model.seg_pointnet.fc.weight.grad[:2]) == 0
    assert model.seg_pointnet.fc.weight.grad[2:].abs().sum() > 0
    assert model.seg_pointnet.seq_per_point[0][0].weight.grad.abs().sum() > 0
    assert not output.quality.requires_grad
    _assert_finite_gradients(model)


def test_history_bc_is_causal_and_current_bc_is_zero():
    batch = _batch(1)
    model = B0Observation(token_count=4).eval()
    captured = []
    hook = model.seg_pointnet.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    model(batch)
    hook.remove()
    point_input = captured[0].transpose(1, 2).reshape(1, 4, 8, 14)
    assert torch.count_nonzero(point_input[:, -1, :, 5:]) == 0
    expected_center_distance = (batch['points'][:, :3, :, :3] - batch['history_boxes'][:, :, None, :3]).norm(dim=-1)
    torch.testing.assert_close(point_input[:, :3, :, 5], expected_center_distance)


@pytest.mark.parametrize('angle', [0., .7, math.pi - 1e-4, -math.pi + 1e-4])
def test_coarse_yaw_uses_both_sine_and_cosine(angle):
    model = B0Observation(token_count=4).eval()
    with torch.no_grad():
        model.coarse_box_head[-1].weight.zero_()
        model.coarse_box_head[-1].bias[3:].copy_(torch.tensor([math.sin(angle), math.cos(angle)]))
    prediction = model(_batch(1)).coarse_box[0, 3]
    torch.testing.assert_close(prediction, torch.tensor(angle), atol=1e-6, rtol=0)
    (1 - torch.cos(prediction - (angle + .2))).backward()
    _assert_finite_gradients(model)


def test_raw_id_dedup_is_per_frame_and_invalid_poison_is_inert():
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    ids = torch.tensor([[[9, 9, 3, -1], [9, 3, 3, 8]]])
    assert unique_valid_mask(valid, ids).tolist() == [[[True, False, True, False], [True, True, False, True]]]
    batch = _batch(1)
    batch['point_ids'][..., 1] = batch['point_ids'][..., 0]
    model = B0Observation(token_count=4).eval()
    expected = model(batch)
    batch['points'][..., 1, :] = float('nan')
    actual = model(batch)
    for name in ('coarse_box', 'source_tokens', 'point_features', 'quality'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
    assert torch.count_nonzero(actual.foreground_probability[..., 1]) == 0


@pytest.mark.parametrize('unique_points', [0, 1])
def test_empty_and_singleton_observation_use_running_bn_and_finite_backward(unique_points):
    batch = _batch(1)
    batch['history_valid'][:] = False
    batch['history_boxes'][:] = float('nan')
    batch['point_valid'][:] = False
    batch['points'][:] = float('nan')
    if unique_points:
        batch['point_valid'][0, -1, 0] = True
        batch['points'][0, -1, 0] = torch.tensor([1., 2., 3., 0., .5])
    model = B0Observation(token_count=4).train()
    output = model(batch)
    assert all(layer.num_batches_tracked == 0 for layer in model.modules() if isinstance(layer, nn.BatchNorm1d))
    assert output.current_valid.item() == bool(unique_points)
    tensors = (output.coarse_box, output.source_tokens, output.bc_prediction, output.segmentation_logits)
    assert all(torch.isfinite(t).all() for t in tensors)
    sum(t.sum() for t in tensors).backward()
    _assert_finite_gradients(model)
    if not unique_points:
        assert all(torch.count_nonzero(t) == 0 for t in tensors)


def test_world_axis_corner_geometry_keeps_center_and_rotates_only_offsets():
    box = torch.tensor([[10., 20., 30., math.pi / 2]])
    corners = box_corners_xyz(box, torch.tensor([[4., 2., 6.]]))
    torch.testing.assert_close(corners.mean(-2), box[:, :3])
    torch.testing.assert_close(corners.amin(-2), torch.tensor([[9., 18., 27.]]))
    torch.testing.assert_close(corners.amax(-2), torch.tensor([[11., 22., 33.]]))


def test_seven_query_boxes_share_one_pose_head_and_preserve_initial_seeds():
    observation, evidence, batch, prior = _decoder_inputs()
    model = SharedHypothesisDecoder().eval()
    counts = []
    hook = model.corner_projection.register_forward_pre_hook(lambda module, args: counts.append(args[0].shape[1]))
    output = model(observation, evidence, batch, prior)
    hook.remove()
    assert counts == [56]
    assert output.hypothesis_boxes.shape == (1, 4, 4)
    assert output.history_boxes.shape == (1, 3, 4)
    assert output.decoder_features.shape == (1, 4, 64)
    torch.testing.assert_close(output.history_boxes, batch['history_boxes'])
    torch.testing.assert_close(output.hypothesis_boxes[:, 0], observation.coarse_box)
    torch.testing.assert_close(output.hypothesis_boxes[:, 1:, :3], evidence.centers_xyz)
    torch.testing.assert_close(output.hypothesis_boxes[:, 1:, 3], observation.coarse_box[:, 3:4].expand(-1, 3))
    assert output.hypothesis_valid.all()
    assert len(model.encoder.layer_stack) == len(model.encoder_global.layer_stack) == len(model.decoder.layer_stack) == 3


def test_decoder_geometry_and_context_have_the_intended_gradient_ownership():
    observation, evidence, batch, prior = _decoder_inputs()
    observation.coarse_box.requires_grad_()
    evidence.centers_xyz.requires_grad_()
    evidence.point_features.requires_grad_()
    prior.mean_xy.requires_grad_()
    prior.log_sigma.requires_grad_()
    prior.feature.requires_grad_()
    prior.box.requires_grad_()
    model = SharedHypothesisDecoder().eval()
    # 零初始化 adapter 先学习输出层，随后应允许 h1 编码器联合学习。
    nn.init.normal_(model.prior_adapter[-1].weight, std=.03)
    output = model(observation, evidence, batch, prior)
    variables = (observation.coarse_box, evidence.centers_xyz, prior.mean_xy,
                 prior.log_sigma, prior.feature, evidence.point_features, prior.box)
    gradients = torch.autograd.grad(output.decoder_features.square().sum(), variables,
                                    allow_unused=True, retain_graph=True)
    assert gradients[0] is not None and gradients[0].abs().sum() > 0
    assert gradients[1] is None  # fixed mode query geometry has no center path
    assert gradients[2] is None and gradients[3] is None and gradients[6] is None
    assert gradients[4] is not None and gradients[4].abs().sum() > 0
    assert gradients[5] is not None and gradients[5].abs().sum() > 0
    center_grad, = torch.autograd.grad(output.hypothesis_boxes[:, 1:, :3].sum(), evidence.centers_xyz)
    torch.testing.assert_close(center_grad, torch.ones_like(center_grad))


def test_fixed_member_attention_prevents_other_modes_coordinate_and_feature_bypass():
    observation, evidence, batch, prior = _decoder_inputs()
    model = SharedHypothesisDecoder().eval()
    before = model(observation, evidence, batch, prior)
    changed_features = evidence.point_features.clone()
    changed_features[:, 2:] += 30 * torch.randn_like(changed_features[:, 2:])
    changed_centers = evidence.centers_xyz.clone()
    changed_centers[:, 1:] += 50
    after = model(observation, replace(evidence, point_features=changed_features, centers_xyz=changed_centers), batch, prior)
    torch.testing.assert_close(before.decoder_features[:, 1], after.decoder_features[:, 1], atol=0, rtol=0)
    assert not torch.allclose(before.decoder_features[:, 0], after.decoder_features[:, 0])


def test_empty_base_can_recover_with_modes_but_empty_sequence_inputs_force_prior():
    observation, evidence, batch, prior = _decoder_inputs()
    observation.current_valid[:] = False
    observation.sequence_valid[:] = False
    observation.source_valid[:] = False
    model = SharedHypothesisDecoder().eval()
    with torch.no_grad():
        model.pose_head.bias[:3] = torch.tensor([.2, -.3, .4])
    recovered = model(observation, evidence, batch, prior)
    torch.testing.assert_close(recovered.hypothesis_boxes[:, 0], prior.box)
    torch.testing.assert_close(recovered.hypothesis_boxes[:, 1:, :3], evidence.centers_xyz + model.pose_head.bias[:3])
    torch.testing.assert_close(recovered.hypothesis_boxes[:, 1:, 3], prior.box[:, 3:4].expand(-1, 3))
    assert recovered.hypothesis_valid.all()
    evidence.point_valid[:] = False
    evidence.point_features[:] = float('nan')
    empty = model(observation, evidence, batch, prior)
    torch.testing.assert_close(empty.hypothesis_boxes, prior.box[:, None].expand(-1, 4, -1))
    assert empty.hypothesis_valid.tolist() == [[True, False, False, False]]


def test_all_masked_decoder_and_zero_sincos_residual_are_numerically_safe():
    observation, evidence, batch, prior = _decoder_inputs()
    batch['history_valid'][:] = False
    batch['history_boxes'][:] = float('nan')
    observation.current_valid[:] = False
    observation.sequence_valid[:] = False
    observation.source_valid[:] = False
    observation.source_tokens[:] = float('nan')
    evidence.point_valid[:] = False
    evidence.memory_valid[:] = False
    evidence.point_features[:] = float('nan')
    evidence.memory_features[:] = float('nan')
    model = SharedHypothesisDecoder().train()
    with torch.no_grad():
        model.pose_head.bias[3:] = 0
    output = model(observation, evidence, batch, prior)
    assert torch.isfinite(output.hypothesis_boxes).all()
    assert torch.isfinite(output.quality_logits).all()
    assert torch.isfinite(output.decoder_features).all()
    (output.hypothesis_boxes.sum() + output.quality_logits.sum() + output.history_boxes.sum()).backward()
    _assert_finite_gradients(model)


def test_invalid_modes_stay_invalid_even_when_other_evidence_is_present():
    observation, evidence, batch, prior = _decoder_inputs()
    evidence.mode_valid[:, 1] = False
    evidence.members[:, 2] = False
    evidence.centers_xyz[:, 1:] = float('nan')
    output = SharedHypothesisDecoder().eval()(observation, evidence, batch, prior)
    assert output.hypothesis_valid.tolist() == [[True, True, False, False]]
    assert torch.count_nonzero(output.decoder_features[:, 2:]) == 0
    assert torch.count_nonzero(output.hypothesis_boxes[:, 2:]) == 0
    assert (output.quality_logits[:, 2:] == -20).all()
