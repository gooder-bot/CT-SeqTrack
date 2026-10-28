"""v39 G 的合约测试；CUDA 恢复审计仅在兼容设备可用时运行。"""
import math
import copy
import os

import numpy as np
import pytest
import torch

from models.ct_v39.g_observation import GObservation, unique_valid, ContiguousCpuGELU, _coarse_vote_max
from models.ct_v39.g_losses import compute_g_losses, correspondence_targets, supervision_targets, masked_mean, _balanced_values
from models.ct_v39.tensor_geometry import pair_box_metrics, rotate_xy
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions


def batch_fixture(batch=2, requires_grad=False):
    generator = torch.Generator().manual_seed(13)
    template = (torch.rand(batch, 8, 3, generator=generator) - .5) * torch.tensor([2., 1., .8])
    size = torch.tensor([4., 2., 1.5]).expand(batch, 3).clone()
    truth = torch.tensor([.2, -.1, .05, .1]).expand(batch, 4).clone()
    current = torch.cat((rotate_xy(template[..., :2], truth[:, None, 3]), template[..., 2:]), -1)
    current = current + truth[:, None, :3]
    background = torch.rand(batch, 4, 3, generator=generator) + 5.
    points = torch.cat((current, background), 1).requires_grad_(requires_grad)
    pool = dict(xyz=points, point_ids=torch.arange(12)[None].expand(batch, 12),
                valid=torch.ones(batch, 12, dtype=torch.bool))
    templates = dict(xyz=template[:, None].expand(batch, 3, 8, 3).clone().requires_grad_(requires_grad),
                     valid=torch.ones(batch, 3, 8, dtype=torch.bool), ages=torch.tensor([10., 1., .5])[None].expand(batch, 3))
    boxes = truth[:, None].expand(batch, 4, 4).clone()
    boxes[:, :, 0] += torch.tensor([0., .2, -.3, .1])
    boxes.requires_grad_(requires_grad)
    targets = dict(target_box=truth.requires_grad_(requires_grad), target_size=size,
                   current_target_mask=torch.arange(12)[None].expand(batch, 12) < 8,
                   template_xyz_gt=templates['xyz'], template_target_mask=templates['valid'])
    return pool, templates, boxes, size, targets


def test_tensor_geometry_matches_double_cpu_oracle():
    generator = torch.Generator().manual_seed(71)
    boxes = torch.randn(12, 4, 4, generator=generator)
    target = torch.randn(12, 4, generator=generator) * .4
    size = torch.rand(12, 3, generator=generator) * 3. + 1.
    target_size = torch.rand(12, 3, generator=generator) * 3. + 1.
    result = pair_box_metrics(boxes, target, size, target_size)
    for b in range(len(boxes)):
        for c in range(boxes.shape[1]):
            pred = LocalYawBox(boxes[b, c].double().numpy(), size[b, [1, 0, 2]].double().numpy())
            gt = LocalYawBox(target[b].double().numpy(), target_size[b, [1, 0, 2]].double().numpy())
            exact, distance = box_metrics(pred, gt, up_axis=(0, 0, 1), mode='geometry_exact')
            compat, _ = box_metrics(pred, gt, up_axis=(0, 0, 1), mode='benchmark_compat')
            success, precision = metric_contributions(compat, distance)
            assert abs(result['exact_iou'][b, c].item() - exact) < 2e-4
            assert abs(result['benchmark_iou'][b, c].item() - compat) < 2e-4
            assert abs(result['distance'][b, c].item() - distance) < 2e-4
            assert abs(result['success'][b, c].item() - success) < 1e-6
            assert abs(result['precision'][b, c].item() - precision) < 1e-6


def test_tensor_geometry_exact_and_compat_height_are_distinct():
    boxes = torch.zeros(1, 3, 4)
    boxes[0, 1, 0] = 10.
    boxes[0, 2, 3] = math.pi / 2.
    size, true_size = torch.tensor([[4., 2., 1.]]), torch.tensor([[4., 2., 3.]])
    result = pair_box_metrics(boxes, torch.tensor([[0., 0., 1.4, 0.]]), size, true_size)
    assert result['exact_iou'][0, 1] == 0
    assert not torch.equal(result['exact_iou'], result['benchmark_iou'])


def test_g_loss_is_finite_and_gradient_isolated_from_host_and_gt():
    torch.manual_seed(4)
    pool, templates, boxes, size, targets = batch_fixture(requires_grad=True)
    model = GObservation()
    measurement = torch.tensor([[True, False, False, True]]).expand(len(boxes), 4)
    verification = model.verify(pool, templates, boxes, size, measurement_eligible=measurement)
    refinement = model.refine(pool, templates, boxes, size,
                              encoded_pool=verification['encoded_pool'], encoded_templates=verification['encoded_templates'])
    losses = compute_g_losses(pool, templates, boxes, size, verification, targets, refinement)
    assert torch.isfinite(torch.stack(list(losses.values()))).all()
    assert verification['logits'].shape == (2, 4, 3)
    assert torch.equal(verification['trust'][:, 1:3], torch.zeros(2, 2))
    losses['loss_g'].backward()
    assert pool['xyz'].grad is None
    assert templates['xyz'].grad is None
    assert boxes.grad is None
    assert targets['target_box'].grad is None
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0
    assert model.log_temperature.grad is not None
    assert refinement['soft_boxes'].requires_grad


def test_empty_pool_template_and_geometry_degeneracy_are_safe():
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    pool['valid'].zero_()
    templates['valid'].zero_()
    model = GObservation()
    verification = model.verify(pool, templates, boxes, size)
    refinement = model.refine(pool, templates, boxes, size)
    losses = compute_g_losses(pool, templates, boxes, size, verification, targets, refinement)
    assert torch.isfinite(losses['loss_g'])
    assert not refinement['valid'].any()
    assert not refinement['yaw_valid'].any()
    assert torch.equal(refinement['boxes'], boxes)
    losses['loss_g'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_empty_recovery_never_dilutes_other_endpoint_correction_loss_or_gradient():
    pool, templates, boxes, size, targets = batch_fixture(batch=2)
    model = GObservation()
    verification = model.verify(pool, templates, boxes, size)
    refinement = model.refine(pool, templates, boxes, size)
    # 确保纠偏损失非零，且本测试不依赖随机网络的初始纠偏是否可观测。
    soft = (targets['target_box'][:, None].expand_as(boxes) + .1).detach().requires_grad_()
    refinement = dict(refinement, soft_boxes=soft,
                      supervision_valid=torch.ones(2, 4, dtype=torch.bool))
    baseline = compute_g_losses(pool, templates, boxes, size, verification, targets,
                                refinement)['loss_g_correction']
    empty_recovery = dict(refinement, reacquire=True,
                          supervision_valid=torch.zeros(2, 4, dtype=torch.bool))
    with_empty = compute_g_losses(pool, templates, boxes, size, verification, targets,
                                  [refinement, empty_recovery])['loss_g_correction']
    assert baseline > 0.
    assert torch.equal(baseline, with_empty)
    reference_gradient, = torch.autograd.grad(baseline, soft, retain_graph=True)
    actual_gradient, = torch.autograd.grad(with_empty, soft)
    assert torch.equal(reference_gradient, actual_gradient)

    # 两个来源分别只监督一个 endpoint，必须等价于一个来源监督两者。
    left = dict(refinement, supervision_valid=torch.tensor([[True]*4, [False]*4]))
    right = dict(refinement, supervision_valid=torch.tensor([[False]*4, [True]*4]))
    split = compute_g_losses(pool, templates, boxes, size, verification, targets,
                             [left, right])['loss_g_correction']
    assert torch.equal(baseline, split)


def test_raw_id_duplicates_are_not_extra_support():
    ids = torch.tensor([[4, 1, 4, -1, 2, 2]])
    valid = torch.tensor([[True, True, True, True, False, True]])
    result = unique_valid(ids, valid)
    assert result.tolist() == [[True, True, False, False, False, True]]


def test_presence_and_matchability_have_different_supervision():
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    targets['template_xyz_gt'] = templates['xyz'].detach() + 50.
    model = GObservation()
    verification = model.verify(pool, templates, boxes, size)
    labels = supervision_targets(pool, templates, boxes, size, verification, targets)
    matching = correspondence_targets(pool, templates, size, verification, targets)
    assert not labels['region_nomatch'].any()
    assert not matching['matched_current'].any()
    assert (matching['row'] == pool['xyz'].shape[1]).all()
    losses = compute_g_losses(pool, templates, boxes, size, verification, targets,
                              model.refine(pool, templates, boxes, size))
    assert losses['loss_g_correction'] == 0.
    losses['loss_g'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_coarse_vote_preserves_spatial_target_center():
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    model = GObservation()
    ep, et = model.encode(pool, size), model.encode_templates(templates, size)
    ep['features'] = torch.zeros(1, 12, 64)
    ep['features'][0, :8, :8] = torch.eye(8)
    et['features'] = torch.zeros(1, 3, 8, 64)
    et['features'][0, :, :, :8] = torch.eye(8)
    proposal = model.coarse_proposals(pool, templates, size, targets['target_box'][:, 3],
                                      encoded_pool=ep, encoded_templates=et)
    assert proposal['boxes'].shape == (1, 2, 4)
    assert torch.isfinite(proposal['boxes']).all()
    assert proposal['valid'][0, 0]
    distance = (proposal['boxes'][0, :, :2] - targets['target_box'][0, :2]).norm(dim=-1)
    assert distance.min() < .5
    assert abs(proposal['boxes'][0, 0, 2] - targets['target_box'][0, 2]) < .1


def test_coarse_group_max_and_candidate_outputs_match_original_cpu_scatter(monkeypatch):
    import models.ct_v39.g_observation as module

    def original(weights, inverse):
        groups = len(torch.unique(inverse))
        return weights.new_zeros(groups).scatter_reduce_(
            0, inverse, weights, reduce='amax', include_self=True)

    # 交错组号、重复最大票、全零/负值与空输入：保留 include_self=0 的定义。
    inverse = torch.tensor([2, 0, 1, 0, 2, 1, 1])
    for weights in (torch.tensor([.2, .8, .3, .8, .7, .4, .4]), torch.zeros(7), -torch.ones(7)):
        torch.testing.assert_close(_coarse_vote_max(weights, inverse), original(weights, inverse),
                                   rtol=0., atol=0.)
    assert _coarse_vote_max(torch.empty(0), torch.empty(0, dtype=torch.long)).numel() == 0
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    model = GObservation().eval()
    ep, et = model.encode(pool, size), model.encode_templates(templates, size)
    ep['features'] = torch.zeros(1, 12, 64)
    ep['features'][0, :8, :8] = torch.eye(8)
    et['features'] = torch.zeros(1, 3, 8, 64)
    et['features'][0, :, :, :8] = torch.eye(8)
    actual = model.coarse_proposals(pool, templates, size, targets['target_box'][:, 3],
                                   encoded_pool=ep, encoded_templates=et)
    monkeypatch.setattr(module, '_coarse_vote_max', original)
    expected = model.coarse_proposals(pool, templates, size, targets['target_box'][:, 3],
                                     encoded_pool=ep, encoded_templates=et)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0., atol=0., msg=key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='strict CUDA recovery audit requires the registered GPU environment')
def test_cuda_coarse_recovery_strict_deterministic_repeats():
    from tools.check_v39_full_batch import audit_recovery_kernel
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        device = torch.device('cuda')
        result = audit_recovery_kernel(GObservation().to(device).eval(), device)
        assert result['passed'] and result['repeated_exact']
        assert result['valid_peak_count'] >= 2
        assert result['raw_id_votes_deduplicated'] and result['strict_deterministic']
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_coarse_recovery_audit_checks_two_peaks_and_unique_raw_votes_on_cpu():
    from tools.check_v39_full_batch import audit_recovery_kernel
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        result = audit_recovery_kernel(GObservation().eval(), torch.device('cpu'))
        assert result['passed'] and result['repeated_exact']
        assert result['valid_peak_count'] == 2
        assert result['scores'] == [[6., 6.]]
        assert result['raw_id_votes_deduplicated'] and result['strict_deterministic']
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_weighted_rigid_fit_is_correct_differentiable_and_rejects_collinear_yaw():
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    model = GObservation()
    ep, et = model.encode(pool, size), model.encode_templates(templates, size)
    weights = torch.zeros(1, 1, 3, 8, 12)
    weights[0, 0, :, torch.arange(8), torch.arange(8)] = 1.
    weights.requires_grad_()
    matching = dict(weights=weights, hard=weights.detach() > .5)
    fit = model._fit(matching, ep, et, torch.zeros(1, 1, 4), size)
    assert fit['yaw_valid'].all()
    assert torch.allclose(fit['soft_boxes'][0, 0], targets['target_box'][0], atol=1e-5)
    fit['soft_boxes'].square().sum().backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    et['xyz'] = et['xyz'].clone()
    et['xyz'][..., 1] = 0.
    collinear = model._fit(matching, ep, et, torch.zeros(1, 1, 4), size)
    assert not collinear['yaw_valid'].any()


def test_zero_length_current_pool_is_safe():
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    pool = {key: value[:, :0] for key, value in pool.items()}
    targets['current_target_mask'] = targets['current_target_mask'][:, :0]
    model = GObservation()
    verification = model.verify(pool, templates, boxes, size)
    refinement = model.refine(pool, templates, boxes, size)
    losses = compute_g_losses(pool, templates, boxes, size, verification, targets, refinement)
    assert torch.isfinite(losses['loss_g'])
    assert not refinement['valid'].any()


def test_endpoint_normalization_does_not_weight_more_points_or_candidates_more():
    values = torch.tensor([[1., 9., 9., 9.], [3., 3., 3., 3.]], requires_grad=True)
    valid = torch.tensor([[True, False, False, False], [True, True, True, True]])
    positive = torch.tensor([[True, False, False, False], [False, False, False, False]])
    assert masked_mean(values, valid) == 2.
    assert _balanced_values(values, positive, valid) == 2.


def test_auxiliary_correction_respects_supervision_mask():
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    model = GObservation()
    verification = model.verify(pool, templates, boxes, size)
    refinement = model.refine(pool, templates, boxes, size)
    refinement['supervision_valid'] = torch.zeros(1, 4, dtype=torch.bool)
    losses = compute_g_losses(pool, templates, boxes, size, verification, targets, refinement)
    assert losses['loss_g_correction'] == 0.


def test_matching_checkpoint_matches_eager_loss_and_gradients():
    pool, templates, boxes, size, targets = batch_fixture(batch=1)
    torch.manual_seed(19)
    checked = GObservation()
    eager = copy.deepcopy(checked)
    eager.checkpoint_matching = False
    totals = []
    for model in (checked, eager):
        verification = model.verify(pool, templates, boxes, size)
        refinement = model.refine(pool, templates, boxes, size)
        loss = compute_g_losses(pool, templates, boxes, size, verification, targets, refinement)['loss_g']
        totals.append(loss.detach())
        loss.backward()
    assert torch.allclose(totals[0], totals[1], atol=1e-6)
    for first, second in zip(checked.parameters(), eager.parameters()):
        assert (first.grad is None) == (second.grad is None)
        if first.grad is not None:
            assert torch.allclose(first.grad, second.grad, atol=1e-6, rtol=1e-5)


def test_region_absence_uses_actual_crop_margin_not_object_box():
    points = torch.tensor([[[0., 0., 0.], [.1, .2, 0.], [-.1, -.2, 0.]]])
    pool = dict(xyz=points, valid=torch.ones(1, 3, dtype=torch.bool), point_ids=torch.arange(3)[None])
    templates = dict(xyz=points[:, None].expand(1, 3, 3, 3),
                     valid=torch.ones(1, 3, 3, dtype=torch.bool), ages=torch.zeros(1, 3))
    boxes = torch.tensor([[[4., 0., 0., 0.], [6., 0., 0., 0.]]])
    size = torch.tensor([[4., 2., 2.]])
    target = dict(target_box=torch.zeros(1, 4), current_target_mask=torch.ones(1, 3, dtype=torch.bool),
                  region_scale=1.25, region_offset=2.)
    verification = GObservation().verify(pool, templates, boxes, size)
    labels = supervision_targets(pool, templates, boxes, size, verification, target)
    assert labels['region_nomatch'].tolist() == [[0., 1.]]
    target['region_offset'] = 0.
    tight = supervision_targets(pool, templates, boxes, size, verification, target)
    assert tight['region_nomatch'].tolist() == [[1., 1.]]


def test_cpu_gelu_noncontiguous_gradient_matches_native_reference():
    generator = torch.Generator().manual_seed(73)
    value = torch.randn(1, 12, 8, 32, generator=generator)
    gradient = torch.randn(32, 8, 12, 1, generator=generator).permute(3, 2, 1, 0)
    assert not gradient.is_contiguous()
    reference_input = value.clone().requires_grad_()
    with torch.backends.mkldnn.flags(enabled=False):
        reference = torch.nn.functional.gelu(reference_input)
        reference.backward(gradient)
    actual_input = value.clone().requires_grad_()
    with torch.backends.mkldnn.flags(enabled=True):
        actual = ContiguousCpuGELU()(actual_input)
        actual.backward(gradient)
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_input.grad, reference_input.grad, atol=1e-6, rtol=1e-5)
