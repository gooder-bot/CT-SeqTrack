"""相同候选训练图：GT仅产生监督/台账，不参与候选使用政策。"""
import numpy as np
import torch
from torch.nn import functional as F

from models.ct_v31.losses import (compute_losses as v35_losses, balanced_mean,
    frame_endpoint_mean, oriented_iou_labels, quality_targets, masked_mean,
    local_observation_box_terms, observation_segmentation_loss, observation_bc_loss)
from models.ct_v31.geometry import anchor_yaw
from models.ct_v31.motion import physical_motion_uncertainty_loss
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions


def endpoint_bce(logits, target, valid):
    values = F.binary_cross_entropy_with_logits(torch.where(valid, logits, 0.),
        torch.where(valid, target.detach().to(logits), 0.), reduction='none')
    return frame_endpoint_mean(values, valid), values


def pairwise_gap_terms(utility, target, valid):
    """有效候选对内等权；标签差确定margin与权重，不额外调阈值。"""
    count = utility.shape[1]
    left, right = torch.triu_indices(count, count, 1, device=utility.device)
    pair_valid = valid[:, left] & valid[:, right]
    gap = (target[:, left] - target[:, right]).detach()
    predicted = utility[:, left] - utility[:, right]
    terms = gap.abs() * F.relu(gap.abs() - gap.sign() * predicted)
    return terms, pair_valid


def pairwise_gap_loss(utility, target, valid):
    terms, pair_valid = pairwise_gap_terms(utility, target, valid)
    return frame_endpoint_mean(terms, pair_valid)


@torch.no_grad()
def supervision_targets(batch, output):
    boxes = output.candidates.boxes.detach()
    target = batch['target_box'].detach().to(boxes)
    size = batch['box_size'].detach().to(boxes)
    target_size = batch.get('target_box_size', size).detach().to(boxes)
    exact = oriented_iou_labels(boxes, target, size, target_size)
    utility = np.zeros((*boxes.shape[:2], 2), np.float32)
    bxs, gts, szs, gszs = [value.cpu().double().numpy() for value in (boxes, target, size, target_size)]
    for row in range(len(boxes)):
        truth = LocalYawBox(gts[row], gszs[row][[1, 0, 2]])
        for candidate in range(4):
            iou, distance = box_metrics(LocalYawBox(bxs[row, candidate], szs[row][[1, 0, 2]]), truth,
                up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
            utility[row, candidate] = metric_contributions(iou, distance)
    quality = quality_targets(boxes, target, size, torch.ones_like(exact), exact)
    return dict(utility=boxes.new_tensor(utility), commit_quality=quality.detach(), exact_iou=exact.detach())


def _record_endpoint(sink, name, values, valid, coefficient):
    if sink is None:
        return
    valid = valid.detach().bool()
    values = values.detach()
    if values.ndim > 1:
        count = valid.sum(-1)
        per_endpoint = torch.where(valid, values, 0.).sum(-1) / count.clamp_min(1)
        active = count > 0
    else:
        per_endpoint, active = values, valid
    numerator = torch.where(active, coefficient * per_endpoint, 0.)
    denominator = active.to(values)
    sink[name] = dict(numerator=numerator, denominator=denominator,
                      batch_contribution=numerator / denominator.sum().clamp_min(1))


def _record_balanced(sink, name, values, foreground, valid, coefficient):
    """原batch FG/BG归约分别记录，不能假装是端点等权loss。"""
    if sink is None:
        return
    groups = (valid.bool() & foreground.bool(), valid.bool() & ~foreground.bool())
    group_count = sum(mask.any().to(values) for mask in groups).clamp_min(1)
    for suffix, mask in zip(('foreground', 'background'), groups):
        denominator = mask.sum(-1).to(values).detach()
        numerator = (torch.where(mask, values.detach(), 0.).sum(-1) * coefficient / group_count).detach()
        sink[name + '_' + suffix] = dict(numerator=numerator, denominator=denominator,
            batch_contribution=numerator / denominator.sum().clamp_min(1))


def _engineering_base_losses(batch, output, q0_exact):
    """v37 专用 B0/B1 适配；逐项保持旧函数(False,False)及日志字典。

旧模型默认函数不变。仅删除乘零的 mode/pad IoU 标签计算，q0 exact
由同次监督缓存提供；仍构造原四槽质量张量以保持归约与 fallback 监督。
    """
    target = batch['target_box']
    size = batch['box_size'].to(target).clamp_min(1e-3)
    observation, prior, decoder = output.observation, output.prior, output.decoder
    point_valid = batch['point_valid'].bool()
    labels = batch['segmentation_labels'].long()
    zero = observation.coarse_box.sum() * 0.
    losses = {}
    yaw_anchor = anchor_yaw(batch, target)
    for name, predicted, truth, valid, center_weight, angle_weight, endpoint_mean in (
        ('coarse', observation.coarse_box, target, observation.sequence_valid, 2., 10., False),
        ('main', decoder.hypothesis_boxes[:, 0], target, observation.sequence_valid, 2., 10., False),
        ('history', decoder.history_boxes, batch['history_target_boxes'],
         batch['history_valid'].bool(), .2, 1., True)):
        center, angle = local_observation_box_terms(predicted, truth, valid, yaw_anchor,
            center_weight, angle_weight, endpoint_mean=endpoint_mean)
        losses['loss_' + name] = center + angle
        losses['loss_' + name + '_center'] = center.detach()
        losses['loss_' + name + '_angle'] = angle.detach()
    (losses['loss_seg'], losses['loss_seg_current'], losses['loss_seg_history']) = observation_segmentation_loss(
        observation.segmentation_logits, labels, point_valid)
    losses['loss_bc'], bc_current, bc_history = observation_bc_loss(
        observation.bc_prediction, batch['bc_targets'], point_valid)
    losses['loss_bc_current'], losses['loss_bc_history'] = bc_current.detach(), bc_history.detach()
    physical_valid = batch['physical_valid'].bool() & prior.valid.bool()
    terms = physical_motion_uncertainty_loss(
        prior.mean_xy, batch['physical_displacement'], prior.log_sigma,
        prior.direction_xy.detach(), physical_valid,
        kinematic_xy=prior.kinematic_xy.detach(), envelope_parallel_perp=prior.envelope.detach(),
        residual_unit_parallel_perp=prior.unit_residual)
    losses['loss_physical'] = masked_mean(terms['mean_per_sample'], terms['valid'])
    losses['loss_sigma'] = masked_mean(terms['nll_per_sample'], terms['valid'])
    error = batch['acquisition_target'].detach() - prior.acquisition_fraction
    pinball = torch.maximum(.9 * error, -.1 * error).mean(-1)
    valid = batch['acquisition_valid'].bool()
    demand = batch.get('acquisition_demand', valid).bool()
    losses['loss_acquisition'] = balanced_mean(pinball, demand, valid)
    purity = target.new_ones((len(target), 4))
    mode_positive = torch.zeros((len(target), 3), dtype=torch.bool, device=target.device)
    losses.update(loss_identity=zero, loss_vote=zero, loss_reliability=zero)
    purity[:, 1:] = 0
    mode_center, mode_angle = zero, zero
    losses['loss_modes'] = mode_center + mode_angle
    losses['loss_modes_center'], losses['loss_modes_angle'] = mode_center.detach(), mode_angle.detach()
    iou = decoder.hypothesis_boxes.new_zeros(decoder.hypothesis_boxes.shape[:2])
    iou[:, :1] = q0_exact
    quality_target = quality_targets(decoder.hypothesis_boxes, target, size, purity, iou)
    quality_error = F.binary_cross_entropy_with_logits(decoder.quality_logits, quality_target, reduction='none')
    main_quality = masked_mean(quality_error[:, 0], decoder.hypothesis_valid[:, 0])
    diagnostic_sink = getattr(output, 'loss_statistics', None)
    if diagnostic_sink is not None:
        quality_valid = decoder.hypothesis_valid[:, 0].detach().bool()
        denominator = quality_valid.to(quality_error)
        numerator = torch.where(quality_valid, .5 * quality_error[:, 0].detach(), 0.)
        diagnostic_sink['loss_main_quality'] = dict(numerator=numerator, denominator=denominator,
            batch_contribution=numerator / denominator.sum().clamp_min(1.))
    mode_quality = zero
    losses['loss_quality'] = main_quality + mode_quality
    losses['loss_total'] = (
        losses['loss_coarse'] + losses['loss_main'] + losses['loss_modes'] + losses['loss_history']
        + .1 * losses['loss_seg'] + losses['loss_bc']
        + .1 * losses['loss_physical'] + .05 * losses['loss_sigma'] + .05 * losses['loss_acquisition']
        + .2 * losses['loss_identity'] + losses['loss_vote'] + .1 * losses['loss_reliability']
        + .5 * losses['loss_quality'])
    losses['loss_contribution_seg'] = (.1 * losses['loss_seg']).detach()
    losses['loss_contribution_quality'] = (.5 * losses['loss_quality']).detach()
    losses['quality_target_mean'] = masked_mean(quality_target, decoder.hypothesis_valid).detach()
    losses['mode_positive_fraction'] = mode_positive.to(target).mean().detach()
    return losses


def compute_losses(batch, output):
    # q0/history/B1逐项计算沿用原函数，包括q0质量loss的原梯度与.5系数。
    from utils.execution_lab import enabled
    targets = None
    if enabled('labels'):
        from .engineering_geometry import supervision_geometry
        targets, q0_exact = supervision_geometry(batch, output)
        losses = _engineering_base_losses(batch, output, q0_exact)
    else:
        losses = v35_losses(batch, output, enable_b1=True, enable_b2=False, enable_b3=False)
    base_total = losses['loss_total']
    target, size = batch['target_box'], batch['box_size'].to(batch['target_box']).clamp_min(1e-3)
    evidence, candidates = output.evidence, output.candidates
    sink = getattr(output, 'loss_statistics', None)
    labels, valid = batch['extension_labels'].detach().to(target), batch['extension_valid'].bool()
    terms = F.binary_cross_entropy_with_logits(evidence.identity_logits, labels.clamp(0, 1), reduction='none')
    losses['loss_identity'] = balanced_mean(terms, labels > .5, valid)
    _record_balanced(sink, 'loss_identity', terms, labels > .5, valid, .2)
    selected_labels = labels.gather(1, evidence.point_indices.clamp_min(0))
    foreground = (selected_labels > .5) & evidence.point_valid
    error = (evidence.vote_xyz - target[:, None, :3].detach()) / size[:, None].detach()
    vote = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(-1)
    losses['loss_vote'] = frame_endpoint_mean(vote, foreground)
    _record_endpoint(sink, 'loss_vote', vote, foreground, 1.)
    scale = (.5 * size[:, :2].norm(dim=1)).clamp_min(1e-3)
    distance = (evidence.vote_xyz.detach() - target[:, None, :3].detach()).norm(dim=-1)
    reliability = (selected_labels.clamp(0, 1) / (1. + distance / scale[:, None])).detach()
    terms = F.binary_cross_entropy_with_logits(evidence.reliability_logits, reliability, reduction='none')
    losses['loss_reliability'] = balanced_mean(terms, foreground, evidence.point_valid)
    _record_balanced(sink, 'loss_reliability', terms, foreground, evidence.point_valid, .1)
    members = evidence.members.detach().bool() & evidence.point_valid[:, None]
    count = (members & foreground[:, None]).sum(-1)
    purity = count.to(target) / members.sum(-1).clamp_min(1)
    positive = evidence.mode_valid & candidates.valid[:, 2:] & (count >= 1) & (purity >= .5)
    seed = F.smooth_l1_loss(candidates.seed_boxes[..., :3], target[:, None, :3].detach().expand(-1, 2, -1),
                           reduction='none').mean(-1)
    final = F.smooth_l1_loss(candidates.boxes[:, 2:, :3], target[:, None, :3].detach().expand(-1, 2, -1),
                            reduction='none').mean(-1)
    angle = 1. - torch.cos(candidates.boxes[:, 2:, 3] - target[:, None, 3].detach())
    center = .5 * (seed + final)
    losses['loss_modes_center'] = 2. * frame_endpoint_mean(center, positive)
    losses['loss_modes_angle'] = 10. * frame_endpoint_mean(angle, positive)
    losses['loss_modes'] = losses['loss_modes_center'] + losses['loss_modes_angle']
    _record_endpoint(sink, 'loss_modes_center', center, positive, 2.)
    _record_endpoint(sink, 'loss_modes_angle', angle, positive, 10.)
    if targets is None:
        targets = supervision_targets(batch, output)
    output.supervision_targets = targets
    logits = output.verification.logits
    losses['loss_success'], st = endpoint_bce(logits[..., 0], targets['utility'][..., 0], candidates.valid)
    losses['loss_precision'], pt = endpoint_bce(logits[..., 1], targets['utility'][..., 1], candidates.valid)
    losses['loss_mode_commit_quality'], qt = endpoint_bce(logits[:, 2:, 2], targets['commit_quality'][:, 2:], candidates.valid[:, 2:])
    ranking, pair_valid = pairwise_gap_terms(output.verification.utility, targets['utility'].mean(-1), candidates.valid)
    losses['loss_ranking'] = frame_endpoint_mean(ranking, pair_valid)
    for name, values, mask, coefficient in (
            ('loss_success', st, candidates.valid, .125), ('loss_precision', pt, candidates.valid, .125),
            ('loss_mode_commit_quality', qt, candidates.valid[:, 2:], .5), ('loss_ranking', ranking, pair_valid, .1)):
        _record_endpoint(sink, name, values, mask, coefficient)
    losses['loss_total'] = (base_total + .2 * losses['loss_identity'] + losses['loss_vote']
        + .1 * losses['loss_reliability'] + losses['loss_modes']
        + .125 * losses['loss_success'] + .125 * losses['loss_precision']
        + .5 * losses['loss_mode_commit_quality'] + .1 * losses['loss_ranking'])
    losses['mode_positive_fraction'] = positive.to(target).mean().detach()
    return losses
