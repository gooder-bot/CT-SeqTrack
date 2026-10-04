"""v36 监督标签只在此处读取；归约与系数显式登记。"""
import numpy as np
import torch
from torch.nn import functional as F
from models.ct_v31.geometry import anchor_yaw
from models.ct_v31.losses import (masked_mean, balanced_mean, frame_endpoint_mean,
    local_observation_box_terms, observation_segmentation_loss, observation_bc_loss,
    oriented_iou_labels)
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions


def endpoint_bce(logits, target, valid):
    """普通 BCE：端点内有效候选平均，再端点等权；没有类别重加权。"""
    clean = torch.where(valid, logits, torch.zeros_like(logits))
    labels = torch.where(valid, target.detach().to(logits), torch.zeros_like(logits))
    return frame_endpoint_mean(F.binary_cross_entropy_with_logits(clean, labels, reduction='none'), valid)


def legacy_mean_loss(prior, target, valid):
    """保留旧 mean 分支；不实例化或训练旧 sigma/acquisition heads。"""
    residual = target.to(prior.mean_xy) - prior.kinematic_xy.detach()
    aligned = torch.einsum('bi,bij->bj', residual, prior.axes.detach())
    unit = aligned / prior.envelope.detach().clamp_min(1e-6)
    inside = unit.abs() <= 1.
    huber = F.smooth_l1_loss(prior.unit_residual, unit.clamp(-1, 1), reduction='none', beta=.25)
    tail = .25 * F.relu(.9 - unit.sign() * prior.unit_residual)
    return masked_mean(torch.where(inside, huber, tail).mean(-1), valid)


def projected_size(size, yaw, axes):
    c, s = yaw.detach().cos(), yaw.detach().sin()
    rotation = torch.stack((c, -s, s, c), -1).reshape(-1, 2, 2)
    projection = axes.detach().transpose(-1, -2) @ rotation
    return (projection.abs() @ size[:, :2, None]).squeeze(-1).clamp_min(1e-3)


def pinball(error):
    return torch.maximum(.9 * error, -.1 * error)


@torch.no_grad()
def supervision_targets(batch, output):
    """最终框标签与真实在线支持集合相同；只用于 loss/提交后的台账。"""
    boxes = output.candidates.boxes.detach()
    target = batch['target_box'].to(boxes)
    size = batch['box_size'].to(boxes)
    target_size = batch.get('target_box_size', size).to(boxes)
    exact_iou = oriented_iou_labels(boxes, target, size, target_size)
    utility = np.zeros((*boxes.shape[:2], 2), dtype=np.float32)
    bxs, gts, szs, gszs = [x.cpu().double().numpy() for x in (boxes, target, size, target_size)]
    for i in range(len(boxes)):
        gt = LocalYawBox(gts[i], gszs[i][[1, 0, 2]])
        for j in range(4):
            iou, distance = box_metrics(LocalYawBox(bxs[i, j], szs[i][[1, 0, 2]]), gt,
                up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
            utility[i, j] = metric_contributions(iou, distance)
    evidence = output.evidence
    selected = batch['extension_labels'].gather(1, evidence.point_indices.clamp_min(0))
    selected = torch.where(evidence.point_valid, selected, torch.zeros_like(selected))
    labels = torch.cat((batch['segmentation_labels'][:, -1], selected), -1).to(boxes)
    support = output.verification.support_mask.bool()
    if support.shape[-1] != labels.shape[-1]:
        raise ValueError('verification pool must retain B0 then selected extension order')
    count = support.sum(-1)
    foreground = (support & (labels > .5)[:, None]).sum(-1)
    purity = foreground.to(boxes) / count.clamp_min(1)
    measurement = (foreground >= 1) & (purity >= .5) & (exact_iou >= .5)
    measurement[:, 1] = False
    return dict(utility=boxes.new_tensor(utility), measurement=measurement.to(boxes),
                exact_iou=exact_iou, foreground_count=foreground, support_count=count,
                support_purity=purity, support_labels=labels)


def compute_losses(batch, output, *, arm='A', time_scale=.5):
    target = batch['target_box']
    size = batch['box_size'].to(target).clamp_min(1e-3)
    obs, prior, evidence, candidates = output.observation, output.prior, output.evidence, output.candidates
    zero = obs.coarse_box.sum() * 0.
    losses = {}
    yaw_anchor = anchor_yaw(batch, target)
    for name, prediction, truth, valid, cw, aw, endpoint in (
        ('coarse', obs.coarse_box, target, obs.sequence_valid, 2., 10., False),
        ('main', candidates.boxes[:, 0], target, obs.sequence_valid, 2., 10., False),
        ('history', candidates.history_boxes, batch['history_target_boxes'], batch['history_valid'], .2, 1., True)):
        center, angle = local_observation_box_terms(prediction, truth, valid, yaw_anchor, cw, aw,
                                                    endpoint_mean=endpoint)
        losses['loss_' + name] = center + angle
        losses['loss_' + name + '_center'] = center.detach()
        losses['loss_' + name + '_angle'] = angle.detach()
    losses['loss_seg'], _, _ = observation_segmentation_loss(obs.segmentation_logits,
        batch['segmentation_labels'], batch['point_valid'])
    losses['loss_bc'], _, _ = observation_bc_loss(obs.bc_prediction, batch['bc_targets'], batch['point_valid'])

    origin_target = batch['prior_origin_target_world'].detach().double()
    current_target = batch['current_target_world'].detach().double()
    dims = projected_size(size, prior.origin_world[:, 3].to(size), prior.axes.to(size))
    if arm == 'D':
        losses['loss_physical'] = zero
    elif arm == 'E':
        valid = batch['physical_valid'].bool() & prior.legacy_valid.bool()
        losses['loss_physical'] = legacy_mean_loss(prior, batch['physical_displacement'], valid)
    else:
        physical = (current_target[:, :2] - origin_target[:, :2]).to(prior.mean_xy)
        error = torch.einsum('bi,bij->bj', prior.mean_xy - physical, prior.axes.detach())
        scale = dims * (prior.horizon / time_scale).clamp_min(1)[:, None]
        terms = F.smooth_l1_loss(error / scale, torch.zeros_like(error), reduction='none').mean(-1)
        losses['loss_physical'] = masked_mean(terms, prior.motion_valid & batch['physical_valid'].bool())
    anchor_error = (origin_target[:, :2] - prior.origin_world.detach()[:, :2]).to(size)
    current_error = (current_target[:, :2] - prior.world_box.detach()[:, :2]).to(size)
    anchor_error = torch.einsum('bi,bij->bj', anchor_error, prior.axes.detach()).abs().detach()
    current_error = torch.einsum('bi,bij->bj', current_error, prior.axes.detach()).abs().detach()
    anchor_term = pinball(anchor_error - prior.range_anchor).mean(-1)
    current_term = pinball(current_error - prior.range_for_loss).mean(-1)
    losses['loss_range_anchor'] = masked_mean(anchor_term, prior.prior_valid)
    losses['loss_range_current'] = masked_mean(current_term, prior.prior_valid)
    losses['loss_range'] = .5 * (losses['loss_range_anchor'] + losses['loss_range_current'])

    positive = torch.zeros_like(evidence.mode_valid)
    if arm != 'C':
        ext_labels, ext_valid = batch['extension_labels'].to(target), batch['extension_valid'].bool()
        terms = F.binary_cross_entropy_with_logits(evidence.identity_logits, ext_labels.clamp(0, 1), reduction='none')
        losses['loss_identity'] = balanced_mean(terms, ext_labels > .5, ext_valid)
        selected_labels = ext_labels.gather(1, evidence.point_indices.clamp_min(0))
        selected_fg = (selected_labels > .5) & evidence.point_valid
        vote_error = (evidence.vote_xyz - target[:, None, :3]) / size[:, None]
        vote_terms = F.smooth_l1_loss(vote_error, torch.zeros_like(vote_error), reduction='none').mean(-1)
        losses['loss_vote'] = frame_endpoint_mean(vote_terms, selected_fg)
        scale = (.5 * size[:, :2].norm(dim=1)).clamp_min(1e-3)
        distance = (evidence.vote_xyz.detach() - target[:, None, :3]).norm(dim=-1)
        reliability = selected_labels.clamp(0, 1) / (1. + distance / scale[:, None])
        terms = F.binary_cross_entropy_with_logits(evidence.reliability_logits, reliability.detach(), reduction='none')
        losses['loss_reliability'] = balanced_mean(terms, selected_fg, evidence.point_valid)
        members = evidence.members & evidence.point_valid[:, None]
        fg_count = (members & selected_fg[:, None]).sum(-1)
        purity = fg_count.to(target) / members.sum(-1).clamp_min(1)
        positive = evidence.mode_valid & candidates.valid[:, 2:] & (fg_count >= 1) & (purity >= .5)
    else:
        losses.update(loss_identity=zero, loss_vote=zero, loss_reliability=zero)
    seed_error = F.smooth_l1_loss(candidates.seed_boxes[..., :3], target[:, None, :3].expand(-1, 2, -1), reduction='none').mean(-1)
    final_error = F.smooth_l1_loss(candidates.boxes[:, 2:, :3], target[:, None, :3].expand(-1, 2, -1), reduction='none').mean(-1)
    angle = 1. - torch.cos(candidates.boxes[:, 2:, 3] - target[:, None, 3])
    losses['loss_modes_center'] = 2. * frame_endpoint_mean(.5 * (seed_error + final_error), positive)
    losses['loss_modes_angle'] = 10. * frame_endpoint_mean(angle, positive)
    losses['loss_modes'] = losses['loss_modes_center'] + losses['loss_modes_angle']

    targets = supervision_targets(batch, output)
    logits = output.verification.logits
    losses['loss_success'] = endpoint_bce(logits[..., 0], targets['utility'][..., 0], candidates.valid)
    losses['loss_precision'] = endpoint_bce(logits[..., 1], targets['utility'][..., 1], candidates.valid)
    measurement_valid = candidates.valid.clone()
    measurement_valid[:, 1] = False
    losses['loss_measurement'] = endpoint_bce(logits[..., 2], targets['measurement'], measurement_valid)
    losses['loss_total'] = (losses['loss_coarse'] + losses['loss_main'] + losses['loss_history']
        + .1 * losses['loss_seg'] + losses['loss_bc'] + .1 * losses['loss_physical']
        + .1 * losses['loss_range'] + .2 * losses['loss_identity'] + losses['loss_vote']
        + .1 * losses['loss_reliability'] + losses['loss_modes']
        + .125 * losses['loss_success'] + .125 * losses['loss_precision'] + .25 * losses['loss_measurement'])
    return losses
