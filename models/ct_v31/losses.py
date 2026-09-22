"""v31 唯一目标集合：定位/身份/物理位移各司其职。"""

import numpy as np
import torch
from torch.nn import functional as F

from .motion import physical_motion_uncertainty_loss
from utils.tracking_metrics import LocalYawBox, box_metrics


def seqtrack_segmentation_cross_entropy(logits, labels, valid_mask=None):
    """原整批加权 CE，避开 CUDA 空间 NLL 的非确定性 mean 归约。

    保留 [B,2,N] 上的原 log_softmax 计算，只把所得 log-probability
    转成普通 NLL 的 [B*N,2] 输入。分母仍为全 batch 的 sum(weight[label])，
    不按 view、帧或点分别平均，也不修改全局 deterministic 设置。
    """
    if (logits.ndim != 3 or logits.shape[1] != 2
            or labels.shape != (logits.shape[0], logits.shape[2])):
        raise ValueError('SeqTrack segmentation requires logits [B,2,N] and labels [B,N]')
    log_probabilities = F.log_softmax(logits, dim=1)
    rows = log_probabilities.movedim(1, -1).reshape(-1, 2)
    if valid_mask is not None:
        valid = valid_mask.bool().reshape(-1)
        target = labels.reshape(-1).masked_fill(~valid, 0)
        weights = logits.new_tensor([.5, 2.])
        loss = F.nll_loss(rows, target, weight=weights, reduction='none')
        return torch.where(valid, loss, torch.zeros_like(loss)).sum() / (
            weights[target] * valid).sum().clamp_min(1e-12)
    return F.nll_loss(rows, labels.reshape(-1),
                      weight=logits.new_tensor([.5, 2.]), reduction='mean')


def masked_mean(value, valid):
    valid = valid.to(device=value.device, dtype=value.dtype)
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(value)
    return torch.where(valid.bool(), value, torch.zeros_like(value)).sum() / valid.sum().clamp_min(1)


def balanced_mean(value, foreground, valid):
    """仅对实际存在的前/背景组等权，不把零个前景当一个假样本。"""
    positive = valid.bool() & foreground.bool()
    negative = valid.bool() & ~foreground.bool()
    p_exists, n_exists = positive.any().to(value), negative.any().to(value)
    return (masked_mean(value, positive) * p_exists + masked_mean(value, negative) * n_exists) / (p_exists + n_exists).clamp_min(1)


def box_loss(predicted, target, valid, center_weight=2., angle_weight=10.):
    xyz = F.smooth_l1_loss(predicted[..., :3], target[..., :3], reduction='none').mean(-1)
    yaw = 1. - torch.cos(predicted[..., 3] - target[..., 3])
    return center_weight * masked_mean(xyz, valid) + angle_weight * masked_mean(yaw, valid)


@torch.no_grad()
def oriented_iou_labels(boxes, target, size, target_size=None):
    """标签用真实直立有向 3D IoU；不能把 GT 几何送入模型 forward。"""
    target_size = size if target_size is None else target_size
    estimates, truth, sizes, true_sizes = [x.detach().cpu().double().numpy()
                                         for x in (boxes, target, size, target_size)]
    result = np.zeros(estimates.shape[:2], dtype=np.float32)
    for i in range(len(estimates)):
        gt = LocalYawBox(truth[i], true_sizes[i][[1, 0, 2]])
        for j in range(estimates.shape[1]):
            pred = LocalYawBox(estimates[i, j], sizes[i][[1, 0, 2]])
            result[i, j] = box_metrics(pred, gt, up_axis=(0, 0, 1), mode='geometry_exact')[0]
    return torch.as_tensor(result, device=boxes.device, dtype=boxes.dtype)


@torch.no_grad()
def quality_targets(boxes, target, size, purity, iou):
    scale = (.5 * size[:, :2].norm(dim=1)).clamp_min(1e-3)
    distance = (boxes[..., :3] - target[:, None, :3]).norm(dim=-1)
    return purity * (.5 + .5 * iou) / (1. + distance / scale[:, None])


def compute_losses(batch, output, *, enable_b1=True, enable_b2=True, enable_b3=True):
    target = batch['target_box']
    size = batch['box_size'].to(target).clamp_min(1e-3)
    observation, prior, evidence, decoder = output.observation, output.prior, output.evidence, output.decoder
    point_valid = batch['point_valid'].bool()
    labels = batch['segmentation_labels'].long()
    zero = observation.coarse_box.sum() * 0.
    losses = {}
    losses['loss_coarse'] = box_loss(observation.coarse_box, target, observation.current_valid)
    losses['loss_main'] = box_loss(decoder.hypothesis_boxes[:, 0], target, observation.current_valid)
    losses['loss_history'] = box_loss(decoder.history_boxes, batch['history_target_boxes'],
                                      batch['history_valid'].bool(), .2, 1.)
    logits = observation.segmentation_logits.permute(0, 3, 1, 2).flatten(2)
    losses['loss_seg'] = seqtrack_segmentation_cross_entropy(logits, labels.flatten(1), point_valid.flatten(1))
    # 无 GT 前景的帧不对背景施加 box-distance 回归；判定仅存在于 loss。
    frame_fg = ((labels > 0) & point_valid).any(-1)
    losses['loss_bc'] = masked_mean(F.smooth_l1_loss(
        observation.bc_prediction, batch['bc_targets'], reduction='none'), point_valid & frame_fg[..., None])

    if enable_b1:
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
    else:
        losses.update(loss_physical=zero, loss_sigma=zero, loss_acquisition=zero)

    purity = target.new_ones((len(target), 4))
    mode_positive = torch.zeros((len(target), 3), dtype=torch.bool, device=target.device)
    if enable_b2:
        extension_labels = batch['extension_labels'].to(target)
        extension_valid = batch['extension_valid'].bool()
        identity_bce = F.binary_cross_entropy_with_logits(evidence.identity_logits, extension_labels, reduction='none')
        losses['loss_identity'] = balanced_mean(identity_bce, extension_labels > .5, extension_valid)
        selected_labels = extension_labels.gather(1, evidence.point_indices.clamp_min(0))
        selected_fg = (selected_labels > .5) & evidence.point_valid
        vote_error = (evidence.vote_xyz - target[:, None, :3]) / size[:, None]
        losses['loss_vote'] = masked_mean(F.smooth_l1_loss(vote_error, torch.zeros_like(vote_error), reduction='none'), selected_fg)
        scale = (.5 * size[:, :2].norm(dim=1)).clamp_min(1e-3)
        distance = (evidence.vote_xyz.detach() - target[:, None, :3]).norm(dim=-1)
        reliability_target = selected_labels / (1. + distance / scale[:, None])
        reliability_bce = F.binary_cross_entropy_with_logits(evidence.reliability_logits, reliability_target.detach(), reduction='none')
        losses['loss_reliability'] = balanced_mean(reliability_bce, selected_fg, evidence.point_valid)
        members = evidence.members.bool() & evidence.point_valid[:, None]
        foreground_count = (members & selected_fg[:, None]).sum(-1)
        purity[:, 1:] = foreground_count.to(target) / members.sum(-1).clamp_min(1)
        mode_positive = (purity[:, 1:] >= .5) & (foreground_count >= 1) & evidence.mode_valid
    else:
        losses.update(loss_identity=zero, loss_vote=zero, loss_reliability=zero)
        purity[:, 1:] = 0

    losses['loss_modes'] = (box_loss(decoder.hypothesis_boxes[:, 1:], target[:, None].expand(-1, 3, -1),
                                     mode_positive & decoder.hypothesis_valid[:, 1:]) if enable_b3 else zero)
    iou = oriented_iou_labels(decoder.hypothesis_boxes, target, size, batch.get('target_box_size'))
    quality_target = quality_targets(decoder.hypothesis_boxes, target, size, purity, iou)
    quality_error = F.binary_cross_entropy_with_logits(decoder.quality_logits, quality_target, reduction='none')
    # q0 与测量模式分别归一化，纯背景模式仍必须学习低质量。
    main_quality = masked_mean(quality_error[:, 0], decoder.hypothesis_valid[:, 0])
    mode_quality = balanced_mean(quality_error[:, 1:], purity[:, 1:] > 0,
                                 decoder.hypothesis_valid[:, 1:]) if enable_b3 else zero
    losses['loss_quality'] = main_quality + mode_quality
    losses['loss_total'] = (
        losses['loss_coarse'] + losses['loss_main'] + losses['loss_modes'] + losses['loss_history']
        + .1 * losses['loss_seg'] + losses['loss_bc']
        + .1 * losses['loss_physical'] + .05 * losses['loss_sigma'] + .05 * losses['loss_acquisition']
        + .2 * losses['loss_identity'] + losses['loss_vote'] + .1 * losses['loss_reliability']
        + .5 * losses['loss_quality'])
    losses['quality_target_mean'] = masked_mean(quality_target, decoder.hypothesis_valid).detach()
    losses['mode_positive_fraction'] = mode_positive.to(target).mean().detach()
    return losses
