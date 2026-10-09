"""相同候选训练图：GT仅产生监督/台账，不参与候选使用政策。"""
import numpy as np
import torch
from torch.nn import functional as F

from models.ct_v31.losses import (compute_losses as v35_losses, balanced_mean,
    frame_endpoint_mean, oriented_iou_labels, quality_targets)
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


def compute_losses(batch, output):
    # q0/history/B1逐项计算沿用原函数，包括q0质量loss的原梯度与.5系数。
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
