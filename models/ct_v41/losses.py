"""v41 原强 B1 + 四候选效用、q0/R 测量资格和 XYZ 恢复监督。"""
import numpy as np
import torch
from torch.nn import functional as F

from .base_losses import compute_losses as base_losses, frame_endpoint_mean
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions
from .verification import current_candidate_support
from .recovery import recovery_loss_terms


def endpoint_bce(logits, target, valid):
    terms = F.binary_cross_entropy_with_logits(torch.where(valid, logits, 0.),
        torch.where(valid, target.detach().to(logits), 0.), reduction='none')
    return frame_endpoint_mean(terms, valid), terms


def pairwise_gap_terms(utility, target, valid):
    """去重后合法无序对的 signed gap SmoothL1，包含真实平局。"""
    left, right = torch.triu_indices(utility.shape[1], utility.shape[1], 1, device=utility.device)
    pair_valid = valid[:, left] & valid[:, right]
    difference = (utility[:, left]-utility[:, right])-(target[:, left]-target[:, right]).detach()
    terms = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction='none', beta=1.)
    return terms, pair_valid


def pairwise_gap_loss(utility, target, valid):
    terms, mask = pairwise_gap_terms(utility, target, valid)
    return frame_endpoint_mean(terms, mask)


@torch.no_grad()
def supervision_targets(batch, output):
    boxes = output.candidates.boxes.detach()
    target = batch['target_box'].detach().to(boxes)
    size = batch['box_size'].detach().to(boxes)
    target_size = batch.get('target_box_size', size).detach().to(boxes)
    # T 是 >=.5 的离散事件，必须保留与候选台账相同的 double 几何精度。
    # 原强 B1 的 quality_labels 仍由原 loss 原样计算，不能修改其数学。
    exact_values = np.zeros(boxes.shape[:2], np.float64)
    metrics = np.zeros((*boxes.shape[:2], 2), np.float32)
    bxs, gts, szs, gszs = [value.cpu().double().numpy() for value in (boxes, target, size, target_size)]
    for row in range(len(boxes)):
        truth = LocalYawBox(gts[row], gszs[row][[1, 0, 2]])
        for candidate in range(boxes.shape[1]):
            prediction = LocalYawBox(bxs[row, candidate], szs[row][[1, 0, 2]])
            exact_values[row, candidate] = box_metrics(prediction, truth,
                up_axis=(0, 0, 1), mode='geometry_exact', dim=3)[0]
            iou, distance = box_metrics(prediction, truth,
                up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
            metrics[row, candidate] = metric_contributions(iou, distance)
    exact = torch.as_tensor(exact_values, device=boxes.device, dtype=torch.float64)
    metrics = boxes.new_tensor(metrics)
    support = current_candidate_support(output.observation, batch, boxes)
    labels = batch['segmentation_labels'][:, -1].detach()
    count = support['count']
    purity = (support['mask'] & (labels[:, None] > .5)).sum(-1).to(boxes)/count.clamp_min(1)
    real = output.candidates.measurement_candidate.detach() & output.candidates.raw_valid.detach()
    trusted = (exact >= .5) & (count >= 3) & (purity >= .5)
    return dict(exact_iou=exact, success=metrics[..., 0], precision=metrics[..., 1],
        utility=metrics.mean(-1), trusted=(trusted & real).to(boxes),
        support_count=count, support_purity=purity, trusted_valid=real)


def _record(sink, name, values, valid, coefficient):
    if sink is None:
        return
    values, valid = values.detach(), valid.detach().bool()
    if values.ndim > 1:
        count = valid.sum(-1)
        endpoint = torch.where(valid, values, 0.).sum(-1)/count.clamp_min(1)
        active = count > 0
    else:
        endpoint, active = values, valid
    numerator = torch.where(active, coefficient*endpoint, 0.)
    denominator = active.to(values)
    sink[name] = dict(numerator=numerator, denominator=denominator,
        batch_contribution=numerator/denominator.sum().clamp_min(1))


def compute_recovery_losses(batch, output, *, recovery_weight=.1):
    """辅助流只调用此函数；不计算主路/B1/M损失，不反传其参数。"""
    if output.verification is None or output.recovery is None:
        raise ValueError('v41 recovery losses require verification and recovery output')
    if not np.isfinite(recovery_weight) or recovery_weight < 0:
        raise ValueError('v41 recovery loss weight must be finite and nonnegative')
    losses = {}
    target = supervision_targets(batch, output)
    output.supervision_targets = target
    logits, valid = output.verification.logits, output.candidates.valid
    losses['loss_success'], success = endpoint_bce(logits[..., 0], target['success'], valid)
    losses['loss_precision'], precision = endpoint_bce(logits[..., 1], target['precision'], valid)
    # 对每个端点内的测量候选平均；q0/R 数目变化不把 T 权重翻倍。
    losses['loss_trusted_write'], trusted = endpoint_bce(logits[..., 2], target['trusted'], target['trusted_valid'])
    gap, pairs = pairwise_gap_terms(output.verification.utility, target['utility'], valid)
    losses['loss_ranking'] = frame_endpoint_mean(gap, pairs)
    recovery = recovery_loss_terms(batch, output.recovery)
    losses['loss_recovery_xyz'] = recovery['loss']
    losses['recovery_supervised_count'] = recovery['valid'].to(logits).sum().detach()
    sink = getattr(output, 'loss_statistics', None)
    for name, terms, mask, weight in (
            ('loss_success', success, valid, .125), ('loss_precision', precision, valid, .125),
            ('loss_trusted_write', trusted, target['trusted_valid'], .5), ('loss_ranking', gap, pairs, .1),
            ('loss_recovery_xyz', recovery['terms'], recovery['valid'], recovery_weight)):
        _record(sink, name, terms, mask, weight)
    losses['loss_total'] = .125*losses['loss_success'] + .125*losses['loss_precision'] \
        + .5*losses['loss_trusted_write'] + .1*losses['loss_ranking'] \
        + recovery_weight*losses['loss_recovery_xyz']
    return losses


def compute_losses(batch, output, *, enable_b1=True, recovery_weight=.1):
    losses = base_losses(batch, output, enable_b1=enable_b1)
    original = losses['loss_total']
    recovery = compute_recovery_losses(batch, output, recovery_weight=recovery_weight)
    losses.update(recovery)
    losses['loss_total'] = original + recovery['loss_total']
    return losses
