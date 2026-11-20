"""v41 原强 B1 + 四候选效用、q0/R 测量资格和 XYZ 恢复监督。"""
import numpy as np
import torch
from torch.nn import functional as F

from .base_losses import compute_losses as base_losses, frame_endpoint_mean
from utils.tracking_metrics import metric_contributions
from .cpu_geometry import training_box_metrics
from .verification import final_candidate_support
from .recovery import recovery_loss_terms
from .input import retained_geometry_sources


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
def _supervision_targets(batch, output, *, include_q0=False, reference_geometry=None):
    boxes = output.candidates.boxes.detach()
    target = batch['target_box'].detach().to(boxes)
    size = batch['box_size'].detach().to(boxes)
    target_size = batch.get('target_box_size', size).detach().to(boxes)
    # T 是 >=.5 的离散事件，必须保留与候选台账相同的 double 几何精度。
    # 原强 B1 的 quality_labels 仍由原 loss 原样计算，不能修改其数学。
    # 旧 quality 的框可以是空观测的 q0，且 size 的下限只属于这一个标签。
    # 非标准混合 dtype 交回旧 quality 实现，保留各输入独立转 double 的规则。
    reuse_q0 = include_q0 and all(value.dtype == boxes.dtype for value in (
        batch['target_box'], batch['box_size'], batch.get('target_box_size', size), output.decoder.q0_box))
    sources = retained_geometry_sources(batch, boxes)
    fields = {'boxes': boxes.flatten(1)}
    if reuse_q0:
        fields['q0'] = output.decoder.q0_box.detach().to(boxes)
    if sources is None:
        fields.update(target=target, size=size, target_size=target_size)
        if reuse_q0:
            fields['quality_size'] = size.clamp_min(1e-3)
    packet = torch.cat(list(fields.values()), dim=1) if len(fields) > 1 else fields['boxes']
    host = packet.cpu().double().numpy()
    offsets = np.cumsum([field.shape[1] for field in fields.values()])[:-1]
    arrays = dict(zip(fields, np.split(host, offsets, axis=1)))
    if sources is not None:
        arrays.update((key, value.double().numpy()) for key, value in zip(('target', 'size', 'target_size'), sources))
        if reuse_q0:
            arrays['quality_size'] = sources[1].clamp_min(1e-3).double().numpy()
    # 此后的 CPU 多边形裁剪可与设备侧前景计数重叠；两者只在构造 trusted 时汇合。
    support = final_candidate_support(output.observation, batch, output.candidates.boxes)
    labels = batch['segmentation_labels'][:, -1].detach()
    count = support['count']
    purity = (support['mask'] & (labels[:, None] > .5)).sum(-1).to(boxes)/count.clamp_min(1)
    real = output.candidates.measurement_candidate.detach() & output.candidates.raw_valid.detach()
    bxs = arrays['boxes'].reshape(boxes.shape)
    gts, szs, gszs = arrays['target'], arrays['size'], arrays['target_size']
    if reference_geometry is None:
        reference_geometry = getattr(output, 'reference_geometry', True)
    geometry, q0_labels = training_box_metrics(bxs, gts, szs, gszs, q0=arrays.get('q0'),
        quality_sizes=arrays.get('quality_size'), explicit_target_size='target_box_size' in batch,
        reference=reference_geometry)
    exact_values, compat_values, distances = geometry.transpose(2, 0, 1)
    metrics = np.stack(metric_contributions(compat_values, distances), axis=-1).astype(np.float32)
    # 一个连续回包保留 exact 的双精度和旧 S/P、quality 的 float32 舍入。
    fields = [exact_values, metrics]
    if reuse_q0:
        fields.append(q0_labels)
    payload = np.concatenate([field.ravel().view(np.uint8) for field in fields])
    packet = torch.from_numpy(payload).to(boxes.device, non_blocking=True)
    exact = packet[:exact_values.nbytes].view(torch.float64).reshape(exact_values.shape)
    scores = packet[exact_values.nbytes:].view(torch.float32).to(boxes.dtype)
    metrics = scores[:metrics.size].reshape(metrics.shape)
    quality = scores[metrics.numel():].reshape(-1, 1) if reuse_q0 else None
    trusted = (exact >= .5) & (count >= 3) & (purity >= .5)
    targets = dict(exact_iou=exact, success=metrics[..., 0], precision=metrics[..., 1],
        utility=metrics.mean(-1), trusted=(trusted & real).to(boxes),
        support_count=count, support_purity=purity, trusted_valid=real)
    return targets, quality


def supervision_targets(batch, output):
    return _supervision_targets(batch, output)[0]


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


def compute_recovery_losses(batch, output, *, recovery_weight=.1, targets=None):
    """辅助流只调用此函数；不计算主路/B1/M损失，不反传其参数。"""
    if output.verification is None or output.recovery is None:
        raise ValueError('v41 recovery losses require verification and recovery output')
    if not np.isfinite(recovery_weight) or recovery_weight < 0:
        raise ValueError('v41 recovery loss weight must be finite and nonnegative')
    losses = {}
    target = supervision_targets(batch, output) if targets is None else targets
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


def compute_losses(batch, output, *, enable_b1=True, recovery_weight=.1, q0_center_weighting='uniform'):
    targets, q0_exact_iou = _supervision_targets(batch, output, include_q0=True)
    losses = base_losses(batch, output, enable_b1=enable_b1, q0_exact_iou=q0_exact_iou,
                         q0_center_weighting=q0_center_weighting)
    original = losses['loss_total']
    recovery = compute_recovery_losses(batch, output, recovery_weight=recovery_weight, targets=targets)
    losses.update(recovery)
    losses['loss_total'] = original + recovery['loss_total']
    return losses
