"""v39 G 的设备端监督；所有 GT 只在本文件进入 loss，不进入候选前向。"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .tensor_geometry import inside_boxes, local_points, pair_box_metrics


DEFAULT_WEIGHTS = dict(success=.125, precision=.125, trusted=.5, ranking=.1,
                       matching=.5, correction=.5, targetness=.2,
                       region_nomatch=.2, quality=.5)


def masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    """先每个 endpoint 内平均，再平均有效 endpoints；点/候选数量不改样本权重。"""
    valid = valid.expand_as(value)
    if value.ndim <= 1:
        return torch.where(valid, value, 0.).sum() / valid.sum().clamp_min(1)
    axes = tuple(range(1, value.ndim))
    count = valid.sum(axes)
    endpoint = torch.where(valid, value, 0.).sum(axes) / count.clamp_min(1)
    return endpoint.sum() / (count > 0).sum().clamp_min(1)


def _bce(logits, target, valid):
    return masked_mean(F.binary_cross_entropy_with_logits(logits.float(), target.detach().float(), reduction='none'), valid)


def _balanced_values(value, positive, valid):
    """matched/unmatched（或 FG/BG）分别平均，避免粗池背景数量改变 loss 比例。"""
    pos, neg = positive & valid, ~positive & valid
    axes = tuple(range(1, value.ndim))
    pos_n, neg_n = pos.sum(axes), neg.sum(axes)
    positive_mean = torch.where(pos, value, 0.).sum(axes) / pos_n.clamp_min(1)
    negative_mean = torch.where(neg, value, 0.).sum(axes) / neg_n.clamp_min(1)
    classes = (pos_n > 0).float() + (neg_n > 0).float()
    endpoint = (positive_mean + negative_mean) / classes.clamp_min(1.)
    return endpoint.sum() / (classes > 0).sum().clamp_min(1)


@torch.no_grad()
def supervision_targets(pool, templates, boxes, size, verification, targets):
    gt = targets['target_box'].detach().float()
    true_size = targets.get('target_size', targets.get('target_box_size', size)).detach().float()
    metrics = pair_box_metrics(boxes, gt, size, true_size)
    ep, et = verification['encoded_pool'], verification['encoded_templates']
    fg = targets.get('current_target_mask')
    if fg is None:
        fg = inside_boxes(ep['xyz'], gt[:, None], true_size)[:, 0]
    fg = fg.detach().bool() & ep['valid']
    support = verification['support_mask'].detach()
    count = support.sum(-1)
    purity = (support & fg[:, None]).sum(-1).float() / count.clamp_min(1)
    trusted = (metrics['exact_iou'] >= .5) & (count >= 3) & (purity >= .5)
    # FP32 几何边界登记为灰区；不 CPU 补算每个候选。
    trusted_valid = verification['measurement_eligible'] & ((metrics['exact_iou'] - .5).abs() > .001)
    scale = (.5 * size.detach().float()[:, :2].norm(dim=-1)).clamp_min(1e-3)
    # 与旧 B0 quality 完全同义（原调用 purity=1）；支持纯度只进入独立 T。
    quality = (.5 + .5 * metrics['exact_iou']) / (1. + metrics['distance'] / scale[:, None])
    # 与实际候选中心 G crop 的范围完全一致；目标在物体框外但仍在
    # 可读取 crop 中时不能错误监督成 absent。只在当前真实证据池计数。
    region = inside_boxes(ep['xyz'], boxes.detach(), size.detach(),
                          scale=targets.get('region_scale', 1.25),
                          offset=targets.get('region_offset', 2.))
    region_count = (region & fg[:, None]).sum(-1)
    metrics.update(trusted=trusted.float(), trusted_valid=trusted_valid, support_purity=purity,
                   quality=quality, region_nomatch=(region_count < 3).float(), foreground=fg)
    return metrics


@torch.no_grad()
def correspondence_targets(pool, templates, size, verification, targets):
    ep, et = verification['encoded_pool'], verification['encoded_templates']
    gt = targets['target_box'].detach().float()
    current = local_points(ep['xyz'], gt[:, None])[:, 0]
    if 'template_xyz_gt' not in targets or 'template_target_mask' not in targets:
        raise ValueError('G correspondence loss requires loss-only template_xyz_gt and template_target_mask')
    target_xyz = targets['template_xyz_gt'].detach().float()
    target_mask = targets['template_target_mask'].detach().bool() & et['valid']
    foreground = targets.get('current_target_mask')
    if foreground is None:
        true_size = targets.get('target_size', targets.get('target_box_size', size))
        foreground = inside_boxes(ep['xyz'], gt[:, None], true_size)[:, 0]
    foreground = foreground.detach().bool() & ep['valid']
    b, t, m, _ = target_xyz.shape
    n = current.shape[1]
    radius = (.1 * size.detach().float().norm(dim=-1)).clamp(.15, .5)
    if n == 0 or m == 0:
        return dict(row=torch.full((b, t, m), n, device=size.device, dtype=torch.long),
                    column=torch.full((b, t, n), m, device=size.device, dtype=torch.long),
                    row_valid=et['valid'], column_valid=ep['valid'][:, None].expand(b, t, n),
                    matched_current=torch.zeros((b, n), device=size.device, dtype=torch.bool))
    distance = torch.cdist(target_xyz.reshape(b, t * m, 3), current).reshape(b, t, m, n)
    possible = target_mask[..., None] & foreground[:, None, None]
    distance = distance.masked_fill(~possible, torch.inf)
    row_distance, row_index = distance.min(-1)
    column_distance, column_index = distance.min(-2)
    row = torch.where(row_distance <= radius[:, None, None], row_index, n)
    column = torch.where(column_distance <= radius[:, None, None], column_index, m)
    return dict(row=row, column=column, row_valid=et['valid'],
                column_valid=ep['valid'][:, None].expand(b, t, n),
                matched_current=(column < m).any(1) & foreground)


def _matching_loss(verification, target, valid):
    matching = verification['matching']
    row_log, column_log = matching['row_log_probabilities'], matching['column_log_probabilities']
    c = row_log.shape[1]
    row_target = target['row'][:, None].expand(-1, c, -1, -1)
    column_target = target['column'][:, None].expand(-1, c, -1, -1)
    row_value = -row_log.gather(-1, row_target[..., None]).squeeze(-1)
    column_value = -column_log.gather(-2, column_target[..., None, :]).squeeze(-2)
    row_valid = target['row_valid'][:, None] & valid[..., None, None]
    column_valid = target['column_valid'][:, None] & valid[..., None, None]
    rows = _balanced_values(row_value, row_target < row_log.shape[-1] - 1, row_valid)
    columns = _balanced_values(column_value, column_target < column_log.shape[-2] - 1, column_valid)
    return .5 * (rows + columns)


def _correction_values(refinement, size, targets, corresponding):
    soft = refinement['soft_boxes'].float()
    original = refinement['input_boxes'].detach().float()
    truth = targets['target_box'].detach().float()[:, None]
    size = size.detach().float().clamp_min(1e-3)
    observable = corresponding['matched_current'].sum(-1) >= 3
    valid = observable[:, None].expand(soft.shape[:2])
    if 'supervision_valid' in refinement:
        valid = valid & refinement['supervision_valid'].detach().bool().expand_as(valid)
    if not refinement.get('reacquire', False):
        difference = truth[..., :3] - original[..., :3]
        limit_xy = (.5 * size[:, :2].norm(dim=-1)).clamp_max(2.)[:, None]
        limit_z = (.25 * size[:, 2]).clamp_max(.5)[:, None]
        valid = valid & (difference[..., :2].norm(dim=-1) <= limit_xy)
        valid &= difference[..., 2].abs() <= limit_z
    normalized_error = (soft[..., :3] - truth[..., :3]) / size[:, None]
    xyz = F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error), beta=.1, reduction='none').mean(-1)
    angle = 1. - (soft[..., 3] - truth[..., 3]).cos()
    yaw_valid = refinement['yaw_valid'].detach() & valid
    if not refinement.get('reacquire', False):
        original_angle = torch.atan2((truth[..., 3] - original[..., 3]).sin(),
                                     (truth[..., 3] - original[..., 3]).cos())
        yaw_valid &= original_angle.abs() <= math.pi / 6.
    # 暂不归约：调用方先合并同 endpoint 的所有纠偏来源，避免一个其他
    # endpoint 触发空恢复分支时，改变本 endpoint 已有的监督权重。
    return xyz, valid, angle, yaw_valid


def compute_g_losses(pool, templates, boxes, size, verification, targets,
                     refinement=None, *, weights: Mapping[str, float] | None = None):
    """返回 loss_g 和各原始 loss_g_*；默认 S/P=.125，不能写回旧版 1/1。"""
    configuration = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(configuration)
        if unknown:
            raise ValueError(f'unknown G loss weights: {sorted(unknown)}')
        configuration.update(weights)
    supervision = supervision_targets(pool, templates, boxes, size, verification, targets)
    correspondence = correspondence_targets(pool, templates, size, verification, targets)
    valid = verification['candidate_valid'].detach().bool()
    logits = verification['logits']
    terms = dict(success=_bce(logits[..., 0], supervision['success'], valid),
                 precision=_bce(logits[..., 1], supervision['precision'], valid),
                 trusted=_bce(logits[..., 2], supervision['trusted'], supervision['trusted_valid']),
                 region_nomatch=_bce(verification['region_no_match_logits'], supervision['region_nomatch'], valid))
    quality_valid = targets.get('quality_candidate_mask', verification['measurement_eligible']).detach().bool() & valid
    terms['quality'] = _bce(verification['q_logits'], supervision['quality'], quality_valid)
    predicted = verification['utility']
    truth_utility = .5 * (supervision['success'] + supervision['precision'])
    left, right = torch.triu_indices(predicted.shape[1], predicted.shape[1], 1, device=predicted.device)
    ranking_error = ((predicted[:, left] - predicted[:, right])
                     - (truth_utility[:, left] - truth_utility[:, right])).square()
    terms['ranking'] = masked_mean(ranking_error, valid[:, left] & valid[:, right])
    point_logits = verification['targetness_logits'].float()
    point_target = supervision['foreground'][:, None].expand_as(point_logits)
    point_valid = verification['encoded_pool']['valid'][:, None] & valid[..., None]
    point_loss = F.binary_cross_entropy_with_logits(point_logits, point_target.float(), reduction='none')
    terms['targetness'] = _balanced_values(point_loss, point_target, point_valid)
    terms['matching'] = _matching_loss(verification, correspondence, valid)
    if refinement is None:
        terms['correction'] = logits.sum() * 0.
    else:
        refinements = refinement if isinstance(refinement, (tuple, list)) else (refinement,)
        values = []
        for item in refinements:
            # 每个 refinement 的可观测性只来自自己的点池；teacher 新 crop 不得
            # 把 prefix 空观测追认为可纠偏。GT 仅生成这一路的 loss mask。
            local_targets = dict(targets)
            target_size = targets.get('target_size', targets.get('target_box_size', size))
            local_targets['current_target_mask'] = inside_boxes(
                item['encoded_pool']['xyz'], targets['target_box'].detach()[:, None], target_size.detach())[:, 0]
            local_correspondence = correspondence_targets(pool, templates, size,
                dict(encoded_pool=item['encoded_pool'], encoded_templates=item['encoded_templates']), local_targets)
            values.append(_correction_values(item, size, targets, local_correspondence))
        if values:
            xyz, xyz_valid, angle, yaw_valid = (torch.cat(parts, dim=1) for parts in zip(*values))
            terms['correction'] = masked_mean(xyz, xyz_valid) + .25 * masked_mean(angle, yaw_valid)
        else:
            terms['correction'] = logits.sum() * 0.
    result = {f'loss_g_{key}': value for key, value in terms.items()}
    result['loss_g'] = sum(configuration[key] * value for key, value in terms.items())
    # 便于调用方记录被动诊断；不参与任何在线选择。
    verification['supervision_targets'] = supervision
    return result
