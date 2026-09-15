"""v30 同状态六动作标签与模式质量监督；不读取未来，不运行影子递归。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.ct_v2.pipeline_contracts import EvidenceModeSet
from utils.tracking_metrics_v27 import local_boxes_metric_gains


def _get(config, key, default):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def legal_action_row_mean(error, valid):
    """先对每行合法动作平均，再对有合法动作的行平均，避免模式数改变行权重。"""
    valid = valid.detach().bool()
    if error.shape != valid.shape or error.ndim != 2:
        raise ValueError('v30 loss reduction requires aligned [B,A] arrays')
    masked = torch.where(valid, error, torch.zeros_like(error))
    row_valid = valid.any(1)
    row_mean = masked.sum(1) / valid.sum(1).clamp_min(1)
    return row_mean.sum() / row_valid.sum().clamp_min(1)


def compute_mode_quality_loss(data, output, config=None):
    """purity × exp(-d²/(2s_obj²)) 的软 BCE；外层默认 .1 由 host 负责。"""
    modes = EvidenceModeSet.from_output(output)
    logits = modes.quality_logit
    with torch.no_grad():
        selected = output['ct_extension_selected_indices'].detach().long()
        labels = data['ct_extension_labels'].detach().to(logits).gather(1, selected)
        point_valid = data['ct_extension_valid_mask'].detach().bool().to(logits.device).gather(1, selected)
        point_valid &= output['ct_extension_selected_valid_mask'].detach().bool()
        member = modes.member_mask.detach() & point_valid[:, None, :]
        finite_labels = torch.isfinite(labels)
        member &= finite_labels[:, None]
        purity = (member.to(logits) * torch.nan_to_num(labels).clamp(0, 1)[:, None]).sum(2)
        purity /= member.sum(2).clamp_min(1)
        target = data['box_label'].detach().to(logits)
        size = data['bbox_size'].detach().to(logits)
        # wlh 的平面半对角线，只有数值 epsilon，不限制小物体尺寸。
        object_scale = (.5 * torch.linalg.norm(size[:, :2], dim=-1)).clamp_min(1e-6)
        distance2 = (modes.centers_xy.detach() - target[:, None, :2]).square().sum(-1)
        quality_target = purity * torch.exp(-distance2 / (2 * object_scale[:, None].square()))
        valid = (modes.valid.detach() & member.any(2)
                 & torch.isfinite(target).all(1)[:, None]
                 & torch.isfinite(size).all(1)[:, None] & (size > 0).all(1)[:, None]
                 & torch.isfinite(modes.centers_xy.detach()).all(-1))
        if 'b0_view_id' in data:
            valid &= data['b0_view_id'].detach().to(valid.device).reshape(-1, 1) == 0
        quality_target = torch.where(valid, quality_target, torch.zeros_like(quality_target))
    loss = legal_action_row_mean(F.binary_cross_entropy_with_logits(logits, quality_target, reduction='none'), valid)
    return dict(loss=loss, target=quality_target, purity=purity, valid=valid.to(logits))


def compute_b3_mode_utility_loss(data, output, config):
    """返回未乘外层 .2 的 S/P MSE + .1 两 BCE + .1 两几何 SmoothL1。"""
    predicted_s = output['ct_b3_action_success_gain']
    predicted_p = output['ct_b3_action_precision_gain']
    if predicted_s.shape != predicted_p.shape or predicted_s.ndim != 2 or predicted_s.shape[1] != 6:
        raise ValueError('v30 utility requires six action gain predictions')
    batch = len(predicted_s)
    observation = output['observation_aux_estimation_boxes'].detach()[:, :4].clone()
    action = output['ct_b3_action_boxes'].detach()[..., :4].clone()
    target = data['box_label'].detach().to(observation)[:, :4].clone()
    prediction_size = data['bbox_size'].detach().to(observation)
    target_size = data['target_bbox_size'].detach().to(observation)
    if bool(_get(config, 'degrees', False)):
        observation[:, 3] = torch.deg2rad(observation[:, 3])
        action[..., 3] = torch.deg2rad(action[..., 3])
        target[:, 3] = torch.deg2rad(target[:, 3])
    valid = output['ct_b3_action_valid'].detach().bool().clone()
    valid &= torch.isfinite(action).all(-1)
    for tensor in (observation, target, prediction_size, target_size):
        valid &= torch.isfinite(tensor).all(-1)[:, None]
    valid &= ((prediction_size > 0).all(1) & (target_size > 0).all(1))[:, None]
    radius = output['ct_router_radius'].detach().to(observation).reshape(batch)
    valid &= (torch.isfinite(radius) & (radius > 0))[:, None]
    if 'b0_view_id' in data:
        valid &= data['b0_view_id'].detach().to(valid.device).reshape(batch, 1) == 0
    targets = {key: torch.zeros_like(predicted_s) for key in ('success', 'precision', 'distance', 'iou')}
    with torch.no_grad():
        if bool(valid.any()):
            expand = lambda value: value[:, None].expand(-1, 6, -1)[valid]
            gains = local_boxes_metric_gains(expand(observation), action[valid], expand(target),
                expand(prediction_size), expand(target_size), up_axis=_get(config, 'up_axis', (0, 0, 1)),
                mode='benchmark_compat', dim=int(_get(config, 'IoU_space', 3)))
            for name, key in (('success', 'success_gain'), ('precision', 'precision_gain'),
                              ('distance', 'center_gain'), ('iou', 'iou_gain')):
                value = torch.as_tensor(gains[key], device=predicted_s.device, dtype=predicted_s.dtype)
                if name == 'distance':
                    value = (value / radius[:, None].expand(-1, 6)[valid]).clamp(-1, 1)
                targets[name][valid] = value
        utility = .5 * (targets['success'] + targets['precision'])
        helpful, harmful = (utility > 1e-6).to(predicted_s), (utility < -1e-6).to(predicted_s)
    losses = {
        'success': legal_action_row_mean((predicted_s - targets['success']).square(), valid),
        'precision': legal_action_row_mean((predicted_p - targets['precision']).square(), valid),
        'help': legal_action_row_mean(F.binary_cross_entropy_with_logits(
            output['ct_b3_action_help_logit'], helpful, reduction='none'), valid),
        'harm': legal_action_row_mean(F.binary_cross_entropy_with_logits(
            output['ct_b3_action_harm_logit'], harmful, reduction='none'), valid),
        'distance': legal_action_row_mean(F.smooth_l1_loss(
            output['ct_b3_action_distance_gain'], targets['distance'], reduction='none'), valid),
        'iou': legal_action_row_mean(F.smooth_l1_loss(
            output['ct_b3_action_iou_gain'], targets['iou'], reduction='none'), valid),
    }
    aux_weight = float(_get(config, 'ct_b3_geometry_aux_weight', .1))
    total = losses['success'] + losses['precision'] + .1 * (losses['help'] + losses['harm'])
    total = total + aux_weight * (losses['distance'] + losses['iou'])
    result = {'loss': total, **{'loss_' + name: value for name, value in losses.items()},
              'action_help_label': helpful, 'action_harm_label': harmful, 'action_valid': valid.to(predicted_s),
              **{'action_h1_' + name + '_gain': value for name, value in targets.items()}}
    # 与 forward 同一候选的标量日志；完整六动作监督保留独立字段。
    index = output.get('ct_b3_diagnostic_action_index', output['ct_b3_best_action_index']).detach().clamp_min(0)
    rows = torch.arange(batch, device=index.device)
    for name, value in (('help_label', helpful), ('harm_label', harmful), ('valid', valid.to(predicted_s)),
                        ('h1_success_gain', targets['success']), ('h1_precision_gain', targets['precision'])):
        result[name] = value[rows, index]
    return result


# Host 接口使用简短名称；完整函数名保留便于独立诊断调用。
mode_quality_loss = compute_mode_quality_loss
