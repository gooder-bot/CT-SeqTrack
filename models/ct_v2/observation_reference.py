"""v28 原 SeqTrack 整批观测目标；与插件事务分别构造，禁止原地累加别名。"""

import torch
import torch.nn.functional as F


def seqtrack_reference_loss(data, output, config, *, use_motion_cls=True,
                            box_aware=True):
    """保留原 batch 分母及采样槽语义，BC 恰好参与一次反传。"""
    box = data['box_label']
    motion = data['motion_label'][:, 0]
    moving = data['motion_state_label'][:, 0]
    refs = data['box_label_prev']
    estimate = output['estimation_boxes']
    observed = output.get('observation_aux_estimation_boxes',
                          output['aux_estimation_boxes'])
    predicted_motion = output['motion_pred']
    updated_refs = output['updated_ref_boxs']
    logits = output['seg_logits']
    losses = {
        'loss_seg': F.cross_entropy(logits, data['seg_label'],
                                    weight=logits.new_tensor([.5, 2.])),
        'loss_center': F.smooth_l1_loss(estimate[:, :3], box[:, :3]),
        'loss_angle': F.smooth_l1_loss(estimate[:, 3].sin(), box[:, 3].sin()),
        'loss_center_aux': F.smooth_l1_loss(observed[:, :3], box[:, :3]),
        'loss_angle_aux': F.smooth_l1_loss(observed[:, 3].sin(), box[:, 3].sin()),
        'loss_center_ref': F.smooth_l1_loss(updated_refs[..., :3], refs[..., :3]),
        'loss_angle_ref': F.smooth_l1_loss(updated_refs[..., 3].sin(), refs[..., 3].sin()),
    }
    if use_motion_cls:
        losses['loss_motion_cls'] = F.cross_entropy(output['motion_cls'], moving)
        center = F.smooth_l1_loss(predicted_motion[:, :3], motion[:, :3], reduction='none')
        angle = F.smooth_l1_loss(predicted_motion[:, 3].sin(), motion[:, 3].sin(), reduction='none')
        losses['loss_center_motion'] = (moving * center.mean(1)).sum() / (moving.sum() + 1e-6)
        losses['loss_angle_motion'] = (moving * angle).sum() / (moving.sum() + 1e-6)
        total = losses['loss_motion_cls'] * config.motion_cls_seg_weight
    else:
        losses['loss_center_motion'] = F.smooth_l1_loss(predicted_motion[:, :3], motion[:, :3])
        losses['loss_angle_motion'] = F.smooth_l1_loss(predicted_motion[:, 3].sin(), motion[:, 3].sin())
        total = 0.0
    # 运算分组与参考 compute_loss 一致，避免无意改变浮点归约次序。
    total = total + (losses['loss_center'] * config.center_weight
                     + losses['loss_angle'] * config.angle_weight)
    total = total + (losses['loss_seg'] * config.seg_weight
        + (losses['loss_center_aux'] * config.center_weight + losses['loss_angle_aux'] * config.angle_weight)
        + (losses['loss_center_motion'] * config.center_weight + losses['loss_angle_motion'] * config.angle_weight)
        + (losses['loss_center_ref'] * config.ref_center_weight + losses['loss_angle_ref'] * config.ref_angle_weight))
    if box_aware:
        target = torch.cat((data['prev_bc'].flatten(1, 2), data['this_bc']), dim=1)
        losses['loss_bc'] = F.smooth_l1_loss(output['pred_bc'], target)
        total = total + config.bc_weight * losses['loss_bc']
    losses['loss_total'] = total
    return losses
