"""v28 原 SeqTrack 整批观测目标；与插件事务分别构造，禁止原地累加别名。"""

import torch
import torch.nn.functional as F


def seqtrack_segmentation_cross_entropy(logits, labels):
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
    return F.nll_loss(rows, labels.reshape(-1),
                      weight=logits.new_tensor([.5, 2.]), reduction='mean')


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
        'loss_seg': seqtrack_segmentation_cross_entropy(logits, data['seg_label']),
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
