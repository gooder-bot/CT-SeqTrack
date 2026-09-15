"""同一端点的获取→三模式→六动作诊断，GT 仅在前向完成后进入。"""
import numpy as np
import torch


def add_v30_endpoint(row, batch, output, *, target, previous_target=None,
                     physical_delta_t=None, lost_length=0, config=None):
    row.update(protocol_version='v30', max_action_q=0.,
        selected_action_index=-1, mode_count=0, action_count=0,
        mode_target_event=False, correct_mode_event=False, selected_correct_mode=False,
        recursive_age=max(int(row['frame_id']) - 1, 0), lost_length=int(lost_length))
    if config is not None:
        from utils.v30_reporting import endpoint_identity
        row.update(endpoint_identity(config))
    raw_count = row.get('b0_raw_point_count')
    row['sparsity_group'] = ('unknown' if raw_count is None else
        'empty' if raw_count == 0 else '1_2' if raw_count <= 2 else '3_10' if raw_count <= 10 else 'dense')
    age = row['recursive_age']
    row['recursive_age_group'] = '0_1' if age < 2 else '2_3' if age < 4 else '4_7' if age < 8 else '8_plus'
    if previous_target is not None and physical_delta_t is not None and physical_delta_t > 0:
        speed = float(np.linalg.norm(target.center - previous_target.center) / physical_delta_t)
        row['physical_speed_mps'] = speed
        row['speed_group'] = 'static' if speed <= .3 else '0.3_5' if speed <= 5 else '5_15' if speed <= 15 else '15_plus'
    else:
        row['physical_speed_mps'], row['speed_group'] = None, 'initial'
    if 'ct_evidence_modes' not in output:
        return
    modes = output['ct_evidence_modes'].detached()
    labels = batch['ct_extension_labels'].gather(1, output['ct_extension_selected_indices']).detach()
    members = modes.member_mask
    target_count = (members.to(labels) * labels[:, None]).sum(-1)
    center_distance = torch.linalg.norm(modes.centers_xy - batch['box_label'][:, None, :2], dim=-1)
    scale = .5 * torch.linalg.norm(batch['bbox_size'][:, :2], dim=-1)
    correct = modes.valid & (target_count > 0) & (center_distance <= scale[:, None])
    row.update(mode_count=int(modes.valid[0].sum().cpu()),
        mode_target_event=bool((modes.valid[0] & (target_count[0] > 0)).any().cpu()),
        correct_mode_event=bool(correct[0].any().cpu()),
        mode_correct_definition='unique_fg>0_and_center_distance<=initial_half_diagonal')
    for name, value in (('valid', modes.valid), ('unique_points', modes.unique_count),
        ('unique_target_points', target_count), ('targetness_mass', modes.mass),
        ('quality', modes.quality), ('center_distance', center_distance), ('correct', correct)):
        row['mode_' + name] = value[0].cpu().tolist()
    if 'ct_b3_action_scores' not in output:
        return
    valid = output['ct_b3_action_valid'][0].detach().bool()
    scores = output['ct_b3_action_scores'][0].detach()
    action = int(output['ct_b3_chosen_action_index'][0].detach().cpu())
    row.update(max_action_q=float(output['ct_b3_max_action_score'][0].detach().cpu()),
        action_count=int(valid.sum().cpu()), selected_action_index=action,
        action_scores=scores.cpu().tolist(), action_valid=valid.cpu().tolist(),
        selected_correct_mode=bool(correct[0, action // 2].cpu()) if action >= 0 else False,
        action_local_boxes=output['ct_b3_action_boxes'][0].detach().cpu().tolist())
    # 全部动作的即时标签与网络训练调用同一度量；不以最终执行选择过滤监督。
    if 'ct_b3_action_success_gain' in output:
        from utils.v30_training import compute_b3_mode_utility_loss
        if config is not None:
            with torch.no_grad():
                utility = compute_b3_mode_utility_loss(batch, output, config)
            for name in ('success', 'precision', 'distance', 'iou'):
                row['action_label_' + name] = utility['action_h1_' + name + '_gain'][0].cpu().tolist()
            gain = .5 * (utility['action_h1_success_gain'][0] + utility['action_h1_precision_gain'][0])
            row['beneficial_action_available'] = bool(((gain > 1e-6) & valid).any().cpu())
            row['selected_action_immediate_gain'] = float(gain[action].cpu()) if action >= 0 else 0.
