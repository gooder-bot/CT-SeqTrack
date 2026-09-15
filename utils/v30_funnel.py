"""v30 全量机制事务漏斗计数；写日志可降频，分子/条件分母不抽样。"""
import torch


@torch.no_grad()
def accumulate_funnel(host, data, output, quality, utility):
    modes = output['ct_evidence_modes']
    device, dtype = modes.centers_xy.device, modes.centers_xy.dtype
    batch = len(modes.valid)
    def value(key):
        item = data.get(key)
        return torch.zeros(batch, device=device, dtype=dtype) if item is None else item.detach().to(device=device, dtype=dtype).reshape(batch)
    indices = output['ct_extension_selected_indices'].detach()
    valid = output['ct_extension_selected_valid_mask'].detach().to(dtype)
    labels = data['ct_extension_labels'].detach().to(device).gather(1, indices)
    selected_target = (labels * valid).sum(1)
    mode_fg = (modes.member_mask.to(dtype) * labels[:, None]).sum(2)
    distance = torch.linalg.norm(modes.centers_xy - data['box_label'].detach()[:, None, :2], dim=-1)
    scale = .5 * torch.linalg.norm(data['bbox_size'].detach()[:, :2], dim=-1)
    correct = modes.valid & (mode_fg > 0) & (distance <= scale[:, None])
    counts = {
        'rows': torch.ones(batch, device=device, dtype=dtype),
        'global_novel_unique_points': value('ct_acquisition_global_novel_point_count'),
        'max_reachable_unique_points': value('ct_acquisition_max_reachable_point_count'),
        'support_novel_unique_points': value('ct_acquisition_support_novel_point_count'),
        'prepool768_unique_points': value('ct_acquisition_prepool_point_count'),
        'selected256_unique_points': valid.sum(1),
        'global_novel_target_points': value('motion_margin_global_novel_target_count'),
        'max_reachable_target_points': value('motion_margin_max_reachable_target_count'),
        'support_novel_target_points': value('ct_acquisition_extension_pool_target_count'),
        'prepool768_target_points': value('ct_acquisition_sampled_target_count'),
        'selected256_target_points': selected_target,
        'valid_modes': modes.valid.sum(1).to(dtype),
        'foreground_modes': (modes.valid & (mode_fg > 0)).sum(1).to(dtype),
        'correct_modes': correct.sum(1).to(dtype),
    }
    stages = ['global_novel', 'max_reachable', 'support_novel', 'prepool768', 'selected256']
    previous = torch.ones(batch, device=device, dtype=torch.bool)
    for stage in stages:
        event = counts[stage + '_target_points'] > 0
        counts[stage + '_target_events'] = event.to(dtype)
        counts[stage + '_conditional_denominator'] = previous.to(dtype)
        counts[stage + '_conditional_retained'] = (previous & event).to(dtype)
        previous = event
    counts['correct_mode_conditional_denominator'] = (selected_target > 0).to(dtype)
    counts['correct_mode_conditional_retained'] = ((selected_target > 0) & correct.any(1)).to(dtype)
    action_valid = output.get('ct_b3_action_valid')
    if action_valid is not None:
        action_valid = action_valid.detach().bool()
        counts['legal_actions'] = action_valid.sum(1).to(dtype)
        counts['legal_action_events'] = action_valid.any(1).to(dtype)
        chosen = output['ct_b3_chosen_action_index'].detach()
        applied = chosen >= 0
        row_ids = torch.arange(batch, device=device)
        counts['selected_actions'] = applied.to(dtype)
        counts['selected_correct_mode'] = (applied & correct[row_ids, chosen.clamp_min(0) // 2]).to(dtype)
        counts['correct_mode_utilization_denominator'] = correct.any(1).to(dtype)
        if 'action_h1_success_gain' in utility:
            gain_s, gain_p = utility['action_h1_success_gain'], utility['action_h1_precision_gain']
            counts['utility_labeled_rows'] = utility['action_valid'].bool().any(1).to(dtype)
            counts['beneficial_action_events'] = (((gain_s + gain_p) > 2e-6) & action_valid).any(1).to(dtype)
            counts['selected_success_gain_sum'] = gain_s[row_ids, chosen.clamp_min(0)] * applied
            counts['selected_precision_gain_sum'] = gain_p[row_ids, chosen.clamp_min(0)] * applied
    raw = data['b0_raw_point_count'].detach()[:, -1].to(device)
    age = value('ct_recursive_state_age')
    speed = torch.linalg.norm(data['velocity_label'].detach().to(device), dim=-1) if 'velocity_label' in data else torch.zeros_like(age)
    groups = {'all': torch.ones_like(age, dtype=torch.bool),
        'sparse_empty': raw == 0, 'sparse_1_2': (raw > 0) & (raw <= 2),
        'sparse_3_10': (raw > 2) & (raw <= 10), 'sparse_dense': raw > 10,
        'age_0_1': (age >= 0) & (age < 2), 'age_2_3': (age >= 2) & (age < 4),
        'age_4_7': (age >= 4) & (age < 8), 'age_8_plus': age >= 8,
        'speed_static': speed <= .3, 'speed_0.3_5': (speed > .3) & (speed <= 5),
        'speed_5_15': (speed > 5) & (speed <= 15), 'speed_15_plus': speed > 15}
    lost = value('ct_recursive_lost_length')
    groups.update(lost_0=lost == 0, lost_1_2=(lost > 0) & (lost <= 2),
                  lost_3_7=(lost > 2) & (lost <= 7), lost_8_plus=lost >= 8)
    store = getattr(host, '_ct_v30_funnel_counts', None)
    if store is None:
        store = host._ct_v30_funnel_counts = {}
    names = list(counts)
    # full 的全轮点计数会超过 float32 的精确整数范围；设备端 double 累计后统一传输。
    matrix = torch.stack([counts[name].detach() for name in names], 1).double()
    for group, mask in groups.items():
        totals = (matrix * mask[:, None]).sum(0)
        key = (group, tuple(names))
        store[key] = store[key] + totals if key in store else totals


def flush_funnel(host):
    store = getattr(host, '_ct_v30_funnel_counts', {})
    summary = {}
    for (group, names), tensor in store.items():
        for name, value in zip(names, tensor.cpu().tolist()):
            key = group + '/' + name
            summary[key] = summary.get(key, 0.) + value
    if summary:
        host.logger.experiment.add_scalars('ct_v30_full_funnel', summary, global_step=host.global_step)
        log_dir = getattr(host.logger, 'log_dir', None)
        if log_dir:
            import json
            from pathlib import Path
            destination = Path(log_dir) / 'ct_v30_funnel'
            destination.mkdir(parents=True, exist_ok=True)
            epoch = int(getattr(host, 'current_epoch', 0)) + 1
            (destination / f'epoch_{epoch:03d}.json').write_text(json.dumps(
                dict(schema='ct_seqtrack.training_funnel.v30', epoch=epoch,
                     global_step=int(host.global_step), counts=summary),
                sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    host._ct_v30_last_funnel = summary
    return summary
