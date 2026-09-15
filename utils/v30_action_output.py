"""将唯一实际动作同步到所有标量兼容接口，避免策略切换后诊断错位。"""
import torch


def apply_selection_output(output, observation, selection):
    chosen = selection['chosen_action_index']
    index = torch.where(selection['applied'], chosen, selection['best_action_index']).clamp_min(0)
    rows = torch.arange(len(observation), device=observation.device)
    pick = lambda value: value[rows, index]
    score = pick(output['ct_b3_action_scores'])
    applied = selection['applied'].to(observation)
    output.update(ct_final_box=selection['final_box'], ct_router_soft_box=selection['final_box'],
        ct_b3_chosen_action_index=chosen, ct_b3_best_action_index=selection['best_action_index'],
        ct_b3_diagnostic_action_index=index, ct_b3_max_action_score=selection['max_action_score'],
        ct_b3_action_score=score, ct_b3_h3_residual=score, ct_b3_h3_utility=score,
        ct_router_logit=score, ct_router_gate=score, ct_router_applied_gate=applied,
        ct_b3_final_gate=applied, ct_router_evidence_valid=output['ct_b3_action_valid'].any(1).to(observation),
        ct_router_bounded_residual_xy=pick(output['ct_b3_action_residual_xy']))
    fields = {
        'ct_b3_help_logit': 'ct_b3_action_help_logit', 'ct_b3_harm_logit': 'ct_b3_action_harm_logit',
        'ct_b3_expected_success_gain': 'ct_b3_action_success_gain',
        'ct_b3_expected_precision_gain': 'ct_b3_action_precision_gain',
        'ct_b3_bounded_action_features': 'ct_b3_all_action_features',
        'ct_b3_mode_summary': 'ct_b3_all_mode_summary',
        'ct_router_residual_xy': 'ct_b3_action_raw_residual_xy',
    }
    for target, source in fields.items():
        if source in output:
            output[target] = pick(output[source])
    if 'ct_b3_action_raw_norm' in output:
        output['ct_router_clip_rate'] = (pick(output['ct_b3_action_raw_norm']) > output['ct_router_radius']).to(observation)
    if 'ct_b3_help_logit' in output:
        output['ct_b3_help_probability'] = output['ct_b3_help_logit'].sigmoid()
        output['ct_b3_harm_probability'] = output['ct_b3_harm_logit'].sigmoid()
    if 'ct_b3_expected_success_gain' in output:
        output['ct_b3_expected_iou_gain'] = output['ct_b3_expected_success_gain']
        output['ct_b3_expected_center_gain'] = output['ct_b3_expected_precision_gain']
    return output
