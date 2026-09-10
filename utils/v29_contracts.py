"""v29 适应化 SeqTrack 身份；旧版本不使用这些字段。"""

V29_CONTRACTS = {
    'ct_observation_contract': 'seqtrack_adapted_rollin_v1',
    'ct_b0_sampling_contract': 'real_sparse_repeat_slots_v1',
    'ct_b0_frame_layout_contract': 'batch_frame_channel_point_v1',
    'ct_b0_attention_mask_contract': 'frame_measurement_corner_query_v1',
    'ct_b0_rollin_contract': 'teacher0_short_current_b0_v1',
    'ct_b0_supervision_contract': 'gt_history_input_anchor_v1',
    'ct_b0_coarse_target_contract': 'physical_delta_input_axes_v1',
    'ct_support_z_contract': 'b0_vertical_hull_v1',
    'ct_mechanism_behavior_contract': 'mechanism_behavior_v1',
    'ct_state_transition_contract': 'bounded_structural_policy_v1',
    'ct_b3_target_contract': 'instantaneous_sp_gain_v1',
    'ct_policy_fitting_contract': 'internal_closed_loop_five_v1',
    'ct_b0_point_feature_source': 'seg_second64_v1',
    'ct_b0_loss_reduction': 'reference_batch',
    'ct_b0_rollin_max_steps': 3,
    'ct_training_state_policy': 'mixed_accepted_v1',
}


def validate_v29_contract(config):
    get = config.get if isinstance(config, dict) else lambda k, d=None: getattr(config, k, d)
    if not get('ct_enable_v29', False):
        return
    expected = dict(V29_CONTRACTS, ct_enable_v28=True, ct_enable_v27=True)
    wrong = [key for key, value in expected.items() if get(key) != value]
    if wrong:
        raise ValueError('v29 contract mismatch: ' + ', '.join(wrong))
