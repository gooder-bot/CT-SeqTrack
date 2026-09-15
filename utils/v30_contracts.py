"""v30 方法/数据/执行身份；旧版本从不进入此分派。"""

import math


V30_CONTRACTS = {
    'ct_observation_contract': 'seqtrack_valid_measurement_rollin_v30',
    'ct_b0_sampling_contract': 'real_sparse_repeat_slots_v1',
    'ct_b0_frame_layout_contract': 'batch_frame_channel_point_v1',
    'ct_b0_attention_mask_contract': 'frame_measurement_corner_query_v1',
    'ct_b0_loss_contract': 'measurement_masked_ce_bc_history_ref_v1',
    'ct_b0_rollin_contract': 'teacher0_mixed_short_long_v30',
    'ct_b0_supervision_contract': 'gt_history_input_anchor_v1',
    'ct_b0_coarse_target_contract': 'physical_delta_input_axes_v1',
    'ct_b0_point_feature_source': 'seg_second64_v1',
    'ct_b0_loss_reduction': 'reference_batch',
    'ct_b1_input_contract': 'causal_history_current_crop21_v1',
    'ct_support_xy_contract': 'b0_boundary_endpoint_band_v1',
    'ct_support_z_contract': 'b0_vertical_hull_v1',
    'ct_acquisition_target_contract': 'reachable_novel_q90_min_background_v30',
    'ct_acquisition_loss_contract': 'demand_age_balanced_pinball_v1',
    'ct_b2_mode_contract': 'weighted_support_k3_1m_nms075_v30',
    'ct_b2_geometry_contract': 'support_halfsize_plus_logsigma7_v30',
    'ct_b2_feature_contract': 'support_normalized_modes64_v30',
    'ct_b2_quality_contract': 'purity_center_gaussian_detached_v30',
    'ct_mechanism_behavior_contract': 'track_hash_mix_2n_1a_1e_4q_v30',
    'ct_state_transition_contract': 'six_action_strict_q_tie_small_motion_v30',
    'ct_b3_target_contract': 'same_state_six_action_h1_aux_geometry_v30',
    'ct_b3_action_contract': 'three_modes_half_full_xy_v1',
    'ct_policy_fitting_contract': 'row_max_q_closed_loop_ten_v1',
    'ct_dataset_protocol': 'ct_seqtrack.dataset_protocol.v30',
    'ct_training_state_policy': 'mixed_accepted_v30',
}

V30_IDENTITY_FIELDS = (
    'ct_enable_v30', *V30_CONTRACTS,
    'ct_v30_ablation', 'ct_b0_long_rollin_enabled', 'ct_b0_rollin_max_steps',
    'ct_b0_masked_bn_recompute',
    'ct_v30_legacy_acquisition', 'ct_acquisition_need_balance',
    'ct_acquisition_margin_min', 'ct_acquisition_margin_max', 'ct_acquisition_margin_initial',
    'ct_mode_count', 'ct_mode_quality_weight', 'ct_b3_geometry_aux_weight',
    'ct_coordinate_mode', 'kitti_frame_period', 'ct_frame_stride',
    'ct_pointcloud_cache_bytes', 'ct_moving_speed_threshold',
    'ct_dataset_manifest_sha256', 'ct_scene_manifest_sha256',
    'ct_dataloader_prefetch_factor', 'ct_dataloader_persistent_workers',
)


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def validate_v30_contract(config):
    if not bool(_get(config, 'ct_enable_v30', False)):
        return
    expected = dict(V30_CONTRACTS, ct_enable_v29=True, ct_enable_v28=True, ct_enable_v27=True)
    legacy = bool(_get(config, 'ct_v30_legacy_acquisition', False))
    if legacy:
        expected.update(ct_support_xy_contract='v29_object_margin_xy_v1',
                        ct_acquisition_target_contract='reachable_novel_q90_min_background_v29')
    if not bool(_get(config, 'ct_acquisition_need_balance', True)):
        expected['ct_acquisition_loss_contract'] = 'age_balanced_pinball_v1'
    if not bool(_get(config, 'ct_b0_long_rollin_enabled', True)):
        expected['ct_b0_rollin_contract'] = 'teacher0_short_current_b0_v1'
    wrong = [key for key, value in expected.items() if _get(config, key) != value]
    if wrong:
        raise ValueError('v30 contract mismatch: ' + ', '.join(wrong))
    for name in ('ct_b0_long_rollin_enabled', 'ct_v30_legacy_acquisition', 'ct_acquisition_need_balance'):
        if type(_get(config, name)) is not bool:
            raise ValueError(f'{name} must be explicitly boolean')
    if type(_get(config, 'ct_b0_masked_bn_recompute', False)) is not bool:
        raise ValueError('ct_b0_masked_bn_recompute must be boolean')
    count = _get(config, 'ct_mode_count')
    if type(count) is not int or count not in (1, 3):
        raise ValueError('v30 mode count must be 1 or 3 within the fixed three-slot schema')
    weight = _get(config, 'ct_mode_quality_weight')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight not in (0., .1):
        raise ValueError('v30 mode quality weight must be 0 or 0.1')
    bands = []
    for name in ('ct_acquisition_margin_min', 'ct_acquisition_margin_max', 'ct_acquisition_margin_initial'):
        value = _get(config, name)
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or x <= 0 for x in value)):
            raise ValueError(f'{name} must contain two positive finite metres')
        bands.append(tuple(value))
    lower, upper, initial = bands
    if not all(a < c < b for a, b, c in zip(lower, upper, initial)):
        raise ValueError('v30 initial band must lie strictly inside its bounds')
    allowed = ((2., 1.), ((6., 3.),)) if legacy else ((.25, .25), ((4., 3.), (2., 1.5)))
    if lower != allowed[0] or upper not in allowed[1]:
        raise ValueError('v30 acquisition bounds differ from registered full/tight/legacy geometry')
    dataset = _get(config, 'dataset')
    if dataset == 'nuscenes_mf':
        if _get(config, 'version') not in ('v1.0-mini', 'v1.0-trainval') or _get(config, 'ct_coordinate_mode') != 'global':
            raise ValueError('v30 nuScenes requires mini/trainval and global coordinates')
    elif dataset in ('kitti_mf', 'kitti'):
        if (_get(config, 'version') != 'kitti_tracking'
                or _get(config, 'ct_coordinate_mode') != 'sensor_relative'
                or _get(config, 'kitti_frame_period') != .1):
            raise ValueError('v30 KITTI requires registered sensor_relative coordinates and 0.1s source time')
    else:
        raise ValueError('v30 supports only nuScenes and KITTI')
    from utils.dataset_protocol_v30 import frame_stride
    frame_stride(_get(config, 'ct_frame_stride'))
    cache = _get(config, 'ct_pointcloud_cache_bytes')
    if type(cache) is not int or cache < 0:
        raise ValueError('v30 pointcloud cache bytes must be a nonnegative integer')
    if _get(config, 'preloading', False):
        raise ValueError('v30 uses a bounded raw-frame cache; preloading must be false')
    v30_dataloader_kwargs(config)
    from utils.v29_performance import validate_performance_contract
    validate_performance_contract(config)


def validate_v30_scratch_contract(config):
    """完整新方法合同；不把旧 mini1262 常数当成新数据选择的替代物。"""
    validate_v30_contract(config)
    expected = {
        'net_model': 'ctseqtrack', 'ct_protocol_status': 'formal',
        'ct_initialization_policy': 'scratch_only', 'ct_b0_initialization_policy': 'scratch_only',
        'ct_online_recursive_training': True, 'ct_formal_resume_contract': True,
        'ct_training_topology': 'dual_stream', 'ct_runtime_protocol': 'safe_seqtrack_auto_v1',
        'ct_optimizer_topology': 'unified_auto', 'ct_separate_optimizers': False,
        'ct_module_isolation': 'strict', 'ct_b0_training_protocol': 'safe_seqtrack_auto_v1',
        'ct_batch_schema': 'ct_seqtrack.train.v4', 'ct_candidate_policy': 'b2_raw',
        'ct_observation_rng_mode': 'stateless_seqtrack', 'ct_validation_rng_mode': 'stateless_tracklet_frame',
        'ct_mechanism_shadow_b0_no_grad': True, 'ct_mechanism_stream': 'online_recursive',
        'ct_mechanism_passes_per_epoch': 1, 'ct_mechanism_b0_view': 'canonical_only',
        'ct_recursive_tracklet_slots': 16, 'ct_recursive_candidate_views': 4,
        'ct_b0_candidate_views': 4, 'ct_b2_candidate_views': 1, 'num_candidates': 4,
        'ct_b0_candidate_mode': 'independent', 'candidate_trajectory_mode': 'independent',
        'ct_recursive_reseed_enabled': False, 'ct_b0_rng_shift_control': False,
        'ct_b0_steps_per_epoch': 0, 'ct_v28_expected_mini_car_updates_per_epoch': None,
        'ct_recovery_candidate_policy': 'off', 'search_v3_use_dynamic_sigma': False,
        'ct_expansion_point_count': 768, 'ct_relation_topk': 128,
        'ct_relation_coverage_count': 96, 'ct_relation_exploration_count': 32,
        'ct_raw_search_weight': 0., 'ct_presence_hard_gate': False,
        'epoch': 60, 'from_epoch': 0, 'seed': 42, 'batch_size': 16, 'workers': 4,
        'trainer_devices': 1, 'category_name': 'Car', 'precision': 32,
        'optimizer': 'Adam', 'lr': .0001, 'ct_b0_lr': .0001, 'ct_b1_lr': .0001,
        'ct_b2_lr': .0001, 'ct_b3_lr': .0001, 'ct_plugin_lr': .0001,
        'motion_v3_warmup_epoch': 0, 'ct_manual_amp_enabled': False,
        'b2_v3_freeze_candidate_producers': False, 'v22_freeze_candidate_producers': False,
        'ct_deterministic_algorithms': True, 'ct_deterministic_warn_only': False,
        'ct_allow_tf32': False, 'ct_cudnn_benchmark': False, 'ct_cudnn_deterministic': True,
        'ct_adam_foreach': False, 'ct_adam_fused': False, 'ct_cublas_workspace_config': ':4096:8',
        'ct_keep_final_window_checkpoints': 3, 'save_top_k': 0, 'check_val_every_n_epoch': 5,
        'ct_b0_ce_contract': 'class_axis_logsoftmax_flat_nll_v1', 'observation_safe_bbox_size': True,
    }
    engineering = _get(config, 'ct_engineering_check', False)
    if type(engineering) is not bool:
        raise ValueError('ct_engineering_check must be explicitly boolean')
    if engineering:
        # 真实数据 smoke 仅覆盖训练执行量；方法、预算和初始化合同仍完整验证。
        from pathlib import Path
        for key in ('epoch', 'workers', 'check_val_every_n_epoch'):
            expected.pop(key)
        wrong = []
        root = (Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks').resolve()
        log_dir = _get(config, 'log_dir')
        target = Path(str(log_dir)).resolve() if log_dir else None
        if target is None or root not in target.parents:
            wrong.append('engineering log_dir must be below artifacts/ct_checks')
        epochs = _get(config, 'epoch')
        if type(epochs) is not int or not 1 <= epochs <= 3:
            wrong.append('engineering epoch must be an integer in [1,3]')
        workers = _get(config, 'workers')
        if type(workers) is not int or workers < 0:
            wrong.append('engineering workers must be a nonnegative integer')
        if _get(config, 'check_val_every_n_epoch') != 1:
            wrong.append('engineering check_val_every_n_epoch must be 1')
        for key in ('limit_train_batches', 'limit_val_batches'):
            value = _get(config, key)
            if type(value) is not int or not 1 <= value <= 100:
                wrong.append(f'engineering {key} must be an integer in [1,100]')
    else:
        wrong = []
        for key in ('limit_train_batches', 'limit_val_batches'):
            if type(_get(config, key, 1.)) is not float or _get(config, key, 1.) != 1.:
                wrong.append(key)
    wrong.extend(key for key, value in expected.items() if _get(config, key) != value)
    if _get(config, 'ct_recursive_rollout_horizons') != [1, 2, 4, 8]:
        wrong.append('ct_recursive_rollout_horizons')
    if _get(config, 'init_checkpoint'):
        wrong.append('init_checkpoint')
    if wrong:
        raise ValueError('invalid v30 scratch contract: ' + ', '.join(sorted(wrong)))


def v30_dataloader_kwargs(config):
    """只增加预取参数；保持每 epoch worker 生命周期及旧 loader 参数原样。"""
    if not bool(_get(config, 'ct_enable_v30', False)):
        return {}
    prefetch = _get(config, 'ct_dataloader_prefetch_factor', 2)
    persistent = _get(config, 'ct_dataloader_persistent_workers', False)
    if type(prefetch) is not int or prefetch <= 0 or type(persistent) is not bool or persistent:
        raise ValueError('v30 loader requires positive integer prefetch and persistent_workers=false')
    if int(_get(config, 'workers', 0)) <= 0:
        return {}
    return dict(prefetch_factor=prefetch, persistent_workers=False)
