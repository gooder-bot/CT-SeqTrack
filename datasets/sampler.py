# Created by zenn at 2021/4/27
# Modified by Aron Lin at Jun 4 20:32:36 CST 2023 

import numpy as np
import torch
from easydict import EasyDict
from nuscenes.utils import geometry_utils

import datasets.points_utils as points_utils

from datasets.misc_utils import get_history_frame_ids_and_masks, \
    create_history_frame_dict, \
    generate_virtual_points, \
    build_time_fields, \
    build_main_time_fields, \
    build_effective_time_fields, \
    normalize_dynamics_time_mode
from utils.twc_utils import (
    build_shared_candidate_offset_map,
    build_shared_point_sampling_seed_map,
    candidate_offsets_for_frame_ids,
    point_sampling_seeds_for_frame_ids,
    sample_candidate_offset,
    sample_point_sampling_seed,
)
from utils.candidate_utils import (
    apply_shared_se2_to_boxes,
    boxes_to_anchor_parameters,
    build_ct_training_histories,
    canonical_dynamics_targets,
    equivalent_local_offsets,
    normalize_candidate_trajectory_mode,
    shared_se2_world_translation,
    validate_shared_se2_transform,
)
from utils.ct_search import (
    build_time_guided_search_box,
    stratified_search_sample,
)


def no_processing(data, *args):
    return data


def siamese_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    """
    first_frame = data['first_frame']
    template_frame = data['template_frame']
    search_frame = data['search_frame']
    candidate_id = data['candidate_id']
    first_pc, first_box = first_frame['pc'], first_frame['3d_bbox']
    template_pc, template_box = template_frame['pc'], template_frame['3d_bbox']
    search_pc, search_box = search_frame['pc'], search_frame['3d_bbox']
    if template_transform is not None:
        template_pc, template_box = template_transform(template_pc, template_box)
        first_pc, first_box = template_transform(first_pc, first_box)
    if search_transform is not None:
        search_pc, search_box = search_transform(search_pc, search_box)
    # generating template. Merging the object from previous and the first frames.
    if candidate_id == 0:
        samplegt_offsets = np.zeros(3)
    else:
        samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        samplegt_offsets[2] = samplegt_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    template_box = points_utils.getOffsetBB(template_box, samplegt_offsets, limit_box=config.data_limit_box,
                                            degrees=config.degrees)
    model_pc, model_box = points_utils.getModel([first_pc, template_pc], [first_box, template_box],
                                                scale=config.model_bb_scale, offset=config.model_bb_offset)

    assert model_pc.nbr_points() > 20, 'not enough template points'

    # generating search area. Use the current gt box to select the nearby region as the search area.

    if candidate_id == 0 and config.num_candidates > 1:
        sample_offset = np.zeros(3)
    else:
        # gaussian = KalmanFiltering(bnd=[1, 1, (5 if config.degrees else np.deg2rad(5))])
        # sample_offset = gaussian.sample(1)[0]
        raise NotImplementedError("Previously used pomegranate's KalmanFiltering here, now disabled. Update required.")
    sample_bb = points_utils.getOffsetBB(search_box, sample_offset, limit_box=config.data_limit_box,
                                         degrees=config.degrees)
    search_pc_crop = points_utils.generate_subwindow(search_pc, sample_bb,
                                                     scale=config.search_bb_scale, offset=config.search_bb_offset)
    assert search_pc_crop.nbr_points() > 20, 'not enough search points'
    search_box = points_utils.transform_box(search_box, sample_bb)
    seg_label = points_utils.get_in_box_mask(search_pc_crop, search_box).astype(int)
    search_bbox_reg = [search_box.center[0], search_box.center[1], search_box.center[2], -sample_offset[2]]

    template_points, idx_t = points_utils.regularize_pc(model_pc.points.T, config.template_size)
    search_points, idx_s = points_utils.regularize_pc(search_pc_crop.points.T, config.search_size)
    seg_label = seg_label[idx_s]
    data_dict = {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
    }
    if getattr(config, 'box_aware', False):
        template_bc = points_utils.get_point_to_box_distance(template_points, model_box)
        search_bc = points_utils.get_point_to_box_distance(search_points, search_box)
        data_dict.update({'points2cc_dist_t': template_bc.astype('float32'),
                          'points2cc_dist_s': search_bc.astype('float32'), })
    return data_dict


def motion_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    point_sample_size
    bb_scale
    bb_offset
    """
    prev_frame = data['prev_frame']
    this_frame = data['this_frame']
    candidate_id = data['candidate_id']
    prev_pc, prev_box = prev_frame['pc'], prev_frame['3d_bbox']
    this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']

    num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points[0:3,:]).sum() 
    assert num_points_in_prev_box > config.limit_num_points_in_prev_box, 'not enough target points'

    if template_transform is not None:
        prev_pc, prev_box = template_transform(prev_pc, prev_box)
    if search_transform is not None:
        this_pc, this_box = search_transform(this_pc, this_box)

    if candidate_id == 0:
        sample_offsets = np.zeros(3) 
    else:
        sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        sample_offsets[2] = sample_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    ref_box = points_utils.getOffsetBB(prev_box, sample_offsets, limit_box=config.data_limit_box,
                                       degrees=config.degrees)
    prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_box,
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)

    this_frame_pc = points_utils.generate_subwindow(this_pc, ref_box,
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)
    # assert this_frame_pc.nbr_points() > config.limit_num_this_frame_subwindow_pc, 'not enough search points'

    this_box = points_utils.transform_box(this_box, ref_box) 
    prev_box = points_utils.transform_box(prev_box, ref_box) 
    ref_box = points_utils.transform_box(ref_box, ref_box)   
    motion_box = points_utils.transform_box(this_box, prev_box) 

    prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T, config.point_sample_size) 
    this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T, config.point_sample_size) 

    seg_label_this = geometry_utils.points_in_box(this_box, this_points.T[:3,:], 1.25).astype(int) 
    seg_label_prev = geometry_utils.points_in_box(prev_box, prev_points.T[:3,:], 1.25).astype(int) 
    seg_mask_prev = geometry_utils.points_in_box(ref_box, prev_points.T[:3,:], 1.25).astype(float) 
    if candidate_id != 0:
        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        seg_mask_prev[seg_mask_prev == 0] = 0.2
        seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

    timestamp_prev = np.full((config.point_sample_size, 1), fill_value=0)
    timestamp_this = np.full((config.point_sample_size, 1), fill_value=0.1)

    prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
    this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)


    stack_points = np.concatenate([prev_points, this_points], axis=0)
    stack_seg_label = np.hstack([seg_label_prev, seg_label_this])
    theta_this = this_box.orientation.degrees * this_box.orientation.axis[-1] if config.degrees else \
        this_box.orientation.radians * this_box.orientation.axis[-1]
    box_label = np.append(this_box.center, theta_this).astype('float32')
    theta_prev = prev_box.orientation.degrees * prev_box.orientation.axis[-1] if config.degrees else \
        prev_box.orientation.radians * prev_box.orientation.axis[-1]
    box_label_prev = np.append(prev_box.center, theta_prev).astype('float32')
    theta_motion = motion_box.orientation.degrees * motion_box.orientation.axis[-1] if config.degrees else \
        motion_box.orientation.radians * motion_box.orientation.axis[-1]
    motion_label = np.append(motion_box.center, theta_motion).astype('float32')

    motion_state_label = np.sqrt(np.sum((this_box.center - prev_box.center) ** 2)) > config.motion_threshold

    data_dict = {
        'points': stack_points.astype('float32'),
        'box_label': box_label,
        'box_label_prev': box_label_prev,
        'motion_label': motion_label,
        'motion_state_label': motion_state_label.astype('int'),
        'bbox_size': this_box.wlh,
        'seg_label': stack_seg_label.astype('int'),
    }

    if getattr(config, 'box_aware', False):
        prev_bc = points_utils.get_point_to_box_distance(stack_points[:config.point_sample_size, :3], prev_box)
        this_bc = points_utils.get_point_to_box_distance(stack_points[config.point_sample_size:, :3], this_box)
        candidate_bc_prev = points_utils.get_point_to_box_distance(stack_points[:config.point_sample_size, :3], ref_box)
        candidate_bc_this = np.zeros_like(candidate_bc_prev)
        candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)

        data_dict.update({'prev_bc': prev_bc.astype('float32'),
                          'this_bc': this_bc.astype('float32'),
                          'candidate_bc': candidate_bc.astype('float32')})
    return data_dict

def motion_processing_mf(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    point_sample_size
    bb_scale
    bb_offset
    """
    prev_frames = data['prev_frames']
    this_frame = data['this_frame']
    candidate_id = data['candidate_id']
    valid_mask = data['valid_mask']
    num_hist = len(valid_mask)
    empty_counter = 0

    prev_pcs  = [prev_frames[key]['pc'] for key in sorted(prev_frames,key=lambda k: abs(int(k)))] # Ordered point clouds, -1, -2, -3
    prev_boxs = [prev_frames[key]['3d_bbox'] for key in sorted(prev_frames,key=lambda k: abs(int(k)))] # Ordered point clouds, -1, -2, -3
    this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']
    # Keep the untouched GT trajectory for M1 labels and invariance audits.
    # ``transform_box`` is non-mutating, but the local variables below are
    # intentionally rebound to candidate-coordinate boxes.
    canonical_prev_boxs = list(prev_boxs)
    canonical_this_box = this_box
    sorted_prev_keys = sorted(prev_frames, key=lambda k: abs(int(k)))
    prev_timestamps = [prev_frames[key].get('timestamp') for key in sorted_prev_keys]
    current_timestamp = this_frame.get('timestamp')
    default_time_step = getattr(config, 'default_time_step', getattr(config, 'time_step', 0.1))
    pseudo_time_step = getattr(config, 'pseudo_time_step', 0.1)
    use_real_time = getattr(config, 'use_real_time', True)

    prev_frame_ids = data.get('prev_frame_ids')
    this_frame_id = data.get('this_frame_id')
    history_offsets = data.get('history_offsets')
    candidate_trajectory_mode = normalize_candidate_trajectory_mode(
        getattr(config, 'candidate_trajectory_mode', 'independent'))
    candidate_offsets = data.get('candidate_offsets')
    candidate_shared_transform = data.get('candidate_shared_transform')
    if candidate_trajectory_mode == 'shared_se2':
        if candidate_offsets is not None:
            raise ValueError(
                "shared_se2 consumes one candidate_shared_transform, not per-frame "
                "candidate_offsets")
        if candidate_shared_transform is None:
            candidate_shared_transform = sample_candidate_offset(candidate_id, config)
        candidate_shared_transform = validate_shared_se2_transform(
            candidate_shared_transform).astype(np.float32)
        if int(candidate_id) == 0 and not np.array_equal(
                candidate_shared_transform, np.zeros(3, dtype=np.float32)):
            raise ValueError("candidate0 must remain the exact identity in shared_se2 mode")
    else:
        if candidate_shared_transform is not None:
            raise ValueError(
                "candidate_shared_transform is only valid for shared_se2 mode")
        if candidate_offsets is not None:
            candidate_offsets = np.asarray(candidate_offsets, dtype=np.float32)
            if candidate_offsets.shape != (num_hist, 3):
                raise ValueError(
                    f"candidate_offsets must have shape {(num_hist, 3)}, "
                    f"got {candidate_offsets.shape}.")
            if not np.isfinite(candidate_offsets).all():
                raise ValueError("candidate_offsets contains non-finite values.")
        else:
            candidate_offsets = np.stack(
                [sample_candidate_offset(candidate_id, config) for _ in range(num_hist)],
                axis=0,
            )
    point_sampling_seeds = data.get('point_sampling_seeds')
    if point_sampling_seeds is not None:
        point_sampling_seeds = np.asarray(point_sampling_seeds, dtype=np.int64)
        if point_sampling_seeds.shape != (num_hist,):
            raise ValueError(
                f"point_sampling_seeds must have shape {(num_hist,)}, "
                f"got {point_sampling_seeds.shape}.")
        prev_sampling_seeds = [int(seed) for seed in point_sampling_seeds]
    else:
        prev_sampling_seeds = [None] * num_hist
    current_sampling_seed = data.get('current_sampling_seed')
    if current_sampling_seed is not None:
        current_sampling_seed = int(current_sampling_seed)

    # Check the number of empty boxes
    for prev_box, prev_pc in zip(prev_boxs, prev_pcs):
        num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points[0:3,:]).sum()
        if num_points_in_prev_box < config.limit_num_points_in_prev_box:
            empty_counter += 1
    assert empty_counter < config.empty_box_limit, 'not enough valid box' 

    if candidate_trajectory_mode == 'shared_se2':
        ref_boxs = apply_shared_se2_to_boxes(
            prev_boxs, candidate_shared_transform, degrees=config.degrees)
        candidate_offsets = equivalent_local_offsets(
            prev_boxs, ref_boxs, degrees=config.degrees)
        candidate_shared_world_translation = shared_se2_world_translation(
            prev_boxs[0], candidate_shared_transform, degrees=config.degrees).astype(np.float32)
    else:
        ref_boxs = []
        for i, prev_box in enumerate(prev_boxs): # Apply a random offset to each box, not uniformly
            sample_offsets = candidate_offsets[i]
            ref_box = points_utils.getOffsetBB(prev_box, sample_offsets, limit_box=config.data_limit_box,
                                            degrees=config.degrees)
            ref_boxs.append(ref_box)
        candidate_shared_transform = np.zeros(3, dtype=np.float32)
        candidate_shared_world_translation = np.zeros(3, dtype=np.float32)

    ct_motion_history_boxs, ct_search_history_boxs = build_ct_training_histories(
        canonical_prev_boxs,
        ref_boxs,
        candidate_offsets,
        candidate_id,
        candidate_trajectory_mode,
        training_mode=getattr(
            config, 'ct_history_training_mode', 'canonical'),
        correlation=float(getattr(
            config, 'ct_history_correlation', 0.75)),
        degrees=config.degrees,
    )
    canonical_ref_boxs = boxes_to_anchor_parameters(
        canonical_prev_boxs, canonical_prev_boxs[0], degrees=config.degrees)
    ct_motion_ref_boxs = boxes_to_anchor_parameters(
        ct_motion_history_boxs,
        canonical_prev_boxs[0],
        degrees=config.degrees,
    )

    real_time_fields = build_time_fields(
        prev_timestamps, current_timestamp,
        frame_ids=prev_frame_ids,
        current_frame_id=this_frame_id,
        use_real_time=use_real_time,
        default_step=default_time_step,
        pseudo_step=pseudo_time_step)
    relative_timestamps, delta_t_list, local_timestamps, current_timestamp = (
        real_time_fields)
    dynamics_time_mode = normalize_dynamics_time_mode(
        this_frame.get('_ct_dynamics_time_mode',
                       getattr(config, 'dynamics_time_mode', 'true')))
    effective_time_fields = build_effective_time_fields(
        dynamics_time_mode,
        real_time_fields,
        effective_frame_timestamps=[
            prev_frames[key].get('_ct_effective_timestamp')
            for key in sorted_prev_keys
        ],
        effective_current_timestamp=this_frame.get('_ct_effective_timestamp'),
        frame_ids=prev_frame_ids,
        current_frame_id=this_frame_id,
        default_step=float(getattr(
            config, 'dynamics_fixed_delta_t', default_time_step)),
        pseudo_step=pseudo_time_step,
    )
    (effective_relative_timestamps, effective_delta_t_list,
     effective_local_timestamps, effective_current_timestamp) = (
        effective_time_fields)
    main_current_value = float(getattr(config, 'main_time_current', 0.0))
    point_timestamps, corner_timestamps, main_timestamps = build_main_time_fields(
        valid_mask,
        relative_timestamps,
        local_timestamps,
        num_hist,
        pseudo_step=pseudo_time_step,
        source=getattr(config, 'main_time_source', 'real'),
        current_value=main_current_value)

    prev_frame_pcs = []
    for i, prev_pc in enumerate(prev_pcs):
        prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(prev_pc, ref_boxs[i], ref_boxs[0],
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)
        prev_frame_pcs.append(prev_frame_pc)

    this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(
        this_pc, ref_boxs[0], ref_boxs[0],
        scale=config.bb_scale,
        offset=config.bb_offset)
    baseline_search_points = this_frame_pc.points.T
    ct_search_box = None
    ct_search_diagnostics = {
        'valid': False,
        'query_delta_t': float(effective_delta_t_list[0]),
    }
    expanded_search_points = np.empty(
        (0, baseline_search_points.shape[1]),
        dtype=baseline_search_points.dtype,
    )
    if bool(getattr(config, 'use_time_guided_search', False)):
        search_history_mode = str(getattr(
            config,
            'ct_search_training_history',
            'canonical',
        )).strip().lower()
        if search_history_mode not in (
                'canonical', 'candidate', 'correlated_candidate'):
            raise ValueError(
                "ct_search_training_history must be canonical, candidate, "
                "or correlated_candidate")
        if search_history_mode == 'canonical':
            search_history_boxes = canonical_prev_boxs
        elif search_history_mode == 'candidate':
            search_history_boxes = ref_boxs
        else:
            search_history_boxes = ct_search_history_boxs
        ct_search_box, ct_search_diagnostics = build_time_guided_search_box(
            search_history_boxes,
            effective_delta_t_list,
            valid_mask=valid_mask,
            base_length=float(getattr(
                config, 'ct_search_base_length', 4.0)),
            base_width=float(getattr(
                config, 'ct_search_base_width', 2.0)),
            max_length=float(getattr(
                config, 'ct_search_max_length', 16.0)),
            max_width=float(getattr(
                config, 'ct_search_max_width', 6.0)),
            max_speed=float(getattr(
                config, 'ct_search_max_speed', 20.0)),
            max_displacement=float(getattr(
                config, 'ct_search_max_displacement', 12.0)),
            width_per_second=float(getattr(
                config, 'ct_search_width_per_second', 0.25)),
            min_displacement=float(getattr(
                config, 'ct_search_min_displacement', 0.2)),
        )
        if ct_search_box is not None:
            expanded_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                this_pc,
                ct_search_box,
                ref_boxs[0],
                scale=1.0,
                offset=0.0,
            )
            expanded_search_points = expanded_search_pc.points.T

    # Preserve the pre-normalization anchor: ref_boxs[0] becomes approximately
    # zero after transform_box(), so the normalized tensor cannot prove that
    # paired views shared the same crop and local coordinate system.
    coordinate_anchor_box = ref_boxs[0]
    coordinate_anchor_theta = (
        coordinate_anchor_box.orientation.degrees * coordinate_anchor_box.orientation.axis[-1]
        if config.degrees else coordinate_anchor_box.orientation.radians
        * coordinate_anchor_box.orientation.axis[-1]
    )
    coordinate_anchor = np.append(
        coordinate_anchor_box.center, coordinate_anchor_theta).astype('float32')

    this_box    = points_utils.transform_box(this_box, ref_boxs[0]) 
    prev_boxs   = [points_utils.transform_box(prev_box, ref_boxs[0]) for prev_box in prev_boxs] 
    ref_boxs    = [points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs]    
    motion_boxs = [points_utils.transform_box(this_box, prev_box) for prev_box in prev_boxs]  

    # Resample each frame of the point cloud to a specific number
    prev_points_list = [
        points_utils.regularize_pc(
            prev_frame_pc.points.T, config.point_sample_size, seed=seed)[0]
        for prev_frame_pc, seed in zip(prev_frame_pcs, prev_sampling_seeds)
    ]
    if bool(getattr(config, 'use_time_guided_search', False)):
        this_points, ct_search_sampling = stratified_search_sample(
            baseline_search_points,
            expanded_search_points,
            config.point_sample_size,
            baseline_ratio=float(getattr(
                config, 'ct_search_baseline_ratio', 0.75)),
            min_expansion_points=int(getattr(
                config, 'ct_search_min_expansion_points', 32)),
            seed=current_sampling_seed,
        )
    else:
        this_points = points_utils.regularize_pc(
            baseline_search_points,
            config.point_sample_size,
            seed=current_sampling_seed,
        )[0]
        ct_search_sampling = {
            'baseline_sample_count': int(config.point_sample_size),
            'expansion_sample_count': 0,
            'expansion_available_count': 0,
        }
    ct_search_active = ct_search_sampling['expansion_sample_count'] > 0
    num_points_in_search = int(len(baseline_search_points))
    if ct_search_active:
        num_points_in_search += int(
            ct_search_sampling['expansion_available_count'])
    search_has_usable_points = num_points_in_search > 2

    seg_label_this = geometry_utils.points_in_box(this_box, this_points.T[:3,:], config.bb_scale).astype(int)
    seg_label_prev_list = [geometry_utils.points_in_box(prev_box, prev_points.T[:3,:], config.bb_scale).astype(int) for prev_box, prev_points in zip(prev_boxs, prev_points_list)] #应当只考虑xyz特征
    seg_mask_prev_list = [geometry_utils.points_in_box(ref_box, prev_points.T[:3,:], config.bb_scale).astype(float) for ref_box,prev_points in zip(ref_boxs,prev_points_list)]#应当只考虑xyz特征
    if candidate_id != 0:
        for seg_mask_prev in seg_mask_prev_list:
            # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
            # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)

    timestamp_prev_list = [
        np.full((config.point_sample_size, 1), fill_value=timestamp, dtype=np.float32)
        for timestamp in point_timestamps
    ]
    timestamp_this = np.full(
        (config.point_sample_size, 1), fill_value=main_current_value, dtype=np.float32)

    prev_points_list = [
        np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]],
                       axis=-1)
        for prev_points, timestamp_prev, seg_mask_prev in zip(
            prev_points_list, timestamp_prev_list, seg_mask_prev_list)
    ]
    this_points = np.concatenate(
        [this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

    stack_points_list = prev_points_list + [this_points]
    stack_points = np.concatenate(stack_points_list, axis=0)

    stack_seg_label_list = seg_label_prev_list + [seg_label_this]
    stack_seg_label = np.hstack(stack_seg_label_list)

    theta_this = this_box.orientation.degrees * this_box.orientation.axis[-1] if config.degrees else \
        this_box.orientation.radians * this_box.orientation.axis[-1]
    box_label = np.append(this_box.center, theta_this).astype('float32')
    theta_prev_list = [
        prev_box.orientation.degrees * prev_box.orientation.axis[-1]
        if config.degrees else prev_box.orientation.radians *
        prev_box.orientation.axis[-1] for prev_box in prev_boxs
    ]
    box_label_prev_list = [
        np.append(prev_box.center, theta_prev).astype('float32')
        for prev_box, theta_prev in zip(prev_boxs, theta_prev_list)
    ]

    # Generate a reference box sequence
    theta_ref_list=[
        ref_box.orientation.degrees * ref_box.orientation.axis[-1]
        if config.degrees else ref_box.orientation.radians *
        ref_box.orientation.axis[-1] for ref_box in ref_boxs
    ]
    ref_box_list = [
        np.append(ref_box.center, theta_ref).astype('float32')
        for ref_box, theta_ref in zip(ref_boxs, theta_ref_list)
    ]

    theta_motion_list = [
        motion_box.orientation.degrees * motion_box.orientation.axis[-1]
        if config.degrees else motion_box.orientation.radians *
        motion_box.orientation.axis[-1] for motion_box in motion_boxs
    ]

    motion_label_list = [
        np.append(motion_box.center, theta_motion).astype('float32')
        for motion_box, theta_motion in zip(motion_boxs, theta_motion_list)
    ]
    motion_state_label_list = [ 
        np.sqrt(np.sum((this_box.center - prev_box.center)**2))
        > config.motion_threshold for prev_box in prev_boxs
    ]
    current_delta_t_real = delta_t_list[0] if len(delta_t_list) > 0 else default_time_step
    current_delta_t_effective = (
        effective_delta_t_list[0] if len(effective_delta_t_list) > 0
        else float(getattr(config, 'dynamics_fixed_delta_t', default_time_step)))
    # Supervision is always defined in physical time. A fixed/shuffled negative
    # control may alter only the time consumed by DynamicsEncoder.
    dynamics_displacement_label, velocity_label = canonical_dynamics_targets(
        canonical_prev_boxs,
        canonical_this_box,
        current_delta_t_real,
        degrees=config.degrees,
        eps=1e-3,
    )

    data_dict = {
        'points': stack_points.astype('float32'), # Historical first, then current
        'box_label': box_label, 
        'ref_boxs':np.stack(ref_box_list, axis=0),
        'box_label_prev': np.stack(box_label_prev_list, axis=0),
        'motion_label': np.stack(motion_label_list, axis=0),
        'motion_state_label': np.stack(motion_state_label_list, axis=0).astype('int'),
        'bbox_size': this_box.wlh, 
        'seg_label': stack_seg_label.astype('int'), 
        'valid_mask': np.array(valid_mask).astype('int'), 
        'timestamps': main_timestamps,
        'delta_t': np.array(delta_t_list, dtype=np.float32),
        'delta_t_real': np.array(delta_t_list, dtype=np.float32),
        'delta_t_effective': np.array(effective_delta_t_list, dtype=np.float32),
        'delta_T': np.array(corner_timestamps, dtype=np.float32),
        'timestamps_real': local_timestamps,
        'delta_T_real': np.array(relative_timestamps, dtype=np.float32),
        'timestamps_effective': np.asarray(effective_local_timestamps, dtype=np.float32),
        'delta_T_effective': np.array(effective_relative_timestamps, dtype=np.float32),
        'current_timestamp': np.float64(current_timestamp if current_timestamp is not None else 0.0),
        'current_effective_timestamp': np.float64(effective_current_timestamp),
        'current_delta_t': np.float32(current_delta_t_real),
        'current_delta_t_real': np.float32(current_delta_t_real),
        'current_delta_t_effective': np.float32(current_delta_t_effective),
        'dynamics_time_mode_id': np.int64(
            {'true': 0, 'fixed': 1, 'shuffled': 2}[dynamics_time_mode]),
        'num_points_in_search': np.float32(num_points_in_search),
        'search_has_usable_points': np.float32(search_has_usable_points),
        'ct_search_used': np.float32(ct_search_active),
        'ct_search_expansion_ratio': np.float32(
            ct_search_sampling['expansion_sample_count']
            / float(config.point_sample_size)),
        'ct_search_baseline_points': np.float32(len(baseline_search_points)),
        'ct_search_expansion_points': np.float32(
            ct_search_sampling['expansion_available_count']),
        'ct_search_query_delta_t': np.float32(
            ct_search_diagnostics.get(
                'query_delta_t', effective_delta_t_list[0])),
        'ct_search_predicted_displacement': np.float32(
            ct_search_diagnostics.get('displacement', 0.0)),
        'velocity_label': velocity_label,
        'dynamics_displacement_label': dynamics_displacement_label,
        'canonical_ref_boxs': canonical_ref_boxs,
        'ct_motion_ref_boxs': ct_motion_ref_boxs,
        'candidate_id': np.int64(candidate_id),
        'candidate_trajectory_mode_id': np.int64(
            {'independent': 0, 'shared_se2': 1}[candidate_trajectory_mode]),
        'coordinate_anchor': coordinate_anchor,
        'candidate_offsets': candidate_offsets.astype('float32'),
        'candidate_shared_transform': np.asarray(
            candidate_shared_transform, dtype=np.float32),
        'candidate_shared_world_translation': candidate_shared_world_translation,
    }
    if prev_frame_ids is not None:
        data_dict['prev_frame_ids'] = np.array(prev_frame_ids, dtype=np.int64)
    if this_frame_id is not None:
        data_dict['this_frame_id'] = np.int64(this_frame_id)
    if history_offsets is not None:
        data_dict['history_offsets'] = np.array(history_offsets, dtype=np.int64)
    if point_sampling_seeds is not None:
        data_dict['point_sampling_seeds'] = point_sampling_seeds
    if current_sampling_seed is not None:
        data_dict['current_sampling_seed'] = np.int64(current_sampling_seed)

    if getattr(config, 'box_aware', False):
        stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
        hist_points_list = stack_points_split[:num_hist] 
        prev_bc_list = [
            points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
            for hist_points, prev_box in zip(hist_points_list, prev_boxs)
        ]
        this_points_split = stack_points_split[-1] 
        this_bc = points_utils.get_point_to_box_distance(this_points_split[:,:3], this_box)


        candidate_bc_prev_list = [
            points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
            for hist_points, prev_box in zip(hist_points_list, ref_boxs)
        ]

        candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
        candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
        candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

        data_dict.update({'prev_bc': np.stack(prev_bc_list, axis=0).astype('float32'),
                          'this_bc': this_bc.astype('float32'),
                          'candidate_bc': candidate_bc.astype('float32')})

    return data_dict


class PointTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, random_sample, sample_per_epoch=10000, processing=siamese_processing, config=None,
                 **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.sample_per_epoch = sample_per_epoch
        self.dataset = dataset
        self.processing = processing
        self.config = config
        self.random_sample = random_sample
        self.num_candidates = getattr(config, 'num_candidates', 1)
        if getattr(self.config, "use_augmentation", False):
            print('using augmentation')
            self.transform = points_utils.apply_augmentation
        else:
            self.transform = None
        if not self.random_sample:
            num_frames_total = 0
            self.tracklet_start_ids = [num_frames_total]
            for i in range(dataset.get_num_tracklets()):
                num_frames_total += dataset.get_num_frames_tracklet(i)
                self.tracklet_start_ids.append(num_frames_total)

    def get_anno_index(self, index):
        return index // self.num_candidates 

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        if self.random_sample:
            return self.sample_per_epoch * self.num_candidates
        else:
            return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            if self.random_sample:
                tracklet_id = torch.randint(0, self.dataset.get_num_tracklets(), size=(1,)).item()
                tracklet_annos = self.dataset.tracklet_anno_list[tracklet_id]
                frame_ids = [0] + points_utils.random_choice(num_samples=2, size=len(tracklet_annos)).tolist()
            else:
                for i in range(0, self.dataset.get_num_tracklets()):
                    if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                        tracklet_id = i 
                        this_frame_id = anno_id - self.tracklet_start_ids[i] 
                        prev_frame_id = max(this_frame_id - 1, 0) 
                        frame_ids = (0, prev_frame_id, this_frame_id) 
            first_frame, template_frame, search_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {"first_frame": first_frame,
                    "template_frame": template_frame,
                    "search_frame": search_frame,
                    "candidate_id": candidate_id}

            return self.processing(data, self.config,
                                   template_transform=None,
                                   search_transform=self.transform)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


class TestTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, config=None, **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)


class MotionTrackingSampler(PointTrackingSampler):
    def __init__(self, dataset, config=None, **kwargs):
        super().__init__(dataset, random_sample=False, config=config, **kwargs)
        self.processing = motion_processing

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index) 
        try:

            for i in range(0, self.dataset.get_num_tracklets()):
                if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_id = max(this_frame_id - 1, 0)
                    frame_ids = (0, prev_frame_id, this_frame_id)
            first_frame, prev_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {
                "first_frame": first_frame, 
                "prev_frame": prev_frame,  
                "this_frame": this_frame,   
                "candidate_id": candidate_id}
        
            return self.processing(data, self.config,
                                   template_transform=self.transform,
                                   search_transform=self.transform)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


class MotionTrackingSamplerMF(PointTrackingSampler):
    def __init__(self, dataset, config=None, **kwargs):
        super().__init__(dataset, random_sample=False, config=config, **kwargs)
        self.processing = motion_processing_mf
        self.use_twc = getattr(self.config, 'use_twc', False)
        self.use_m3_path_distillation = bool(getattr(
            self.config, 'use_m3_path_distillation', False))
        self.use_paired_history = bool(
            self.use_twc or self.use_m3_path_distillation)
        self.candidate_trajectory_mode = normalize_candidate_trajectory_mode(
            getattr(self.config, 'candidate_trajectory_mode', 'independent'))
        self.paired_candidate_zero_only = bool(getattr(
            self.config,
            'm3_candidate_zero_only'
            if self.use_m3_path_distillation else 'twc_candidate_zero_only',
            getattr(self.config, 'twc_candidate_zero_only', True),
        ))
        default_twc_b_offsets = [1 + 2 * i for i in range(self.dataset.hist_num)]
        self.twc_view_a_offsets = list(getattr(
            self.config,
            'm3_view_a_offsets' if self.use_m3_path_distillation else 'twc_view_a_offsets',
            getattr(
                self.config,
                'twc_view_a_offsets',
                list(range(1, self.dataset.hist_num + 1)))))
        self.twc_view_b_offsets = list(getattr(
            self.config,
            'm3_view_b_offsets' if self.use_m3_path_distillation else 'twc_view_b_offsets',
            getattr(self.config, 'twc_view_b_offsets', default_twc_b_offsets)))
        if self.use_paired_history and getattr(self.config, "use_augmentation", False):
            raise ValueError(
                "Paired history views require explicit shared transforms; "
                "keep use_augmentation=False.")
        if self.use_paired_history:
            if (len(self.twc_view_a_offsets) != self.dataset.hist_num
                    or len(self.twc_view_b_offsets) != self.dataset.hist_num):
                raise ValueError(
                    "Each paired history view must provide exactly hist_num offsets.")
            if self.twc_view_a_offsets[0] != 1 or self.twc_view_b_offsets[0] != 1:
                raise ValueError(
                    "Paired views must share the nearest t-1 anchor.")

    def _locate_tracklet(self, anno_id):
        for i in range(0, self.dataset.get_num_tracklets()):
            if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                return i, anno_id - self.tracklet_start_ids[i]
        raise IndexError(f"anno_id {anno_id} is outside tracklet ranges.")

    def _build_view(self, tracklet_id, this_frame_id, first_frame, this_frame,
                    candidate_id, offsets, candidate_offset_map=None,
                    point_sampling_seed_map=None, current_sampling_seed=None,
                    candidate_shared_transform=None):
        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
            this_frame_id, self.dataset.hist_num, offsets=offsets)
        prev_frames_tuple = self.dataset.get_frames(tracklet_id, frame_ids=prev_frame_ids)
        prev_frames_dict = create_history_frame_dict(prev_frames_tuple)
        data = {
            "first_frame": first_frame,
            "prev_frames": prev_frames_dict,
            "this_frame": this_frame,
            "candidate_id": candidate_id,
            "valid_mask": valid_mask,
            "prev_frame_ids": prev_frame_ids,
            "this_frame_id": this_frame_id,
            "history_offsets": offsets,
        }
        if candidate_offset_map is not None:
            data["candidate_offsets"] = candidate_offsets_for_frame_ids(
                prev_frame_ids, candidate_offset_map)
        if candidate_shared_transform is not None:
            data["candidate_shared_transform"] = np.asarray(
                candidate_shared_transform, dtype=np.float32)
        if point_sampling_seed_map is not None:
            data["point_sampling_seeds"] = point_sampling_seeds_for_frame_ids(
                prev_frame_ids, point_sampling_seed_map)
        if current_sampling_seed is not None:
            data["current_sampling_seed"] = current_sampling_seed
        return self.processing(data, self.config,
                               template_transform=self.transform,
                               search_transform=self.transform)

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index) 
        try:
            tracklet_id, this_frame_id = self._locate_tracklet(anno_id)
            frame_ids = (0, this_frame_id)
            first_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            if self.use_paired_history:
                paired_candidate_id = (
                    0 if self.paired_candidate_zero_only else candidate_id)
                prev_frame_ids_a, _ = get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num, offsets=self.twc_view_a_offsets)
                prev_frame_ids_b, _ = get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num, offsets=self.twc_view_b_offsets)
                # Sample perturbations and regularization seeds once per absolute
                # frame id. Any physical history frame appearing in both paths,
                # especially t-1, must share crop, coordinates, and sampled XYZ.
                if self.candidate_trajectory_mode == 'shared_se2':
                    candidate_offset_map = None
                    candidate_shared_transform = sample_candidate_offset(
                        paired_candidate_id, self.config)
                else:
                    candidate_offset_map = build_shared_candidate_offset_map(
                        paired_candidate_id,
                        list(prev_frame_ids_a) + list(prev_frame_ids_b),
                        self.config,
                    )
                    candidate_shared_transform = None
                point_sampling_seed_map = build_shared_point_sampling_seed_map(
                    list(prev_frame_ids_a) + list(prev_frame_ids_b))
                current_sampling_seed = sample_point_sampling_seed()
                view_a = self._build_view(tracklet_id, this_frame_id, first_frame, this_frame,
                                          paired_candidate_id, self.twc_view_a_offsets,
                                          candidate_offset_map=candidate_offset_map,
                                          point_sampling_seed_map=point_sampling_seed_map,
                                          current_sampling_seed=current_sampling_seed,
                                          candidate_shared_transform=candidate_shared_transform)
                view_b = self._build_view(tracklet_id, this_frame_id, first_frame, this_frame,
                                          paired_candidate_id, self.twc_view_b_offsets,
                                          candidate_offset_map=candidate_offset_map,
                                          point_sampling_seed_map=point_sampling_seed_map,
                                          current_sampling_seed=current_sampling_seed,
                                          candidate_shared_transform=candidate_shared_transform)
                return {"view_a": view_a, "view_b": view_b}

            offsets = list(range(1, self.dataset.hist_num + 1))
            return self._build_view(tracklet_id, this_frame_id, first_frame, this_frame,
                                    candidate_id, offsets)
        except AssertionError:
            # return 1
            return self[torch.randint(0, len(self), size=(1,)).item()]
