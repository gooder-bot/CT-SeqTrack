# Created by zenn at 2021/4/27
# Modified by Aron Lin at Jun 4 20:32:36 CST 2023 

import copy
import numpy as np
import torch
from easydict import EasyDict
from nuscenes.utils import geometry_utils
from pyquaternion import Quaternion

import datasets.points_utils as points_utils
from utils.ct_history import (
    b2_v3_history_mode_id,
    build_alternating_aux_history_offsets,
    build_irregular_history_offsets,
    select_b2_v3_history_mode,
)

from datasets.misc_utils import get_history_frame_ids_and_masks, \
    create_history_frame_dict, \
    generate_virtual_points, \
    build_time_fields, \
    build_main_time_fields, \
    build_effective_time_fields, \
    normalize_dynamics_time_mode
from utils.sampling_utils import (
    build_shared_candidate_offset_map,
    build_shared_point_sampling_seed_map,
    candidate_offsets_for_frame_ids,
    point_sampling_seeds_for_frame_ids,
    sample_candidate_offset,
    sample_point_sampling_seed,
    deterministic_candidate_offset,
    deterministic_candidate_retry_index,
    deterministic_point_seed,
    prune_seqtrack_observation_payload,
    stable_uint32_seed,
)
from utils.candidate_utils import (
    anchor_relative_trajectory_targets,
    apply_shared_se2_to_boxes,
    boxes_to_anchor_parameters,
    build_b1_physical_contract,
    build_ct_training_histories,
    canonical_dynamics_targets,
    equivalent_local_offsets,
    normalize_candidate_trajectory_mode,
    physical_motion_targets,
    reexpress_motion_prediction,
    shared_se2_world_translation,
    validate_shared_se2_transform,
)
from utils.ct_search import (
    build_causal_history_corridor,
    build_ordered_trajectory_search_box,
    build_time_guided_search_box,
    combined_search_support_statistics,
    resolve_b1_search_support,
    resolve_joint_search_geometry,
    sample_padded_search_extension,
    sample_joint_novel_extensions,
    sample_bounded_novel_prepool,
    sample_source_aware_endpoint_points,
    sample_search_extension,
    stratified_search_sample,
    useful_search_coverage_need,
)

from utils.replay_cache import RecursiveReplayCache, replay_config_sha256
from utils.recursive_state import (
    OnlineRecursiveBatchSampler,
    stable_tracklet_partition,
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
        'bbox_size': (
            data['first_frame']['3d_bbox'].wlh
            if bool(getattr(
                config, 'observation_safe_bbox_size', False))
            else this_box.wlh),
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
    v27 = bool(getattr(config, 'ct_enable_v27', False))
    from utils.point_identity import raw_point_ids, sampled_identity
    from functools import partial
    crop_subwindow = partial(points_utils.generate_subwindow_with_aroundboxs,
                             canonicalize=v27)
    prev_frames = data['prev_frames']
    this_frame = data['this_frame']
    candidate_id = data['candidate_id']
    recursive_replay = data.get('recursive_replay')
    online_recursive_state = data.get('online_recursive_state')
    if recursive_replay is not None and online_recursive_state is not None:
        raise ValueError(
            "recursive replay and online recursive state are mutually exclusive")
    valid_mask = (
        list(recursive_replay['history_valid_mask'])
        if recursive_replay is not None else
        list(online_recursive_state['history_valid_mask'])
        if online_recursive_state is not None else data['valid_mask'])
    num_hist = len(valid_mask)
    empty_counter = 0

    prev_pcs  = [prev_frames[key]['pc'] for key in sorted(prev_frames,key=lambda k: abs(int(k)))] # Ordered point clouds, -1, -2, -3
    prev_boxs = [prev_frames[key]['3d_bbox'] for key in sorted(prev_frames,key=lambda k: abs(int(k)))] # Ordered point clouds, -1, -2, -3
    this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']
    joint_contract_v2 = bool(
        getattr(config, 'ct_joint_contract_version', 1) >= 2)
    joint_contract_v3 = bool(
        getattr(config, 'ct_joint_contract_version', 1) >= 3)
    # Keep the untouched GT trajectory for M1 labels and invariance audits.
    # ``transform_box`` is non-mutating, but the local variables below are
    # intentionally rebound to candidate-coordinate boxes.
    ground_truth_history = list(prev_boxs)
    recursive_history = list(prev_boxs)
    candidate_history = None
    # Contract-v1-only compatibility alias.  Contract v2 never uses this
    # name to stand for GT, recursive state, and candidate state at once.
    legacy_canonical_prev_boxs = list(prev_boxs)
    canonical_this_box = this_box
    if online_recursive_state is not None:
        state_rows = np.asarray(
            online_recursive_state['history_boxes_world'], dtype=np.float64)
        if state_rows.shape != (num_hist, 7):
            raise ValueError(
                f"online recursive history must have shape {(num_hist, 7)}")
        online_boxs = []
        for source_box, row in zip(prev_boxs, state_rows):
            state_box = copy.deepcopy(source_box)
            state_box.center = row[:3].copy()
            state_box.wlh = row[3:6].copy()
            state_box.orientation = Quaternion(
                axis=[0, 0, 1], radians=float(row[6]))
            online_boxs.append(state_box)
        # Every input branch starts from the same deployed recursive state.
        # Current-frame GT remains available only for labels below.
        prev_boxs = online_boxs
        legacy_canonical_prev_boxs = list(online_boxs)
        recursive_history = list(online_boxs)
    motion_anchor_box = recursive_history[0]
    sorted_prev_keys = sorted(prev_frames, key=lambda k: abs(int(k)))
    prev_timestamps = (
        list(online_recursive_state['history_timestamps'])
        if online_recursive_state is not None
        else [prev_frames[key].get('timestamp') for key in sorted_prev_keys])
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
            candidate_shared_transform = (
                deterministic_candidate_offset(
                    candidate_id, config,
                    data.get('tracklet_key', data.get('tracklet_id', 'unknown')),
                    data.get('this_frame_id', data.get('sample_index', 0)),
                    'online_recovery_view')
                if online_recursive_state is not None
                else sample_candidate_offset(candidate_id, config))
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
            if online_recursive_state is not None:
                raise ValueError(
                    "online recursive training requires candidate_trajectory_mode=shared_se2")
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
        if online_recursive_state is not None:
            tracklet_seed_key = data.get(
                'tracklet_key', data.get('tracklet_id', 'unknown'))
            current_seed_frame = data.get(
                'this_frame_id', data.get('sample_index', 0))
            prev_sampling_seeds = [
                deterministic_point_seed(
                    config, tracklet_seed_key, current_seed_frame,
                    int(candidate_id), 'history', index)
                for index in range(num_hist)]
        else:
            prev_sampling_seeds = [None] * num_hist
    current_sampling_seed = data.get('current_sampling_seed')
    if current_sampling_seed is not None:
        current_sampling_seed = int(current_sampling_seed)
    elif online_recursive_state is not None:
        current_sampling_seed = deterministic_point_seed(
            config,
            data.get('tracklet_key', data.get('tracklet_id', 'unknown')),
            data.get('this_frame_id', data.get('sample_index', 0)),
            int(candidate_id), 'current')
    sample_index = int(data.get(
        'sample_index', this_frame_id if this_frame_id is not None else 0))

    # Check the number of empty boxes
    for prev_box, prev_pc in zip(prev_boxs, prev_pcs):
        num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points[0:3,:]).sum()
        if num_points_in_prev_box < config.limit_num_points_in_prev_box:
            empty_counter += 1
    if online_recursive_state is None and not v27:
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
    candidate_history = list(ref_boxs)

    if recursive_replay is not None:
        if int(candidate_id) != 0:
            raise ValueError(
                "recursive replay is defined for candidate_id=0 only")
        replay_rows = np.asarray(
            recursive_replay['history_boxes_world'], dtype=np.float64)
        if replay_rows.shape != (num_hist, 7):
            raise ValueError(
                f"recursive replay history must have shape {(num_hist, 7)}")
        ref_boxs = []
        for source_box, row in zip(
                legacy_canonical_prev_boxs, replay_rows):
            replay_box = copy.deepcopy(source_box)
            replay_box.center = row[:3].copy()
            replay_box.wlh = row[3:6].copy()
            replay_box.orientation = Quaternion(
                axis=[0, 0, 1], radians=float(row[6]))
            ref_boxs.append(replay_box)
        candidate_offsets = equivalent_local_offsets(
            legacy_canonical_prev_boxs, ref_boxs, degrees=config.degrees)
        candidate_shared_transform = np.zeros(3, dtype=np.float32)
        candidate_shared_world_translation = np.zeros(3, dtype=np.float32)

    use_ct_joint_full = bool(getattr(config, 'use_ct_joint_full', False))
    observation_only = bool(data.get('ct_observation_only', False))
    use_search_evidence_v3 = (
        bool(getattr(config, 'use_motion_conditioned_search_v3', False))
        or use_ct_joint_full) and not observation_only
    if use_search_evidence_v3 and not bool(getattr(
            config, 'use_b1motion_v3', False)):
        raise ValueError("B2-v3 requires B1motion-v3 shared history")
    b2_v3_history_mode = None
    if use_search_evidence_v3:
        b2_v3_history_mode = (
            'recursive_replay' if recursive_replay is not None
            else select_b2_v3_history_mode(
                data.get('tracklet_key', data.get('tracklet_id', 'unknown')),
                this_frame_id if this_frame_id is not None else sample_index,
                candidate_id,
                seed=int(getattr(config, 'seed', 42)),
            ))
    history_base_boxs = (
        recursive_history if joint_contract_v2
        else legacy_canonical_prev_boxs)
    ct_motion_history_boxs, ct_search_history_boxs = build_ct_training_histories(
        history_base_boxs,
        ref_boxs,
        candidate_offsets,
        candidate_id,
        candidate_trajectory_mode,
        training_mode=(
            b2_v3_history_mode if use_search_evidence_v3
            else getattr(config, 'ct_history_training_mode', 'canonical')),
        correlation=float(getattr(
            config, 'ct_history_correlation', 0.75)),
        recursive_error_scale=float(getattr(
            config, 'ct_history_recursive_error_scale', 1.0)),
        degrees=config.degrees,
    )
    if recursive_replay is not None:
        # The cache already contains the causal frozen-B0 rollout.  Applying
        # another synthetic perturbation here would recreate the train/test
        # mismatch that the replay path is designed to remove.
        ct_motion_history_boxs = ref_boxs
        ct_search_history_boxs = ref_boxs
    if online_recursive_state is not None:
        # Candidate recovery views are coherent transformations of the full
        # deployed state.  Do not synthesize a second, independently perturbed
        # motion or Search history.
        ct_motion_history_boxs = candidate_history
        ct_search_history_boxs = candidate_history
    canonical_label_boxs = (
        ground_truth_history if joint_contract_v2
        else legacy_canonical_prev_boxs)
    canonical_ref_boxs = boxes_to_anchor_parameters(
        canonical_label_boxs, canonical_label_boxs[0],
        degrees=config.degrees)
    use_ordered_trajectory = bool(getattr(
        config, 'use_ordered_trajectory_encoder', False))
    if use_ordered_trajectory:
        # The online path receives recursive estimated boxes expressed in the
        # latest estimated crop anchor.  Reuse the candidate-anchored search
        # history here so training sees the same coordinate/error contract.
        ordered_motion_history_boxs = ct_search_history_boxs
        ordered_motion_anchor = ref_boxs[0]
    else:
        # Frozen legacy B1 checkpoints keep their original GT-anchor contract.
        ordered_motion_history_boxs = ct_motion_history_boxs
        ordered_motion_anchor = legacy_canonical_prev_boxs[0]
    ct_motion_ref_boxs = boxes_to_anchor_parameters(
        ordered_motion_history_boxs,
        ordered_motion_anchor,
        degrees=config.degrees,
    )
    use_motion_v3 = bool(getattr(config, 'use_b1motion_v3', False))
    motion_main_ref_boxs = None
    if use_motion_v3:
        # Match recursive inference: the newest trajectory box is the actual
        # crop anchor (therefore exactly zero in local coordinates), while
        # older estimated-box errors evolve smoothly.  Using the clean newest
        # GT box here would expose candidate anchor error that is unavailable
        # online and recreate the v2 identifiability bug.
        if joint_contract_v2:
            # B1 is candidate-view invariant.  Recovery views alter the B0
            # crop and Search anchor, not the physical history signal.
            motion_main_ref_boxs = boxes_to_anchor_parameters(
                recursive_history,
                motion_anchor_box,
                degrees=config.degrees,
            )
        else:
            motion_main_ref_boxs = boxes_to_anchor_parameters(
                ct_search_history_boxs,
                ref_boxs[0],
                degrees=config.degrees,
            )
        if not np.allclose(
                motion_main_ref_boxs[0], 0.0, rtol=0.0, atol=1e-5):
            raise ValueError(
                "B1motion-v3 newest main history must equal the crop anchor")
        if use_search_evidence_v3:
            shared_search_ref_boxs = (
                motion_main_ref_boxs.copy()
                if joint_contract_v2 else boxes_to_anchor_parameters(
                    ct_search_history_boxs,
                    ref_boxs[0],
                    degrees=config.degrees,
                ))
            if not np.array_equal(
                    motion_main_ref_boxs, shared_search_ref_boxs):
                raise RuntimeError(
                    "B2-v3 requires byte-identical B1/B2 history tensors")

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
    if recursive_replay is not None:
        replay_delta_t = np.asarray(
            recursive_replay['delta_t'], dtype=np.float32)
        if replay_delta_t.shape != (num_hist,):
            raise ValueError(
                "recursive replay delta_t must match history length")
        if (not np.isfinite(replay_delta_t).all()
                or np.any(replay_delta_t <= 0)):
            raise ValueError(
                "recursive replay delta_t must be finite and positive")
        replay_current_delta_t = float(
            recursive_replay['current_delta_t'])
        if (not np.isfinite(replay_current_delta_t)
                or replay_current_delta_t <= 0):
            raise ValueError(
                "recursive replay current_delta_t must be finite and positive")
        effective_delta_t_list = replay_delta_t.tolist()
        # Keep relative clock diagnostics internally consistent with the
        # cached physical intervals (newest-to-oldest ordering).
        cumulative = np.cumsum(replay_delta_t)
        effective_relative_timestamps = (-cumulative).tolist()
        effective_local_timestamps = np.asarray(
            effective_relative_timestamps + [0.0], dtype=np.float32)
    v27_b1_input = None
    if v27 and use_motion_v3:
        from utils.b1_acquisition import build_b1_input_arrays
        v27_b1_input = build_b1_input_arrays(
            recursive_history, effective_delta_t_list, valid_mask,
            history_quality=(online_recursive_state or {}).get('history_quality'),
            recursive_age=(online_recursive_state or {}).get('recursive_age', 0.),
            first_frame_wlh=data['first_frame']['3d_bbox'].wlh,
            degrees=config.degrees,
            time_scale=float(getattr(config, 'time_scale', .5)))
        motion_main_ref_boxs = v27_b1_input['ref_boxs']
    motion_aux_contract = None
    if use_motion_v3 and not observation_only and not data.get('_ct_inference', False):
        motion_aux_prev_frames = data.get('motion_aux_prev_frames')
        if motion_aux_prev_frames is None:
            raise KeyError(
                "B1motion-v3 training requires motion_aux_prev_frames")
        motion_aux_keys = sorted(
            motion_aux_prev_frames, key=lambda key: abs(int(key)))
        motion_aux_ground_truth_boxs = [
            motion_aux_prev_frames[key]['3d_bbox']
            for key in motion_aux_keys
        ]
        motion_aux_canonical_boxs = list(motion_aux_ground_truth_boxs)
        motion_aux_valid_mask = list(data['motion_aux_valid_mask'])
        motion_aux_frame_ids = list(data['motion_aux_frame_ids'])
        online_motion_aux_state = data.get('online_motion_aux_state')
        if online_motion_aux_state is not None:
            aux_rows = np.asarray(
                online_motion_aux_state['history_boxes_world'],
                dtype=np.float64)
            if aux_rows.shape != (num_hist, 7):
                raise ValueError(
                    f"online auxiliary history must have shape "
                    f"{(num_hist, 7)}")
            online_aux_boxs = []
            for source_box, row in zip(
                    motion_aux_canonical_boxs, aux_rows):
                state_box = copy.deepcopy(source_box)
                state_box.center = row[:3].copy()
                state_box.wlh = row[3:6].copy()
                state_box.orientation = Quaternion(
                    axis=[0, 0, 1], radians=float(row[6]))
                online_aux_boxs.append(state_box)
            motion_aux_canonical_boxs = online_aux_boxs
        if candidate_trajectory_mode == 'shared_se2':
            motion_aux_candidate_boxs = apply_shared_se2_to_boxes(
                motion_aux_canonical_boxs,
                candidate_shared_transform,
                degrees=config.degrees,
            )
            motion_aux_offsets = equivalent_local_offsets(
                motion_aux_canonical_boxs,
                motion_aux_candidate_boxs,
                degrees=config.degrees,
            )
        else:
            motion_aux_offsets = candidate_offsets.copy()
            motion_aux_candidate_boxs = [
                points_utils.getOffsetBB(
                    box,
                    motion_aux_offsets[index],
                    limit_box=config.data_limit_box,
                    degrees=config.degrees,
                )
                for index, box in enumerate(motion_aux_canonical_boxs)
            ]
        _, motion_aux_history_boxs = build_ct_training_histories(
            motion_aux_canonical_boxs,
            motion_aux_candidate_boxs,
            motion_aux_offsets,
            candidate_id,
            candidate_trajectory_mode,
            training_mode=getattr(
                config,
                'motion_v3_history_training_mode',
                'correlated_candidate',
            ),
            correlation=float(getattr(
                config, 'motion_v3_history_correlation', 0.75)),
            recursive_error_scale=1.0,
            degrees=config.degrees,
        )
        if online_motion_aux_state is not None:
            # Auxiliary gaps obey the same causal state contract as the main
            # B1 history; no independently synthesized history is allowed.
            motion_aux_history_boxs = motion_aux_candidate_boxs
        if joint_contract_v2:
            motion_aux_anchor = motion_aux_canonical_boxs[0]
            motion_aux_ref_boxs = boxes_to_anchor_parameters(
                motion_aux_canonical_boxs,
                motion_aux_anchor,
                degrees=config.degrees,
            )
        else:
            motion_aux_anchor = motion_aux_candidate_boxs[0]
            motion_aux_ref_boxs = boxes_to_anchor_parameters(
                motion_aux_history_boxs,
                motion_aux_anchor,
                degrees=config.degrees,
            )
        if not np.allclose(
                motion_aux_ref_boxs[0], 0.0, rtol=0.0, atol=1e-5):
            raise ValueError(
                "B1motion-v3 newest auxiliary history must equal its anchor")
        motion_aux_prev_timestamps = (
            list(online_motion_aux_state['history_timestamps'])
            if online_motion_aux_state is not None
            else [motion_aux_prev_frames[key].get('timestamp')
                  for key in motion_aux_keys])
        motion_aux_real_time_fields = build_time_fields(
            motion_aux_prev_timestamps,
            current_timestamp,
            frame_ids=motion_aux_frame_ids,
            current_frame_id=this_frame_id,
            use_real_time=use_real_time,
            default_step=default_time_step,
            pseudo_step=pseudo_time_step,
        )
        motion_aux_effective_time_fields = build_effective_time_fields(
            dynamics_time_mode,
            motion_aux_real_time_fields,
            effective_frame_timestamps=[
                motion_aux_prev_frames[key].get('_ct_effective_timestamp')
                for key in motion_aux_keys
            ],
            effective_current_timestamp=this_frame.get(
                '_ct_effective_timestamp'),
            frame_ids=motion_aux_frame_ids,
            current_frame_id=this_frame_id,
            default_step=float(getattr(
                config, 'dynamics_fixed_delta_t', default_time_step)),
            pseudo_step=pseudo_time_step,
        )
        motion_aux_delta_t_real = motion_aux_real_time_fields[1]
        motion_aux_delta_t_effective = motion_aux_effective_time_fields[1]
        motion_aux_current_delta_t_real = (
            motion_aux_delta_t_real[0]
            if motion_aux_delta_t_real else default_time_step)
        motion_aux_current_delta_t_effective = (
            motion_aux_delta_t_effective[0]
            if motion_aux_delta_t_effective else float(getattr(
                config, 'dynamics_fixed_delta_t', default_time_step)))
        v27_aux_input = None
        if v27:
            from utils.b1_acquisition import build_b1_input_arrays
            v27_aux_input = build_b1_input_arrays(
                motion_aux_canonical_boxs, motion_aux_delta_t_effective,
                motion_aux_valid_mask,
                history_quality=(online_motion_aux_state or {}).get('history_quality'),
                recursive_age=(online_motion_aux_state or {}).get('recursive_age', 0.),
                first_frame_wlh=data['first_frame']['3d_bbox'].wlh,
                degrees=config.degrees,
                time_scale=float(getattr(config, 'time_scale', .5)))
            motion_aux_ref_boxs = v27_aux_input['ref_boxs']
        if joint_contract_v2:
            motion_aux_physical = build_b1_physical_contract(
                canonical_this_box,
                motion_aux_ground_truth_boxs,
                motion_aux_canonical_boxs,
                motion_aux_current_delta_t_real,
                degrees=config.degrees,
                eps=1e-3,
            )
            same_aux_axes = (np.allclose(
                motion_aux_ref_boxs, motion_aux_physical['ref_boxs'],
                rtol=1e-6, atol=1e-6) if v27 else np.array_equal(
                    motion_aux_ref_boxs, motion_aux_physical['ref_boxs']))
            if not same_aux_axes:
                raise RuntimeError(
                    "B1 auxiliary input and physical-label axes diverged")
            motion_aux_target_xy = motion_aux_physical['target_xy']
        else:
            motion_aux_target_xy, _ = physical_motion_targets(
                canonical_this_box,
                motion_aux_canonical_boxs[0],
                motion_aux_anchor,
                motion_aux_current_delta_t_real,
                degrees=config.degrees,
                eps=1e-3,
            )
        motion_aux_contract = {
            'motion_aux_ref_boxs': motion_aux_ref_boxs.astype('float32'),
            'motion_aux_delta_t': np.asarray(
                motion_aux_delta_t_effective, dtype=np.float32),
            'motion_aux_current_delta_t': np.float32(
                motion_aux_current_delta_t_effective),
            'motion_aux_valid_mask': np.asarray(
                motion_aux_valid_mask, dtype=np.int64),
            'motion_aux_target_xy': motion_aux_target_xy.astype('float32'),
            'motion_aux_query_gap_frames': np.int64(
                data['motion_aux_offsets'][0]),
        }
        if v27_aux_input is not None:
            motion_aux_contract['motion_aux_acquisition_features'] = (
                v27_aux_input['acquisition_features'])
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
        prev_frame_pc = crop_subwindow(prev_pc, ref_boxs[i], ref_boxs[0],
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)
        prev_frame_pcs.append(prev_frame_pc)

    this_frame_pc = crop_subwindow(
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
    corridor_box = None
    corridor_points = np.empty(
        (0, baseline_search_points.shape[1]),
        dtype=baseline_search_points.dtype,
    )
    corridor_diagnostics = {
        'valid': False, 'reason': 'v26_disabled', 'source_id': 0}
    v26_recovery_enabled = bool(getattr(
        config, 'ct_enable_v26_recovery', False))
    inner_core_point_count = None
    b1_prior_valid = True
    use_trajectory_search = (
        bool(getattr(config, 'use_trajectory_search', False))
        or use_ct_joint_full) and not observation_only
    if bool(getattr(config, 'use_time_guided_search', False)) and use_trajectory_search:
        raise ValueError(
            "legacy time-guided search and ordered trajectory search are "
            "mutually exclusive")
    if bool(getattr(config, 'use_time_guided_search', False)) or use_trajectory_search:
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
            search_history_boxes = (
                recursive_history if joint_contract_v2
                else legacy_canonical_prev_boxs)
        elif search_history_mode == 'candidate':
            search_history_boxes = ref_boxs
        else:
            search_history_boxes = ct_search_history_boxs
        if use_trajectory_search:
            ct_search_box, ct_search_diagnostics = (
                build_ordered_trajectory_search_box(
                    search_history_boxes,
                    effective_delta_t_list,
                    valid_mask=valid_mask,
                    base_length=float(getattr(
                        config, 'trajectory_search_base_length', 4.0)),
                    base_width=float(getattr(
                        config, 'trajectory_search_base_width', 2.0)),
                    max_length=float(getattr(
                        config, 'ct_tube_max_length', 24.0)
                        if use_ct_joint_full else getattr(
                            config, 'trajectory_search_max_length', 20.0)),
                    max_width=float(getattr(
                        config, 'ct_tube_max_width', 8.0)
                        if use_ct_joint_full else getattr(
                            config, 'trajectory_search_max_width', 8.0)),
                    max_speed=float(getattr(
                        config, 'ct_motion_max_speed', 20.0)
                        if use_ct_joint_full else getattr(
                            config, 'trajectory_search_max_speed', 20.0)),
                    max_acceleration=float(getattr(
                        config, 'ct_motion_max_acceleration', 8.0)
                        if use_ct_joint_full else getattr(
                            config, 'trajectory_search_max_acceleration', 8.0)),
                    max_displacement=float(getattr(
                        config, 'ct_motion_max_displacement', 12.0)
                        if use_ct_joint_full else getattr(
                            config, 'trajectory_search_max_displacement', 12.0)),
                    acceleration_weight=float(getattr(
                        config, 'ct_motion_acceleration_weight', 0.5)
                        if use_ct_joint_full else getattr(
                            config, 'trajectory_search_acceleration_weight', 0.5)),
                    sigma_parallel_scale=float(getattr(
                        config, 'trajectory_search_sigma_parallel_scale', 2.0)),
                    sigma_perpendicular_scale=float(getattr(
                        config, 'trajectory_search_sigma_perpendicular_scale', 2.0)),
                    min_displacement=float(getattr(
                        config, 'trajectory_search_min_displacement', 0.2)),
                    min_delta_t=float(getattr(
                        config, 'trajectory_search_min_delta_t', 0.75)),
                    min_gap_ratio=float(getattr(
                        config, 'trajectory_search_min_gap_ratio', 1.5)),
                    allow_normal_cadence=(
                        True if use_ct_joint_full else bool(getattr(
                            config,
                            'trajectory_search_allow_normal_cadence', False))),
                    require_recent_transition=use_ct_joint_full,
                ))
        else:
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
            expanded_search_pc = crop_subwindow(
                this_pc,
                ct_search_box,
                ref_boxs[0],
                scale=1.0,
                offset=0.0,
            )
            expanded_search_points = expanded_search_pc.points.T

    # B2-v2 is deliberately independent of the legacy long-tube paths above.
    # Candidate zero receives clean history.  Other candidates alternate
    # deterministically between correlated and recursive histories, matching
    # the recursive online error contract without consuming point-sampling RNG.
    use_search_evidence_v2 = bool(getattr(
        config, 'use_search_evidence_v2', False))
    use_search_evidence_v21 = bool(getattr(
        config, 'use_search_evidence_v21', False))
    use_search_evidence_v22 = bool(getattr(
        config, 'use_motion_conditioned_search_v22', False))
    if sum(map(bool, (
            use_search_evidence_v2,
            use_search_evidence_v21,
            use_search_evidence_v22,
            use_search_evidence_v3))) > 1:
        raise ValueError(
            "Search Evidence v2, v2.1, v2.2, and v3 are exclusive")
    use_endpoint_search_evidence = (
        use_search_evidence_v2
        or use_search_evidence_v21
        or use_search_evidence_v22
        or use_search_evidence_v3)
    search_config_prefix = (
        'search_v3' if use_search_evidence_v3
        else 'search_v22' if use_search_evidence_v22
        else 'search_v21' if use_search_evidence_v21
        else 'search_v2')

    def search_config_value(name, default):
        if use_ct_joint_full:
            joint_mapping = {
                'point_count': 'ct_endpoint_quota',
                'extension_quota': 'ct_endpoint_quota',
                'min_points': 'ct_search_min_points',
                'max_length': 'ct_tube_max_length',
                'max_width': 'ct_tube_max_width',
                'max_speed': 'ct_motion_max_speed',
                'max_acceleration': 'ct_motion_max_acceleration',
                'max_displacement': 'ct_motion_max_displacement',
                'acceleration_weight': 'ct_motion_acceleration_weight',
            }
            field = joint_mapping.get(name)
            if field is not None:
                return getattr(config, field, default)
        return getattr(config, f'{search_config_prefix}_{name}', default)

    search_v2_box = None
    search_v2_diagnostics = {
        'valid': False,
        'query_delta_t': float(effective_delta_t_list[0]),
        'gap_ratio': 1.0,
        'sigma_parallel': 0.0,
        'sigma_perpendicular': 0.0,
    }
    search_v2_expanded_points = np.empty(
        (0, baseline_search_points.shape[1]),
        dtype=baseline_search_points.dtype,
    )
    search_v2_endpoint_xy = np.zeros((2,), dtype=np.float32)
    if use_endpoint_search_evidence:
        if use_search_evidence_v3:
            search_v2_history_mode = b2_v3_history_mode
            search_v2_history_boxes = (
                recursive_history if joint_contract_v2
                else ct_search_history_boxs)
        elif int(candidate_id) == 0:
            search_v2_history_mode = 'canonical'
        elif sample_index % 2 == 0:
            search_v2_history_mode = 'correlated_candidate'
        else:
            search_v2_history_mode = 'recursive_candidate'
        if not use_search_evidence_v3:
            _, search_v2_history_boxes = build_ct_training_histories(
                history_base_boxs,
                ref_boxs,
                candidate_offsets,
                candidate_id,
                candidate_trajectory_mode,
                training_mode=search_v2_history_mode,
                correlation=float(search_config_value(
                    'history_correlation', 0.75)),
                recursive_error_scale=1.0,
                degrees=config.degrees,
            )
        replay_b1 = (
            recursive_replay.get('b1')
            if recursive_replay is not None else None)
        use_prepass_support = (
            bool(getattr(config, 'use_b1_prepass_support', False))
            if (not use_ct_joint_full or joint_contract_v2)
            else False)
        support_prediction = data.get('motion_prediction', replay_b1)
        if isinstance(replay_b1, dict) and recursive_replay is not None:
            support_prediction = dict(replay_b1)
            support_prediction.setdefault(
                'current_delta_t', recursive_replay['current_delta_t'])
        if (joint_contract_v2 and isinstance(support_prediction, dict)
                and bool(support_prediction.get('valid', False))):
            support_prediction = reexpress_motion_prediction(
                support_prediction, motion_anchor_box, ref_boxs[0])
        support_kwargs = dict(
            prediction=support_prediction,
            use_b1_prepass=use_prepass_support,
            use_dynamic_sigma=bool(getattr(
                config, 'search_v3_use_dynamic_sigma', False)),
            use_acquisition_margin=bool(getattr(
                config, 'ct_adaptive_acquisition_margin', False)),
            fixed_margins=(
                float(getattr(
                    config, 'search_v3_fixed_margin_parallel', 2.0)),
                float(getattr(
                    config, 'search_v3_fixed_margin_perpendicular', 1.0)),
            ),
            coverage_scale=float(getattr(
                config, 'search_v3_coverage_scale', 2.448)),
            standardized_residual_quantile=tuple(getattr(
                config,
                'search_v3_standardized_residual_q90_parallel_perpendicular',
                (1.0, 1.0))),
            min_direction_speed=float(getattr(
                config, 'motion_v3_min_direction_speed', 0.2)),
            max_length=float(search_config_value('max_length', 24.0)),
            max_width=float(search_config_value('max_width', 10.0)),
            fallback_max_speed=float(search_config_value('max_speed', 20.0)),
            fallback_max_acceleration=float(search_config_value(
                'max_acceleration', 8.0)),
            fallback_max_displacement=float(search_config_value(
                'max_displacement', 12.0)),
            fallback_acceleration_weight=float(search_config_value(
                'acceleration_weight', 0.5)),
            fallback_max_yaw_rate=float(search_config_value(
                'max_yaw_rate', np.pi / 2.0)),
            fallback_min_displacement=float(search_config_value(
                'min_displacement', 0.2))
            if not (use_ct_joint_full and joint_contract_v2) else 0.0,
            fallback_require_recent_transition=use_ct_joint_full,
        )
        if v27:
            support_kwargs.update(enable_v27=True,
                                  first_frame_size=data['first_frame']['3d_bbox'].wlh)
        if use_ct_joint_full and joint_contract_v2:
            (search_v2_box,
             ct_search_box,
             search_v2_diagnostics) = resolve_joint_search_geometry(
                search_v2_history_boxes,
                effective_delta_t_list,
                valid_mask,
                **support_kwargs,
            )
            if ct_search_box is not None:
                ct_search_diagnostics = dict(search_v2_diagnostics)
                expanded_search_pc = (
                    crop_subwindow(
                        this_pc, ct_search_box, ref_boxs[0],
                        scale=1.0, offset=0.0))
                expanded_search_points = expanded_search_pc.points.T
        else:
            search_v2_box, search_v2_diagnostics = resolve_b1_search_support(
                search_v2_history_boxes,
                effective_delta_t_list,
                valid_mask,
                **support_kwargs,
            )
        if search_v2_box is not None:
            learned_prior_support = (
                search_v2_diagnostics.get('prior_source') == 'b1')
            bounded_support = bool(
                learned_prior_support or v26_recovery_enabled)
            search_v2_expanded_pc = (
                crop_subwindow(
                    this_pc,
                    search_v2_box,
                    ref_boxs[0],
                    scale=(1.0 if bounded_support
                           else config.bb_scale),
                    offset=(0.0 if bounded_support
                            else config.bb_offset),
                ))
            search_v2_expanded_points = search_v2_expanded_pc.points.T
            endpoint_center = search_v2_diagnostics.get('endpoint_center')
            if endpoint_center is not None:
                endpoint_box = copy.deepcopy(ref_boxs[0])
                endpoint_box.center = np.asarray(
                    endpoint_center, dtype=np.float64)
                endpoint_local = points_utils.transform_box(
                    endpoint_box, ref_boxs[0])
                search_v2_endpoint_xy = np.asarray(
                    endpoint_local.center[:2], dtype=np.float32)
            else:
                search_v2_local_box = points_utils.transform_box(
                    search_v2_box, ref_boxs[0])
                search_v2_endpoint_xy = np.asarray(
                    search_v2_local_box.center[:2], dtype=np.float32)

        if v26_recovery_enabled:
            inner_core_pc = crop_subwindow(
                this_pc, ref_boxs[0], ref_boxs[0], scale=1.0, offset=0.0)
            b1_prior_valid = bool(
                isinstance(support_prediction, dict)
                and support_prediction.get('valid', False))
            inner_core_point_count = len(inner_core_pc.points.T)
            corridor_needed, _ = useful_search_coverage_need(
                search_v2_diagnostics.get(
                    'query_delta_t', effective_delta_t_list[0]),
                search_v2_diagnostics.get('gap_ratio', 1.0),
                search_v2_endpoint_xy,
                ref_boxs[0].wlh,
                len(baseline_search_points),
                min_delta_t=float(getattr(
                    config, 'trajectory_search_min_delta_t', 0.75)),
                min_gap_ratio=float(getattr(
                    config, 'trajectory_search_min_gap_ratio', 1.5)),
                min_endpoint_ratio=float(getattr(
                    config, 'ct_search_endpoint_ratio', 0.6)),
                sparse_base_points=int(getattr(
                    config, 'ct_search_sparse_base_points', 64)),
                inner_core_point_count=inner_core_point_count,
                min_inner_core_points=int(getattr(
                    config, 'ct_corridor_min_inner_core_points', 3)),
                b1_valid=b1_prior_valid,
                constraint_clipped=bool(search_v2_diagnostics.get(
                    'constraint_clipped', False)),
                bb_scale=float(config.bb_scale),
                bb_offset=float(config.bb_offset),
            )
            corridor_box, corridor_diagnostics = build_causal_history_corridor(
                search_v2_history_boxes,
                effective_delta_t_list,
                valid_mask,
                enabled=corridor_needed,
                first_frame_size=data['first_frame']['3d_bbox'].wlh,
                max_speed=float(getattr(
                    config, 'ct_motion_max_speed', 20.0)),
                max_acceleration=float(getattr(
                    config, 'ct_motion_max_acceleration', 8.0)),
                max_displacement=float(getattr(
                    config, 'ct_motion_max_displacement', 12.0)),
                max_length=float(getattr(
                    config, 'ct_corridor_max_length', 16.0)),
                width_padding=float(getattr(
                    config, 'ct_corridor_width_padding', 2.0)),
                max_width=float(getattr(
                    config, 'ct_corridor_max_width', 6.0)),
                **({'enable_v27': True} if v27 else {}),
            )
            if corridor_box is not None:
                corridor_pc = crop_subwindow(
                    this_pc, corridor_box, ref_boxs[0],
                    scale=1.0, offset=0.0)
                corridor_points = corridor_pc.points.T

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
    motion_anchor_theta = (
        motion_anchor_box.orientation.degrees
        * motion_anchor_box.orientation.axis[-1]
        if config.degrees else motion_anchor_box.orientation.radians
        * motion_anchor_box.orientation.axis[-1])
    motion_anchor = np.append(
        motion_anchor_box.center, motion_anchor_theta).astype('float32')

    this_box    = points_utils.transform_box(this_box, ref_boxs[0]) 
    prev_boxs   = [points_utils.transform_box(prev_box, ref_boxs[0]) for prev_box in prev_boxs] 
    ref_boxs    = [points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs]    
    motion_boxs = [points_utils.transform_box(this_box, prev_box) for prev_box in prev_boxs]  

    # Resample each frame of the point cloud to a specific number
    prev_regularized = [
        points_utils.regularize_pc(
            prev_frame_pc.points.T, config.point_sample_size, seed=seed)
        for prev_frame_pc, seed in zip(prev_frame_pcs, prev_sampling_seeds)
    ]
    prev_points_list = [item[0] for item in prev_regularized]
    this_sample_indices = None
    trajectory_search_points = np.zeros(
        (int(getattr(config, 'ct_tube_quota', 128)
             if use_ct_joint_full else getattr(
                 config, 'trajectory_search_point_count', 128)),
         baseline_search_points.shape[1]),
        dtype=np.float32,
    )
    trajectory_search_sampling = {
        'active': False,
        'sample_count': 0,
        'available_count': 0,
    }
    trajectory_search_point_valid_mask = np.zeros(
        (trajectory_search_points.shape[0],), dtype=np.float32)
    trajectory_search_point_source = np.zeros(
        (trajectory_search_points.shape[0],), dtype=np.int64)
    if use_trajectory_search:
        # Keep every baseline token exactly as in B0.  The extension is encoded
        # by a separate lightweight branch instead of stealing a fixed quota.
        this_points, this_sample_indices = points_utils.regularize_pc(
            baseline_search_points,
            config.point_sample_size,
            seed=current_sampling_seed,
        )
        if use_ct_joint_full:
            (trajectory_search_points,
             trajectory_search_point_valid_mask,
             trajectory_search_point_source,
             trajectory_search_sampling) = (
                sample_source_aware_endpoint_points(
                    baseline_search_points,
                    expanded_search_points,
                    sample_size=int(getattr(config, 'ct_tube_quota', 128)),
                    extension_quota=int(getattr(
                        config, 'ct_tube_quota', 128)),
                    min_points=int(getattr(
                        config, 'ct_search_min_points', 3)),
                    seed=current_sampling_seed,
                ))
        else:
            trajectory_search_points, trajectory_search_sampling = (
                sample_search_extension(
                    baseline_search_points,
                    expanded_search_points,
                    int(getattr(config, 'trajectory_search_point_count', 128)),
                    min_expansion_points=int(getattr(
                        config, 'trajectory_search_min_points', 16)),
                    seed=current_sampling_seed,
                ))
            trajectory_search_point_valid_mask.fill(
                float(trajectory_search_sampling['active']))
        ct_search_sampling = {
            'baseline_sample_count': int(config.point_sample_size),
            'expansion_sample_count': int(
                trajectory_search_sampling['sample_count']),
            'expansion_available_count': int(
                trajectory_search_sampling['available_count']),
        }
    elif bool(getattr(config, 'use_time_guided_search', False)):
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
        this_points, this_sample_indices = points_utils.regularize_pc(
            baseline_search_points,
            config.point_sample_size,
            seed=current_sampling_seed,
        )
        ct_search_sampling = {
            'baseline_sample_count': int(config.point_sample_size),
            'expansion_sample_count': 0,
            'expansion_available_count': 0,
        }

    search_v2_point_count = int(search_config_value('point_count', 128))
    search_v2_points = np.zeros(
        (search_v2_point_count, baseline_search_points.shape[1]),
        dtype=np.float32,
    )
    search_v2_point_valid_mask = np.zeros(
        (search_v2_point_count,), dtype=np.float32)
    search_v2_point_source = np.zeros(
        (search_v2_point_count,), dtype=np.int64)
    search_v2_sampling = {
        'active': False,
        'sample_count': 0,
        'available_count': 0,
        'extension_count': 0,
        'overlap_count': 0,
    }
    if use_endpoint_search_evidence and search_v2_box is not None:
        independent_seed_base = (
            current_sampling_seed
            if current_sampling_seed is not None else sample_index)
        search_v2_seed = (
            int(independent_seed_base) * 1664525 + 1013904223
        ) & 0xFFFFFFFF
        if (use_search_evidence_v21 or use_search_evidence_v22
                or use_search_evidence_v3):
            (search_v2_points,
             search_v2_point_valid_mask,
             search_v2_point_source,
             search_v2_sampling) = sample_source_aware_endpoint_points(
                baseline_search_points,
                search_v2_expanded_points,
                sample_size=search_v2_point_count,
                extension_quota=int(search_config_value(
                    'extension_quota', 64)),
                min_points=int(search_config_value('min_points', 3)),
                seed=search_v2_seed,
            )
        else:
            (search_v2_points,
             search_v2_point_valid_mask,
             search_v2_sampling) = sample_padded_search_extension(
                baseline_search_points,
                search_v2_expanded_points,
                sample_size=search_v2_point_count,
                min_expansion_points=int(search_config_value(
                    'min_points', 3)),
                seed=search_v2_seed,
            )
    joint_extension_sampling = None
    joint_extension_source = None
    identity_kwargs = {}
    if v27:
        empty_ids = np.empty(0, dtype=np.int64)
        identity_kwargs = dict(
            baseline_ids=raw_point_ids(this_frame_pc),
            endpoint_ids=(raw_point_ids(search_v2_expanded_pc)
                          if search_v2_box is not None else empty_ids),
            tube_ids=(raw_point_ids(expanded_search_pc)
                      if ct_search_box is not None else empty_ids),
            corridor_ids=(raw_point_ids(corridor_pc)
                          if corridor_box is not None else empty_ids),
            enable_v27=True)
    if use_search_evidence_v3 and joint_contract_v3:
        independent_seed_base = (
            current_sampling_seed
            if current_sampling_seed is not None else sample_index)
        joint_extension_seed = (
            int(independent_seed_base) * 22695477 + 1) & 0xFFFFFFFF
        if v26_recovery_enabled:
            (joint_extension_points,
             joint_extension_valid_mask,
             joint_extension_source,
             joint_extension_sampling) = sample_bounded_novel_prepool(
                baseline_search_points,
                search_v2_expanded_points,
                expanded_search_points,
                corridor_points,
                local_quota=(
                    int(getattr(config, 'ct_endpoint_quota', 256))
                    + int(getattr(config, 'ct_tube_quota', 256))),
                corridor_quota=int(getattr(
                    config, 'ct_corridor_quota', 256)),
                voxel_size=float(getattr(
                    config, 'ct_search_extension_voxel_size', 0.2)),
                **identity_kwargs,
            )
        else:
            (joint_extension_points,
             joint_extension_valid_mask,
             joint_extension_source,
             joint_extension_sampling) = sample_joint_novel_extensions(
                baseline_search_points,
                search_v2_expanded_points,
                expanded_search_points,
                endpoint_quota=int(getattr(config, 'ct_endpoint_quota', 128)),
                tube_quota=int(getattr(config, 'ct_tube_quota', 128)),
                seed=joint_extension_seed,
            )
        endpoint_quota = int(getattr(config, 'ct_endpoint_quota', 128))
        search_v2_points = joint_extension_points[:endpoint_quota]
        search_v2_point_valid_mask = joint_extension_valid_mask[
            :endpoint_quota]
        search_v2_point_source = joint_extension_source[:endpoint_quota]
        trajectory_search_points = joint_extension_points[endpoint_quota:]
        trajectory_search_point_valid_mask = joint_extension_valid_mask[
            endpoint_quota:]
        trajectory_search_point_source = joint_extension_source[endpoint_quota:]
        search_v2_sampling = {
            'active': bool(joint_extension_sampling['active']),
            'sample_count': int(search_v2_point_valid_mask.sum()),
            'available_count': int(
                joint_extension_sampling.get(
                    'endpoint_available_count',
                    joint_extension_sampling.get(
                        'local_available_count', 0))),
            'extension_count': int(search_v2_point_valid_mask.sum()),
            'overlap_count': 0,
            'selected_extension_count': int(
                search_v2_point_valid_mask.sum()),
            'selected_overlap_count': 0,
        }
        trajectory_search_sampling = {
            'active': bool(trajectory_search_point_valid_mask.any()),
            'sample_count': int(
                trajectory_search_point_valid_mask.sum()),
            'available_count': int(
                joint_extension_sampling.get(
                    'tube_available_count',
                    joint_extension_sampling.get(
                        'corridor_available_count', 0))),
        }
    joint_support = combined_search_support_statistics(
        (search_v2_points, trajectory_search_points),
        (search_v2_point_valid_mask,
         trajectory_search_point_valid_mask),
        (search_v2_point_source, trajectory_search_point_source),
        voxel_size=float(getattr(
            config, 'ct_search_extension_voxel_size', 0.2)),
        **({'point_ids': (
            joint_extension_sampling['_selected_point_ids'][:len(search_v2_points)],
            joint_extension_sampling['_selected_point_ids'][len(search_v2_points):])}
           if v27 and use_search_evidence_v3 and joint_contract_v3 else {}),
    )
    coverage_need, endpoint_ratio = useful_search_coverage_need(
        search_v2_diagnostics.get(
            'query_delta_t', effective_delta_t_list[0]),
        search_v2_diagnostics.get('gap_ratio', 1.0),
        search_v2_endpoint_xy,
        coordinate_anchor_box.wlh,
        len(baseline_search_points),
        min_delta_t=float(getattr(
            config, 'trajectory_search_min_delta_t', 0.75)),
        min_gap_ratio=float(getattr(
            config, 'trajectory_search_min_gap_ratio', 1.5)),
        min_endpoint_ratio=float(getattr(
            config, 'ct_search_endpoint_ratio', 0.6)),
        sparse_base_points=int(getattr(
            config, 'ct_search_sparse_base_points', 64)),
        inner_core_point_count=inner_core_point_count,
        min_inner_core_points=int(getattr(
            config, 'ct_corridor_min_inner_core_points', 3)),
        b1_valid=(b1_prior_valid if v26_recovery_enabled else True),
        constraint_clipped=(bool(search_v2_diagnostics.get(
            'constraint_clipped', False)) if v26_recovery_enabled else False),
        bb_scale=float(config.bb_scale),
        bb_offset=float(config.bb_offset),
    )
    recent_history_valid = bool(
        len(valid_mask) >= 2 and int(valid_mask[0]) and int(valid_mask[1]))
    query_dt_value = float(search_v2_diagnostics.get(
        'query_delta_t', effective_delta_t_list[0]))
    time_valid = bool(np.isfinite(query_dt_value) and query_dt_value > 0.0)
    proposal_valid = bool(
        search_v2_diagnostics.get('valid', False)
        and not search_v2_diagnostics.get('constraint_clipped', False)
        and np.isfinite(search_v2_endpoint_xy).all()
        and float(search_v2_diagnostics.get('displacement', 0.0))
        >= float(getattr(
            config, 'trajectory_search_min_displacement', 0.2)))
    point_support_valid = bool(
        joint_support['total_count'] >= int(getattr(
            config, 'ct_search_min_total_points', 16))
        and joint_support['extension_count'] >= int(getattr(
            config, 'ct_search_min_extension_points', 8))
        and joint_support['extension_voxels'] >= int(getattr(
            config, 'ct_search_min_extension_voxels', 4)))
    geometry_valid = bool(
        recent_history_valid and time_valid
        and search_v2_box is not None and ct_search_box is not None
        and search_v2_diagnostics.get('valid', False)
        and np.isfinite(search_v2_endpoint_xy).all())
    structural_point_valid = bool(joint_support['total_count'] >= 3)
    new_support_valid = bool(
        joint_support['extension_count'] >= 1
        and joint_support['extension_voxels'] >= 1)
    if use_search_evidence_v3 and joint_contract_v3:
        # Availability is a deterministic structural contract: valid B1
        # geometry plus at least one finite novel extension point.  No GT
        # label, learned score, density heuristic or utility estimate enters.
        causal_support_valid = bool(
            geometry_valid or (
                v26_recovery_enabled
                and corridor_diagnostics.get('valid', False)))
        search_support_valid = bool(
            causal_support_valid
            and joint_extension_sampling is not None
            and joint_extension_sampling['sample_count'] > 0)
    elif use_ct_joint_full and joint_contract_v2 and bool(getattr(
            config, 'ct_search_relaxed_validity', True)):
        search_support_valid = bool(
            geometry_valid and structural_point_valid and new_support_valid)
    else:
        search_support_valid = bool(
            recent_history_valid and time_valid and proposal_valid
            and coverage_need and point_support_valid)
    ct_search_active = ct_search_sampling['expansion_sample_count'] > 0
    num_points_in_search = int(len(baseline_search_points))
    if ct_search_active:
        num_points_in_search += int(
            ct_search_sampling['expansion_available_count'])
    search_has_usable_points = num_points_in_search > 2

    seg_label_this = geometry_utils.points_in_box(this_box, this_points.T[:3,:], config.bb_scale).astype(int)
    evidence_label_scale = 1.0 if v27 else config.bb_scale
    search_v2_point_labels = geometry_utils.points_in_box(
        this_box,
        search_v2_points.T[:3, :],
        evidence_label_scale,
    ).astype(np.float32)
    search_v2_point_labels *= search_v2_point_valid_mask
    trajectory_search_point_labels = geometry_utils.points_in_box(
        this_box,
        trajectory_search_points.T[:3, :],
        evidence_label_scale,
    ).astype(np.float32)
    trajectory_search_point_labels *= trajectory_search_point_valid_mask
    if joint_extension_sampling is not None:
        extension_pool_points = joint_extension_sampling.pop(
            '_pool_points', np.zeros(
                (0, baseline_search_points.shape[1]), dtype=np.float32))
    else:
        extension_pool_points = np.zeros(
            (0, baseline_search_points.shape[1]), dtype=np.float32)
    base_target_count = int(np.sum(seg_label_this > 0))
    expansion_regions = np.concatenate((
        search_v2_expanded_points, expanded_search_points,
        corridor_points), axis=0)
    if len(expansion_regions) and not v27:
        _, unique_expansion_indices = np.unique(
            np.rint(expansion_regions[:, :3] / 1e-6).astype(np.int64),
            axis=0, return_index=True)
        expansion_regions = expansion_regions[
            np.sort(unique_expansion_indices)]
    elif len(expansion_regions) and v27:
        raw_expansion_ids = np.concatenate((identity_kwargs['endpoint_ids'],
                                            identity_kwargs['tube_ids'],
                                            identity_kwargs['corridor_ids']))
        _, first = np.unique(raw_expansion_ids, return_index=True)
        expansion_regions = expansion_regions[np.sort(first)]
    expansion_target_count = int(np.sum(geometry_utils.points_in_box(
        this_box, expansion_regions.T[:3, :],
        evidence_label_scale))) if len(expansion_regions) else 0
    extension_pool_target_count = int(np.sum(geometry_utils.points_in_box(
        this_box, extension_pool_points.T[:3, :],
        evidence_label_scale))) if len(extension_pool_points) else 0
    extension_sampled_target_count = int(
        np.sum(search_v2_point_labels > 0)
        + np.sum(trajectory_search_point_labels > 0))
    recovery_role = int(candidate_id)
    recovery_positive = False
    recovery_fallback = False
    if candidate_id == 1:
        recovery_positive = bool(
            base_target_count <= 2
            and extension_pool_target_count > 0
            and extension_sampled_target_count > 0)
        recovery_fallback = not recovery_positive
    elif candidate_id == 2:
        recovery_positive = bool(
            base_target_count == 0
            and extension_pool_target_count > 0
            and extension_sampled_target_count > 0)
        recovery_fallback = not recovery_positive
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
    trajectory_timestamp = np.full(
        (trajectory_search_points.shape[0], 1),
        fill_value=main_current_value,
        dtype=np.float32,
    )
    trajectory_prior = np.full(
        (trajectory_search_points.shape[0], 1),
        fill_value=0.5,
        dtype=np.float32,
    )
    trajectory_search_points = np.concatenate(
        (trajectory_search_points, trajectory_timestamp, trajectory_prior),
        axis=-1,
    )
    search_v2_timestamp = np.full(
        (search_v2_points.shape[0], 1),
        fill_value=main_current_value,
        dtype=np.float32,
    )
    search_v2_prior = np.full(
        (search_v2_points.shape[0], 1),
        fill_value=0.5,
        dtype=np.float32,
    )
    search_v2_points = np.concatenate(
        (search_v2_points, search_v2_timestamp, search_v2_prior),
        axis=-1,
    )

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
    if recursive_replay is not None:
        current_delta_t_effective = float(
            recursive_replay['current_delta_t'])
    # Supervision is always defined in physical time. A fixed/shuffled negative
    # control may alter only the time consumed by DynamicsEncoder.
    dynamics_displacement_label, velocity_label = canonical_dynamics_targets(
        canonical_label_boxs,
        canonical_this_box,
        current_delta_t_real,
        degrees=config.degrees,
        eps=1e-3,
    )
    trajectory_displacement_label, trajectory_velocity_label = (
        anchor_relative_trajectory_targets(
            canonical_this_box,
            coordinate_anchor_box,
            current_delta_t_real,
            degrees=config.degrees,
            eps=1e-3,
        ))
    motion_main_target_xy = None
    if use_motion_v3:
        if joint_contract_v2:
            motion_main_physical = build_b1_physical_contract(
                canonical_this_box,
                ground_truth_history,
                recursive_history,
                current_delta_t_real,
                degrees=config.degrees,
                eps=1e-3,
            )
            same_main_axes = (np.allclose(
                motion_main_ref_boxs, motion_main_physical['ref_boxs'],
                rtol=1e-6, atol=1e-6) if v27 else np.array_equal(
                    motion_main_ref_boxs, motion_main_physical['ref_boxs']))
            if not same_main_axes:
                raise RuntimeError(
                    "B1 main input and physical-label axes diverged")
            motion_main_target_xy = motion_main_physical['target_xy']
        else:
            motion_main_target_xy, _ = physical_motion_targets(
                canonical_this_box,
                legacy_canonical_prev_boxs[0],
                coordinate_anchor_box,
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
        'motion_state_label': np.stack(motion_state_label_list, axis=0).astype(
            np.int64 if v27 else 'int'),
        'bbox_size': (
            np.asarray(data['first_frame']['3d_bbox'].wlh, dtype=np.float32)
            if use_ct_joint_full or bool(getattr(
                config, 'observation_safe_bbox_size', False))
            else this_box.wlh),
        'seg_label': stack_seg_label.astype(np.int64 if v27 else 'int'),
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
        'ct_search_used': np.float32(
            search_support_valid if use_ct_joint_full else ct_search_active),
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
        'ct_search_support_valid': np.float32(search_support_valid),
        'ct_search_geometry_valid': np.float32(geometry_valid),
        'ct_search_structural_point_valid': np.float32(
            structural_point_valid),
        'ct_search_new_support_valid': np.float32(new_support_valid),
        'ct_search_quality_valid': np.float32(point_support_valid),
        # Temporary compatibility alias; new code must use the explicit name.
        'candidate_valid': np.float32(search_support_valid),
        'ct_search_history_valid': np.float32(recent_history_valid),
        'ct_search_time_valid': np.float32(time_valid),
        'ct_search_proposal_valid': np.float32(proposal_valid),
        'ct_search_point_support_valid': np.float32(point_support_valid),
        'ct_search_coverage_need': np.float32(coverage_need),
        'ct_search_endpoint_ratio': np.float32(endpoint_ratio),
        'ct_search_total_point_count': np.float32(
            joint_support['total_count']),
        'ct_search_extension_count': np.float32(
            joint_support['extension_count']),
        'ct_search_extension_voxels': np.float32(
            joint_support['extension_voxels']),
        'ct_corridor_valid': np.float32(
            bool(corridor_diagnostics.get('valid', False))),
        'ct_corridor_constraint_clipped': np.float32(
            bool(corridor_diagnostics.get('constraint_clipped', False))),
        'ct_corridor_source_id': np.int64(
            corridor_diagnostics.get('source_id', 0)),
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
    if use_search_evidence_v3 and joint_contract_v3:
        extension_points = np.concatenate(
            (search_v2_points, trajectory_search_points), axis=0)
        extension_labels = np.concatenate((
            search_v2_point_labels,
            trajectory_search_point_labels,
        ), axis=0)
        extension_valid_mask = np.concatenate((
            search_v2_point_valid_mask,
            trajectory_search_point_valid_mask,
        ), axis=0)
        if joint_extension_source is None:
            raise RuntimeError("contract-v3 extension source was not built")
        data_dict.update({
            # This is the exact current-frame tensor appended to ``points``;
            # no independent crop or resampling is allowed.
            'ct_base_evidence_points': this_points.astype('float32'),
            'ct_base_evidence_labels': seg_label_this.astype('float32'),
            'ct_base_evidence_valid_mask': np.ones(
                (config.point_sample_size,), dtype=np.float32),
            'ct_extension_points': extension_points.astype('float32'),
            'ct_extension_labels': extension_labels.astype('float32'),
            'ct_extension_valid_mask':
                extension_valid_mask.astype('float32'),
            # v25: 0/1/2/3. v26 bitmask: endpoint=1,tube=2,corridor=4.
            'ct_extension_source': joint_extension_source.astype('int64'),
            'ct_acquisition_base_target_count': np.float32(
                base_target_count),
            'ct_acquisition_expansion_target_count': np.float32(
                expansion_target_count),
            'ct_acquisition_extension_pool_target_count': np.float32(
                extension_pool_target_count),
            'ct_acquisition_sampled_target_count': np.float32(
                extension_sampled_target_count),
            'ct_acquisition_extension_pool_count': np.float32(
                joint_extension_sampling['available_count']),
            'ct_acquisition_sampled_count': np.float32(
                joint_extension_sampling['sample_count']),
            'ct_view_role': np.int64(0 if candidate_id == 0 else 1),
            # 0=identity observation, 1=weak recovery-positive,
            # 2=strict miss (possibly explicit fallback), 3=natural control.
            'ct_recovery_role': np.int64(recovery_role),
            'ct_recovery_positive': np.float32(recovery_positive),
            'ct_recovery_fallback': np.float32(recovery_fallback),
        })
        if v26_recovery_enabled:
            data_dict.update({
                'ct_extension_prepool_points':
                    extension_points.astype('float32'),
                'ct_extension_prepool_labels':
                    extension_labels.astype('float32'),
                'ct_extension_prepool_valid_mask':
                    extension_valid_mask.astype('float32'),
                'ct_extension_prepool_source':
                    joint_extension_source.astype('int64'),
            })
    if (use_trajectory_search
            or bool(getattr(config, 'use_ordered_trajectory_encoder', False))):
        data_dict.update({
            'trajectory_displacement_label':
                trajectory_displacement_label.astype('float32'),
            'trajectory_velocity_label':
                trajectory_velocity_label.astype('float32'),
            'trajectory_search_points':
                trajectory_search_points.astype('float32'),
            'trajectory_search_point_labels':
                trajectory_search_point_labels.astype('float32'),
            'trajectory_search_point_valid_mask':
                trajectory_search_point_valid_mask.astype('float32'),
            'trajectory_search_point_source':
                trajectory_search_point_source.astype('int64'),
            # Stable branch contract: 0=baseline, 1=endpoint, 2=tube.
            'trajectory_search_branch_source': np.full(
                trajectory_search_point_valid_mask.shape, 2, dtype=np.int64),
            'trajectory_search_valid': np.float32(
                search_support_valid if use_ct_joint_full
                else trajectory_search_sampling['active']),
            'trajectory_search_gap_ratio': np.float32(
                ct_search_diagnostics.get('gap_ratio', 1.0)),
            'trajectory_search_sigma_parallel': np.float32(
                ct_search_diagnostics.get('sigma_parallel', 0.0)),
            'trajectory_search_sigma_perpendicular': np.float32(
                ct_search_diagnostics.get('sigma_perpendicular', 0.0)),
        })
    if use_search_evidence_v2:
        data_dict.update({
            'search_v2_points': search_v2_points.astype('float32'),
            'search_v2_point_valid_mask':
                search_v2_point_valid_mask.astype('float32'),
            'search_v2_point_labels':
                search_v2_point_labels.astype('float32'),
            'search_v2_geometry_valid': np.float32(
                search_v2_sampling['active']),
            'search_v2_endpoint_xy':
                search_v2_endpoint_xy.astype('float32'),
            'search_v2_query_delta_t': np.float32(
                search_v2_diagnostics.get(
                    'query_delta_t', effective_delta_t_list[0])),
            'search_v2_gap_ratio': np.float32(
                search_v2_diagnostics.get('gap_ratio', 1.0)),
            'search_v2_sigma_parallel': np.float32(
                search_v2_diagnostics.get('sigma_parallel', 0.0)),
            'search_v2_sigma_perpendicular': np.float32(
                search_v2_diagnostics.get('sigma_perpendicular', 0.0)),
            'search_v2_available_count': np.float32(
                search_v2_sampling['available_count']),
        })
    if use_search_evidence_v21:
        data_dict.update({
            'search_v21_points': search_v2_points.astype('float32'),
            'search_v21_point_valid_mask':
                search_v2_point_valid_mask.astype('float32'),
            'search_v21_point_source':
                search_v2_point_source.astype('int64'),
            'search_v21_point_labels':
                search_v2_point_labels.astype('float32'),
            'search_v21_geometry_valid': np.float32(
                search_v2_sampling['active']),
            'search_v21_endpoint_xy':
                search_v2_endpoint_xy.astype('float32'),
            'search_v21_query_delta_t': np.float32(
                search_v2_diagnostics.get(
                    'query_delta_t', effective_delta_t_list[0])),
            'search_v21_gap_ratio': np.float32(
                search_v2_diagnostics.get('gap_ratio', 1.0)),
            'search_v21_sigma_parallel': np.float32(
                search_v2_diagnostics.get('sigma_parallel', 0.0)),
            'search_v21_sigma_perpendicular': np.float32(
                search_v2_diagnostics.get('sigma_perpendicular', 0.0)),
            'search_v21_available_count': np.float32(
                search_v2_sampling['available_count']),
            'search_v21_extension_count': np.float32(
                search_v2_sampling['extension_count']),
            'search_v21_overlap_count': np.float32(
                search_v2_sampling['overlap_count']),
        })
    if use_search_evidence_v22:
        data_dict.update({
            'search_v22_points': search_v2_points.astype('float32'),
            'search_v22_point_valid_mask':
                search_v2_point_valid_mask.astype('float32'),
            'search_v22_point_source':
                search_v2_point_source.astype('int64'),
            'search_v22_point_labels':
                search_v2_point_labels.astype('float32'),
            'search_v22_geometry_valid': np.float32(
                search_v2_sampling['active']),
            'search_v22_support_anchor_xy':
                search_v2_endpoint_xy.astype('float32'),
            'search_v22_query_delta_t': np.float32(
                search_v2_diagnostics.get(
                    'query_delta_t', effective_delta_t_list[0])),
            'search_v22_gap_ratio': np.float32(
                search_v2_diagnostics.get('gap_ratio', 1.0)),
            'search_v22_sigma_parallel': np.float32(
                search_v2_diagnostics.get('sigma_parallel', 0.0)),
            'search_v22_sigma_perpendicular': np.float32(
                search_v2_diagnostics.get('sigma_perpendicular', 0.0)),
            'search_v22_available_count': np.float32(
                search_v2_sampling['available_count']),
            'search_v22_extension_count': np.float32(
                search_v2_sampling['extension_count']),
            'search_v22_overlap_count': np.float32(
                search_v2_sampling['overlap_count']),
        })
    if use_search_evidence_v3:
        v3_query_delta_t = np.float32(search_v2_diagnostics.get(
            'query_delta_t', effective_delta_t_list[0]))
        if v3_query_delta_t != np.float32(effective_delta_t_list[0]):
            raise RuntimeError(
                "B2-v3 query delta_t diverged from the shared B1 clock")
        data_dict.update({
            'search_v3_points': search_v2_points.astype('float32'),
            'search_v3_point_valid_mask':
                search_v2_point_valid_mask.astype('float32'),
            'search_v3_point_source':
                search_v2_point_source.astype('int64'),
            'search_v3_branch_source': np.ones(
                search_v2_point_valid_mask.shape, dtype=np.int64),
            'search_v3_point_labels':
                search_v2_point_labels.astype('float32'),
            'search_v3_geometry_valid': np.float32(
                geometry_valid if joint_contract_v2 else search_support_valid),
            'search_v3_support_valid': np.float32(search_support_valid),
            'search_v3_total_point_count': np.float32(
                joint_support['total_count']),
            'search_v3_joint_extension_count': np.float32(
                joint_support['extension_count']),
            'search_v3_extension_voxels': np.float32(
                joint_support['extension_voxels']),
            'search_v3_endpoint_ratio': np.float32(endpoint_ratio),
            'search_v3_support_anchor_xy':
                search_v2_endpoint_xy.astype('float32'),
            'search_v3_query_delta_t': v3_query_delta_t,
            'search_v3_gap_ratio': np.float32(
                search_v2_diagnostics.get('gap_ratio', 1.0)),
            'search_v3_sigma_parallel': np.float32(
                search_v2_diagnostics.get('sigma_parallel', 0.0)),
            'search_v3_sigma_perpendicular': np.float32(
                search_v2_diagnostics.get('sigma_perpendicular', 0.0)),
            'search_v3_available_count': np.float32(
                search_v2_sampling['available_count']),
            'search_v3_extension_count': np.float32(
                search_v2_sampling['extension_count']),
            'search_v3_overlap_count': np.float32(
                search_v2_sampling['overlap_count']),
            'search_v3_prior_source_id': np.int64(
                search_v2_diagnostics.get('source_id', 0)),
            'search_v3_support_truncated': np.float32(
                bool(search_v2_diagnostics.get('truncated', False))),
            'search_v3_support_requested_extent': np.asarray((
                search_v2_diagnostics.get('requested_length', 0.0),
                search_v2_diagnostics.get('requested_width', 0.0),
            ), dtype=np.float32),
            'search_v3_support_actual_extent': np.asarray((
                search_v2_diagnostics.get('length', 0.0),
                search_v2_diagnostics.get('width', 0.0),
            ), dtype=np.float32),
            'b2_v3_history_ref_boxs':
                motion_main_ref_boxs.astype('float32'),
            'b2_v3_history_valid_mask':
                np.asarray(valid_mask, dtype=np.int64),
            'b2_v3_history_delta_t':
                np.asarray(effective_delta_t_list, dtype=np.float32),
            'b2_v3_history_mode_id': np.int64(
                b2_v3_history_mode_id(b2_v3_history_mode)),
            'b2_v3_history_anchor': (
                motion_anchor if joint_contract_v2 else coordinate_anchor),
        })
        if replay_b1 is not None:
            data_dict.update({
                'replay_b1_contract_present': np.float32(1.0),
                'replay_b1_mu_xy': np.asarray(
                    replay_b1['mu_xy'], dtype=np.float32),
                'replay_b1_direction_xy': np.asarray(
                    replay_b1['direction_xy'], dtype=np.float32),
                'replay_b1_log_sigma_parallel_perp': np.asarray(
                    replay_b1['log_sigma_parallel_perp'], dtype=np.float32),
                'replay_b1_gap_ratio': np.float32(
                    replay_b1['gap_ratio']),
                'replay_b1_valid': np.float32(
                    bool(replay_b1['valid'])),
            })
    if use_motion_v3:
        data_dict.update({
            'motion_main_ref_boxs': motion_main_ref_boxs.astype('float32'),
            'motion_main_delta_t': np.asarray(
                effective_delta_t_list, dtype=np.float32),
            'motion_main_current_delta_t': np.float32(
                current_delta_t_effective),
            'motion_main_valid_mask': np.asarray(valid_mask, dtype=np.int64),
            'motion_main_target_xy': motion_main_target_xy.astype('float32'),
            'motion_main_anchor': (
                motion_anchor if joint_contract_v2 else coordinate_anchor),
            'motion_source_anchor': (
                motion_anchor if joint_contract_v2 else coordinate_anchor),
        })
        if motion_aux_contract is not None:
            data_dict.update(motion_aux_contract)
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

    if v27:
        import hashlib
        margin_target = None
        identities = [sampled_identity(pc, result[1], config.point_sample_size)
                      for pc, result in zip(prev_frame_pcs, prev_regularized)]
        identities.append(sampled_identity(this_frame_pc, this_sample_indices,
                                           config.point_sample_size))
        data_dict.update({
            'b0_point_ids': np.stack([item[0] for item in identities]),
            'b0_point_valid_mask': np.stack([item[1] for item in identities]),
            'b0_point_unique_mask': np.stack([item[2] for item in identities]),
            'b0_valid_mask': np.stack([item[1] for item in identities]),
            'b0_unique_mask': np.stack([item[2] for item in identities]),
            'b0_raw_point_count': np.asarray(
                [len(pc.points.T) for pc in prev_frame_pcs] +
                [len(baseline_search_points)], dtype=np.int64),
            'b0_frame_uid': np.asarray([int.from_bytes(hashlib.sha256(str(
                frame.get('frame_id', frame.get('timestamp', ''))).encode()).digest()[:8],
                'big') & ((1 << 63) - 1) for frame in
                [prev_frames[key] for key in sorted_prev_keys] + [this_frame]], dtype=np.int64),
            'ct_current_observation_valid': np.float32(len(baseline_search_points) > 0),
            'ct_recursive_state_age': np.float32(
                (online_recursive_state or {}).get('recursive_age', 0.)),
            'target_bbox_size': np.asarray(canonical_this_box.wlh, dtype=np.float32),
            'timestamps_effective': np.asarray(effective_local_timestamps, dtype=np.float32),
        })
        if v27_b1_input is not None:
            from utils.b1_acquisition import b1_input_digest, acquisition_margin_grid_target
            data_dict['motion_acquisition_features'] = v27_b1_input['acquisition_features']
            data_dict['motion_input_digest'] = b1_input_digest(v27_b1_input)
            data_dict['motion_main_ref_boxs'] = v27_b1_input['ref_boxs']
            data_dict['motion_acquisition_target'] = np.asarray((2., 1.), dtype=np.float32)
            data_dict['motion_acquisition_target_valid'] = np.float32(0.)
            # 不同slot会同时包含首个query和成熟历史。没有合法获取几何时
            # 仍返回固定字段，target_valid=0屏蔽监督，不能让合批取决于首行。
            margin_count_keys = ('global_novel_target_count', 'max_reachable_target_count',
                                 'selected_target_count', 'selected_background_count')
            for key in margin_count_keys:
                data_dict['motion_margin_' + key] = np.float32(0.)
            if not data.get('_ct_inference', False) and search_v2_box is not None:
                margin_target = acquisition_margin_grid_target(
                    this_pc.points.T, raw_point_ids(this_pc),
                    geometry_utils.points_in_box(canonical_this_box, this_pc.points, 1.0),
                    raw_point_ids(this_frame_pc),
                    anchor_center=coordinate_anchor_box.center,
                    endpoint_center=search_v2_diagnostics['endpoint_center'],
                    object_wlh=data['first_frame']['3d_bbox'].wlh,
                    object_yaw=float(coordinate_anchor_box.orientation.radians *
                                     coordinate_anchor_box.orientation.axis[-1]),
                    corridor_box=corridor_box)
                data_dict['motion_acquisition_target'] = margin_target['target_margin']
                data_dict['motion_acquisition_target_valid'] = np.float32(margin_target['valid'])
                for key in margin_count_keys:
                    data_dict['motion_margin_' + key] = np.float32(margin_target[key])
        if use_search_evidence_v3 and joint_contract_v3:
            data_dict['ct_extension_point_ids'] = np.asarray(
                joint_extension_sampling['_selected_point_ids'], dtype=np.int64)
            data_dict['ct_base_evidence_valid_mask'] = identities[-1][1]
            data_dict['ct_base_evidence_unique_mask'] = identities[-1][2]
            data_dict['ct_base_evidence_point_ids'] = identities[-1][0]
            data_dict['ct_base_point_ids'] = identities[-1][0]
            data_dict['ct_base_unique_mask'] = identities[-1][2]
            evidence_labels = geometry_utils.points_in_box(
                this_box, extension_points[:, :3].T, 1.0).astype(np.float32)
            evidence_labels *= extension_valid_mask
            data_dict['ct_extension_labels'] = evidence_labels
            data_dict['ct_extension_prepool_labels'] = evidence_labels.copy()
            data_dict['ct_base_evidence_labels'] = geometry_utils.points_in_box(
                this_box, this_points[:, :3].T, 1.0).astype(np.float32) * identities[-1][1]
            data_dict.update({
                'ct_acquisition_base_target_count': np.float32(np.sum(
                    geometry_utils.points_in_box(this_box, this_frame_pc.points[:3], 1.0))),
                'ct_acquisition_expansion_target_count': np.float32(expansion_target_count),
                'ct_acquisition_extension_pool_target_count': np.float32(extension_pool_target_count),
                'ct_acquisition_sampled_target_count': np.float32(np.sum(evidence_labels)),
            })
            source_prediction = support_prediction or {}
            learned_acquisition = search_v2_diagnostics.get('prior_source') == 'b1'
            actual_margin = (source_prediction.get('acquisition_margin_parallel_perp', (2., 1.))
                             if learned_acquisition else (2., 1.))
            acquisition_yaw = (float(search_v2_box.orientation.radians *
                                     search_v2_box.orientation.axis[-1])
                               if search_v2_box is not None else float(coordinate_anchor_theta))
            anchor_yaw = float(coordinate_anchor_theta)
            if config.degrees:
                anchor_yaw = np.deg2rad(anchor_yaw)
            angle = acquisition_yaw - anchor_yaw
            acquisition_direction = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float32)
            if learned_acquisition:
                statistical_direction = np.asarray(source_prediction['direction_xy'], dtype=np.float32)
                statistical_log_sigma = np.asarray(source_prediction['log_sigma_parallel_perp'], dtype=np.float32)
            else:
                # The resolved CV prior owns these statistics.  An invalid
                # learned prior may contain NaNs and cannot enter B2 features.
                statistical_direction = acquisition_direction.copy()
                coverage_scale = max(float(getattr(config, 'search_v3_coverage_scale', 2.448)), 1e-6)
                fallback_sigma = np.asarray((
                    search_v2_diagnostics.get('sigma_parallel', 2. / coverage_scale),
                    search_v2_diagnostics.get('sigma_perpendicular', 1. / coverage_scale)), dtype=np.float32)
                fallback_sigma = np.where(np.isfinite(fallback_sigma) & (fallback_sigma > 0),
                                          fallback_sigma, np.asarray((2., 1.)) / coverage_scale)
                statistical_log_sigma = np.log(fallback_sigma).astype(np.float32)
            data_dict.update({
                'ct_acquisition_direction_xy': acquisition_direction,
                'ct_acquisition_margin': np.asarray(actual_margin, dtype=np.float32),
                'ct_acquisition_statistical_direction_xy': statistical_direction,
                'ct_acquisition_log_sigma': statistical_log_sigma,
                'ct_acquisition_learned_valid': np.float32(source_prediction.get('valid', False)),
                'ct_acquisition_resolved_valid': np.float32(search_v2_diagnostics.get('valid', False)),
                'ct_acquisition_fallback_reason': str(search_v2_diagnostics.get('fallback_reason',
                    search_v2_diagnostics.get('reason', ''))),
                'ct_acquisition_parameter_revision': np.int64(source_prediction.get('parameter_revision', 0)),
            })
            sidecar = data.get('_ct_diagnostic_sidecar')
            if sidecar is not None:
                sidecar['_construction'] = dict(
                    global_pc=this_pc, gt_box=canonical_this_box, anchor_box=coordinate_anchor_box,
                    baseline_pc=this_frame_pc, sampled_base_xyz=this_points[:, :3],
                    sampled_base_ids=identities[-1][0],
                    endpoint_pc=search_v2_expanded_pc if search_v2_box is not None else None,
                    tube_pc=expanded_search_pc if ct_search_box is not None else None,
                    corridor_pc=corridor_pc if corridor_box is not None else None,
                    pool_xyz=extension_pool_points,
                    pool_ids=joint_extension_sampling['_pool_point_ids'],
                    prepool_xyz=extension_points[:, :3],
                    prepool_ids=data_dict['ct_extension_point_ids'],
                    prepool_valid=extension_valid_mask, source=joint_extension_source,
                    support_boxes=(search_v2_box, ct_search_box, corridor_box))
                from utils.v27_diagnostics import build_acquisition_diagnostics
                sidecar['acquisition'] = build_acquisition_diagnostics(
                    **sidecar.pop('_construction'), margin_target=margin_target)
                data_dict.update({
                    'ct_acquisition_base_target_count': np.float32(sidecar['acquisition']['base_raw_target_count']),
                    'ct_acquisition_expansion_target_count': np.float32(sidecar['acquisition']['expansion_target_count']),
                    'ct_acquisition_extension_pool_target_count': np.float32(sidecar['acquisition']['pool_target_count']),
                    'ct_acquisition_sampled_target_count': np.float32(sidecar['acquisition']['sampled_target_count']),
                })
        sidecar = data.get('_ct_diagnostic_sidecar')
        if isinstance(sidecar, dict) and 'acquisition' not in sidecar:
            # Observation-only arms still report the complete raw/base funnel.
            # GT is read exclusively in this sidecar and does not alter B0 labels.
            from utils.v27_diagnostics import build_acquisition_diagnostics
            empty_xyz = np.empty((0, 3), dtype=np.float32)
            empty_ids = np.empty(0, dtype=np.int64)
            sidecar['acquisition'] = build_acquisition_diagnostics(
                this_pc, canonical_this_box, coordinate_anchor_box, this_frame_pc,
                this_points[:, :3], identities[-1][0], None, None, None,
                empty_xyz, empty_ids, empty_xyz, empty_ids,
                np.empty(0, dtype=np.float32), empty_ids)
            sidecar['acquisition'].update(
                acquisition_enabled=False,
                acquisition_disabled_reason=('observation_only' if observation_only else 'modules_disabled'),
                selected_point_count=0, selected_target_count=0,
                selected_background_count=0, selected_target_bearing=0)

    if str(getattr(
            config, 'ct_observation_payload_mode', 'legacy'
            )).strip().lower() == 'seqtrack_core':
        data_dict = prune_seqtrack_observation_payload(data_dict)

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
        self.epoch = 0
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

    def set_epoch(self, epoch):
        """Select the stateless observation RNG domain for one epoch."""
        self.epoch = int(epoch)

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


class PartitionedTestTrackingSampler(TestTrackingSampler):
    """Evaluation view over one atomic tracklet hash partition."""

    def __init__(self, dataset, config=None, partition='dev', **kwargs):
        super().__init__(dataset, config=config, **kwargs)
        self.partition = str(partition)
        seed = int(getattr(
            self.config, 'ct_partition_seed',
            getattr(self.config, 'seed', 42)) or 42)
        self.tracklet_indices = []
        for tracklet_id in range(dataset.get_num_tracklets()):
            key = (
                dataset.get_tracklet_key(tracklet_id)
                if hasattr(dataset, 'get_tracklet_key') else str(tracklet_id))
            if stable_tracklet_partition(key, seed) == self.partition:
                self.tracklet_indices.append(tracklet_id)
        if not self.tracklet_indices:
            raise ValueError(
                f"tracklet partition {self.partition!r} is empty")

    def __len__(self):
        return len(self.tracklet_indices)

    def __getitem__(self, index):
        tracklet_id = self.tracklet_indices[index]
        frame_count = self.dataset.get_num_frames_tracklet(tracklet_id)
        return self.dataset.get_frames(
            tracklet_id, list(range(frame_count)))

    def get_tracklet_key(self, index):
        tracklet_id = self.tracklet_indices[index]
        if hasattr(self.dataset, 'get_tracklet_key'):
            return self.dataset.get_tracklet_key(tracklet_id)
        return str(tracklet_id)


def online_recursive_collate(items):
    """Keep raw point-cloud objects out of PyTorch's tensor collator."""
    return items


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
        self.use_b4_paired_views = bool(getattr(
            self.config, 'use_b4_paired_views', False))
        self.use_paired_history = self.use_b4_paired_views
        self.use_b1motion_v3 = bool(getattr(
            self.config, 'use_b1motion_v3', False))
        self.use_recursive_replay_cache = bool(getattr(
            self.config, 'use_recursive_replay_cache', False))
        self.online_recursive_training = bool(getattr(
            self.config, 'ct_online_recursive_training', False))
        self.recursive_replay_cache = None
        if self.use_recursive_replay_cache:
            cache_dir = getattr(
                self.config, 'recursive_replay_cache_dir', None)
            if not cache_dir:
                raise ValueError(
                    "use_recursive_replay_cache requires "
                    "recursive_replay_cache_dir")
            expected_manifest = {
                'dataset': str(getattr(
                    self.config, 'dataset', 'unknown')),
                'split': str(getattr(
                    self.config, 'train_split', 'train')),
                'replay_config_sha256': replay_config_sha256(self.config),
            }
            self.recursive_replay_cache = RecursiveReplayCache(
                cache_dir, expected_manifest=expected_manifest)
        self.recursive_replay_require_all = bool(getattr(
            self.config, 'recursive_replay_require_all', True))
        self.candidate_trajectory_mode = normalize_candidate_trajectory_mode(
            getattr(self.config, 'candidate_trajectory_mode', 'independent'))
        if self.online_recursive_training:
            if self.use_recursive_replay_cache or self.use_paired_history:
                raise ValueError(
                    "online recursive training is incompatible with replay/paired views")
            if self.candidate_trajectory_mode != 'shared_se2':
                raise ValueError(
                    "online recursive training requires shared_se2 candidates")
            if int(self.num_candidates) != int(getattr(
                    self.config, 'ct_recursive_candidate_views', 4)):
                raise ValueError(
                    "num_candidates must match ct_recursive_candidate_views")
            if int(self.num_candidates) != int(getattr(
                    self.config, 'ct_b0_candidate_views', 1)):
                raise ValueError(
                    "num_candidates must match ct_b0_candidate_views")
        self.trajectory_training_irregular_probability = float(getattr(
            self.config, 'trajectory_training_irregular_probability', 0.0))
        self.trajectory_training_query_gaps = [
            int(value) for value in getattr(
                self.config, 'trajectory_training_query_gaps', [2, 4])
        ]
        self.trajectory_training_transition_gaps = [
            int(value) for value in getattr(
                self.config, 'trajectory_training_transition_gaps',
                [1, 1, 2, 4])
        ]
        if not 0.0 <= self.trajectory_training_irregular_probability <= 1.0:
            raise ValueError(
                "trajectory_training_irregular_probability must be in [0,1]")
        if (any(value <= 0 for value in self.trajectory_training_query_gaps)
                or any(value <= 0 for value in
                       self.trajectory_training_transition_gaps)):
            raise ValueError("trajectory training gaps must be positive")
        if (self.trajectory_training_irregular_probability > 0
                and not self.trajectory_training_query_gaps):
            raise ValueError(
                "irregular trajectory training requires query gap choices")
        self.motion_v3_aux_query_gaps = [
            int(value) for value in getattr(
                self.config, 'motion_v3_aux_query_gaps', [2, 4])
        ]
        self.motion_v3_aux_transition_gaps = [
            int(value) for value in getattr(
                self.config, 'motion_v3_aux_transition_gaps', [1, 2])
        ]
        if self.use_b1motion_v3:
            if self.use_paired_history:
                raise ValueError(
                    "B1motion-v3 box-only auxiliary training is incompatible "
                    "with paired point-cloud views")
            if self.trajectory_training_irregular_probability != 0.0:
                raise ValueError(
                    "B1motion-v3 keeps the main view continuous; set legacy "
                    "trajectory_training_irregular_probability to zero")
            if (not self.motion_v3_aux_query_gaps
                    or any(value <= 0 for value in
                           self.motion_v3_aux_query_gaps)
                    or any(value <= 0 for value in
                           self.motion_v3_aux_transition_gaps)):
                raise ValueError("B1motion-v3 auxiliary gaps must be positive")
        self.paired_candidate_zero_only = bool(getattr(
            self.config, 'b4_candidate_zero_only', True))
        default_b4_b_offsets = [
            1 + 2 * i for i in range(self.dataset.hist_num)]
        self.b4_view_a_offsets = list(getattr(
            self.config, 'b4_view_a_offsets',
            list(range(1, self.dataset.hist_num + 1))))
        self.b4_view_b_offsets = list(getattr(
            self.config, 'b4_view_b_offsets', default_b4_b_offsets))
        if self.use_paired_history and getattr(self.config, "use_augmentation", False):
            raise ValueError(
                "Paired history views require explicit shared transforms; "
                "keep use_augmentation=False.")
        if self.use_paired_history:
            if (len(self.b4_view_a_offsets) != self.dataset.hist_num
                    or len(self.b4_view_b_offsets) != self.dataset.hist_num):
                raise ValueError(
                    "Each paired history view must provide exactly hist_num offsets.")
            if self.b4_view_a_offsets[0] != 1 or self.b4_view_b_offsets[0] != 1:
                raise ValueError(
                    "Paired views must share the nearest t-1 anchor.")

    def _locate_tracklet(self, anno_id):
        for i in range(0, self.dataset.get_num_tracklets()):
            if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                return i, anno_id - self.tracklet_start_ids[i]
        raise IndexError(f"anno_id {anno_id} is outside tracklet ranges.")

    def _sample_history_offsets(self):
        normal = list(range(1, self.dataset.hist_num + 1))
        if self.trajectory_training_irregular_probability <= 0:
            return normal
        if torch.rand(()) >= self.trajectory_training_irregular_probability:
            return normal
        query_index = torch.randint(
            0, len(self.trajectory_training_query_gaps), size=(1,)).item()
        query_gap = self.trajectory_training_query_gaps[query_index]
        if len(self.trajectory_training_transition_gaps) > 1:
            start = torch.randint(
                0,
                len(self.trajectory_training_transition_gaps),
                size=(1,),
            ).item()
            transition_gaps = (
                self.trajectory_training_transition_gaps[start:]
                + self.trajectory_training_transition_gaps[:start]
            )
        else:
            transition_gaps = self.trajectory_training_transition_gaps
        return build_irregular_history_offsets(
            self.dataset.hist_num, query_gap, transition_gaps)

    def _motion_v3_aux_offsets(self, sample_index):
        return build_alternating_aux_history_offsets(
            self.dataset.hist_num,
            sample_index,
            query_gaps=self.motion_v3_aux_query_gaps,
            transition_gaps=self.motion_v3_aux_transition_gaps,
        )

    def _build_view(self, tracklet_id, this_frame_id, first_frame, this_frame,
                    candidate_id, offsets, candidate_offset_map=None,
                    point_sampling_seed_map=None, current_sampling_seed=None,
                    candidate_shared_transform=None,
                    motion_aux_offsets=None, sample_index=0):
        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
            this_frame_id, self.dataset.hist_num, offsets=offsets)
        prev_frames_tuple = self.dataset.get_frames(tracklet_id, frame_ids=prev_frame_ids)
        prev_frames_dict = create_history_frame_dict(prev_frames_tuple)
        tracklet_key = (
            self.dataset.get_tracklet_key(tracklet_id)
            if hasattr(self.dataset, "get_tracklet_key")
            else str(tracklet_id))
        data = {
            "first_frame": first_frame,
            "prev_frames": prev_frames_dict,
            "this_frame": this_frame,
            "candidate_id": candidate_id,
            "valid_mask": valid_mask,
            "prev_frame_ids": prev_frame_ids,
            "this_frame_id": this_frame_id,
            "history_offsets": offsets,
            "sample_index": int(sample_index),
            "tracklet_id": int(tracklet_id),
            "tracklet_key": tracklet_key,
        }
        if self.recursive_replay_cache is not None:
            replay = self.recursive_replay_cache.get(
                tracklet_key, this_frame_id)
            if replay is None and self.recursive_replay_require_all:
                raise KeyError(
                    f"recursive replay missing {tracklet_key} frame "
                    f"{this_frame_id}")
            if replay is not None:
                data["recursive_replay"] = replay
        if motion_aux_offsets is not None:
            motion_aux_frame_ids, motion_aux_valid_mask = (
                get_history_frame_ids_and_masks(
                    this_frame_id,
                    self.dataset.hist_num,
                    offsets=motion_aux_offsets,
                ))
            motion_aux_frames = self.dataset.get_frames(
                tracklet_id, frame_ids=motion_aux_frame_ids)
            data.update({
                "motion_aux_prev_frames": create_history_frame_dict(
                    motion_aux_frames),
                "motion_aux_valid_mask": motion_aux_valid_mask,
                "motion_aux_frame_ids": motion_aux_frame_ids,
                "motion_aux_offsets": motion_aux_offsets,
            })
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

    def _online_raw_view(
            self, epoch, batch_index, slot, tracklet_id, this_frame_id,
            candidate_id, build_shadow=False):
        offsets = list(range(1, self.dataset.hist_num + 1))
        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
            this_frame_id, self.dataset.hist_num, offsets=offsets)
        first_frame, this_frame = self.dataset.get_frames(
            tracklet_id, frame_ids=(0, this_frame_id))
        prev_frames = self.dataset.get_frames(
            tracklet_id, frame_ids=prev_frame_ids)
        tracklet_key = (
            self.dataset.get_tracklet_key(tracklet_id)
            if hasattr(self.dataset, 'get_tracklet_key') else str(tracklet_id))
        seed_parts = (
            (tracklet_key, int(this_frame_id), int(candidate_id))
            if int(candidate_id) == 0 else
            (tracklet_key, int(epoch), int(this_frame_id),
             int(candidate_id)))
        raw = {
            'online_recursive_raw': True,
            'online_epoch': int(epoch),
            'online_batch_index': int(batch_index),
            'online_slot': int(slot),
            'first_frame': first_frame,
            'prev_frames': create_history_frame_dict(prev_frames),
            'this_frame': this_frame,
            'candidate_id': int(candidate_id),
            'b0_view_id': int(candidate_id),
            'ct_b0_auxiliary_only': bool(int(candidate_id) != 0),
            'valid_mask': list(valid_mask),
            'prev_frame_ids': list(prev_frame_ids),
            'this_frame_id': int(this_frame_id),
            'history_offsets': offsets,
            'sample_index': int(this_frame_id),
            'tracklet_id': int(tracklet_id),
            'tracklet_key': tracklet_key,
            'candidate_shared_transform': deterministic_candidate_offset(
                candidate_id, self.config, *seed_parts,
                'coherent_recursive_view'),
            'point_sampling_seeds': np.asarray([
                deterministic_point_seed(
                    self.config, *seed_parts, 'history', frame_id)
                for frame_id in prev_frame_ids], dtype=np.int64),
            'current_sampling_seed': deterministic_point_seed(
                self.config, *seed_parts, 'current'),
            'shadow_future': [],
            'shadow_scheduled': bool(build_shadow),
            'shadow_future_exists': [
                this_frame_id + horizon < self.dataset.get_num_frames_tracklet(tracklet_id)
                for horizon in (1, 2)],
        }
        if self.use_b1motion_v3:
            motion_aux_offsets = self._motion_v3_aux_offsets(this_frame_id)
            motion_aux_frame_ids, motion_aux_valid_mask = (
                get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num,
                    offsets=motion_aux_offsets))
            motion_aux_frames = self.dataset.get_frames(
                tracklet_id, frame_ids=motion_aux_frame_ids)
            raw.update({
                'motion_aux_prev_frames': create_history_frame_dict(
                    motion_aux_frames),
                'motion_aux_valid_mask': list(motion_aux_valid_mask),
                'motion_aux_frame_ids': list(motion_aux_frame_ids),
                'motion_aux_offsets': list(motion_aux_offsets),
            })
        if build_shadow:
            for future_id in (this_frame_id + 1, this_frame_id + 2):
                if future_id >= self.dataset.get_num_frames_tracklet(tracklet_id):
                    continue
                future_raw = self._online_raw_view(
                    epoch, batch_index, slot, tracklet_id, future_id,
                    candidate_id=0, build_shadow=False)
                future_raw['ct_observation_only'] = True
                raw['shadow_future'].append(future_raw)
        return raw

    def __getitem__(self, index):
        if self.online_recursive_training:
            if not isinstance(index, tuple) or len(index) != 7:
                raise TypeError(
                    "online recursive sampler requires structured batch indices")
            return self._online_raw_view(*index)
        retry_attempt = 0
        expected_candidate_id = None
        if isinstance(index, tuple):
            if (len(index) != 4
                    or index[0] != 'ct_stateless_observation_retry'):
                raise TypeError(
                    "observation sampler received an invalid structured index")
            _, index, retry_attempt, expected_candidate_id = index
            index = int(index)
            retry_attempt = int(retry_attempt)
            expected_candidate_id = int(expected_candidate_id)
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        if (expected_candidate_id is not None
                and int(candidate_id) != expected_candidate_id):
            raise RuntimeError(
                "stateless observation retry changed candidate_id")
        try:
            tracklet_id, this_frame_id = self._locate_tracklet(anno_id)
            frame_ids = (0, this_frame_id)
            first_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            if self.use_paired_history:
                paired_candidate_id = (
                    0 if self.paired_candidate_zero_only else candidate_id)
                prev_frame_ids_a, _ = get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num, offsets=self.b4_view_a_offsets)
                prev_frame_ids_b, _ = get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num, offsets=self.b4_view_b_offsets)
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
                                          paired_candidate_id, self.b4_view_a_offsets,
                                          candidate_offset_map=candidate_offset_map,
                                          point_sampling_seed_map=point_sampling_seed_map,
                                          current_sampling_seed=current_sampling_seed,
                                          candidate_shared_transform=candidate_shared_transform,
                                          sample_index=anno_id)
                view_b = self._build_view(tracklet_id, this_frame_id, first_frame, this_frame,
                                          paired_candidate_id, self.b4_view_b_offsets,
                                          candidate_offset_map=candidate_offset_map,
                                          point_sampling_seed_map=point_sampling_seed_map,
                                          current_sampling_seed=current_sampling_seed,
                                          candidate_shared_transform=candidate_shared_transform,
                                          sample_index=anno_id)
                return {"view_a": view_a, "view_b": view_b}

            if self.use_b1motion_v3:
                offsets = list(range(1, self.dataset.hist_num + 1))
                motion_aux_offsets = self._motion_v3_aux_offsets(
                    index)
            else:
                offsets = self._sample_history_offsets()
                motion_aux_offsets = None
            if str(getattr(
                    self.config, 'ct_observation_rng_mode', 'legacy'
                    )).strip().lower() == 'stateless_seqtrack':
                prev_frame_ids, _ = get_history_frame_ids_and_masks(
                    this_frame_id, self.dataset.hist_num, offsets=offsets)
                tracklet_key = (
                    self.dataset.get_tracklet_key(tracklet_id)
                    if hasattr(self.dataset, 'get_tracklet_key')
                    else str(tracklet_id))
                candidate_offset_map = {
                    int(frame_id): deterministic_candidate_offset(
                        candidate_id, self.config, 'observation', self.epoch,
                        tracklet_key, int(this_frame_id), int(frame_id))
                    for frame_id in prev_frame_ids
                }
                point_sampling_seed_map = {
                    int(frame_id): deterministic_point_seed(
                        self.config, 'observation', self.epoch, tracklet_key,
                        int(this_frame_id), int(candidate_id), 'history',
                        int(frame_id))
                    for frame_id in prev_frame_ids
                }
                current_sampling_seed = deterministic_point_seed(
                    self.config, 'observation', self.epoch, tracklet_key,
                    int(this_frame_id), int(candidate_id), 'current')
                return self._build_view(
                    tracklet_id, this_frame_id, first_frame, this_frame,
                    candidate_id, offsets,
                    candidate_offset_map=candidate_offset_map,
                    point_sampling_seed_map=point_sampling_seed_map,
                    current_sampling_seed=current_sampling_seed,
                    motion_aux_offsets=motion_aux_offsets,
                    sample_index=anno_id)
            return self._build_view(
                tracklet_id, this_frame_id, first_frame, this_frame,
                candidate_id, offsets,
                motion_aux_offsets=motion_aux_offsets,
                sample_index=anno_id)
        except AssertionError as error:
            if str(getattr(
                    self.config, 'ct_observation_rng_mode', 'legacy'
                    )).strip().lower() == 'stateless_seqtrack':
                max_attempts = int(getattr(
                    self.config, 'ct_observation_retry_max_attempts', 64))
                if max_attempts <= 0:
                    raise ValueError(
                        "ct_observation_retry_max_attempts must be positive")
                if retry_attempt >= max_attempts:
                    raise RuntimeError(
                        "stateless observation retry exhausted "
                        f"{max_attempts} attempts for candidate{candidate_id}"
                    ) from error
                retry_index = deterministic_candidate_retry_index(
                    index, len(self), int(self.num_candidates),
                    int(getattr(self.config, 'seed', 42) or 42),
                    self.epoch, retry_attempt)
                return self[(
                    'ct_stateless_observation_retry', retry_index,
                    retry_attempt + 1, int(candidate_id))]
            return self[torch.randint(0, len(self), size=(1,)).item()]
