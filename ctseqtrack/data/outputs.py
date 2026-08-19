"""Flat output-dictionary construction for v25 samples."""

import numpy as np
from nuscenes.utils import geometry_utils

import datasets.points_utils as points_utils
from utils.candidate_utils import (
    anchor_relative_trajectory_targets,
    build_b1_physical_contract,
    canonical_dynamics_targets,
)
from ctseqtrack.data.search import (
    combined_search_support_statistics,
    sample_joint_novel_extensions,
    useful_search_coverage_need,
)
from utils.ct_history import b2_v3_history_mode_id


def build_base_output(context):
    baseline_search_points = context["baseline_search_points"]
    box_label = context["box_label"]
    box_label_prev_list = context["box_label_prev_list"]
    candidate_id = context["candidate_id"]
    candidate_offsets = context["candidate_offsets"]
    candidate_shared_transform = context["candidate_shared_transform"]
    candidate_shared_world_translation = context["candidate_shared_world_translation"]
    candidate_trajectory_mode = context["candidate_trajectory_mode"]
    canonical_ref_boxs = context["canonical_ref_boxs"]
    config = context["config"]
    coordinate_anchor = context["coordinate_anchor"]
    corner_timestamps = context["corner_timestamps"]
    coverage_need = context["coverage_need"]
    ct_motion_ref_boxs = context["ct_motion_ref_boxs"]
    ct_search_active = context["ct_search_active"]
    ct_search_diagnostics = context["ct_search_diagnostics"]
    ct_search_sampling = context["ct_search_sampling"]
    current_delta_t_effective = context["current_delta_t_effective"]
    current_delta_t_real = context["current_delta_t_real"]
    current_timestamp = context["current_timestamp"]
    data = context["data"]
    delta_t_list = context["delta_t_list"]
    dynamics_displacement_label = context["dynamics_displacement_label"]
    dynamics_time_mode = context["dynamics_time_mode"]
    effective_current_timestamp = context["effective_current_timestamp"]
    effective_delta_t_list = context["effective_delta_t_list"]
    effective_local_timestamps = context["effective_local_timestamps"]
    effective_relative_timestamps = context["effective_relative_timestamps"]
    endpoint_ratio = context["endpoint_ratio"]
    geometry_valid = context["geometry_valid"]
    history_offsets = context["history_offsets"]
    joint_support = context["joint_support"]
    local_timestamps = context["local_timestamps"]
    main_timestamps = context["main_timestamps"]
    motion_label_list = context["motion_label_list"]
    motion_state_label_list = context["motion_state_label_list"]
    new_support_valid = context["new_support_valid"]
    num_points_in_search = context["num_points_in_search"]
    point_support_valid = context["point_support_valid"]
    proposal_valid = context["proposal_valid"]
    recent_history_valid = context["recent_history_valid"]
    ref_box_list = context["ref_box_list"]
    relative_timestamps = context["relative_timestamps"]
    search_has_usable_points = context["search_has_usable_points"]
    search_support_valid = context["search_support_valid"]
    stack_points = context["stack_points"]
    stack_seg_label = context["stack_seg_label"]
    structural_point_valid = context["structural_point_valid"]
    this_box = context["this_box"]
    time_valid = context["time_valid"]
    use_ct_joint_full = context["use_ct_joint_full"]
    valid_mask = context["valid_mask"]
    velocity_label = context["velocity_label"]
    data_dict = {
        "points": stack_points.astype("float32"),  # Historical first, then current
        "box_label": box_label,
        "ref_boxs": np.stack(ref_box_list, axis=0),
        "box_label_prev": np.stack(box_label_prev_list, axis=0),
        "motion_label": np.stack(motion_label_list, axis=0),
        "motion_state_label": np.stack(motion_state_label_list, axis=0).astype("int"),
        "bbox_size": (
            np.asarray(data["first_frame"]["3d_bbox"].wlh, dtype=np.float32)
            if use_ct_joint_full
            or bool(getattr(config, "observation_safe_bbox_size", False))
            else this_box.wlh
        ),
        "seg_label": stack_seg_label.astype("int"),
        "valid_mask": np.array(valid_mask).astype("int"),
        "timestamps": main_timestamps,
        "delta_t": np.array(delta_t_list, dtype=np.float32),
        "delta_t_real": np.array(delta_t_list, dtype=np.float32),
        "delta_t_effective": np.array(effective_delta_t_list, dtype=np.float32),
        "delta_T": np.array(corner_timestamps, dtype=np.float32),
        "timestamps_real": local_timestamps,
        "delta_T_real": np.array(relative_timestamps, dtype=np.float32),
        "timestamps_effective": np.asarray(
            effective_local_timestamps, dtype=np.float32
        ),
        "delta_T_effective": np.array(effective_relative_timestamps, dtype=np.float32),
        "current_timestamp": np.float64(
            current_timestamp if current_timestamp is not None else 0.0
        ),
        "current_effective_timestamp": np.float64(effective_current_timestamp),
        "current_delta_t": np.float32(current_delta_t_real),
        "current_delta_t_real": np.float32(current_delta_t_real),
        "current_delta_t_effective": np.float32(current_delta_t_effective),
        "dynamics_time_mode_id": np.int64(
            {"true": 0, "fixed": 1, "shuffled": 2}[dynamics_time_mode]
        ),
        "num_points_in_search": np.float32(num_points_in_search),
        "search_has_usable_points": np.float32(search_has_usable_points),
        "ct_search_used": np.float32(
            search_support_valid if use_ct_joint_full else ct_search_active
        ),
        "ct_search_expansion_ratio": np.float32(
            ct_search_sampling["expansion_sample_count"]
            / float(config.point_sample_size)
        ),
        "ct_search_baseline_points": np.float32(len(baseline_search_points)),
        "ct_search_expansion_points": np.float32(
            ct_search_sampling["expansion_available_count"]
        ),
        "ct_search_query_delta_t": np.float32(
            ct_search_diagnostics.get("query_delta_t", effective_delta_t_list[0])
        ),
        "ct_search_predicted_displacement": np.float32(
            ct_search_diagnostics.get("displacement", 0.0)
        ),
        "ct_search_support_valid": np.float32(search_support_valid),
        "ct_search_geometry_valid": np.float32(geometry_valid),
        "ct_search_structural_point_valid": np.float32(structural_point_valid),
        "ct_search_new_support_valid": np.float32(new_support_valid),
        "ct_search_quality_valid": np.float32(point_support_valid),
        # Temporary compatibility alias; new code must use the explicit name.
        "candidate_valid": np.float32(search_support_valid),
        "ct_search_history_valid": np.float32(recent_history_valid),
        "ct_search_time_valid": np.float32(time_valid),
        "ct_search_proposal_valid": np.float32(proposal_valid),
        "ct_search_point_support_valid": np.float32(point_support_valid),
        "ct_search_coverage_need": np.float32(coverage_need),
        "ct_search_endpoint_ratio": np.float32(endpoint_ratio),
        "ct_search_total_point_count": np.float32(joint_support["total_count"]),
        "ct_search_extension_count": np.float32(joint_support["extension_count"]),
        "ct_search_extension_voxels": np.float32(joint_support["extension_voxels"]),
        "velocity_label": velocity_label,
        "dynamics_displacement_label": dynamics_displacement_label,
        "canonical_ref_boxs": canonical_ref_boxs,
        "ct_motion_ref_boxs": ct_motion_ref_boxs,
        "candidate_id": np.int64(candidate_id),
        "candidate_gap_frames": np.int64(
            data.get(
                "candidate_gap_frames",
                (
                    int(np.asarray(history_offsets).reshape(-1)[0])
                    if history_offsets is not None
                    else 1
                ),
            )
        ),
        "candidate_role": np.int64(data.get("candidate_role", candidate_id)),
        "candidate_available": np.float32(data.get("candidate_available", 1.0)),
        "candidate_boundary_ratio": np.float32(
            data.get("candidate_boundary_ratio", endpoint_ratio)
        ),
        "candidate_role_satisfied": np.float32(
            data.get("candidate_role_satisfied", 1.0 if candidate_id == 0 else 0.0)
        ),
        "candidate_trajectory_mode_id": np.int64(
            {"independent": 0, "shared_se2": 1}[candidate_trajectory_mode]
        ),
        "coordinate_anchor": coordinate_anchor,
        "candidate_offsets": candidate_offsets.astype("float32"),
        "candidate_shared_transform": np.asarray(
            candidate_shared_transform, dtype=np.float32
        ),
        "candidate_shared_world_translation": candidate_shared_world_translation,
    }
    return data_dict


def build_sample_output(context):
    base_regularized = context["base_regularized"]
    baseline_search_points = context["baseline_search_points"]
    candidate_id = context["candidate_id"]
    candidate_offsets = context["candidate_offsets"]
    candidate_shared_transform = context["candidate_shared_transform"]
    candidate_shared_world_translation = context["candidate_shared_world_translation"]
    candidate_trajectory_mode = context["candidate_trajectory_mode"]
    canonical_label_boxs = context["canonical_label_boxs"]
    canonical_ref_boxs = context["canonical_ref_boxs"]
    canonical_this_box = context["canonical_this_box"]
    causal_temporal_policy = context["causal_temporal_policy"]
    config = context["config"]
    coordinate_anchor = context["coordinate_anchor"]
    corner_timestamps = context["corner_timestamps"]
    coordinate_anchor_box = context["coordinate_anchor_box"]
    ct_search_box = context["ct_search_box"]
    ct_search_diagnostics = context["ct_search_diagnostics"]
    ct_search_sampling = context["ct_search_sampling"]
    current_sampling_seed = context["current_sampling_seed"]
    current_timestamp = context["current_timestamp"]
    data = context["data"]
    default_time_step = context["default_time_step"]
    delta_t_list = context["delta_t_list"]
    effective_delta_t_list = context["effective_delta_t_list"]
    dynamics_time_mode = context["dynamics_time_mode"]
    effective_current_timestamp = context["effective_current_timestamp"]
    effective_local_timestamps = context["effective_local_timestamps"]
    effective_relative_timestamps = context["effective_relative_timestamps"]
    expanded_search_points = context["expanded_search_points"]
    ground_truth_history = context["ground_truth_history"]
    history_offsets = context["history_offsets"]
    main_current_value = context["main_current_value"]
    main_timestamps = context["main_timestamps"]
    local_timestamps = context["local_timestamps"]
    motion_boxs = context["motion_boxs"]
    ct_motion_ref_boxs = context["ct_motion_ref_boxs"]
    motion_main_ref_boxs = context["motion_main_ref_boxs"]
    point_timestamps = context["point_timestamps"]
    prev_boxs = context["prev_boxs"]
    recursive_history = context["recursive_history"]
    ref_boxs = context["ref_boxs"]
    relative_timestamps = context["relative_timestamps"]
    sample_index = context["sample_index"]
    search_v2_box = context["search_v2_box"]
    search_v2_diagnostics = context["search_v2_diagnostics"]
    search_v2_endpoint_xy = context["search_v2_endpoint_xy"]
    search_v2_expanded_points = context["search_v2_expanded_points"]
    this_box = context["this_box"]
    use_ct_joint_full = context["use_ct_joint_full"]
    use_motion_v3 = context["use_motion_v3"]
    valid_mask = context["valid_mask"]
    if use_ct_joint_full:
        independent_seed_base = (
            current_sampling_seed if current_sampling_seed is not None else sample_index
        )
        joint_extension_seed = (int(independent_seed_base) * 22695477 + 1) & 0xFFFFFFFF
        (
            joint_extension_points,
            joint_extension_valid_mask,
            joint_extension_source,
            joint_extension_sampling,
        ) = sample_joint_novel_extensions(
            baseline_search_points,
            search_v2_expanded_points,
            expanded_search_points,
            endpoint_quota=int(getattr(config, "ct_endpoint_quota", 128)),
            tube_quota=int(getattr(config, "ct_tube_quota", 128)),
            seed=joint_extension_seed,
        )
        endpoint_quota = int(getattr(config, "ct_endpoint_quota", 128))
        search_v2_points = joint_extension_points[:endpoint_quota]
        search_v2_point_valid_mask = joint_extension_valid_mask[:endpoint_quota]
        search_v2_point_source = (search_v2_point_valid_mask > 0).astype(np.int64)
        trajectory_search_points = joint_extension_points[endpoint_quota:]
        trajectory_search_point_valid_mask = joint_extension_valid_mask[endpoint_quota:]
        trajectory_search_point_source = (
            trajectory_search_point_valid_mask > 0
        ).astype(np.int64)
        search_v2_sampling = {
            "active": bool(joint_extension_sampling["active"]),
            "sample_count": int(search_v2_point_valid_mask.sum()),
            "available_count": int(
                joint_extension_sampling["endpoint_available_count"]
            ),
            "extension_count": int(search_v2_point_valid_mask.sum()),
            "overlap_count": 0,
            "selected_extension_count": int(search_v2_point_valid_mask.sum()),
            "selected_overlap_count": 0,
        }
        trajectory_search_sampling = {
            "active": bool(trajectory_search_point_valid_mask.any()),
            "sample_count": int(trajectory_search_point_valid_mask.sum()),
            "available_count": int(joint_extension_sampling["tube_available_count"]),
        }
    joint_support = combined_search_support_statistics(
        (search_v2_points, trajectory_search_points),
        (search_v2_point_valid_mask, trajectory_search_point_valid_mask),
        (search_v2_point_source, trajectory_search_point_source),
        voxel_size=float(getattr(config, "ct_search_extension_voxel_size", 0.2)),
    )
    coverage_need, endpoint_ratio = useful_search_coverage_need(
        search_v2_diagnostics.get("query_delta_t", effective_delta_t_list[0]),
        search_v2_diagnostics.get("gap_ratio", 1.0),
        search_v2_endpoint_xy,
        coordinate_anchor_box.wlh,
        len(baseline_search_points),
        min_delta_t=float(getattr(config, "trajectory_search_min_delta_t", 0.75)),
        min_gap_ratio=float(getattr(config, "trajectory_search_min_gap_ratio", 1.5)),
        min_endpoint_ratio=float(getattr(config, "ct_search_endpoint_ratio", 0.6)),
        sparse_base_points=int(getattr(config, "ct_search_sparse_base_points", 64)),
        bb_scale=float(config.bb_scale),
        bb_offset=float(config.bb_offset),
    )
    recent_history_valid = bool(
        len(valid_mask) >= 2 and int(valid_mask[0]) and int(valid_mask[1])
    )
    query_dt_value = float(
        search_v2_diagnostics.get("query_delta_t", effective_delta_t_list[0])
    )
    time_valid = bool(np.isfinite(query_dt_value) and query_dt_value > 0.0)
    proposal_valid = bool(
        search_v2_diagnostics.get("valid", False)
        and not search_v2_diagnostics.get("constraint_clipped", False)
        and np.isfinite(search_v2_endpoint_xy).all()
        and float(search_v2_diagnostics.get("displacement", 0.0))
        >= float(getattr(config, "trajectory_search_min_displacement", 0.2))
    )
    point_support_valid = bool(
        joint_support["total_count"]
        >= int(getattr(config, "ct_search_min_total_points", 16))
        and joint_support["extension_count"]
        >= int(getattr(config, "ct_search_min_extension_points", 8))
        and joint_support["extension_voxels"]
        >= int(getattr(config, "ct_search_min_extension_voxels", 4))
    )
    geometry_valid = bool(
        recent_history_valid
        and time_valid
        and search_v2_box is not None
        and ct_search_box is not None
        and search_v2_diagnostics.get("valid", False)
        and np.isfinite(search_v2_endpoint_xy).all()
    )
    structural_point_valid = bool(joint_support["total_count"] >= 3)
    new_support_valid = bool(
        joint_support["extension_count"] >= 1 and joint_support["extension_voxels"] >= 1
    )
    if use_ct_joint_full:
        # Availability is a deterministic structural contract: valid B1
        # geometry plus at least one finite novel extension point.  No GT
        # label, learned score, density heuristic or utility estimate enters.
        search_support_valid = bool(
            geometry_valid
            and joint_extension_sampling is not None
            and joint_extension_sampling["sample_count"] > 0
        )
    else:
        search_support_valid = bool(
            recent_history_valid
            and time_valid
            and proposal_valid
            and coverage_need
            and point_support_valid
        )
    ct_search_active = ct_search_sampling["expansion_sample_count"] > 0
    num_points_in_search = int(len(baseline_search_points))
    if ct_search_active:
        num_points_in_search += int(ct_search_sampling["expansion_available_count"])
    search_has_usable_points = num_points_in_search > 2

    # B0 keeps the SeqTrack3D 1.25-scale segmentation definition.  B2 owns an
    # independent exact-box target contract in v25.
    seg_label_this = geometry_utils.points_in_box(
        this_box, this_points.T[:3, :], config.bb_scale
    ).astype(int)
    b2_target_bb_scale = float(
        getattr(config, "ct_b2_target_bb_scale", config.bb_scale)
    )
    b2_base_labels = geometry_utils.points_in_box(
        this_box, this_points.T[:3, :], b2_target_bb_scale
    ).astype(np.float32)
    if base_regularized is None:
        raise RuntimeError("v25 base evidence metadata was not produced")
    base_unique_valid_mask = base_regularized.unique_valid_mask
    search_v2_point_labels = geometry_utils.points_in_box(
        this_box,
        search_v2_points.T[:3, :],
        b2_target_bb_scale,
    ).astype(np.float32)
    search_v2_point_labels *= search_v2_point_valid_mask
    trajectory_search_point_labels = geometry_utils.points_in_box(
        this_box,
        trajectory_search_points.T[:3, :],
        b2_target_bb_scale,
    ).astype(np.float32)
    trajectory_search_point_labels *= trajectory_search_point_valid_mask
    if joint_extension_sampling is not None:
        extension_pool_points = joint_extension_sampling.pop(
            "_pool_points",
            np.zeros((0, baseline_search_points.shape[1]), dtype=np.float32),
        )
    else:
        extension_pool_points = np.zeros(
            (0, baseline_search_points.shape[1]), dtype=np.float32
        )
    base_target_count = int(np.sum((b2_base_labels > 0) * (base_unique_valid_mask > 0)))
    expansion_regions = np.concatenate(
        (search_v2_expanded_points, expanded_search_points), axis=0
    )
    if len(expansion_regions):
        _, unique_expansion_indices = np.unique(
            np.rint(expansion_regions[:, :3] / 1e-6).astype(np.int64),
            axis=0,
            return_index=True,
        )
        expansion_regions = expansion_regions[np.sort(unique_expansion_indices)]
    expansion_target_count = (
        int(
            np.sum(
                geometry_utils.points_in_box(
                    this_box, expansion_regions.T[:3, :], b2_target_bb_scale
                )
            )
        )
        if len(expansion_regions)
        else 0
    )
    extension_pool_target_count = (
        int(
            np.sum(
                geometry_utils.points_in_box(
                    this_box, extension_pool_points.T[:3, :], b2_target_bb_scale
                )
            )
        )
        if len(extension_pool_points)
        else 0
    )
    extension_sampled_target_count = int(
        np.sum(search_v2_point_labels > 0) + np.sum(trajectory_search_point_labels > 0)
    )
    recovery_role = int(candidate_id)
    recovery_positive = False
    recovery_fallback = False
    if candidate_id == 1:
        recovery_positive = bool(
            base_target_count <= 2
            and extension_pool_target_count > 0
            and extension_sampled_target_count > 0
        )
        recovery_fallback = not recovery_positive
    elif candidate_id == 2:
        recovery_positive = bool(
            base_target_count == 0
            and extension_pool_target_count > 0
            and extension_sampled_target_count > 0
        )
        recovery_fallback = not recovery_positive
    seg_label_prev_list = [
        geometry_utils.points_in_box(
            prev_box, prev_points.T[:3, :], config.bb_scale
        ).astype(int)
        for prev_box, prev_points in zip(prev_boxs, prev_points_list)
    ]  # 应当只考虑xyz特征
    seg_mask_prev_list = [
        geometry_utils.points_in_box(
            ref_box, prev_points.T[:3, :], config.bb_scale
        ).astype(float)
        for ref_box, prev_points in zip(ref_boxs, prev_points_list)
    ]  # 应当只考虑xyz特征
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
        (config.point_sample_size, 1), fill_value=main_current_value, dtype=np.float32
    )

    prev_points_list = [
        np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
        for prev_points, timestamp_prev, seg_mask_prev in zip(
            prev_points_list, timestamp_prev_list, seg_mask_prev_list
        )
    ]
    this_points = np.concatenate(
        [this_points, timestamp_this, seg_mask_this[:, None]], axis=-1
    )
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

    theta_this = (
        this_box.orientation.degrees * this_box.orientation.axis[-1]
        if config.degrees
        else this_box.orientation.radians * this_box.orientation.axis[-1]
    )
    box_label = np.append(this_box.center, theta_this).astype("float32")
    theta_prev_list = [
        (
            prev_box.orientation.degrees * prev_box.orientation.axis[-1]
            if config.degrees
            else prev_box.orientation.radians * prev_box.orientation.axis[-1]
        )
        for prev_box in prev_boxs
    ]
    box_label_prev_list = [
        np.append(prev_box.center, theta_prev).astype("float32")
        for prev_box, theta_prev in zip(prev_boxs, theta_prev_list)
    ]

    # Generate a reference box sequence
    theta_ref_list = [
        (
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if config.degrees
            else ref_box.orientation.radians * ref_box.orientation.axis[-1]
        )
        for ref_box in ref_boxs
    ]
    ref_box_list = [
        np.append(ref_box.center, theta_ref).astype("float32")
        for ref_box, theta_ref in zip(ref_boxs, theta_ref_list)
    ]

    theta_motion_list = [
        (
            motion_box.orientation.degrees * motion_box.orientation.axis[-1]
            if config.degrees
            else motion_box.orientation.radians * motion_box.orientation.axis[-1]
        )
        for motion_box in motion_boxs
    ]

    motion_label_list = [
        np.append(motion_box.center, theta_motion).astype("float32")
        for motion_box, theta_motion in zip(motion_boxs, theta_motion_list)
    ]
    motion_state_label_list = [
        np.sqrt(np.sum((this_box.center - prev_box.center) ** 2))
        > config.motion_threshold
        for prev_box in prev_boxs
    ]
    current_delta_t_real = (
        delta_t_list[0] if len(delta_t_list) > 0 else default_time_step
    )
    current_delta_t_effective = (
        effective_delta_t_list[0]
        if len(effective_delta_t_list) > 0
        else float(getattr(config, "dynamics_fixed_delta_t", default_time_step))
    )
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
        )
    )
    motion_main_target_xy = None
    if use_motion_v3:
        motion_main_physical = build_b1_physical_contract(
            canonical_this_box,
            ground_truth_history,
            recursive_history,
            current_delta_t_real,
            degrees=config.degrees,
            eps=1e-3,
        )
        if not np.array_equal(motion_main_ref_boxs, motion_main_physical["ref_boxs"]):
            raise RuntimeError("B1 main input and physical-label axes diverged")
        motion_main_target_xy = motion_main_physical["target_xy"]

    data_dict = build_base_output(locals())
    if use_ct_joint_full:
        extension_points = np.concatenate(
            (search_v2_points, trajectory_search_points), axis=0
        )
        extension_labels = np.concatenate(
            (
                search_v2_point_labels,
                trajectory_search_point_labels,
            ),
            axis=0,
        )
        extension_valid_mask = np.concatenate(
            (
                search_v2_point_valid_mask,
                trajectory_search_point_valid_mask,
            ),
            axis=0,
        )
        if joint_extension_source is None:
            raise RuntimeError("contract-v3 extension source was not built")
        data_dict.update(
            {
                # This is the exact current-frame tensor appended to ``points``;
                # no independent crop or resampling is allowed.
                "ct_base_evidence_points": this_points.astype("float32"),
                "ct_base_evidence_labels": b2_base_labels.astype("float32"),
                "ct_base_evidence_valid_mask": base_unique_valid_mask.astype("float32"),
                "ct_extension_points": extension_points.astype("float32"),
                "ct_extension_labels": extension_labels.astype("float32"),
                "ct_extension_valid_mask": extension_valid_mask.astype("float32"),
                # 0=padding, 1=endpoint, 2=tube, 3=both.
                "ct_extension_source": joint_extension_source.astype("int64"),
                "ct_acquisition_base_target_count": np.float32(base_target_count),
                "ct_acquisition_expansion_target_count": np.float32(
                    expansion_target_count
                ),
                "ct_acquisition_extension_pool_target_count": np.float32(
                    extension_pool_target_count
                ),
                "ct_acquisition_sampled_target_count": np.float32(
                    extension_sampled_target_count
                ),
                "ct_acquisition_extension_pool_count": np.float32(
                    joint_extension_sampling["available_count"]
                ),
                "ct_acquisition_sampled_count": np.float32(
                    joint_extension_sampling["sample_count"]
                ),
                "ct_evidence_raw_point_count": np.float32(len(baseline_search_points)),
                "ct_evidence_base_unique_count": np.float32(
                    np.sum(base_unique_valid_mask > 0)
                ),
                "ct_evidence_extension_unique_count": np.float32(
                    np.sum(extension_valid_mask > 0)
                ),
                "ct_evidence_foreground_count": np.float32(
                    base_target_count + extension_sampled_target_count
                ),
                "ct_view_role": np.int64(0 if candidate_id == 0 else 1),
            }
        )
        if not causal_temporal_policy:
            # Explicit legacy_spatial_gt_ablation diagnostics only.  Formal
            # causal batches expose role/gap metadata above instead.
            data_dict.update(
                {
                    "ct_recovery_role": np.int64(recovery_role),
                    "ct_recovery_positive": np.float32(recovery_positive),
                    "ct_recovery_fallback": np.float32(recovery_fallback),
                }
            )
    b2_v3_history_mode = context["b2_v3_history_mode"]
    coordinate_anchor = context["coordinate_anchor"]
    ct_search_diagnostics = context["ct_search_diagnostics"]
    history_offsets = context["history_offsets"]
    motion_anchor = context["motion_anchor"]
    motion_aux_contract = context["motion_aux_contract"]
    num_hist = context["num_hist"]
    point_sampling_seeds = context["point_sampling_seeds"]
    prev_frame_ids = context["prev_frame_ids"]
    this_frame_id = context["this_frame_id"]
    use_search_evidence_v3 = context["use_search_evidence_v3"]
    use_trajectory_search = context["use_trajectory_search"]
    if use_trajectory_search:
        data_dict.update(
            {
                "trajectory_displacement_label": trajectory_displacement_label.astype(
                    "float32"
                ),
                "trajectory_velocity_label": trajectory_velocity_label.astype(
                    "float32"
                ),
                "trajectory_search_points": trajectory_search_points.astype("float32"),
                "trajectory_search_point_labels": trajectory_search_point_labels.astype(
                    "float32"
                ),
                "trajectory_search_point_valid_mask": trajectory_search_point_valid_mask.astype(
                    "float32"
                ),
                "trajectory_search_point_source": trajectory_search_point_source.astype(
                    "int64"
                ),
                # Stable branch contract: 0=baseline, 1=endpoint, 2=tube.
                "trajectory_search_branch_source": np.full(
                    trajectory_search_point_valid_mask.shape, 2, dtype=np.int64
                ),
                "trajectory_search_valid": np.float32(
                    search_support_valid
                    if use_ct_joint_full
                    else trajectory_search_sampling["active"]
                ),
                "trajectory_search_gap_ratio": np.float32(
                    ct_search_diagnostics.get("gap_ratio", 1.0)
                ),
                "trajectory_search_sigma_parallel": np.float32(
                    ct_search_diagnostics.get("sigma_parallel", 0.0)
                ),
                "trajectory_search_sigma_perpendicular": np.float32(
                    ct_search_diagnostics.get("sigma_perpendicular", 0.0)
                ),
            }
        )
    if use_search_evidence_v3:
        v3_query_delta_t = np.float32(
            search_v2_diagnostics.get("query_delta_t", effective_delta_t_list[0])
        )
        if v3_query_delta_t != np.float32(effective_delta_t_list[0]):
            raise RuntimeError("B2-v3 query delta_t diverged from the shared B1 clock")
        data_dict.update(
            {
                "search_v3_points": search_v2_points.astype("float32"),
                "search_v3_point_valid_mask": search_v2_point_valid_mask.astype(
                    "float32"
                ),
                "search_v3_point_source": search_v2_point_source.astype("int64"),
                "search_v3_branch_source": np.ones(
                    search_v2_point_valid_mask.shape, dtype=np.int64
                ),
                "search_v3_point_labels": search_v2_point_labels.astype("float32"),
                "search_v3_geometry_valid": np.float32(geometry_valid),
                "search_v3_support_valid": np.float32(search_support_valid),
                "search_v3_total_point_count": np.float32(joint_support["total_count"]),
                "search_v3_joint_extension_count": np.float32(
                    joint_support["extension_count"]
                ),
                "search_v3_extension_voxels": np.float32(
                    joint_support["extension_voxels"]
                ),
                "search_v3_endpoint_ratio": np.float32(endpoint_ratio),
                "search_v3_support_anchor_xy": search_v2_endpoint_xy.astype("float32"),
                "search_v3_query_delta_t": v3_query_delta_t,
                "search_v3_gap_ratio": np.float32(
                    search_v2_diagnostics.get("gap_ratio", 1.0)
                ),
                "search_v3_sigma_parallel": np.float32(
                    search_v2_diagnostics.get("sigma_parallel", 0.0)
                ),
                "search_v3_sigma_perpendicular": np.float32(
                    search_v2_diagnostics.get("sigma_perpendicular", 0.0)
                ),
                "search_v3_available_count": np.float32(
                    search_v2_sampling["available_count"]
                ),
                "search_v3_extension_count": np.float32(
                    search_v2_sampling["extension_count"]
                ),
                "search_v3_overlap_count": np.float32(
                    search_v2_sampling["overlap_count"]
                ),
                "search_v3_prior_source_id": np.int64(
                    search_v2_diagnostics.get("source_id", 0)
                ),
                "search_v3_support_truncated": np.float32(
                    bool(search_v2_diagnostics.get("truncated", False))
                ),
                "search_v3_support_requested_extent": np.asarray(
                    (
                        search_v2_diagnostics.get("requested_length", 0.0),
                        search_v2_diagnostics.get("requested_width", 0.0),
                    ),
                    dtype=np.float32,
                ),
                "search_v3_support_actual_extent": np.asarray(
                    (
                        search_v2_diagnostics.get("length", 0.0),
                        search_v2_diagnostics.get("width", 0.0),
                    ),
                    dtype=np.float32,
                ),
                "b2_v3_history_ref_boxs": motion_main_ref_boxs.astype("float32"),
                "b2_v3_history_valid_mask": np.asarray(valid_mask, dtype=np.int64),
                "b2_v3_history_delta_t": np.asarray(
                    effective_delta_t_list, dtype=np.float32
                ),
                "b2_v3_history_mode_id": np.int64(
                    b2_v3_history_mode_id(b2_v3_history_mode)
                ),
                "b2_v3_history_anchor": motion_anchor,
            }
        )
    if use_motion_v3:
        data_dict.update(
            {
                "motion_main_ref_boxs": motion_main_ref_boxs.astype("float32"),
                "motion_main_delta_t": np.asarray(
                    effective_delta_t_list, dtype=np.float32
                ),
                "motion_main_current_delta_t": np.float32(current_delta_t_effective),
                "motion_main_valid_mask": np.asarray(valid_mask, dtype=np.int64),
                "motion_main_target_xy": motion_main_target_xy.astype("float32"),
                "motion_main_anchor": motion_anchor,
                "motion_source_anchor": motion_anchor,
            }
        )
        if motion_aux_contract is not None:
            data_dict.update(motion_aux_contract)
    if prev_frame_ids is not None:
        data_dict["prev_frame_ids"] = np.array(prev_frame_ids, dtype=np.int64)
    if this_frame_id is not None:
        data_dict["this_frame_id"] = np.int64(this_frame_id)
    if history_offsets is not None:
        data_dict["history_offsets"] = np.array(history_offsets, dtype=np.int64)
    if point_sampling_seeds is not None:
        data_dict["point_sampling_seeds"] = point_sampling_seeds
    if current_sampling_seed is not None:
        data_dict["current_sampling_seed"] = np.int64(current_sampling_seed)

    if getattr(config, "box_aware", False):
        stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
        hist_points_list = stack_points_split[:num_hist]
        prev_bc_list = [
            points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
            for hist_points, prev_box in zip(hist_points_list, prev_boxs)
        ]
        this_points_split = stack_points_split[-1]
        this_bc = points_utils.get_point_to_box_distance(
            this_points_split[:, :3], this_box
        )

        candidate_bc_prev_list = [
            points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
            for hist_points, prev_box in zip(hist_points_list, ref_boxs)
        ]

        candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
        candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
        candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

        data_dict.update(
            {
                "prev_bc": np.stack(prev_bc_list, axis=0).astype("float32"),
                "this_bc": this_bc.astype("float32"),
                "candidate_bc": candidate_bc.astype("float32"),
            }
        )

    return data_dict
