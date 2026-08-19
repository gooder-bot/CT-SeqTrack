"""Formal v25 online inference input construction.

This function is mechanically extracted from the historical model base. The
public MotionBaseModelMF method remains stable while data responsibilities live
in ctseqtrack.data.
"""

import copy

import numpy as np
import torch
from nuscenes.utils import geometry_utils

from datasets import points_utils
from datasets.misc_utils import (
    build_effective_time_fields,
    build_main_time_fields,
    build_time_fields,
    get_history_frame_ids_and_masks,
    get_last_n_bounding_boxes,
    normalize_dynamics_time_mode,
)
from ctseqtrack.data.search import (
    build_ordered_trajectory_search_box,
    combined_search_support_statistics,
    resolve_b1_search_support,
    resolve_joint_search_geometry,
    sample_joint_novel_extensions,
    sample_source_aware_endpoint_points,
    useful_search_coverage_need,
)
from ctseqtrack.data.recursive import (
    RecursiveTrackState,
    build_recursive_input_contract,
)


def build_v25_input_dict(
    self, sequence, frame_id, results_bbs, **kwargs
):  # Note: There may be cases of input with empty point clouds
    assert frame_id > 0, "no need to construct an input_dict at frame 0"

    if (
        bool(getattr(self.config, "use_b1_prepass_support", False))
        and kwargs.get("motion_prediction") is None
    ):
        predictor = getattr(self, "predict_motion_prepass", None)
        if predictor is None:
            raise RuntimeError("B1 pre-pass support requires a motion predictor")
        kwargs["motion_prediction"] = predictor(sequence, frame_id, results_bbs)

    recursive_state = kwargs.get("recursive_state")
    use_recursive_contract = bool(
        getattr(self.config, "use_ct_joint_full", False)
    ) or bool(getattr(self.config, "observation_safe_bbox_size", False))
    if use_recursive_contract and recursive_state is not None:
        recursive_contract = build_recursive_input_contract(
            recursive_state, frame_id, self.hist_num, self.config, candidate_id=0
        )
        prev_frame_ids = recursive_contract["history_frame_ids"]
        valid_mask = recursive_contract["history_valid_mask"].tolist()
        ref_boxs = recursive_state.history_boxes(prev_frame_ids, valid_mask)
        prev_sampling_seeds = recursive_contract["point_sampling_seeds"].tolist()
        current_sampling_seed = int(recursive_contract["current_sampling_seed"])
    elif use_recursive_contract:
        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
            frame_id, self.hist_num
        )
        ref_boxs = get_last_n_bounding_boxes(results_bbs, valid_mask)
        fallback_state = RecursiveTrackState(
            tracklet_id=0,
            tracklet_key=str(
                sequence[0].get("tracklet_key", sequence[0].get("tracklet_id", "eval"))
            ),
            first_box=results_bbs[0],
        )
        for prediction_id, prediction in enumerate(results_bbs[1:], 1):
            fallback_state.append(prediction_id, prediction)
        recursive_contract = build_recursive_input_contract(
            fallback_state, frame_id, self.hist_num, self.config, candidate_id=0
        )
        prev_sampling_seeds = recursive_contract["point_sampling_seeds"].tolist()
        current_sampling_seed = int(recursive_contract["current_sampling_seed"])
    else:
        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(
            frame_id, self.hist_num
        )
        ref_boxs = get_last_n_bounding_boxes(results_bbs, valid_mask)
        prev_sampling_seeds = [None] * len(prev_frame_ids)
        current_sampling_seed = 1
        recursive_contract = None
    prev_frames = [sequence[id] for id in prev_frame_ids]
    this_frame = sequence[frame_id]
    this_pc = this_frame["pc"]
    prev_pcs = [frame["pc"] for frame in prev_frames]
    bbox_size = (
        recursive_contract["target_size"]
        if use_recursive_contract
        else this_frame["3d_bbox"].wlh
    )
    num_hist = len(valid_mask)
    default_time_step = getattr(
        self.config, "default_time_step", getattr(self.config, "time_step", 0.1)
    )
    pseudo_time_step = getattr(self.config, "pseudo_time_step", 0.1)
    use_real_time = getattr(self.config, "use_real_time", True)
    prev_timestamps = [frame.get("timestamp") for frame in prev_frames]
    current_timestamp = this_frame.get("timestamp")
    real_time_fields = build_time_fields(
        prev_timestamps,
        current_timestamp,
        frame_ids=prev_frame_ids,
        current_frame_id=frame_id,
        use_real_time=use_real_time,
        default_step=default_time_step,
        pseudo_step=pseudo_time_step,
    )
    relative_timestamps, delta_t_list, local_timestamps, current_timestamp = (
        real_time_fields
    )
    dynamics_time_mode = normalize_dynamics_time_mode(
        this_frame.get(
            "_ct_dynamics_time_mode", getattr(self.config, "dynamics_time_mode", "true")
        )
    )
    effective_time_fields = build_effective_time_fields(
        dynamics_time_mode,
        real_time_fields,
        effective_frame_timestamps=[
            frame.get("_ct_effective_timestamp") for frame in prev_frames
        ],
        effective_current_timestamp=this_frame.get("_ct_effective_timestamp"),
        frame_ids=prev_frame_ids,
        current_frame_id=frame_id,
        default_step=float(
            getattr(self.config, "dynamics_fixed_delta_t", default_time_step)
        ),
        pseudo_step=pseudo_time_step,
    )
    (
        effective_relative_timestamps,
        effective_delta_t_list,
        effective_local_timestamps,
        effective_current_timestamp,
    ) = effective_time_fields
    main_current_value = float(getattr(self.config, "main_time_current", 0.0))
    point_timestamps, corner_timestamps, main_timestamps = build_main_time_fields(
        valid_mask,
        relative_timestamps,
        local_timestamps,
        num_hist,
        pseudo_step=pseudo_time_step,
        source=getattr(self.config, "main_time_source", "real"),
        current_value=main_current_value,
    )

    prev_frame_pcs = []
    for i, prev_pc in enumerate(prev_pcs):
        prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(
            prev_pc,
            ref_boxs[i],
            ref_boxs[0],
            scale=self.config.bb_scale,
            offset=self.config.bb_offset,
        )
        prev_frame_pcs.append(prev_frame_pc)

    this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(
        this_pc,
        ref_boxs[0],
        ref_boxs[0],
        scale=self.config.bb_scale,
        offset=self.config.bb_offset,
    )
    baseline_search_points = this_frame_pc.points.T
    expanded_search_points = np.empty(
        (0, baseline_search_points.shape[1]),
        dtype=baseline_search_points.dtype,
    )
    ct_search_box = None
    ct_search_diagnostics = {
        "valid": False,
        "query_delta_t": float(effective_delta_t_list[0]),
    }
    use_ct_joint_full = bool(getattr(self.config, "use_ct_joint_full", False))
    if int(getattr(self.config, "ct_joint_contract_version", 3)) != 3:
        raise ValueError("formal CT inference requires contract version 3")
    if int(getattr(self.config, "ct_point_evidence_contract_version", 2)) != 2:
        raise ValueError(
            "formal CT inference requires point-evidence contract version 2"
        )
    use_trajectory_search = use_ct_joint_full
    if use_trajectory_search:
        ct_search_box, ct_search_diagnostics = build_ordered_trajectory_search_box(
            ref_boxs,
            effective_delta_t_list,
            valid_mask=valid_mask,
            base_length=float(
                getattr(self.config, "trajectory_search_base_length", 4.0)
            ),
            base_width=float(getattr(self.config, "trajectory_search_base_width", 2.0)),
            max_length=float(getattr(self.config, "ct_tube_max_length", 24.0)),
            max_width=float(getattr(self.config, "trajectory_search_max_width", 8.0)),
            max_speed=float(getattr(self.config, "ct_motion_max_speed", 20.0)),
            max_acceleration=float(
                getattr(self.config, "ct_motion_max_acceleration", 8.0)
            ),
            max_displacement=float(
                getattr(self.config, "ct_motion_max_displacement", 12.0)
            ),
            acceleration_weight=float(
                getattr(self.config, "ct_motion_acceleration_weight", 0.5)
            ),
            sigma_parallel_scale=float(
                getattr(self.config, "trajectory_search_sigma_parallel_scale", 2.0)
            ),
            sigma_perpendicular_scale=float(
                getattr(self.config, "trajectory_search_sigma_perpendicular_scale", 2.0)
            ),
            min_displacement=float(
                getattr(self.config, "trajectory_search_min_displacement", 0.2)
            ),
            min_delta_t=float(
                getattr(self.config, "trajectory_search_min_delta_t", 0.75)
            ),
            min_gap_ratio=float(
                getattr(self.config, "trajectory_search_min_gap_ratio", 1.5)
            ),
            allow_normal_cadence=True,
            require_recent_transition=True,
        )
        if ct_search_box is not None:
            ct_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                this_pc, ct_search_box, ref_boxs[0], scale=1.0, offset=0.0
            )
            expanded_search_points = ct_search_pc.points.T

    use_search_evidence_v3 = use_ct_joint_full
    use_endpoint_search_evidence = use_search_evidence_v3
    search_config_prefix = "search_v3" if use_ct_joint_full else "search_v2"

    def search_config_value(name, default):
        if use_ct_joint_full:
            joint_mapping = {
                "point_count": "ct_endpoint_quota",
                "extension_quota": "ct_endpoint_quota",
                "min_points": "ct_search_min_points",
                "max_length": "ct_tube_max_length",
                "max_width": "ct_tube_max_width",
                "max_speed": "ct_motion_max_speed",
                "max_acceleration": "ct_motion_max_acceleration",
                "max_displacement": "ct_motion_max_displacement",
                "acceleration_weight": "ct_motion_acceleration_weight",
            }
            field = joint_mapping.get(name)
            if field is not None:
                return getattr(self.config, field, default)
        return getattr(self.config, f"{search_config_prefix}_{name}", default)

    search_v2_box = None
    search_v2_diagnostics = {
        "valid": False,
        "query_delta_t": float(effective_delta_t_list[0]),
        "gap_ratio": 1.0,
        "sigma_parallel": 0.0,
        "sigma_perpendicular": 0.0,
    }
    search_v2_expanded_points = np.empty(
        (0, baseline_search_points.shape[1]),
        dtype=baseline_search_points.dtype,
    )
    search_v2_endpoint_xy = np.zeros((2,), dtype=np.float32)
    if use_endpoint_search_evidence:
        motion_prediction = kwargs.get("motion_prediction")
        use_prepass = (
            bool(getattr(self.config, "use_b1_prepass_support", False))
            if (
                not use_ct_joint_full
                or int(getattr(self.config, "ct_joint_contract_version", 1)) >= 2
            )
            else False
        )
        support_kwargs = dict(
            prediction=motion_prediction,
            use_b1_prepass=use_prepass,
            use_dynamic_sigma=bool(
                getattr(self.config, "search_v3_use_dynamic_sigma", False)
            ),
            fixed_margins=(
                float(getattr(self.config, "search_v3_fixed_margin_parallel", 2.0)),
                float(
                    getattr(self.config, "search_v3_fixed_margin_perpendicular", 1.0)
                ),
            ),
            coverage_scale=float(
                getattr(self.config, "search_v3_coverage_scale", 2.448)
            ),
            standardized_residual_quantile=tuple(
                getattr(
                    self.config,
                    "search_v3_standardized_residual_q90_parallel_perpendicular",
                    (1.0, 1.0),
                )
            ),
            min_direction_speed=float(
                getattr(self.config, "motion_v3_min_direction_speed", 0.2)
            ),
            max_length=float(search_config_value("max_length", 24.0)),
            max_width=float(search_config_value("max_width", 10.0)),
            fallback_max_speed=float(search_config_value("max_speed", 20.0)),
            fallback_max_acceleration=float(
                search_config_value("max_acceleration", 8.0)
            ),
            fallback_max_displacement=float(
                search_config_value("max_displacement", 12.0)
            ),
            fallback_acceleration_weight=float(
                search_config_value("acceleration_weight", 0.5)
            ),
            fallback_max_yaw_rate=float(
                search_config_value("max_yaw_rate", np.pi / 2.0)
            ),
            fallback_min_displacement=(
                float(search_config_value("min_displacement", 0.2))
                if not (
                    use_ct_joint_full
                    and int(getattr(self.config, "ct_joint_contract_version", 1)) >= 2
                )
                else 0.0
            ),
            fallback_require_recent_transition=use_ct_joint_full,
        )
        if (
            use_ct_joint_full
            and int(getattr(self.config, "ct_joint_contract_version", 1)) >= 2
        ):
            (search_v2_box, ct_search_box, search_v2_diagnostics) = (
                resolve_joint_search_geometry(
                    ref_boxs,
                    effective_delta_t_list,
                    valid_mask,
                    **support_kwargs,
                )
            )
            if ct_search_box is not None:
                ct_search_diagnostics = dict(search_v2_diagnostics)
                ct_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                    this_pc, ct_search_box, ref_boxs[0], scale=1.0, offset=0.0
                )
                expanded_search_points = ct_search_pc.points.T
        else:
            search_v2_box, search_v2_diagnostics = resolve_b1_search_support(
                ref_boxs,
                effective_delta_t_list,
                valid_mask,
                **support_kwargs,
            )
        if search_v2_box is not None:
            learned_prior_support = search_v2_diagnostics.get("prior_source") == "b1"
            search_v2_pc = points_utils.generate_subwindow_with_aroundboxs(
                this_pc,
                search_v2_box,
                ref_boxs[0],
                scale=(1.0 if learned_prior_support else self.config.bb_scale),
                offset=(0.0 if learned_prior_support else self.config.bb_offset),
            )
            search_v2_expanded_points = search_v2_pc.points.T
            endpoint_center = search_v2_diagnostics.get("endpoint_center")
            if endpoint_center is not None:
                endpoint_box = copy.deepcopy(ref_boxs[0])
                endpoint_box.center = np.asarray(endpoint_center, dtype=np.float64)
                endpoint_local = points_utils.transform_box(endpoint_box, ref_boxs[0])
                search_v2_endpoint_xy = np.asarray(
                    endpoint_local.center[:2], dtype=np.float32
                )
            else:
                search_v2_local_box = points_utils.transform_box(
                    search_v2_box, ref_boxs[0]
                )
                search_v2_endpoint_xy = np.asarray(
                    search_v2_local_box.center[:2], dtype=np.float32
                )
    num_points_in_search = this_frame_pc.nbr_points()

    coordinate_anchor_box = ref_boxs[0]
    coordinate_anchor_theta = (
        coordinate_anchor_box.orientation.degrees
        * coordinate_anchor_box.orientation.axis[-1]
        if self.config.degrees
        else coordinate_anchor_box.orientation.radians
        * coordinate_anchor_box.orientation.axis[-1]
    )
    coordinate_anchor = np.append(
        coordinate_anchor_box.center, coordinate_anchor_theta
    ).astype(np.float32)
    # canonical_box = points_utils.transform_box(ref_boxs[0], ref_boxs[0])
    ref_boxs = [
        points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs
    ]

    prev_points_list = [
        points_utils.regularize_pc_with_metadata(
            prev_frame_pc.points.T, self.config.point_sample_size, seed=seed
        ).points
        for prev_frame_pc, seed in zip(prev_frame_pcs, prev_sampling_seeds)
    ]
    base_regularized = None

    trajectory_search_points = np.zeros(
        (
            int(
                getattr(self.config, "ct_tube_quota", 128)
                if use_ct_joint_full
                else getattr(self.config, "trajectory_search_point_count", 128)
            ),
            baseline_search_points.shape[1],
        ),
        dtype=np.float32,
    )
    trajectory_search_sampling = {
        "active": False,
        "sample_count": 0,
        "available_count": 0,
    }
    trajectory_search_point_valid_mask = np.zeros(
        (trajectory_search_points.shape[0],), dtype=np.float32
    )
    trajectory_search_point_source = np.zeros(
        (trajectory_search_points.shape[0],), dtype=np.int64
    )
    if use_trajectory_search:
        base_regularized = points_utils.regularize_pc_with_metadata(
            baseline_search_points,
            self.config.point_sample_size,
            seed=current_sampling_seed,
        )
        this_points = base_regularized.points
        (
            trajectory_search_points,
            trajectory_search_point_valid_mask,
            trajectory_search_point_source,
            trajectory_search_sampling,
        ) = sample_source_aware_endpoint_points(
            baseline_search_points,
            expanded_search_points,
            sample_size=int(getattr(self.config, "ct_tube_quota", 128)),
            extension_quota=int(getattr(self.config, "ct_tube_quota", 128)),
            min_points=int(getattr(self.config, "ct_search_min_points", 3)),
            seed=current_sampling_seed,
        )
        ct_search_sampling = {
            "baseline_sample_count": int(self.config.point_sample_size),
            "expansion_sample_count": int(trajectory_search_sampling["sample_count"]),
            "expansion_available_count": int(
                trajectory_search_sampling["available_count"]
            ),
        }
        ct_search_active = bool(trajectory_search_sampling["active"])
        num_points_in_search = int(len(baseline_search_points))
        if ct_search_active:
            num_points_in_search += int(trajectory_search_sampling["available_count"])
    else:
        base_regularized = points_utils.regularize_pc_with_metadata(
            baseline_search_points,
            self.config.point_sample_size,
            seed=current_sampling_seed,
        )
        this_points = base_regularized.points
        ct_search_sampling = {
            "baseline_sample_count": int(self.config.point_sample_size),
            "expansion_sample_count": 0,
            "expansion_available_count": 0,
        }
        ct_search_active = False

    search_v2_point_count = int(search_config_value("point_count", 128))
    search_v2_points = np.zeros(
        (search_v2_point_count, baseline_search_points.shape[1]),
        dtype=np.float32,
    )
    search_v2_point_valid_mask = np.zeros((search_v2_point_count,), dtype=np.float32)
    search_v2_point_source = np.zeros((search_v2_point_count,), dtype=np.int64)
    search_v2_sampling = {
        "active": False,
        "sample_count": 0,
        "available_count": 0,
        "extension_count": 0,
        "overlap_count": 0,
    }
    if use_endpoint_search_evidence and search_v2_box is not None:
        search_v2_seed = (
            int(current_sampling_seed) * 1664525 + 1013904223
        ) & 0xFFFFFFFF
        (
            search_v2_points,
            search_v2_point_valid_mask,
            search_v2_point_source,
            search_v2_sampling,
        ) = sample_source_aware_endpoint_points(
            baseline_search_points,
            search_v2_expanded_points,
            sample_size=search_v2_point_count,
            extension_quota=int(search_config_value("extension_quota", 64)),
            min_points=int(search_config_value("min_points", 3)),
            seed=search_v2_seed,
        )
    joint_extension_source = None
    joint_extension_sampling = None
    if use_ct_joint_full:
        joint_extension_seed = (int(current_sampling_seed) * 22695477 + 1) & 0xFFFFFFFF
        (
            joint_extension_points,
            joint_extension_valid_mask,
            joint_extension_source,
            joint_extension_sampling,
        ) = sample_joint_novel_extensions(
            baseline_search_points,
            search_v2_expanded_points,
            expanded_search_points,
            endpoint_quota=int(getattr(self.config, "ct_endpoint_quota", 128)),
            tube_quota=int(getattr(self.config, "ct_tube_quota", 128)),
            seed=joint_extension_seed,
        )
        joint_extension_sampling.pop("_pool_points", None)
        endpoint_quota = int(getattr(self.config, "ct_endpoint_quota", 128))
        search_v2_points = joint_extension_points[:endpoint_quota]
        search_v2_point_valid_mask = joint_extension_valid_mask[:endpoint_quota]
        search_v2_point_source = (search_v2_point_valid_mask > 0).astype(np.int64)
        trajectory_search_points = joint_extension_points[endpoint_quota:]
        trajectory_search_point_valid_mask = joint_extension_valid_mask[endpoint_quota:]
        trajectory_search_point_source = (
            trajectory_search_point_valid_mask > 0
        ).astype(np.int64)
    joint_support = combined_search_support_statistics(
        (search_v2_points, trajectory_search_points),
        (search_v2_point_valid_mask, trajectory_search_point_valid_mask),
        (search_v2_point_source, trajectory_search_point_source),
        voxel_size=float(getattr(self.config, "ct_search_extension_voxel_size", 0.2)),
    )
    coverage_need, endpoint_ratio = useful_search_coverage_need(
        search_v2_diagnostics.get("query_delta_t", effective_delta_t_list[0]),
        search_v2_diagnostics.get("gap_ratio", 1.0),
        search_v2_endpoint_xy,
        coordinate_anchor_box.wlh,
        len(baseline_search_points),
        min_delta_t=float(getattr(self.config, "trajectory_search_min_delta_t", 0.75)),
        min_gap_ratio=float(
            getattr(self.config, "trajectory_search_min_gap_ratio", 1.5)
        ),
        min_endpoint_ratio=float(getattr(self.config, "ct_search_endpoint_ratio", 0.6)),
        sparse_base_points=int(
            getattr(self.config, "ct_search_sparse_base_points", 64)
        ),
        bb_scale=float(self.config.bb_scale),
        bb_offset=float(self.config.bb_offset),
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
        >= float(getattr(self.config, "trajectory_search_min_displacement", 0.2))
    )
    point_support_valid = bool(
        joint_support["total_count"]
        >= int(getattr(self.config, "ct_search_min_total_points", 16))
        and joint_support["extension_count"]
        >= int(getattr(self.config, "ct_search_min_extension_points", 8))
        and joint_support["extension_voxels"]
        >= int(getattr(self.config, "ct_search_min_extension_voxels", 4))
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
    search_has_usable_points = num_points_in_search > 2
    seg_mask_prev_list = [
        geometry_utils.points_in_box(ref_box, prev_points.T[:3, :], 1.25).astype(float)
        for ref_box, prev_points in zip(ref_boxs, prev_points_list)
    ]  # 搴斿綋鍙€冭檻xyz鐗瑰緛

    # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
    # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
    if frame_id != 1:
        for seg_mask_prev in seg_mask_prev_list:
            # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
            # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
            seg_mask_prev[seg_mask_prev == 0] = 0.2
            seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)

    timestamp_prev_list = [
        np.full(
            (self.config.point_sample_size, 1), fill_value=timestamp, dtype=np.float32
        )
        for timestamp in point_timestamps
    ]
    timestamp_this = np.full(
        (self.config.point_sample_size, 1),
        fill_value=main_current_value,
        dtype=np.float32,
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

    ref_box_thetas = [
        (
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if self.config.degrees
            else ref_box.orientation.radians * ref_box.orientation.axis[-1]
        )
        for ref_box in ref_boxs
    ]
    ref_box_list = [
        np.append(ref_box.center, theta).astype("float32")
        for ref_box, theta in zip(ref_boxs, ref_box_thetas)
    ]
    ref_boxs_np = np.stack(ref_box_list, axis=0)

    current_delta_t = delta_t_list[0] if len(delta_t_list) > 0 else default_time_step
    current_delta_t_effective = (
        effective_delta_t_list[0]
        if len(effective_delta_t_list) > 0
        else float(getattr(self.config, "dynamics_fixed_delta_t", default_time_step))
    )

    data_dict = {
        "points": torch.tensor(
            stack_points[None, :], device=self.device, dtype=torch.float32
        ),
        "ref_boxs": torch.tensor(
            ref_boxs_np[None, :], device=self.device, dtype=torch.float32
        ),
        "valid_mask": torch.tensor(
            valid_mask, device=self.device, dtype=torch.float32
        ).unsqueeze(0),
        "bbox_size": torch.tensor(
            bbox_size[None, :], device=self.device, dtype=torch.float32
        ),
        "coordinate_anchor": torch.tensor(
            coordinate_anchor[None, :], device=self.device, dtype=torch.float32
        ),
        "timestamps": torch.tensor(
            main_timestamps[None, :], device=self.device, dtype=torch.float32
        ),
        "delta_t": torch.tensor(
            np.array(delta_t_list, dtype=np.float32)[None, :],
            device=self.device,
            dtype=torch.float32,
        ),
        "delta_t_real": torch.tensor(
            np.array(delta_t_list, dtype=np.float32)[None, :],
            device=self.device,
            dtype=torch.float32,
        ),
        "delta_t_effective": torch.tensor(
            np.array(effective_delta_t_list, dtype=np.float32)[None, :],
            device=self.device,
            dtype=torch.float32,
        ),
        "delta_T": torch.tensor(
            np.array(corner_timestamps, dtype=np.float32)[None, :],
            device=self.device,
            dtype=torch.float32,
        ),
        "timestamps_real": torch.tensor(
            local_timestamps[None, :], device=self.device, dtype=torch.float32
        ),
        "delta_T_real": torch.tensor(
            np.array(relative_timestamps, dtype=np.float32)[None, :],
            device=self.device,
            dtype=torch.float32,
        ),
        "timestamps_effective": torch.tensor(
            np.asarray(effective_local_timestamps, dtype=np.float32)[None, :],
            device=self.device,
            dtype=torch.float32,
        ),
        "delta_T_effective": torch.tensor(
            np.array(effective_relative_timestamps, dtype=np.float32)[None, :],
            device=self.device,
            dtype=torch.float32,
        ),
        "current_timestamp": torch.tensor(
            [current_timestamp], device=self.device, dtype=torch.float64
        ),
        "current_effective_timestamp": torch.tensor(
            [effective_current_timestamp], device=self.device, dtype=torch.float64
        ),
        "current_delta_t": torch.tensor(
            [current_delta_t], device=self.device, dtype=torch.float32
        ),
        "current_delta_t_real": torch.tensor(
            [current_delta_t], device=self.device, dtype=torch.float32
        ),
        "current_delta_t_effective": torch.tensor(
            [current_delta_t_effective], device=self.device, dtype=torch.float32
        ),
        "dynamics_time_mode_id": torch.tensor(
            [{"true": 0, "fixed": 1, "shuffled": 2}[dynamics_time_mode]],
            device=self.device,
            dtype=torch.int64,
        ),
        "num_points_in_search": torch.tensor(
            [num_points_in_search], device=self.device, dtype=torch.float32
        ),
        "search_has_usable_points": torch.tensor(
            [search_has_usable_points], device=self.device, dtype=torch.float32
        ),
        "ct_search_used": torch.tensor(
            [search_support_valid if use_ct_joint_full else ct_search_active],
            device=self.device,
            dtype=torch.float32,
        ),
        "ct_search_expansion_ratio": torch.tensor(
            [
                ct_search_sampling["expansion_sample_count"]
                / float(self.config.point_sample_size)
            ],
            device=self.device,
            dtype=torch.float32,
        ),
        "ct_search_baseline_points": torch.tensor(
            [len(baseline_search_points)], device=self.device, dtype=torch.float32
        ),
        "ct_search_expansion_points": torch.tensor(
            [ct_search_sampling["expansion_available_count"]],
            device=self.device,
            dtype=torch.float32,
        ),
        "ct_search_query_delta_t": torch.tensor(
            [ct_search_diagnostics.get("query_delta_t", effective_delta_t_list[0])],
            device=self.device,
            dtype=torch.float32,
        ),
        "ct_search_predicted_displacement": torch.tensor(
            [ct_search_diagnostics.get("displacement", 0.0)],
            device=self.device,
            dtype=torch.float32,
        ),
        "ct_search_support_valid": torch.tensor(
            [search_support_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_geometry_valid": torch.tensor(
            [geometry_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_structural_point_valid": torch.tensor(
            [structural_point_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_new_support_valid": torch.tensor(
            [new_support_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_quality_valid": torch.tensor(
            [point_support_valid], device=self.device, dtype=torch.float32
        ),
        "candidate_valid": torch.tensor(
            [search_support_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_history_valid": torch.tensor(
            [recent_history_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_time_valid": torch.tensor(
            [time_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_proposal_valid": torch.tensor(
            [proposal_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_point_support_valid": torch.tensor(
            [point_support_valid], device=self.device, dtype=torch.float32
        ),
        "ct_search_coverage_need": torch.tensor(
            [coverage_need], device=self.device, dtype=torch.float32
        ),
        "ct_search_endpoint_ratio": torch.tensor(
            [endpoint_ratio], device=self.device, dtype=torch.float32
        ),
        "ct_search_total_point_count": torch.tensor(
            [joint_support["total_count"]], device=self.device, dtype=torch.float32
        ),
        "ct_search_extension_count": torch.tensor(
            [joint_support["extension_count"]], device=self.device, dtype=torch.float32
        ),
        "ct_search_extension_voxels": torch.tensor(
            [joint_support["extension_voxels"]], device=self.device, dtype=torch.float32
        ),
    }
    if use_ct_joint_full:
        if joint_extension_source is None:
            raise RuntimeError("contract-v3 inference extension source was not built")
        extension_points = np.concatenate(
            (search_v2_points, trajectory_search_points), axis=0
        )
        extension_valid_mask = np.concatenate(
            (search_v2_point_valid_mask, trajectory_search_point_valid_mask), axis=0
        )
        data_dict.update(
            {
                "ct_base_evidence_points": torch.tensor(
                    this_points[None, :], device=self.device, dtype=torch.float32
                ),
                "ct_base_evidence_valid_mask": torch.tensor(
                    base_regularized.unique_valid_mask[None, :],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "ct_extension_points": torch.tensor(
                    extension_points[None, :], device=self.device, dtype=torch.float32
                ),
                "ct_extension_valid_mask": torch.tensor(
                    extension_valid_mask[None, :],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "ct_extension_source": torch.tensor(
                    joint_extension_source[None, :],
                    device=self.device,
                    dtype=torch.long,
                ),
                "ct_evidence_raw_point_count": torch.tensor(
                    [len(baseline_search_points)],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "ct_evidence_base_unique_count": torch.tensor(
                    [base_regularized.unique_point_count],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "ct_evidence_extension_unique_count": torch.tensor(
                    [float(extension_valid_mask.sum())],
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
        )
    if bool(getattr(self.config, "use_b1motion_v3", False)):
        # Online history is already recursive and expressed in the latest
        # predicted anchor.  Expose an explicit motion contract rather than
        # reusing legacy dynamics fields with different target semantics.
        data_dict.update(
            {
                "motion_main_ref_boxs": data_dict["ref_boxs"],
                "motion_main_delta_t": data_dict["delta_t_effective"],
                "motion_main_current_delta_t": data_dict["current_delta_t_effective"],
                "motion_main_valid_mask": data_dict["valid_mask"],
                "motion_main_anchor": torch.tensor(
                    coordinate_anchor[None, :], device=self.device, dtype=torch.float32
                ),
                "motion_source_anchor": torch.tensor(
                    coordinate_anchor[None, :], device=self.device, dtype=torch.float32
                ),
            }
        )
        if use_search_evidence_v3:
            # Both branches own references to the same online recursive
            # state tensors.  No reconstruction or second clock is allowed.
            data_dict.update(
                {
                    "b2_v3_history_ref_boxs": data_dict["ref_boxs"],
                    "b2_v3_history_delta_t": data_dict["delta_t_effective"],
                    "b2_v3_history_valid_mask": data_dict["valid_mask"],
                    "b2_v3_history_mode_id": torch.tensor(
                        [2], device=self.device, dtype=torch.int64
                    ),
                    "b2_v3_history_anchor": data_dict["motion_main_anchor"],
                }
            )
    if use_trajectory_search:
        data_dict.update(
            {
                "trajectory_search_points": torch.tensor(
                    trajectory_search_points[None, :],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "trajectory_search_point_valid_mask": torch.tensor(
                    trajectory_search_point_valid_mask[None, :],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "trajectory_search_point_source": torch.tensor(
                    trajectory_search_point_source[None, :],
                    device=self.device,
                    dtype=torch.long,
                ),
                # Stable branch contract: 0=baseline, 1=endpoint, 2=tube.
                "trajectory_search_branch_source": torch.full(
                    (1, trajectory_search_points.shape[0]),
                    2,
                    device=self.device,
                    dtype=torch.long,
                ),
                "trajectory_search_valid": torch.tensor(
                    [
                        (
                            search_support_valid
                            if use_ct_joint_full
                            else trajectory_search_sampling["active"]
                        )
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "trajectory_search_gap_ratio": torch.tensor(
                    [ct_search_diagnostics.get("gap_ratio", 1.0)],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "trajectory_search_sigma_parallel": torch.tensor(
                    [ct_search_diagnostics.get("sigma_parallel", 0.0)],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "trajectory_search_sigma_perpendicular": torch.tensor(
                    [ct_search_diagnostics.get("sigma_perpendicular", 0.0)],
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
        )
    if use_search_evidence_v3:
        data_dict.update(
            {
                "search_v3_points": torch.tensor(
                    search_v2_points[None, :], device=self.device, dtype=torch.float32
                ),
                "search_v3_point_valid_mask": torch.tensor(
                    search_v2_point_valid_mask[None, :],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_point_source": torch.tensor(
                    search_v2_point_source[None, :],
                    device=self.device,
                    dtype=torch.long,
                ),
                "search_v3_branch_source": torch.ones(
                    (1, search_v2_points.shape[0]), device=self.device, dtype=torch.long
                ),
                "search_v3_geometry_valid": torch.tensor(
                    [geometry_valid],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_support_valid": torch.tensor(
                    [search_support_valid], device=self.device, dtype=torch.float32
                ),
                "search_v3_total_point_count": torch.tensor(
                    [joint_support["total_count"]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_joint_extension_count": torch.tensor(
                    [joint_support["extension_count"]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_extension_voxels": torch.tensor(
                    [joint_support["extension_voxels"]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_endpoint_ratio": torch.tensor(
                    [endpoint_ratio], device=self.device, dtype=torch.float32
                ),
                "search_v3_support_anchor_xy": torch.tensor(
                    search_v2_endpoint_xy[None, :],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_query_delta_t": torch.tensor(
                    [
                        search_v2_diagnostics.get(
                            "query_delta_t", effective_delta_t_list[0]
                        )
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_gap_ratio": torch.tensor(
                    [search_v2_diagnostics.get("gap_ratio", 1.0)],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_sigma_parallel": torch.tensor(
                    [search_v2_diagnostics.get("sigma_parallel", 0.0)],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_sigma_perpendicular": torch.tensor(
                    [search_v2_diagnostics.get("sigma_perpendicular", 0.0)],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_available_count": torch.tensor(
                    [search_v2_sampling["available_count"]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_extension_count": torch.tensor(
                    [search_v2_sampling["extension_count"]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_overlap_count": torch.tensor(
                    [search_v2_sampling["overlap_count"]],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_prior_source_id": torch.tensor(
                    [search_v2_diagnostics.get("source_id", 0)],
                    device=self.device,
                    dtype=torch.long,
                ),
                "search_v3_support_truncated": torch.tensor(
                    [bool(search_v2_diagnostics.get("truncated", False))],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_support_requested_extent": torch.tensor(
                    [
                        [
                            search_v2_diagnostics.get("requested_length", 0.0),
                            search_v2_diagnostics.get("requested_width", 0.0),
                        ]
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                "search_v3_support_actual_extent": torch.tensor(
                    [
                        [
                            search_v2_diagnostics.get("length", 0.0),
                            search_v2_diagnostics.get("width", 0.0),
                        ]
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
        )
    if getattr(self.config, "box_aware", False):
        stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
        hist_points_list = stack_points_split[:num_hist]
        candidate_bc_prev_list = [
            points_utils.get_point_to_box_distance(hist_points[:, :3], ref_box)
            for hist_points, ref_box in zip(hist_points_list, ref_boxs)
        ]
        candidate_bc_this = np.zeros_like(candidate_bc_prev_list[0])
        candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
        candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

        data_dict.update(
            {
                "candidate_bc": points_utils.np_to_torch_tensor(
                    candidate_bc.astype("float32"), device=self.device
                )
            }
        )
    return data_dict, results_bbs[-1]
