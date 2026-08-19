"""Canonical CT-SeqTrack v25 multi-frame sample construction."""

import copy

import numpy as np
from nuscenes.utils import geometry_utils
from pyquaternion import Quaternion

import datasets.points_utils as points_utils
from ctseqtrack.data.auxiliary import build_motion_aux_contract
from ctseqtrack.data.outputs import (
    build_sample_output,
)
from datasets.misc_utils import (
    build_effective_time_fields,
    build_main_time_fields,
    build_time_fields,
    normalize_dynamics_time_mode,
)
from utils.candidate_utils import (
    apply_shared_se2_to_boxes,
    boxes_to_anchor_parameters,
    build_ct_training_histories,
    equivalent_local_offsets,
    normalize_candidate_trajectory_mode,
    reexpress_motion_prediction,
    shared_se2_world_translation,
    validate_shared_se2_transform,
)
from utils.ct_history import (
    select_b2_v3_history_mode,
)
from ctseqtrack.data.search import (
    build_ordered_trajectory_search_box,
    resolve_joint_search_geometry,
    sample_source_aware_endpoint_points,
)
from utils.sampling_utils import (
    deterministic_candidate_offset,
    sample_candidate_offset,
)


def motion_processing_mf(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    point_sample_size
    bb_scale
    bb_offset
    """
    prev_frames = data["prev_frames"]
    this_frame = data["this_frame"]
    candidate_id = data["candidate_id"]
    online_recursive_state = data.get("online_recursive_state")
    valid_mask = (
        list(online_recursive_state["history_valid_mask"])
        if online_recursive_state is not None
        else data["valid_mask"]
    )
    num_hist = len(valid_mask)
    empty_counter = 0

    prev_pcs = [
        prev_frames[key]["pc"] for key in sorted(prev_frames, key=lambda k: abs(int(k)))
    ]  # Ordered point clouds, -1, -2, -3
    prev_boxs = [
        prev_frames[key]["3d_bbox"]
        for key in sorted(prev_frames, key=lambda k: abs(int(k)))
    ]  # Ordered point clouds, -1, -2, -3
    this_pc, this_box = this_frame["pc"], this_frame["3d_bbox"]
    if int(getattr(config, "ct_joint_contract_version", 3)) != 3:
        raise ValueError("formal CT samples require joint contract version 3")
    if int(getattr(config, "ct_point_evidence_contract_version", 2)) != 2:
        raise ValueError("formal CT samples require point-evidence contract version 2")
    # Keep the untouched GT trajectory for M1 labels and invariance audits.
    # ``transform_box`` is non-mutating, but the local variables below are
    # intentionally rebound to candidate-coordinate boxes.
    ground_truth_history = list(prev_boxs)
    recursive_history = list(prev_boxs)
    candidate_history = None
    canonical_this_box = this_box
    if online_recursive_state is not None:
        state_rows = np.asarray(
            online_recursive_state["history_boxes_world"], dtype=np.float64
        )
        if state_rows.shape != (num_hist, 7):
            raise ValueError(
                f"online recursive history must have shape {(num_hist, 7)}"
            )
        online_boxs = []
        for source_box, row in zip(prev_boxs, state_rows):
            state_box = copy.deepcopy(source_box)
            state_box.center = row[:3].copy()
            state_box.wlh = row[3:6].copy()
            state_box.orientation = Quaternion(axis=[0, 0, 1], radians=float(row[6]))
            online_boxs.append(state_box)
        # Every input branch starts from the same deployed recursive state.
        # Current-frame GT remains available only for labels below.
        prev_boxs = online_boxs
        recursive_history = list(online_boxs)
    motion_anchor_box = recursive_history[0]
    sorted_prev_keys = sorted(prev_frames, key=lambda k: abs(int(k)))
    prev_timestamps = (
        list(online_recursive_state["history_timestamps"])
        if online_recursive_state is not None
        else [prev_frames[key].get("timestamp") for key in sorted_prev_keys]
    )
    current_timestamp = this_frame.get("timestamp")
    default_time_step = getattr(
        config, "default_time_step", getattr(config, "time_step", 0.1)
    )
    pseudo_time_step = getattr(config, "pseudo_time_step", 0.1)
    use_real_time = getattr(config, "use_real_time", True)

    prev_frame_ids = data.get("prev_frame_ids")
    this_frame_id = data.get("this_frame_id")
    history_offsets = data.get("history_offsets")
    candidate_trajectory_mode = normalize_candidate_trajectory_mode(
        getattr(config, "candidate_trajectory_mode", "independent")
    )
    candidate_offsets = data.get("candidate_offsets")
    candidate_shared_transform = data.get("candidate_shared_transform")
    if candidate_trajectory_mode == "shared_se2":
        if candidate_offsets is not None:
            raise ValueError(
                "shared_se2 consumes one candidate_shared_transform, not per-frame "
                "candidate_offsets"
            )
        if candidate_shared_transform is None:
            candidate_shared_transform = (
                deterministic_candidate_offset(
                    candidate_id,
                    config,
                    data.get("tracklet_key", data.get("tracklet_id", "unknown")),
                    data.get("this_frame_id", data.get("sample_index", 0)),
                    "online_recovery_view",
                )
                if online_recursive_state is not None
                else sample_candidate_offset(candidate_id, config)
            )
        candidate_shared_transform = validate_shared_se2_transform(
            candidate_shared_transform
        ).astype(np.float32)
        if int(candidate_id) == 0 and not np.array_equal(
            candidate_shared_transform, np.zeros(3, dtype=np.float32)
        ):
            raise ValueError(
                "candidate0 must remain the exact identity in shared_se2 mode"
            )
    else:
        if candidate_shared_transform is not None:
            raise ValueError(
                "candidate_shared_transform is only valid for shared_se2 mode"
            )
        if candidate_offsets is not None:
            candidate_offsets = np.asarray(candidate_offsets, dtype=np.float32)
            if candidate_offsets.shape != (num_hist, 3):
                raise ValueError(
                    f"candidate_offsets must have shape {(num_hist, 3)}, "
                    f"got {candidate_offsets.shape}."
                )
            if not np.isfinite(candidate_offsets).all():
                raise ValueError("candidate_offsets contains non-finite values.")
        else:
            if online_recursive_state is not None:
                raise ValueError(
                    "online recursive training requires candidate_trajectory_mode=shared_se2"
                )
            candidate_offsets = np.stack(
                [
                    sample_candidate_offset(candidate_id, config)
                    for _ in range(num_hist)
                ],
                axis=0,
            )
    point_sampling_seeds = data.get("point_sampling_seeds")
    if point_sampling_seeds is not None:
        point_sampling_seeds = np.asarray(point_sampling_seeds, dtype=np.int64)
        if point_sampling_seeds.shape != (num_hist,):
            raise ValueError(
                f"point_sampling_seeds must have shape {(num_hist,)}, "
                f"got {point_sampling_seeds.shape}."
            )
        prev_sampling_seeds = [int(seed) for seed in point_sampling_seeds]
    else:
        if online_recursive_state is not None:
            tracklet_seed_key = data.get(
                "tracklet_key", data.get("tracklet_id", "unknown")
            )
            current_seed_frame = data.get("this_frame_id", data.get("sample_index", 0))
            prev_sampling_seeds = [
                deterministic_point_seed(
                    config,
                    tracklet_seed_key,
                    current_seed_frame,
                    int(candidate_id),
                    "history",
                    index,
                )
                for index in range(num_hist)
            ]
        else:
            prev_sampling_seeds = [None] * num_hist
    current_sampling_seed = data.get("current_sampling_seed")
    if current_sampling_seed is not None:
        current_sampling_seed = int(current_sampling_seed)
    elif online_recursive_state is not None:
        current_sampling_seed = deterministic_point_seed(
            config,
            data.get("tracklet_key", data.get("tracklet_id", "unknown")),
            data.get("this_frame_id", data.get("sample_index", 0)),
            int(candidate_id),
            "current",
        )
    sample_index = int(
        data.get("sample_index", this_frame_id if this_frame_id is not None else 0)
    )

    # Check the number of empty boxes
    for prev_box, prev_pc in zip(prev_boxs, prev_pcs):
        num_points_in_prev_box = geometry_utils.points_in_box(
            prev_box, prev_pc.points[0:3, :]
        ).sum()
        if num_points_in_prev_box < config.limit_num_points_in_prev_box:
            empty_counter += 1
    if online_recursive_state is None:
        assert empty_counter < config.empty_box_limit, "not enough valid box"

    if candidate_trajectory_mode == "shared_se2":
        ref_boxs = apply_shared_se2_to_boxes(
            prev_boxs, candidate_shared_transform, degrees=config.degrees
        )
        candidate_offsets = equivalent_local_offsets(
            prev_boxs, ref_boxs, degrees=config.degrees
        )
        candidate_shared_world_translation = shared_se2_world_translation(
            prev_boxs[0], candidate_shared_transform, degrees=config.degrees
        ).astype(np.float32)
    else:
        ref_boxs = []
        for i, prev_box in enumerate(
            prev_boxs
        ):  # Apply a random offset to each box, not uniformly
            sample_offsets = candidate_offsets[i]
            ref_box = points_utils.getOffsetBB(
                prev_box,
                sample_offsets,
                limit_box=config.data_limit_box,
                degrees=config.degrees,
            )
            ref_boxs.append(ref_box)
        candidate_shared_transform = np.zeros(3, dtype=np.float32)
        candidate_shared_world_translation = np.zeros(3, dtype=np.float32)
    candidate_history = list(ref_boxs)

    use_ct_joint_full = bool(getattr(config, "use_ct_joint_full", False))
    observation_only = bool(data.get("ct_observation_only", False))
    use_search_evidence_v3 = use_ct_joint_full and not observation_only
    if use_search_evidence_v3 and not bool(getattr(config, "use_b1motion_v3", False)):
        raise ValueError("B2-v3 requires B1motion-v3 shared history")
    b2_v3_history_mode = None
    if use_search_evidence_v3:
        b2_v3_history_mode = select_b2_v3_history_mode(
            data.get("tracklet_key", data.get("tracklet_id", "unknown")),
            this_frame_id if this_frame_id is not None else sample_index,
            candidate_id,
            seed=int(getattr(config, "seed", 42)),
        )
    ct_motion_history_boxs, ct_search_history_boxs = build_ct_training_histories(
        recursive_history,
        ref_boxs,
        candidate_offsets,
        candidate_id,
        candidate_trajectory_mode,
        training_mode=(
            b2_v3_history_mode
            if use_search_evidence_v3
            else getattr(config, "ct_history_training_mode", "canonical")
        ),
        correlation=float(getattr(config, "ct_history_correlation", 0.75)),
        recursive_error_scale=float(
            getattr(config, "ct_history_recursive_error_scale", 1.0)
        ),
        degrees=config.degrees,
    )
    if online_recursive_state is not None:
        # Candidate recovery views are coherent transformations of the full
        # deployed state.  Do not synthesize a second, independently perturbed
        # motion or Search history.
        ct_motion_history_boxs = candidate_history
        ct_search_history_boxs = candidate_history
    canonical_label_boxs = ground_truth_history
    canonical_ref_boxs = boxes_to_anchor_parameters(
        canonical_label_boxs, canonical_label_boxs[0], degrees=config.degrees
    )
    # Preserve the formal B1 auxiliary tensor in the observation anchor. The
    # deployed B1 contract below independently uses recursive_history and
    # motion_anchor_box, so recovery candidates cannot perturb its prior.
    ordered_motion_history_boxs = ct_motion_history_boxs
    ordered_motion_anchor = recursive_history[0]
    ct_motion_ref_boxs = boxes_to_anchor_parameters(
        ordered_motion_history_boxs,
        ordered_motion_anchor,
        degrees=config.degrees,
    )
    use_motion_v3 = bool(getattr(config, "use_b1motion_v3", False))
    causal_temporal_policy = str(
        getattr(config, "ct_candidate_policy", "legacy_spatial")
    ).strip().lower() in ("causal_b1_boundary", "causal_temporal_uniform")
    motion_main_ref_boxs = None
    if use_motion_v3:
        # Match recursive inference: the newest trajectory box is the actual
        # crop anchor (therefore exactly zero in local coordinates), while
        # older estimated-box errors evolve smoothly.  Using the clean newest
        # GT box here would expose candidate anchor error that is unavailable
        # online and recreate the v2 identifiability bug.
        # B1 is candidate-view invariant. Recovery views alter the B0 crop and
        # Search anchor, not the physical history signal.
        motion_main_ref_boxs = boxes_to_anchor_parameters(
            recursive_history,
            motion_anchor_box,
            degrees=config.degrees,
        )
        if not np.allclose(motion_main_ref_boxs[0], 0.0, rtol=0.0, atol=1e-5):
            raise ValueError(
                "B1motion-v3 newest main history must equal the crop anchor"
            )
        if use_search_evidence_v3:
            shared_search_ref_boxs = motion_main_ref_boxs.copy()
            if not np.array_equal(motion_main_ref_boxs, shared_search_ref_boxs):
                raise RuntimeError(
                    "B2-v3 requires byte-identical B1/B2 history tensors"
                )

    real_time_fields = build_time_fields(
        prev_timestamps,
        current_timestamp,
        frame_ids=prev_frame_ids,
        current_frame_id=this_frame_id,
        use_real_time=use_real_time,
        default_step=default_time_step,
        pseudo_step=pseudo_time_step,
    )
    relative_timestamps, delta_t_list, local_timestamps, current_timestamp = (
        real_time_fields
    )
    dynamics_time_mode = normalize_dynamics_time_mode(
        this_frame.get(
            "_ct_dynamics_time_mode", getattr(config, "dynamics_time_mode", "true")
        )
    )
    effective_time_fields = build_effective_time_fields(
        dynamics_time_mode,
        real_time_fields,
        effective_frame_timestamps=[
            prev_frames[key].get("_ct_effective_timestamp") for key in sorted_prev_keys
        ],
        effective_current_timestamp=this_frame.get("_ct_effective_timestamp"),
        frame_ids=prev_frame_ids,
        current_frame_id=this_frame_id,
        default_step=float(
            getattr(config, "dynamics_fixed_delta_t", default_time_step)
        ),
        pseudo_step=pseudo_time_step,
    )
    (
        effective_relative_timestamps,
        effective_delta_t_list,
        effective_local_timestamps,
        effective_current_timestamp,
    ) = effective_time_fields
    motion_aux_contract = build_motion_aux_contract(
        data=data,
        config=config,
        use_motion_v3=use_motion_v3,
        observation_only=observation_only,
        causal_temporal_policy=causal_temporal_policy,
        num_hist=num_hist,
        this_frame=this_frame,
        canonical_this_box=canonical_this_box,
        current_timestamp=current_timestamp,
        this_frame_id=this_frame_id,
        dynamics_time_mode=dynamics_time_mode,
        use_real_time=use_real_time,
        default_time_step=default_time_step,
        pseudo_time_step=pseudo_time_step,
    )
    main_current_value = float(getattr(config, "main_time_current", 0.0))
    point_timestamps, corner_timestamps, main_timestamps = build_main_time_fields(
        valid_mask,
        relative_timestamps,
        local_timestamps,
        num_hist,
        pseudo_step=pseudo_time_step,
        source=getattr(config, "main_time_source", "real"),
        current_value=main_current_value,
    )

    prev_frame_pcs = []
    for i, prev_pc in enumerate(prev_pcs):
        prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(
            prev_pc,
            ref_boxs[i],
            ref_boxs[0],
            scale=config.bb_scale,
            offset=config.bb_offset,
        )
        prev_frame_pcs.append(prev_frame_pc)

    this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(
        this_pc,
        ref_boxs[0],
        ref_boxs[0],
        scale=config.bb_scale,
        offset=config.bb_offset,
    )
    baseline_search_points = this_frame_pc.points.T
    ct_search_box = None
    ct_search_diagnostics = {
        "valid": False,
        "query_delta_t": float(effective_delta_t_list[0]),
    }
    expanded_search_points = np.empty(
        (0, baseline_search_points.shape[1]),
        dtype=baseline_search_points.dtype,
    )
    use_trajectory_search = use_ct_joint_full and not observation_only
    if use_trajectory_search:
        search_history_mode = (
            str(getattr(config, "ct_search_training_history", "canonical"))
            .strip()
            .lower()
        )
        if search_history_mode not in (
            "canonical",
            "candidate",
            "correlated_candidate",
        ):
            raise ValueError(
                "ct_search_training_history must be canonical, candidate, "
                "or correlated_candidate"
            )
        if search_history_mode == "canonical":
            search_history_boxes = recursive_history
        elif search_history_mode == "candidate":
            search_history_boxes = ref_boxs
        else:
            search_history_boxes = ct_search_history_boxs
        ct_search_box, ct_search_diagnostics = build_ordered_trajectory_search_box(
            search_history_boxes,
            effective_delta_t_list,
            valid_mask=valid_mask,
            base_length=float(getattr(config, "trajectory_search_base_length", 4.0)),
            base_width=float(getattr(config, "trajectory_search_base_width", 2.0)),
            max_length=float(getattr(config, "ct_tube_max_length", 24.0)),
            max_width=float(getattr(config, "ct_tube_max_width", 8.0)),
            max_speed=float(getattr(config, "ct_motion_max_speed", 20.0)),
            max_acceleration=float(getattr(config, "ct_motion_max_acceleration", 8.0)),
            max_displacement=float(getattr(config, "ct_motion_max_displacement", 12.0)),
            acceleration_weight=float(
                getattr(config, "ct_motion_acceleration_weight", 0.5)
            ),
            sigma_parallel_scale=float(
                getattr(config, "trajectory_search_sigma_parallel_scale", 2.0)
            ),
            sigma_perpendicular_scale=float(
                getattr(config, "trajectory_search_sigma_perpendicular_scale", 2.0)
            ),
            min_displacement=float(
                getattr(config, "trajectory_search_min_displacement", 0.2)
            ),
            min_delta_t=float(getattr(config, "trajectory_search_min_delta_t", 0.75)),
            min_gap_ratio=float(
                getattr(config, "trajectory_search_min_gap_ratio", 1.5)
            ),
            allow_normal_cadence=True,
            require_recent_transition=True,
        )
        if ct_search_box is not None:
            expanded_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                this_pc, ct_search_box, ref_boxs[0], scale=1.0, offset=0.0
            )
            expanded_search_points = expanded_search_pc.points.T

    use_endpoint_search_evidence = use_search_evidence_v3

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
                return getattr(config, field, default)
        return getattr(config, f"search_v3_{name}", default)

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
        search_v2_history_mode = b2_v3_history_mode
        search_v2_history_boxes = recursive_history
        use_prepass_support = bool(getattr(config, "use_b1_prepass_support", False))
        support_prediction = data.get("motion_prediction")
        if isinstance(support_prediction, dict) and bool(
            support_prediction.get("valid", False)
        ):
            support_prediction = reexpress_motion_prediction(
                support_prediction, motion_anchor_box, ref_boxs[0]
            )
        support_kwargs = dict(
            prediction=support_prediction,
            use_b1_prepass=use_prepass_support,
            use_dynamic_sigma=bool(
                getattr(config, "search_v3_use_dynamic_sigma", False)
            ),
            fixed_margins=(
                float(getattr(config, "search_v3_fixed_margin_parallel", 2.0)),
                float(getattr(config, "search_v3_fixed_margin_perpendicular", 1.0)),
            ),
            coverage_scale=float(getattr(config, "search_v3_coverage_scale", 2.448)),
            standardized_residual_quantile=tuple(
                getattr(
                    config,
                    "search_v3_standardized_residual_q90_parallel_perpendicular",
                    (1.0, 1.0),
                )
            ),
            min_direction_speed=float(
                getattr(config, "motion_v3_min_direction_speed", 0.2)
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
                0.0
                if use_ct_joint_full
                else float(search_config_value("min_displacement", 0.2))
            ),
            fallback_require_recent_transition=use_ct_joint_full,
        )
        (search_v2_box, ct_search_box, search_v2_diagnostics) = (
            resolve_joint_search_geometry(
                search_v2_history_boxes,
                effective_delta_t_list,
                valid_mask,
                **support_kwargs,
            )
        )
        if ct_search_box is not None:
            ct_search_diagnostics = dict(search_v2_diagnostics)
            expanded_search_pc = points_utils.generate_subwindow_with_aroundboxs(
                this_pc, ct_search_box, ref_boxs[0], scale=1.0, offset=0.0
            )
            expanded_search_points = expanded_search_pc.points.T
        if search_v2_box is not None:
            learned_prior_support = search_v2_diagnostics.get("prior_source") == "b1"
            search_v2_expanded_pc = points_utils.generate_subwindow_with_aroundboxs(
                this_pc,
                search_v2_box,
                ref_boxs[0],
                scale=(1.0 if learned_prior_support else config.bb_scale),
                offset=(0.0 if learned_prior_support else config.bb_offset),
            )
            search_v2_expanded_points = search_v2_expanded_pc.points.T
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

    # Preserve the pre-normalization anchor: ref_boxs[0] becomes approximately
    # zero after transform_box(), so the normalized tensor cannot prove that
    # paired views shared the same crop and local coordinate system.
    coordinate_anchor_box = ref_boxs[0]
    coordinate_anchor_theta = (
        coordinate_anchor_box.orientation.degrees
        * coordinate_anchor_box.orientation.axis[-1]
        if config.degrees
        else coordinate_anchor_box.orientation.radians
        * coordinate_anchor_box.orientation.axis[-1]
    )
    coordinate_anchor = np.append(
        coordinate_anchor_box.center, coordinate_anchor_theta
    ).astype("float32")
    motion_anchor_theta = (
        motion_anchor_box.orientation.degrees * motion_anchor_box.orientation.axis[-1]
        if config.degrees
        else motion_anchor_box.orientation.radians
        * motion_anchor_box.orientation.axis[-1]
    )
    motion_anchor = np.append(motion_anchor_box.center, motion_anchor_theta).astype(
        "float32"
    )

    this_box = points_utils.transform_box(this_box, ref_boxs[0])
    prev_boxs = [
        points_utils.transform_box(prev_box, ref_boxs[0]) for prev_box in prev_boxs
    ]
    ref_boxs = [
        points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs
    ]
    motion_boxs = [
        points_utils.transform_box(this_box, prev_box) for prev_box in prev_boxs
    ]

    # Resample each frame of the point cloud to a specific number
    prev_regularized = [
        points_utils.regularize_pc_with_metadata(
            prev_frame_pc.points.T, config.point_sample_size, seed=seed
        )
        for prev_frame_pc, seed in zip(prev_frame_pcs, prev_sampling_seeds)
    ]
    prev_points_list = [item.points for item in prev_regularized]
    base_regularized = None
    trajectory_search_points = np.zeros(
        (
            int(
                getattr(config, "ct_tube_quota", 128)
                if use_ct_joint_full
                else getattr(config, "trajectory_search_point_count", 128)
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
            config.point_sample_size,
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
            sample_size=int(getattr(config, "ct_tube_quota", 128)),
            extension_quota=int(getattr(config, "ct_tube_quota", 128)),
            min_points=int(getattr(config, "ct_search_min_points", 3)),
            seed=current_sampling_seed,
        )
        ct_search_sampling = {
            "baseline_sample_count": int(config.point_sample_size),
            "expansion_sample_count": int(trajectory_search_sampling["sample_count"]),
            "expansion_available_count": int(
                trajectory_search_sampling["available_count"]
            ),
        }
    else:
        base_regularized = points_utils.regularize_pc_with_metadata(
            baseline_search_points,
            config.point_sample_size,
            seed=current_sampling_seed,
        )
        this_points = base_regularized.points
        ct_search_sampling = {
            "baseline_sample_count": int(config.point_sample_size),
            "expansion_sample_count": 0,
            "expansion_available_count": 0,
        }

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
        independent_seed_base = (
            current_sampling_seed if current_sampling_seed is not None else sample_index
        )
        search_v2_seed = (
            int(independent_seed_base) * 1664525 + 1013904223
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
    joint_extension_sampling = None
    joint_extension_source = None
    data_dict = build_sample_output(locals())
    return data_dict
