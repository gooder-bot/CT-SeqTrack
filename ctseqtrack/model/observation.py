"""Formal v25 observation and B1 dispatch path.

This module contains only the B0 observation computation and the physical
time prior used by contract-v3.  Historical dynamics, fusion and router
experiments deliberately do not participate in this path.
"""

import torch

from datasets import points_utils
from datasets.misc_utils import (
    create_corner_timestamps_from_deltas,
    get_tensor_corners_batch,
)
from ctseqtrack.model.forward import forward_contract_v3


def resolve_observation_delta_t(input_dict, reference, default_time_step):
    """Return the real and effective gaps used by v25 observation logs."""
    batch_size = reference.shape[0]
    real_delta_t = input_dict.get(
        "current_delta_t_real", input_dict.get("current_delta_t")
    )
    if real_delta_t is None:
        real_delta_t = reference.new_full((batch_size,), float(default_time_step))
    elif not torch.is_tensor(real_delta_t):
        real_delta_t = torch.as_tensor(
            real_delta_t, device=reference.device, dtype=reference.dtype
        )
    else:
        real_delta_t = real_delta_t.to(device=reference.device, dtype=reference.dtype)
    if real_delta_t.numel() == 1 and batch_size > 1:
        real_delta_t = real_delta_t.repeat(batch_size)
    real_delta_t = real_delta_t.reshape(batch_size)

    effective_delta_t = input_dict.get("current_delta_t_effective", real_delta_t)
    if not torch.is_tensor(effective_delta_t):
        effective_delta_t = torch.as_tensor(
            effective_delta_t,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        effective_delta_t = effective_delta_t.to(
            device=reference.device, dtype=reference.dtype
        )
    if effective_delta_t.numel() == 1 and batch_size > 1:
        effective_delta_t = effective_delta_t.repeat(batch_size)
    return real_delta_t, effective_delta_t.reshape(batch_size)


def encode_point_time(model, points):
    encoded_time = model.time_encoder(points[..., 3:4])
    return torch.cat((points[..., :3], encoded_time, points[..., 4:]), dim=-1)


def build_observability_stats(model, input_dict, seg_logits, chunk_size):
    batch_size = seg_logits.shape[0]
    device = seg_logits.device
    dtype = seg_logits.dtype
    if "num_points_in_search" in input_dict:
        num_points = input_dict["num_points_in_search"]
        if not torch.is_tensor(num_points):
            num_points = torch.as_tensor(num_points, device=device, dtype=dtype)
        num_points = num_points.to(device=device, dtype=dtype).reshape(batch_size)
    else:
        num_points = seg_logits.new_full((batch_size,), float(chunk_size))

    current_logits = seg_logits[:, :, -chunk_size:]
    foreground_probability = torch.softmax(current_logits, dim=1)[:, 1, :]
    if getattr(model.config, "obs_stats_detach_seg", True):
        foreground_probability = foreground_probability.detach()
    soft_foreground_count = foreground_probability.sum(dim=1)
    mean_foreground_score = foreground_probability.mean(dim=1)
    bounded_probability = torch.clamp(foreground_probability, min=1e-6, max=1.0 - 1e-6)
    segmentation_entropy = -(
        bounded_probability * torch.log(bounded_probability)
        + (1.0 - bounded_probability) * torch.log(1.0 - bounded_probability)
    ).mean(dim=1)
    estimated_foreground_points = mean_foreground_score * torch.clamp(
        num_points, min=0.0
    )

    history_ratio = input_dict["valid_mask"].to(device=device, dtype=dtype).mean(dim=1)
    default_step = getattr(
        model.config, "default_time_step", getattr(model.config, "time_step", 0.5)
    )
    time_scale = max(float(getattr(model.config, "time_scale", default_step)), 1e-6)
    real_delta_t, effective_delta_t = resolve_observation_delta_t(
        input_dict, seg_logits, default_step
    )
    # Formal v25 never enables the removed CT-v2 dynamics branch, so its
    # observation gap is the real physical gap.
    current_delta_t_ratio = real_delta_t / time_scale
    real_delta_t_ratio = real_delta_t / time_scale
    effective_delta_t_ratio = effective_delta_t / time_scale

    statistics = torch.stack(
        (
            torch.log1p(torch.clamp(num_points, min=0.0)),
            torch.log1p(torch.clamp(estimated_foreground_points, min=0.0)),
            mean_foreground_score,
            history_ratio,
            current_delta_t_ratio,
        ),
        dim=1,
    )
    statistics = torch.nan_to_num(statistics, nan=0.0, posinf=0.0, neginf=0.0)
    auxiliary = {
        "obs_stats": statistics,
        "obs_num_points_search": num_points,
        "obs_soft_fg_count": soft_foreground_count,
        "obs_estimated_fg_points": estimated_foreground_points,
        "obs_mean_fg_score": mean_foreground_score,
        "obs_segmentation_entropy": segmentation_entropy,
        "obs_valid_history_ratio": history_ratio,
        "obs_current_delta_t_ratio": current_delta_t_ratio,
        "obs_current_delta_t_real_ratio": real_delta_t_ratio,
        "obs_current_delta_t_effective_ratio": effective_delta_t_ratio,
    }
    return statistics, auxiliary


def _observation_backbone(model, input_dict):
    output = {}
    points = encode_point_time(model, input_dict["points"])
    point_tensor = points.transpose(1, 2)
    if model.box_aware:
        candidate_bc = input_dict["candidate_bc"].transpose(1, 2)
        point_tensor = torch.cat((point_tensor, candidate_bc), dim=1)

    batch_size, _, point_count = point_tensor.shape
    history_length = input_dict["valid_mask"].shape[1]
    frame_count = history_length + 1
    chunk_size = point_count // frame_count
    segmentation = model.seg_pointnet(point_tensor)
    segmentation_logits = segmentation[:, :2, :]
    observation_stats, observation_aux = build_observability_stats(
        model, input_dict, segmentation_logits, chunk_size
    )
    prediction_class = torch.argmax(segmentation_logits, dim=1, keepdim=True)
    masked_points = point_tensor[:, :4, :] * prediction_class
    if model.box_aware:
        predicted_bc = segmentation[:, 2:, :]
        masked_points = torch.cat(
            (masked_points, predicted_bc * prediction_class), dim=1
        )
        output["pred_bc"] = predicted_bc.transpose(1, 2)

    point_feature = model.mini_pointnet(masked_points)
    motion_prediction = model.motion_mlp(point_feature)
    if model.use_motion_cls:
        motion_state_logits = model.motion_state_mlp(point_feature)
        motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)
        masked_motion = motion_prediction * motion_mask
        output["motion_cls"] = motion_state_logits
    else:
        masked_motion = motion_prediction
    coarse_box = points_utils.get_offset_box_tensor(
        torch.zeros_like(motion_prediction), masked_motion
    )

    repeated_size = input_dict["bbox_size"].repeat_interleave(frame_count, dim=0)
    box_sequence = torch.cat((input_dict["ref_boxs"], coarse_box.unsqueeze(1)), dim=1)
    flat_boxes = box_sequence.reshape(batch_size * frame_count, 4)
    corners = get_tensor_corners_batch(
        flat_boxes[:, :3], repeated_size, flat_boxes[:, -1]
    )
    corners = corners.reshape(batch_size, frame_count * 8, -1)
    corner_stamps = create_corner_timestamps_from_deltas(
        input_dict["delta_T"],
        8,
        current_time=getattr(model.config, "main_time_current", 0.0),
    ).to(device=corners.device, dtype=corners.dtype)
    corners = torch.cat((corners, model.time_encoder(corner_stamps)), dim=-1)

    per_frame_points = point_tensor.reshape(batch_size * frame_count, -1, chunk_size)
    collect_b2_features = bool(
        model.use_ct_joint_full
        and model.ct_joint_contract_version >= 3
        and model.ct_enable_b2
    )
    feature_result = model.feature_pointnet(
        per_frame_points, return_point_features=collect_b2_features
    )
    if collect_b2_features:
        feature, point_features = feature_result
        point_features = point_features.transpose(1, 2).reshape(
            batch_size, frame_count, chunk_size, -1
        )
        if point_features.shape[-1] != 64:
            raise RuntimeError("FeaturePointNet second-layer features must be 64d")
        output["b0_point_aligned_features"] = point_features
        raw_frame_points = input_dict["points"].reshape(
            batch_size, frame_count, chunk_size, -1
        )
        explicit_base = input_dict.get("ct_base_evidence_points")
        if explicit_base is None:
            raise KeyError("contract-v3 B2 requires ct_base_evidence_points")
        explicit_base = explicit_base.to(
            device=raw_frame_points.device, dtype=raw_frame_points.dtype
        )
        if explicit_base.shape != raw_frame_points[:, -1].shape:
            raise ValueError("base evidence must have shape [B,1024,5]")
        if not torch.equal(explicit_base, raw_frame_points[:, -1]):
            raise RuntimeError("base evidence diverged from B0 current 1024 points")
        output["ct_base_evidence_points"] = explicit_base
    else:
        feature = feature_result

    feature = feature.transpose(1, 2)
    token_count = feature.shape[1]
    sequence_features = feature.reshape(batch_size, frame_count * token_count, -1)
    if model.use_ct_joint_full:
        delta_motion, decoder_state = model.Transformer(
            corners,
            sequence_features,
            input_dict["valid_mask"],
            return_decoder_state=True,
        )
        observation_query = decoder_state[:, -1]
        output["decoder_state"] = decoder_state
        output["observation_query"] = observation_query
    else:
        delta_motion = model.Transformer(
            corners, sequence_features, input_dict["valid_mask"]
        )
        observation_query = None
    if model.training and model.ct_b0_rng_shift_control:
        torch.rand((batch_size,), device=feature.device)

    return {
        "output": output,
        "point_feature": point_feature,
        "segmentation_logits": segmentation_logits,
        "observation_stats": observation_stats,
        "observation_aux": observation_aux,
        "observation_query": observation_query,
        "motion_prediction": motion_prediction,
        "coarse_box": coarse_box,
        "updated_reference_boxes": delta_motion[:, :history_length, :],
        "observation_box": delta_motion[:, -1, :],
        "batch_size": batch_size,
        "frame_count": frame_count,
        "chunk_size": chunk_size,
    }


def _physical_prior(model, input_dict, context):
    output = context["output"]
    required = (
        "motion_main_ref_boxs",
        "motion_main_delta_t",
        "motion_main_current_delta_t",
        "motion_main_valid_mask",
    )
    missing = [key for key in required if key not in input_dict]
    if missing:
        raise KeyError("B1motion-v3 input is missing: " + ", ".join(missing))
    main_reference_boxes = input_dict["motion_main_ref_boxs"]
    if bool(getattr(model.config, "shuffle_b1_signal", False)):
        main_reference_boxes = torch.flip(main_reference_boxes, dims=(1,))
    if model.use_ct_joint_full and not model.ct_enable_b1:
        main_motion = model.physical_motion_encoder.kinematic_fallback(
            main_reference_boxes,
            input_dict["motion_main_delta_t"],
            input_dict["motion_main_valid_mask"],
            input_dict["motion_main_current_delta_t"],
        )
    else:
        main_motion = model.physical_motion_encoder(
            main_reference_boxes,
            input_dict["motion_main_delta_t"],
            input_dict["motion_main_valid_mask"],
            input_dict["motion_main_current_delta_t"],
        )
    if bool(getattr(model.config, "force_b1_invalid", False)):
        main_motion = dict(main_motion)
        main_motion["valid"] = torch.zeros_like(main_motion["valid"])
        main_motion["source_id"] = torch.zeros_like(main_motion["source_id"])

    origin = input_dict["motion_main_ref_boxs"][:, 0, :2].to(
        device=main_motion["prior_xy"].device, dtype=main_motion["prior_xy"].dtype
    )
    output.update(
        {
            "motion_prior_xy": main_motion["prior_xy"],
            "motion_prior_origin_xy": origin,
            "motion_prior_proposal_xy": main_motion["prior_xy"],
            "motion_prior_basis_velocity_xy": main_motion["basis_velocity_xy"],
            "motion_prior_velocity_xy": main_motion["velocity_xy"],
            "motion_prior_kinematic_xy": main_motion["kinematic_prior_xy"],
            "motion_prior_residual_xy": main_motion["residual_xy"],
            "motion_prior_residual_unit_parallel_perp": main_motion[
                "residual_unit_parallel_perp"
            ],
            "motion_prior_envelope_parallel_perp": main_motion[
                "envelope_parallel_perp"
            ],
            "motion_prior_valid": main_motion["valid"],
            "motion_prior_log_sigma_xy": main_motion["log_sigma_xy"],
            "motion_prior_log_sigma_parallel_perp": main_motion[
                "log_sigma_parallel_perp"
            ],
            "motion_prior_covariance_xy": main_motion["covariance_xy"],
            "motion_prior_direction_xy": main_motion["motion_direction_xy"],
            "motion_prior_source_id": main_motion["source_id"],
            "motion_prior_gap_ratio": main_motion["gap_ratio"],
        }
    )
    if (
        model.training
        and "motion_aux_ref_boxs" in input_dict
        and not (model.use_ct_joint_full and not model.ct_enable_b1)
    ):
        auxiliary_motion = model.physical_motion_encoder(
            input_dict["motion_aux_ref_boxs"],
            input_dict["motion_aux_delta_t"],
            input_dict["motion_aux_valid_mask"],
            input_dict["motion_aux_current_delta_t"],
        )
        output.update(
            {
                "motion_aux_prior_xy": auxiliary_motion["prior_xy"],
                "motion_aux_prior_velocity_xy": auxiliary_motion["velocity_xy"],
                "motion_aux_prior_kinematic_xy": auxiliary_motion["kinematic_prior_xy"],
                "motion_aux_prior_valid": auxiliary_motion["valid"],
                "motion_aux_prior_gap_ratio": auxiliary_motion["gap_ratio"],
                "motion_aux_prior_log_sigma_xy": auxiliary_motion["log_sigma_xy"],
                "motion_aux_prior_log_sigma_parallel_perp": auxiliary_motion[
                    "log_sigma_parallel_perp"
                ],
                "motion_aux_prior_direction_xy": auxiliary_motion[
                    "motion_direction_xy"
                ],
            }
        )
    history_ratio = (
        input_dict["motion_main_valid_mask"]
        .to(
            device=context["point_feature"].device, dtype=context["point_feature"].dtype
        )
        .mean(dim=1)
    )
    return main_motion, history_ratio


def forward_v25(model, input_dict):
    """Run B0, then B1, B2 and B3 in the registered v25 order."""
    context = _observation_backbone(model, input_dict)
    output = context["output"]
    final_box = context["observation_box"]
    main_motion = None
    history_ratio = None
    if model.use_b1motion_v3:
        main_motion, history_ratio = _physical_prior(model, input_dict, context)

    if model.use_ct_joint_full:
        required = (
            "search_v3_points",
            "search_v3_point_valid_mask",
            "search_v3_point_source",
            "search_v3_branch_source",
            "trajectory_search_points",
            "trajectory_search_point_valid_mask",
            "trajectory_search_point_source",
            "trajectory_search_branch_source",
            "trajectory_search_valid",
            "search_v3_support_valid",
            "search_v3_query_delta_t",
            "search_v3_gap_ratio",
        )
        missing = [key for key in required if key not in input_dict]
        if missing:
            raise KeyError("CT joint Full input is missing: " + ", ".join(missing))
        if context["observation_query"] is None:
            raise RuntimeError(
                "CT joint Full requires the final observation decoder query"
            )
        if model.ct_enable_b2:
            final_box = forward_contract_v3(
                model,
                input_dict,
                output,
                context["observation_box"],
                context["observation_stats"],
                main_motion,
                history_ratio,
                context["batch_size"],
                context["frame_count"],
                context["chunk_size"],
                coarse_box=context["coarse_box"],
            )

    output.update(
        {
            "estimation_boxes": context["coarse_box"],
            "seg_logits": context["segmentation_logits"],
            "motion_pred": context["motion_prediction"],
            "observation_aux_estimation_boxes": context["observation_box"],
            "aux_estimation_boxes": final_box,
            "ref_boxs": input_dict["ref_boxs"],
            "valid_mask": input_dict["valid_mask"],
            "updated_ref_boxs": context["updated_reference_boxes"],
        }
    )
    output.update(context["observation_aux"])
    for key in (
        "ct_search_used",
        "ct_search_expansion_ratio",
        "ct_search_baseline_points",
        "ct_search_expansion_points",
        "ct_search_query_delta_t",
        "ct_search_predicted_displacement",
        "trajectory_search_valid",
        "trajectory_search_gap_ratio",
        "trajectory_search_sigma_parallel",
        "trajectory_search_sigma_perpendicular",
        "search_has_usable_points",
    ):
        if key in input_dict:
            output[key] = input_dict[key]
    return output


__all__ = [
    "build_observability_stats",
    "encode_point_time",
    "forward_v25",
    "resolve_observation_delta_t",
]
