"""Current contract-v3 B0--B3 forward transaction."""

import torch

from ctseqtrack.contracts import (
    DecisionOutput,
    EvidenceOutput,
    MotionPriorOutput,
    reexpress_motion_prior,
    validate_motion_prior_support_alignment,
)
from ctseqtrack.model.evidence import (
    apply_memory_control,
    build_box_memory_tokens,
)
from ctseqtrack.runtime.calibration import require_selective_calibration


def forward_contract_v3(
    self,
    input_dict,
    output_dict,
    observation_box,
    observation_stats,
    main_motion,
    history_valid_ratio,
    batch_size,
    frame_count,
    chunk_size,
    coarse_box=None,
):
    """Run the extension-query/full-base-memory contract-v3 path."""
    required = (
        "ct_base_evidence_points",
        "ct_base_evidence_valid_mask",
        "ct_extension_points",
        "ct_extension_valid_mask",
        "ct_extension_source",
        "search_v3_query_delta_t",
        "search_v3_gap_ratio",
        "search_v3_support_anchor_xy",
        "motion_source_anchor",
        "coordinate_anchor",
        "ref_boxs",
        "bbox_size",
        "valid_mask",
    )
    missing = [key for key in required if key not in input_dict]
    if missing:
        raise KeyError("CT contract-v3 input is missing: " + ", ".join(missing))
    aligned_features = output_dict["b0_point_aligned_features"]
    if aligned_features.shape != (batch_size, frame_count, chunk_size, 64):
        raise ValueError("B0 point-aligned feature contract must be [B,L,1024,64]")
    if chunk_size != 1024:
        raise ValueError("contract-v3 requires exactly 1024 B0 points")
    raw_frames = input_dict["points"].reshape(batch_size, frame_count, chunk_size, -1)
    observation_contract = self.b0_observation_contract(
        observation_box,
        observation_stats,
        current_features=aligned_features[:, -1],
        history_features=aligned_features[:, :-1],
    )
    observation_box = observation_contract.box
    observation_stats = observation_contract.statistics
    history_features = observation_contract.history_features
    current_base_features = observation_contract.current_features
    timestamps_real = input_dict.get("timestamps_real")
    history_timestamps = None
    current_timestamp = None
    if timestamps_real is not None:
        timestamps_real = timestamps_real.to(
            device=current_base_features.device, dtype=current_base_features.dtype
        )
        history_timestamps = timestamps_real[:, :-1]
        current_timestamp = timestamps_real[:, -1]
    memory_tokens, memory_valid, memory_metadata = build_box_memory_tokens(
        history_features,
        raw_frames[:, :-1],
        input_dict["ref_boxs"],
        input_dict["bbox_size"],
        input_dict["valid_mask"],
        foreground_tokens=8,
        context_tokens=4,
        history_timestamps=history_timestamps,
        current_timestamp=current_timestamp,
        current_box=observation_box.detach(),
        return_metadata=True,
    )
    if memory_tokens.shape[1] != 36:
        raise RuntimeError("contract-v3 requires exactly 36 memory slots")
    memory_mode = str(getattr(self.config, "ct_memory_mode", "real")).strip().lower()
    memory_tokens, memory_valid, memory_metadata = apply_memory_control(
        memory_tokens, memory_valid, memory_metadata, memory_mode
    )
    base_evidence_mode = (
        str(getattr(self.config, "ct_base_evidence_mode", "full")).strip().lower()
    )
    base_valid_mask = input_dict["ct_base_evidence_valid_mask"].to(
        current_base_features.device
    )
    if base_evidence_mode == "empty":
        base_valid_mask = torch.zeros_like(base_valid_mask)
    elif base_evidence_mode != "full":
        raise ValueError("ct_base_evidence_mode must be full or empty")
    extension_points = input_dict["ct_extension_points"].to(
        device=current_base_features.device, dtype=current_base_features.dtype
    )
    extension_points = self.encode_point_time(extension_points)
    if extension_points.shape[1:] != (256, 5):
        raise ValueError("extension points must have shape [B,256,5]")
    recursive_age = input_dict.get("ct_recursive_state_age")
    if recursive_age is not None:
        recursive_age = recursive_age.to(
            device=current_base_features.device, dtype=current_base_features.dtype
        )
    canonical_prior_contract = MotionPriorOutput(
        center_xy=main_motion["prior_xy"].detach(),
        direction_xy=main_motion["motion_direction_xy"].detach(),
        log_sigma=main_motion["log_sigma_parallel_perp"].detach(),
        valid=main_motion["valid"].detach(),
        source=input_dict["search_v3_prior_source_id"]
        .to(current_base_features.device)
        .detach(),
        mode_centers_xy=main_motion["mode_centers_xy"].detach(),
        mode_probabilities=main_motion["mode_probabilities"].detach(),
        motion_quantiles_pp=main_motion["motion_quantiles_pp"].detach(),
        support_quantiles_pp=main_motion["support_quantiles_pp"].detach(),
        recoverability_probability=main_motion["recoverability_probability"].detach(),
        expert_disagreement=main_motion["expert_disagreement"].detach(),
        residual_acceleration_pp=main_motion["residual_acceleration_pp"].detach(),
        residual_gate=main_motion["residual_gate"].detach(),
    )
    prior_contract = reexpress_motion_prior(
        canonical_prior_contract,
        input_dict["motion_source_anchor"],
        input_dict["coordinate_anchor"],
        degrees=bool(getattr(self.config, "degrees", False)),
    )
    support_alignment_error = validate_motion_prior_support_alignment(
        prior_contract,
        input_dict["search_v3_support_anchor_xy"],
        tolerance=1e-3,
    )
    with self.ct_plugin_rng.fork(current_base_features.device):
        joint_output = self.ct_joint_search_refiner(
            extension_points=extension_points,
            extension_valid_mask=input_dict["ct_extension_valid_mask"].to(
                current_base_features.device
            ),
            extension_source=input_dict["ct_extension_source"].to(
                current_base_features.device
            ),
            current_base_features=current_base_features,
            current_base_valid_mask=base_valid_mask,
            memory_tokens=memory_tokens,
            memory_valid_mask=memory_valid,
            memory_metadata=memory_metadata,
            observation_box=observation_box,
            observation_stats=observation_stats,
            b1_center_xy=prior_contract.center_xy,
            b1_sigma_parallel_perp=torch.exp(prior_contract.log_sigma),
            b1_direction_xy=prior_contract.direction_xy,
            b1_valid=prior_contract.valid,
            query_delta_t=input_dict["search_v3_query_delta_t"].to(
                current_base_features.device
            ),
            gap_ratio=input_dict["search_v3_gap_ratio"].to(
                current_base_features.device
            ),
            recursive_age=recursive_age,
        )
    evidence_contract = EvidenceOutput(
        raw_box=joint_output["ct_b2_raw_box"],
        structural_available=joint_output["ct_b2_available"],
        presence_logit=joint_output["ct_b2_extension_presence_logit"],
        targetness=joint_output["ct_search_targetness_logits"],
        evidence_summary=torch.cat(
            (
                joint_output["ct_b2_base_evidence"],
                joint_output["ct_b2_extension_evidence"],
            ),
            dim=1,
        ),
        point_diagnostics={
            "entropy": joint_output["ct_search_targetness_entropy"],
            "ess": joint_output["ct_search_normalized_ess"],
            "count": joint_output["ct_search_extension_selected_count"],
        },
    )
    raw_box = evidence_contract.raw_box
    if self.ct_enable_b3:
        # B3 is currently deterministic, but keeping it in the plugin
        # stream makes the RNG ownership contract explicit and protects
        # canonical B0 if stochastic routing is introduced later.
        with self.ct_plugin_rng.fork(current_base_features.device):
            final_box, router_output = self.ct_joint_router(
                observation_box=observation_box,
                raw_box=raw_box,
                availability=evidence_contract.structural_available,
                base_evidence=joint_output["ct_b2_base_evidence"],
                extension_evidence=joint_output["ct_b2_extension_evidence"],
                base_presence_probability=joint_output[
                    "ct_b2_base_presence_probability"
                ],
                extension_presence_probability=joint_output[
                    "ct_b2_extension_presence_probability"
                ],
                observation_stats=observation_stats,
                b1_sigma_parallel_perp=torch.exp(prior_contract.log_sigma),
                query_delta_t=input_dict["search_v3_query_delta_t"].to(
                    current_base_features.device
                ),
                gap_ratio=input_dict["search_v3_gap_ratio"].to(
                    current_base_features.device
                ),
                recursive_age=recursive_age,
                enabled=True,
                coarse_box=coarse_box,
                b1_center_xy=prior_contract.center_xy,
                targetness_entropy=evidence_contract.point_diagnostics["entropy"],
                normalized_ess=evidence_contract.point_diagnostics["ess"],
                extension_point_count=evidence_contract.point_diagnostics["count"],
                extension_voxel_count=input_dict.get("ct_search_extension_voxels"),
                targetness_mean=joint_output["ct_search_targetness_mean"],
                targetness_max=joint_output["ct_search_targetness_max"],
                b1_recoverability=prior_contract.recoverability_probability,
                b1_motion_q95=(
                    None
                    if prior_contract.motion_quantiles_pp is None
                    else prior_contract.motion_quantiles_pp[:, 2]
                ),
                b1_support_q95=(
                    None
                    if prior_contract.support_quantiles_pp is None
                    else prior_contract.support_quantiles_pp[:, 2]
                ),
                b1_mode_probabilities=prior_contract.mode_probabilities,
                b1_expert_disagreement=prior_contract.expert_disagreement,
                b1_support_saturation=input_dict.get("search_v3_support_saturated"),
            )
    else:
        dt = (
            input_dict["search_v3_query_delta_t"]
            .to(current_base_features.device)
            .reshape(batch_size)
            .clamp(min=0.0)
        )
        residual = raw_box[:, :2] - observation_box[:, :2]
        residual_norm = torch.linalg.norm(residual, dim=1)
        radius = torch.clamp(
            float(getattr(self.config, "ct_router_radius_base", 0.5))
            + float(getattr(self.config, "ct_router_radius_per_second", 0.5)) * dt,
            max=float(getattr(self.config, "ct_router_radius_max", 2.0)),
        )
        scale = torch.clamp(radius / residual_norm.clamp_min(1e-6), max=1.0)
        bounded_residual = residual * scale.unsqueeze(1)
        zeros = observation_box.new_zeros((batch_size,))
        final_box = observation_box
        router_output = {
            "ct_b3_help_logit": zeros,
            "ct_b3_harm_logit": zeros,
            "ct_b3_help_probability": zeros,
            "ct_b3_harm_probability": zeros,
            "ct_b3_expected_center_gain": zeros,
            "ct_b3_expected_iou_gain": zeros,
            "ct_b3_action_score": zeros,
            "ct_b3_calibrated": zeros,
            "ct_b3_h3_residual": zeros,
            "ct_b3_h3_utility": zeros,
            "ct_b3_final_gate": zeros,
            "ct_router_logit": zeros,
            "ct_router_gate": zeros,
            "ct_router_applied_gate": zeros,
            "ct_router_evidence_valid": joint_output["ct_search_candidate_valid"],
            "ct_router_bounded_residual_xy": bounded_residual,
            "ct_router_residual_xy": residual,
            "ct_router_radius": radius,
            "ct_router_clip_rate": (residual_norm > radius).to(observation_box.dtype),
            "ct_router_soft_box": observation_box,
        }
    if not self.training:
        mode = self.proposal_inference_mode
        if mode in ("obs", "obs_only", "observation"):
            final_box = observation_box
        elif mode == "raw_search":
            final_box = torch.where(
                joint_output["ct_search_candidate_valid"]
                .reshape(batch_size, 1)
                .to(torch.bool),
                raw_box,
                observation_box,
            )
        elif self.ct_enable_b3 and mode in ("full", "full_selective", "selective"):
            require_selective_calibration(self.ct_joint_router.calibrated, mode)
    decision_contract = DecisionOutput(
        final_box=final_box,
        help_logit=router_output["ct_b3_help_logit"],
        harm_logit=router_output["ct_b3_harm_logit"],
        expected_center_gain=router_output["ct_b3_expected_center_gain"],
        expected_iou_gain=router_output["ct_b3_expected_iou_gain"],
        applied=router_output["ct_b3_final_gate"],
        bounded_residual=router_output["ct_router_bounded_residual_xy"],
    )
    final_box = decision_contract.final_box
    joint_output.update(router_output)
    joint_output.update(
        {
            "ct_final_box": final_box,
            "candidate_valid": joint_output["ct_search_candidate_valid"],
            "ct_b1_candidate_center_xy": prior_contract.center_xy,
            "ct_b1_candidate_direction_xy": prior_contract.direction_xy,
            "ct_b1_support_alignment_error": support_alignment_error,
            "ct_search_geometry_valid": input_dict["ct_search_geometry_valid"]
            .to(device=current_base_features.device, dtype=current_base_features.dtype)
            .reshape(batch_size),
            "ct_b1_geometry_source_id": input_dict["search_v3_prior_source_id"]
            .to(device=current_base_features.device, dtype=current_base_features.dtype)
            .reshape(batch_size),
            "ct_memory_mode_id": current_base_features.new_full(
                (batch_size,),
                {"real": 0, "empty": 1, "time_misaligned": 2, "none": 3}[memory_mode],
            ),
            "ct_base_evidence_mode_id": current_base_features.new_full(
                (batch_size,), {"full": 0, "empty": 1}[base_evidence_mode]
            ),
        }
    )
    output_dict.update(joint_output)
    return final_box
