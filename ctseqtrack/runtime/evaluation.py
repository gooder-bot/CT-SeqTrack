"""Paper-facing v25 evaluator diagnostics.

Ground truth is consumed only here, outside the model forward path.
"""

import numpy as np
from nuscenes.utils import geometry_utils

from datasets import points_utils
from utils.metrics import estimateAccuracy, estimateOverlap


def build_ct_joint_diagnostic_row(
    self, output, data_dict, this_box, reference_box, frame_id
):
    """Export paper-facing joint-Full diagnostics; GT stays outside forward."""
    target_box = points_utils.transform_box(this_box, reference_box)
    target_xy = np.asarray(target_box.center[:2], dtype=np.float64)

    def xy(key, fallback):
        value = output.get(key)
        if value is None:
            return np.asarray(fallback, dtype=np.float64)
        return (
            value.detach()
            .cpu()
            .numpy()
            .reshape(-1, value.shape[-1])[0, :2]
            .astype(np.float64)
        )

    observation = xy("observation_aux_estimation_boxes", (0.0, 0.0))
    kinematic = xy("motion_prior_kinematic_xy", observation)
    learned = xy("motion_prior_xy", kinematic)
    raw_search = xy("ct_search_unmasked_raw_xy", observation)
    raw_obs = xy("ct_search_raw_obs_xy", raw_search)
    raw_motion = xy("ct_search_raw_motion_xy", raw_search)
    raw_alpha = xy("ct_search_raw_alpha_xy", raw_search)
    final = xy("aux_estimation_boxes", observation)
    observation_local4 = (
        output["observation_aux_estimation_boxes"]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 4)[0]
    )
    final_local4 = (
        output["aux_estimation_boxes"].detach().cpu().numpy().reshape(-1, 4)[0]
    )
    observation_world = points_utils.getOffsetBB(
        reference_box,
        observation_local4,
        degrees=self.config.degrees,
        use_z=self.config.use_z,
        limit_box=self.config.limit_box,
    )
    final_world = points_utils.getOffsetBB(
        reference_box,
        final_local4,
        degrees=self.config.degrees,
        use_z=self.config.use_z,
        limit_box=self.config.limit_box,
    )
    raw_search_local4 = observation_local4.copy()
    raw_search_local4[:2] = raw_search
    raw_search_world = points_utils.getOffsetBB(
        reference_box,
        raw_search_local4,
        degrees=self.config.degrees,
        use_z=self.config.use_z,
        limit_box=self.config.limit_box,
    )
    router_gate_applied = self._proposal_scalar(output, "ct_router_applied_gate")
    router_residual = xy("ct_router_bounded_residual_xy", (0.0, 0.0))
    bounded_local4 = observation_local4.copy()
    bounded_local4[:2] = observation + router_residual
    bounded_world = points_utils.getOffsetBB(
        reference_box,
        bounded_local4,
        degrees=self.config.degrees,
        use_z=self.config.use_z,
        limit_box=self.config.limit_box,
    )
    selective_local4 = observation_local4.copy()
    selective_local4[:2] = observation + router_gate_applied * router_residual
    selective_world = points_utils.getOffsetBB(
        reference_box,
        selective_local4,
        degrees=self.config.degrees,
        use_z=self.config.use_z,
        limit_box=self.config.limit_box,
    )

    def foreground_count(points_key, mask_key, source_key=None):
        points = data_dict.get(points_key)
        mask = data_dict.get(mask_key)
        if points is None or mask is None:
            return 0
        points_np = points.detach().cpu().numpy()[0, :, :3]
        mask_np = mask.detach().cpu().numpy().reshape(-1) > 0
        foreground = geometry_utils.points_in_box(
            target_box,
            points_np.T,
            float(getattr(self.config, "ct_b2_target_bb_scale", self.config.bb_scale)),
        )
        if source_key is not None:
            source = data_dict.get(source_key)
            if source is None:
                return 0
            source_np = source.detach().cpu().numpy().reshape(-1) > 0
            foreground = foreground & source_np
        return int(np.sum(foreground & mask_np))

    endpoint_foreground = foreground_count(
        "search_v3_points", "search_v3_point_valid_mask"
    )
    tube_foreground = foreground_count(
        "trajectory_search_points", "trajectory_search_point_valid_mask"
    )
    endpoint_extension_foreground = foreground_count(
        "search_v3_points", "search_v3_point_valid_mask", "search_v3_point_source"
    )
    tube_extension_foreground = foreground_count(
        "trajectory_search_points",
        "trajectory_search_point_valid_mask",
        "trajectory_search_point_source",
    )
    base_foreground = foreground_count(
        "ct_base_evidence_points", "ct_base_evidence_valid_mask"
    )
    sigma = output.get("motion_prior_log_sigma_parallel_perp")
    sigma_np = (
        np.exp(sigma.detach().cpu().numpy().reshape(-1, 2)[0])
        if sigma is not None
        else np.asarray((np.nan, np.nan))
    )
    residual_unit = output.get("motion_prior_residual_unit_parallel_perp")
    residual_unit_np = (
        residual_unit.detach().cpu().numpy().reshape(-1, 2)[0]
        if residual_unit is not None
        else np.zeros(2, dtype=np.float32)
    )
    b1_valid = bool(self._proposal_scalar(output, "motion_prior_valid") > 0.0)
    b1_nll = float("nan")
    b1_mahalanobis_sq = float("nan")
    direction = output.get("motion_prior_direction_xy")
    log_sigma = output.get("motion_prior_log_sigma_parallel_perp")
    if b1_valid and direction is not None and log_sigma is not None:
        direction_np = direction.detach().cpu().numpy().reshape(-1, 2)[0]
        direction_norm = float(np.linalg.norm(direction_np))
        log_sigma_np = log_sigma.detach().cpu().numpy().reshape(-1, 2)[0]
        if (
            np.isfinite(direction_norm)
            and direction_norm > 1e-8
            and np.isfinite(log_sigma_np).all()
        ):
            direction_np = direction_np / direction_norm
            perpendicular_np = np.asarray(
                (-direction_np[1], direction_np[0]), dtype=np.float64
            )
            learned_error_xy = target_xy - learned
            aligned_error = np.asarray(
                (
                    np.dot(learned_error_xy, direction_np),
                    np.dot(learned_error_xy, perpendicular_np),
                ),
                dtype=np.float64,
            )
            safe_log_sigma = np.clip(log_sigma_np, -4.0, 2.5)
            b1_nll = float(
                np.sum(
                    0.5
                    * (
                        aligned_error**2 * np.exp(-2.0 * safe_log_sigma)
                        + 2.0 * safe_log_sigma
                    )
                )
            )
            b1_mahalanobis_sq = float(
                np.sum(aligned_error**2 * np.exp(-2.0 * safe_log_sigma))
            )
            if not np.isfinite(b1_nll):
                b1_nll = float("nan")
            if not np.isfinite(b1_mahalanobis_sq):
                b1_mahalanobis_sq = float("nan")
    raw_obs_error = float(np.linalg.norm(raw_obs - target_xy))
    raw_motion_error = float(np.linalg.norm(raw_motion - target_xy))
    counterfactual_margin = float(
        getattr(self.config, "ct_query_counterfactual_margin", 0.05)
    )
    motion_helpful = bool(raw_motion_error + counterfactual_margin < raw_obs_error)
    motion_harmful = bool(raw_obs_error + counterfactual_margin < raw_motion_error)
    observation_error = float(np.linalg.norm(observation - target_xy))
    bounded_error = float(np.linalg.norm(bounded_local4[:2] - target_xy))
    observation_iou = float(
        estimateOverlap(
            this_box,
            observation_world,
            dim=self.config.IoU_space,
            up_axis=self.config.up_axis,
        )
    )
    bounded_iou = float(
        estimateOverlap(
            this_box,
            bounded_world,
            dim=self.config.IoU_space,
            up_axis=self.config.up_axis,
        )
    )
    return {
        "frame_id": int(frame_id),
        "b2_version": "ct_joint_full",
        "candidate_id": int(
            self._proposal_scalar(data_dict, "candidate_id", default=0.0)
        ),
        "candidate_gap_frames": int(
            self._proposal_scalar(data_dict, "candidate_gap_frames", default=1.0)
        ),
        "candidate_role": int(
            self._proposal_scalar(data_dict, "candidate_role", default=0.0)
        ),
        "candidate_available": int(
            self._proposal_scalar(data_dict, "candidate_available", default=1.0) > 0.0
        ),
        "boundary_ratio": self._proposal_scalar(
            data_dict, "candidate_boundary_ratio", default=0.0
        ),
        "role_satisfied": int(
            self._proposal_scalar(data_dict, "candidate_role_satisfied", default=0.0)
            > 0.0
        ),
        "base_target_count": self._proposal_scalar(
            data_dict, "ct_acquisition_base_target_count"
        ),
        "pool_target_count": self._proposal_scalar(
            data_dict, "ct_acquisition_extension_pool_target_count"
        ),
        "sampled_target_count": self._proposal_scalar(
            data_dict, "ct_acquisition_sampled_target_count"
        ),
        "extension_pool_count": self._proposal_scalar(
            data_dict, "ct_acquisition_extension_pool_count"
        ),
        "sampled_count": self._proposal_scalar(
            data_dict, "ct_acquisition_sampled_count"
        ),
        "target_in_support": int(
            self._proposal_scalar(
                data_dict, "ct_acquisition_extension_pool_target_count"
            )
            > 0.0
        ),
        "current_target_points": self._proposal_scalar(
            data_dict, "ct_acquisition_base_target_count"
        ),
        "evidence_raw_point_count": self._proposal_scalar(
            data_dict,
            "ct_evidence_raw_point_count",
            default=self._proposal_scalar(data_dict, "ct_search_baseline_points"),
        ),
        "evidence_base_unique_count": self._proposal_scalar(
            data_dict, "ct_evidence_base_unique_count"
        ),
        "evidence_extension_unique_count": self._proposal_scalar(
            data_dict, "ct_evidence_extension_unique_count"
        ),
        "evidence_foreground_count": int(
            base_foreground + endpoint_foreground + tube_foreground
        ),
        "recursive_age": self._proposal_scalar(data_dict, "ct_recursive_state_age"),
        "support_actual_length": self._proposal_scalar(
            data_dict, "search_v3_support_actual_extent", column=0
        ),
        "support_actual_width": self._proposal_scalar(
            data_dict, "search_v3_support_actual_extent", column=1
        ),
        "support_volume": (
            self._proposal_scalar(
                data_dict, "search_v3_support_actual_extent", column=0
            )
            * self._proposal_scalar(
                data_dict, "search_v3_support_actual_extent", column=1
            )
            * self._proposal_scalar(data_dict, "bbox_size", column=2)
        ),
        "available": int(self._proposal_scalar(output, "ct_b2_available") > 0.0),
        "structural_available": int(
            self._proposal_scalar(output, "ct_b2_available") > 0.0
        ),
        "recovery_positive": int(
            self._proposal_scalar(data_dict, "ct_recovery_positive") > 0.0
        ),
        "recovery_fallback": int(
            self._proposal_scalar(data_dict, "ct_recovery_fallback") > 0.0
        ),
        "query_delta_t": self._proposal_scalar(data_dict, "search_v3_query_delta_t"),
        "gap_ratio": self._proposal_scalar(
            data_dict, "search_v3_gap_ratio", default=1.0
        ),
        "endpoint_foreground_count": endpoint_foreground,
        "tube_foreground_count": tube_foreground,
        "foreground_count": endpoint_foreground + tube_foreground,
        "extension_foreground_count": (
            endpoint_extension_foreground + tube_extension_foreground
        ),
        "search_valid": int(
            self._proposal_scalar(output, "ct_search_candidate_valid") > 0.0
        ),
        "search_geometry_valid": int(
            self._proposal_scalar(data_dict, "ct_search_geometry_valid") > 0.0
        ),
        "search_new_support_valid": int(
            self._proposal_scalar(output, "ct_search_new_support_valid") > 0.0
        ),
        "search_available": int(
            self._proposal_scalar(output, "ct_search_available") > 0.0
        ),
        "search_geometry_source_id": int(
            self._proposal_scalar(data_dict, "search_v3_prior_source_id")
        ),
        "search_quality_valid": int(
            self._proposal_scalar(data_dict, "ct_search_quality_valid") > 0.0
        ),
        "search_coverage_need": int(
            self._proposal_scalar(data_dict, "ct_search_coverage_need") > 0.0
        ),
        "search_total_point_count": self._proposal_scalar(
            data_dict, "ct_search_total_point_count"
        ),
        "search_extension_count": self._proposal_scalar(
            data_dict, "ct_search_extension_count"
        ),
        "search_extension_voxels": self._proposal_scalar(
            data_dict, "ct_search_extension_voxels"
        ),
        "observation_error": observation_error,
        "observation_iou": observation_iou,
        "observation_distance": float(
            estimateAccuracy(
                this_box,
                observation_world,
                dim=self.config.IoU_space,
                up_axis=self.config.up_axis,
            )
        ),
        "kinematic_error": float(np.linalg.norm(kinematic - target_xy)),
        "learned_motion_error": float(np.linalg.norm(learned - target_xy)),
        "b1_valid": int(b1_valid and np.isfinite(b1_nll)),
        "b1_nll": b1_nll,
        "b1_mahalanobis_sq": b1_mahalanobis_sq,
        "b1_coverage_50": int(
            np.isfinite(b1_mahalanobis_sq) and b1_mahalanobis_sq <= 1.38629436112
        ),
        "b1_coverage_80": int(
            np.isfinite(b1_mahalanobis_sq) and b1_mahalanobis_sq <= 3.21887582487
        ),
        "b1_coverage_95": int(
            np.isfinite(b1_mahalanobis_sq) and b1_mahalanobis_sq <= 5.99146454711
        ),
        "raw_search_error": float(np.linalg.norm(raw_search - target_xy)),
        "raw_search_iou": float(
            estimateOverlap(
                this_box,
                raw_search_world,
                dim=self.config.IoU_space,
                up_axis=self.config.up_axis,
            )
        ),
        "raw_search_distance": float(
            estimateAccuracy(
                this_box,
                raw_search_world,
                dim=self.config.IoU_space,
                up_axis=self.config.up_axis,
            )
        ),
        "selective_error": float(np.linalg.norm(selective_local4[:2] - target_xy)),
        "selective_iou": float(
            estimateOverlap(
                this_box,
                selective_world,
                dim=self.config.IoU_space,
                up_axis=self.config.up_axis,
            )
        ),
        "selective_distance": float(
            estimateAccuracy(
                this_box,
                selective_world,
                dim=self.config.IoU_space,
                up_axis=self.config.up_axis,
            )
        ),
        "raw_obs_error": raw_obs_error,
        "raw_motion_error": raw_motion_error,
        "raw_alpha_error": float(np.linalg.norm(raw_alpha - target_xy)),
        "alpha_counterfactual_uplift": (raw_obs_error - raw_motion_error),
        "alpha_motion_helpful": int(motion_helpful),
        "alpha_motion_harmful": int(motion_harmful),
        "alpha_ambiguous": int(not (motion_helpful or motion_harmful)),
        "final_error": float(np.linalg.norm(final - target_xy)),
        "final_iou": float(
            estimateOverlap(
                this_box,
                final_world,
                dim=self.config.IoU_space,
                up_axis=self.config.up_axis,
            )
        ),
        "final_distance": float(
            estimateAccuracy(
                this_box,
                final_world,
                dim=self.config.IoU_space,
                up_axis=self.config.up_axis,
            )
        ),
        "query_gate": self._proposal_scalar(output, "ct_query_gate_probability"),
        "query_gate_applied": self._proposal_scalar(output, "ct_query_gate_internal"),
        "query_shift_norm": self._proposal_scalar(output, "ct_query_shift_norm"),
        "router_gate": self._proposal_scalar(output, "ct_router_gate"),
        "action_score": self._proposal_scalar(output, "ct_b3_action_score"),
        "helpful_probability": self._proposal_scalar(output, "ct_b3_help_probability"),
        "harmful_probability": self._proposal_scalar(output, "ct_b3_harm_probability"),
        "expected_center_gain": self._proposal_scalar(
            output, "ct_b3_expected_center_gain"
        ),
        "expected_iou_gain": self._proposal_scalar(output, "ct_b3_expected_iou_gain"),
        "center_gain": observation_error - bounded_error,
        "iou_gain": bounded_iou - observation_iou,
        "bounded_action_error": bounded_error,
        "bounded_action_iou": bounded_iou,
        "b3_calibrated": self._proposal_scalar(output, "ct_b3_calibrated"),
        "router_applied_gate": router_gate_applied,
        "router_evidence_valid": self._proposal_scalar(
            output, "ct_router_evidence_valid"
        ),
        "router_radius": self._proposal_scalar(output, "ct_router_radius"),
        "residual_unit_parallel": float(residual_unit_np[0]),
        "residual_unit_perpendicular": float(residual_unit_np[1]),
        "residual_saturation": self._proposal_scalar(
            output, "ct_motion_residual_saturation"
        ),
        "sigma_parallel": float(sigma_np[0]),
        "sigma_perpendicular": float(sigma_np[1]),
        "targetness_mean": self._proposal_scalar(output, "ct_search_targetness_mean"),
        "targetness_max": self._proposal_scalar(output, "ct_search_targetness_max"),
        "targetness_entropy": self._proposal_scalar(
            output, "ct_search_targetness_entropy"
        ),
        "normalized_ess": self._proposal_scalar(output, "ct_search_normalized_ess"),
        "extension_mass_ratio": self._proposal_scalar(
            output, "ct_search_extension_mass_ratio"
        ),
        "extension_vote_rms": self._proposal_scalar(
            output, "ct_search_extension_vote_rms"
        ),
        "presence_probability": self._proposal_scalar(
            output, "ct_search_presence_probability"
        ),
        "presence_score": self._proposal_scalar(
            output, "ct_search_presence_probability"
        ),
        "presence_target": int(
            endpoint_extension_foreground + tube_extension_foreground >= 1
        ),
    }
