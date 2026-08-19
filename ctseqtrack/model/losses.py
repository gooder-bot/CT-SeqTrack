"""Current contract-v3 B1/B2/B3 loss transaction."""

import torch
import torch.nn.functional as F

from ctseqtrack.model.evidence import extension_target_bearing_mask
from ctseqtrack.runtime.optimization import candidate_stratified_mean


def compute_contract_v3_loss(self, data, output, target_xy):
    """B2 acquisition and detached B3 action-risk objectives."""
    dtype = output["ct_search_targetness_logits"].dtype
    device = output["ct_search_targetness_logits"].device
    extension_labels = data["ct_extension_labels"].to(device=device, dtype=dtype)
    extension_valid = data["ct_extension_valid_mask"].to(device=device, dtype=dtype)
    base_labels = data["ct_base_evidence_labels"].to(device=device, dtype=dtype)
    base_valid = data["ct_base_evidence_valid_mask"].to(device=device, dtype=dtype)
    candidate_id = data.get("candidate_id")
    if candidate_id is None:
        candidate_id = torch.zeros(target_xy.shape[0], device=device, dtype=torch.long)
    else:
        candidate_id = candidate_id.to(device=device).reshape(-1)
    canonical_row = (candidate_id == 0).to(dtype)
    candidate_available = data.get("candidate_available")
    if candidate_available is None:
        candidate_available = torch.ones_like(canonical_row)
    else:
        candidate_available = candidate_available.to(
            device=device, dtype=dtype
        ).reshape(-1)
    candidate_boundary_ratio = data.get("candidate_boundary_ratio")
    if candidate_boundary_ratio is None:
        candidate_boundary_ratio = target_xy.new_zeros(target_xy.shape[0])
    else:
        candidate_boundary_ratio = candidate_boundary_ratio.to(
            device=device, dtype=dtype
        ).reshape(-1)
    candidate_role_satisfied = data.get("candidate_role_satisfied")
    if candidate_role_satisfied is None:
        candidate_role_satisfied = canonical_row
    else:
        candidate_role_satisfied = candidate_role_satisfied.to(
            device=device, dtype=dtype
        ).reshape(-1)

    def weighted_mean(values, valid=None):
        """Candidate-stratified numerator/denominator aggregation."""
        availability_mask = candidate_available
        if valid is None:
            valid = availability_mask
        else:
            valid = valid * availability_mask
        return candidate_stratified_mean(values, valid, candidate_id)

    targetness_error = F.binary_cross_entropy_with_logits(
        output["ct_search_targetness_logits"], extension_labels, reduction="none"
    )
    positive_weight = float(getattr(self.config, "ct_targetness_positive_weight", 1.0))
    negative_weight = float(getattr(self.config, "ct_targetness_negative_weight", 1.0))
    targetness_error = targetness_error * (
        extension_labels * positive_weight + (1.0 - extension_labels) * negative_weight
    )
    loss_targetness = weighted_mean(targetness_error, extension_valid)

    foreground = extension_valid * extension_labels * candidate_available.unsqueeze(1)
    vote_target = target_xy.unsqueeze(1).expand_as(output["ct_search_point_votes"])
    vote_error = F.smooth_l1_loss(
        output["ct_search_point_votes"], vote_target, reduction="none"
    ).mean(dim=2)
    loss_vote = weighted_mean(vote_error, foreground)

    availability = output["ct_b2_available"].detach()
    observation_xy = output["observation_aux_estimation_boxes"][:, :2].detach()
    raw_xy = output["ct_b2_raw_box"][:, :2]
    raw_error_per_sample = F.smooth_l1_loss(raw_xy, target_xy, reduction="none").mean(
        dim=1
    )
    base_presence_target = ((base_labels * base_valid).sum(dim=1) > 0).to(dtype)
    extension_presence_target = (
        (extension_labels * extension_valid).sum(dim=1) > 0
    ).to(dtype)
    target_bearing = extension_target_bearing_mask(
        availability, extension_labels, extension_valid
    )
    target_bearing = target_bearing * candidate_available
    # A raw candidate is identifiable as extension evidence only on rows
    # where the extension actually contains target points.  Absence rows
    # still train presence below, but never receive a GT center gradient.
    loss_raw = weighted_mean(raw_error_per_sample, target_bearing)
    base_presence_error = F.binary_cross_entropy_with_logits(
        output["ct_b2_base_presence_logit"], base_presence_target, reduction="none"
    )
    extension_presence_error = F.binary_cross_entropy_with_logits(
        output["ct_b2_extension_presence_logit"],
        extension_presence_target,
        reduction="none",
    )
    presence_valid = candidate_available
    presence_scope = (
        str(getattr(self.config, "ct_presence_training_scope", "all_candidates"))
        .strip()
        .lower()
    )
    if presence_scope == "candidate0":
        presence_valid = presence_valid * canonical_row
    elif presence_scope != "all_candidates":
        raise ValueError(
            "ct_presence_training_scope must be all_candidates or candidate0"
        )
    loss_base_presence = weighted_mean(base_presence_error, presence_valid)
    loss_extension_presence = weighted_mean(extension_presence_error, presence_valid)
    loss_presence = 0.5 * (loss_base_presence + loss_extension_presence)

    observation_error = torch.linalg.norm(observation_xy - target_xy, dim=1)
    bounded_xy = observation_xy + output["ct_router_bounded_residual_xy"].detach()
    bounded_distance_error = torch.linalg.norm(bounded_xy - target_xy, dim=1)
    center_gain_h1 = observation_error - bounded_distance_error

    # Same-size axis-aligned BEV IoU is a stable differentiable proxy for
    # the expected-IoU head.  Exact oriented IoU remains an evaluation and
    # calibration metric.
    size_xy = data["bbox_size"][:, :2].to(device=device, dtype=dtype).clamp_min(1e-3)

    def same_size_iou(center):
        overlap = torch.clamp(size_xy - torch.abs(center - target_xy), min=0.0)
        intersection = overlap[:, 0] * overlap[:, 1]
        area = size_xy[:, 0] * size_xy[:, 1]
        return intersection / torch.clamp(2.0 * area - intersection, min=1e-6)

    iou_gain_h1 = same_size_iou(bounded_xy) - same_size_iou(observation_xy)
    h3_center_gain = center_gain_h1.new_zeros(center_gain_h1.shape)
    h3_iou_gain = iou_gain_h1.new_zeros(iou_gain_h1.shape)
    h3_valid = availability.new_zeros(availability.shape)
    if "ct_h3_center_gain" in data:
        h3_center_gain = (
            data["ct_h3_center_gain"]
            .to(device=device, dtype=dtype)
            .reshape(-1)
            .detach()
        )
    elif "ct_h3_gain" in data:
        h3_center_gain = (
            data["ct_h3_gain"].to(device=device, dtype=dtype).reshape(-1).detach()
        )
    if "ct_h3_iou_gain" in data:
        h3_iou_gain = (
            data["ct_h3_iou_gain"].to(device=device, dtype=dtype).reshape(-1).detach()
        )
    if "ct_h3_valid" in data:
        h3_valid = (
            data["ct_h3_valid"].to(device=device, dtype=dtype).reshape(-1).detach()
        )

    combined_center_gain = torch.where(
        h3_valid > 0, 0.5 * (center_gain_h1 + h3_center_gain), center_gain_h1
    )
    combined_iou_gain = torch.where(
        h3_valid > 0, 0.5 * (iou_gain_h1 + h3_iou_gain), iou_gain_h1
    )
    helpful = (
        (center_gain_h1 > self.ct_router_help_margin)
        & (iou_gain_h1 >= 0.0)
        & (
            (h3_valid <= 0)
            | ((h3_center_gain > self.ct_router_h3_margin) & (h3_iou_gain >= 0.0))
        )
    )
    harmful = (
        (center_gain_h1 < -self.ct_router_help_margin)
        | (iou_gain_h1 < 0.0)
        | (
            (h3_valid > 0)
            & ((h3_center_gain < -self.ct_router_h3_margin) | (h3_iou_gain < 0.0))
        )
        | ~extension_presence_target.to(torch.bool)
    )
    # B3 risk labels are action labels on the canonical state
    # distribution.  Auxiliary acquisition views cannot train B3.
    b3_valid = availability * canonical_row
    helpful_error = F.binary_cross_entropy_with_logits(
        output["ct_b3_help_logit"], helpful.to(dtype), reduction="none"
    )
    harmful_error = F.binary_cross_entropy_with_logits(
        output["ct_b3_harm_logit"], harmful.to(dtype), reduction="none"
    )
    loss_helpful = weighted_mean(helpful_error, b3_valid)
    loss_harmful = weighted_mean(harmful_error, b3_valid)
    loss_center_gain = weighted_mean(
        F.smooth_l1_loss(
            output["ct_b3_expected_center_gain"],
            combined_center_gain.detach(),
            reduction="none",
        ),
        b3_valid,
    )
    loss_iou_gain = weighted_mean(
        F.smooth_l1_loss(
            output["ct_b3_expected_iou_gain"],
            combined_iou_gain.detach(),
            reduction="none",
        ),
        b3_valid,
    )
    loss_b3 = target_xy.new_zeros(())
    if self.ct_enable_b3:
        loss_b3 = loss_helpful + loss_harmful + loss_center_gain + loss_iou_gain

    def acquisition_metric(key):
        value = data.get(key)
        if value is None:
            return target_xy.new_zeros((target_xy.shape[0],))
        return value.to(device=device, dtype=dtype).reshape(-1)

    base_target_count = (
        acquisition_metric("ct_acquisition_base_target_count") * candidate_available
    )
    expansion_target_count = (
        acquisition_metric("ct_acquisition_expansion_target_count")
        * candidate_available
    )
    pool_target_count = (
        acquisition_metric("ct_acquisition_extension_pool_target_count")
        * candidate_available
    )
    sampled_target_count = (
        acquisition_metric("ct_acquisition_sampled_target_count") * candidate_available
    )
    recovery_positive = acquisition_metric("ct_recovery_positive") * candidate_available
    recovery_fallback = acquisition_metric("ct_recovery_fallback") * candidate_available
    support_truncated = (
        acquisition_metric("search_v3_support_truncated") * candidate_available
    )
    support_extent = data.get("search_v3_support_actual_extent")
    if support_extent is None:
        support_volume = target_xy.new_zeros((target_xy.shape[0],))
    else:
        support_extent = support_extent.to(device=device, dtype=dtype)
        support_volume = (
            support_extent[:, 0]
            * support_extent[:, 1]
            * data["bbox_size"][:, 2].to(device=device, dtype=dtype)
        ) * candidate_available
    eligible_rows = pool_target_count > 0
    retained_rows = eligible_rows & (sampled_target_count > 0)
    eligible_row_count = eligible_rows.to(dtype).sum()
    retained_row_count = retained_rows.to(dtype).sum()
    acquisition_row_recall = retained_row_count / torch.clamp(
        eligible_row_count, min=1.0
    )
    pool_target_sum = pool_target_count.sum()
    sampled_target_sum = sampled_target_count.sum()
    acquisition_point_recall = sampled_target_sum / torch.clamp(
        pool_target_sum, min=1.0
    )

    b2_total = (
        self.ct_targetness_weight * loss_targetness
        + self.ct_vote_weight * loss_vote
        + self.ct_raw_search_weight * loss_raw
        + self.ct_presence_weight * loss_presence
    )
    b3_total = self.ct_router_weight * loss_b3
    plugin_total = b2_total + b3_total
    return {
        "loss_ct_plugin_total": plugin_total,
        "loss_ct_b2_total": b2_total,
        "loss_ct_b3_total": b3_total,
        "loss_ct_targetness": loss_targetness,
        "loss_ct_vote": loss_vote,
        "loss_ct_raw_search": loss_raw,
        "loss_ct_presence": loss_presence,
        "loss_ct_base_presence": loss_base_presence,
        "loss_ct_extension_presence": loss_extension_presence,
        "loss_ct_utility": loss_b3,
        "loss_ct_utility_classification": (loss_helpful + loss_harmful),
        "loss_ct_expected_gain": (loss_center_gain + loss_iou_gain),
        "loss_ct_helpful": loss_helpful,
        "loss_ct_harmful": loss_harmful,
        "loss_ct_expected_center_gain": loss_center_gain,
        "loss_ct_expected_iou_gain": loss_iou_gain,
        "loss_ct_h3": loss_b3,
        "ct_h1_signed_gain": weighted_mean(center_gain_h1, availability),
        "ct_h1_helpful_rate": weighted_mean(helpful.to(dtype), b3_valid),
        "ct_h1_harmful_rate": weighted_mean(harmful.to(dtype), b3_valid),
        "ct_b2_target_bearing_rate": target_bearing.mean(),
        "ct_no_extension_counterfactual_gain": target_xy.new_zeros(()),
        "ct_b2_availability_rate": availability.float().mean(),
        "ct_candidate_available_rate": candidate_available.mean(),
        "ct_candidate_boundary_ratio_mean": (
            candidate_boundary_ratio * candidate_available
        ).sum()
        / candidate_available.sum().clamp_min(1.0),
        "ct_candidate_role_satisfied_rate": (
            candidate_role_satisfied * candidate_available
        ).sum()
        / candidate_available.sum().clamp_min(1.0),
        "ct_candidate_available_row_count": candidate_available.sum(),
        "ct_candidate_role_satisfied_row_count": (
            candidate_role_satisfied * candidate_available
        ).sum(),
        "ct_candidate_boundary_ratio_sum": (
            candidate_boundary_ratio * candidate_available
        ).sum(),
        "ct_candidate_boundary_ratio_count": candidate_available.sum(),
        "ct_support_truncated_row_count": support_truncated.sum(),
        "ct_support_volume_sum": support_volume.sum(),
        "ct_support_volume_count": candidate_available.sum(),
        "ct_b2_base_presence_target_rate": base_presence_target.mean(),
        "ct_b2_extension_presence_target_rate": extension_presence_target.mean(),
        "ct_acquisition_base_presence_rate": (base_target_count > 0).to(dtype).mean(),
        "ct_acquisition_expansion_coverage_rate": (expansion_target_count > 0)
        .to(dtype)
        .mean(),
        "ct_acquisition_pool_presence_rate": (pool_target_count > 0).to(dtype).mean(),
        "ct_acquisition_sampled_presence_rate": (sampled_target_count > 0)
        .to(dtype)
        .mean(),
        "ct_acquisition_eligible_row_count": eligible_row_count,
        "ct_acquisition_retained_row_count": retained_row_count,
        "ct_acquisition_pool_target_sum": pool_target_sum,
        "ct_acquisition_sampled_target_sum": sampled_target_sum,
        "ct_acquisition_row_recall": acquisition_row_recall,
        "ct_acquisition_point_recall": acquisition_point_recall,
        "ct_recovery_positive_row_count": recovery_positive.sum(),
        "ct_recovery_fallback_row_count": recovery_fallback.sum(),
        # Compatibility alias.  From contract-v3 onward "recall" means
        # retained eligible rows, never a macro point ratio.
        "ct_acquisition_target_recall": acquisition_row_recall,
    }
