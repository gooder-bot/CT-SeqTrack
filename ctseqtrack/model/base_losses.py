"""Formal v25 B0/B1 losses and transaction ownership."""

import torch
import torch.nn.functional as F

from ctseqtrack.model.losses import compute_contract_v3_loss
from ctseqtrack.model.prior import physical_motion_uncertainty_loss


def masked_mean(per_sample, valid):
    valid = valid.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    per_sample = per_sample.reshape(-1)
    # Invalid recursive rows may legitimately carry undefined diagnostics
    # (for example an oracle error of +inf when no expert is available).
    # Multiplying those values by zero is not a mask: IEEE inf*0 and nan*0
    # remain NaN and would poison the first optimizer step.  Select the safe
    # branch before multiplying instead.
    active = valid > 0
    safe_per_sample = torch.where(active, per_sample, torch.zeros_like(per_sample))
    safe_valid = torch.where(active, valid, torch.zeros_like(valid))
    return (safe_per_sample * safe_valid).sum() / torch.clamp(
        safe_valid.sum(), min=1.0
    )


def _weighted_masked_mean(per_sample, valid, weight=None):
    valid = valid.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    per_sample = per_sample.reshape(-1)
    if weight is None:
        weight = torch.ones_like(valid)
    else:
        weight = weight.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    active = valid > 0
    safe_per_sample = torch.where(active, per_sample, torch.zeros_like(per_sample))
    safe_weight = torch.where(active, weight, torch.zeros_like(weight))
    effective = valid * safe_weight
    return (safe_per_sample * effective).sum() / torch.clamp(
        effective.sum(), min=1.0
    )


def _aligned_absolute_error(mean_xy, target_xy, direction_xy):
    direction = direction_xy.detach()
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    error = target_xy - mean_xy
    return torch.stack(
        ((error * direction).sum(dim=1), (error * perpendicular).sum(dim=1)),
        dim=1,
    ).abs()


def _pinball_loss(quantiles, target):
    levels = quantiles.new_tensor((0.50, 0.80, 0.95)).view(1, 3, 1)
    residual = target.unsqueeze(1) - quantiles
    return torch.maximum(levels * residual, (levels - 1.0) * residual).mean(dim=(1, 2))


def _hard_motion_weights(model, data, prefix, valid):
    key = f"{prefix}_gt_cv_difficulty"
    if key not in data:
        return torch.ones_like(valid)
    difficulty = data[key].to(device=valid.device, dtype=valid.dtype).reshape(-1)
    q50 = float(model.motion_v3_hard_q50)
    q90 = float(model.motion_v3_hard_q90)
    if q90 <= q50:
        raise ValueError("GT-only hard-motion q90 must be greater than q50")
    raw = 1.0 + 2.0 * torch.clamp((difficulty - q50) / (q90 - q50), 0.0, 1.0)
    normalizer = masked_mean(raw, valid).detach().clamp(min=1e-6)
    return raw / normalizer


def _balanced_binary_loss(probability, target, valid):
    probability = probability.clamp(min=1e-6, max=1.0 - 1e-6)
    positive = valid * target
    negative = valid * (1.0 - target)
    positive_count = positive.sum()
    negative_count = negative.sum()
    positive_weight = torch.where(
        positive_count > 0,
        0.5 * valid.sum() / positive_count.clamp(min=1.0),
        valid.new_zeros(()),
    )
    negative_weight = torch.where(
        negative_count > 0,
        0.5 * valid.sum() / negative_count.clamp(min=1.0),
        valid.new_zeros(()),
    )
    class_weight = target * positive_weight + (1.0 - target) * negative_weight
    return _weighted_masked_mean(
        F.binary_cross_entropy(probability, target, reduction="none"),
        valid,
        class_weight,
    )


def _ra_pmm_view_loss(model, data, output, prefix, output_prefix, outer_weight):
    """RA-PMM objective for either the main or a gap auxiliary view."""
    mean = output[f"{output_prefix}_prior_xy"]
    target = (
        data[f"{prefix}_physical_target_xy"]
        .to(device=mean.device, dtype=mean.dtype)
        .detach()
    )
    if target.requires_grad:
        raise RuntimeError("B1 physical target must be detached")
    endpoint_target = (
        data[f"{prefix}_endpoint_target_xy"]
        .to(device=mean.device, dtype=mean.dtype)
        .detach()
    )
    anchor_drift = (
        data[f"{prefix}_anchor_drift_xy"]
        .to(device=mean.device, dtype=mean.dtype)
        .detach()
    )
    valid = output[f"{output_prefix}_prior_valid"].to(mean.dtype).reshape(-1)
    if prefix == "motion_main" and "candidate_available" in data:
        valid = valid * data["candidate_available"].to(
            device=valid.device, dtype=valid.dtype
        ).reshape(-1)
    hard_weight = _hard_motion_weights(model, data, prefix, valid)

    mean_per_sample = F.smooth_l1_loss(mean, target, reduction="none").mean(dim=1)
    mean_loss = _weighted_masked_mean(mean_per_sample, valid, hard_weight)

    mode_centers = output[f"{output_prefix}_prior_mode_centers_xy"]
    mode_probability = output[f"{output_prefix}_prior_mode_probabilities"]
    expert_valid = output[f"{output_prefix}_prior_expert_valid_mask"].bool()
    expert_error = torch.linalg.norm(target.unsqueeze(1) - mode_centers, dim=2)
    expert_error = expert_error.masked_fill(~expert_valid, float("inf"))
    sorted_error, sorted_index = torch.sort(expert_error, dim=1)
    # Mode supervision is defined only when at least two finite experts are
    # available.  In particular, early recursive rows can have zero experts;
    # evaluating inf/inf on those rows creates a NaN even though their B1
    # validity weight is zero.
    has_two_finite_experts = torch.isfinite(sorted_error[:, 0]) & torch.isfinite(
        sorted_error[:, 1]
    )
    safe_best = torch.where(
        has_two_finite_experts, sorted_error[:, 0], torch.zeros_like(sorted_error[:, 0])
    )
    safe_second = torch.where(
        has_two_finite_experts, sorted_error[:, 1], torch.ones_like(sorted_error[:, 1])
    )
    ratio = safe_best / (safe_second + 1e-6)
    ratio = torch.where(safe_second <= 1e-6, torch.ones_like(ratio), ratio)
    mode_confidence = torch.clamp((0.9 - ratio) / 0.1, 0.0, 1.0)
    distinguishable = has_two_finite_experts.to(mean.dtype)
    mode_confidence = mode_confidence * distinguishable
    log_probability = torch.log(mode_probability.clamp(min=1e-8))
    top_one = sorted_index[:, 0]
    hard_ce = -torch.gather(log_probability, 1, top_one.unsqueeze(1)).squeeze(1)
    valid_expert_count = expert_valid.to(mean.dtype).sum(dim=1).clamp(min=1.0)
    smooth_ce = (
        -(log_probability * expert_valid.to(mean.dtype)).sum(dim=1) / valid_expert_count
    )
    mode_per_sample = 0.95 * hard_ce + 0.05 * smooth_ce
    mode_loss = _weighted_masked_mean(
        mode_per_sample, valid, hard_weight * mode_confidence
    )

    direction = output[f"{output_prefix}_prior_direction_xy"].detach()
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    physics_residual = target - output[f"{output_prefix}_prior_kinematic_xy"].detach()
    residual_pp = torch.stack(
        (
            (physics_residual * direction).sum(dim=1),
            (physics_residual * perpendicular).sum(dim=1),
        ),
        dim=1,
    )
    dt = (
        data.get(f"{prefix}_physical_delta_t", data[f"{prefix}_current_delta_t"])
        .to(device=mean.device, dtype=mean.dtype)
        .reshape(-1)
    )
    dt_floor = max(float(model.motion_v3_dt_floor), 1e-3)
    target_acceleration = (
        2.0 * residual_pp / torch.clamp(dt, min=dt_floor).pow(2).unsqueeze(1)
    )
    max_acceleration = float(model.physical_motion_encoder.max_acceleration)
    target_acceleration = target_acceleration.clamp(
        min=-max_acceleration, max=max_acceleration
    ).detach()
    predicted_acceleration = output[
        f"{output_prefix}_prior_residual_acceleration_pp"
    ] * output[f"{output_prefix}_prior_residual_gate"].unsqueeze(1)
    acc_per_sample = F.smooth_l1_loss(
        predicted_acceleration, target_acceleration, reduction="none"
    ).mean(dim=1)
    acc_norm_loss = _weighted_masked_mean(acc_per_sample, valid, hard_weight)
    acc_reg_loss = _weighted_masked_mean(
        output[f"{output_prefix}_prior_residual_acceleration_pp"].pow(2).mean(dim=1),
        valid,
    )

    motion_quantiles = output[f"{output_prefix}_prior_motion_quantiles_pp"]
    motion_residual = _aligned_absolute_error(mean.detach(), target, direction)
    motion_quantile_loss = _weighted_masked_mean(
        _pinball_loss(motion_quantiles, motion_residual), valid, hard_weight
    )

    support_quantiles = output[f"{output_prefix}_prior_support_quantiles_pp"]
    support_residual = _aligned_absolute_error(
        mean.detach(), endpoint_target, direction
    )
    support_cap = data[f"{prefix}_support_cap_pp"].to(
        device=mean.device, dtype=mean.dtype
    )
    if support_cap.dim() == 1:
        support_cap = support_cap.unsqueeze(0).expand_as(support_residual)
    capped_residual = torch.minimum(support_residual, support_cap)
    recoverable = (support_residual <= support_cap).all(dim=1).to(mean.dtype)
    boundary_band = (
        (support_residual >= 0.8 * support_cap).any(dim=1) * (recoverable > 0)
    ).to(mean.dtype)
    support_weight = torch.where(
        recoverable > 0,
        1.0 + boundary_band,
        recoverable.new_full(recoverable.shape, 0.25),
    )
    support_quantile_loss = _weighted_masked_mean(
        _pinball_loss(support_quantiles, capped_residual), valid, support_weight
    )
    censored_axis = (support_residual > support_cap).to(mean.dtype)
    censor_per_sample = (
        F.relu(support_cap - support_quantiles[:, 2]) * censored_axis
    ).sum(dim=1) / censored_axis.sum(dim=1).clamp(min=1.0)
    censor_loss = _weighted_masked_mean(censor_per_sample, valid * (1.0 - recoverable))
    recoverability_probability = output[
        f"{output_prefix}_prior_recoverability_probability"
    ]
    recoverability_loss = _balanced_binary_loss(
        recoverability_probability, recoverable, valid
    )

    view_loss = (
        mean_loss
        + model.motion_v3_mode_weight * mode_loss
        + model.motion_v3_acc_norm_weight * acc_norm_loss
        + model.motion_v3_motion_quantile_weight * motion_quantile_loss
        + model.motion_v3_support_quantile_weight * support_quantile_loss
        + model.motion_v3_recoverability_weight * recoverability_loss
        + model.motion_v3_censor_weight * censor_loss
        + model.motion_v3_acc_reg_weight * acc_reg_loss
    )
    transaction = float(outer_weight) * view_loss

    physical_error = torch.linalg.norm(mean.detach() - target, dim=1)
    endpoint_error = torch.linalg.norm(mean.detach() - endpoint_target, dim=1)
    anchor_drift_error = torch.linalg.norm(anchor_drift, dim=1)
    mode_entropy = -(
        mode_probability * torch.log(mode_probability.clamp(min=1e-8))
    ).sum(dim=1)
    predicted_mode_error = torch.linalg.norm(
        (mode_probability.unsqueeze(2) * mode_centers).sum(dim=1).detach() - target,
        dim=1,
    )
    finite_oracle_error = torch.where(
        torch.isfinite(sorted_error[:, 0]),
        sorted_error[:, 0],
        torch.zeros_like(sorted_error[:, 0]),
    ).detach()
    oracle_regret = predicted_mode_error - finite_oracle_error
    selected_mode = torch.argmax(mode_probability.detach(), dim=1)
    joint_motion_coverage = (
        (motion_residual <= motion_quantiles[:, 2].detach()).all(dim=1).to(mean.dtype)
    )
    joint_support_coverage = (
        (support_residual <= support_quantiles[:, 2].detach()).all(dim=1).to(mean.dtype)
    )
    capped_support_coverage = (
        (capped_residual <= support_quantiles[:, 2].detach()).all(dim=1).to(mean.dtype)
    )
    predicted_unrecoverable = (recoverability_probability.detach() < 0.5).to(mean.dtype)

    stem = "motion_v3" if prefix == "motion_main" else "motion_v3_aux"
    losses = {
        f"loss_{stem}_prior": mean_loss,
        f"loss_{stem}_mode": mode_loss,
        f"loss_{stem}_acc_norm": acc_norm_loss,
        f"loss_{stem}_motion_quantile": motion_quantile_loss,
        f"loss_{stem}_support_quantile": support_quantile_loss,
        f"loss_{stem}_recoverability": recoverability_loss,
        f"loss_{stem}_censor": censor_loss,
        f"loss_{stem}_acc_reg": acc_reg_loss,
        f"{stem}_physical_rmse": torch.sqrt(masked_mean(physical_error.pow(2), valid)),
        f"{stem}_endpoint_rmse": torch.sqrt(masked_mean(endpoint_error.pow(2), valid)),
        f"{stem}_anchor_drift_rmse": torch.sqrt(
            masked_mean(anchor_drift_error.pow(2), valid)
        ),
        f"{stem}_mode_skip_rate": masked_mean(
            (mode_confidence <= 0).to(mean.dtype), valid
        ),
        f"{stem}_mode_entropy": masked_mean(mode_entropy, valid),
        f"{stem}_oracle_regret": masked_mean(oracle_regret, valid),
        f"{stem}_motion_joint_q95_coverage": masked_mean(joint_motion_coverage, valid),
        f"{stem}_support_conditional_q95_coverage": masked_mean(
            joint_support_coverage, valid * recoverable
        ),
        f"{stem}_support_capped_q95_coverage": masked_mean(
            capped_support_coverage, valid
        ),
        f"{stem}_boundary_q95_coverage": masked_mean(
            joint_support_coverage, valid * boundary_band
        ),
        f"{stem}_unrecoverable_recall": masked_mean(
            predicted_unrecoverable, valid * (1.0 - recoverable)
        ),
        f"{stem}_recoverable_rate": masked_mean(recoverable, valid),
        f"{stem}_dt_floor_rate": masked_mean((dt < dt_floor).to(mean.dtype), valid),
        f"{stem}_valid_rate": valid.float().mean(),
    }
    for expert_index, expert_name in enumerate(("cv", "ca", "ctrv")):
        losses[f"{stem}_expert_usage_{expert_name}"] = masked_mean(
            (selected_mode == expert_index).to(mean.dtype), valid
        )
    if prefix == "motion_aux" and "motion_aux_query_gap_frames" in data:
        query_gap = data["motion_aux_query_gap_frames"].to(valid.device).reshape(-1)
        for gap in model.motion_v3_aux_query_gaps:
            gap_valid = valid * (query_gap == gap).to(valid.dtype)
            losses[f"{stem}_acc_norm_gap{gap}"] = masked_mean(acc_per_sample, gap_valid)
            losses[f"{stem}_acc_count_gap{gap}"] = gap_valid.sum()
            losses[f"{stem}_acc_grad_proxy_gap{gap}"] = masked_mean(
                (predicted_acceleration.detach() - target_acceleration)
                .abs()
                .mean(dim=1),
                gap_valid,
            )
    return transaction, [transaction], losses


def _labels(data):
    with torch.no_grad():
        box_label = data["box_label"]
        motion_label = data["motion_label"]
        reference_label = data["box_label_prev"]
        return {
            "segmentation": data["seg_label"],
            "motion_state": data["motion_state_label"][:, 0],
            "center": box_label[:, :3],
            "angle": torch.sin(box_label[:, 3]),
            "motion_center": motion_label[:, 0, :3],
            "motion_angle": torch.sin(motion_label[:, 0, 3]),
            "reference_center": reference_label[:, :, :3],
            "reference_angle": torch.sin(reference_label[:, :, 3]),
        }


def _b0_loss(model, data, output, labels):
    loss_total = 0.0
    losses = {}
    motion_prediction = output["motion_pred"]
    segmentation_logits = output["seg_logits"]
    loss_segmentation = F.cross_entropy(
        segmentation_logits,
        labels["segmentation"],
        weight=segmentation_logits.new_tensor([0.5, 2.0]),
    )
    if model.use_motion_cls:
        loss_motion_classification = F.cross_entropy(
            output["motion_cls"], labels["motion_state"]
        )
        loss_total += loss_motion_classification * model.config.motion_cls_seg_weight
        losses["loss_motion_cls"] = loss_motion_classification
        motion_center = F.smooth_l1_loss(
            motion_prediction[:, :3], labels["motion_center"], reduction="none"
        )
        loss_motion_center = (
            labels["motion_state"] * motion_center.mean(dim=1)
        ).sum() / (labels["motion_state"].sum() + 1e-6)
        motion_angle = F.smooth_l1_loss(
            torch.sin(motion_prediction[:, 3]), labels["motion_angle"], reduction="none"
        )
        loss_motion_angle = (labels["motion_state"] * motion_angle).sum() / (
            labels["motion_state"].sum() + 1e-6
        )
    else:
        loss_motion_center = F.smooth_l1_loss(
            motion_prediction[:, :3], labels["motion_center"]
        )
        loss_motion_angle = F.smooth_l1_loss(
            torch.sin(motion_prediction[:, 3]), labels["motion_angle"]
        )

    coarse_boxes = output["estimation_boxes"]
    loss_center = F.smooth_l1_loss(coarse_boxes[:, :3], labels["center"])
    loss_angle = F.smooth_l1_loss(torch.sin(coarse_boxes[:, 3]), labels["angle"])
    loss_total += (
        loss_center * model.config.center_weight
        + loss_angle * model.config.angle_weight
    )
    losses["loss_center"] = loss_center
    losses["loss_angle"] = loss_angle

    observation_boxes = output.get(
        "observation_aux_estimation_boxes", output["aux_estimation_boxes"]
    )
    loss_center_aux = F.smooth_l1_loss(observation_boxes[:, :3], labels["center"])
    loss_angle_aux = F.smooth_l1_loss(
        torch.sin(observation_boxes[:, 3]), labels["angle"]
    )
    updated_reference_boxes = output["updated_ref_boxs"]
    loss_center_ref = F.smooth_l1_loss(
        updated_reference_boxes[:, :, :3], labels["reference_center"]
    )
    loss_angle_ref = F.smooth_l1_loss(
        torch.sin(updated_reference_boxes[:, :, 3]), labels["reference_angle"]
    )
    loss_total += (
        loss_segmentation * model.config.seg_weight
        + loss_center_aux * model.config.center_weight
        + loss_angle_aux * model.config.angle_weight
        + loss_motion_center * model.config.center_weight
        + loss_motion_angle * model.config.angle_weight
        + loss_center_ref * model.config.ref_center_weight
        + loss_angle_ref * model.config.ref_angle_weight
    )
    losses.update(
        {
            "loss_total": loss_total,
            "loss_seg": loss_segmentation,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_motion_center,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_motion_angle,
            "loss_center_ref": loss_center_ref,
            "loss_angle_ref": loss_angle_ref,
        }
    )
    return loss_total, losses


def _main_prior_loss(model, data, output):
    if model.motion_v3_ra_pmm:
        return _ra_pmm_view_loss(
            model,
            data,
            output,
            "motion_main",
            "motion",
            model.motion_v3_prior_weight,
        )
    target = data["motion_main_target_xy"].to(
        device=output["motion_prior_xy"].device, dtype=output["motion_prior_xy"].dtype
    )
    per_sample = F.smooth_l1_loss(
        output["motion_prior_xy"], target, reduction="none"
    ).mean(dim=1)
    valid = output["motion_prior_valid"]
    uncertainty = None
    if model.use_calibrated_motion_uncertainty or model.use_ct_joint_full:
        uncertainty = physical_motion_uncertainty_loss(
            output["motion_prior_xy"],
            target,
            output["motion_prior_log_sigma_parallel_perp"],
            output["motion_prior_direction_xy"],
            valid,
        )
        per_sample = uncertainty["mean_per_sample"]
        valid = uncertainty["valid"]
    if "candidate_available" in data:
        valid = valid * data["candidate_available"].to(
            device=valid.device, dtype=valid.dtype
        ).reshape(-1)
    prior_loss = masked_mean(per_sample, valid)
    prior_addition = model.motion_v3_prior_weight * prior_loss
    additions = [prior_addition]
    transaction = prior_addition
    losses = {"loss_motion_v3_prior": prior_loss}
    if uncertainty is not None:
        nll_loss = masked_mean(uncertainty["nll_per_sample"], valid)
        nll_addition = model.motion_v3_nll_weight * nll_loss
        additions.append(nll_addition)
        transaction = transaction + nll_addition
        losses["loss_motion_v3_nll"] = nll_loss

    prior_error = torch.linalg.norm(output["motion_prior_xy"].detach() - target, dim=1)
    kinematic_error = torch.linalg.norm(
        output["motion_prior_kinematic_xy"].detach() - target, dim=1
    )
    losses.update(
        {
            "motion_v3_prior_rmse": torch.sqrt(masked_mean(prior_error.pow(2), valid)),
            "motion_v3_kinematic_rmse": torch.sqrt(
                masked_mean(kinematic_error.pow(2), valid)
            ),
            "motion_v3_prior_valid_rate": valid.float().mean(),
            "motion_v3_history_valid_ratio": data["motion_main_valid_mask"]
            .float()
            .mean(),
        }
    )
    if uncertainty is not None:
        normalized_error = uncertainty["aligned_error"] * torch.exp(
            -output["motion_prior_log_sigma_parallel_perp"]
        )
        mahalanobis_squared = normalized_error.pow(2).sum(dim=1)
        coverage_errors = []
        for label, threshold, nominal in (
            ("50", 1.38629436112, 0.50),
            ("80", 3.21887582487, 0.80),
            ("95", 5.99146454711, 0.95),
        ):
            empirical = masked_mean(
                (mahalanobis_squared <= threshold).to(mahalanobis_squared.dtype), valid
            )
            losses[f"motion_v3_coverage_{label}"] = empirical
            coverage_errors.append(torch.abs(empirical - empirical.new_tensor(nominal)))
        losses.update(
            {
                "motion_v3_sigma_parallel_mean": masked_mean(
                    torch.exp(output["motion_prior_log_sigma_parallel_perp"][:, 0]),
                    valid,
                ),
                "motion_v3_sigma_perpendicular_mean": masked_mean(
                    torch.exp(output["motion_prior_log_sigma_parallel_perp"][:, 1]),
                    valid,
                ),
                "motion_v3_coverage_ece": torch.stack(coverage_errors).mean(),
            }
        )
    if "candidate_id" in data:
        candidate_id = data["candidate_id"].to(device=valid.device).reshape(-1)
        for bucket, mask in (
            ("candidate0", candidate_id == 0),
            ("candidate_nonzero", candidate_id != 0),
        ):
            bucket_valid = valid * mask.to(valid.dtype)
            losses.update(
                {
                    f"motion_v3_prior_rmse_{bucket}": torch.sqrt(
                        masked_mean(prior_error.pow(2), bucket_valid)
                    ),
                    f"motion_v3_kinematic_rmse_{bucket}": torch.sqrt(
                        masked_mean(kinematic_error.pow(2), bucket_valid)
                    ),
                    f"motion_v3_count_{bucket}": bucket_valid.sum(),
                }
            )
    return transaction, additions, losses


def _auxiliary_prior_loss(model, data, output):
    if not ("motion_aux_prior_xy" in output and "motion_aux_target_xy" in data):
        zero = output["motion_prior_xy"].new_zeros(())
        return zero, [], {}
    if model.motion_v3_ra_pmm:
        return _ra_pmm_view_loss(
            model,
            data,
            output,
            "motion_aux",
            "motion_aux",
            model.motion_v3_aux_prior_weight,
        )
    target = data["motion_aux_target_xy"].to(
        device=output["motion_aux_prior_xy"].device,
        dtype=output["motion_aux_prior_xy"].dtype,
    )
    per_sample = F.smooth_l1_loss(
        output["motion_aux_prior_xy"], target, reduction="none"
    ).mean(dim=1)
    valid = output["motion_aux_prior_valid"]
    uncertainty = None
    if model.use_calibrated_motion_uncertainty or model.use_ct_joint_full:
        uncertainty = physical_motion_uncertainty_loss(
            output["motion_aux_prior_xy"],
            target,
            output["motion_aux_prior_log_sigma_parallel_perp"],
            output["motion_aux_prior_direction_xy"],
            valid,
        )
        per_sample = uncertainty["mean_per_sample"]
        valid = uncertainty["valid"]
    prior_loss = masked_mean(per_sample, valid)
    additions = [model.motion_v3_aux_prior_weight * prior_loss]
    losses = {"loss_motion_v3_aux_prior": prior_loss}
    if uncertainty is not None:
        nll_loss = masked_mean(uncertainty["nll_per_sample"], valid)
        additions.append(model.motion_v3_aux_nll_weight * nll_loss)
        losses["loss_motion_v3_aux_nll"] = nll_loss
    prior_error = torch.linalg.norm(
        output["motion_aux_prior_xy"].detach() - target, dim=1
    )
    kinematic_error = torch.linalg.norm(
        output["motion_aux_prior_kinematic_xy"].detach() - target, dim=1
    )
    losses.update(
        {
            "motion_v3_aux_prior_rmse": torch.sqrt(
                masked_mean(prior_error.pow(2), valid)
            ),
            "motion_v3_aux_kinematic_rmse": torch.sqrt(
                masked_mean(kinematic_error.pow(2), valid)
            ),
            "motion_v3_aux_gap_ratio": masked_mean(
                output["motion_aux_prior_gap_ratio"], valid
            ),
            "motion_v3_aux_history_valid_ratio": data["motion_aux_valid_mask"]
            .float()
            .mean(),
        }
    )
    if "motion_aux_query_gap_frames" in data:
        query_gap = (
            data["motion_aux_query_gap_frames"].to(device=valid.device).reshape(-1)
        )
        for gap in model.motion_v3_aux_query_gaps:
            gap_valid = valid * (query_gap == gap).to(valid.dtype)
            losses.update(
                {
                    f"motion_v3_aux_prior_rmse_gap{gap}": torch.sqrt(
                        masked_mean(prior_error.pow(2), gap_valid)
                    ),
                    f"motion_v3_aux_kinematic_rmse_gap{gap}": torch.sqrt(
                        masked_mean(kinematic_error.pow(2), gap_valid)
                    ),
                    f"motion_v3_aux_count_gap{gap}": gap_valid.sum(),
                }
            )
    transaction = sum(additions[1:], additions[0])
    return transaction, additions, losses


def _box_aware_loss(model, data, output):
    if not model.box_aware:
        return output["motion_pred"].new_zeros(()), {}
    previous_bc = torch.flatten(data["prev_bc"], start_dim=1, end_dim=2)
    bc_label = torch.cat((previous_bc, data["this_bc"]), dim=1)
    loss_bc = F.smooth_l1_loss(output["pred_bc"], bc_label)
    return loss_bc * model.config.bc_weight, {"loss_bc": loss_bc}


def _append_observation_diagnostics(model, output, losses):
    if getattr(model.config, "obs_gate_log_stats", False):
        for log_key, output_key in {
            "obs_num_points_search_mean": "obs_num_points_search",
            "obs_soft_fg_count_mean": "obs_soft_fg_count",
            "obs_estimated_fg_points_mean": "obs_estimated_fg_points",
            "obs_mean_fg_score": "obs_mean_fg_score",
            "obs_valid_history_ratio": "obs_valid_history_ratio",
            "obs_current_delta_t_ratio": "obs_current_delta_t_ratio",
            "obs_current_delta_t_real_ratio": "obs_current_delta_t_real_ratio",
            "obs_current_delta_t_effective_ratio": "obs_current_delta_t_effective_ratio",
        }.items():
            if output_key in output:
                losses[log_key] = output[output_key].mean()
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
        if key in output:
            losses[f"{key}_mean"] = output[key].float().mean()


def compute_v25_loss(model, data, output):
    """Compute the unchanged v25 objective and disjoint transactions."""
    labels = _labels(data)
    loss_total, losses = _b0_loss(model, data, output, labels)
    b0_transaction = loss_total
    zero = loss_total.new_zeros(())
    b1_transaction = zero
    b2_transaction = zero
    b3_transaction = zero

    if model.use_b1motion_v3 and not (
        model.use_ct_joint_full and not model.ct_enable_b1
    ):
        b1_transaction, main_additions, b1_losses = _main_prior_loss(
            model, data, output
        )
        for addition in main_additions:
            loss_total = loss_total + addition
        losses.update(b1_losses)
        auxiliary_transaction, auxiliary_additions, auxiliary_losses = (
            _auxiliary_prior_loss(model, data, output)
        )
        b1_transaction = b1_transaction + auxiliary_transaction
        for addition in auxiliary_additions:
            loss_total = loss_total + addition
        losses.update(auxiliary_losses)
        losses["loss_total"] = loss_total

    if model.use_ct_joint_full and model.ct_enable_b2:
        target_xy = labels["center"][:, :2].to(
            device=output["ct_b2_raw_box"].device, dtype=output["ct_b2_raw_box"].dtype
        )
        plugin_losses = compute_contract_v3_loss(model, data, output, target_xy)
        loss_total = loss_total + plugin_losses["loss_ct_plugin_total"]
        b2_transaction = plugin_losses["loss_ct_b2_total"]
        b3_transaction = plugin_losses["loss_ct_b3_total"]
        losses.update(plugin_losses)
        losses["loss_total"] = loss_total

    box_addition, box_losses = _box_aware_loss(model, data, output)
    loss_total = loss_total + box_addition
    b0_transaction = b0_transaction + box_addition
    losses.update(box_losses)
    losses["loss_total"] = loss_total
    _append_observation_diagnostics(model, output, losses)
    losses.update(
        {
            "loss_b0_transaction": b0_transaction,
            "loss_b1_transaction": b1_transaction,
            "loss_b2_transaction": b2_transaction,
            "loss_b3_transaction": b3_transaction,
            "loss_plugin_transaction": (
                b1_transaction + b2_transaction + b3_transaction
            ),
        }
    )
    return losses


__all__ = ["compute_v25_loss", "masked_mean"]
