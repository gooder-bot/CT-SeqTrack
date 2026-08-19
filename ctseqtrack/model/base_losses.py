"""Formal v25 B0/B1 losses and transaction ownership."""

import torch
import torch.nn.functional as F

from ctseqtrack.model.losses import compute_contract_v3_loss
from ctseqtrack.model.prior import physical_motion_uncertainty_loss


def masked_mean(per_sample, valid):
    valid = valid.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
    return (per_sample.reshape(-1) * valid).sum() / torch.clamp(valid.sum(), min=1.0)


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
        return [], {}
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
    return additions, losses


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
        auxiliary_additions, auxiliary_losses = _auxiliary_prior_loss(
            model, data, output
        )
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
