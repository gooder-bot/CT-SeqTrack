"""Training diagnostics and epoch-boundary transactions for v25."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ctseqtrack.runtime.optimization import restore_global_rng_state


def parameter_group_sha256(model, group_name):
    digest = hashlib.sha256()
    for parameter_name, parameter in sorted(
        model._ct_named_parameters_by_module[group_name]
    ):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(parameter_name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def record_parameter_hash(model, event):
    timeline = getattr(model, "_ct_parameter_hash_timeline", None)
    if timeline is None:
        timeline = []
        model._ct_parameter_hash_timeline = timeline
    timeline.append(
        {
            "event": str(event),
            "epoch": int(getattr(model, "current_epoch", 0)),
            "b0_update_step": int(model.ct_b0_update_step.item()),
            "b0_sha256": parameter_group_sha256(model, "b0"),
        }
    )


def binary_curve_metrics_numpy(scores, targets):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores) & np.isfinite(targets)
    scores = np.clip(scores[finite], 0.0, 1.0)
    targets = targets[finite] > 0.5
    positive_count = int(targets.sum())
    negative_count = int((~targets).sum())
    auroc = 0.5
    if positive_count and negative_count:
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        ends = np.r_[starts[1:], len(scores)]
        for start, end in zip(starts, ends):
            ranks[start:end] = 0.5 * (start + 1 + end)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        positive_rank_sum = ranks[inverse][targets].sum()
        auroc = float(
            (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
            / (positive_count * negative_count)
        )
    auprc = 0.0
    if positive_count:
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_targets = targets[order].astype(np.float64)
        true_positive = np.cumsum(sorted_targets)
        threshold_ends = np.r_[
            np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),
            len(sorted_scores) - 1,
        ]
        true_positive = true_positive[threshold_ends]
        predicted_positive = threshold_ends.astype(np.float64) + 1.0
        precision = true_positive / predicted_positive
        recall = true_positive / positive_count
        auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    calibration = []
    for bin_index in range(5):
        lower = bin_index / 5.0
        upper = (bin_index + 1) / 5.0
        selected = (scores >= lower) & (
            scores < upper if bin_index < 4 else scores <= upper
        )
        calibration.append(
            {
                "count": int(selected.sum()),
                "confidence": float(scores[selected].mean()) if selected.any() else 0.0,
                "positive_rate": (
                    float(targets[selected].mean()) if selected.any() else 0.0
                ),
            }
        )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "positive_mean": float(scores[targets].mean()) if positive_count else 0.0,
        "negative_mean": float(scores[~targets].mean()) if negative_count else 0.0,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "calibration": calibration,
    }


def accumulate_joint_binary_rows(model, data, output):
    if not (model.training and model.use_ct_joint_full and model.ct_enable_b2):
        return
    if not hasattr(model, "_ct_epoch_binary_rows"):
        model._ct_epoch_binary_rows = {"presence": [], "alpha": [], "alpha_uplift": []}
    labels = data["ct_extension_labels"].detach()
    point_valid = data["ct_extension_valid_mask"].detach()
    support_valid = output["ct_b2_available"].detach().reshape(-1)
    presence_target = (labels * point_valid).sum(dim=1) >= 1
    presence_select = support_valid > 0
    if bool(presence_select.any()):
        model._ct_epoch_binary_rows["presence"].append(
            (
                output["ct_b2_extension_presence_probability"]
                .detach()[presence_select]
                .float()
                .cpu()
                .numpy(),
                presence_target[presence_select].float().cpu().numpy(),
            )
        )
    target_xy = data["box_label"][:, :2].to(
        device=output["ct_b2_raw_box"].device, dtype=output["ct_b2_raw_box"].dtype
    )
    observation_xy = output["observation_aux_estimation_boxes"][:, :2].detach()
    raw_xy = output["ct_b2_raw_box"][:, :2].detach()
    gain = torch.linalg.norm(observation_xy - target_xy, dim=1) - torch.linalg.norm(
        raw_xy - target_xy, dim=1
    )
    helpful = gain > model.ct_router_help_margin
    harmful = (gain < -model.ct_router_help_margin) | ~presence_target
    utility_valid = (support_valid > 0) & (helpful | harmful)
    if bool(utility_valid.any()):
        model._ct_epoch_binary_rows["alpha"].append(
            (
                torch.sigmoid(output["ct_b2_utility_logit"].detach())[utility_valid]
                .float()
                .cpu()
                .numpy(),
                helpful[utility_valid].float().cpu().numpy(),
            )
        )
        model._ct_epoch_binary_rows["alpha_uplift"].append(
            gain[utility_valid].float().cpu().numpy()
        )


def on_train_epoch_start(model):
    pending_rng = getattr(model, "_ct_pending_global_rng_state", None)
    if pending_rng is not None:
        restore_global_rng_state(pending_rng)
        model._ct_pending_global_rng_state = None
    model._ct_epoch_boundary_complete = False
    for module_name in ("b0", "b1", "b2", "b3", "plugin"):
        setattr(model, f"_ct_{module_name}_updated_this_epoch", False)
    model._ct_epoch_binary_rows = {"presence": [], "alpha": [], "alpha_uplift": []}
    model._ct_epoch_acquisition_totals = {
        population: {
            "eligible_rows": 0.0,
            "retained_rows": 0.0,
            "pool_targets": 0.0,
            "sampled_targets": 0.0,
            "available_rows": 0.0,
            "role_satisfied_rows": 0.0,
            "boundary_ratio_sum": 0.0,
            "boundary_ratio_count": 0.0,
            "support_truncated_rows": 0.0,
            "support_volume_sum": 0.0,
            "support_volume_count": 0.0,
            "recovery_positive_rows": 0.0,
            "recovery_fallback_rows": 0.0,
        }
        for population in ("candidate0", "auxiliary_train")
    }
    model._ct_selector_epoch = {
        "gap_counts": {"1": {}, "2": {}},
        "available": {"1": 0, "2": 0},
        "satisfied": {"1": 0, "2": 0},
        "migration_comparisons": 0,
        "migrations": 0,
        "current": {},
    }
    if bool(getattr(model.config, "ct_online_recursive_training", False)):
        pending_boundary = getattr(model, "_ct_pending_recursive_state_boundary", None)
        if pending_boundary is not None:
            if pending_boundary.get("next_epoch_reset") is not True:
                raise RuntimeError(
                    "resume recursive-state contract does not reset at the "
                    "epoch boundary"
                )
            model._ct_pending_recursive_state_boundary = None
        model._ct_recursive_states = {}
        model._ct_online_batch_context = []
        train_loader = getattr(model.trainer, "train_dataloader", None)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(int(model.current_epoch))


def _finish_acquisition(model):
    acquisition = getattr(model, "_ct_epoch_acquisition_totals", {})
    for population, totals in acquisition.items():
        totals["row_recall"] = (
            totals["retained_rows"] / totals["eligible_rows"]
            if totals["eligible_rows"] > 0
            else None
        )
        totals["point_recall"] = (
            totals["sampled_targets"] / totals["pool_targets"]
            if totals["pool_targets"] > 0
            else None
        )
        totals["role_satisfaction_rate"] = (
            totals["role_satisfied_rows"] / totals["available_rows"]
            if totals["available_rows"] > 0
            else None
        )
        totals["boundary_ratio_mean"] = (
            totals["boundary_ratio_sum"] / totals["boundary_ratio_count"]
            if totals["boundary_ratio_count"] > 0
            else None
        )
        totals["support_truncation_rate"] = (
            totals["support_truncated_rows"] / totals["available_rows"]
            if totals["available_rows"] > 0
            else None
        )
        totals["support_volume_mean"] = (
            totals["support_volume_sum"] / totals["support_volume_count"]
            if totals["support_volume_count"] > 0
            else None
        )
        for metric in (
            "eligible_rows",
            "retained_rows",
            "pool_targets",
            "sampled_targets",
            "available_rows",
            "role_satisfied_rows",
            "boundary_ratio_sum",
            "boundary_ratio_count",
            "support_truncated_rows",
            "support_volume_sum",
            "support_volume_count",
            "recovery_positive_rows",
            "recovery_fallback_rows",
        ):
            model.log(
                f"ct_acquisition/{population}_{metric}",
                float(totals[metric]),
                on_step=False,
                on_epoch=True,
            )
        for metric in (
            "row_recall",
            "point_recall",
            "role_satisfaction_rate",
            "boundary_ratio_mean",
            "support_truncation_rate",
            "support_volume_mean",
        ):
            if totals[metric] is not None:
                model.log(
                    f"ct_acquisition/{population}_{metric}",
                    float(totals[metric]),
                    on_step=False,
                    on_epoch=True,
                )
    if not acquisition or int(getattr(model, "global_rank", 0)) != 0:
        return
    logger = getattr(model, "logger", None)
    log_dir = getattr(logger, "log_dir", None)
    if log_dir is None:
        log_dir = getattr(logger, "save_dir", ".")
    output = (
        Path(log_dir)
        / "acquisition_supply"
        / (f"epoch_{int(model.current_epoch) + 1:02d}.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    selector = getattr(model, "_ct_selector_epoch", {})
    comparisons = int(selector.get("migration_comparisons", 0))
    selector_summary = {
        "gap_counts": selector.get("gap_counts", {}),
        "boundary_available": int(selector.get("available", {}).get("1", 0)),
        "outside_available": int(selector.get("available", {}).get("2", 0)),
        "boundary_satisfied_rate": (
            selector.get("satisfied", {}).get("1", 0)
            / max(selector.get("available", {}).get("1", 0), 1)
        ),
        "outside_satisfied_rate": (
            selector.get("satisfied", {}).get("2", 0)
            / max(selector.get("available", {}).get("2", 0), 1)
        ),
        "migration_comparisons": comparisons,
        "migration_rate": (
            selector.get("migrations", 0) / comparisons if comparisons else None
        ),
    }
    output.write_text(
        json.dumps(
            {
                "schema": "ct_seqtrack.acquisition_training_supply.v2",
                "experiment_name": str(
                    getattr(model.config, "experiment_name", "unknown")
                ),
                "seed": int(getattr(model.config, "seed", 42) or 42),
                "epoch": int(model.current_epoch) + 1,
                "populations": acquisition,
                "selector": selector_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    model._ct_selector_previous = dict(selector.get("current", {}))


def on_train_epoch_end(model):
    if int(getattr(model, "current_epoch", 0)) == 0:
        missing_updates = [
            name
            for name in getattr(model, "_ct_optimizer_names", ())
            if int(getattr(model, f"ct_{name}_update_step").item()) <= 0
        ]
        if missing_updates:
            raise RuntimeError(
                "v25 epoch 1 requires a nonzero update count for every "
                "enabled module: " + ", ".join(missing_updates)
            )
    if model.config.optimizer.lower() != "adamonecycle":
        schedulers = model.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for name, scheduler in zip(model._ct_optimizer_names, schedulers):
            if getattr(model, f"_ct_{name}_updated_this_epoch", False):
                scheduler.step()
    model._ct_epoch_boundary_complete = True
    if hasattr(model, "_ct_named_parameters_by_module"):
        record_parameter_hash(model, f"epoch_{int(model.current_epoch) + 1}_end")
    _finish_acquisition(model)
    rows = getattr(model, "_ct_epoch_binary_rows", {})
    epoch_metrics = {}
    for name in ("presence", "alpha"):
        entries = rows.get(name, [])
        if not entries:
            continue
        metrics = binary_curve_metrics_numpy(
            np.concatenate([entry[0] for entry in entries]),
            np.concatenate([entry[1] for entry in entries]),
        )
        for metric in (
            "auroc",
            "auprc",
            "positive_mean",
            "negative_mean",
            "positive_count",
            "negative_count",
        ):
            epoch_metrics[f"{name}_{metric}"] = metrics[metric]
        for index, calibration in enumerate(metrics["calibration"]):
            for metric in ("count", "confidence", "positive_rate"):
                epoch_metrics[f"{name}_calibration_bin{index}_{metric}"] = calibration[
                    metric
                ]
    if rows.get("alpha_uplift"):
        epoch_metrics["alpha_counterfactual_uplift"] = float(
            np.mean(np.concatenate(rows["alpha_uplift"]))
        )
    if epoch_metrics:
        model.logger.experiment.add_scalars(
            "ct_epoch_calibration", epoch_metrics, global_step=model.global_step
        )


__all__ = [
    "accumulate_joint_binary_rows",
    "binary_curve_metrics_numpy",
    "on_train_epoch_end",
    "on_train_epoch_start",
    "parameter_group_sha256",
    "record_parameter_hash",
]
