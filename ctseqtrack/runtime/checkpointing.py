"""Exact-resume checkpoint state for formal v25 training."""

import copy
import hashlib

from ctseqtrack.runtime.acquisition import validate_preflight_artifact
from ctseqtrack.runtime.calibration import (
    b1_calibration_config_sha256,
    validate_b1_calibration_state,
)
from ctseqtrack.runtime.contracts import (
    build_online_resume_contract,
    validate_b2_method_promotion,
)
from ctseqtrack.runtime.optimization import capture_global_rng_state


def _module_audit(model):
    optimizer_lrs = {}
    trainer_optimizers = list(
        getattr(getattr(model, "_trainer", None), "optimizers", [])
    )
    for name, optimizer in zip(
        getattr(model, "_ct_optimizer_names", ()), trainer_optimizers
    ):
        optimizer_lrs[name] = [float(group["lr"]) for group in optimizer.param_groups]
    module_hashes = {}
    for name, parameters in getattr(
        model, "_ct_named_parameters_by_module", {}
    ).items():
        digest = hashlib.sha256()
        for parameter_name, parameter in sorted(parameters):
            tensor = parameter.detach().cpu().contiguous()
            digest.update(parameter_name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        module_hashes[name] = digest.hexdigest()
    return {
        "schema": "ct_seqtrack.module_audit.v1",
        "epoch": int(getattr(model, "current_epoch", 0)),
        "parameter_sha256": module_hashes,
        "optimizer_lr": optimizer_lrs,
        "update_steps": {
            name: int(getattr(model, f"ct_{name}_update_step").item())
            for name in getattr(model, "_ct_optimizer_names", ())
        },
        "last_gradient_norm": dict(getattr(model, "_ct_last_gradient_norm", {})),
        "b0_hash_timeline": copy.deepcopy(
            getattr(model, "_ct_parameter_hash_timeline", [])
        ),
    }


def save_v25_checkpoint(model, checkpoint):
    if bool(getattr(model.config, "ct_online_recursive_training", False)):
        checkpoint["ct_online_resume_contract"] = build_online_resume_contract(
            model.config
        )
        checkpoint["ct_global_rng_state"] = capture_global_rng_state()
        batch_progress = getattr(
            getattr(
                getattr(getattr(model, "_trainer", None), "fit_loop", None),
                "epoch_loop",
                None,
            ),
            "batch_progress",
            None,
        )
        checkpoint["ct_epoch_boundary_complete"] = bool(
            model._ct_epoch_boundary_complete
            or getattr(batch_progress, "is_last_batch", False)
        )
        checkpoint["ct_recursive_state_boundary"] = {
            "schema": "ct_seqtrack.recursive_state_boundary.v1",
            "next_epoch_reset": True,
            "completed_epoch": int(getattr(model, "current_epoch", 0)),
            "active_state_count": len(getattr(model, "_ct_recursive_states", {})),
        }
    for attribute, key in (
        ("_ct_b2_method_promotion", "ct_b2_method_promotion"),
        ("_ct_acquisition_preflight", "ct_acquisition_preflight"),
        ("_ct_b2_promotion", "ct_b2_promotion"),
        ("_b1_uncertainty_calibration", "b1_uncertainty_calibration"),
    ):
        if hasattr(model, attribute):
            checkpoint[key] = copy.deepcopy(getattr(model, attribute))
    checkpoint["ct_isolated_scalers"] = {
        name: scaler.state_dict() for name, scaler in model._ct_scalers.items()
    }
    checkpoint["ct_module_audit"] = _module_audit(model)


def _restore_b1_calibration(model, checkpoint):
    calibration = checkpoint.get("b1_uncertainty_calibration")
    if isinstance(calibration, dict):
        validate_b1_calibration_state(calibration, checkpoint.get("state_dict", {}))
    if model.require_b1_calibration_artifact:
        if (
            not isinstance(calibration, dict)
            or calibration.get("schema") != "ct_seqtrack.b1_uncertainty_calibration.v2"
            or len(calibration.get("fixed_margin_parallel_perpendicular_95", [])) != 2
        ):
            raise RuntimeError(
                "this configuration requires a verified v2 B1 "
                "calibration artifact with fixed residual margins"
            )
        if (
            len(
                calibration.get(
                    "standardized_abs_residual_q90_parallel_perpendicular", []
                )
            )
            != 2
        ):
            raise RuntimeError(
                "contract-v3 calibration lacks standardized residual q90"
            )
        source = calibration.get("source_artifact", {})
        if (
            source.get("partition") != "calibration"
            or source.get("dataset") != str(getattr(model.config, "dataset", "unknown"))
            or source.get("split") != str(getattr(model.config, "train_split", "train"))
            or source.get("b1_config_sha256")
            != b1_calibration_config_sha256(model.config)
        ):
            raise RuntimeError("B1 calibration dataset/partition/config mismatch")
    if model.require_b1_calibration_passed and (
        not isinstance(calibration, dict)
        or not bool(calibration.get("promotion", {}).get("passed"))
    ):
        raise RuntimeError("this configuration requires a promoted B1 calibration")
    if not isinstance(calibration, dict):
        return
    model._b1_uncertainty_calibration = copy.deepcopy(calibration)
    margins = calibration.get("fixed_margin_parallel_perpendicular_95")
    if isinstance(margins, (list, tuple)) and len(margins) == 2:
        model.config.search_v3_fixed_margin_parallel = float(margins[0])
        model.config.search_v3_fixed_margin_perpendicular = float(margins[1])
    standardized_q90 = calibration.get(
        "standardized_abs_residual_q90_parallel_perpendicular"
    )
    if isinstance(standardized_q90, (list, tuple)) and len(standardized_q90) == 2:
        model.config["search_v3_standardized_residual_q90_parallel_perpendicular"] = [
            float(value) for value in standardized_q90
        ]


def load_v25_checkpoint(model, checkpoint):
    module_audit = checkpoint.get("ct_module_audit")
    if isinstance(module_audit, dict):
        timeline = module_audit.get("b0_hash_timeline")
        if isinstance(timeline, list):
            model._ct_parameter_hash_timeline = copy.deepcopy(timeline)
    if bool(getattr(model.config, "ct_online_recursive_training", False)) and not bool(
        getattr(model.config, "test", False)
    ):
        rng_state = checkpoint.get("ct_global_rng_state")
        if (
            not isinstance(rng_state, dict)
            or rng_state.get("schema") != "ct_seqtrack.global_rng.v1"
        ):
            raise ValueError("exact online resume requires ct_seqtrack.global_rng.v1")
        model._ct_pending_global_rng_state = copy.deepcopy(rng_state)
        recursive_boundary = checkpoint.get("ct_recursive_state_boundary")
        if (
            not isinstance(recursive_boundary, dict)
            or recursive_boundary.get("schema")
            != "ct_seqtrack.recursive_state_boundary.v1"
            or recursive_boundary.get("next_epoch_reset") is not True
        ):
            raise ValueError(
                "exact online resume requires the recursive-state epoch "
                "boundary contract"
            )
        model._ct_pending_recursive_state_boundary = copy.deepcopy(recursive_boundary)
    if model.ct_enable_b2:
        model._ct_acquisition_preflight = validate_preflight_artifact(
            checkpoint.get("ct_acquisition_preflight"), model.config
        )
    model._ct_pending_scaler_state = copy.deepcopy(
        checkpoint.get("ct_isolated_scalers")
    )
    if model.ct_enable_b3:
        model._ct_b2_method_promotion = validate_b2_method_promotion(
            checkpoint.get("ct_b2_method_promotion"), model.config
        )
        final_promotion = checkpoint.get("ct_b2_promotion")
        if (
            isinstance(final_promotion, dict)
            and final_promotion.get("schema") == "ct_seqtrack.b2_evidence_promotion.v4"
            and bool(final_promotion.get("passed"))
        ):
            model._ct_b2_promotion = copy.deepcopy(final_promotion)
    _restore_b1_calibration(model, checkpoint)


__all__ = ["load_v25_checkpoint", "save_v25_checkpoint"]
