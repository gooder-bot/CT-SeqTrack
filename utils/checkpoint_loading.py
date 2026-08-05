"""Side-effect-free checkpoint loading for offline CT-SeqTrack exporters."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from utils.replay_cache import (
    b1_calibration_config_sha256,
    validate_b1_calibration_state,
)


def apply_b1_calibration_contract(model, payload, state_dict):
    """Validate and attach B1 calibration metadata for manual loaders.

    Lightning invokes ``on_load_checkpoint`` for ordinary resume paths, but
    offline exporters and the B3 rollout loader restore tensors directly.
    Keeping this check here prevents those paths from silently bypassing the
    dataset/config/promotion and scale-buffer binding contract.
    """
    calibration = payload.get("b1_uncertainty_calibration")
    if isinstance(calibration, dict):
        validate_b1_calibration_state(calibration, state_dict)
    if bool(getattr(model, "require_b1_calibration_artifact", False)):
        if (not isinstance(calibration, dict)
                or calibration.get("schema")
                != "ct_seqtrack.b1_uncertainty_calibration.v2"
                or len(calibration.get(
                    "fixed_margin_parallel_perpendicular_95", [])) != 2):
            raise RuntimeError(
                "checkpoint lacks the required B1 v2 calibration artifact")
        source_metadata = calibration.get("source_artifact", {})
        if (source_metadata.get("partition") != "calibration"
                or source_metadata.get("dataset") != str(getattr(
                    model.config, "dataset", "unknown"))
                or source_metadata.get("split") != str(getattr(
                    model.config, "train_split", "train"))
                or source_metadata.get("b1_config_sha256")
                != b1_calibration_config_sha256(model.config)):
            raise RuntimeError("B1 calibration provenance mismatch")
    if bool(getattr(model, "require_b1_calibration_passed", False)):
        if (not isinstance(calibration, dict)
                or not bool(calibration.get(
                    "promotion", {}).get("passed"))):
            raise RuntimeError("checkpoint lacks promoted B1 uncertainty")
    if isinstance(calibration, dict):
        model._b1_uncertainty_calibration = copy.deepcopy(calibration)
        margins = calibration.get(
            "fixed_margin_parallel_perpendicular_95")
        if isinstance(margins, (list, tuple)) and len(margins) == 2:
            model.config.search_v3_fixed_margin_parallel = float(margins[0])
            model.config.search_v3_fixed_margin_perpendicular = float(
                margins[1])
    return calibration


def load_initial_weights(model, checkpoint_path, report_path=None):
    """Load shape-compatible model tensors without importing ``main.py``.

    Offline exporters deliberately disable B2/B3 training paths, so they need
    the frozen B0/B1 state and calibration metadata but no Trainer or CLI
    initialization side effects.
    """
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    source = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(source, dict):
        raise TypeError("checkpoint does not contain a state_dict mapping")
    target = model.state_dict()
    candidates = [("none", source)]
    for prefix in ("model.", "module."):
        candidates.append((prefix, {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in source.items()
        }))
    selected_prefix, normalized = max(
        candidates,
        key=lambda item: sum(
            key in target and target[key].shape == value.shape
            for key, value in item[1].items()),
    )
    matched = {
        key: value for key, value in normalized.items()
        if key in target and target[key].shape == value.shape
    }
    critical = (
        "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
        "feature_pointnet.", "Transformer.", "physical_motion_encoder.",
    )
    missing = [
        prefix for prefix in critical
        if not any(key.startswith(prefix) for key in matched)]
    if missing:
        raise RuntimeError(
            "checkpoint is missing baseline prefixes: " + ", ".join(missing))
    nonfinite = [
        key for key, value in matched.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())]
    if nonfinite:
        raise RuntimeError(
            "checkpoint contains non-finite tensors: "
            + ", ".join(nonfinite[:20]))
    target.update(matched)
    model.load_state_dict(target, strict=True)
    apply_b1_calibration_contract(model, payload, normalized)
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "selected_prefix_strip": selected_prefix,
        "matched_tensor_count": len(matched),
        "target_tensor_count": len(target),
    }
