"""Causal box/time-only B1 pre-pass shared by training and inference."""

import numpy as np
import torch

from datasets import points_utils
from datasets.misc_utils import (
    build_effective_time_fields,
    build_time_fields,
    get_history_frame_ids_and_masks,
    get_last_n_bounding_boxes,
    normalize_dynamics_time_mode,
)


@torch.no_grad()
def predict_motion_from_history(model, ref_boxs, delta_t, valid_mask, current_delta_t):
    if not model.use_b1motion_v3:
        raise RuntimeError("predict_motion_from_history requires use_b1motion_v3")
    parameter = next(model.physical_motion_encoder.parameters())
    device, dtype = parameter.device, parameter.dtype

    def as_tensor(value):
        return torch.as_tensor(value, device=device, dtype=dtype)

    ref_boxs = as_tensor(ref_boxs)
    delta_t = as_tensor(delta_t)
    valid_mask = as_tensor(valid_mask)
    current_delta_t = as_tensor(current_delta_t)
    if ref_boxs.dim() == 2:
        ref_boxs = ref_boxs.unsqueeze(0)
    if delta_t.dim() == 1:
        delta_t = delta_t.unsqueeze(0)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    if current_delta_t.dim() == 0:
        current_delta_t = current_delta_t.unsqueeze(0)
    motion_ref_boxs = (
        torch.flip(ref_boxs, dims=(1,))
        if bool(getattr(model.config, "shuffle_b1_signal", False))
        else ref_boxs
    )
    if model.use_ct_joint_full and not model.ct_enable_b1:
        prediction = model.physical_motion_encoder.kinematic_fallback(
            motion_ref_boxs, delta_t, valid_mask, current_delta_t
        )
    else:
        prediction = model.physical_motion_encoder(
            motion_ref_boxs, delta_t, valid_mask, current_delta_t
        )
    if bool(getattr(model.config, "force_b1_invalid", False)):
        prediction = dict(prediction)
        prediction["valid"] = torch.zeros_like(prediction["valid"])
        prediction["source_id"] = torch.zeros_like(prediction["source_id"])
    return {
        key: prediction[key].detach()
        for key in (
            "mu_xy",
            "kinematic_prior_xy",
            "log_sigma_parallel_perp",
            "covariance_xy",
            "basis_velocity_xy",
            "direction_xy",
            "velocity_xy",
            "feature",
            "valid",
            "gap_ratio",
            "source_id",
        )
    }


def build_motion_prepass_inputs(
    model,
    history_boxes,
    history_ids,
    valid_mask,
    history_timestamps,
    current_timestamp,
    effective_history_timestamps,
    effective_current_timestamp,
    dynamics_time_mode_value,
    current_frame_id,
):
    if int(current_frame_id) <= 0:
        raise ValueError("motion pre-pass is only defined after frame 0")
    if len(history_boxes) != model.hist_num:
        return None
    default_step = float(
        getattr(
            model.config, "default_time_step", getattr(model.config, "time_step", 0.1)
        )
    )
    pseudo_step = float(getattr(model.config, "pseudo_time_step", 0.1))
    real_fields = build_time_fields(
        history_timestamps,
        current_timestamp,
        frame_ids=history_ids,
        current_frame_id=current_frame_id,
        use_real_time=bool(getattr(model.config, "use_real_time", True)),
        default_step=default_step,
        pseudo_step=pseudo_step,
    )
    effective_fields = build_effective_time_fields(
        normalize_dynamics_time_mode(dynamics_time_mode_value),
        real_fields,
        effective_frame_timestamps=effective_history_timestamps,
        effective_current_timestamp=effective_current_timestamp,
        frame_ids=history_ids,
        current_frame_id=current_frame_id,
        default_step=float(
            getattr(model.config, "dynamics_fixed_delta_t", default_step)
        ),
        pseudo_step=pseudo_step,
    )
    effective_delta_t = effective_fields[1]
    anchor = history_boxes[0]
    local_rows = []
    for box in history_boxes:
        local_box = points_utils.transform_box(box, anchor)
        yaw = (
            local_box.orientation.degrees * local_box.orientation.axis[-1]
            if bool(getattr(model.config, "degrees", True))
            else local_box.orientation.radians * local_box.orientation.axis[-1]
        )
        local_rows.append(np.append(local_box.center, yaw).astype(np.float32))
    return {
        "ref_boxs": np.stack(local_rows, axis=0),
        "delta_t": np.asarray(effective_delta_t, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=np.float32),
        "current_delta_t": np.float32(effective_delta_t[0]),
    }


def empty_motion_prepass_prediction(model):
    feature_dim = int(getattr(model.config, "motion_v3_hidden_dim", 128))
    return {
        "mu_xy": np.zeros(2, dtype=np.float32),
        "kinematic_prior_xy": np.zeros(2, dtype=np.float32),
        "log_sigma_parallel_perp": np.zeros(2, dtype=np.float32),
        "covariance_xy": np.eye(2, dtype=np.float32),
        "basis_velocity_xy": np.zeros(2, dtype=np.float32),
        "direction_xy": np.asarray((1.0, 0.0), dtype=np.float32),
        "velocity_xy": np.zeros(2, dtype=np.float32),
        "feature": np.zeros(feature_dim, dtype=np.float32),
        "valid": False,
        "gap_ratio": 1.0,
        "source_id": 0,
        "current_delta_t": float(getattr(model.config, "default_time_step", 0.5)),
    }


def unbatch_motion_prepass_predictions(tensor_prediction, current_delta_t):
    results = []
    for row in range(len(current_delta_t)):
        result = {}
        for key, value in tensor_prediction.items():
            item = value[row].detach().cpu()
            if key == "valid":
                result[key] = bool(item.item() > 0)
            elif key == "source_id":
                result[key] = int(item.item())
            elif key == "gap_ratio":
                result[key] = float(item.item())
            else:
                result[key] = item.numpy()
        result["current_delta_t"] = float(current_delta_t[row])
        finite_keys = (
            "mu_xy",
            "log_sigma_parallel_perp",
            "direction_xy",
            "velocity_xy",
        )
        if all(
            key in result and bool(np.isfinite(result[key]).all())
            for key in finite_keys
        ):
            result["log_sigma_parallel_perp"] = np.clip(
                np.asarray(result["log_sigma_parallel_perp"], dtype=np.float32),
                -4.0,
                2.5,
            )
        else:
            result["valid"] = False
            result["source_id"] = 0
        results.append(result)
    return results


@torch.no_grad()
def predict_motion_prepass_contract(
    model,
    history_boxes,
    history_ids,
    valid_mask,
    history_timestamps,
    current_timestamp,
    effective_history_timestamps,
    effective_current_timestamp,
    dynamics_time_mode_value,
    current_frame_id,
):
    inputs = build_motion_prepass_inputs(
        model,
        history_boxes,
        history_ids,
        valid_mask,
        history_timestamps,
        current_timestamp,
        effective_history_timestamps,
        effective_current_timestamp,
        dynamics_time_mode_value,
        current_frame_id,
    )
    if inputs is None:
        return empty_motion_prepass_prediction(model)
    prediction = predict_motion_from_history(
        model,
        inputs["ref_boxs"],
        inputs["delta_t"],
        inputs["valid_mask"],
        inputs["current_delta_t"],
    )
    return unbatch_motion_prepass_predictions(prediction, [inputs["current_delta_t"]])[
        0
    ]


@torch.no_grad()
def predict_motion_prepass(model, sequence, frame_id, results_bbs):
    """Predict before cropping without reading the current annotation."""
    if frame_id <= 0:
        raise ValueError("motion pre-pass is only defined after frame 0")
    history_ids, valid_mask = get_history_frame_ids_and_masks(frame_id, model.hist_num)
    history_boxes = get_last_n_bounding_boxes(results_bbs, valid_mask)
    previous_frames = [sequence[index] for index in history_ids]
    current_frame = sequence[frame_id]
    return predict_motion_prepass_contract(
        model,
        history_boxes,
        history_ids,
        valid_mask,
        [frame.get("timestamp") for frame in previous_frames],
        current_frame.get("timestamp"),
        [frame.get("_ct_effective_timestamp") for frame in previous_frames],
        current_frame.get("_ct_effective_timestamp"),
        current_frame.get(
            "_ct_dynamics_time_mode",
            getattr(model.config, "dynamics_time_mode", "true"),
        ),
        frame_id,
    )


__all__ = [
    "build_motion_prepass_inputs",
    "empty_motion_prepass_prediction",
    "predict_motion_from_history",
    "predict_motion_prepass",
    "predict_motion_prepass_contract",
    "unbatch_motion_prepass_predictions",
]
