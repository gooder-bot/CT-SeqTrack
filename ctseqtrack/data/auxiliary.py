"""Pure B1 auxiliary-history construction for v25 samples."""

import copy

import numpy as np
from pyquaternion import Quaternion

from datasets.misc_utils import build_effective_time_fields, build_time_fields
from utils.candidate_utils import (
    boxes_to_anchor_parameters,
    build_b1_physical_contract,
)


def build_motion_aux_contract(
    *,
    data,
    config,
    use_motion_v3,
    observation_only,
    causal_temporal_policy,
    num_hist,
    this_frame,
    canonical_this_box,
    current_timestamp,
    this_frame_id,
    dynamics_time_mode,
    use_real_time,
    default_time_step,
    pseudo_time_step,
):
    motion_aux_contract = None
    if use_motion_v3 and not observation_only and not causal_temporal_policy:
        motion_aux_prev_frames = data.get("motion_aux_prev_frames")
        if motion_aux_prev_frames is None:
            raise KeyError("B1motion-v3 training requires motion_aux_prev_frames")
        motion_aux_keys = sorted(motion_aux_prev_frames, key=lambda key: abs(int(key)))
        motion_aux_ground_truth_boxs = [
            motion_aux_prev_frames[key]["3d_bbox"] for key in motion_aux_keys
        ]
        motion_aux_canonical_boxs = list(motion_aux_ground_truth_boxs)
        motion_aux_valid_mask = list(data["motion_aux_valid_mask"])
        motion_aux_frame_ids = list(data["motion_aux_frame_ids"])
        online_motion_aux_state = data.get("online_motion_aux_state")
        if online_motion_aux_state is not None:
            aux_rows = np.asarray(
                online_motion_aux_state["history_boxes_world"], dtype=np.float64
            )
            if aux_rows.shape != (num_hist, 7):
                raise ValueError(
                    f"online auxiliary history must have shape " f"{(num_hist, 7)}"
                )
            online_aux_boxs = []
            for source_box, row in zip(motion_aux_canonical_boxs, aux_rows):
                state_box = copy.deepcopy(source_box)
                state_box.center = row[:3].copy()
                state_box.wlh = row[3:6].copy()
                state_box.orientation = Quaternion(
                    axis=[0, 0, 1], radians=float(row[6])
                )
                online_aux_boxs.append(state_box)
            motion_aux_canonical_boxs = online_aux_boxs
        motion_aux_anchor = motion_aux_canonical_boxs[0]
        motion_aux_ref_boxs = boxes_to_anchor_parameters(
            motion_aux_canonical_boxs,
            motion_aux_anchor,
            degrees=config.degrees,
        )
        if not np.allclose(motion_aux_ref_boxs[0], 0.0, rtol=0.0, atol=1e-5):
            raise ValueError(
                "B1motion-v3 newest auxiliary history must equal its anchor"
            )
        motion_aux_prev_timestamps = (
            list(online_motion_aux_state["history_timestamps"])
            if online_motion_aux_state is not None
            else [
                motion_aux_prev_frames[key].get("timestamp") for key in motion_aux_keys
            ]
        )
        motion_aux_real_time_fields = build_time_fields(
            motion_aux_prev_timestamps,
            current_timestamp,
            frame_ids=motion_aux_frame_ids,
            current_frame_id=this_frame_id,
            use_real_time=use_real_time,
            default_step=default_time_step,
            pseudo_step=pseudo_time_step,
        )
        motion_aux_effective_time_fields = build_effective_time_fields(
            dynamics_time_mode,
            motion_aux_real_time_fields,
            effective_frame_timestamps=[
                motion_aux_prev_frames[key].get("_ct_effective_timestamp")
                for key in motion_aux_keys
            ],
            effective_current_timestamp=this_frame.get("_ct_effective_timestamp"),
            frame_ids=motion_aux_frame_ids,
            current_frame_id=this_frame_id,
            default_step=float(
                getattr(config, "dynamics_fixed_delta_t", default_time_step)
            ),
            pseudo_step=pseudo_time_step,
        )
        motion_aux_delta_t_real = motion_aux_real_time_fields[1]
        motion_aux_delta_t_effective = motion_aux_effective_time_fields[1]
        motion_aux_current_delta_t_real = (
            motion_aux_delta_t_real[0] if motion_aux_delta_t_real else default_time_step
        )
        motion_aux_current_delta_t_effective = (
            motion_aux_delta_t_effective[0]
            if motion_aux_delta_t_effective
            else float(getattr(config, "dynamics_fixed_delta_t", default_time_step))
        )
        motion_aux_physical = build_b1_physical_contract(
            canonical_this_box,
            motion_aux_ground_truth_boxs,
            motion_aux_canonical_boxs,
            motion_aux_current_delta_t_real,
            degrees=config.degrees,
            eps=1e-3,
            history_delta_t=motion_aux_delta_t_real,
            history_valid_mask=motion_aux_valid_mask,
            dt_floor=float(getattr(config, "motion_v3_dt_floor", 0.05)),
            support_cap_pp=(
                float(getattr(config, "motion_v3_support_cap_parallel", 4.0)),
                float(getattr(config, "motion_v3_support_cap_perpendicular", 3.0)),
            ),
        )
        if not np.array_equal(motion_aux_ref_boxs, motion_aux_physical["ref_boxs"]):
            raise RuntimeError("B1 auxiliary input and physical-label axes diverged")
        motion_aux_target_xy = motion_aux_physical["target_xy"]
        motion_aux_diagnostics = np.zeros((num_hist, 6), dtype=np.float32)
        motion_aux_diagnostic_valid = np.zeros(num_hist, dtype=np.int64)
        if online_motion_aux_state is not None:
            motion_aux_diagnostics = np.asarray(
                online_motion_aux_state.get(
                    "history_observation_diagnostics", motion_aux_diagnostics
                ),
                dtype=np.float32,
            )
            motion_aux_diagnostic_valid = np.asarray(
                online_motion_aux_state.get(
                    "history_diagnostic_valid_mask", motion_aux_diagnostic_valid
                ),
                dtype=np.int64,
            )
        motion_aux_contract = {
            "motion_aux_ref_boxs": motion_aux_ref_boxs.astype("float32"),
            "motion_aux_delta_t": np.asarray(
                motion_aux_delta_t_effective, dtype=np.float32
            ),
            "motion_aux_current_delta_t": np.float32(
                motion_aux_current_delta_t_effective
            ),
            "motion_aux_physical_delta_t": np.float32(motion_aux_current_delta_t_real),
            "motion_aux_valid_mask": np.asarray(motion_aux_valid_mask, dtype=np.int64),
            "motion_aux_history_observation_diagnostics": motion_aux_diagnostics,
            "motion_aux_history_diagnostic_valid_mask": motion_aux_diagnostic_valid,
            "motion_aux_physical_target_xy": motion_aux_physical[
                "physical_target_xy"
            ].astype("float32"),
            "motion_aux_endpoint_target_xy": motion_aux_physical[
                "endpoint_target_xy"
            ].astype("float32"),
            "motion_aux_anchor_drift_xy": motion_aux_physical["anchor_drift_xy"].astype(
                "float32"
            ),
            "motion_aux_gt_cv_difficulty": np.float32(
                motion_aux_physical["gt_cv_difficulty"]
            ),
            "motion_aux_support_cap_pp": motion_aux_physical["support_cap_pp"].astype(
                "float32"
            ),
            "motion_aux_support_censored": np.float32(
                motion_aux_physical["support_censored"]
            ),
            "motion_aux_target_xy": motion_aux_target_xy.astype("float32"),
            "motion_aux_query_gap_frames": np.int64(data["motion_aux_offsets"][0]),
        }
    return motion_aux_contract
