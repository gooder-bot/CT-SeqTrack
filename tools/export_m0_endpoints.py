"""Export frozen recursive tracking outputs for the CT-SeqTrack M0 stage.

The exporter intentionally lives outside the training/evaluation classes.  It
replays the existing recursive prediction path, writes one row per endpoint,
and can optionally run two passive history paths without feeding either result
back into the tracker.  This keeps M0 diagnostics from changing the model that
is being diagnosed.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


crop_diag = None
estimate_accuracy = None
estimate_overlap = None
get_dataset = None
get_model = None
points_utils = None
rec_diag = None
torch = None


def load_runtime_dependencies():
    global crop_diag, estimate_accuracy, estimate_overlap
    global get_dataset, get_model, points_utils, rec_diag, torch
    if torch is not None:
        return

    import torch as torch_module

    from datasets import get_dataset as get_dataset_function
    from datasets import points_utils as points_utils_module
    from models import get_model as get_model_function
    from tools import diagnose_recursive_crop_reachability as rec_diag_module
    from utils.metrics import estimateAccuracy, estimateOverlap

    rec_diag_module.load_runtime_dependencies()
    crop_diag = rec_diag_module.crop_diag
    estimate_accuracy = estimateAccuracy
    estimate_overlap = estimateOverlap
    get_dataset = get_dataset_function
    get_model = get_model_function
    points_utils = points_utils_module
    rec_diag = rec_diag_module
    torch = torch_module


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def tensor_scalar(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Expected scalar tensor for {key}, got shape={tuple(value.shape)}"
            )
        value = value.detach().reshape(-1)[0].cpu().item()
    return finite_float(value)


def tensor_vector(mapping, key, width):
    value = mapping.get(key)
    if value is None:
        return [None] * int(width)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value).reshape(-1)
    if value.size < width:
        raise ValueError(f"{key} has {value.size} values, expected at least {width}")
    return [finite_float(item) for item in value[:width]]


def tensor_json(mapping, key):
    value = mapping.get(key)
    if value is None:
        return ""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return json.dumps(array.tolist(), separators=(",", ":"), allow_nan=False)


def box_center(box):
    return np.asarray(box.center, dtype=np.float64)


def box_yaw(box, degrees=False):
    if degrees:
        return float(box.orientation.degrees * box.orientation.axis[-1])
    return float(box.orientation.radians * box.orientation.axis[-1])


def wrapped_angle_difference(left, right, degrees=False):
    period = 360.0 if degrees else 2.0 * math.pi
    half = period / 2.0
    return abs(float((float(left) - float(right) + half) % period - half))


def box_fields(prefix, box, degrees=False):
    if box is None:
        return {
            f"{prefix}_x": None,
            f"{prefix}_y": None,
            f"{prefix}_z": None,
            f"{prefix}_yaw": None,
        }
    center = box_center(box)
    return {
        f"{prefix}_x": float(center[0]),
        f"{prefix}_y": float(center[1]),
        f"{prefix}_z": float(center[2]),
        f"{prefix}_yaw": box_yaw(box, degrees=degrees),
    }


def git_state():
    def run(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def parse_offsets(text, hist_num):
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != int(hist_num):
        raise ValueError(
            f"History path must contain hist_num={hist_num} offsets, got {values}"
        )
    if any(value <= 0 for value in values):
        raise ValueError("History offsets must be positive")
    if any(values[index] >= values[index + 1] for index in range(len(values) - 1)):
        raise ValueError("History offsets must increase from recent to older")
    return values


def stable_endpoint_seed(seed, tracklet_id, frame_index):
    payload = f"{int(seed)}:{int(tracklet_id)}:{int(frame_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def synthetic_path_indices(frame_index, offsets):
    history = [int(frame_index) - int(offset) for offset in offsets]
    if any(index < 0 for index in history):
        return None
    return list(reversed(history)) + [int(frame_index)]


def build_candidate_box(model, output, reference_box):
    estimation_box = output["aux_estimation_boxes"]
    estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()
    if len(estimation_box.shape) == 3:
        best_box_idx = estimation_box_cpu[:, 4].argmax()
        estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]
    return points_utils.getOffsetBB(
        reference_box,
        estimation_box_cpu,
        degrees=model.config.degrees,
        use_z=model.config.use_z,
        limit_box=model.config.limit_box,
    )


def run_forward(model, data_dict, reference_box):
    empty_fallback = bool(torch.sum(data_dict["points"][:, :, :3]).item() == 0.0)
    if empty_fallback:
        return {
            "candidate_box": copy.deepcopy(reference_box),
            "reference_box": reference_box,
            "data": data_dict,
            "output": None,
            "empty_fallback": True,
            "forward_ran": False,
        }

    output = model(data_dict)
    candidate_box = build_candidate_box(model, output, reference_box)
    return {
        "candidate_box": candidate_box,
        "reference_box": reference_box,
        "data": data_dict,
        "output": output,
        "empty_fallback": False,
        "forward_ran": True,
    }


def run_active_branch(model, sequence, frame_index, results_bbs):
    data_dict, reference_box = model.build_input_dict(sequence, frame_index, results_bbs)
    return run_forward(model, data_dict, reference_box)


def run_passive_path(model, sequence, results_bbs, frame_index, offsets):
    indices = synthetic_path_indices(frame_index, offsets)
    if indices is None:
        return None
    synthetic_sequence = [sequence[index] for index in indices]
    synthetic_results = [copy.deepcopy(results_bbs[index]) for index in indices[:-1]]
    data_dict, reference_box = model.build_input_dict(
        synthetic_sequence,
        len(synthetic_sequence) - 1,
        synthetic_results,
    )
    branch = run_forward(model, data_dict, reference_box)
    branch["absolute_frame_indices"] = indices
    return branch


def run_paired_passive_paths(
    model,
    sequence,
    results_bbs,
    frame_index,
    offsets_a,
    offsets_b,
    seed,
    tracklet_id,
    fail_on_mismatch,
):
    if frame_index < max(max(offsets_a), max(offsets_b)):
        return None, None, None

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    endpoint_seed = stable_endpoint_seed(seed, tracklet_id, frame_index)

    def seed_branch():
        random.seed(endpoint_seed)
        np.random.seed(endpoint_seed)
        torch.manual_seed(endpoint_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(endpoint_seed)

    try:
        seed_branch()
        branch_a = run_passive_path(
            model, sequence, results_bbs, frame_index, offsets_a
        )
        seed_branch()
        branch_b = run_passive_path(
            model, sequence, results_bbs, frame_index, offsets_b
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    if branch_a is None or branch_b is None:
        return branch_a, branch_b, None

    anchor_gap = float(
        np.max(
            np.abs(
                np.concatenate(
                    [
                        box_center(branch_a["reference_box"]),
                        [box_yaw(branch_a["reference_box"], model.config.degrees)],
                    ]
                )
                - np.concatenate(
                    [
                        box_center(branch_b["reference_box"]),
                        [box_yaw(branch_b["reference_box"], model.config.degrees)],
                    ]
                )
            )
        )
    )
    point_sample_size = int(model.config.point_sample_size)
    current_a = branch_a["data"]["points"][:, -point_sample_size:, :3]
    current_b = branch_b["data"]["points"][:, -point_sample_size:, :3]
    current_point_gap = float(torch.max(torch.abs(current_a - current_b)).item())
    if fail_on_mismatch and (anchor_gap > 1e-6 or current_point_gap > 1e-6):
        raise RuntimeError(
            "Passive history paths do not share the endpoint geometry: "
            f"anchor_gap={anchor_gap:.6g}, current_point_gap={current_point_gap:.6g}"
        )
    checks = {
        "anchor_gap_max": anchor_gap,
        "current_point_gap_max": current_point_gap,
        "endpoint_seed": endpoint_seed,
    }
    return branch_a, branch_b, checks


def output_fields(output):
    fields = {
        "coarse_motion_x": None,
        "coarse_motion_y": None,
        "coarse_motion_z": None,
        "coarse_motion_yaw": None,
        "observation_motion_x": None,
        "observation_motion_y": None,
        "observation_motion_z": None,
        "observation_motion_yaw": None,
        "dynamics_pred_x": None,
        "dynamics_pred_y": None,
        "dynamics_pred_z": None,
        "dynamics_clamped_x": None,
        "dynamics_clamped_y": None,
        "dynamics_clamped_z": None,
        "dynamics_residual_x": None,
        "dynamics_residual_y": None,
        "dynamics_residual_z": None,
        "dynamics_innovation_raw_x": None,
        "dynamics_innovation_raw_y": None,
        "dynamics_innovation_raw_z": None,
        "dynamics_innovation_clamped_x": None,
        "dynamics_innovation_clamped_y": None,
        "dynamics_innovation_clamped_z": None,
        "dynamics_innovation_applied_x": None,
        "dynamics_innovation_applied_y": None,
        "dynamics_innovation_applied_z": None,
        "velocity_pred_x": None,
        "velocity_pred_y": None,
        "velocity_pred_z": None,
        "dynamics_valid": None,
        "dynamics_residual_alpha": None,
        "dynamics_residual_scale_effective": None,
        "dynamics_residual_raw_norm": None,
        "dynamics_residual_clamped_norm": None,
        "dynamics_residual_clamp_mask": None,
        "dynamics_residual_applied": None,
        "dynamics_innovation_raw_norm": None,
        "dynamics_innovation_clamped_norm": None,
        "dynamics_innovation_applied_norm": None,
        "dynamics_innovation_radius": None,
        "dynamics_innovation_alpha": None,
        "dynamics_innovation_scale_effective": None,
        "dynamics_innovation_clamp_mask": None,
        "dynamics_innovation_applied_mask": None,
        "dynamics_innovation_invalid_fallback": None,
        "dynamics_innovation_valid": None,
        "physical_time_adapter_norm": None,
        "physical_time_adapter_scale_effective": None,
        "obs_num_points_search": None,
        "obs_soft_fg_count": None,
        "obs_mean_fg_score": None,
        "obs_estimated_fg_points": None,
        "obs_valid_history_ratio": None,
        "obs_current_delta_t_ratio": None,
        "obs_dyn_center_gap": None,
    }
    if output is None:
        return fields

    motion = tensor_vector(output, "motion_pred", 4)
    observation_motion = tensor_vector(output, "motion_obs_pred", 4)
    dynamics = tensor_vector(output, "dynamics_displacement_pred", 3)
    dynamics_clamped = tensor_vector(output, "dynamics_displacement_clamped", 3)
    residual_key = (
        "motion_dynamics_residual"
        if output.get("motion_dynamics_residual") is not None
        else "motion_dyn_residual"
    )
    dynamics_residual = tensor_vector(output, residual_key, 3)
    innovation_raw = tensor_vector(output, "dynamics_innovation_raw", 3)
    innovation_clamped = tensor_vector(
        output, "dynamics_innovation_clamped", 3)
    innovation_applied = tensor_vector(
        output, "dynamics_innovation_applied", 3)
    velocity = tensor_vector(output, "velocity_pred", 3)
    fields.update(
        {
            "coarse_motion_x": motion[0],
            "coarse_motion_y": motion[1],
            "coarse_motion_z": motion[2],
            "coarse_motion_yaw": motion[3],
            "observation_motion_x": observation_motion[0],
            "observation_motion_y": observation_motion[1],
            "observation_motion_z": observation_motion[2],
            "observation_motion_yaw": observation_motion[3],
            "dynamics_pred_x": dynamics[0],
            "dynamics_pred_y": dynamics[1],
            "dynamics_pred_z": dynamics[2],
            "dynamics_clamped_x": dynamics_clamped[0],
            "dynamics_clamped_y": dynamics_clamped[1],
            "dynamics_clamped_z": dynamics_clamped[2],
            "dynamics_residual_x": dynamics_residual[0],
            "dynamics_residual_y": dynamics_residual[1],
            "dynamics_residual_z": dynamics_residual[2],
            "dynamics_innovation_raw_x": innovation_raw[0],
            "dynamics_innovation_raw_y": innovation_raw[1],
            "dynamics_innovation_raw_z": innovation_raw[2],
            "dynamics_innovation_clamped_x": innovation_clamped[0],
            "dynamics_innovation_clamped_y": innovation_clamped[1],
            "dynamics_innovation_clamped_z": innovation_clamped[2],
            "dynamics_innovation_applied_x": innovation_applied[0],
            "dynamics_innovation_applied_y": innovation_applied[1],
            "dynamics_innovation_applied_z": innovation_applied[2],
            "velocity_pred_x": velocity[0],
            "velocity_pred_y": velocity[1],
            "velocity_pred_z": velocity[2],
            "dynamics_valid": tensor_scalar(output, "dynamics_valid"),
            "dynamics_residual_alpha": tensor_scalar(
                output, "dynamics_residual_alpha"
            ),
            "dynamics_residual_scale_effective": tensor_scalar(
                output, "dynamics_residual_scale_effective"
            ),
            "dynamics_residual_raw_norm": tensor_scalar(
                output, "dynamics_residual_raw_norm"
            ),
            "dynamics_residual_clamped_norm": tensor_scalar(
                output, "dynamics_residual_clamped_norm"
            ),
            "dynamics_residual_clamp_mask": tensor_scalar(
                output, "dynamics_residual_clamp_mask"
            ),
            "dynamics_residual_applied": tensor_scalar(
                output, "dynamics_residual_applied_mask"
            ),
            "dynamics_innovation_raw_norm": tensor_scalar(
                output, "dynamics_innovation_raw_norm"
            ),
            "dynamics_innovation_clamped_norm": tensor_scalar(
                output, "dynamics_innovation_clamped_norm"
            ),
            "dynamics_innovation_applied_norm": tensor_scalar(
                output, "dynamics_innovation_applied_norm"
            ),
            "dynamics_innovation_radius": tensor_scalar(
                output, "dynamics_innovation_radius"
            ),
            "dynamics_innovation_alpha": tensor_scalar(
                output, "dynamics_innovation_alpha"
            ),
            "dynamics_innovation_scale_effective": tensor_scalar(
                output, "dynamics_innovation_scale_effective"
            ),
            "dynamics_innovation_clamp_mask": tensor_scalar(
                output, "dynamics_innovation_clamp_mask"
            ),
            "dynamics_innovation_applied_mask": tensor_scalar(
                output, "dynamics_innovation_applied_mask"
            ),
            "dynamics_innovation_invalid_fallback": tensor_scalar(
                output, "dynamics_innovation_invalid_fallback"
            ),
            "dynamics_innovation_valid": tensor_scalar(
                output, "dynamics_innovation_valid"
            ),
            "physical_time_adapter_norm": tensor_scalar(
                output, "physical_time_adapter_norm"
            ),
            "physical_time_adapter_scale_effective": tensor_scalar(
                output, "physical_time_adapter_scale"
            ),
            "obs_num_points_search": tensor_scalar(
                output, "obs_num_points_search"
            ),
            "obs_soft_fg_count": tensor_scalar(output, "obs_soft_fg_count"),
            "obs_mean_fg_score": tensor_scalar(output, "obs_mean_fg_score"),
            "obs_estimated_fg_points": tensor_scalar(
                output, "obs_estimated_fg_points"
            ),
            "obs_valid_history_ratio": tensor_scalar(
                output, "obs_valid_history_ratio"
            ),
            "obs_current_delta_t_ratio": tensor_scalar(
                output, "obs_current_delta_t_ratio"
            ),
            "obs_dyn_center_gap": tensor_scalar(output, "obs_dyn_center_gap"),
        }
    )
    return fields


def active_crop_fields(active_crop):
    """Normalize the existing crop diagnostic contract for the M0 CSV."""
    return {
        "active_crop_target_point_count": active_crop["target_point_count"],
        "active_crop_retained_target_points": active_crop[
            "retained_target_point_count"
        ],
        "active_crop_target_point_recall": finite_float(
            active_crop["target_point_recall"]
        ),
        "active_crop_has_target_point": active_crop["has_target_point"],
        "active_crop_center_inside": active_crop["center_inside"],
    }


def passive_path_fields(branch_a, branch_b, checks, degrees=False):
    fields = {
        "path_variance_available": False,
        "path_a_empty_fallback": None,
        "path_b_empty_fallback": None,
        "path_a_pred_x": None,
        "path_a_pred_y": None,
        "path_a_pred_z": None,
        "path_a_pred_yaw": None,
        "path_b_pred_x": None,
        "path_b_pred_y": None,
        "path_b_pred_z": None,
        "path_b_pred_yaw": None,
        "path_center_gap": None,
        "path_yaw_gap": None,
        "path_anchor_gap_max": None,
        "path_current_point_gap_max": None,
        "path_endpoint_seed": None,
    }
    if branch_a is None or branch_b is None:
        return fields

    box_a = branch_a["candidate_box"]
    box_b = branch_b["candidate_box"]
    fields.update(
        {
            "path_variance_available": True,
            "path_a_empty_fallback": bool(branch_a["empty_fallback"]),
            "path_b_empty_fallback": bool(branch_b["empty_fallback"]),
            "path_center_gap": float(np.linalg.norm(box_center(box_a) - box_center(box_b))),
            "path_yaw_gap": wrapped_angle_difference(
                box_yaw(box_a, degrees), box_yaw(box_b, degrees), degrees=degrees
            ),
            "path_anchor_gap_max": checks["anchor_gap_max"],
            "path_current_point_gap_max": checks["current_point_gap_max"],
            "path_endpoint_seed": checks["endpoint_seed"],
        }
    )
    fields.update(box_fields("path_a_pred", box_a, degrees))
    fields.update(box_fields("path_b_pred", box_b, degrees))
    return fields


def source_tracklet_id(dataset, tracklet_id):
    metadata = getattr(dataset, "virtual_rate_meta", None)
    if metadata and tracklet_id < len(metadata):
        return int(metadata[tracklet_id].get("source_tracklet", tracklet_id))
    return int(tracklet_id)


def source_frame_index(dataset, tracklet_id, frame_index):
    metadata = getattr(dataset, "virtual_rate_meta", None)
    if metadata and tracklet_id < len(metadata):
        keep_indices = metadata[tracklet_id].get("keep_indices", [])
        if frame_index < len(keep_indices):
            return int(keep_indices[frame_index])
    return int(frame_index)


def stable_tracklet_key(dataset, sequence, tracklet_id, version, split):
    get_tracklet_key = getattr(dataset, "get_tracklet_key", None)
    if callable(get_tracklet_key):
        return str(get_tracklet_key(tracklet_id))
    meta = sequence[0].get("meta", {}) if sequence else {}
    box_anno = meta.get("box_anno", meta) if isinstance(meta, dict) else {}
    if isinstance(box_anno, dict) and box_anno.get("instance_token"):
        return str(box_anno["instance_token"])
    return (
        f"{version}:{split}:source_tracklet:"
        f"{source_tracklet_id(dataset, tracklet_id)}"
    )


def tracking_scores(ious, distances):
    overlaps = np.asarray(ious, dtype=np.float64)
    accuracy = np.asarray(distances, dtype=np.float64)
    success_thresholds = np.linspace(0.0, 1.0, 21)
    precision_thresholds = np.linspace(0.0, 2.0, 21)
    success_curve = np.asarray(
        [np.mean(overlaps >= threshold) for threshold in success_thresholds]
    )
    precision_curve = np.asarray(
        [np.mean(accuracy <= threshold) for threshold in precision_thresholds]
    )
    return {
        "success": float(np.trapz(success_curve, x=success_thresholds) * 100.0),
        "precision": float(np.trapz(precision_curve, x=precision_thresholds) * 50.0),
        "mean_iou": float(np.mean(overlaps)),
        "mean_center_error": float(np.mean(accuracy)),
    }


def load_config(path):
    rec_diag.load_runtime_dependencies()
    return rec_diag.load_config(path)


def merge_protocol_config(cfg, path):
    if path is None:
        return
    with open(path, "r", encoding="utf-8") as config_file:
        protocol_cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    for key, value in protocol_cfg.items():
        if "virtual_rate" in key:
            setattr(cfg, key, value)


def apply_overrides(cfg, args):
    merge_protocol_config(cfg, args.protocol_cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    cfg.preloading = bool(args.preloading)

    if args.virtual_rate_mode is not None:
        cfg.test_virtual_rate_mode = args.virtual_rate_mode
    if args.virtual_rate_gap_pattern is not None:
        cfg.test_virtual_rate_gap_pattern = args.virtual_rate_gap_pattern
    if args.virtual_rate_max_gap is not None:
        cfg.test_virtual_rate_max_gap = args.virtual_rate_max_gap
    if args.virtual_rate_manifest is not None:
        cfg.virtual_rate_manifest_test = args.virtual_rate_manifest
        cfg.test_virtual_rate_manifest_allow_create = False
        cfg.test_virtual_rate_manifest_strict = True
    if args.manifest_commit_match is not None:
        cfg.test_virtual_rate_manifest_require_commit_match = bool(
            args.manifest_commit_match
        )

    if args.dynamics_time_mode is not None:
        cfg.dynamics_time_mode = args.dynamics_time_mode
        cfg.test_dynamics_time_mode = args.dynamics_time_mode
    if args.dynamics_fixed_delta_t is not None:
        cfg.dynamics_fixed_delta_t = args.dynamics_fixed_delta_t
        cfg.test_dynamics_fixed_delta_t = args.dynamics_fixed_delta_t
    if args.dynamics_time_manifest is not None:
        cfg.dynamics_time_manifest = args.dynamics_time_manifest
        cfg.dynamics_time_manifest_test = args.dynamics_time_manifest
        cfg.dynamics_time_manifest_strict = True
        cfg.test_dynamics_time_manifest_strict = True
    if args.manifest_commit_match is not None:
        cfg.test_dynamics_time_manifest_require_commit_match = bool(
            args.manifest_commit_match
        )


def load_model(cfg, weights_path, device):
    model_class = get_model(cfg.net_model)
    model = model_class.load_from_checkpoint(
        str(weights_path),
        config=cfg,
        map_location="cpu",
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def dataset_metadata(dataset):
    return {
        "virtual_rate_summary": getattr(dataset, "virtual_rate_summary", None),
        "virtual_rate_selection_sha256": getattr(
            dataset, "virtual_rate_selection_sha256", None
        ),
        "virtual_rate_manifest_content_sha256": getattr(
            dataset, "virtual_rate_manifest_content_sha256", None
        ),
        "virtual_rate_manifest_file_sha256": getattr(
            dataset, "virtual_rate_manifest_file_sha256", None
        ),
        "dynamics_time_summary": getattr(dataset, "dynamics_time_summary", None),
        "dynamics_time_manifest_content_sha256": getattr(
            dataset, "dynamics_time_manifest_content_sha256", None
        ),
        "dynamics_time_manifest_file_sha256": getattr(
            dataset, "dynamics_time_manifest_file_sha256", None
        ),
        "dynamics_time_permutation_sha256": getattr(
            dataset, "dynamics_time_permutation_sha256", None
        ),
    }


def initial_row(
    args,
    dataset,
    sequence,
    tracklet_id,
    tracklet_key,
    version,
    split,
    cfg_sha256,
    resolved_cfg_sha256,
    weights_sha256,
    initial_iou,
    initial_distance,
):
    gt_box = sequence[0]["3d_bbox"]
    token = crop_diag.frame_token(sequence[0], f"{tracklet_id}:0")
    row = {
        "run_label": args.run_label,
        "protocol": args.protocol_name,
        "time_mode": args.resolved_time_mode,
        "tracklet_id": int(tracklet_id),
        "source_tracklet_id": source_tracklet_id(dataset, tracklet_id),
        "tracklet_key": tracklet_key,
        "frame_index": 0,
        "source_frame_index": source_frame_index(dataset, tracklet_id, 0),
        "frame_token": token,
        "is_initial_frame": True,
        "full_history": False,
        "current_timestamp": crop_diag.frame_timestamp(sequence[0], 0),
        "current_delta_t_real": None,
        "current_delta_t_effective": None,
        "delta_t_real": "",
        "delta_t_effective": "",
        "history_frame_indices": "",
        "history_source_frame_indices": "",
        "valid_history_ratio": None,
        "num_points_in_search": None,
        "empty_fallback": False,
        "forward_ran": False,
        "previous_prediction_error": 0.0,
        "gt_displacement_from_previous_gt": 0.0,
        # Keep the persisted endpoint identical to the values used by the
        # in-process aggregate.  Some box backends return an overlap a few
        # ulps below 1.0 even when a box is compared with itself; hard-coding
        # 1.0 here makes the CSV-derived Success disagree at threshold 1.0.
        "iou": float(initial_iou),
        "center_error": float(initial_distance),
        "checkpoint_sha256": weights_sha256,
        "config_sha256": cfg_sha256,
        "resolved_config_sha256": resolved_cfg_sha256,
        "version": version,
        "split": split,
    }
    row.update(box_fields("prediction", gt_box, args.degrees))
    row.update(box_fields("ground_truth", gt_box, args.degrees))
    row.update(box_fields("reference", gt_box, args.degrees))
    row.update(box_fields("gt_local_disp", None, args.degrees))
    row.update(output_fields(None))
    row.update(passive_path_fields(None, None, None, args.degrees))
    row.update(
        {
            "active_crop_target_point_count": None,
            "active_crop_retained_target_points": None,
            "active_crop_target_point_recall": None,
            "active_crop_has_target_point": None,
            "active_crop_center_inside": None,
        }
    )
    return row


def self_test():
    if parse_offsets("1,3,5", 3) != [1, 3, 5]:
        raise RuntimeError("offset parsing self-test failed")
    if synthetic_path_indices(8, [1, 3, 5]) != [3, 5, 7, 8]:
        raise RuntimeError("synthetic path index self-test failed")
    if synthetic_path_indices(3, [1, 3, 5]) is not None:
        raise RuntimeError("unavailable path self-test failed")
    seed_a = stable_endpoint_seed(42, 3, 9)
    seed_b = stable_endpoint_seed(42, 3, 9)
    seed_c = stable_endpoint_seed(42, 3, 10)
    if seed_a != seed_b or seed_a == seed_c:
        raise RuntimeError("stable endpoint seed self-test failed")
    scores = tracking_scores([1.0, 0.0], [0.0, 2.0])
    if not np.isclose(scores["success"], 51.25) or not np.isclose(
        scores["precision"], 51.25
    ):
        raise RuntimeError(f"tracking score self-test failed: {scores}")
    crop = active_crop_fields(
        {
            "target_point_count": 3,
            "retained_target_point_count": 2,
            "target_point_recall": 2.0 / 3.0,
            "has_target_point": True,
            "center_inside": False,
        }
    )
    if crop["active_crop_retained_target_points"] != 2:
        raise RuntimeError(f"active crop field self-test failed: {crop}")
    m2_fields = output_fields(None)
    required_m2_fields = {
        "dynamics_innovation_raw_norm",
        "dynamics_innovation_applied_norm",
        "dynamics_innovation_radius",
        "dynamics_innovation_alpha",
        "dynamics_innovation_scale_effective",
        "dynamics_innovation_clamp_mask",
        "dynamics_innovation_applied_mask",
        "dynamics_innovation_valid",
        "physical_time_adapter_norm",
        "physical_time_adapter_scale_effective",
    }
    missing = sorted(required_m2_fields - set(m2_fields))
    if missing:
        raise RuntimeError(f"M2 endpoint fields missing: {missing}")
    print("M0 endpoint exporter self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export frozen recursive endpoint predictions and optional passive "
            "history-path variance without changing the active prediction path."
        )
    )
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--protocol-cfg", default=None)
    parser.add_argument("--run-label", default="model")
    parser.add_argument("--protocol-name", default="standard")
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--max-endpoints", type=int, default=None)
    parser.add_argument("--require-full-history", action="store_true")
    parser.add_argument("--omit-initial-frame", action="store_true")
    parser.add_argument("--reference-endpoints-csv", default=None)
    parser.add_argument("--allow-partial-reference", action="store_true")
    parser.add_argument("--virtual-rate-mode", default=None)
    parser.add_argument("--virtual-rate-gap-pattern", nargs="*", type=int, default=None)
    parser.add_argument("--virtual-rate-max-gap", type=int, default=None)
    parser.add_argument("--virtual-rate-manifest", default=None)
    parser.add_argument(
        "--dynamics-time-mode", choices=("true", "fixed", "shuffled"), default=None
    )
    parser.add_argument("--dynamics-fixed-delta-t", type=float, default=None)
    parser.add_argument("--dynamics-time-manifest", default=None)
    commit_group = parser.add_mutually_exclusive_group()
    commit_group.add_argument(
        "--require-manifest-commit-match",
        dest="manifest_commit_match",
        action="store_true",
    )
    commit_group.add_argument(
        "--allow-manifest-commit-mismatch",
        dest="manifest_commit_match",
        action="store_false",
    )
    parser.set_defaults(manifest_commit_match=None)
    parser.add_argument("--passive-path-variance", action="store_true")
    parser.add_argument("--path-a-offsets", default="1,2,3")
    parser.add_argument("--path-b-offsets", default="1,3,5")
    parser.add_argument("--fail-on-path-mismatch", action="store_true")
    parser.add_argument("--target-wlh-factor", type=float, default=1.0)
    parser.add_argument("--output-dir", default="output/diagnostics/m0_endpoints")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--model-load-smoke", action="store_true")
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Abort before loading the model when the repository is not clean.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    load_runtime_dependencies()
    if args.cfg is None or args.weights is None:
        parser.error("--cfg and --weights are required unless --self-test is used")

    initial_git_state = git_state()
    if args.require_clean_git and initial_git_state["dirty"]:
        raise RuntimeError(
            "--require-clean-git was requested, but the repository has changes: "
            f"{initial_git_state['status_porcelain']}"
        )

    cfg_path = Path(args.cfg).resolve()
    weights_path = Path(args.weights).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    cfg = load_config(cfg_path)
    apply_overrides(cfg, args)
    resolved_cfg = dict(cfg)
    resolved_cfg_sha256 = canonical_sha256(resolved_cfg)
    split = args.split if args.split is not None else cfg.test_split
    version = str(getattr(cfg, "version", "unknown"))
    args.degrees = bool(cfg.degrees)
    args.resolved_time_mode = str(
        args.dynamics_time_mode or getattr(cfg, "dynamics_time_mode", "true")
    )
    hist_num = int(cfg.hist_num)
    offsets_a = parse_offsets(args.path_a_offsets, hist_num)
    offsets_b = parse_offsets(args.path_b_offsets, hist_num)

    device = rec_diag.resolve_device(args.device)
    model = load_model(cfg, weights_path, device)
    weights_sha256 = sha256_file(weights_path)
    cfg_sha256 = sha256_file(cfg_path)
    if args.model_load_smoke:
        print(
            "M0 endpoint model-load smoke: PASS "
            f"device={device} parameters={sum(p.numel() for p in model.parameters())} "
            f"weights_sha256={weights_sha256}"
        )
        return

    sampler = get_dataset(cfg, type="test", split=split, protocol_role="test")
    dataset = getattr(sampler, "dataset", sampler)
    reference_path = (
        Path(args.reference_endpoints_csv).resolve()
        if args.reference_endpoints_csv is not None
        else None
    )
    reference_keys = (
        rec_diag.load_reference_endpoints(reference_path) if reference_path else None
    )

    tracklet_limit = dataset.get_num_tracklets()
    if args.max_tracklets is not None:
        tracklet_limit = min(tracklet_limit, args.max_tracklets)

    rows = []
    observed_keys = []
    active_ious = []
    active_distances = []
    active_empty_count = 0
    path_center_gaps = []
    path_yaw_gaps = []
    path_anchor_gaps = []
    path_current_point_gaps = []
    endpoint_count = 0
    stop = False
    start_time = time.time()

    with torch.no_grad():
        for tracklet_id in range(tracklet_limit):
            tracklet_length = dataset.get_num_frames_tracklet(tracklet_id)
            if tracklet_length <= 0:
                continue
            sequence = [
                dataset.get_frames(tracklet_id, [frame_index])[0]
                for frame_index in range(tracklet_length)
            ]
            tracklet_key = stable_tracklet_key(
                dataset, sequence, tracklet_id, version, split
            )
            results_bbs = [copy.deepcopy(sequence[0]["3d_bbox"])]

            initial_iou = float(
                estimate_overlap(
                    sequence[0]["3d_bbox"],
                    results_bbs[0],
                    dim=cfg.IoU_space,
                    up_axis=cfg.up_axis,
                )
            )
            initial_distance = float(
                estimate_accuracy(
                    sequence[0]["3d_bbox"],
                    results_bbs[0],
                    dim=cfg.IoU_space,
                    up_axis=cfg.up_axis,
                )
            )
            active_ious.append(initial_iou)
            active_distances.append(initial_distance)
            if not args.omit_initial_frame and not args.require_full_history:
                rows.append(
                    initial_row(
                        args,
                        dataset,
                        sequence,
                        tracklet_id,
                        tracklet_key,
                        version,
                        split,
                        cfg_sha256,
                        resolved_cfg_sha256,
                        weights_sha256,
                        initial_iou,
                        initial_distance,
                    )
                )
                endpoint_count += 1

            for frame_index in range(1, tracklet_length):
                previous_gt = sequence[frame_index - 1]["3d_bbox"]
                current_gt = sequence[frame_index]["3d_bbox"]
                previous_prediction = results_bbs[-1]
                branch = run_active_branch(model, sequence, frame_index, results_bbs)
                prediction = branch["candidate_box"]
                active_empty_count += int(branch["empty_fallback"])

                if args.passive_path_variance:
                    branch_a, branch_b, path_checks = run_paired_passive_paths(
                        model,
                        sequence,
                        results_bbs,
                        frame_index,
                        offsets_a,
                        offsets_b,
                        args.seed,
                        tracklet_id,
                        args.fail_on_path_mismatch,
                    )
                else:
                    branch_a, branch_b, path_checks = None, None, None

                results_bbs.append(prediction)
                iou = float(
                    estimate_overlap(
                        current_gt,
                        prediction,
                        dim=cfg.IoU_space,
                        up_axis=cfg.up_axis,
                    )
                )
                distance = float(
                    estimate_accuracy(
                        current_gt,
                        prediction,
                        dim=cfg.IoU_space,
                        up_axis=cfg.up_axis,
                    )
                )
                active_ious.append(iou)
                active_distances.append(distance)

                full_history = bool(frame_index >= hist_num)
                if args.require_full_history and not full_history:
                    continue

                data_dict = branch["data"]
                reference_box = branch["reference_box"]
                local_gt = points_utils.transform_box(current_gt, reference_box)
                active_crop = crop_diag.evaluate_crop(
                    sequence[frame_index]["pc"],
                    current_gt,
                    reference_box,
                    float(cfg.bb_scale),
                    float(cfg.bb_offset),
                    args.target_wlh_factor,
                )
                history_indices = [
                    max(0, frame_index - offset)
                    for offset in range(1, hist_num + 1)
                ]
                history_source_indices = [
                    source_frame_index(dataset, tracklet_id, index)
                    for index in history_indices
                ]
                token = crop_diag.frame_token(
                    sequence[frame_index], f"{tracklet_id}:{frame_index}"
                )
                row = {
                    "run_label": args.run_label,
                    "protocol": args.protocol_name,
                    "time_mode": args.resolved_time_mode,
                    "tracklet_id": int(tracklet_id),
                    "source_tracklet_id": source_tracklet_id(dataset, tracklet_id),
                    "tracklet_key": tracklet_key,
                    "frame_index": int(frame_index),
                    "source_frame_index": source_frame_index(
                        dataset, tracklet_id, frame_index
                    ),
                    "frame_token": token,
                    "is_initial_frame": False,
                    "full_history": full_history,
                    "current_timestamp": tensor_scalar(
                        data_dict, "current_timestamp"
                    ),
                    "current_delta_t_real": tensor_scalar(
                        data_dict, "current_delta_t_real"
                    ),
                    "current_delta_t_effective": tensor_scalar(
                        data_dict, "current_delta_t_effective"
                    ),
                    "delta_t_real": tensor_json(data_dict, "delta_t_real"),
                    "delta_t_effective": tensor_json(
                        data_dict, "delta_t_effective"
                    ),
                    "history_frame_indices": json.dumps(
                        history_indices, separators=(",", ":")
                    ),
                    "history_source_frame_indices": json.dumps(
                        history_source_indices, separators=(",", ":")
                    ),
                    "valid_history_ratio": float(
                        data_dict["valid_mask"].detach().float().mean().cpu().item()
                    ),
                    "num_points_in_search": tensor_scalar(
                        data_dict, "num_points_in_search"
                    ),
                    "empty_fallback": bool(branch["empty_fallback"]),
                    "forward_ran": bool(branch["forward_ran"]),
                    "previous_prediction_error": rec_diag.center_error(
                        previous_prediction, previous_gt
                    ),
                    "gt_displacement_from_previous_gt": rec_diag.center_error(
                        previous_gt, current_gt
                    ),
                    "iou": iou,
                    "center_error": distance,
                    "checkpoint_sha256": weights_sha256,
                    "config_sha256": cfg_sha256,
                    "resolved_config_sha256": resolved_cfg_sha256,
                    "version": version,
                    "split": split,
                }
                row.update(active_crop_fields(active_crop))
                row.update(box_fields("prediction", prediction, cfg.degrees))
                row.update(box_fields("ground_truth", current_gt, cfg.degrees))
                row.update(box_fields("reference", reference_box, cfg.degrees))
                row.update(box_fields("gt_local_disp", local_gt, cfg.degrees))
                row.update(output_fields(branch["output"]))
                path_fields = passive_path_fields(
                    branch_a, branch_b, path_checks, cfg.degrees
                )
                row.update(path_fields)
                if path_fields["path_variance_available"]:
                    path_center_gaps.append(path_fields["path_center_gap"])
                    path_yaw_gaps.append(path_fields["path_yaw_gap"])
                    path_anchor_gaps.append(path_fields["path_anchor_gap_max"])
                    path_current_point_gaps.append(
                        path_fields["path_current_point_gap_max"]
                    )

                rows.append(row)
                observed_keys.append(
                    rec_diag.endpoint_key(tracklet_id, frame_index, token)
                )
                endpoint_count += 1
                if args.max_endpoints is not None and endpoint_count >= args.max_endpoints:
                    stop = True
                    break
            if stop:
                break

    if not rows:
        raise RuntimeError("No M0 endpoint rows were produced")

    reference_report = rec_diag.validate_reference(
        observed_keys,
        reference_keys,
        tracklet_limit,
        args.allow_partial_reference,
    )
    summary = {
        "schema": "ct_seqtrack.m0_endpoint_export",
        "schema_version": 1,
        "run_label": args.run_label,
        "protocol": args.protocol_name,
        "time_mode": args.resolved_time_mode,
        "cfg": str(cfg_path),
        "cfg_sha256": cfg_sha256,
        "resolved_cfg_sha256": resolved_cfg_sha256,
        "protocol_cfg": str(Path(args.protocol_cfg).resolve())
        if args.protocol_cfg
        else None,
        "protocol_cfg_sha256": sha256_file(Path(args.protocol_cfg).resolve())
        if args.protocol_cfg
        else None,
        "weights": str(weights_path),
        "weights_sha256": weights_sha256,
        "split": split,
        "version": version,
        "seed": args.seed,
        "device": str(device),
        "tracklet_limit": tracklet_limit,
        "logged_row_count": len(rows),
        "active_frame_count": len(active_ious),
        "active_empty_fallback_count": active_empty_count,
        "active_metrics": tracking_scores(active_ious, active_distances),
        "passive_path_variance": {
            "enabled": bool(args.passive_path_variance),
            "path_a_offsets": offsets_a,
            "path_b_offsets": offsets_b,
            "eligible_count": len(path_center_gaps),
            "center_gap": crop_diag.finite_summary(path_center_gaps),
            "yaw_gap": crop_diag.finite_summary(path_yaw_gaps),
            "anchor_gap_max": crop_diag.finite_summary(path_anchor_gaps)["max"],
            "current_point_gap_max": crop_diag.finite_summary(
                path_current_point_gaps
            )["max"],
        },
        "require_full_history": bool(args.require_full_history),
        "omit_initial_frame": bool(args.omit_initial_frame),
        "reference_endpoints_csv": str(reference_path) if reference_path else None,
        "reference_endpoints_sha256": sha256_file(reference_path)
        if reference_path
        else None,
        "reference_match": reference_report,
        "dataset": dataset_metadata(dataset),
        "git": initial_git_state,
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "runtime_seconds": float(time.time() - start_time),
        "note": (
            "Only the active canonical prediction updates recursive history. Passive path A/B "
            "forwards, when enabled, are evaluation-only and cannot alter later endpoints."
        ),
    }

    tag = args.tag or f"{args.run_label}_{args.protocol_name}_{summary['time_mode']}"
    output_dir = Path(args.output_dir) / crop_diag.safe_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "m0_endpoints.csv"
    summary_path = output_dir / "m0_summary.json"
    resolved_cfg_path = output_dir / "resolved_config.json"
    crop_diag.write_rows(csv_path, rows)
    with resolved_cfg_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            resolved_cfg,
            config_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        )
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"endpoint csv: {csv_path}")
    print(f"summary json: {summary_path}")
    print(f"resolved config: {resolved_cfg_path}")


if __name__ == "__main__":
    main()
