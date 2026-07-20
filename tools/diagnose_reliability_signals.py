import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


crop_diag = None
points_utils = None
rec_diag = None
torch = None


OBS_SIGNAL_KEYS = (
    "obs_search_point_count",
    "obs_empty_fallback",
    "obs_forward_ran",
    "obs_soft_fg_count",
    "obs_estimated_fg_points",
    "obs_mean_fg_score",
    "obs_fg_hard_ratio",
    "obs_estimated_hard_fg_points",
    "obs_fg_probability_p50",
    "obs_fg_probability_p90",
    "obs_fg_probability_p95",
    "obs_fg_probability_max",
    "obs_fg_entropy_mean",
    "obs_fg_margin_mean",
    "obs_motion_dynamic_probability",
)


def load_runtime_dependencies():
    global crop_diag, points_utils, rec_diag, torch
    if rec_diag is not None:
        return

    from datasets import points_utils as points_utils_module
    from tools import diagnose_recursive_crop_reachability as rec_diag_module

    rec_diag_module.load_runtime_dependencies()
    crop_diag = rec_diag_module.crop_diag
    points_utils = points_utils_module
    rec_diag = rec_diag_module
    torch = rec_diag_module.torch


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def box_center(box):
    return np.asarray(box.center, dtype=np.float64)


def tensor_scalar(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor for {key}, got shape={tuple(value.shape)}")
        value = value.detach().reshape(-1)[0].cpu().item()
    return finite_float(value)


def box_yaw(box, degrees=False):
    if degrees:
        return float(box.orientation.degrees * box.orientation.axis[-1])
    return float(box.orientation.radians * box.orientation.axis[-1])


def box_fields(prefix, box, degrees=False):
    if box is None:
        return {
            f"{prefix}_center_x": None,
            f"{prefix}_center_y": None,
            f"{prefix}_center_z": None,
            f"{prefix}_yaw": None,
        }
    center = box_center(box)
    return {
        f"{prefix}_center_x": float(center[0]),
        f"{prefix}_center_y": float(center[1]),
        f"{prefix}_center_z": float(center[2]),
        f"{prefix}_yaw": box_yaw(box, degrees=degrees),
    }


def wrapped_angle_difference(left, right, degrees=False):
    period = 360.0 if degrees else 2.0 * math.pi
    half = period / 2.0
    difference = (float(left) - float(right) + half) % period - half
    return abs(float(difference))


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


def extract_forward_signals(data_dict, output):
    valid_mask = data_dict["valid_mask"]
    history_length = int(valid_mask.shape[1])
    point_count = int(data_dict["points"].shape[1])
    chunk_size = point_count // (history_length + 1)
    if chunk_size <= 0:
        raise RuntimeError(
            f"Invalid current-frame chunk size: points={point_count}, history={history_length}"
        )

    current_logits = output["seg_logits"][:, :, -chunk_size:]
    foreground_probability = torch.softmax(current_logits, dim=1)[:, 1, :]
    probability = foreground_probability.detach().reshape(-1).cpu().numpy().astype(np.float64)
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    entropy = -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))
    margin = np.abs(2.0 * probability - 1.0)

    search_point_count = tensor_scalar(data_dict, "num_points_in_search")
    hard_ratio = float(np.mean(probability >= 0.5))
    signals = {
        "search_point_count": search_point_count,
        "soft_fg_count": tensor_scalar(output, "obs_soft_fg_count"),
        "estimated_fg_points": tensor_scalar(output, "obs_estimated_fg_points"),
        "mean_fg_score": tensor_scalar(output, "obs_mean_fg_score"),
        "fg_hard_ratio": hard_ratio,
        "estimated_hard_fg_points": (
            hard_ratio * search_point_count if search_point_count is not None else None
        ),
        "fg_probability_p50": float(np.quantile(probability, 0.50)),
        "fg_probability_p90": float(np.quantile(probability, 0.90)),
        "fg_probability_p95": float(np.quantile(probability, 0.95)),
        "fg_probability_max": float(np.max(probability)),
        "fg_entropy_mean": float(np.mean(entropy)),
        "fg_margin_mean": float(np.mean(margin)),
        "motion_dynamic_probability": None,
    }
    if "motion_cls" in output:
        motion_probability = torch.softmax(output["motion_cls"], dim=1)[:, 1]
        signals["motion_dynamic_probability"] = float(
            motion_probability.detach().reshape(-1)[0].cpu().item()
        )
    return signals


def empty_branch_signals(search_point_count):
    return {
        "search_point_count": finite_float(search_point_count),
        "soft_fg_count": None,
        "estimated_fg_points": None,
        "mean_fg_score": None,
        "fg_hard_ratio": None,
        "estimated_hard_fg_points": None,
        "fg_probability_p50": None,
        "fg_probability_p90": None,
        "fg_probability_p95": None,
        "fg_probability_max": None,
        "fg_entropy_mean": None,
        "fg_margin_mean": None,
        "motion_dynamic_probability": None,
    }


def run_branch(model, sequence, frame_index, results_bbs):
    data_dict, reference_box = model.build_input_dict(sequence, frame_index, results_bbs)
    search_point_count = int(data_dict["num_points_in_search"].item())
    empty_fallback = bool(torch.sum(data_dict["points"][:, :, :3]).item() == 0.0)
    if empty_fallback:
        return {
            "candidate_box": copy.deepcopy(reference_box),
            "reference_box": reference_box,
            "empty_fallback": True,
            "forward_ran": False,
            "signals": empty_branch_signals(search_point_count),
        }

    output = model(data_dict)
    candidate_box = build_candidate_box(model, output, reference_box)
    return {
        "candidate_box": candidate_box,
        "reference_box": reference_box,
        "empty_fallback": False,
        "forward_ran": True,
        "signals": extract_forward_signals(data_dict, output),
    }


def branch_fields(prefix, branch):
    fields = {
        f"{prefix}_empty_fallback": bool(branch["empty_fallback"]),
        f"{prefix}_forward_ran": bool(branch["forward_ran"]),
    }
    fields.update(
        {f"{prefix}_{key}": value for key, value in branch["signals"].items()}
    )
    return fields


def crop_fields(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def predicted_cv_geometry(
    results_bbs,
    older_time,
    previous_time,
    current_time,
):
    previous_box = results_bbs[-1]
    fallback = copy.deepcopy(previous_box)
    if len(results_bbs) < 2:
        return fallback, False, None, None, None, None

    history_gap = float(previous_time) - float(older_time)
    query_gap = float(current_time) - float(previous_time)
    if (
        history_gap <= 0.0
        or query_gap <= 0.0
        or not np.isfinite(history_gap)
        or not np.isfinite(query_gap)
    ):
        return fallback, False, history_gap, query_gap, None, None

    older_box = results_bbs[-2]
    velocity = (box_center(previous_box) - box_center(older_box)) / history_gap
    speed = float(np.linalg.norm(velocity))
    shift = float(speed * query_gap)
    anchor = copy.deepcopy(previous_box)
    anchor.center = box_center(previous_box) + velocity * query_gap
    return anchor, True, history_gap, query_gap, speed, shift


def source_tracklet_id(dataset, tracklet_id):
    metadata = getattr(dataset, "virtual_rate_meta", None)
    if metadata and tracklet_id < len(metadata):
        return int(metadata[tracklet_id].get("source_tracklet", tracklet_id))
    return int(tracklet_id)


def tracklet_key(dataset, sequence, tracklet_id, version, split):
    meta = sequence[0].get("meta", {}) if sequence else {}
    box_anno = meta.get("box_anno", meta) if isinstance(meta, dict) else {}
    if isinstance(box_anno, dict):
        instance_token = box_anno.get("instance_token")
        if instance_token:
            return str(instance_token)
    source_id = source_tracklet_id(dataset, tracklet_id)
    return f"{version}:{split}:source_tracklet:{source_id}"


def source_frame_index(dataset, tracklet_id, frame_index):
    metadata = getattr(dataset, "virtual_rate_meta", None)
    if metadata and tracklet_id < len(metadata):
        keep_indices = metadata[tracklet_id].get("keep_indices", [])
        if frame_index < len(keep_indices):
            return int(keep_indices[frame_index])
    return int(frame_index)


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


def set_next_labels(previous_row, current_row):
    if previous_row is None:
        return
    previous_row["next_current_delta_t"] = current_row["current_delta_t"]
    previous_row["next_cv_speed"] = current_row["cv_speed"]
    previous_row["next_cv_shift"] = current_row["cv_shift"]
    previous_row["next_obs_crop_miss"] = current_row["current_obs_crop_miss"]
    previous_row["next_obs_center_outside"] = current_row["current_obs_center_outside"]
    previous_row["next_obs_drift"] = current_row["current_obs_drift"]
    previous_row["next_obs_empty_fallback"] = current_row["obs_empty_fallback"]


def copy_previous_observation_signals(row, previous_runtime_row):
    for key in OBS_SIGNAL_KEYS:
        row[f"prev_{key}"] = (
            previous_runtime_row.get(key) if previous_runtime_row is not None else None
        )


def summarize_rows(rows):
    visible = [row for row in rows if row["current_target_visible"]]
    next_labeled = [row for row in rows if row["next_obs_crop_miss"] is not None]
    selector_labeled = [row for row in rows if row["selector_label"] is not None]
    obs_any = [row["obs_has_target_point"] for row in visible]
    traj_any = [row["traj_has_target_point"] for row in visible]
    union_any = [left or right for left, right in zip(obs_any, traj_any)]
    obs_recall = [row["obs_target_point_recall"] for row in visible]
    traj_recall = [row["traj_target_point_recall"] for row in visible]
    union_recall = [max(left, right) for left, right in zip(obs_recall, traj_recall)]

    return {
        "endpoint_count": len(rows),
        "tracklet_count": len({row["tracklet_key"] for row in rows}),
        "visible_endpoint_count": len(visible),
        "current_obs_crop_miss_prevalence": (
            float(np.mean([row["current_obs_crop_miss"] for row in visible]))
            if visible
            else None
        ),
        "current_obs_drift_prevalence": float(np.mean([row["current_obs_drift"] for row in rows])),
        "next_obs_crop_miss_labeled_count": len(next_labeled),
        "next_obs_crop_miss_prevalence": (
            float(np.mean([row["next_obs_crop_miss"] for row in next_labeled]))
            if next_labeled
            else None
        ),
        "selector_labeled_count": len(selector_labeled),
        "trajectory_better_prevalence": (
            float(np.mean([row["selector_label"] for row in selector_labeled]))
            if selector_labeled
            else None
        ),
        "obs_empty_fallback_rate": float(np.mean([row["obs_empty_fallback"] for row in rows])),
        "traj_empty_fallback_rate": float(np.mean([row["traj_empty_fallback"] for row in rows])),
        "obs_candidate_error": crop_diag.finite_summary(
            [row["obs_candidate_error"] for row in rows]
        ),
        "traj_candidate_error": crop_diag.finite_summary(
            [row["traj_candidate_error"] for row in rows]
        ),
        "cv_speed": crop_diag.finite_summary([row["cv_speed"] for row in rows]),
        "cv_shift": crop_diag.finite_summary([row["cv_shift"] for row in rows]),
        "candidate_center_distance": crop_diag.finite_summary(
            [row["candidate_center_distance"] for row in rows]
        ),
        "passive_crop_complementarity": {
            "obs_has_target_point_rate": float(np.mean(obs_any)) if obs_any else None,
            "traj_has_target_point_rate": float(np.mean(traj_any)) if traj_any else None,
            "dual_union_has_target_point_rate": float(np.mean(union_any)) if union_any else None,
            "traj_only_endpoint_count": int(
                np.sum([right and not left for left, right in zip(obs_any, traj_any)])
            ),
            "obs_only_endpoint_count": int(
                np.sum([left and not right for left, right in zip(obs_any, traj_any)])
            ),
            "both_miss_endpoint_count": int(
                np.sum([not left and not right for left, right in zip(obs_any, traj_any)])
            ),
            "obs_target_point_recall_mean": float(np.mean(obs_recall)) if obs_recall else None,
            "traj_target_point_recall_mean": float(np.mean(traj_recall)) if traj_recall else None,
            "dual_oracle_target_point_recall_mean": (
                float(np.mean(union_recall)) if union_recall else None
            ),
        },
    }


def self_test():
    class DummyOrientation:
        radians = 0.0
        degrees = 0.0
        axis = np.array([0.0, 0.0, 1.0])

    class DummyBox:
        def __init__(self, center):
            self.center = np.asarray(center, dtype=np.float64)
            self.orientation = DummyOrientation()

    first = DummyBox([0.0, 0.0, 0.0])
    second = DummyBox([2.0, 0.0, 0.0])
    anchor, available, history_gap, query_gap, speed, shift = predicted_cv_geometry(
        [first, second], 0.0, 2.0, 5.0
    )
    if not available or not np.allclose(anchor.center, [5.0, 0.0, 0.0]):
        raise RuntimeError(f"CV geometry self-test failed: {anchor.center}")
    if (history_gap, query_gap, speed, shift) != (2.0, 3.0, 1.0, 3.0):
        raise RuntimeError("CV scalar self-test failed")

    previous = {"next_obs_crop_miss": None}
    current = {
        "current_delta_t": 1.5,
        "cv_speed": 4.0,
        "cv_shift": 6.0,
        "current_obs_crop_miss": True,
        "current_obs_center_outside": False,
        "current_obs_drift": True,
        "obs_empty_fallback": False,
    }
    set_next_labels(previous, current)
    if previous["next_obs_crop_miss"] is not True or previous["next_cv_shift"] != 6.0:
        raise RuntimeError("next-label alignment self-test failed")
    if wrapped_angle_difference(179.0, -179.0, degrees=True) != 2.0:
        raise RuntimeError("wrapped-angle self-test failed")
    print("reliability signal diagnostic self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the normal A1 recursive observation trajectory while passively forwarding a "
            "predicted-history real-dt CV anchor. Record only test-time signals as features and "
            "use GT exclusively for offline labels and diagnostic metrics."
        )
    )
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--reference-endpoints-csv", default=None)
    parser.add_argument("--allow-partial-reference", action="store_true")
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--max-endpoints", type=int, default=None)
    parser.add_argument("--require-full-history", action="store_true")
    parser.add_argument("--target-wlh-factor", type=float, default=1.0)
    parser.add_argument("--prediction-error-threshold", type=float, default=4.0)
    parser.add_argument("--selector-margin", type=float, default=0.25)
    parser.add_argument("--output-dir", default="output/diagnostics/reliability_signals")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--model-load-smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    load_runtime_dependencies()
    if args.cfg is None or args.weights is None:
        parser.error("--cfg and --weights are required unless --self-test is used.")

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

    cfg = rec_diag.load_config(cfg_path)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    cfg.preloading = bool(args.preloading)
    split = args.split if args.split is not None else cfg.train_split
    version = str(getattr(cfg, "version", "unknown"))

    device = rec_diag.resolve_device(args.device)
    model = rec_diag.load_a1_model(cfg, weights_path, device)
    if args.model_load_smoke:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            "reliability model-load smoke: PASS "
            f"device={device} parameters={parameter_count} "
            f"weights_sha256={rec_diag.sha256_file(weights_path)}"
        )
        return

    sampler = rec_diag.get_dataset(
        cfg, type="test", split=split, protocol_role="test")
    dataset = getattr(sampler, "dataset", sampler)
    reference_path = (
        Path(args.reference_endpoints_csv).resolve()
        if args.reference_endpoints_csv is not None
        else None
    )
    reference_keys = rec_diag.load_reference_endpoints(reference_path) if reference_path else None

    base_scale = float(cfg.bb_scale)
    base_offset = float(cfg.bb_offset)
    hist_num = int(cfg.hist_num)
    tracklet_limit = dataset.get_num_tracklets()
    if args.max_tracklets is not None:
        tracklet_limit = min(tracklet_limit, args.max_tracklets)

    rows = []
    observed_keys = []
    endpoint_count = 0
    stop = False
    start_time = time.time()

    with torch.no_grad():
        for tracklet_id in range(tracklet_limit):
            tracklet_length = dataset.get_num_frames_tracklet(tracklet_id)
            if tracklet_length < 2:
                continue
            sequence = [
                dataset.get_frames(tracklet_id, [frame_index])[0]
                for frame_index in range(tracklet_length)
            ]
            stable_tracklet_key = tracklet_key(
                dataset, sequence, tracklet_id, version, split
            )
            stable_source_tracklet = source_tracklet_id(dataset, tracklet_id)
            results_bbs = [copy.deepcopy(sequence[0]["3d_bbox"])]
            previous_runtime_row = None
            previous_logged_row = None

            for frame_index in range(1, tracklet_length):
                current_frame = sequence[frame_index]
                previous_frame = sequence[frame_index - 1]
                current_gt = current_frame["3d_bbox"]
                previous_gt = previous_frame["3d_bbox"]
                previous_pred = results_bbs[-1]

                previous_time = crop_diag.frame_timestamp(previous_frame, frame_index - 1)
                current_time = crop_diag.frame_timestamp(current_frame, frame_index)
                current_delta_t = float(current_time - previous_time)
                if frame_index >= 2:
                    older_frame = sequence[frame_index - 2]
                    older_time = crop_diag.frame_timestamp(older_frame, frame_index - 2)
                else:
                    older_time = previous_time

                (
                    trajectory_anchor,
                    pred_cv_available,
                    history_gap,
                    query_gap,
                    cv_speed,
                    cv_shift,
                ) = predicted_cv_geometry(
                    results_bbs,
                    older_time,
                    previous_time,
                    current_time,
                )

                observation_branch = run_branch(model, sequence, frame_index, results_bbs)
                observation_candidate = observation_branch["candidate_box"]

                if pred_cv_available:
                    trajectory_results_bbs = list(results_bbs)
                    trajectory_results_bbs[-1] = trajectory_anchor
                    trajectory_branch = run_branch(
                        model, sequence, frame_index, trajectory_results_bbs
                    )
                else:
                    trajectory_branch = {
                        "candidate_box": copy.deepcopy(trajectory_anchor),
                        "reference_box": trajectory_anchor,
                        "empty_fallback": True,
                        "forward_ran": False,
                        "signals": empty_branch_signals(None),
                    }
                trajectory_candidate = trajectory_branch["candidate_box"]

                observation_crop = crop_diag.evaluate_crop(
                    current_frame["pc"],
                    current_gt,
                    previous_pred,
                    base_scale,
                    base_offset,
                    args.target_wlh_factor,
                )
                trajectory_crop = crop_diag.evaluate_crop(
                    current_frame["pc"],
                    current_gt,
                    trajectory_anchor,
                    base_scale,
                    base_offset,
                    args.target_wlh_factor,
                )

                observation_error = rec_diag.center_error(observation_candidate, current_gt)
                trajectory_error = rec_diag.center_error(trajectory_candidate, current_gt)
                candidate_error_gain = observation_error - trajectory_error
                selector_label = None
                if abs(candidate_error_gain) >= args.selector_margin:
                    selector_label = bool(candidate_error_gain > 0.0)

                token = crop_diag.frame_token(
                    current_frame, f"{tracklet_id}:{frame_index}"
                )
                visible = observation_crop["target_point_count"] > 0
                row = {
                    "tracklet_id": int(tracklet_id),
                    "source_tracklet_id": stable_source_tracklet,
                    "tracklet_key": stable_tracklet_key,
                    "frame_index": int(frame_index),
                    "source_frame_index": source_frame_index(
                        dataset, tracklet_id, frame_index
                    ),
                    "frame_token": token,
                    "current_timestamp": float(current_time),
                    "current_delta_t": current_delta_t,
                    "history_gap": history_gap,
                    "query_gap": query_gap,
                    "full_history": bool(frame_index >= hist_num),
                    "pred_cv_available": bool(pred_cv_available),
                    "cv_speed": cv_speed,
                    "cv_shift": cv_shift,
                    "previous_prediction_error": rec_diag.center_error(
                        previous_pred, previous_gt
                    ),
                    "current_gt_displacement": rec_diag.center_error(previous_gt, current_gt),
                    "current_target_visible": bool(visible),
                    "current_obs_crop_miss": (
                        bool(not observation_crop["has_target_point"]) if visible else None
                    ),
                    "current_obs_center_outside": bool(
                        not observation_crop["center_inside"]
                    ),
                    "current_obs_drift": bool(
                        observation_error > args.prediction_error_threshold
                    ),
                    "current_traj_crop_miss": (
                        bool(not trajectory_crop["has_target_point"]) if visible else None
                    ),
                    "current_traj_drift": bool(
                        trajectory_error > args.prediction_error_threshold
                    ),
                    "dual_has_target_point": (
                        bool(
                            observation_crop["has_target_point"]
                            or trajectory_crop["has_target_point"]
                        )
                        if visible
                        else None
                    ),
                    "obs_candidate_error": observation_error,
                    "traj_candidate_error": trajectory_error,
                    "candidate_error_gain_traj_over_obs": candidate_error_gain,
                    "selector_label": selector_label,
                    "next_current_delta_t": None,
                    "next_cv_speed": None,
                    "next_cv_shift": None,
                    "next_obs_crop_miss": None,
                    "next_obs_center_outside": None,
                    "next_obs_drift": None,
                    "next_obs_empty_fallback": None,
                }
                row.update(branch_fields("obs", observation_branch))
                row.update(branch_fields("traj", trajectory_branch))
                row.update(crop_fields("obs", observation_crop))
                row.update(crop_fields("traj", trajectory_crop))
                row.update(box_fields("obs_anchor", previous_pred, cfg.degrees))
                row.update(box_fields("traj_anchor", trajectory_anchor, cfg.degrees))
                row.update(box_fields("obs_candidate", observation_candidate, cfg.degrees))
                row.update(box_fields("traj_candidate", trajectory_candidate, cfg.degrees))
                row["anchor_center_distance"] = rec_diag.center_error(
                    previous_pred, trajectory_anchor
                )
                row["anchor_yaw_difference"] = wrapped_angle_difference(
                    box_yaw(previous_pred, cfg.degrees),
                    box_yaw(trajectory_anchor, cfg.degrees),
                    degrees=cfg.degrees,
                )
                row["candidate_center_distance"] = rec_diag.center_error(
                    observation_candidate, trajectory_candidate
                )
                row["candidate_yaw_difference"] = wrapped_angle_difference(
                    box_yaw(observation_candidate, cfg.degrees),
                    box_yaw(trajectory_candidate, cfg.degrees),
                    degrees=cfg.degrees,
                )
                copy_previous_observation_signals(row, previous_runtime_row)
                set_next_labels(previous_logged_row, row)

                results_bbs.append(observation_candidate)
                previous_runtime_row = row

                if args.require_full_history and not row["full_history"]:
                    continue

                key = rec_diag.endpoint_key(tracklet_id, frame_index, token)
                observed_keys.append(key)
                rows.append(row)
                previous_logged_row = row
                endpoint_count += 1
                if args.max_endpoints is not None and endpoint_count >= args.max_endpoints:
                    stop = True
                    break
            if stop:
                break

    if not rows:
        raise RuntimeError("No reliability diagnostic endpoints were produced.")

    reference_report = rec_diag.validate_reference(
        observed_keys,
        reference_keys,
        tracklet_limit,
        args.allow_partial_reference,
    )
    summary = summarize_rows(rows)
    summary.update(
        {
            "cfg": str(cfg_path),
            "cfg_sha256": rec_diag.sha256_file(cfg_path),
            "weights": str(weights_path),
            "weights_sha256": rec_diag.sha256_file(weights_path),
            "reference_endpoints_csv": str(reference_path) if reference_path else None,
            "reference_endpoints_sha256": (
                rec_diag.sha256_file(reference_path) if reference_path else None
            ),
            "reference_match": reference_report,
            "split": split,
            "version": version,
            "virtual_rate_mode": str(getattr(cfg, "virtual_rate_mode", "none")),
            "base_scale": base_scale,
            "base_offset": base_offset,
            "target_wlh_factor": args.target_wlh_factor,
            "prediction_error_threshold": args.prediction_error_threshold,
            "selector_margin": args.selector_margin,
            "require_full_history": args.require_full_history,
            "seed": args.seed,
            "device": str(device),
            "git": git_state(),
            "runtime_seconds": float(time.time() - start_time),
            "feature_boundary": (
                "Fields prefixed previous_prediction_error/current_gt/current_target/"
                "obs_candidate_error/traj_candidate_error/selector_label and crop GT metrics are "
                "offline labels only. Trigger features must use prev_obs_* plus current delta_t/CV "
                "geometry. Current obs/traj foreground features are post-crop selector evidence."
            ),
            "note": (
                "The observation candidate alone updates the recursive history. The raw real-dt "
                "predicted-history CV branch is passive and cannot change later endpoints. This "
                "tool diagnoses reliability and candidate complementarity; it is not an active "
                "dual-anchor tracker and does not train any parameter."
            ),
        }
    )

    tag = args.tag or f"{Path(args.cfg).stem}_{split}"
    output_dir = Path(args.output_dir) / crop_diag.safe_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reliability_endpoints.csv"
    summary_path = output_dir / "reliability_summary.json"
    crop_diag.write_rows(csv_path, rows)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"endpoint csv: {csv_path}")
    print(f"summary json: {summary_path}")


if __name__ == "__main__":
    main()
