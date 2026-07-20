import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EasyDict = None
Quaternion = None
geometry_utils = None
get_dataset = None
points_utils = None


def load_runtime_dependencies():
    global EasyDict, Quaternion, geometry_utils, get_dataset, points_utils
    if EasyDict is not None:
        return
    from easydict import EasyDict as EasyDictClass
    from nuscenes.utils import geometry_utils as geometry_utils_module
    from pyquaternion import Quaternion as QuaternionClass

    from datasets import get_dataset as get_dataset_function
    from datasets import points_utils as points_utils_module

    EasyDict = EasyDictClass
    Quaternion = QuaternionClass
    geometry_utils = geometry_utils_module
    get_dataset = get_dataset_function
    points_utils = points_utils_module


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        cfg = EasyDict(yaml.load(config_file, Loader=yaml.FullLoader))
    if "preloading" not in cfg:
        cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def parse_float_list(value):
    if value is None or str(value).strip() == "":
        return []
    return sorted(float(item.strip()) for item in str(value).split(",") if item.strip())


def safe_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "diagnostic"


def finite_summary(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(quantiles[5]),
    }


def frame_timestamp(frame, fallback):
    value = frame.get("timestamp")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(value):
        return float(fallback)
    return value


def frame_token(frame, fallback):
    frame_id = frame.get("frame_id")
    if frame_id:
        return str(frame_id)
    meta = frame.get("meta", {})
    sample_data = meta.get("sample_data_lidar", {}) if isinstance(meta, dict) else {}
    return str(sample_data.get("token", fallback))


def bounds_membership(target_center, target_corners, search_min, search_max):
    target_center = np.asarray(target_center, dtype=np.float64)
    target_corners = np.asarray(target_corners, dtype=np.float64)
    search_min = np.asarray(search_min, dtype=np.float64)
    search_max = np.asarray(search_max, dtype=np.float64)
    center_inside = bool(
        np.all(target_center > search_min) and np.all(target_center < search_max)
    )
    corners_inside = np.logical_and(
        target_corners > search_min.reshape(3, 1),
        target_corners < search_max.reshape(3, 1),
    )
    all_corners_inside = bool(np.all(corners_inside))
    target_min = np.min(target_corners, axis=1)
    target_max = np.max(target_corners, axis=1)
    box_intersects = bool(
        np.all(target_max > search_min) and np.all(target_min < search_max)
    )
    return {
        "center_inside": center_inside,
        "all_corners_inside": all_corners_inside,
        "box_intersects": box_intersects,
    }


def local_search_geometry(anchor_box, target_box, scale, offset):
    rotation_to_anchor = np.asarray(anchor_box.rotation_matrix, dtype=np.float64).T
    target_center_local = rotation_to_anchor @ (
        np.asarray(target_box.center, dtype=np.float64)
        - np.asarray(anchor_box.center, dtype=np.float64)
    )
    target_corners_local = rotation_to_anchor @ (
        np.asarray(target_box.corners(), dtype=np.float64)
        - np.asarray(anchor_box.center, dtype=np.float64).reshape(3, 1)
    )

    local_anchor = copy.deepcopy(anchor_box)
    local_anchor.translate(-np.asarray(anchor_box.center, dtype=np.float64))
    local_anchor.rotate(Quaternion(matrix=rotation_to_anchor))
    search_box = copy.deepcopy(local_anchor)
    search_box.wlh = np.asarray(search_box.wlh, dtype=np.float64) * float(scale)
    search_corners = np.asarray(search_box.corners(), dtype=np.float64)
    search_min = np.min(search_corners, axis=1) - float(offset)
    search_max = np.max(search_corners, axis=1) + float(offset)

    return bounds_membership(
        target_center_local,
        target_corners_local,
        search_min,
        search_max,
    )


def search_crop_mask(point_cloud, anchor_box, scale, offset):
    local_pc = copy.deepcopy(point_cloud)
    local_anchor = copy.deepcopy(anchor_box)
    translation = -np.asarray(anchor_box.center, dtype=np.float64)
    rotation_to_anchor = np.asarray(anchor_box.rotation_matrix, dtype=np.float64).T
    local_pc.translate(translation)
    local_anchor.translate(translation)
    local_pc.rotate(rotation_to_anchor)
    local_anchor.rotate(Quaternion(matrix=rotation_to_anchor))
    _, mask = points_utils.crop_pc_axis_aligned(
        local_pc,
        local_anchor,
        scale=float(scale),
        offset=float(offset),
        return_mask=True,
    )
    return np.asarray(mask, dtype=bool)


def evaluate_crop(point_cloud, target_box, anchor_box, scale, offset, target_wlh_factor):
    crop_mask = search_crop_mask(point_cloud, anchor_box, scale, offset)
    target_mask = geometry_utils.points_in_box(
        target_box,
        point_cloud.points[:3, :],
        wlh_factor=float(target_wlh_factor),
    )
    target_mask = np.asarray(target_mask, dtype=bool)
    total_target_points = int(np.sum(target_mask))
    retained_target_points = int(np.sum(np.logical_and(target_mask, crop_mask)))
    target_point_recall = (
        float(retained_target_points / total_target_points)
        if total_target_points > 0
        else float("nan")
    )
    result = local_search_geometry(anchor_box, target_box, scale, offset)
    result.update(
        {
            "crop_point_count": int(np.sum(crop_mask)),
            "target_point_count": total_target_points,
            "retained_target_point_count": retained_target_points,
            "target_point_recall": target_point_recall,
            "has_target_point": bool(retained_target_points > 0),
            "scale": float(scale),
            "offset": float(offset),
        }
    )
    return result


def constant_velocity_anchor(previous_previous_frame, previous_frame, current_frame):
    previous_box = previous_frame["3d_bbox"]
    anchor = copy.deepcopy(previous_box)
    if previous_previous_frame is None:
        return anchor, False

    previous_previous_box = previous_previous_frame["3d_bbox"]
    previous_previous_time = frame_timestamp(previous_previous_frame, -1.0)
    previous_time = frame_timestamp(previous_frame, 0.0)
    current_time = frame_timestamp(current_frame, 1.0)
    history_gap = previous_time - previous_previous_time
    query_gap = current_time - previous_time
    if history_gap <= 0.0 or query_gap <= 0.0:
        return anchor, False

    velocity = (
        np.asarray(previous_box.center, dtype=np.float64)
        - np.asarray(previous_previous_box.center, dtype=np.float64)
    ) / history_gap
    anchor.center = np.asarray(previous_box.center, dtype=np.float64) + velocity * query_gap
    return anchor, True


def bucket_label(value, boundaries):
    if not np.isfinite(value):
        return "non_finite"
    lower = float("-inf")
    for boundary in boundaries:
        if value <= boundary:
            if np.isneginf(lower):
                return f"<= {boundary:g}"
            return f"({lower:g}, {boundary:g}]"
        lower = boundary
    if not boundaries:
        return "all"
    return f"> {boundaries[-1]:g}"


def summarize_mode_subset(subset):
    valid_target = [row for row in subset if row["target_point_count"] > 0]
    center_inside_rate = float(np.mean([row["center_inside"] for row in subset]))
    has_target_point_rate = (
        float(np.mean([row["has_target_point"] for row in valid_target]))
        if valid_target
        else None
    )
    return {
        "endpoint_count": len(subset),
        "center_inside_rate": center_inside_rate,
        "center_outside_rate": 1.0 - center_inside_rate,
        "box_intersection_rate": float(
            np.mean([row["box_intersects"] for row in subset])
        ),
        "all_corners_inside_rate": float(
            np.mean([row["all_corners_inside"] for row in subset])
        ),
        "target_visible_endpoint_count": len(valid_target),
        "has_target_point_rate": has_target_point_rate,
        "no_target_point_rate": (
            1.0 - has_target_point_rate
            if has_target_point_rate is not None
            else None
        ),
        "target_point_recall": finite_summary(
            [row["target_point_recall"] for row in valid_target]
        ),
        "crop_point_count": finite_summary([row["crop_point_count"] for row in subset]),
    }


def summarize_buckets(rows, field, boundaries):
    buckets = {}
    for row in rows:
        bucket = bucket_label(float(row[field]), boundaries)
        mode = row["crop_mode"]
        buckets.setdefault(bucket, {}).setdefault(mode, []).append(row)
    for bucket, by_mode in buckets.items():
        for mode, subset in list(by_mode.items()):
            by_mode[mode] = summarize_mode_subset(subset)
    return buckets


def summarize_rows(rows, delta_t_bins, displacement_bins, target_point_bins):
    summary = {
        "endpoint_count": len({(row["tracklet_id"], row["frame_index"]) for row in rows}),
        "current_delta_t": finite_summary(
            [row["current_delta_t"] for row in rows if row["crop_mode"] == "base"]
        ),
        "displacement_norm": finite_summary(
            [row["displacement_norm"] for row in rows if row["crop_mode"] == "base"]
        ),
        "modes": {},
        "delta_t_buckets": summarize_buckets(
            rows, "current_delta_t", delta_t_bins
        ),
        "displacement_buckets": summarize_buckets(
            rows, "displacement_norm", displacement_bins
        ),
        "target_point_buckets": summarize_buckets(
            rows, "target_point_count", target_point_bins
        ),
    }
    modes = sorted({row["crop_mode"] for row in rows})
    for mode in modes:
        subset = [row for row in rows if row["crop_mode"] == mode]
        summary["modes"][mode] = summarize_mode_subset(subset)
    return summary


def write_rows(path, rows):
    if not rows:
        raise RuntimeError("No endpoint rows were produced.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    near = bounds_membership(
        np.array([0.0, 0.0, 0.0]),
        np.array(
            [
                [-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5],
                [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    if not near["center_inside"] or not near["all_corners_inside"]:
        raise RuntimeError(f"near-target crop self-test failed: {near}")

    far = bounds_membership(
        np.array([8.0, 0.0, 0.0]),
        np.array(
            [
                [7.5, 7.5, 7.5, 7.5, 8.5, 8.5, 8.5, 8.5],
                [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
            ]
        ),
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    if far["center_inside"] or far["box_intersects"]:
        raise RuntimeError(f"far-target crop self-test failed: {far}")
    summary = finite_summary([0.0, 1.0, 2.0, float("nan")])
    if summary["count"] != 3 or summary["p50"] != 1.0:
        raise RuntimeError(f"summary self-test failed: {summary}")
    mock_rows = []
    for mode, center_inside, has_target_point in (
        ("base", False, False),
        ("expanded", True, True),
        ("cv_recenter", True, True),
    ):
        mock_rows.append(
            {
                "tracklet_id": 0,
                "frame_index": 3,
                "crop_mode": mode,
                "current_delta_t": 1.0,
                "displacement_norm": 2.0,
                "target_point_count": 5,
                "center_inside": center_inside,
                "box_intersects": center_inside,
                "all_corners_inside": center_inside,
                "has_target_point": has_target_point,
                "target_point_recall": float(has_target_point),
                "crop_point_count": 10,
            }
        )
    crop_summary = summarize_rows(mock_rows, [0.5, 1.0], [1.0, 2.0], [0, 5])
    if crop_summary["modes"]["base"]["center_outside_rate"] != 1.0:
        raise RuntimeError(f"crop summary self-test failed: {crop_summary}")
    if crop_summary["modes"]["expanded"]["target_point_recall"]["p50"] != 1.0:
        raise RuntimeError(f"crop recall summary self-test failed: {crop_summary}")
    print("crop reachability self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure model-independent search-crop reachability using the previous GT box, "
            "a larger crop, and a GT-history constant-velocity recenter anchor."
        )
    )
    parser.add_argument("--cfg")
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--max-endpoints", type=int, default=None)
    parser.add_argument(
        "--require-full-history",
        action="store_true",
        help="Only emit endpoints with at least cfg.hist_num preceding frames.",
    )
    parser.add_argument("--expanded-scale-multiplier", type=float, default=2.0)
    parser.add_argument("--expanded-offset-multiplier", type=float, default=2.0)
    parser.add_argument("--target-wlh-factor", type=float, default=1.0)
    parser.add_argument("--delta-t-bins", default="0.5,1.0,2.0")
    parser.add_argument("--displacement-bins", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--target-point-bins", default="0,5,20")
    parser.add_argument("--output-dir", default="output/diagnostics/crop_reachability")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.cfg is None:
        parser.error("--cfg is required unless --self-test is used.")
    if args.expanded_scale_multiplier < 1.0:
        raise ValueError("--expanded-scale-multiplier must be >= 1.0.")
    if args.expanded_offset_multiplier < 1.0:
        raise ValueError("--expanded-offset-multiplier must be >= 1.0.")

    load_runtime_dependencies()
    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    cfg.preloading = bool(args.preloading)
    split = args.split if args.split is not None else cfg.train_split
    sampler = get_dataset(cfg, type="test", split=split)
    dataset = getattr(sampler, "dataset", sampler)

    base_scale = float(cfg.bb_scale)
    base_offset = float(cfg.bb_offset)
    crop_modes = {
        "base": (base_scale, base_offset),
        "expanded": (
            base_scale * args.expanded_scale_multiplier,
            base_offset * args.expanded_offset_multiplier,
        ),
    }

    rows = []
    endpoint_count = 0
    tracklet_limit = dataset.get_num_tracklets()
    if args.max_tracklets is not None:
        tracklet_limit = min(tracklet_limit, args.max_tracklets)

    stop = False
    for tracklet_id in range(tracklet_limit):
        tracklet_length = dataset.get_num_frames_tracklet(tracklet_id)
        if tracklet_length < 2:
            continue
        previous_previous_frame = None
        previous_frame = dataset.get_frames(tracklet_id, [0])[0]
        for frame_index in range(1, tracklet_length):
            current_frame = dataset.get_frames(tracklet_id, [frame_index])[0]
            previous_time = frame_timestamp(previous_frame, frame_index - 1)
            current_time = frame_timestamp(current_frame, frame_index)
            current_delta_t = current_time - previous_time
            displacement_norm = float(
                np.linalg.norm(
                    np.asarray(current_frame["3d_bbox"].center, dtype=np.float64)
                    - np.asarray(previous_frame["3d_bbox"].center, dtype=np.float64)
                )
            )
            cv_anchor, cv_available = constant_velocity_anchor(
                previous_previous_frame, previous_frame, current_frame
            )
            full_history = frame_index >= int(cfg.hist_num)
            if args.require_full_history and not full_history:
                previous_previous_frame = previous_frame
                previous_frame = current_frame
                continue
            anchors = {
                "base": previous_frame["3d_bbox"],
                "expanded": previous_frame["3d_bbox"],
                "cv_recenter": cv_anchor,
            }
            mode_settings = dict(crop_modes)
            mode_settings["cv_recenter"] = (base_scale, base_offset)

            endpoint_metadata = {
                "tracklet_id": tracklet_id,
                "frame_index": frame_index,
                "frame_token": frame_token(current_frame, f"{tracklet_id}:{frame_index}"),
                "current_delta_t": float(current_delta_t),
                "displacement_norm": displacement_norm,
                "cv_available": bool(cv_available),
                "full_history": bool(full_history),
            }
            for mode, (scale, offset) in mode_settings.items():
                metrics = evaluate_crop(
                    current_frame["pc"],
                    current_frame["3d_bbox"],
                    anchors[mode],
                    scale,
                    offset,
                    args.target_wlh_factor,
                )
                row = dict(endpoint_metadata)
                row["crop_mode"] = mode
                row.update(metrics)
                rows.append(row)

            endpoint_count += 1
            previous_previous_frame = previous_frame
            previous_frame = current_frame
            if args.max_endpoints is not None and endpoint_count >= args.max_endpoints:
                stop = True
                break
        if stop:
            break

    delta_t_bins = parse_float_list(args.delta_t_bins)
    displacement_bins = parse_float_list(args.displacement_bins)
    target_point_bins = parse_float_list(args.target_point_bins)
    summary = summarize_rows(
        rows,
        delta_t_bins,
        displacement_bins,
        target_point_bins,
    )
    summary.update(
        {
            "cfg": str(Path(args.cfg).resolve()),
            "split": split,
            "virtual_rate_mode": str(getattr(cfg, "virtual_rate_mode", "none")),
            "base_scale": base_scale,
            "base_offset": base_offset,
            "expanded_scale_multiplier": args.expanded_scale_multiplier,
            "expanded_offset_multiplier": args.expanded_offset_multiplier,
            "target_wlh_factor": args.target_wlh_factor,
            "require_full_history": args.require_full_history,
            "note": (
                "base and expanded use the previous GT box; cv_recenter uses only GT history "
                "with a constant-velocity center extrapolation. This is an oracle reachability "
                "diagnostic, not an online tracking result."
            ),
        }
    )

    tag = args.tag or f"{Path(args.cfg).stem}_{split}"
    output_dir = Path(args.output_dir) / safe_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "crop_reachability_endpoints.csv"
    summary_path = output_dir / "crop_reachability_summary.json"
    write_rows(csv_path, rows)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"endpoint csv: {csv_path}")
    print(f"summary json: {summary_path}")


if __name__ == "__main__":
    main()
