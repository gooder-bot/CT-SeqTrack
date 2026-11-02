"""KITTI 双臂准备：标准库只读标签/标定与点云文件大小，不导入模型。

它不下载数据、不创建训练任务、不加载点云内容，
仅把共同数据清单与迁移风险写入新的 artifacts/ct_checks 子目录。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid


SCHEMA = "ct_seqtrack.kitti_two_arm_input_inventory.v1"
ROLE_SCENES = {
    "train": [f"{i:04d}" for i in range(17)],
    "dev": ["0017", "0018"],
    "test": ["0019", "0020"],
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values),
            "mean": math.fsum(values) / len(values), "max": max(values)}


def determinant3(values):
    a, b, c, d, e, f, g, h, i = values
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def orthogonality_error3(values):
    return max(abs(sum(values[3 * row + a] * values[3 * row + b] for row in range(3))
                   - float(a == b)) for a in range(3) for b in range(3))


def inspect_calibration(path):
    if not path.is_file():
        return {"status": "missing"}
    entries = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        key, *values = line.split()
        entries[key.rstrip(":")] = [float(v) for v in values]
    transform = next((entries[k] for k in ("Tr_velo_cam", "Tr_velo_to_cam") if k in entries), None)
    rectify = next((entries[k] for k in ("R_rect", "R0_rect") if k in entries), None)
    valid = (transform is not None and len(transform) == 12 and rectify is not None
             and len(rectify) == 9 and all(math.isfinite(x) for x in transform + rectify))
    result = {"status": "structurally_valid" if valid else "invalid", "sha256": file_hash(path)}
    if valid:
        rotation = [transform[i] for i in (0, 1, 2, 4, 5, 6, 8, 9, 10)]
        result.update(rotation_determinant=determinant3(rotation),
                      rectification_determinant=determinant3(rectify),
                      rotation_orthogonality_error=orthogonality_error3(rotation),
                      rectification_orthogonality_error=orthogonality_error3(rectify))
        if any(abs(result[k] - 1.0) > 0.002 for k in
               ("rotation_determinant", "rectification_determinant")) or any(result[k] > 0.002 for k in
               ("rotation_orthogonality_error", "rectification_orthogonality_error")):
            result["status"] = "invalid_rotation_geometry"
    return result


def inspect_scene(data_root, scene, category):
    label = data_root / "label_02" / (scene + ".txt")
    calibration = inspect_calibration(data_root / "calib" / (scene + ".txt"))
    result = {"scene": scene, "calibration": calibration, "tracks": [],
              "velodyne_directory_exists": (data_root / "velodyne" / scene).is_dir()}
    if not label.is_file():
        return dict(result, status="missing_labels", label_sha256=None)
    by_track = defaultdict(list)
    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) not in (17, 18):
            raise ValueError(f"Malformed label {label}:{line_number}")
        try:
            frame, track, occluded = int(fields[0]), int(fields[1]), int(fields[4])
            numbers = [float(fields[3])] + [float(x) for x in fields[5:]]
            geometry = [float(x) for x in fields[10:17]]
        except ValueError as error:
            raise ValueError(f"Malformed numeric label {label}:{line_number}") from error
        if frame < 0 or not all(math.isfinite(x) for x in numbers):
            raise ValueError(f"Invalid frame/nonfinite label {label}:{line_number}")
        if fields[2].casefold() != category.casefold() or track < 0:
            continue
        if min(geometry[:3]) <= 0:
            raise ValueError(f"Nonpositive object dimensions {label}:{line_number}")
        by_track[track].append({"frame": frame, "hwl": geometry[:3]})
    unique_frames = sorted({row["frame"] for rows in by_track.values() for row in rows})
    files = []
    for frame in unique_frames:
        relative = f"velodyne/{scene}/{frame:06d}.bin"
        cloud = data_root / relative
        size = cloud.stat().st_size if cloud.is_file() else None
        status = "missing" if size is None else "empty" if size == 0 else "malformed" if size % 16 else "present"
        files.append({"path": relative, "status": status, "bytes": size})
    cloud_by_frame = dict(zip(unique_frames, files))
    gaps, h2, h4, size_drifts = [], [], [], []
    weighted_cloud_status = Counter()
    for track, rows in sorted(by_track.items()):
        rows.sort(key=lambda row: row["frame"])
        frames = [row["frame"] for row in rows]
        if len(frames) != len(set(frames)):
            raise ValueError(f"Duplicate frame in scene={scene}, track={track}")
        gaps.extend((b - a) / 10.0 for a, b in zip(frames, frames[1:]))
        drift = max((max(abs(v - base) for v, base in zip(row["hwl"], rows[0]["hwl"]))
                     for row in rows), default=0.0)
        size_drifts.append(drift)
        for frame in frames:
            weighted_cloud_status[cloud_by_frame[frame]["status"]] += 1
        for t in range(1, len(rows)):
            for horizon, values in ((2, h2), (4, h4)):
                future = t - 1 + horizon
                if future < len(rows):
                    values.append((frames[future] - frames[t - 1]) / 10.0)
        result["tracks"].append({"track_id": track, "frames": frames,
            "count": len(frames), "initial_hwl": rows[0]["hwl"],
            "max_size_change_m": drift,
            "singleton": len(frames) == 1})
    result.update(status="scanned", label_sha256=file_hash(label),
        pointcloud_files=files, unique_pointcloud_status=dict(Counter(row["status"] for row in files)),
        object_frame_pointcloud_status=dict(weighted_cloud_status),
        object_frames=sum(len(rows) for rows in by_track.values()),
        nonfirst_endpoints=sum(max(len(rows) - 1, 0) for rows in by_track.values()),
        nominal_original_seqtrack_rows=4 * sum(len(rows) for rows in by_track.values()),
        nominal_ct_rows=4 * sum(max(len(rows) - 1, 0) for rows in by_track.values()),
        frame_gaps_seconds=distribution(gaps), mtp_h2_dt_seconds=distribution(h2),
        mtp_h4_dt_seconds=distribution(h4), size_change_m=distribution(size_drifts),
        tracks_with_size_change_over_1mm=sum(x > 0.001 for x in size_drifts))
    return result


def inspect_dataset(data_root, category):
    source = Path(data_root).expanduser().resolve()
    if (source / "training" / "label_02").is_dir():
        source = source / "training"
    if not all((source / name).is_dir() for name in ("label_02", "calib", "velodyne")):
        raise ValueError("Data root must contain label_02, calib and velodyne, directly or under training/")
    scenes = [inspect_scene(source, f"{i:04d}", category) for i in range(21)]
    by_scene = {row["scene"]: row for row in scenes}
    roles = {}
    for role, names in ROLE_SCENES.items():
        selected = [by_scene[name] for name in names]
        roles[role] = {"scenes": names,
            "tracks": sum(len(row["tracks"]) for row in selected),
            "object_frames": sum(row.get("object_frames", 0) for row in selected),
            "nonfirst_endpoints": sum(row.get("nonfirst_endpoints", 0) for row in selected),
            "nominal_original_seqtrack_rows": sum(row.get("nominal_original_seqtrack_rows", 0) for row in selected),
            "nominal_ct_rows": sum(row.get("nominal_ct_rows", 0) for row in selected)}
    result = {"schema": SCHEMA, "data_root": str(source), "category": category,
        "proposed_roles": roles, "scenes": scenes,
        "semantics": {"frame_period_seconds": 0.1, "ct_frame_stride": 1,
            "initialization_and_singletons_counted": True,
            "missing_cloud_policy": "fail_closed_no_frames_removed",
            "empty_cloud_policy": "valid_zero_point_observation_warn_and_preserve_frame",
            "gt_preload_crop": False,
            "check_scope": "metadata_and_file_sizes_only_no_model_no_environment_probe",
            "not_checked": ["cloud contents", "box geometry alignment", "CUDA", "training", "model inputs"],
            "training_rows": "nominal_only_original_resampling_and_drop_last_change_actual_exposure"}}
    identity = {"schema": SCHEMA, "category": category, "roles": ROLE_SCENES,
        "scenes": [{"scene": row["scene"], "label_sha256": row.get("label_sha256"),
                    "calibration_sha256": row["calibration"].get("sha256"),
                    "files": row.get("pointcloud_files", [])} for row in scenes]}
    result["inventory_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    result["blocking_findings"] = [f"{row['scene']}: {row['status']}" for row in scenes if row["status"] != "scanned"]
    result["blocking_findings"] += [f"{row['scene']}: calibration {row['calibration']['status']}"
        for row in scenes if row["calibration"]["status"] != "structurally_valid"]
    result["blocking_findings"] += [f"velodyne/{row['scene']}: missing_directory"
        for row in scenes if not row["velodyne_directory_exists"]]
    result["blocking_findings"] += [item["path"] + ": malformed byte count" for row in scenes
        for item in row.get("pointcloud_files", []) if item["status"] == "malformed"]
    result["missing_files"] = [item["path"] for row in scenes
        for item in row.get("pointcloud_files", []) if item["status"] == "missing"]
    result["empty_files"] = [item["path"] for row in scenes
        for item in row.get("pointcloud_files", []) if item["status"] == "empty"]
    result["warnings"] = [path + ": zero_point_observation_frame_preserved" for path in result["empty_files"]]
    result["blocking_findings"] += [item["path"] + ": " + item["status"] for row in scenes
        for item in row.get("pointcloud_files", []) if item["status"] == "missing"]
    result["role_readiness"] = {role: {"has_objects": value["object_frames"] > 0,
        "has_predictable_endpoints": value["nonfirst_endpoints"] > 0,
        "ready": value["object_frames"] > 0 and value["nonfirst_endpoints"] > 0,
        "reason": "no_objects" if value["object_frames"] == 0 else
                  "no_predictable_endpoints" if value["nonfirst_endpoints"] == 0 else "metadata_coverage_present"}
        for role, value in roles.items()}
    result["input_readiness"] = {"ready": not result["blocking_findings"] and all(
        row["ready"] for row in result["role_readiness"].values()),
        "scope": "metadata_only_not_training_or_geometry_acceptance"}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--path", required=True, help="KITTI root or its training/ directory")
    parser.add_argument("--category", default="all", choices=("all", "Car", "Pedestrian", "Van", "Cyclist"))
    parser.add_argument("--output-dir", help="New directory under this repository's artifacts/ct_checks")
    args = parser.parse_args(argv)
    repo = next((path for path in Path(__file__).resolve().parents
                 if (path / "main.py").is_file() and (path / "cfgs").is_dir()), None)
    if repo is None:
        raise ValueError("Place this tool inside CT-SeqTrack before running it")
    allowed = (repo / "artifacts" / "ct_checks").resolve()
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    dest = Path(args.output_dir).resolve() if args.output_dir else allowed / ("kitti_inventory_" + suffix)
    if allowed not in dest.parents or dest.exists():
        raise ValueError("Output must be a new child directory under artifacts/ct_checks")
    categories = ("Car", "Pedestrian", "Van", "Cyclist") if args.category == "all" else (args.category,)
    reports = {category: inspect_dataset(args.path, category) for category in categories}
    report = {"schema": SCHEMA, "categories": reports,
        "blocking_findings": sorted({finding for value in reports.values() for finding in value["blocking_findings"]})}
    report["input_readiness"] = {"ready": all(value["input_readiness"]["ready"] for value in reports.values()),
        "scope": "metadata_only_not_training_or_geometry_acceptance"}
    report["inventory_sha256"] = hashlib.sha256(json.dumps(
        {category: value["inventory_sha256"] for category, value in reports.items()},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / "inventory.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"report": str(dest / "inventory.json"),
        "inventory_sha256": report["inventory_sha256"],
        "roles_by_category": {category: value["proposed_roles"] for category, value in reports.items()},
        "readiness_by_category": {category: value["role_readiness"] for category, value in reports.items()},
        "input_readiness": report["input_readiness"],
        "blocking_findings": report["blocking_findings"],
        "missing_files": len({path for value in reports.values() for path in value["missing_files"]}),
        "empty_files": len({path for value in reports.values() for path in value["empty_files"]})}, ensure_ascii=False))
    return 0 if report["input_readiness"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
