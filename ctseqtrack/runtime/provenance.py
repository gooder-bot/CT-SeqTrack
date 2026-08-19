import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path):
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root):
    root = Path(root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
        ).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root),
            text=True,
        ).splitlines()
        full_status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root), text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, tracked_status, full_status = "unknown", ["unknown"], ["unknown"]
    return {
        "commit": commit,
        "dirty_tracked": bool(tracked_status),
        "dirty_any": bool(full_status),
        "status_porcelain": full_status,
    }


def dataset_provenance(wrapped):
    dataset = getattr(wrapped, "dataset", wrapped)
    partition_manifest = getattr(wrapped, "partition_manifest", None)
    return {
        "dataset": dataset.__class__.__name__,
        "version": getattr(dataset, "version", None),
        "split": getattr(dataset, "split", None),
        "protocol_role": getattr(dataset, "protocol_role", None),
        "kitti_hv_intervals": getattr(dataset, "kitti_hv_intervals", None),
        "tracklets": dataset.get_num_tracklets(),
        "frames": dataset.get_num_frames_total(),
        "virtual_rate_summary": getattr(dataset, "virtual_rate_summary", {}),
        "virtual_rate_selection_sha256": getattr(
            dataset, "virtual_rate_selection_sha256", None
        ),
        "virtual_rate_manifest_content_sha256": getattr(
            dataset, "virtual_rate_manifest_content_sha256", None
        ),
        "virtual_rate_manifest_file_sha256": getattr(
            dataset, "virtual_rate_manifest_file_sha256", None
        ),
        "dynamics_time_summary": getattr(dataset, "dynamics_time_summary", {}),
        "partition_scheme": getattr(
            wrapped,
            "partition_scheme",
            getattr(getattr(wrapped, "config", None), "ct_partition_scheme", None),
        ),
        "partition_manifest_sha256": (
            partition_manifest.get("content_sha256")
            if isinstance(partition_manifest, dict)
            else None
        ),
        "partition_manifest_partitions": (
            partition_manifest.get("partitions", {})
            if isinstance(partition_manifest, dict)
            else {}
        ),
    }


def write_run_provenance(output_dir, cfg, datasets, mode, root):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(str(getattr(cfg, "cfg", "")))
    checkpoint = getattr(cfg, "checkpoint", None)
    resolved_config = dict(cfg)
    resolved_config_json = json.dumps(
        resolved_config, sort_keys=True, separators=(",", ":"), default=str
    )
    dataset_records = {}
    for name, wrapped in datasets.items():
        record = dataset_provenance(wrapped)
        partition_manifest = getattr(wrapped, "partition_manifest", None)
        if isinstance(partition_manifest, dict):
            manifest_path = output_dir / f"{name}_scene_partition_manifest.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    partition_manifest, handle, indent=2, sort_keys=True, default=str
                )
                handle.write("\n")
            record["partition_manifest_path"] = str(manifest_path)
            record["partition_manifest_file_sha256"] = sha256_file(manifest_path)
        dataset_records[name] = record
    payload = {
        "schema": "ct_seqtrack.run_provenance",
        "schema_version": 1,
        "mode": mode,
        "git": git_state(root),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "resolved_config_sha256": hashlib.sha256(
            resolved_config_json.encode("utf-8")
        ).hexdigest(),
        "resolved_config": resolved_config,
        "seed": getattr(cfg, "seed", None),
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_rule": (
            "train: final/last is primary; precision/test top-k is diagnostic only"
            if mode == "train"
            else "test: use the explicitly supplied checkpoint; no threshold retuning"
        ),
        "datasets": dataset_records,
    }
    path = output_dir / "run_provenance.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(f"run provenance: {path}")
    return path
