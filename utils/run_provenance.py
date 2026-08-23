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
            cwd=str(root), text=True).splitlines()
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
    return {
        "dataset": dataset.__class__.__name__,
        "version": getattr(dataset, "version", None),
        "split": getattr(dataset, "split", None),
        "protocol_role": getattr(dataset, "protocol_role", None),
        "kitti_hv_intervals": getattr(
            dataset, "kitti_hv_intervals", None),
        "tracklets": dataset.get_num_tracklets(),
        "frames": dataset.get_num_frames_total(),
        "virtual_rate_summary": getattr(dataset, "virtual_rate_summary", {}),
        "virtual_rate_selection_sha256": getattr(
            dataset, "virtual_rate_selection_sha256", None),
        "virtual_rate_manifest_content_sha256": getattr(
            dataset, "virtual_rate_manifest_content_sha256", None),
        "virtual_rate_manifest_file_sha256": getattr(
            dataset, "virtual_rate_manifest_file_sha256", None),
        "dynamics_time_summary": getattr(dataset, "dynamics_time_summary", {}),
    }


def write_run_provenance(output_dir, cfg, datasets, mode, root):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(str(getattr(cfg, "cfg", "")))
    checkpoint = getattr(cfg, "checkpoint", None)
    init_checkpoint = getattr(cfg, "init_checkpoint", None)
    resolved_config = dict(cfg)
    resolved_config_json = json.dumps(
        resolved_config, sort_keys=True, separators=(",", ":"), default=str)
    payload = {
        "schema": "ct_seqtrack.run_provenance",
        "schema_version": 3,
        "mode": mode,
        "git": git_state(root),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "resolved_config_sha256": hashlib.sha256(
            resolved_config_json.encode("utf-8")).hexdigest(),
        "resolved_config": resolved_config,
        "seed": getattr(cfg, "seed", None),
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "init_checkpoint_path": init_checkpoint,
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "checkpoint_rule": (
            "train: final/last and late-3 are primary; mini_val is never "
            "used to select different best epochs across arms"
            if mode == "train" else
            "test: use the explicitly supplied frozen checkpoint; no threshold retuning"
        ),
        "safe_seqtrack": {
            "runtime_protocol": getattr(
                cfg, "ct_runtime_protocol", None),
            "optimizer_topology": getattr(
                cfg, "ct_optimizer_topology", None),
            "batch_schema": getattr(cfg, "ct_batch_schema", None),
            "observation_rng_mode": getattr(
                cfg, "ct_observation_rng_mode", None),
            "validation_rng_mode": getattr(
                cfg, "ct_validation_rng_mode", None),
            "candidate_policy": getattr(
                cfg, "ct_candidate_policy", None),
            "b0_candidate_weights": list(getattr(
                cfg, "ct_b0_candidate_weights", [])),
            "b2_candidate_views": getattr(
                cfg, "ct_b2_candidate_views", None),
            "mechanism_shadow_b0_no_grad": getattr(
                cfg, "ct_mechanism_shadow_b0_no_grad", None),
            "cuda_stage_audit": getattr(
                cfg, "ct_cuda_stage_audit", None),
            "observation_fingerprint_steps": getattr(
                cfg, "ct_observation_fingerprint_steps", None),
            "evaluator_identity": getattr(
                cfg, "ct_evaluator_identity", None),
            "seqtrack_reference": {
                "url": getattr(cfg, "ct_seqtrack_reference_url", None),
                "commit": getattr(
                    cfg, "ct_seqtrack_reference_commit", None),
            },
        },
        "datasets": {
            name: dataset_provenance(dataset) for name, dataset in datasets.items()
        },
        "training_streams": {
            "topology": getattr(cfg, "ct_training_topology", None),
            "observation_source": "train",
            "mechanism_source": (
                "mechanism" if "mechanism" in datasets else None),
            "observation_steps_per_epoch": getattr(
                cfg, "ct_observation_steps_per_epoch_observed", None),
            "mechanism_steps_per_epoch": getattr(
                cfg, "ct_mechanism_steps_per_epoch_observed", None),
            "b0_optimizer_steps_per_epoch": getattr(
                cfg, "ct_observation_steps_per_epoch_observed", None),
            "mechanism_passes_per_epoch": getattr(
                cfg, "ct_mechanism_passes_per_epoch", None),
            "mechanism_partition": getattr(
                cfg, "ct_router_partition", None),
            "mechanism_tracklets": getattr(
                cfg, "ct_mechanism_tracklets_observed", None),
            "mechanism_prediction_frames": getattr(
                cfg, "ct_mechanism_prediction_frames_observed", None),
            "mechanism_selection_sha256": getattr(
                cfg, "ct_mechanism_selection_sha256", None),
        },
    }
    path = output_dir / "run_provenance.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(f"run provenance: {path}")
    return path
