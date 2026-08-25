"""Versioned, leakage-resistant recursive replay cache contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch


REPLAY_SCHEMA_VERSION = 3
B0_STATE_PREFIXES = (
    "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
    "motion_state_mlp.", "feature_pointnet.", "Transformer.",
)
B1_STATE_PREFIXES = ("physical_motion_encoder.",)
REPLAY_CONFIG_FIELDS = (
    "dataset", "category_name", "hist_num", "num_candidates",
    "bb_scale", "bb_offset", "point_sample_size", "degrees",
    "observation_safe_bbox_size",
    "use_real_time", "default_time_step", "pseudo_time_step", "time_scale",
    "dynamics_time_mode", "dynamics_fixed_delta_t",
    "use_b1motion_v3", "motion_v3_hidden_dim", "motion_v3_step_dim",
    "motion_v3_temporal_backend", "motion_v3_cfc_backbone_units",
    "motion_v3_beta_nll_beta", "motion_v3_tail_direction_weight",
    "motion_v3_tail_direction_margin", "motion_v3_prior_weight",
    "motion_v3_aux_prior_weight", "motion_v3_nll_weight",
    "motion_v3_aux_nll_weight", "motion_v3_aux_query_gaps",
    "motion_v3_aux_transition_gaps",
    "motion_v3_min_delta_t", "motion_v3_max_delta_t",
    "motion_v3_max_speed", "motion_v3_max_displacement",
    "motion_v3_initial_sigma", "motion_v3_residual_velocity_scale",
    "motion_v3_eps", "ct_motion_max_acceleration",
    "ct_motion_max_displacement",
    "ct_motion_acceleration_weight", "ct_enable_shared_motion_anchor",
    "ct_enable_dynamic_residual_bound",
    "motion_v3_min_direction_speed", "motion_v3_log_sigma_min",
    "motion_v3_log_sigma_max", "use_calibrated_motion_uncertainty",
)
B1_CALIBRATION_CONFIG_FIELDS = (
    "dataset", "category_name", "hist_num", "degrees",
    "observation_safe_bbox_size", "use_real_time", "default_time_step",
    "time_scale",
    "pseudo_time_step", "dynamics_time_mode", "dynamics_fixed_delta_t",
    "use_b1motion_v3", "motion_v3_hidden_dim", "motion_v3_step_dim",
    "motion_v3_temporal_backend", "motion_v3_cfc_backbone_units",
    "motion_v3_beta_nll_beta", "motion_v3_tail_direction_weight",
    "motion_v3_tail_direction_margin", "motion_v3_prior_weight",
    "motion_v3_aux_prior_weight", "motion_v3_nll_weight",
    "motion_v3_aux_nll_weight", "motion_v3_aux_query_gaps",
    "motion_v3_aux_transition_gaps",
    "motion_v3_time_scale", "motion_v3_min_delta_t",
    "motion_v3_max_delta_t", "motion_v3_max_speed",
    "motion_v3_max_displacement", "motion_v3_min_direction_speed",
    "motion_v3_initial_sigma", "motion_v3_residual_velocity_scale",
    "motion_v3_eps", "ct_motion_max_acceleration",
    "ct_motion_max_displacement",
    "ct_motion_acceleration_weight", "ct_enable_shared_motion_anchor",
    "ct_enable_dynamic_residual_bound",
    "motion_v3_log_sigma_min", "motion_v3_log_sigma_max",
    "use_calibrated_motion_uncertainty",
)
B2_CANDIDATE_CONFIG_EXCLUDED_FIELDS = frozenset({
    # Experiment/runtime identity does not alter the candidate function for a
    # fixed input and checkpoint.
    "experiment_name", "path", "train_split", "val_split", "test_split",
    "version", "key_frame_only", "preloading", "preload_offset", "seed",
    "cfg", "tag", "log_dir", "checkpoint", "init_checkpoint", "test",
    "batch_size", "workers", "epoch", "save_top_k",
    "check_val_every_n_epoch", "limit_train_batches",
    "b1_calibration_artifact_path", "use_recursive_replay_cache",
    "recursive_replay_cache_dir", "recursive_replay_require_all",
    "force_b1_invalid", "shuffle_b1_signal",
    # B3 changes between config 17 and 18 by design and must not change the
    # identity of the already-promoted B0/B1/B2 candidate producer.
    "proposal_inference_mode", "use_action_consistent_router_v3",
    "b2_v3_require_packaged_router", "export_b3_rollouts",
    "require_b2_candidate_config_contract",
})
REQUIRED_MANIFEST_FIELDS = (
    "dataset",
    "split",
    "replay_config_sha256",
    "commit",
    "b0_state_sha256",
    "b1_state_sha256",
    "b1_calibration_sha256",
    "b0_checkpoint_sha256",
    "b1_checkpoint_sha256",
    "source_checkpoint_sha256",
)
FORBIDDEN_RECORD_KEYS = {
    "gt", "ground_truth", "target", "label", "current_gt",
    "current_ground_truth", "current_box_label",
}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_config_contract(config):
    """Return the stable subset that determines recursive B0/B1 records."""
    getter = config.get if isinstance(config, dict) else lambda key, default=None: getattr(
        config, key, default)
    return {key: getter(key, None) for key in REPLAY_CONFIG_FIELDS}


def replay_config_sha256(config):
    return sha256_json(replay_config_contract(config))


def b1_calibration_config_contract(config):
    """Return every resolved field that changes B1 calibration meaning."""
    getter = config.get if isinstance(config, dict) else lambda key, default=None: getattr(
        config, key, default)
    return {
        key: getter(key, None) for key in B1_CALIBRATION_CONFIG_FIELDS}


def b1_calibration_config_sha256(config):
    return sha256_json(b1_calibration_config_contract(config))


def b2_candidate_config_contract(config):
    """Resolved model config that determines B0/B1/B2 candidate behavior."""
    values = dict(config) if isinstance(config, dict) else dict(vars(config))
    return {
        key: value for key, value in values.items()
        if key not in B2_CANDIDATE_CONFIG_EXCLUDED_FIELDS
        and not str(key).startswith("router_v3_")
    }


def b2_candidate_config_sha256(config):
    return sha256_json(b2_candidate_config_contract(config))


def tensor_prefixes_sha256(state_dict, prefixes):
    """Content hash tensor names, shapes, dtypes and bytes for prefixes."""
    prefixes = tuple(prefixes)
    keys = sorted(
        key for key in state_dict if key.startswith(prefixes))
    if not keys:
        raise ValueError(
            "state hash matched no tensors for prefixes " + repr(prefixes))
    digest = hashlib.sha256()
    for key in keys:
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_b1_calibration_state(calibration, state_dict):
    """Bind calibration metadata to the checkpoint scale buffer."""
    if not isinstance(calibration, dict):
        return False
    if calibration.get("schema") != "ct_seqtrack.b1_uncertainty_calibration.v3":
        raise RuntimeError(
            "B1 calibration artifact uses an incompatible schema")
    expected = calibration.get("log_scale_parallel_perpendicular")
    keys = [
        key for key in state_dict
        if key.endswith(
            "physical_motion_encoder.log_sigma_calibration")]
    if len(keys) != 1 or not isinstance(expected, (list, tuple)):
        raise RuntimeError(
            "B1 calibration metadata/buffer contract is incomplete")
    actual = state_dict[keys[0]].detach()
    expected_tensor = torch.as_tensor(
        expected, dtype=actual.dtype, device=actual.device)
    if (expected_tensor.shape != actual.shape
            or not torch.equal(expected_tensor, actual)):
        raise RuntimeError(
            "B1 calibration metadata does not match checkpoint buffer")
    return True


def _validate_no_current_gt(value, path="record"):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_RECORD_KEYS:
                raise ValueError(
                    f"recursive replay contains forbidden GT field: {path}.{key}")
            _validate_no_current_gt(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_no_current_gt(item, f"{path}[{index}]")


def validate_replay_record(record):
    if not isinstance(record, dict):
        raise TypeError("recursive replay record must be a mapping")
    _validate_no_current_gt(record)
    required = (
        "tracklet_key", "frame_id", "history_boxes_world",
        "history_valid_mask", "delta_t", "current_delta_t", "anchor_world",
        "b1", "source",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            "recursive replay record is missing: " + ", ".join(missing))
    history = record["history_boxes_world"]
    if not isinstance(history, list) or not history:
        raise ValueError("history_boxes_world must be a non-empty list")
    if any(not isinstance(row, list) or len(row) != 7 for row in history):
        raise ValueError(
            "history_boxes_world rows must be [x,y,z,w,l,h,yaw]")
    if len(record["history_valid_mask"]) != len(history):
        raise ValueError("history_valid_mask length does not match history")
    if len(record["delta_t"]) != len(history):
        raise ValueError("delta_t length does not match history")
    b1_required = (
        "mu_xy", "log_sigma_parallel_perp", "covariance_xy",
        "basis_velocity_xy", "direction_xy", "velocity_xy", "feature",
        "valid", "gap_ratio", "source_id",
    )
    missing_b1 = [key for key in b1_required if key not in record["b1"]]
    if missing_b1:
        raise ValueError(
            "recursive replay B1 output is missing: " + ", ".join(missing_b1))


def replay_key(tracklet_key, frame_id):
    return f"{str(tracklet_key)}::{int(frame_id)}"


def write_recursive_replay_cache(directory, manifest, records):
    """Write deterministic JSONL plus a content-addressed manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    missing = [key for key in REQUIRED_MANIFEST_FIELDS if key not in manifest]
    if missing:
        raise ValueError("replay manifest is missing: " + ", ".join(missing))
    normalized_records = []
    seen = set()
    for record in records:
        validate_replay_record(record)
        key = replay_key(record["tracklet_key"], record["frame_id"])
        if key in seen:
            raise ValueError(f"duplicate recursive replay key: {key}")
        seen.add(key)
        normalized_records.append(copy.deepcopy(record))
    normalized_records.sort(
        key=lambda row: replay_key(row["tracklet_key"], row["frame_id"]))
    records_path = directory / "records.jsonl"
    lines = [json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in normalized_records]
    records_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest.update({
        "schema_version": REPLAY_SCHEMA_VERSION,
        "record_count": len(normalized_records),
        "records_sha256": sha256_file(records_path),
    })
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


def validate_replay_cache_manifest(directory, expected_manifest=None):
    """Validate formal cache identity without materializing JSONL records."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    records_path = directory / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"incomplete recursive replay cache: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_MANIFEST_FIELDS if key not in manifest]
    if missing:
        raise ValueError("replay manifest is missing: " + ", ".join(missing))
    if int(manifest.get("schema_version", -1)) != REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported recursive replay schema version")
    if sha256_file(records_path) != manifest.get("records_sha256"):
        raise ValueError("recursive replay records SHA256 mismatch")
    for key, expected in dict(expected_manifest or {}).items():
        if str(manifest.get(key)) != str(expected):
            raise ValueError(
                f"recursive replay manifest mismatch for {key}: "
                f"expected {expected!r}, got {manifest.get(key)!r}")
    return manifest


class RecursiveReplayCache:
    """Read-only replay lookup that rejects stale or tampered inputs."""

    def __init__(self, directory, expected_manifest=None):
        self.directory = Path(directory)
        records_path = self.directory / "records.jsonl"
        self.manifest = validate_replay_cache_manifest(
            self.directory, expected_manifest=expected_manifest)
        self._records = {}
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                validate_replay_record(record)
                key = replay_key(record["tracklet_key"], record["frame_id"])
                if key in self._records:
                    raise ValueError(f"duplicate recursive replay key: {key}")
                self._records[key] = record
        if len(self._records) != int(self.manifest.get("record_count", -1)):
            raise ValueError("recursive replay record count mismatch")

    def get(self, tracklet_key, frame_id, default=None):
        value = self._records.get(replay_key(tracklet_key, frame_id))
        return default if value is None else copy.deepcopy(value)

    def __len__(self):
        return len(self._records)
