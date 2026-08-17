"""Acquisition row/point accounting shared by preflight and promotion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PREFLIGHT_SCHEMA = "ct_seqtrack.acquisition_preflight.v2"


def acquisition_config_identity(config):
    """Canonical geometry/sampling identity shared by export and training."""
    if isinstance(config, dict):
        source = config
    else:
        source = vars(config)
    keys = {
        "batch_size", "num_candidates", "seed", "hist_num",
        "ct_prior_mode", "ct_time_mode", "dynamics_time_mode",
        "dynamics_fixed_delta_t", "dynamics_time_manifest",
        "point_sample_size", "bb_scale", "bb_offset", "degrees",
        "candidate_trajectory_mode", "use_ct_joint_full",
        "use_b1_prepass_support", "use_calibrated_motion_uncertainty",
        "use_trajectory_search", "use_b1motion_v3",
        "ct_joint_contract_version", "ct_recursive_candidate_views",
        "ct_recursive_tracklet_slots", "ct_recursive_rollout_horizons",
        "ct_recursive_reseed_enabled", "ct_partition_seed",
        "ct_router_partition", "ct_auxiliary_microbatch_size",
        "ct_recovery_candidate_policy", "ct_endpoint_quota",
        "ct_tube_quota", "ct_expansion_point_count",
        "ct_search_training_history", "ct_search_min_points",
        "ct_tube_max_length", "ct_tube_max_width",
        "ct_motion_max_speed", "ct_motion_max_acceleration",
        "ct_motion_max_displacement", "ct_motion_acceleration_weight",
        "ct_search_extension_voxel_size", "ct_search_min_total_points",
        "ct_search_min_extension_points", "ct_search_min_extension_voxels",
        "ct_search_endpoint_ratio", "ct_search_sparse_base_points",
        "search_v3_use_dynamic_sigma", "search_v3_fixed_margin_parallel",
        "search_v3_fixed_margin_perpendicular", "search_v3_coverage_scale",
        "search_v3_standardized_residual_q90_parallel_perpendicular",
        "trajectory_search_base_length", "trajectory_search_base_width",
        "trajectory_search_sigma_parallel_scale",
        "trajectory_search_sigma_perpendicular_scale",
        "trajectory_search_min_displacement",
        "trajectory_search_min_delta_t", "trajectory_search_min_gap_ratio",
        "motion_v3_min_direction_speed",
    }
    selected = {
        str(key): source[key] for key in sorted(keys) if key in source
    }
    # JSON round-trip normalizes EasyDict/list/tuple scalar containers and
    # rejects accidental tensors or opaque runtime objects.
    return json.loads(json.dumps(selected, sort_keys=True))


def sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number(row, key, default=0.0):
    value = float(row.get(key, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"acquisition row has invalid {key}: {value}")
    return value


def summarize_acquisition_rows(rows):
    """Summarize each partition/candidate without mixing populations."""
    groups = {}
    for source in rows:
        row = dict(source)
        partition = str(row.get("partition", "unknown"))
        candidate_id = int(row.get("candidate_id", -1))
        if candidate_id < 0:
            raise ValueError("acquisition row lacks a valid candidate_id")
        key = (partition, candidate_id)
        stats = groups.setdefault(key, {
            "partition": partition,
            "candidate_id": candidate_id,
            "rows": 0,
            "available_rows": 0,
            "eligible_rows": 0,
            "retained_eligible_rows": 0,
            "pool_target_count": 0.0,
            "sampled_target_count": 0.0,
            "sampled_valid_count": 0.0,
            "recovery_positive_rows": 0,
            "recovery_fallback_rows": 0,
        })
        pool_targets = _number(row, "pool_target_count")
        sampled_targets = _number(row, "sampled_target_count")
        has_explicit_sampled_count = "sampled_count" in row
        sampled_valid = _number(
            row, "sampled_count",
            _number(row, "extension_pool_count"))
        if sampled_targets > pool_targets:
            raise ValueError(
                "sampled_target_count cannot exceed pool_target_count")
        # Older diagnostic exports did not include ``sampled_count`` and may
        # use synthetic point totals solely to test row-vs-point accounting.
        # New formal exports always carry the field, so enforce the physical
        # invariant exactly where it is observable without invalidating those
        # historical artifacts/fixtures.
        if has_explicit_sampled_count and sampled_targets > sampled_valid:
            raise ValueError(
                "sampled_target_count cannot exceed sampled_count")
        eligible = pool_targets > 0.0
        retained = eligible and sampled_targets > 0.0
        stats["rows"] += 1
        stats["available_rows"] += int(bool(row.get(
            "available", _number(row, "extension_pool_count") > 0.0)))
        stats["eligible_rows"] += int(eligible)
        stats["retained_eligible_rows"] += int(retained)
        stats["pool_target_count"] += pool_targets
        stats["sampled_target_count"] += sampled_targets
        stats["sampled_valid_count"] += sampled_valid
        stats["recovery_positive_rows"] += int(bool(row.get(
            "recovery_positive", False)))
        stats["recovery_fallback_rows"] += int(bool(row.get(
            "recovery_fallback", False)))
    output = []
    for key in sorted(groups):
        stats = groups[key]
        eligible = stats["eligible_rows"]
        pool_targets = stats["pool_target_count"]
        stats["row_recall"] = (
            stats["retained_eligible_rows"] / eligible
            if eligible else None)
        stats["point_recall"] = (
            stats["sampled_target_count"] / pool_targets
            if pool_targets else None)
        output.append(stats)
    return output


def build_preflight_artifact(
        rows, config_identity, data_manifest_identity, seed,
        primary_partition="dev", min_target_bearing_rows=100,
        min_row_retention=0.5):
    groups = summarize_acquisition_rows(rows)

    def group(partition, candidate_id):
        return next((item for item in groups
                     if item["partition"] == partition
                     and item["candidate_id"] == candidate_id), None)

    primary = group(str(primary_partition), 0)
    weak = group("train", 1)
    strict = group("train", 2)
    train_positive = sum(
        item["sampled_target_count"] for item in groups
        if item["partition"] == "train")
    train_total = sum(
        item["sampled_valid_count"] for item in groups
        if item["partition"] == "train")
    train_negative = max(train_total - train_positive, 0.0)
    criteria = {
        "dev_candidate0_present": primary is not None,
        "dev_candidate0_target_bearing_extension_nonzero": bool(
            primary and primary["eligible_rows"] > 0),
        "dev_candidate0_availability_nonzero": bool(
            primary and primary["available_rows"] > 0),
        "dev_candidate0_retained_nonzero": bool(
            primary and primary["retained_eligible_rows"] > 0),
        "dev_candidate0_target_bearing_rows_at_least_minimum": bool(
            primary and primary["eligible_rows"]
            >= int(min_target_bearing_rows)),
        "dev_candidate0_row_retention_at_least_minimum": bool(
            primary and primary["row_recall"] is not None
            and primary["row_recall"] >= float(min_row_retention)),
        "train_candidate1_recovery_positive_nonzero": bool(
            weak and weak["recovery_positive_rows"] > 0),
        "train_candidate2_strict_recovery_positive_nonzero": bool(
            strict and strict["recovery_positive_rows"] > 0),
        "train_targetness_positive_nonzero": train_positive > 0,
        "train_targetness_negative_nonzero": train_negative > 0,
    }
    class_total = train_positive + train_negative
    targetness_class_weights = {
        "positive": (
            class_total / (2.0 * train_positive)
            if train_positive > 0 else 0.0),
        "negative": (
            class_total / (2.0 * train_negative)
            if train_negative > 0 else 0.0),
        "positive_points": train_positive,
        "negative_points": train_negative,
    }
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "seed": int(seed),
        "primary_population": {
            "partition": str(primary_partition),
            "candidate_id": 0,
        },
        "requirements": {
            "min_target_bearing_rows": int(min_target_bearing_rows),
            "min_row_retention": float(min_row_retention),
        },
        "config": config_identity,
        "data_manifest": data_manifest_identity,
        "groups": groups,
        "targetness_class_weights": targetness_class_weights,
    }
    payload["statistics_sha256"] = sha256_json({
        "schema": PREFLIGHT_SCHEMA,
        "seed": payload["seed"],
        "primary_population": payload["primary_population"],
        "requirements": payload["requirements"],
        "config": payload["config"],
        "data_manifest": payload["data_manifest"],
        "groups": groups,
        "targetness_class_weights": targetness_class_weights,
    })
    return payload


def validate_preflight_artifact(artifact, config):
    """Verify schema, statistics hash and runtime sampling identity."""
    if (not isinstance(artifact, dict)
            or artifact.get("schema") != PREFLIGHT_SCHEMA
            or not bool(artifact.get("passed"))):
        raise ValueError("training requires a passed acquisition preflight v2")
    expected_hash = sha256_json({
        "schema": PREFLIGHT_SCHEMA,
        "seed": artifact.get("seed"),
        "primary_population": artifact.get("primary_population"),
        "requirements": artifact.get("requirements"),
        "config": artifact.get("config"),
        "data_manifest": artifact.get("data_manifest"),
        "groups": artifact.get("groups"),
        "targetness_class_weights": artifact.get(
            "targetness_class_weights"),
    })
    if artifact.get("statistics_sha256") != expected_hash:
        raise ValueError("acquisition preflight statistics hash mismatch")
    if int(artifact.get("seed", -1)) != int(_config_get(
            config, "seed", 42) or 42):
        raise ValueError("acquisition preflight seed mismatch")
    if artifact.get("primary_population") != {
            "partition": "dev", "candidate_id": 0}:
        raise ValueError(
            "acquisition preflight primary population must be dev candidate0")
    requirements = artifact.get("requirements", {})
    if (int(requirements.get("min_target_bearing_rows", -1)) < 100
            or float(requirements.get("min_row_retention", -1.0)) < 0.5):
        raise ValueError(
            "acquisition preflight requirements are weaker than the formal "
            "100-row/50%-retention gate")
    identity = artifact.get("config", {})
    expected_identity = acquisition_config_identity(config)
    if identity.get("acquisition") != expected_identity:
        raise ValueError("acquisition preflight/config identity mismatch")
    manifest_identity = artifact.get("data_manifest", {})
    manifest = manifest_identity.get("manifest")
    if (not isinstance(manifest, dict)
            or manifest.get("schema")
            != "ct_seqtrack.acquisition_data_manifest.v1"
            or manifest.get("checkpoint_loaded") is not False
            or manifest.get("complete") is not True):
        raise ValueError(
            "acquisition preflight lacks a complete checkpoint-free data "
            "manifest")
    if manifest_identity.get("manifest_sha256") != sha256_json(manifest):
        raise ValueError("acquisition data manifest hash mismatch")
    expected_manifest = {
        "dataset": str(_config_get(config, "dataset", "")),
        "split": str(_config_get(config, "train_split", "")),
        "seed": int(_config_get(config, "seed", 42) or 42),
    }
    manifest_mismatches = [
        key for key, value in expected_manifest.items()
        if manifest.get(key) != value]
    runtime_path = _config_get(config, "path")
    manifest_path = manifest.get("path")
    if runtime_path is None or manifest_path is None:
        manifest_mismatches.append("path")
    elif Path(str(runtime_path)).expanduser().resolve() != Path(
            str(manifest_path)).expanduser().resolve():
        manifest_mismatches.append("path")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list):
        manifest_mismatches.append("partitions")
    else:
        indexed = {
            str(item.get("partition")): item for item in partitions
            if isinstance(item, dict)}
        for partition in ("train", "dev"):
            item = indexed.get(partition)
            if (not isinstance(item, dict)
                    or not item.get("tracklet_identity_sha256")
                    or item.get("complete") is not True
                    or int(item.get("exported_rows", -1))
                    != int(item.get("expected_rows", -2))):
                manifest_mismatches.append(f"partitions.{partition}")
    if manifest_mismatches:
        raise ValueError(
            "acquisition preflight data manifest mismatch: "
            + ", ".join(sorted(set(manifest_mismatches))))
    return artifact


def _config_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
