"""Acquisition row/point accounting shared by preflight and promotion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PREFLIGHT_SCHEMA = "ct_seqtrack.acquisition_preflight.v3"


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
        "ct_recovery_candidate_policy", "ct_candidate_policy",
        "ct_temporal_candidate_gaps", "ct_temporal_boundary_band",
        "ct_presence_training_scope",
        "ct_endpoint_quota",
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


def _quantile(values, probability):
    """Dependency-light linear quantile for immutable JSON artifacts."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
            "role_satisfied_rows": 0,
            "boundary_ratio_sum": 0.0,
            "boundary_ratio_count": 0,
            "gap_counts": {},
            "support_truncated_rows": 0,
            "processing_time_ms_sum": 0.0,
            "processing_time_ms_count": 0,
            "processing_times_ms": [],
            "selection_pool_ratios": {},
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
        role_satisfied = bool(row.get(
            "role_satisfied", candidate_id == 0))
        boundary_ratio = _number(row, "boundary_ratio", 0.0)
        candidate_gap = int(row.get(
            "candidate_gap_frames", 1 if candidate_id == 0 else 0))
        stats["role_satisfied_rows"] += int(role_satisfied)
        stats["boundary_ratio_sum"] += boundary_ratio
        stats["boundary_ratio_count"] += 1
        gap_key = str(candidate_gap)
        stats["gap_counts"][gap_key] = stats["gap_counts"].get(gap_key, 0) + 1
        stats["support_truncated_rows"] += int(bool(row.get(
            "support_truncated", False)))
        if "processing_time_ms" in row:
            processing_time_ms = _number(row, "processing_time_ms")
            stats["processing_time_ms_sum"] += processing_time_ms
            stats["processing_time_ms_count"] += 1
            stats["processing_times_ms"].append(processing_time_ms)
        # The grouped carrier records the full pre-selection pool only on c0,
        # so every logical slot contributes exactly once to each gap stratum.
        if candidate_id == 0:
            for gap, ratio in dict(row.get(
                    "candidate_gap_pool_ratios", {})).items():
                numeric = float(ratio)
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(
                        "candidate gap-pool ratios must be finite and non-negative")
                stats["selection_pool_ratios"].setdefault(
                    str(int(gap)), []).append(numeric)
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
        stats["boundary_ratio_mean"] = (
            stats["boundary_ratio_sum"] / stats["boundary_ratio_count"]
            if stats["boundary_ratio_count"] else None)
        stats["support_truncation_rate"] = (
            stats["support_truncated_rows"] / stats["available_rows"]
            if stats["available_rows"] else None)
        stats["processing_time_ms_mean"] = (
            stats["processing_time_ms_sum"]
            / stats["processing_time_ms_count"]
            if stats["processing_time_ms_count"] else None)
        stats["processing_time_ms_p95"] = _quantile(
            stats["processing_times_ms"], 0.95)
        del stats["processing_times_ms"]
        # Replace raw values by stable distribution summaries before hashing.
        pool_summary = {}
        for gap, values in sorted(stats["selection_pool_ratios"].items()):
            pool_summary[gap] = {
                "count": len(values),
                "min": min(values),
                "q10": _quantile(values, 0.10),
                "q50": _quantile(values, 0.50),
                "q90": _quantile(values, 0.90),
                "max": max(values),
            }
        stats["selection_pool_ratio_distribution"] = pool_summary
        del stats["selection_pool_ratios"]
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
    boundary = group(str(primary_partition), 1)
    outside = group(str(primary_partition), 2)
    train_positive = sum(
        item["sampled_target_count"] for item in groups
        if item["partition"] == "train")
    train_total = sum(
        item["sampled_valid_count"] for item in groups
        if item["partition"] == "train")
    train_negative = max(train_total - train_positive, 0.0)
    acquisition_identity = (
        config_identity.get("acquisition", {})
        if isinstance(config_identity, dict) else {})
    candidate_policy = str(acquisition_identity.get(
        "ct_candidate_policy", "causal_b1_boundary"))
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
        "train_targetness_positive_nonzero": train_positive > 0,
        "train_targetness_negative_nonzero": train_negative > 0,
    }
    if candidate_policy == "causal_temporal_uniform":
        criteria.update({
            "dev_candidate1_uniform_available_rows_at_least_minimum": bool(
                boundary and boundary["available_rows"]
                >= int(min_target_bearing_rows)),
            "dev_candidate2_uniform_available_rows_at_least_minimum": bool(
                outside and outside["available_rows"]
                >= int(min_target_bearing_rows)),
        })
    else:
        criteria.update({
            "dev_candidate1_boundary_role_rows_at_least_minimum": bool(
                boundary and boundary["role_satisfied_rows"]
                >= int(min_target_bearing_rows)),
            "dev_candidate2_outside_role_rows_at_least_minimum": bool(
                outside and outside["role_satisfied_rows"]
                >= int(min_target_bearing_rows)),
        })
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
        raise ValueError("training requires a passed causal acquisition preflight v3")
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
            != "ct_seqtrack.acquisition_data_manifest.v2"
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
    if manifest.get("history_source") != (
            "past_observation_fixed_cv_causal_geometry_audit"):
        manifest_mismatches.append("history_source")
    if manifest.get("current_gt_role") != {
            "candidate0": "target-count-label-only",
            "candidate1_2": "target-count-label-only"}:
        manifest_mismatches.append("current_gt_role")
    runtime_policy = str(_config_get(
        config, "ct_candidate_policy", "causal_b1_boundary"))
    expected_selection = (
        "uniform valid temporal gaps; current GT hidden"
        if runtime_policy == "causal_temporal_uniform" else
        "fixed-cv endpoint boundary/outside; current GT hidden")
    if manifest.get("candidate_selection") != expected_selection:
        manifest_mismatches.append("candidate_selection")
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
