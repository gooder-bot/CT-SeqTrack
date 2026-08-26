#!/usr/bin/env python3
"""Report legacy or schema-v2 B2 acquisition diagnostics on candidate0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


COUNTERFACTUAL_ARMS = (
    "learned_2_1_z0",
    "learned_2_1_z05",
    "learned_2_1_z10",
    "learned_3_1p5_z0",
    "learned_4_2_z0",
    "learned_6_3_z0",
    "cv_2_1_z0",
    "learned_plus_cv_endpoint",
)

COUNTERFACTUAL_METRICS = (
    "valid",
    "xy_target_count",
    "xyz_target_count",
    "target_bearing",
    "raw_point_count",
    "background_count",
    "support_volume",
    "truncated",
    "endpoint_error_xy",
    "endpoint_error_z",
)

V2_REQUIRED_NUMERIC_FIELDS = (
    "acquisition_schema_version",
    "global_target_count_exact",
    "global_target_count_label",
    "global_raw_point_count",
    "base_target_count",
    "base_raw_target_count",
    "base_sampled_target_count",
    "base_raw_point_count",
    "base_sampled_point_count",
    "endpoint_raw_target_count",
    "tube_raw_target_count",
    "endpoint_raw_point_count",
    "tube_raw_point_count",
    "support_union_target_count",
    "support_union_raw_point_count",
    "support_union_background_count",
    "support_xy_target_count",
    "support_xyz_target_count",
    "support_z_clip_target_count",
    "endpoint_target_center_inside_xy",
    "endpoint_target_center_inside_xyz",
    "tube_target_center_inside_xy",
    "tube_target_center_inside_xyz",
    "active_endpoint_error_xy",
    "active_endpoint_error_z",
    "active_tube_error_xy",
    "active_tube_error_z",
    "active_support_truncated",
    "active_endpoint_width",
    "active_endpoint_length",
    "active_endpoint_height",
    "active_tube_width",
    "active_tube_length",
    "active_tube_height",
    "expansion_target_count",
    "pool_target_count",
    "sampled_target_count",
    "extension_pool_count",
    "sampled_count",
    "pool_background_count",
    "sampled_background_count",
    "pool_endpoint_only_target_count",
    "pool_tube_only_target_count",
    "pool_overlap_target_count",
    "sampled_endpoint_only_target_count",
    "sampled_tube_only_target_count",
    "sampled_overlap_target_count",
    "learned_motion_error",
    "kinematic_error",
    "learned_error_parallel",
    "learned_error_perpendicular",
    "learned_cv_disagreement",
    "b1_valid",
    "b1_nll",
    "b1_mahalanobis_sq",
    "b1_coverage_50",
    "b1_coverage_80",
    "b1_coverage_95",
    "sigma_parallel",
    "sigma_perpendicular",
    "query_delta_t",
    "gap_ratio",
    "support_actual_length",
    "support_actual_width",
    "support_volume",
    "recursive_age",
    "recursive_age_valid",
    "search_geometry_source_id",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if "epoch" not in payload:
        raise ValueError("selected checkpoint lacks its completed epoch")
    return int(payload["epoch"]) + 1


def success_auc(overlaps):
    # Match utils.metrics.TorchSuccess exactly: repository evaluation uses
    # float32 overlap tensors and float32 threshold endpoints.
    overlaps = torch.as_tensor(overlaps, dtype=torch.float32)
    thresholds = torch.linspace(0.0, 1.0, steps=21, dtype=torch.float32)
    curve = torch.stack([
        (overlaps >= threshold).to(torch.float32).mean()
        for threshold in thresholds])
    return float(torch.trapz(curve, x=thresholds) * 100.0)


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError("candidate diagnostics contain no rows")
    return rows


def as_float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"candidate diagnostic lacks numeric {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"candidate diagnostic has non-finite {key}")
    return value


def _is_v2_rows(rows):
    schema_present = ["acquisition_schema_version" in row for row in rows]
    if any(schema_present) and not all(schema_present):
        raise ValueError("B2 diagnostics contain mixed acquisition schemas")
    has_v2_markers = any(
        "global_target_count_label" in row
        or any(key.startswith("cf_learned_") for key in row)
        for row in rows)
    if has_v2_markers and not all(schema_present):
        raise ValueError(
            "B2 diagnostics contain v2 fields without a complete schema")
    if not any(schema_present):
        return False
    versions = {as_float(row, "acquisition_schema_version") for row in rows}
    if versions != {2.0}:
        raise ValueError(
            "B2 diagnostics require a homogeneous acquisition schema v2")
    return True


def _validate_v2_rows(rows):
    required = set(V2_REQUIRED_NUMERIC_FIELDS)
    required.update(
        f"cf_{arm}_{metric}"
        for arm in COUNTERFACTUAL_ARMS
        for metric in COUNTERFACTUAL_METRICS)
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(
                "acquisition schema v2 row lacks fields: "
                + ", ".join(missing))
        if not str(row.get("tracklet_key", "")).strip():
            raise ValueError("acquisition schema v2 requires tracklet_key")
        if not str(row.get("active_prior_source", "")).strip():
            raise ValueError("acquisition schema v2 requires active_prior_source")
        values = {key: as_float(row, key) for key in required}
        for key, value in values.items():
            if key.endswith("_count") and value < 0.0:
                raise ValueError(f"acquisition schema v2 {key} is negative")
        for key in (
                "active_endpoint_error_xy", "active_endpoint_error_z",
                "active_tube_error_xy", "active_tube_error_z",
                "active_endpoint_width", "active_endpoint_length",
                "active_endpoint_height", "active_tube_width",
                "active_tube_length", "active_tube_height",
                "learned_motion_error", "kinematic_error",
                "learned_cv_disagreement", "b1_mahalanobis_sq",
                "sigma_parallel", "sigma_perpendicular", "query_delta_t",
                "gap_ratio", "support_actual_length",
                "support_actual_width", "support_volume"):
            if values[key] < 0.0:
                raise ValueError(f"acquisition schema v2 {key} is negative")

        global_exact = values["global_target_count_exact"]
        global_label = values["global_target_count_label"]
        sampled = values["sampled_target_count"]
        pool = values["pool_target_count"]
        expansion = values["expansion_target_count"]
        support_xy = values["support_xy_target_count"]
        support_xyz = values["support_xyz_target_count"]
        if not (0.0 <= sampled <= pool <= expansion <= global_label):
            raise ValueError(
                "acquisition schema v2 requires sampled <= pool <= "
                "expansion <= global label target count")
        if not (0.0 <= global_exact <= global_label):
            raise ValueError(
                "acquisition schema v2 requires exact target <= label target")
        if not (0.0 <= support_xyz <= support_xy <= global_label):
            raise ValueError(
                "acquisition schema v2 requires xyz target <= xy target "
                "<= global label target")
        if support_xyz != expansion:
            raise ValueError(
                "actual support XYZ and expansion target counts disagree")
        if values["support_z_clip_target_count"] != support_xy - support_xyz:
            raise ValueError("z-clip target count is inconsistent")
        if values["support_union_target_count"] != expansion:
            raise ValueError("support union and expansion counts disagree")
        if values["base_target_count"] != values["base_raw_target_count"]:
            raise ValueError(
                "legacy base target and v2 raw base target counts disagree")

        for prefix, total_key in (
                ("pool", "pool_target_count"),
                ("sampled", "sampled_target_count")):
            source_sum = sum(values[
                f"{prefix}_{source}_target_count"] for source in (
                    "endpoint_only", "tube_only", "overlap"))
            if source_sum != values[total_key]:
                raise ValueError(
                    f"{prefix} source target counts do not sum to total")

        count_pairs = (
            ("global_target_count_label", "global_raw_point_count"),
            ("base_raw_target_count", "base_raw_point_count"),
            ("base_sampled_target_count", "base_sampled_point_count"),
            ("endpoint_raw_target_count", "endpoint_raw_point_count"),
            ("tube_raw_target_count", "tube_raw_point_count"),
            ("support_union_target_count", "support_union_raw_point_count"),
            ("pool_target_count", "extension_pool_count"),
            ("sampled_target_count", "sampled_count"),
        )
        for target_key, point_key in count_pairs:
            if not (0.0 <= values[target_key] <= values[point_key]):
                raise ValueError(
                    f"target count {target_key} exceeds {point_key}")
        for prefix, raw_key, target_key in (
                ("support_union", "support_union_raw_point_count",
                 "support_union_target_count"),
                ("pool", "extension_pool_count", "pool_target_count"),
                ("sampled", "sampled_count", "sampled_target_count")):
            if values[f"{prefix}_background_count"] != (
                    values[raw_key] - values[target_key]):
                raise ValueError(
                    f"{prefix} background count is inconsistent")

        binary_fields = (
            "b1_valid", "recursive_age_valid", "active_support_truncated",
            "b1_coverage_50", "b1_coverage_80", "b1_coverage_95",
            "endpoint_target_center_inside_xy",
            "endpoint_target_center_inside_xyz",
            "tube_target_center_inside_xy",
            "tube_target_center_inside_xyz",
        )
        for field in binary_fields:
            if values[field] not in (0.0, 1.0):
                raise ValueError(f"acquisition schema v2 {field} must be binary")
        for arm in COUNTERFACTUAL_ARMS:
            for field in ("valid", "target_bearing", "truncated"):
                key = f"cf_{arm}_{field}"
                if values[key] not in (0.0, 1.0):
                    raise ValueError(f"counterfactual {key} must be binary")
            xy = values[f"cf_{arm}_xy_target_count"]
            xyz = values[f"cf_{arm}_xyz_target_count"]
            raw = values[f"cf_{arm}_raw_point_count"]
            background = values[f"cf_{arm}_background_count"]
            if not (0.0 <= xyz <= xy <= global_label and xyz <= raw):
                raise ValueError(
                    f"counterfactual {arm} target counts are inconsistent")
            if background != raw - xyz:
                raise ValueError(
                    f"counterfactual {arm} background count is inconsistent")
            if any(values[f"cf_{arm}_{field}"] < 0.0 for field in (
                    "support_volume", "endpoint_error_xy",
                    "endpoint_error_z")):
                raise ValueError(
                    f"counterfactual {arm} geometry metric is negative")
            if values[f"cf_{arm}_target_bearing"] != float(xyz > 0.0):
                raise ValueError(
                    f"counterfactual {arm} target-bearing flag disagrees")

        def require_monotonic(keys, label):
            sequence = [values[key] for key in keys]
            if any(right < left for left, right in zip(
                    sequence, sequence[1:])):
                raise ValueError(
                    f"counterfactual {label} target count is not monotonic")

        for suffix in ("xy_target_count", "xyz_target_count"):
            require_monotonic([
                f"cf_learned_2_1_z0_{suffix}",
                f"cf_learned_2_1_z05_{suffix}",
                f"cf_learned_2_1_z10_{suffix}",
            ], f"z-margin {suffix}")
            require_monotonic([
                f"cf_learned_2_1_z0_{suffix}",
                f"cf_learned_3_1p5_z0_{suffix}",
                f"cf_learned_4_2_z0_{suffix}",
                f"cf_learned_6_3_z0_{suffix}",
            ], f"margin {suffix}")
            if (values[f"cf_learned_plus_cv_endpoint_{suffix}"]
                    < values[f"cf_learned_2_1_z0_{suffix}"]):
                raise ValueError(
                    "learned-plus-CV union loses learned support targets")
        if (values["b1_valid"] == 1.0
                and values["search_geometry_source_id"] == 1.0
                and values["cf_learned_2_1_z0_valid"] == 1.0
                and values["cf_learned_2_1_z0_xyz_target_count"]
                != support_xyz):
            raise ValueError(
                "learned_2_1_z0 and actual learned support counts disagree")


def _validated_primary_rows(rows):
    if any("partition" not in row or "candidate_id" not in row
           or "tracklet_id" not in row or "frame_id" not in row
           for row in rows):
        raise ValueError(
            "B2 diagnostics require explicit partition, candidate_id, "
            "tracklet_id and frame_id fields")
    if any(str(row["partition"]) != "dev"
           or int(float(row["candidate_id"])) != 0 for row in rows):
        raise ValueError(
            "B2 diagnostic files may contain only dev candidate0")
    is_v2 = _is_v2_rows(rows)
    if is_v2 and any(
            not str(row.get("tracklet_key", "")).strip() for row in rows):
        raise ValueError("acquisition schema v2 requires tracklet_key")
    row_keys = [(
        str(row["tracklet_key"])
        if is_v2 else int(float(row["tracklet_id"])),
        int(float(row["frame_id"])),
        int(float(row["candidate_id"]))) for row in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("B2 diagnostics contain duplicate rows")
    for row in rows:
        base_count = as_float(row, "base_target_count")
        expansion_count = as_float(row, "expansion_target_count")
        pool_count = as_float(row, "pool_target_count")
        sampled_count = as_float(row, "sampled_target_count")
        if not (0.0 <= sampled_count <= pool_count <= expansion_count):
            raise ValueError(
                "B2 diagnostics require 0 <= sampled_target_count <= "
                "pool_target_count <= expansion_target_count")
        if base_count < 0.0:
            raise ValueError("B2 diagnostics have negative base target count")
    if is_v2:
        _validate_v2_rows(rows)
    if not rows:
        raise ValueError("B2 report population has no dev candidate0 rows")
    return list(rows)


def acquisition_stage(row, weak_base_limit=2.0):
    base_count = as_float(row, "base_target_count")
    expansion_count = as_float(row, "expansion_target_count")
    pool_count = as_float(row, "pool_target_count")
    sampled_count = as_float(row, "sampled_target_count")
    if base_count > float(weak_base_limit):
        return "base_sufficient"
    if expansion_count <= 0.0:
        return "geometry_miss"
    if pool_count <= 0.0:
        return "no_novel_target"
    if sampled_count <= 0.0:
        return "sampling_loss"
    return "retained"


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _percentile(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _v2_subset_summary(rows):
    observable = [
        row for row in rows
        if as_float(row, "global_target_count_label") > 0.0]
    strict = [
        row for row in observable
        if as_float(row, "base_raw_target_count") == 0.0]
    current_bearing = [
        row for row in strict
        if as_float(row, "support_xyz_target_count") > 0.0]
    retained = [
        row for row in strict
        if as_float(row, "sampled_target_count") > 0.0]
    return {
        "rows": len(rows),
        "observable_rows": len(observable),
        "observable_strict_miss_rows": len(strict),
        "current_support_recall": _ratio(len(current_bearing), len(strict)),
        "end_to_end_retention": _ratio(len(retained), len(strict)),
    }


def _stratify(rows, definitions):
    result = {}
    for name, predicate in definitions:
        selected = [row for row in rows if predicate(row)]
        result[name] = _v2_subset_summary(selected)
    return result


def _build_v2_metrics(primary):
    exact_visible = [
        row for row in primary
        if as_float(row, "global_target_count_exact") > 0.0]
    label_visible = [
        row for row in primary
        if as_float(row, "global_target_count_label") > 0.0]
    sensor_unobservable = [
        row for row in primary
        if as_float(row, "global_target_count_label") == 0.0]
    boundary_only = [
        row for row in primary
        if as_float(row, "global_target_count_exact") == 0.0
        and as_float(row, "global_target_count_label") > 0.0]

    raw_strict_miss = [
        row for row in primary
        if as_float(row, "base_raw_target_count") == 0.0]
    base_sample_loss = [
        row for row in primary
        if as_float(row, "base_raw_target_count") > 0.0
        and as_float(row, "base_sampled_target_count") == 0.0]
    observable_strict = [
        row for row in label_visible
        if as_float(row, "base_raw_target_count") == 0.0]

    xy_miss = [
        row for row in observable_strict
        if as_float(row, "support_xy_target_count") == 0.0]
    z_clip_miss = [
        row for row in observable_strict
        if as_float(row, "support_xy_target_count") > 0.0
        and as_float(row, "support_xyz_target_count") == 0.0]
    xyz_geometry_miss = [
        row for row in observable_strict
        if as_float(row, "support_xyz_target_count") == 0.0]
    no_novel_target = [
        row for row in observable_strict
        if as_float(row, "support_xyz_target_count") > 0.0
        and as_float(row, "pool_target_count") == 0.0]
    sampling_loss = [
        row for row in observable_strict
        if as_float(row, "pool_target_count") > 0.0
        and as_float(row, "sampled_target_count") == 0.0]
    retained = [
        row for row in observable_strict
        if as_float(row, "sampled_target_count") > 0.0]
    exclusive_total = sum(map(len, (
        xy_miss, z_clip_miss, no_novel_target, sampling_loss, retained)))
    if exclusive_total != len(observable_strict):
        raise ValueError(
            "observable strict funnel does not partition its population")

    endpoint_only = [
        row for row in observable_strict
        if as_float(row, "endpoint_raw_target_count") > 0.0
        and as_float(row, "tube_raw_target_count") == 0.0]
    tube_only = [
        row for row in observable_strict
        if as_float(row, "endpoint_raw_target_count") == 0.0
        and as_float(row, "tube_raw_target_count") > 0.0]
    both = [
        row for row in observable_strict
        if as_float(row, "endpoint_raw_target_count") > 0.0
        and as_float(row, "tube_raw_target_count") > 0.0]
    neither = [
        row for row in observable_strict
        if as_float(row, "endpoint_raw_target_count") == 0.0
        and as_float(row, "tube_raw_target_count") == 0.0]

    counterfactual_geometry = {}
    for arm in COUNTERFACTUAL_ARMS:
        valid = [
            row for row in observable_strict
            if as_float(row, f"cf_{arm}_valid") == 1.0]
        bearing = [
            row for row in valid
            if as_float(row, f"cf_{arm}_target_bearing") == 1.0]
        recovered = [
            row for row in valid
            if as_float(row, "support_xyz_target_count") == 0.0
            and as_float(row, f"cf_{arm}_xyz_target_count") > 0.0]
        raw_points = [
            as_float(row, f"cf_{arm}_raw_point_count") for row in valid]
        background = [
            as_float(row, f"cf_{arm}_background_count") for row in valid]
        volumes = [
            as_float(row, f"cf_{arm}_support_volume") for row in valid]
        counterfactual_geometry[arm] = {
            "population": "observable_strict_miss",
            "population_rows": len(observable_strict),
            "valid_rows": len(valid),
            "valid_coverage": _ratio(len(valid), len(observable_strict)),
            "target_bearing_rows": len(bearing),
            "target_bearing_recall": _ratio(len(bearing), len(valid)),
            "median_raw_points": _percentile(raw_points, 50),
            "p95_raw_points": _percentile(raw_points, 95),
            "median_background_points": _percentile(background, 50),
            "p95_background_points": _percentile(background, 95),
            "median_support_volume": _percentile(volumes, 50),
            "p95_support_volume": _percentile(volumes, 95),
            "newly_recovered_vs_current_rows": len(recovered),
        }

    learned_valid_rows = [
        row for row in primary if as_float(row, "b1_valid") == 1.0]
    recursive_valid_rows = [
        row for row in primary
        if as_float(row, "recursive_age_valid") == 1.0]
    if recursive_valid_rows:
        ages = [as_float(row, "recursive_age") for row in recursive_valid_rows]
        age_values = sorted(set(ages))
        recursive_age = {
            "valid_rows": len(recursive_valid_rows),
            "median": _percentile(ages, 50),
            "p95": _percentile(ages, 95),
            "by_age": {
                str(age): _v2_subset_summary([
                    row for row in recursive_valid_rows
                    if as_float(row, "recursive_age") == age])
                for age in age_values},
        }
    else:
        recursive_age = None

    return {
        "acquisition_schema_version": 2,
        "observability": {
            "rows": len(primary),
            "exact_visible_rows": len(exact_visible),
            "label_visible_rows": len(label_visible),
            "sensor_unobservable_rows": len(sensor_unobservable),
            "boundary_only_rows": len(boundary_only),
            "label_visible_rate": _ratio(len(label_visible), len(primary)),
        },
        "base_acquisition": {
            "rows": len(primary),
            "raw_strict_miss_rows": len(raw_strict_miss),
            "sample_loss_rows": len(base_sample_loss),
            "observable_strict_miss_rows": len(observable_strict),
            "raw_target_bearing_recall_on_label_visible": _ratio(
                len(label_visible) - len(observable_strict),
                len(label_visible)),
        },
        "observable_strict_funnel": {
            "population": (
                "global_target_count_label > 0 and "
                "base_raw_target_count == 0"),
            "rows": len(observable_strict),
            "xy_miss_rows": len(xy_miss),
            "z_clip_miss_rows": len(z_clip_miss),
            "xyz_geometry_miss_rows": len(xyz_geometry_miss),
            "no_novel_target_rows": len(no_novel_target),
            "sampling_loss_rows": len(sampling_loss),
            "retained_rows": len(retained),
            "retained_rate": _ratio(len(retained), len(observable_strict)),
        },
        "branch_complementarity": {
            "population": "observable_strict_miss",
            "rows": len(observable_strict),
            "endpoint_only_rows": len(endpoint_only),
            "tube_only_rows": len(tube_only),
            "both_rows": len(both),
            "neither_rows": len(neither),
            "endpoint_recall": _ratio(
                len(endpoint_only) + len(both), len(observable_strict)),
            "tube_recall": _ratio(
                len(tube_only) + len(both), len(observable_strict)),
            "union_recall": _ratio(
                len(endpoint_only) + len(tube_only) + len(both),
                len(observable_strict)),
        },
        "counterfactual_geometry": counterfactual_geometry,
        "strata": {
            "global_target_points": _stratify(primary, (
                ("1", lambda row: as_float(
                    row, "global_target_count_label") == 1.0),
                ("2", lambda row: as_float(
                    row, "global_target_count_label") == 2.0),
                ("3-5", lambda row: 3.0 <= as_float(
                    row, "global_target_count_label") <= 5.0),
                (">=6", lambda row: as_float(
                    row, "global_target_count_label") >= 6.0),
            )),
            "learned_error": _stratify(learned_valid_rows, (
                ("<2m", lambda row: as_float(
                    row, "learned_motion_error") < 2.0),
                ("2-4m", lambda row: 2.0 <= as_float(
                    row, "learned_motion_error") < 4.0),
                ("4-8m", lambda row: 4.0 <= as_float(
                    row, "learned_motion_error") < 8.0),
                (">=8m", lambda row: as_float(
                    row, "learned_motion_error") >= 8.0),
            )),
            "gap_ratio": _stratify(primary, (
                ("<=1.25", lambda row: as_float(
                    row, "gap_ratio") <= 1.25),
                ("1.25-2", lambda row: 1.25 < as_float(
                    row, "gap_ratio") <= 2.0),
                (">2", lambda row: as_float(row, "gap_ratio") > 2.0),
            )),
            "recursive_age": recursive_age,
        },
    }


def _summarize_recovery_population(rows, name, definition):
    support_rows = [
        row for row in rows
        if as_float(row, "expansion_target_count") > 0.0]
    pool_rows = [
        row for row in support_rows
        if as_float(row, "pool_target_count") > 0.0]
    sampled_rows = [
        row for row in pool_rows
        if as_float(row, "sampled_target_count") > 0.0]
    pool_target_sum = sum(
        as_float(row, "pool_target_count") for row in pool_rows)
    sampled_target_sum = sum(
        as_float(row, "sampled_target_count") for row in pool_rows)
    return {
        "name": name,
        "definition": definition,
        "rows": len(rows),
        "geometry_miss_rows": len(rows) - len(support_rows),
        "no_novel_target_rows": len(support_rows) - len(pool_rows),
        "sampling_loss_rows": len(pool_rows) - len(sampled_rows),
        "retained_rows": len(sampled_rows),
        "support_row_recall": _ratio(len(support_rows), len(rows)),
        "pool_row_recall": _ratio(len(pool_rows), len(support_rows)),
        "sampling_row_recall": _ratio(
            len(sampled_rows), len(pool_rows)),
        "sampling_point_recall": _ratio(
            sampled_target_sum, pool_target_sum),
        "end_to_end_row_retention": _ratio(
            len(sampled_rows), len(rows)),
    }


def _build_acquisition_metrics(primary):
    is_v2 = _is_v2_rows(primary)
    eligible = [
        row for row in primary
        if as_float(row, "pool_target_count") > 0.0]
    retained = [
        row for row in eligible
        if as_float(row, "sampled_target_count") > 0.0]
    pool_sum = sum(
        as_float(row, "pool_target_count") for row in eligible)
    sampled_sum = sum(
        as_float(row, "sampled_target_count") for row in eligible)
    stages = {
        name: 0 for name in (
            "base_sufficient", "geometry_miss", "no_novel_target",
            "sampling_loss", "retained")}
    for row in primary:
        stages[acquisition_stage(row)] += 1
    weak_rows = [
        row for row in primary
        if as_float(row, "base_target_count") <= 2.0]
    strict_rows = [
        row for row in primary
        if as_float(row, "base_target_count") == 0.0]
    metrics = {
        "population": "dev_candidate0",
        "diagnostic_rows": len(primary),
        "diagnostic_tracklets": len({
            str(row["tracklet_key"])
            if is_v2
            else int(float(row["tracklet_id"]))
            for row in primary}),
        "acquisition_stage_counts": stages,
        "acquisition_weak_recovery": _summarize_recovery_population(
            weak_rows, "weak_recovery", "base_target_count <= 2"),
        "acquisition_strict_miss": _summarize_recovery_population(
            strict_rows, "strict_miss", "base_target_count == 0"),
        # Preserve the original pool-to-sample summary for downstream users.
        "acquisition_eligible_rows": len(eligible),
        "acquisition_retained_rows": len(retained),
        "acquisition_row_recall": _ratio(len(retained), len(eligible)),
        "acquisition_point_recall": _ratio(sampled_sum, pool_sum),
    }
    if is_v2:
        metrics.update(_build_v2_metrics(primary))
    return metrics


def build_acquisition_metrics(rows):
    return _build_acquisition_metrics(_validated_primary_rows(rows))


def _stable_row_key(row):
    tracklet_key = str(row.get("tracklet_key", "")).strip()
    if not tracklet_key:
        raise ValueError("paired B2 comparison requires tracklet_key")
    return (
        tracklet_key,
        int(as_float(row, "frame_id")),
        int(as_float(row, "candidate_id")),
    )


def _serialized_key(key):
    return {
        "tracklet_key": key[0],
        "frame_id": key[1],
        "candidate_id": key[2],
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def build_reference_comparison(primary_rows, reference_rows):
    primary = _validated_primary_rows(primary_rows)
    reference = _validated_primary_rows(reference_rows)
    if not (_is_v2_rows(primary) and _is_v2_rows(reference)):
        raise ValueError("paired B2 comparison requires schema v2 on both sides")
    primary_by_key = {_stable_row_key(row): row for row in primary}
    reference_by_key = {_stable_row_key(row): row for row in reference}
    primary_keys = set(primary_by_key)
    reference_keys = set(reference_by_key)
    intersection = sorted(primary_keys & reference_keys)
    if not intersection:
        raise ValueError("paired B2 comparison has no common stable row keys")
    for key in intersection:
        for field in (
                "global_target_count_exact", "global_target_count_label",
                "global_raw_point_count"):
            if as_float(primary_by_key[key], field) != as_float(
                    reference_by_key[key], field):
                raise ValueError(
                    "paired B2 rows disagree on immutable current-frame "
                    f"observability field {field}")

    def paired_binary_metric(primary_key, reference_key=None):
        reference_key = reference_key or primary_key
        primary_values = [
            float(as_float(primary_by_key[key], primary_key) > 0.0)
            for key in intersection]
        reference_values = [
            float(as_float(reference_by_key[key], reference_key) > 0.0)
            for key in intersection]
        primary_rate = _mean(primary_values)
        reference_rate = _mean(reference_values)
        return {
            "primary_rate": primary_rate,
            "reference_rate": reference_rate,
            "paired_rate_delta": primary_rate - reference_rate,
        }

    both_b1_valid = [
        key for key in intersection
        if as_float(primary_by_key[key], "b1_valid") == 1.0
        and as_float(reference_by_key[key], "b1_valid") == 1.0]
    primary_errors = [
        as_float(primary_by_key[key], "learned_motion_error")
        for key in both_b1_valid]
    reference_errors = [
        as_float(reference_by_key[key], "learned_motion_error")
        for key in both_b1_valid]
    missing_in_primary = sorted(reference_keys - primary_keys)
    missing_in_reference = sorted(primary_keys - reference_keys)
    return {
        "comparison_type": "stable-key paired acquisition diagnostics",
        "interpretation_guard": (
            "All deltas use the stable-key intersection. Unpaired rows are "
            "reported but must not be interpreted as backend promotion."),
        "primary_rows": len(primary_keys),
        "reference_rows": len(reference_keys),
        "paired_rows": len(intersection),
        "row_sets_identical": primary_keys == reference_keys,
        "primary_intersection_coverage": _ratio(
            len(intersection), len(primary_keys)),
        "reference_intersection_coverage": _ratio(
            len(intersection), len(reference_keys)),
        "missing_in_primary": [
            _serialized_key(key) for key in missing_in_primary],
        "missing_in_reference": [
            _serialized_key(key) for key in missing_in_reference],
        "paired_metrics": {
            "b1_valid": paired_binary_metric("b1_valid"),
            "current_support_target_bearing": paired_binary_metric(
                "support_xyz_target_count"),
            "sampled_target_bearing": paired_binary_metric(
                "sampled_target_count"),
            "learned_motion_error_both_valid": {
                "paired_rows": len(both_b1_valid),
                "primary_mean": _mean(primary_errors),
                "reference_mean": _mean(reference_errors),
                "paired_mean_delta": (
                    _mean(primary_errors) - _mean(reference_errors)
                    if both_b1_valid else None),
            },
            "counterfactual_target_bearing": {
                arm: paired_binary_metric(f"cf_{arm}_target_bearing")
                for arm in COUNTERFACTUAL_ARMS},
        },
    }


def build_metrics(
        rows, raw_success, observation_success, margin=0.05,
        action_epsilon=1e-8):
    primary = _validated_primary_rows(rows)
    acquisition_metrics = _build_acquisition_metrics(primary)
    for row in primary:
        if as_float(row, "observation_error") < 0 or as_float(
                row, "raw_search_error") < 0:
            raise ValueError("B2 diagnostics have negative errors")
        if as_float(row, "search_valid") not in (0.0, 1.0):
            raise ValueError("search_valid must be binary")
        if as_float(row, "router_applied_gate") not in (0.0, 1.0):
            raise ValueError("router_applied_gate must be binary")
        for key in ("observation_iou", "raw_search_iou", "selective_iou"):
            if not 0.0 <= as_float(row, key) <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")
    raw_actions = [row for row in primary
                   if bool(int(as_float(row, "search_valid")))
                   and abs(as_float(row, "raw_search_error")
                           - as_float(row, "observation_error"))
                   > action_epsilon]
    center_gains = [
        as_float(row, "observation_error")
        - as_float(row, "raw_search_error") for row in raw_actions]
    iou_gains = [
        as_float(row, "raw_search_iou")
        - as_float(row, "observation_iou") for row in raw_actions]
    helpful = [gain for gain, iou_gain in zip(center_gains, iou_gains)
               if gain > margin and iou_gain >= 0.0]
    harmful = [gain for gain, iou_gain in zip(center_gains, iou_gains)
               if gain < -margin or iou_gain < 0.0]
    action_count = len(raw_actions)
    selective_actions = [
        row for row in primary
        if bool(int(as_float(row, "router_applied_gate")))]
    selective_helpful = [
        row for row in selective_actions
        if as_float(row, "observation_error")
        - as_float(row, "selective_error") > margin]
    selective_harmful = [
        row for row in selective_actions
        if as_float(row, "selective_error")
        - as_float(row, "observation_error") > margin]
    # Diagnostics begin at frame 1.  Tracking initializes frame 0 from GT,
    # so add one IoU=1 endpoint for every dev tracklet before reproducing the
    # repository's exact 21-threshold Success AUC.
    tracklet_count = len({
        int(float(row["tracklet_id"])) for row in primary})
    frame0 = [1.0] * tracklet_count
    computed_observation_success = success_auc(
        frame0 + [as_float(row, "observation_iou") for row in primary])
    computed_raw_success = success_auc(
        frame0 + [as_float(row, "raw_search_iou") for row in primary])
    computed_selective_success = success_auc(
        frame0 + [as_float(row, "selective_iou") for row in primary])
    if abs(float(raw_success) - computed_raw_success) > 1e-4:
        raise ValueError(
            "--raw-search-success does not match candidate diagnostics")
    if abs(float(observation_success) - computed_observation_success) > 1e-4:
        raise ValueError(
            "--observation-success does not match candidate diagnostics")
    return {
        **acquisition_metrics,
        "raw_action_count": action_count,
        "raw_action_rate": action_count / len(primary),
        "raw_helpful_precision": (
            len(helpful) / action_count if action_count else 0.0),
        "raw_harmful_rate": (
            len(harmful) / action_count if action_count else 0.0),
        "raw_center_gain": (
            sum(center_gains) / action_count if action_count else 0.0),
        "raw_iou_gain": (
            sum(iou_gains) / action_count if action_count else 0.0),
        "raw_oracle_center_headroom": (
            sum(max(value, 0.0) for value in center_gains) / action_count
            if action_count else 0.0),
        "raw_oracle_iou_headroom": (
            sum(max(value, 0.0) for value in iou_gains) / action_count
            if action_count else 0.0),
        "selective_action_count": len(selective_actions),
        "selective_action_rate": len(selective_actions) / len(primary),
        "selective_helpful_precision": (
            len(selective_helpful) / len(selective_actions)
            if selective_actions else 0.0),
        "selective_harmful_rate": (
            len(selective_harmful) / len(selective_actions)
            if selective_actions else 0.0),
        "selective_success": computed_selective_success,
        "selective_success_delta": (
            computed_selective_success - computed_observation_success),
        "raw_search_success": float(raw_success),
        "observation_success": float(observation_success),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-diagnostics", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="source B1/B2 evaluation checkpoint")
    parser.add_argument(
        "--reference-candidate-diagnostics", nargs="+",
        help="optional schema-v2 reference CSV(s) for stable-key pairing")
    parser.add_argument(
        "--reference-checkpoint",
        help="checkpoint that produced the reference diagnostics")
    parser.add_argument("--raw-search-success", type=float)
    parser.add_argument("--observation-success", type=float)
    parser.add_argument("--help-margin", type=float, default=0.05)
    parser.add_argument(
        "--acquisition-only", action="store_true",
        help="report support/pool/sample retention without B2 predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if bool(args.reference_candidate_diagnostics) != bool(
            args.reference_checkpoint):
        parser.error(
            "reference comparison requires both "
            "--reference-candidate-diagnostics and --reference-checkpoint")
    rows = load_rows(args.candidate_diagnostics)
    selected_epoch = checkpoint_epoch(args.checkpoint)
    if any("epoch" not in row for row in rows):
        raise ValueError(
            "B2 diagnostics require an explicit validation epoch")
    diagnostic_epochs = {int(float(row["epoch"])) for row in rows}
    if diagnostic_epochs != {selected_epoch}:
        raise ValueError(
            "candidate diagnostics do not match the selected checkpoint "
            f"epoch {selected_epoch}: observed {sorted(diagnostic_epochs)}")
    if args.acquisition_only:
        metrics = build_acquisition_metrics(rows)
    else:
        missing = [
            name for name, value in (
                ("--raw-search-success", args.raw_search_success),
                ("--observation-success", args.observation_success))
            if value is None]
        if missing:
            parser.error(
                "full B2 reporting requires " + ", ".join(missing))
        metrics = build_metrics(
            rows,
            args.raw_search_success, args.observation_success,
            margin=args.help_margin)
    metrics.update({
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_checkpoint_epoch": selected_epoch,
        "acquisition_only": bool(args.acquisition_only),
        "candidate_diagnostics_sha256": [
            sha256_file(path) for path in args.candidate_diagnostics],
    })
    if args.reference_candidate_diagnostics:
        reference_rows = load_rows(args.reference_candidate_diagnostics)
        reference_epoch = checkpoint_epoch(args.reference_checkpoint)
        if any("epoch" not in row for row in reference_rows):
            raise ValueError(
                "reference B2 diagnostics require an explicit validation epoch")
        reference_epochs = {
            int(float(row["epoch"])) for row in reference_rows}
        if reference_epochs != {reference_epoch}:
            raise ValueError(
                "reference diagnostics do not match the selected checkpoint "
                f"epoch {reference_epoch}: observed "
                f"{sorted(reference_epochs)}")
        metrics["paired_acquisition_comparison"] = (
            build_reference_comparison(rows, reference_rows))
        metrics.update({
            "reference_checkpoint_sha256": sha256_file(
                args.reference_checkpoint),
            "reference_checkpoint_epoch": reference_epoch,
            "reference_candidate_diagnostics_sha256": [
                sha256_file(path)
                for path in args.reference_candidate_diagnostics],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
