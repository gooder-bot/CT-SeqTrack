"""Summarize and pair frozen endpoint exports produced by M0.

This script has no PyTorch, nuScenes, or scikit-learn dependency.  It treats a
tracklet as the statistical unit, validates endpoint identity before paired
comparisons, and reports recursive failure timing, path variance, and tracking
metrics from the raw endpoint CSV files.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def safe_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "m0"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_summary(values):
    array = np.asarray(
        [value for value in (finite_float(item) for item in values) if value is not None],
        dtype=np.float64,
    )
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


def tracking_scores(ious, distances):
    overlaps = np.asarray(ious, dtype=np.float64)
    accuracy = np.asarray(distances, dtype=np.float64)
    if overlaps.size == 0 or accuracy.size == 0:
        return {
            "success": None,
            "precision": None,
            "mean_iou": None,
            "mean_center_error": None,
        }
    success_x = np.linspace(0.0, 1.0, 21)
    precision_x = np.linspace(0.0, 2.0, 21)
    success_curve = np.asarray([np.mean(overlaps >= value) for value in success_x])
    precision_curve = np.asarray([np.mean(accuracy <= value) for value in precision_x])
    return {
        "success": float(np.trapz(success_curve, x=success_x) * 100.0),
        "precision": float(np.trapz(precision_curve, x=precision_x) * 50.0),
        "mean_iou": float(np.mean(overlaps)),
        "mean_center_error": float(np.mean(accuracy)),
    }


def endpoint_key(row):
    return (
        str(row.get("tracklet_key", "")),
        int(float(row.get("source_frame_index", row.get("frame_index", 0)))),
        str(row.get("frame_token", "")),
    )


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "tracklet_key",
            "frame_index",
            "source_frame_index",
            "frame_token",
            "iou",
            "center_error",
            "checkpoint_sha256",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"Missing required columns in {path}: {missing}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    seen = set()
    for row in rows:
        for field in ("iou", "center_error"):
            if finite_float(row.get(field)) is None:
                raise RuntimeError(
                    f"Non-finite {field} in {path} at row {len(seen) + 2}: "
                    f"{row.get(field)!r}"
                )
        key = endpoint_key(row)
        if key in seen:
            raise RuntimeError(f"Duplicate endpoint in {path}: {key}")
        seen.add(key)
    return rows


def parse_labeled_path(value):
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path.strip()).resolve()
    if not label:
        raise ValueError(f"Empty label in {value!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def parse_comparison(value):
    if ":" not in value:
        raise ValueError(f"Expected LEFT:RIGHT, got {value!r}")
    left, right = [item.strip() for item in value.split(":", 1)]
    if not left or not right:
        raise ValueError(f"Invalid comparison: {value!r}")
    return left, right


def group_by_tracklet(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["tracklet_key"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(float(row["frame_index"])))
    return dict(grouped)


def max_true_streak(values):
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def first_persistent_failure(rows, threshold, consecutive):
    failures = [finite_float(row.get("center_error")) > threshold for row in rows]
    current = 0
    for index, failed in enumerate(failures):
        current = current + 1 if failed else 0
        if current >= consecutive:
            start_index = index - consecutive + 1
            row = rows[start_index]
            return {
                "frame_index": int(float(row["frame_index"])),
                "source_frame_index": int(
                    float(row.get("source_frame_index", row["frame_index"]))
                ),
                "frame_token": str(row.get("frame_token", "")),
                "current_delta_t_real": finite_float(
                    row.get("current_delta_t_real")
                ),
            }
    return None


def summarize_tracklet(rows, failure_threshold, failure_consecutive):
    metrics = tracking_scores(
        [finite_float(row["iou"]) for row in rows],
        [finite_float(row["center_error"]) for row in rows],
    )
    failures = [finite_float(row["center_error"]) > failure_threshold for row in rows]
    first_failure = next(
        (row for row, failed in zip(rows, failures) if failed), None
    )
    persistent = first_persistent_failure(
        rows, failure_threshold, failure_consecutive
    )
    path_rows = [row for row in rows if parse_bool(row.get("path_variance_available"))]
    result = {
        **metrics,
        "frame_count": len(rows),
        "empty_fallback_count": int(
            sum(parse_bool(row.get("empty_fallback")) for row in rows)
        ),
        "failure_count": int(sum(failures)),
        "max_failure_streak": max_true_streak(failures),
        "first_failure_frame_index": int(float(first_failure["frame_index"]))
        if first_failure
        else None,
        "first_failure_source_frame_index": int(
            float(first_failure.get("source_frame_index", first_failure["frame_index"]))
        )
        if first_failure
        else None,
        "first_persistent_failure_frame_index": persistent["frame_index"]
        if persistent
        else None,
        "first_persistent_failure_source_frame_index": persistent[
            "source_frame_index"
        ]
        if persistent
        else None,
        "path_variance_count": len(path_rows),
        "path_center_gap_mean": finite_summary(
            [row.get("path_center_gap") for row in path_rows]
        )["mean"],
        "path_yaw_gap_mean": finite_summary(
            [row.get("path_yaw_gap") for row in path_rows]
        )["mean"],
    }
    return result


def bucket_label(value, boundaries):
    value = finite_float(value)
    if value is None:
        return "missing"
    lower = None
    for boundary in boundaries:
        if value < boundary:
            return f"[{lower if lower is not None else '-inf'},{boundary})"
        lower = boundary
    return f"[{lower if lower is not None else '-inf'},inf)"


def summarize_buckets(rows, field, boundaries):
    grouped = defaultdict(list)
    for row in rows:
        grouped[bucket_label(row.get(field), boundaries)].append(row)
    result = {}
    for label, subset in sorted(grouped.items()):
        scores = tracking_scores(
            [finite_float(row["iou"]) for row in subset],
            [finite_float(row["center_error"]) for row in subset],
        )
        result[label] = {
            "count": len(subset),
            **scores,
            "empty_fallback_rate": float(
                np.mean([parse_bool(row.get("empty_fallback")) for row in subset])
            ),
        }
    return result


def summarize_run(rows, failure_threshold, failure_consecutive, bins):
    grouped = group_by_tracklet(rows)
    tracklets = {
        key: summarize_tracklet(values, failure_threshold, failure_consecutive)
        for key, values in grouped.items()
    }
    scores = tracking_scores(
        [finite_float(row["iou"]) for row in rows],
        [finite_float(row["center_error"]) for row in rows],
    )
    path_rows = [row for row in rows if parse_bool(row.get("path_variance_available"))]
    return {
        "endpoint_count": len(rows),
        "tracklet_count": len(grouped),
        "protocols": sorted({str(row.get("protocol", "")) for row in rows}),
        "time_modes": sorted({str(row.get("time_mode", "")) for row in rows}),
        "checkpoint_sha256": sorted(
            {str(row.get("checkpoint_sha256", "")) for row in rows}
        ),
        **scores,
        "empty_fallback_count": int(
            sum(parse_bool(row.get("empty_fallback")) for row in rows)
        ),
        "empty_fallback_rate": float(
            np.mean([parse_bool(row.get("empty_fallback")) for row in rows])
        ),
        "tracklet_failure": {
            "first_failure_frame": finite_summary(
                [value["first_failure_frame_index"] for value in tracklets.values()]
            ),
            "first_persistent_failure_frame": finite_summary(
                [
                    value["first_persistent_failure_frame_index"]
                    for value in tracklets.values()
                ]
            ),
            "max_failure_streak": finite_summary(
                [value["max_failure_streak"] for value in tracklets.values()]
            ),
        },
        "path_variance": {
            "eligible_count": len(path_rows),
            "center_gap": finite_summary(
                [row.get("path_center_gap") for row in path_rows]
            ),
            "yaw_gap": finite_summary([row.get("path_yaw_gap") for row in path_rows]),
            "anchor_gap_max": finite_summary(
                [row.get("path_anchor_gap_max") for row in path_rows]
            )["max"],
            "current_point_gap_max": finite_summary(
                [row.get("path_current_point_gap_max") for row in path_rows]
            )["max"],
        },
        "by_current_delta_t_real": summarize_buckets(
            rows, "current_delta_t_real", bins["delta_t"]
        ),
        "by_gt_displacement": summarize_buckets(
            rows, "gt_displacement_from_previous_gt", bins["displacement"]
        ),
        "by_search_points": summarize_buckets(
            rows, "num_points_in_search", bins["points"]
        ),
        "tracklets": tracklets,
    }


def validate_endpoint_identity(left_rows, right_rows, allow_mismatch):
    left_keys = [endpoint_key(row) for row in left_rows]
    right_keys = [endpoint_key(row) for row in right_rows]
    left_set = set(left_keys)
    right_set = set(right_keys)
    report = {
        "left_count": len(left_keys),
        "right_count": len(right_keys),
        "common_count": len(left_set & right_set),
        "left_only_count": len(left_set - right_set),
        "right_only_count": len(right_set - left_set),
        "same_order": left_keys == right_keys,
        "exact_match": left_keys == right_keys,
        "left_only_first": [list(key) for key in sorted(left_set - right_set)[:3]],
        "right_only_first": [list(key) for key in sorted(right_set - left_set)[:3]],
    }
    if not report["exact_match"] and not allow_mismatch:
        raise RuntimeError(f"Endpoint identity mismatch: {report}")
    return report


def bootstrap_mean_ci(values, seed, iterations, alpha=0.05):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(int(seed))
    if array.size == 1:
        samples = np.repeat(array, int(iterations))
    else:
        indices = rng.integers(0, array.size, size=(int(iterations), array.size))
        samples = array[indices].mean(axis=1)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "low": float(np.quantile(samples, alpha / 2.0)),
        "high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
    }


def paired_comparison(
    left_label,
    right_label,
    left_rows,
    right_rows,
    left_summary,
    right_summary,
    allow_mismatch,
    seed,
    bootstrap_iterations,
):
    identity = validate_endpoint_identity(left_rows, right_rows, allow_mismatch)
    right_by_key = {endpoint_key(row): row for row in right_rows}
    endpoint_rows = []
    for left in left_rows:
        key = endpoint_key(left)
        right = right_by_key.get(key)
        if right is None:
            continue
        left_error = finite_float(left["center_error"])
        right_error = finite_float(right["center_error"])
        endpoint_rows.append(
            {
                "comparison": f"{left_label}:{right_label}",
                "tracklet_key": key[0],
                "source_frame_index": key[1],
                "frame_token": key[2],
                "left_iou": finite_float(left["iou"]),
                "right_iou": finite_float(right["iou"]),
                "iou_delta_left_minus_right": finite_float(left["iou"])
                - finite_float(right["iou"]),
                "left_center_error": left_error,
                "right_center_error": right_error,
                "center_error_delta_left_minus_right": left_error - right_error,
                "center_error_gain_left_over_right": right_error - left_error,
                "left_empty_fallback": parse_bool(left.get("empty_fallback")),
                "right_empty_fallback": parse_bool(right.get("empty_fallback")),
                "left_path_center_gap": finite_float(left.get("path_center_gap")),
                "right_path_center_gap": finite_float(right.get("path_center_gap")),
            }
        )

    left_tracklets = left_summary["tracklets"]
    right_tracklets = right_summary["tracklets"]
    common_tracklets = sorted(set(left_tracklets) & set(right_tracklets))
    tracklet_rows = []
    for key in common_tracklets:
        left = left_tracklets[key]
        right = right_tracklets[key]
        tracklet_rows.append(
            {
                "comparison": f"{left_label}:{right_label}",
                "tracklet_key": key,
                "success_delta_left_minus_right": left["success"] - right["success"],
                "precision_delta_left_minus_right": left["precision"]
                - right["precision"],
                "center_error_delta_left_minus_right": left["mean_center_error"]
                - right["mean_center_error"],
                "center_error_gain_left_over_right": right["mean_center_error"]
                - left["mean_center_error"],
                "empty_fallback_delta_left_minus_right": left[
                    "empty_fallback_count"
                ]
                - right["empty_fallback_count"],
                "max_failure_streak_delta_left_minus_right": left[
                    "max_failure_streak"
                ]
                - right["max_failure_streak"],
                "path_center_gap_delta_left_minus_right": (
                    left["path_center_gap_mean"] - right["path_center_gap_mean"]
                    if left["path_center_gap_mean"] is not None
                    and right["path_center_gap_mean"] is not None
                    else None
                ),
            }
        )

    fields = (
        "success_delta_left_minus_right",
        "precision_delta_left_minus_right",
        "center_error_delta_left_minus_right",
        "center_error_gain_left_over_right",
        "empty_fallback_delta_left_minus_right",
        "max_failure_streak_delta_left_minus_right",
        "path_center_gap_delta_left_minus_right",
    )
    statistics = {}
    for index, field in enumerate(fields):
        values = [row[field] for row in tracklet_rows if row[field] is not None]
        statistics[field] = {
            "distribution": finite_summary(values),
            "tracklet_bootstrap_mean_95ci": bootstrap_mean_ci(
                values,
                seed=int(seed) + index * 1009,
                iterations=bootstrap_iterations,
            ),
        }

    return {
        "left": left_label,
        "right": right_label,
        "identity": identity,
        "endpoint_delta_count": len(endpoint_rows),
        "tracklet_delta_count": len(tracklet_rows),
        "aggregate_delta": {
            "success_left_minus_right": left_summary["success"]
            - right_summary["success"],
            "precision_left_minus_right": left_summary["precision"]
            - right_summary["precision"],
            "mean_center_error_left_minus_right": left_summary[
                "mean_center_error"
            ]
            - right_summary["mean_center_error"],
            "empty_fallback_left_minus_right": left_summary[
                "empty_fallback_count"
            ]
            - right_summary["empty_fallback_count"],
        },
        "tracklet_statistics": statistics,
        "endpoint_rows": endpoint_rows,
        "tracklet_rows": tracklet_rows,
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(summary):
    lines = [
        f"# M0 endpoint summary: {summary['tag']}",
        "",
        "## Run metrics",
        "",
        "| run | endpoints | tracklets | Success | Precision | mean error | empty fallback | path gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, run in summary["runs"].items():
        path_mean = run["path_variance"]["center_gap"]["mean"]
        lines.append(
            "| {label} | {endpoint_count} | {tracklet_count} | {success:.4f} | "
            "{precision:.4f} | {mean_error:.6f} | {empty} | {path_gap} |".format(
                label=label,
                endpoint_count=run["endpoint_count"],
                tracklet_count=run["tracklet_count"],
                success=run["success"],
                precision=run["precision"],
                mean_error=run["mean_center_error"],
                empty=run["empty_fallback_count"],
                path_gap=f"{path_mean:.6f}" if path_mean is not None else "NA",
            )
        )

    if summary["comparisons"]:
        lines.extend(["", "## Paired comparisons", ""])
        for name, comparison in summary["comparisons"].items():
            delta = comparison["aggregate_delta"]
            lines.extend(
                [
                    f"### {name}",
                    "",
                    f"- Endpoint exact match: `{comparison['identity']['exact_match']}`",
                    f"- Success left-right: `{delta['success_left_minus_right']:.6f}`",
                    f"- Precision left-right: `{delta['precision_left_minus_right']:.6f}`",
                    f"- Mean center error left-right: `{delta['mean_center_error_left_minus_right']:.6f}`",
                    f"- Empty fallback left-right: `{delta['empty_fallback_left_minus_right']}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "All deltas are paired by tracklet/endpoint. Epochs and frames are not treated as independent statistical samples. This report is a frozen-checkpoint diagnostic and does not promote a method by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test():
    dummy = [
        {
            "tracklet_key": "t0",
            "frame_index": "0",
            "source_frame_index": "0",
            "frame_token": "f0",
            "iou": "1.0",
            "center_error": "0.0",
            "empty_fallback": "False",
            "path_variance_available": "False",
            "protocol": "standard",
            "time_mode": "true",
            "checkpoint_sha256": "a",
            "current_delta_t_real": "",
            "gt_displacement_from_previous_gt": "0",
            "num_points_in_search": "",
        },
        {
            "tracklet_key": "t0",
            "frame_index": "1",
            "source_frame_index": "1",
            "frame_token": "f1",
            "iou": "0.0",
            "center_error": "3.0",
            "empty_fallback": "True",
            "path_variance_available": "True",
            "path_center_gap": "0.2",
            "path_yaw_gap": "0.1",
            "path_anchor_gap_max": "0",
            "path_current_point_gap_max": "0",
            "protocol": "standard",
            "time_mode": "true",
            "checkpoint_sha256": "a",
            "current_delta_t_real": "1.0",
            "gt_displacement_from_previous_gt": "2.0",
            "num_points_in_search": "10",
        },
    ]
    scores = tracking_scores([1.0, 0.0], [0.0, 2.0])
    if not np.isclose(scores["success"], 51.25):
        raise RuntimeError(f"tracking score self-test failed: {scores}")
    tracklet = summarize_tracklet(dummy, 2.0, 1)
    if tracklet["first_failure_frame_index"] != 1 or tracklet["max_failure_streak"] != 1:
        raise RuntimeError(f"failure summary self-test failed: {tracklet}")
    identity = validate_endpoint_identity(dummy, list(dummy), allow_mismatch=False)
    if not identity["exact_match"]:
        raise RuntimeError("identity self-test failed")
    ci = bootstrap_mean_ci([1.0, 1.0, 1.0], seed=42, iterations=100)
    if ci["low"] != 1.0 or ci["high"] != 1.0:
        raise RuntimeError(f"bootstrap self-test failed: {ci}")
    print("M0 endpoint summarizer self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and pair M0 endpoint CSV exports."
    )
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--allow-endpoint-mismatch", action="store_true")
    parser.add_argument("--require-same-checkpoint", action="store_true")
    parser.add_argument("--failure-threshold", type=float, default=2.0)
    parser.add_argument("--failure-consecutive", type=int, default=2)
    parser.add_argument("--delta-t-bins", default="0.75,1.0,2.0")
    parser.add_argument("--displacement-bins", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--point-bins", default="5,20,128")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/diagnostics/m0_analysis")
    parser.add_argument("--tag", default="m0_endpoint_analysis")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.input:
        parser.error("At least one --input LABEL=PATH is required")
    if args.failure_consecutive <= 0:
        parser.error("--failure-consecutive must be positive")
    if args.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")

    paths = {}
    rows_by_label = {}
    for item in args.input:
        label, path = parse_labeled_path(item)
        if label in rows_by_label:
            raise ValueError(f"Duplicate input label: {label}")
        paths[label] = path
        rows_by_label[label] = read_rows(path)

    bins = {
        "delta_t": [float(value) for value in args.delta_t_bins.split(",") if value],
        "displacement": [
            float(value) for value in args.displacement_bins.split(",") if value
        ],
        "points": [float(value) for value in args.point_bins.split(",") if value],
    }
    run_summaries = {
        label: summarize_run(
            rows,
            args.failure_threshold,
            args.failure_consecutive,
            bins,
        )
        for label, rows in rows_by_label.items()
    }

    if args.require_same_checkpoint:
        hashes = {
            tuple(summary["checkpoint_sha256"])
            for summary in run_summaries.values()
        }
        if len(hashes) != 1:
            raise RuntimeError(f"Checkpoint mismatch under --require-same-checkpoint: {hashes}")

    comparisons = {}
    endpoint_delta_rows = []
    tracklet_delta_rows = []
    for item in args.comparison:
        left, right = parse_comparison(item)
        if left not in rows_by_label or right not in rows_by_label:
            raise KeyError(f"Unknown comparison labels: {left}:{right}")
        comparison = paired_comparison(
            left,
            right,
            rows_by_label[left],
            rows_by_label[right],
            run_summaries[left],
            run_summaries[right],
            args.allow_endpoint_mismatch,
            args.seed,
            args.bootstrap_iterations,
        )
        endpoint_delta_rows.extend(comparison.pop("endpoint_rows"))
        tracklet_delta_rows.extend(comparison.pop("tracklet_rows"))
        comparisons[f"{left}:{right}"] = comparison

    run_csv_rows = []
    tracklet_csv_rows = []
    for label, run in run_summaries.items():
        run_csv_rows.append(
            {
                "run": label,
                "endpoint_count": run["endpoint_count"],
                "tracklet_count": run["tracklet_count"],
                "success": run["success"],
                "precision": run["precision"],
                "mean_iou": run["mean_iou"],
                "mean_center_error": run["mean_center_error"],
                "empty_fallback_count": run["empty_fallback_count"],
                "path_center_gap_mean": run["path_variance"]["center_gap"]["mean"],
                "path_yaw_gap_mean": run["path_variance"]["yaw_gap"]["mean"],
            }
        )
        for tracklet_key, metrics in run["tracklets"].items():
            tracklet_csv_rows.append(
                {"run": label, "tracklet_key": tracklet_key, **metrics}
            )

    summary = {
        "schema": "ct_seqtrack.m0_endpoint_summary",
        "schema_version": 1,
        "tag": args.tag,
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in paths.items()
        },
        "failure_threshold": args.failure_threshold,
        "failure_consecutive": args.failure_consecutive,
        "bins": bins,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "runs": run_summaries,
        "comparisons": comparisons,
        "note": (
            "Tracklets are the bootstrap unit. Endpoint identity is checked before paired "
            "comparisons; frames and epochs are not treated as independent samples."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_tag(args.tag)
    summary_path = output_dir / f"{prefix}_summary.json"
    report_path = output_dir / f"{prefix}_report.md"
    runs_path = output_dir / f"{prefix}_runs.csv"
    tracklets_path = output_dir / f"{prefix}_tracklets.csv"
    endpoint_deltas_path = output_dir / f"{prefix}_endpoint_deltas.csv"
    tracklet_deltas_path = output_dir / f"{prefix}_tracklet_deltas.csv"

    with open(summary_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2, allow_nan=False)
    with open(report_path, "w", encoding="utf-8") as output_file:
        output_file.write(markdown_report(summary))
    write_csv(runs_path, run_csv_rows)
    write_csv(tracklets_path, tracklet_csv_rows)
    write_csv(endpoint_deltas_path, endpoint_delta_rows)
    write_csv(tracklet_deltas_path, tracklet_delta_rows)

    print(json.dumps({"runs": run_csv_rows, "comparisons": comparisons}, indent=2))
    print(f"summary json: {summary_path}")
    print(f"report markdown: {report_path}")
    print(f"run csv: {runs_path}")
    print(f"tracklet csv: {tracklets_path}")
    if endpoint_delta_rows:
        print(f"endpoint delta csv: {endpoint_deltas_path}")
    if tracklet_delta_rows:
        print(f"tracklet delta csv: {tracklet_deltas_path}")


if __name__ == "__main__":
    main()
