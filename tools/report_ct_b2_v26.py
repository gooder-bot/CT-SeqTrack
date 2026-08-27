#!/usr/bin/env python3
"""Validate and summarize CT-SeqTrack v26 acquisition diagnostics.

The report is intentionally candidate0-only.  It treats schema-v3 raw support,
novel support, the 768-point pre-pool, the 256-point selection and consensus
voting as separate funnel stages; the historical schema-v2 ``target_bearing``
field is neither accepted nor reconstructed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


COUNTERFACTUAL_ARMS = (
    "fixed_2_1", "adaptive_local", "adaptive_dual_support")
COUNTERFACTUAL_METRICS = (
    "valid", "raw_target_bearing", "novel_target_bearing",
    "support_raw_target_count", "support_raw_point_count",
    "support_raw_background_count", "support_novel_target_count",
    "support_novel_point_count", "support_novel_background_count",
    "support_volume", "truncated", "uses_endpoint", "uses_tube",
    "uses_corridor")

REQUIRED = (
    "acquisition_schema_version", "candidate_id", "tracklet_id",
    "frame_id", "global_target_count_label", "search_coverage_need",
    "base_raw_target_count", "base_raw_point_count",
    "support_raw_target_count", "support_raw_point_count",
    "support_raw_background_count", "support_novel_target_count",
    "support_novel_point_count", "support_novel_background_count",
    "raw_target_bearing", "novel_target_bearing",
    "novel_pool_target_count", "novel_pool_point_count",
    "novel_pool_background_count", "prepool_target_count",
    "prepool_point_count", "prepool_background_count",
    "selected_target_count", "selected_point_count",
    "selected_target_bearing", "consensus_inlier_count",
    "selected_relation_point_count", "selected_relation_target_count",
    "selected_spatial_point_count", "selected_spatial_target_count",
    "selected_exploration_point_count", "selected_exploration_target_count",
    "selected_borrowed_point_count", "selected_borrowed_target_count",
    "relation_auroc", "relation_ap", "relation_auprc", "relation_ece",
    "vote_consistency", "vote_covariance_xx", "vote_covariance_xy",
    "vote_covariance_yy", "vote_inlier_ratio",
    "vote_effective_mass",
    "vote_candidate_margin", "vote_compatible_hypothesis_count",
    "observation_error", "raw_search_error", "observation_iou",
    "raw_search_iou", "action_score", "presence_score",
    "router_applied_gate", "center_gain", "iou_gain", "gap_ratio",
    "recursive_age", "active_support_truncated", "support_volume",
    "endpoint_support_novel_target_count",
    "tube_support_novel_target_count",
    "corridor_support_novel_target_count")


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError("v26 candidate diagnostics contain no rows")
    return rows


def value(row, key):
    try:
        result = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"v26 diagnostic lacks numeric {key}") from error
    if not math.isfinite(result):
        raise ValueError(f"v26 diagnostic has non-finite {key}")
    return result


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def aggregate(rows):
    if not rows:
        return {"rows": 0}
    return {
        "rows": len(rows),
        "novel_pool_target_bearing": ratio(sum(
            value(row, "novel_pool_target_count") > 0 for row in rows),
            len(rows)),
        "selection_row_recall": ratio(sum(
            value(row, "selected_target_count") > 0 for row in rows
            if value(row, "prepool_target_count") > 0), sum(
                value(row, "prepool_target_count") > 0 for row in rows)),
        "selection_point_recall": ratio(sum(
            value(row, "selected_target_count") for row in rows), sum(
                value(row, "prepool_target_count") for row in rows)),
        "support_novel_points_mean": mean(
            value(row, "support_novel_point_count") for row in rows),
        "novel_pool_points_mean": mean(
            value(row, "novel_pool_point_count") for row in rows),
        "prepool_points_mean": mean(
            value(row, "prepool_point_count") for row in rows),
        "selected_points_mean": mean(
            value(row, "selected_point_count") for row in rows),
        "consensus_inliers_mean": mean(
            value(row, "consensus_inlier_count") for row in rows),
    }


def validate(rows):
    required = set(REQUIRED)
    required.update(
        f"cf_{arm}_{metric}"
        for arm in COUNTERFACTUAL_ARMS
        for metric in COUNTERFACTUAL_METRICS)
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(
                f"v26 row {index} is missing: " + ", ".join(missing))
        if value(row, "acquisition_schema_version") != 3.0:
            raise ValueError("v26 report accepts acquisition schema v3 only")
        if int(value(row, "candidate_id")) != 0:
            raise ValueError("v26 formal diagnostics must contain candidate0 only")
        if "target_bearing" in row:
            raise ValueError(
                "ambiguous schema-v2 target_bearing is forbidden in v26")

        raw_target = value(row, "support_raw_target_count")
        novel_target = value(row, "support_novel_target_count")
        raw_points = value(row, "support_raw_point_count")
        novel_points = value(row, "support_novel_point_count")
        pool_target = value(row, "novel_pool_target_count")
        pool_points = value(row, "novel_pool_point_count")
        prepool_target = value(row, "prepool_target_count")
        prepool_points = value(row, "prepool_point_count")
        selected_target = value(row, "selected_target_count")
        selected_points = value(row, "selected_point_count")
        inliers = value(row, "consensus_inlier_count")
        if not (0 <= novel_target <= raw_target
                and 0 <= novel_points <= raw_points):
            raise ValueError("novel support is not a subset of raw support")
        if not (0 <= pool_target <= novel_target
                and 0 <= pool_points <= novel_points):
            raise ValueError("novel pool is not a subset of novel support")
        if not (0 <= prepool_target <= pool_target
                and 0 <= prepool_points <= min(pool_points, 768)):
            raise ValueError("768 pre-pool is not a subset of the novel pool")
        if not (0 <= selected_target <= prepool_target
                and 0 <= selected_points <= min(prepool_points, 256)):
            raise ValueError("256 selection is not a subset of the pre-pool")
        if not 0 <= inliers <= selected_points:
            raise ValueError("consensus inliers are not a selected-point subset")
        if value(row, "raw_target_bearing") != float(raw_target > 0):
            raise ValueError("raw target-bearing flag disagrees with count")
        if value(row, "novel_target_bearing") != float(novel_target > 0):
            raise ValueError("novel target-bearing flag disagrees with count")
        if value(row, "selected_target_bearing") != float(
                selected_target > 0):
            raise ValueError("selected target-bearing flag disagrees with count")
        for metric in (
                "relation_auroc", "relation_ap", "relation_auprc",
                "relation_ece", "vote_consistency", "vote_inlier_ratio"):
            if not 0.0 <= value(row, metric) <= 1.0:
                raise ValueError(f"{metric} must be in [0,1]")
        if abs(value(row, "relation_ap")
               - value(row, "relation_auprc")) > 1e-9:
            raise ValueError("relation AP/AUPRC aliases disagree")
        if abs(value(row, "vote_covariance_xy")) > math.sqrt(max(
                value(row, "vote_covariance_xx"), 0.0) * max(
                    value(row, "vote_covariance_yy"), 0.0)) + 1e-5:
            raise ValueError("reported vote covariance is not PSD")


def stratified(rows, key):
    if key == "gap_ratio":
        groups = {
            "gap_le_1": lambda row: value(row, key) <= 1.0,
            "gap_1_2": lambda row: 1.0 < value(row, key) <= 2.0,
            "gap_gt_2": lambda row: value(row, key) > 2.0,
        }
    elif key == "base_raw_point_count":
        groups = {
            "b0_raw_lt_64": lambda row: value(row, key) < 64,
            "b0_raw_ge_64": lambda row: value(row, key) >= 64,
        }
    else:
        groups = {
            "age_0_2": lambda row: 0 <= value(row, key) <= 2,
            "age_3_5": lambda row: 3 <= value(row, key) <= 5,
            "age_gt_5": lambda row: value(row, key) > 5,
        }
    return {name: aggregate([row for row in rows if predicate(row)])
            for name, predicate in groups.items()}


def build_report(rows):
    validate(rows)
    observable_need = [row for row in rows if (
        value(row, "global_target_count_label") > 0
        and value(row, "search_coverage_need") > 0)]
    target_prepool = [row for row in rows
                      if value(row, "prepool_target_count") > 0]
    selected = [row for row in rows
                if value(row, "router_applied_gate") > 0]
    counterfactuals = {}
    for arm in COUNTERFACTUAL_ARMS:
        valid_rows = [row for row in observable_need
                      if value(row, f"cf_{arm}_valid") > 0]
        counterfactuals[arm] = {
            "rows": len(valid_rows),
            "raw_target_recall": ratio(sum(
                value(row, f"cf_{arm}_raw_target_bearing") > 0
                for row in valid_rows), len(valid_rows)),
            "novel_target_recall": ratio(sum(
                value(row, f"cf_{arm}_novel_target_bearing") > 0
                for row in valid_rows), len(valid_rows)),
            "raw_background_mean": mean([
                value(row, f"cf_{arm}_support_raw_background_count")
                for row in valid_rows]) if valid_rows else 0.0,
            "novel_background_mean": mean([
                value(row, f"cf_{arm}_support_novel_background_count")
                for row in valid_rows]) if valid_rows else 0.0,
            "support_volume_mean": mean([
                value(row, f"cf_{arm}_support_volume")
                for row in valid_rows]) if valid_rows else 0.0,
            "truncation_rate": ratio(sum(
                value(row, f"cf_{arm}_truncated") > 0
                for row in valid_rows), len(valid_rows)),
            "sources": {
                source: int(any(value(
                    row, f"cf_{arm}_uses_{source}") > 0
                    for row in valid_rows))
                for source in ("endpoint", "tube", "corridor")},
        }

    selection_row_recall = ratio(sum(
        value(row, "selected_target_count") > 0
        for row in target_prepool), len(target_prepool))
    selection_point_recall = ratio(sum(
        value(row, "selected_target_count") for row in target_prepool), sum(
            value(row, "prepool_target_count") for row in target_prepool))
    novel_pool_need_recall = ratio(sum(
        value(row, "novel_pool_target_count") > 0
        for row in observable_need), len(observable_need))
    group_target_rates = {}
    for group in ("relation", "spatial", "exploration", "borrowed"):
        group_target_rates[group] = ratio(sum(
            value(row, f"selected_{group}_target_count") for row in rows),
            sum(value(row, f"selected_{group}_point_count") for row in rows))
    report = {
        "schema": "ct_seqtrack.b2_acquisition_report.v26",
        "rows": len(rows),
        "tracklets": len({row["tracklet_id"] for row in rows}),
        "funnel": aggregate(rows),
        "globally_observable_need": {
            "rows": len(observable_need),
            "novel_pool_target_bearing": novel_pool_need_recall,
        },
        "geometry": {
            "active_raw_target_recall": ratio(sum(
                value(row, "raw_target_bearing") > 0
                for row in observable_need), len(observable_need)),
            "active_novel_target_recall": ratio(sum(
                value(row, "novel_target_bearing") > 0
                for row in observable_need), len(observable_need)),
            "raw_background_mean": mean([
                value(row, "support_raw_background_count") for row in rows]),
            "novel_background_mean": mean([
                value(row, "support_novel_background_count") for row in rows]),
            "support_volume_mean": mean(
                value(row, "support_volume") for row in rows),
            "truncation_rate": ratio(sum(
                value(row, "active_support_truncated") > 0
                for row in rows), len(rows)),
            "novel_source_target_bearing": {
                source: ratio(sum(value(
                    row, f"{source}_support_novel_target_count") > 0
                    for row in observable_need), len(observable_need))
                for source in ("endpoint", "tube", "corridor")},
        },
        "selection": {
            "eligible_rows": len(target_prepool),
            "row_recall": selection_row_recall,
            "point_recall": selection_point_recall,
            "target_enrichment": ratio(sum(
                value(row, "selected_target_count") for row in rows) / max(
                    sum(value(row, "selected_point_count")
                        for row in rows), 1.0), sum(
                            value(row, "prepool_target_count")
                            for row in rows) / max(sum(
                                value(row, "prepool_point_count")
                                for row in rows), 1.0)),
            "group_target_rates": group_target_rates,
            "relation_auroc_mean": mean(
                value(row, "relation_auroc") for row in rows),
            "relation_ap_mean": mean(
                value(row, "relation_ap") for row in rows),
            "relation_auprc_mean": mean(
                value(row, "relation_auprc") for row in rows),
            "relation_ece_mean": mean(
                value(row, "relation_ece") for row in rows),
            "relation_vs_spatial_enrichment": ratio(
                group_target_rates["relation"],
                group_target_rates["spatial"]),
            "relation_vs_exploration_enrichment": ratio(
                group_target_rates["relation"],
                group_target_rates["exploration"]),
        },
        "voting": {
            "consistency_mean": mean(
                value(row, "vote_consistency") for row in rows),
            "covariance_trace_mean": mean(
                value(row, "vote_covariance_xx")
                + value(row, "vote_covariance_yy") for row in rows),
            "inlier_ratio_mean": mean(
                value(row, "vote_inlier_ratio") for row in rows),
            "effective_mass_mean": mean(
                value(row, "vote_effective_mass") for row in rows),
            "covariance_anisotropy_mean": mean(
                math.sqrt(
                    (value(row, "vote_covariance_xx")
                     - value(row, "vote_covariance_yy")) ** 2
                    + 4.0 * value(row, "vote_covariance_xy") ** 2)
                / max(value(row, "vote_covariance_xx")
                      + value(row, "vote_covariance_yy"), 1e-6)
                for row in rows),
            "candidate_margin_mean": mean(
                value(row, "vote_candidate_margin") for row in rows),
            "raw_center_gain_mean": mean(
                value(row, "observation_error")
                - value(row, "raw_search_error") for row in rows),
            "raw_iou_gain_mean": mean(
                value(row, "raw_search_iou")
                - value(row, "observation_iou") for row in rows),
            "raw_harm_rate": ratio(sum(
                value(row, "raw_search_error")
                > value(row, "observation_error")
                or value(row, "raw_search_iou")
                < value(row, "observation_iou") for row in rows), len(rows)),
            "oracle_center_headroom_mean": mean([
                value(row, "observation_error") for row in rows
                if value(row, "selected_target_count") > 0]
            ) if any(value(row, "selected_target_count") > 0
                     for row in rows) else 0.0,
            "oracle_iou_headroom_mean": mean([
                1.0 - value(row, "observation_iou") for row in rows
                if value(row, "selected_target_count") > 0]
            ) if any(value(row, "selected_target_count") > 0
                     for row in rows) else 0.0,
        },
        "b3": {
            "action_coverage": ratio(len(selected), len(rows)),
            "harmful_rate": ratio(sum(
                value(row, "center_gain") < 0
                or value(row, "iou_gain") < 0 for row in selected),
                len(selected)),
            "center_gain_mean": mean([
                value(row, "center_gain") for row in selected]
            ) if selected else 0.0,
            "iou_gain_mean": mean([
                value(row, "iou_gain") for row in selected]
            ) if selected else 0.0,
        },
        "counterfactual_geometry": counterfactuals,
        "strata": {
            "gap": stratified(rows, "gap_ratio"),
            "b0_core_sparsity": stratified(rows, "base_raw_point_count"),
            "recursive_age": stratified(rows, "recursive_age"),
        },
        "mechanism_thresholds": {
            "novel_pool_target_bearing_min": 0.15,
            "selection_row_recall_min": 0.90,
            "selection_point_recall_min": 0.70,
            "passed": bool(
                novel_pool_need_recall >= 0.15
                and selection_row_recall >= 0.90
                and selection_point_recall >= 0.70),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(load_rows(args.rows))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
