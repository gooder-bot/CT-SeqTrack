#!/usr/bin/env python3
"""Compare matched GRU/CfC B1-only endpoint exports and apply promotion gates."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ctseqtrack.runtime.scene_bootstrap import tracking_metrics
from tools.report_ct_b1 import build_report, load_rows


def _row_key(row, include_candidate):
    partition = row.get("partition_group_key", row.get("scene_id", ""))
    tracklet = row.get("tracklet_key", row.get("tracklet_id", ""))
    if partition == "" and tracklet == "":
        raise ValueError("comparison rows require a scene or tracklet identity")
    key = (str(partition), str(tracklet), int(float(row["frame_id"])))
    if include_candidate:
        key += (int(float(row.get("candidate_id", 0))),)
    return key


def _aligned_rows(first, second, include_candidate):
    first_map = {_row_key(row, include_candidate): row for row in first}
    second_map = {_row_key(row, include_candidate): row for row in second}
    if len(first_map) != len(first) or len(second_map) != len(second):
        raise ValueError("comparison inputs contain duplicate endpoint identities")
    if set(first_map) != set(second_map):
        missing_first = sorted(set(second_map) - set(first_map))[:5]
        missing_second = sorted(set(first_map) - set(second_map))[:5]
        raise ValueError(
            "GRU/CfC endpoint identities differ: "
            f"missing_gru={missing_first}, missing_cfc={missing_second}"
        )
    keys = sorted(first_map)
    return [first_map[key] for key in keys], [second_map[key] for key in keys]


def _tracking_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _tracking_summary(rows):
    return {"frames": len(rows), **tracking_metrics(rows)}


def _last_nonempty(strata):
    nonempty = [row for row in strata if int(row.get("count", 0)) > 0]
    if not nonempty:
        raise ValueError("time-gap report contains no valid stratum")
    return nonempty[-1]


def compare(gru_proposals, cfc_proposals, gru_tracking, cfc_tracking, tolerance=1e-6):
    gru_proposals, cfc_proposals = _aligned_rows(
        gru_proposals, cfc_proposals, include_candidate=True
    )
    gru_tracking, cfc_tracking = _aligned_rows(
        gru_tracking, cfc_tracking, include_candidate=False
    )
    gru_report = build_report(gru_proposals)
    cfc_report = build_report(cfc_proposals)
    gru_overall = gru_report["overall"]
    cfc_overall = cfc_report["overall"]
    gru_long = _last_nonempty(gru_report["strata"]["time_gap"])
    cfc_long = _last_nonempty(cfc_report["strata"]["time_gap"])
    gru_track = _tracking_summary(gru_tracking)
    cfc_track = _tracking_summary(cfc_tracking)

    sampled_improved = (
        "sampled_target_count" in gru_overall
        and "sampled_target_count" in cfc_overall
        and cfc_overall["sampled_target_count"]
        > gru_overall["sampled_target_count"]
    )
    support_improved = (
        cfc_overall["target_in_support_recall"]
        > gru_overall["target_in_support_recall"]
        or sampled_improved
    )
    volume_tolerance = max(
        float(tolerance), abs(gru_overall["support_volume"]) * float(tolerance)
    )
    gates = {
        "cfc_rmse_better_than_gru": (
            cfc_overall["learned_rmse"] < gru_overall["learned_rmse"]
        ),
        "cfc_rmse_better_than_cv": (
            cfc_overall["learned_rmse"] < cfc_overall["cv_rmse"]
        ),
        "cfc_long_gap_rmse_better_than_gru": (
            cfc_long["learned_rmse"] < gru_long["learned_rmse"]
        ),
        "cfc_nll_not_worse": cfc_overall["nll"] <= gru_overall["nll"],
        "cfc_coverage_ece_not_worse": (
            cfc_overall["coverage_ece"] <= gru_overall["coverage_ece"]
        ),
        "cfc_coverage95_not_worse": (
            cfc_overall["coverage"]["95"] >= gru_overall["coverage"]["95"]
        ),
        "cfc_support_evidence_improved": support_improved,
        "cfc_support_volume_not_larger": (
            cfc_overall["support_volume"]
            <= gru_overall["support_volume"] + volume_tolerance
        ),
    }
    tracking_delta = {
        key: cfc_track[key] - gru_track[key] for key in ("success", "precision")
    }
    tracking_isolated = all(
        abs(value) <= float(tolerance) for value in tracking_delta.values()
    )
    metric_gates_passed = bool(tracking_isolated and all(gates.values()))
    return {
        "schema": "ct_seqtrack.b1_backbone_comparison.v1",
        "aligned_proposal_rows": len(gru_proposals),
        "aligned_tracking_rows": len(gru_tracking),
        "gru": {"b1": gru_report, "tracking": gru_track},
        "cfc": {"b1": cfc_report, "tracking": cfc_track},
        "tracking_delta_cfc_minus_gru": tracking_delta,
        "tracking_isolation_passed": tracking_isolated,
        "b0_hash_audit_required": True,
        "promotion_gates": gates,
        "metric_gates_passed": metric_gates_passed,
        "promoted_to_full_screen": None,
        "promotion_status": (
            "pending_b0_hash_audit" if metric_gates_passed else "metric_gates_failed"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru-proposals", required=True)
    parser.add_argument("--cfc-proposals", required=True)
    parser.add_argument("--gru-tracking", required=True)
    parser.add_argument("--cfc-tracking", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = compare(
        load_rows(args.gru_proposals),
        load_rows(args.cfc_proposals),
        _tracking_rows(args.gru_tracking),
        _tracking_rows(args.cfc_tracking),
        tolerance=args.tolerance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
