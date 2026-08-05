#!/usr/bin/env python3
"""Apply the preregistered raw-Search B2 promotion protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.selective_innovation_common import torch_load  # noqa: E402
from utils.replay_cache import sha256_file  # noqa: E402


MODES = (
    "observation", "motion", "raw_search", "legacy_clipped", "selective")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--frame-diagnostics", required=True)
    parser.add_argument(
        "--metrics", required=True,
        help="same-checkpoint five-mode metrics and paired bootstrap JSON")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def finite_number(mapping, key):
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing/non-numeric metric: {key}") from error
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric: {key}")
    return value


def binary_auc(label, score):
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positive = int(label.sum())
    negative = int((~label).sum())
    if positive == 0 or negative == 0:
        raise RuntimeError("presence AUROC needs both classes")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return float((ranks[label].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def average_precision(label, score):
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    sorted_label = label[order]
    positive = int(sorted_label.sum())
    if positive == 0:
        raise RuntimeError("presence AUPRC needs positive rows")
    sorted_score = score[order]
    true_positive = 0
    predicted = 0
    ap = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_score[end] == sorted_score[start]:
            end += 1
        group_positive = int(sorted_label[start:end].sum())
        true_positive += group_positive
        predicted += end - start
        ap += (group_positive / positive) * (true_positive / predicted)
        start = end
    return float(ap)


def column(rows, name, aliases=(), allow_nonfinite=False):
    keys = (name, *aliases)
    for key in keys:
        if rows and key in rows[0]:
            values = np.asarray([float(row[key]) for row in rows])
            if not allow_nonfinite and not np.isfinite(values).all():
                raise ValueError(f"diagnostic column {key} is non-finite")
            return values
    raise ValueError(f"diagnostics lack {name}")


def main():
    args = parse_args()
    checkpoint_sha = sha256_file(args.candidate_checkpoint)
    checkpoint_payload = torch_load(args.candidate_checkpoint)
    checkpoint_config_sha = checkpoint_payload.get(
        "b2_v3_candidate_config_sha256")
    if not checkpoint_config_sha:
        raise RuntimeError(
            "candidate checkpoint lacks the B2 config contract hash")
    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get(
            "schema") != "ct_seqtrack.b2_v3_five_mode_metrics.v2":
        raise RuntimeError("promotion requires recomputed five-mode metrics v2")
    if metrics.get("candidate_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("five-mode metrics use a different checkpoint")
    if metrics.get("candidate_config_sha256") != checkpoint_config_sha:
        raise RuntimeError(
            "five-mode metrics use a different B2 candidate config")
    mode_metrics = metrics.get("modes", {})
    if set(mode_metrics) != set(MODES):
        raise ValueError("metrics must contain exactly the five formal modes")
    parsed_modes = {
        mode: {
            "success": finite_number(mode_metrics[mode], "success"),
            "precision": finite_number(mode_metrics[mode], "precision"),
        }
        for mode in MODES
    }
    oracle = metrics.get("oracle_obs_raw", {})
    oracle_success = finite_number(oracle, "success")
    oracle_precision = finite_number(oracle, "precision")
    observation = parsed_modes["observation"]
    oracle_gain = {
        "success": oracle_success - observation["success"],
        "precision": oracle_precision - observation["precision"],
    }

    diagnostics_path = Path(args.frame_diagnostics)
    if metrics.get("diagnostic_sha256", {}).get(
            "observation") != sha256_file(diagnostics_path):
        raise RuntimeError(
            "promotion diagnostics differ from the observation-mode evidence")
    with diagnostics_path.open(
            "r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise RuntimeError("frame diagnostics are empty")
    structural = column(rows, "geometry_valid") > 0.5
    presence_label = column(
        rows, "presence_target", aliases=("valid_foreground",)) > 0.5
    presence_score = column(rows, "presence_probability")
    evaluated = structural & np.isfinite(presence_score)
    if not np.any(evaluated):
        raise RuntimeError("no structurally valid presence rows")
    label = presence_label[evaluated]
    score = presence_score[evaluated]
    prevalence = float(label.mean())
    auroc = binary_auc(label, score)
    auprc = average_precision(label, score)
    foreground_valid = float(np.mean(column(rows, "valid_foreground") > 0.5))
    support_calibration = metrics.get("support_calibration", {})
    if (support_calibration.get("partition") != "calibration"
            or int(support_calibration.get("rows", 0)) <= 0
            or int(support_calibration.get("tracklets", 0)) <= 0):
        raise RuntimeError(
            "promotion lacks training-tracklet support calibration evidence")
    truncation_rate = finite_number(
        support_calibration, "truncation_rate")

    base = column(rows, "base_reachable")
    prior = column(rows, "prior_reachable")
    gap = column(rows, "gap_ratio")
    previous = column(rows, "previous_error", allow_nonfinite=True)
    finite_previous = np.isfinite(previous)
    high_previous_cut = (
        float(np.quantile(previous[finite_previous], 0.75))
        if np.any(finite_previous) else float("inf"))
    strata = {
        "gap": gap > 1.0 + 1e-6,
        "high_previous_error": finite_previous
        & (previous >= high_previous_cut),
    }
    reachability = {}
    reachability_pass = True
    for name, mask in strata.items():
        if not np.any(mask):
            reachability[name] = {"rows": 0, "passed": False}
            reachability_pass = False
            continue
        base_rate = float(np.mean(base[mask] > 0.5))
        prior_rate = float(np.mean(prior[mask] > 0.5))
        passed = prior_rate > base_rate
        reachability[name] = {
            "rows": int(mask.sum()), "base": base_rate,
            "prior": prior_rate, "gain": prior_rate - base_rate,
            "passed": bool(passed),
        }
        reachability_pass &= passed

    invariance = metrics.get("observation_invariance", {})
    forced_delta = finite_number(invariance, "forced_invalid_max_abs")
    shuffled_delta = finite_number(invariance, "shuffled_b1_max_abs")
    bootstrap = metrics.get("paired_tracklet_bootstrap", {})
    success_ci = bootstrap.get("oracle_success_gain_ci95")
    precision_ci = bootstrap.get("oracle_precision_gain_ci95")
    if (not isinstance(success_ci, list) or len(success_ci) != 2
            or not isinstance(precision_ci, list) or len(precision_ci) != 2):
        raise ValueError("paired bootstrap must provide two 95% intervals")
    bootstrap_direction = (
        (oracle_gain["success"] >= 0.5
         and float(success_ci[0]) >= 0.0)
        or (oracle_gain["precision"] >= 1.0
            and float(precision_ci[0]) >= 0.0))

    checks = {
        "oracle_headroom": bool(
            oracle_gain["success"] >= 0.5
            or oracle_gain["precision"] >= 1.0),
        "presence_auroc": bool(auroc >= 0.65),
        "presence_auprc": bool(auprc >= prevalence + 0.15),
        "foreground_valid": bool(foreground_valid >= 0.15),
        "support_truncation": bool(truncation_rate <= 0.01),
        "stratified_reachability": bool(reachability_pass),
        "observation_invariance": bool(
            forced_delta <= 1e-7 and shuffled_delta <= 1e-7),
        "bootstrap_direction": bool(bootstrap_direction),
    }
    result = {
        "schema": "ct_seqtrack.b2_v3_promotion.v2",
        "status": "passed" if all(checks.values()) else "failed",
        "candidate_checkpoint": str(Path(
            args.candidate_checkpoint).resolve()),
        "candidate_checkpoint_sha256": checkpoint_sha,
        "candidate_config_sha256": checkpoint_config_sha,
        "metrics_sha256": sha256_file(metrics_path),
        "frame_diagnostics_sha256": sha256_file(diagnostics_path),
        "checks": checks,
        "modes": parsed_modes,
        "oracle_obs_raw": {
            "success": oracle_success, "precision": oracle_precision,
            "gain": oracle_gain,
        },
        "presence": {
            "rows": int(evaluated.sum()), "prevalence": prevalence,
            "auroc": auroc, "auprc": auprc,
            "foreground_valid_rate": foreground_valid,
            "support_truncation_rate": truncation_rate,
        },
        "support_calibration": support_calibration,
        "reachability": reachability,
        "observation_invariance": {
            "forced_invalid_max_abs": forced_delta,
            "shuffled_b1_max_abs": shuffled_delta,
        },
        "paired_tracklet_bootstrap": bootstrap,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise RuntimeError("B2 raw-Search promotion gate failed")


if __name__ == "__main__":
    main()
