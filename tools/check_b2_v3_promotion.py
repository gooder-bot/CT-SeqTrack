#!/usr/bin/env python3
"""Apply the preregistered Seed42 B2-v3 promotion gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v2 import SELECTIVE_V3_ROUTER_SCHEMA  # noqa: E402
from tools.selective_innovation_common import torch_load  # noqa: E402


MODES = ("obs_only", "obs_vs_motion", "obs_vs_refined", "obs_vs_all")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", required=True)
    parser.add_argument(
        "--candidate-diagnostics", required=True,
        help="epoch20/refiner candidate CSV or packaged mini_val CSV")
    parser.add_argument(
        "--metrics", required=True,
        help="JSON containing seed, four mode metrics, and mini_val routing")
    parser.add_argument("--output")
    return parser.parse_args()


def require_number(mapping, key):
    value = float(mapping[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric: {key}")
    return value


def main():
    args = parse_args()
    router = torch_load(args.router)
    if router.get("schema") != SELECTIVE_V3_ROUTER_SCHEMA:
        raise ValueError("router is not a B2-v3 sidecar")
    calibration = router.get("calibration", {})
    with Path(args.metrics).open("r", encoding="utf-8") as input_file:
        metrics = json.load(input_file)
    if int(metrics.get("seed", -1)) != 42:
        raise ValueError("the first promotion gate is fixed to seed42")
    mode_metrics = metrics.get("modes", {})
    if sorted(mode_metrics) != sorted(MODES):
        raise ValueError("metrics JSON must contain exactly four B2-v3 modes")
    parsed_modes = {
        mode: {
            "success": require_number(mode_metrics[mode], "success"),
            "precision": require_number(mode_metrics[mode], "precision"),
        }
        for mode in MODES
    }
    mini_val = metrics.get("mini_val", {})
    mini_precision = require_number(mini_val, "helpful_precision")
    mini_harm = require_number(mini_val, "harm_rate")

    with Path(args.candidate_diagnostics).open(
            "r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    valid_rows = [
        row for row in rows if int(float(row.get("valid_foreground", 0))) == 1]
    if not valid_rows:
        raise RuntimeError("candidate diagnostics contain no valid foreground")
    candidate_rmse = {
        "motion": float(np.sqrt(np.mean([
            float(row["motion_error"]) ** 2 for row in valid_rows]))),
        "raw_search": float(np.sqrt(np.mean([
            float(row["raw_search_error"]) ** 2
            for row in valid_rows]))),
        "refined": float(np.sqrt(np.mean([
            float(row["search_error"]) ** 2 for row in valid_rows]))),
        "count": len(valid_rows),
    }
    all_mode = parsed_modes["obs_vs_all"]
    obs_mode = parsed_modes["obs_only"]
    candidate_pass = (
        candidate_rmse["refined"] < candidate_rmse["motion"]
        and candidate_rmse["refined"] < candidate_rmse["raw_search"])
    calibration_pass = (
        calibration.get("status") == "passed"
        and calibration.get("partition") == "calibration"
        and float(calibration.get("helpful_precision", 0.0)) >= 0.75
        and float(calibration.get("harm_rate", 1.0)) <= 0.10
        and 0.05 <= float(calibration.get("coverage", -1.0)) <= 0.25
        and int(calibration.get("selected_count", 0)) >= 100)
    mini_val_pass = mini_precision >= 0.70 and mini_harm <= 0.10
    leaderboard_pass = (
        all_mode["success"] > 54.132
        and all_mode["precision"] > 64.755
        and all_mode["success"] >= parsed_modes[
            "obs_vs_motion"]["success"]
        and all_mode["precision"] >= parsed_modes[
            "obs_vs_motion"]["precision"]
        and all_mode["success"] >= parsed_modes[
            "obs_vs_refined"]["success"]
        and all_mode["precision"] >= parsed_modes[
            "obs_vs_refined"]["precision"]
        and all_mode["success"] - obs_mode["success"] >= 0.5
        and all_mode["precision"] - obs_mode["precision"] >= 1.0)
    checks = {
        "candidate_quality": candidate_pass,
        "calibration": calibration_pass,
        "mini_val_routing": mini_val_pass,
        "leaderboard": leaderboard_pass,
    }
    result = {
        "schema": "ct_seqtrack.b2_v3_promotion.v1",
        "seed": 42,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "candidate_rmse_valid_foreground": candidate_rmse,
        "calibration": calibration,
        "mini_val": mini_val,
        "modes": parsed_modes,
        "obs_vs_all_gain": {
            "success": all_mode["success"] - obs_mode["success"],
            "precision": all_mode["precision"] - obs_mode["precision"],
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise RuntimeError("B2-v3 Seed42 promotion gate failed")


if __name__ == "__main__":
    main()
