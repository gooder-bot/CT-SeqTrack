#!/usr/bin/env python3
"""Recompute the formal five-mode B2 metrics from frame diagnostics."""

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
OBSERVATION_FIELDS = (
    "observation_x", "observation_y", "observation_z", "observation_yaw")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build reproducible B2-v3 five-mode metrics")
    parser.add_argument("--candidate-checkpoint", required=True)
    for mode in MODES:
        parser.add_argument(
            f"--{mode.replace('_', '-')}", required=True,
            help=f"{mode} proposal_diagnostics.csv")
    parser.add_argument("--forced-invalid", required=True)
    parser.add_argument("--shuffled-b1", required=True)
    parser.add_argument(
        "--support-calibration", required=True,
        help="mini_train diagnostics used only for support truncation")
    parser.add_argument("--calibration-split", default="mini_train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty frame diagnostics: {path}")
    required = {
        "tracklet_key", "frame_id", "final_iou", "final_distance",
        "candidate_config_sha256", *OBSERVATION_FIELDS,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            f"frame diagnostics {path} lack: {', '.join(missing)}")
    indexed = {}
    for row in rows:
        key = (str(row["tracklet_key"]), int(row["frame_id"]))
        if key in indexed:
            raise ValueError(f"duplicate frame diagnostic key: {key}")
        indexed[key] = row
    return indexed


def stable_tracklet_partition(tracklet_key, seed=42):
    import hashlib
    digest = hashlib.sha256(
        f"{int(seed)}::{str(tracklet_key)}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "calibration"


def validate_candidate_config(indexed, expected):
    observed = {str(row["candidate_config_sha256"]) for row in indexed.values()}
    if observed != {str(expected)}:
        raise RuntimeError("frame diagnostics use a different candidate config")


def ope_metrics(ious, distances):
    ious = np.asarray(ious, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if (ious.size == 0 or distances.size != ious.size
            or not np.isfinite(ious).all()
            or not np.isfinite(distances).all()):
        raise ValueError("tracking metric arrays are empty, misaligned, or non-finite")
    success_x = np.linspace(0.0, 1.0, 21)
    precision_x = np.linspace(0.0, 2.0, 21)
    success_curve = np.asarray([(ious >= x).mean() for x in success_x])
    precision_curve = np.asarray([
        (distances <= x).mean() for x in precision_x])
    return {
        "success": float(np.trapz(success_curve, success_x) * 100.0),
        "precision": float(
            np.trapz(precision_curve, precision_x) * 100.0 / 2.0),
        "frame_count": int(ious.size),
    }


def arrays_by_tracklet(indexed, *, oracle=False):
    grouped = {}
    for (tracklet_key, frame_id), row in sorted(indexed.items()):
        if oracle:
            for field in (
                    "observation_iou", "observation_distance",
                    "raw_search_iou", "raw_search_distance", "search_valid"):
                if field not in row:
                    raise ValueError(f"observation diagnostics lack {field}")
            observation_iou = float(row["observation_iou"])
            observation_distance = float(row["observation_distance"])
            raw_valid = float(row["search_valid"]) > 0.5
            raw_iou = float(row["raw_search_iou"])
            raw_distance = float(row["raw_search_distance"])
            iou = max(observation_iou, raw_iou) if raw_valid else observation_iou
            distance = (
                min(observation_distance, raw_distance)
                if raw_valid else observation_distance)
        else:
            iou = float(row["final_iou"])
            distance = float(row["final_distance"])
        grouped.setdefault(tracklet_key, []).append(
            (frame_id, iou, distance))
    result = {}
    for tracklet_key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        # Protocol initialization uses the GT box and is therefore exact.
        result[tracklet_key] = (
            np.asarray([1.0] + [item[1] for item in values]),
            np.asarray([0.0] + [item[2] for item in values]),
        )
    return result


def concatenate_tracklets(grouped, keys=None):
    keys = list(grouped) if keys is None else list(keys)
    return (
        np.concatenate([grouped[key][0] for key in keys]),
        np.concatenate([grouped[key][1] for key in keys]),
    )


def paired_tracklet_bootstrap(
        observation, oracle, samples=2000, seed=42):
    keys = sorted(observation)
    if keys != sorted(oracle) or not keys:
        raise ValueError("observation/oracle tracklet sets are empty or different")
    rng = np.random.default_rng(int(seed))
    success_gain = np.empty(int(samples), dtype=np.float64)
    precision_gain = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sampled = rng.choice(keys, size=len(keys), replace=True).tolist()
        obs_metric = ope_metrics(*concatenate_tracklets(observation, sampled))
        oracle_metric = ope_metrics(*concatenate_tracklets(oracle, sampled))
        success_gain[index] = oracle_metric["success"] - obs_metric["success"]
        precision_gain[index] = (
            oracle_metric["precision"] - obs_metric["precision"])
    return {
        "method": "paired_tracklet_bootstrap",
        "samples": int(samples),
        "seed": int(seed),
        "tracklets": len(keys),
        "oracle_success_gain_ci95": np.quantile(
            success_gain, (0.025, 0.975)).tolist(),
        "oracle_precision_gain_ci95": np.quantile(
            precision_gain, (0.025, 0.975)).tolist(),
    }


def observation_max_abs(reference, control):
    if set(reference) != set(control):
        raise ValueError("observation invariance diagnostics are not aligned")
    maximum = 0.0
    for key in reference:
        left = np.asarray([
            float(reference[key][field]) for field in OBSERVATION_FIELDS])
        right = np.asarray([
            float(control[key][field]) for field in OBSERVATION_FIELDS])
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("observation invariance values are non-finite")
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    checkpoint = torch_load(args.candidate_checkpoint)
    candidate_config_sha = checkpoint.get("b2_v3_candidate_config_sha256")
    if not candidate_config_sha:
        raise RuntimeError("candidate checkpoint lacks its B2 config hash")

    paths = {
        mode: Path(getattr(args, mode)) for mode in MODES}
    indexed = {mode: read_rows(path) for mode, path in paths.items()}
    for rows in indexed.values():
        validate_candidate_config(rows, candidate_config_sha)
    expected_keys = set(indexed["observation"])
    for mode in MODES[1:]:
        if set(indexed[mode]) != expected_keys:
            raise ValueError(f"{mode} diagnostics do not align with observation")

    grouped = {
        mode: arrays_by_tracklet(indexed[mode]) for mode in MODES}
    mode_metrics = {
        mode: ope_metrics(*concatenate_tracklets(grouped[mode]))
        for mode in MODES
    }
    oracle_grouped = arrays_by_tracklet(indexed["observation"], oracle=True)
    oracle_metric = ope_metrics(*concatenate_tracklets(oracle_grouped))
    bootstrap = paired_tracklet_bootstrap(
        grouped["observation"], oracle_grouped,
        samples=args.bootstrap_samples, seed=args.seed)

    forced = read_rows(args.forced_invalid)
    shuffled = read_rows(args.shuffled_b1)
    support_calibration = read_rows(args.support_calibration)
    for control in (forced, shuffled, support_calibration):
        validate_candidate_config(control, candidate_config_sha)
    calibration_rows = [
        row for (tracklet_key, _), row in support_calibration.items()
        if stable_tracklet_partition(tracklet_key, args.seed) == "calibration"]
    if not calibration_rows:
        raise RuntimeError("support diagnostics contain no calibration tracklets")
    splits = {row.get("dataset_split") for row in calibration_rows}
    if splits != {args.calibration_split}:
        raise RuntimeError(
            "support truncation must use the declared training split")
    if "support_truncated" not in calibration_rows[0]:
        raise ValueError("support calibration diagnostics lack truncation")
    support_calibration_result = {
        "dataset_split": args.calibration_split,
        "partition": "calibration",
        "rows": len(calibration_rows),
        "tracklets": len({
            str(row["tracklet_key"]) for row in calibration_rows}),
        "truncation_rate": float(np.mean([
            float(row["support_truncated"]) > 0.5
            for row in calibration_rows])),
        "diagnostics_sha256": sha256_file(args.support_calibration),
    }
    invariance = {
        "forced_invalid_max_abs": observation_max_abs(
            indexed["observation"], forced),
        "shuffled_b1_max_abs": observation_max_abs(
            indexed["observation"], shuffled),
    }
    payload = {
        "schema": "ct_seqtrack.b2_v3_five_mode_metrics.v2",
        "candidate_checkpoint_sha256": sha256_file(
            args.candidate_checkpoint),
        "candidate_config_sha256": candidate_config_sha,
        "modes": mode_metrics,
        "oracle_obs_raw": oracle_metric,
        "observation_invariance": invariance,
        "paired_tracklet_bootstrap": bootstrap,
        "support_calibration": support_calibration_result,
        "diagnostic_sha256": {
            **{mode: sha256_file(path) for mode, path in paths.items()},
            "forced_invalid": sha256_file(args.forced_invalid),
            "shuffled_b1": sha256_file(args.shuffled_b1),
            "support_calibration": sha256_file(args.support_calibration),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
