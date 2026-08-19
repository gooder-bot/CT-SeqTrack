#!/usr/bin/env python3
"""Fit B1 parallel/perpendicular scale buffers on a tracklet calibration split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctseqtrack.runtime.calibration import sha256_file


CHI2_THRESHOLDS = {"50": 1.38629436112, "80": 3.21887582487,
                   "95": 5.99146454711}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="NPZ with error_xy, velocity_xy, log_sigma_pp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--min-direction-speed", type=float, default=0.2)
    return parser.parse_args()


def aligned_errors(error_xy, direction_xy, min_speed=0.2):
    error_xy = np.asarray(error_xy, dtype=np.float64)
    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-12)
    perpendicular = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
    return np.stack((
        np.sum(error_xy * direction, axis=1),
        np.sum(error_xy * perpendicular, axis=1),
    ), axis=1)


def gaussian_nll(error, log_sigma):
    return float(np.mean(np.sum(0.5 * (
        error ** 2 * np.exp(-2.0 * log_sigma) + 2.0 * log_sigma), axis=1)))


def metrics(error, log_sigma):
    normalized_sq = np.sum((error * np.exp(-log_sigma)) ** 2, axis=1)
    coverage = {
        label: float(np.mean(normalized_sq <= threshold))
        for label, threshold in CHI2_THRESHOLDS.items()
    }
    nominal = {"50": 0.50, "80": 0.80, "95": 0.95}
    return {
        "nll": gaussian_nll(error, log_sigma),
        "coverage": coverage,
        "coverage_ece": float(np.mean([
            abs(coverage[key] - nominal[key]) for key in nominal])),
    }


def fit_calibration(arrays, min_direction_speed=0.2):
    error_xy = arrays["error_xy"]
    # ``direction_xy`` is the formal v2 contract.  The velocity fallback is
    # retained only for unit-level compatibility with historical fixtures;
    # the CLI rejects v1 artifacts before this function is called.
    direction_xy = (
        arrays["direction_xy"] if "direction_xy" in arrays
        else arrays["velocity_xy"])
    log_sigma = np.asarray(arrays["log_sigma_pp"], dtype=np.float64)
    valid = np.asarray(
        arrays["valid"] if "valid" in arrays
        else np.ones(len(error_xy)), dtype=bool)
    finite = (
        np.isfinite(error_xy).all(axis=1)
        & np.isfinite(direction_xy).all(axis=1)
        & np.isfinite(log_sigma).all(axis=1))
    valid &= finite
    kinematic_error_xy = (
        arrays["kinematic_error_xy"]
        if "kinematic_error_xy" in arrays else None)
    if kinematic_error_xy is not None:
        kinematic_error_xy = np.asarray(
            kinematic_error_xy, dtype=np.float64)
        if kinematic_error_xy.shape != np.asarray(error_xy).shape:
            raise ValueError(
                "kinematic_error_xy must match learned error_xy")
        valid &= np.isfinite(kinematic_error_xy).all(axis=1)
    if int(valid.sum()) < 10:
        raise RuntimeError("B1 calibration requires at least ten valid rows")
    aligned = aligned_errors(
        np.asarray(error_xy)[valid], np.asarray(direction_xy)[valid],
        min_speed=min_direction_speed)
    base_log_sigma = np.clip(log_sigma[valid], -4.0, 2.5)
    standardized_mse = np.mean(
        aligned ** 2 * np.exp(-2.0 * base_log_sigma), axis=0)
    log_scale = 0.5 * np.log(np.maximum(standardized_mse, 1e-12))
    calibrated = np.clip(base_log_sigma + log_scale[None, :], -4.0, 2.5)
    standardized_after_scale = np.abs(aligned) * np.exp(-calibrated)
    fixed_sigma = np.sqrt(np.maximum(np.mean(aligned ** 2, axis=0), 1e-12))
    fixed_log_sigma = np.broadcast_to(
        np.log(fixed_sigma)[None, :], calibrated.shape)
    result = {
        "valid_rows": int(valid.sum()),
        "log_scale_parallel_perpendicular": log_scale.tolist(),
        "scale_parallel_perpendicular": np.exp(log_scale).tolist(),
        "fixed_margin_parallel_perpendicular_95": np.quantile(
            np.abs(aligned), 0.95, axis=0).tolist(),
        "standardized_abs_residual_q90_parallel_perpendicular": np.quantile(
            standardized_after_scale, 0.90, axis=0).tolist(),
        "uncalibrated": metrics(aligned, base_log_sigma),
        "calibrated": metrics(aligned, calibrated),
        "fixed_sigma_baseline": metrics(aligned, fixed_log_sigma),
        "promotion": {},
        "gap_strata": {},
    }
    learned_mean_rmse = float(np.sqrt(np.mean(np.sum(
        np.asarray(error_xy, dtype=np.float64)[valid] ** 2, axis=1))))
    kinematic_rmse = (
        float(np.sqrt(np.mean(np.sum(
            kinematic_error_xy[valid] ** 2, axis=1))))
        if kinematic_error_xy is not None else None)
    result["mean_prediction"] = {
        "learned_rmse": learned_mean_rmse,
        "kinematic_rmse": kinematic_rmse,
    }
    learned_beats_kinematic = bool(
        kinematic_rmse is not None and learned_mean_rmse < kinematic_rmse)
    promoted = (
        result["calibrated"]["coverage_ece"] <= 0.05
        and result["calibrated"]["coverage"]["95"] >= 0.90
        and result["calibrated"]["nll"]
        < result["fixed_sigma_baseline"]["nll"]
        and learned_beats_kinematic)
    result["promotion"] = {
        "passed": bool(promoted),
        "criteria": {
            "coverage_ece_le_0.05": bool(
                result["calibrated"]["coverage_ece"] <= 0.05),
            "coverage95_ge_0.90": bool(
                result["calibrated"]["coverage"]["95"] >= 0.90),
            "nll_better_than_fixed_sigma": bool(
                result["calibrated"]["nll"]
                < result["fixed_sigma_baseline"]["nll"]),
            "learned_mean_beats_kinematic": learned_beats_kinematic,
        },
    }
    if "gap_ratio" in arrays:
        gaps = np.asarray(arrays["gap_ratio"])[valid]
        for gap in (1.0, 2.0, 4.0):
            mask = np.isclose(gaps, gap)
            if np.any(mask):
                result["gap_strata"][str(int(gap))] = metrics(
                    aligned[mask], calibrated[mask])
    return result


def update_checkpoint(checkpoint_path, output_path, result):
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("state_dict", payload)
    matching = [key for key in state if key.endswith(
        "physical_motion_encoder.log_sigma_calibration")]
    if len(matching) != 1:
        raise RuntimeError(
            "checkpoint must contain exactly one B1 calibration buffer")
    state[matching[0]] = torch.as_tensor(
        result["log_scale_parallel_perpendicular"],
        dtype=state[matching[0]].dtype)
    if isinstance(payload, dict):
        payload["b1_uncertainty_calibration"] = dict(result)
        # Calibration is a post-selection evaluation transform, not an epoch
        # boundary of the optimizer trajectory.  It must never be resumed as
        # if it were the original scratch run.
        payload["ct_posthoc_calibrated"] = True
        payload["ct_epoch_boundary_complete"] = False
    torch.save(payload, output_path)


def load_and_validate_manifest(input_path, checkpoint_path=None):
    input_path = Path(input_path)
    manifest_path = input_path.with_suffix(
        input_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError("B1 calibration manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ct_seqtrack.b1_calibration.v2":
        raise RuntimeError("formal B1 calibration rejects non-v2 artifacts")
    if manifest.get("partition") != "calibration":
        raise RuntimeError(
            "B1 scale calibration is restricted to calibration tracklets")
    if sha256_file(input_path) != manifest.get("artifact_sha256"):
        raise RuntimeError("B1 residual artifact SHA256 mismatch")
    if checkpoint_path is not None:
        actual = sha256_file(checkpoint_path)
        if actual != manifest.get("checkpoint_sha256"):
            raise RuntimeError(
                "calibration may only update the checkpoint that generated "
                "the residual artifact")
    for key in (
            "dataset", "split", "config_sha256", "b1_config_sha256",
            "checkpoint_sha256"):
        if not manifest.get(key):
            raise RuntimeError(f"B1 calibration manifest lacks {key}")
    return manifest


def main():
    args = parse_args()
    if bool(args.checkpoint) != bool(args.output_checkpoint):
        raise ValueError(
            "--checkpoint and --output-checkpoint must be supplied together")
    manifest = load_and_validate_manifest(args.input, args.checkpoint)
    arrays = np.load(args.input, allow_pickle=False)
    for key in ("error_xy", "kinematic_error_xy", "direction_xy",
                "basis_velocity_xy",
                "velocity_xy", "log_sigma_pp", "valid", "gap_ratio"):
        if key not in arrays:
            raise RuntimeError(f"B1 v2 residual artifact lacks {key}")
    result = fit_calibration(
        arrays, min_direction_speed=args.min_direction_speed)
    result = {
        "schema": "ct_seqtrack.b1_uncertainty_calibration.v2",
        "source_artifact": manifest,
        **result,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if args.checkpoint:
        update_checkpoint(
            args.checkpoint, args.output_checkpoint,
            result)
    print(json.dumps(result["promotion"], sort_keys=True))


if __name__ == "__main__":
    main()
