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

from utils.replay_cache import sha256_file, sha256_json


CHI2_THRESHOLDS = {"50": 1.38629436112, "80": 3.21887582487,
                   "95": 5.99146454711}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-input", "--input", dest="fit_input", required=True,
        help="calibration-partition NPZ used only to fit two scale values")
    parser.add_argument(
        "--eval-input", required=True,
        help="independent dev-partition NPZ used for promotion metrics")
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


def _prepared_arrays(
        arrays, min_direction_speed=0.2,
        log_sigma_min=-2.302585092994046, log_sigma_max=2.5):
    error_xy = arrays["error_xy"]
    # ``direction_xy`` is the formal contract.  The velocity fallback is
    # retained only for unit-level compatibility with historical fixtures;
    # the CLI rejects incomplete artifacts before this function is called.
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
    base_log_sigma = np.clip(
        log_sigma[valid], float(log_sigma_min), float(log_sigma_max))
    return {
        "error_xy": np.asarray(error_xy, dtype=np.float64)[valid],
        "kinematic_error_xy": (
            kinematic_error_xy[valid]
            if kinematic_error_xy is not None else None),
        "aligned": aligned,
        "base_log_sigma": base_log_sigma,
        "gap_ratio": (
            np.asarray(arrays["gap_ratio"])[valid]
            if "gap_ratio" in arrays else None),
        "tracklet_key": (
            np.asarray(arrays["tracklet_key"])[valid]
            if "tracklet_key" in arrays else None),
        "valid_rows": int(valid.sum()),
    }


def paired_tracklet_bootstrap_rmse_delta(
        learned_error_xy, kinematic_error_xy, tracklet_keys,
        samples=2000, seed=20260825):
    """Paired cluster bootstrap of learned-minus-CV RMSE by tracklet."""
    learned_sq = np.sum(
        np.asarray(learned_error_xy, dtype=np.float64) ** 2, axis=1)
    kinematic_sq = np.sum(
        np.asarray(kinematic_error_xy, dtype=np.float64) ** 2, axis=1)
    keys = np.asarray(tracklet_keys).astype(str)
    unique, inverse = np.unique(keys, return_inverse=True)
    if unique.size == 0:
        raise ValueError("paired bootstrap requires at least one tracklet")
    counts = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    learned_sums = np.bincount(
        inverse, weights=learned_sq, minlength=unique.size)
    kinematic_sums = np.bincount(
        inverse, weights=kinematic_sq, minlength=unique.size)
    point = float(
        np.sqrt(learned_sums.sum() / counts.sum())
        - np.sqrt(kinematic_sums.sum() / counts.sum()))
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        selected = rng.integers(0, unique.size, size=unique.size)
        denominator = counts[selected].sum()
        deltas[index] = (
            np.sqrt(learned_sums[selected].sum() / denominator)
            - np.sqrt(kinematic_sums[selected].sum() / denominator))
    return {
        "unit": "tracklet",
        "samples": int(samples),
        "tracklets": int(unique.size),
        "point": point,
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975))],
    }


def fit_calibration(
        arrays, min_direction_speed=0.2, evaluation_arrays=None,
        log_sigma_min=-2.302585092994046, log_sigma_max=2.5):
    """Fit on calibration rows and evaluate promotion on independent rows."""
    if not float(log_sigma_min) < float(log_sigma_max):
        raise ValueError("B1 calibration log-sigma bounds must be ordered")
    fitted = _prepared_arrays(
        arrays, min_direction_speed, log_sigma_min, log_sigma_max)
    evaluated = (
        fitted if evaluation_arrays is None
        else _prepared_arrays(
            evaluation_arrays, min_direction_speed,
            log_sigma_min, log_sigma_max))
    aligned_fit = fitted["aligned"]
    base_log_sigma_fit = fitted["base_log_sigma"]
    standardized_mse = np.mean(
        aligned_fit ** 2 * np.exp(-2.0 * base_log_sigma_fit), axis=0)
    log_scale = 0.5 * np.log(np.maximum(standardized_mse, 1e-12))
    calibrated_fit = np.clip(
        base_log_sigma_fit + log_scale[None, :],
        float(log_sigma_min), float(log_sigma_max))
    standardized_after_scale = (
        np.abs(aligned_fit) * np.exp(-calibrated_fit))

    aligned = evaluated["aligned"]
    base_log_sigma = evaluated["base_log_sigma"]
    calibrated = np.clip(
        base_log_sigma + log_scale[None, :],
        float(log_sigma_min), float(log_sigma_max))
    fixed_sigma = np.sqrt(np.maximum(
        np.mean(aligned_fit ** 2, axis=0), 1e-12))
    fixed_log_sigma = np.broadcast_to(
        np.log(fixed_sigma)[None, :], calibrated.shape)
    result = {
        "fit_valid_rows": fitted["valid_rows"],
        "valid_rows": evaluated["valid_rows"],
        "log_sigma_bounds": [
            float(log_sigma_min), float(log_sigma_max)],
        "log_scale_parallel_perpendicular": log_scale.tolist(),
        "scale_parallel_perpendicular": np.exp(log_scale).tolist(),
        "fixed_margin_parallel_perpendicular_95": np.quantile(
            np.abs(aligned_fit), 0.95, axis=0).tolist(),
        "standardized_abs_residual_q90_parallel_perpendicular": np.quantile(
            standardized_after_scale, 0.90, axis=0).tolist(),
        "fit_uncalibrated": metrics(aligned_fit, base_log_sigma_fit),
        "fit_calibrated": metrics(aligned_fit, calibrated_fit),
        "uncalibrated": metrics(aligned, base_log_sigma),
        "calibrated": metrics(aligned, calibrated),
        "fixed_sigma_baseline": metrics(aligned, fixed_log_sigma),
        "promotion": {},
        "gap_strata": {},
    }
    learned_mean_rmse = float(np.sqrt(np.mean(np.sum(
        evaluated["error_xy"] ** 2, axis=1))))
    kinematic_error_xy = evaluated["kinematic_error_xy"]
    kinematic_rmse = (
        float(np.sqrt(np.mean(np.sum(
            kinematic_error_xy ** 2, axis=1))))
        if kinematic_error_xy is not None else None)
    rmse_delta = (
        learned_mean_rmse - kinematic_rmse
        if kinematic_rmse is not None else None)
    paired_bootstrap = None
    if (kinematic_error_xy is not None
            and evaluated["tracklet_key"] is not None):
        paired_bootstrap = paired_tracklet_bootstrap_rmse_delta(
            evaluated["error_xy"], kinematic_error_xy,
            evaluated["tracklet_key"])
    result["mean_prediction"] = {
        "learned_rmse": learned_mean_rmse,
        "kinematic_rmse": kinematic_rmse,
        "learned_minus_kinematic_rmse": rmse_delta,
        "paired_tracklet_bootstrap": paired_bootstrap,
    }
    learned_beats_kinematic = bool(kinematic_rmse is not None and (
        paired_bootstrap["ci95"][1] < 0.0
        if paired_bootstrap is not None else rmse_delta < 0.0))
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
    if evaluated["gap_ratio"] is not None:
        gaps = evaluated["gap_ratio"]
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


def load_and_validate_manifest(
        input_path, checkpoint_path=None, expected_partition="calibration"):
    input_path = Path(input_path)
    manifest_path = input_path.with_suffix(
        input_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError("B1 calibration manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ct_seqtrack.b1_calibration.v3":
        raise RuntimeError("formal B1 calibration rejects non-v3 artifacts")
    if manifest.get("partition") != expected_partition:
        raise RuntimeError(
            "B1 residual artifact has the wrong atomic partition")
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
            "checkpoint_sha256", "tracklet_keys_sha256"):
        if not manifest.get(key):
            raise RuntimeError(f"B1 calibration manifest lacks {key}")
    b1_config = manifest.get("b1_config")
    if (not isinstance(b1_config, dict)
            or sha256_json(b1_config) != manifest.get("b1_config_sha256")):
        raise RuntimeError("B1 calibration config payload/hash mismatch")
    return manifest


def main():
    args = parse_args()
    if bool(args.checkpoint) != bool(args.output_checkpoint):
        raise ValueError(
            "--checkpoint and --output-checkpoint must be supplied together")
    fit_manifest = load_and_validate_manifest(
        args.fit_input, args.checkpoint, expected_partition="calibration")
    eval_manifest = load_and_validate_manifest(
        args.eval_input, args.checkpoint, expected_partition="dev")
    for key in (
            "dataset", "split", "config_sha256", "b1_config_sha256",
            "checkpoint_sha256", "seed"):
        if fit_manifest.get(key) != eval_manifest.get(key):
            raise RuntimeError(
                f"B1 fit/dev calibration identity mismatch: {key}")
    arrays = np.load(args.fit_input, allow_pickle=False)
    evaluation_arrays = np.load(args.eval_input, allow_pickle=False)
    for key in ("error_xy", "kinematic_error_xy", "direction_xy",
                "basis_velocity_xy",
                "velocity_xy", "log_sigma_pp", "valid", "gap_ratio",
                "tracklet_key", "recursive_age", "recursive_age_valid"):
        if key not in arrays or key not in evaluation_arrays:
            raise RuntimeError(f"B1 v3 residual artifact lacks {key}")
    fit_tracklets = sorted(set(
        np.asarray(arrays["tracklet_key"]).astype(str).tolist()))
    eval_tracklets = sorted(set(
        np.asarray(evaluation_arrays["tracklet_key"]).astype(str).tolist()))
    if sha256_json(fit_tracklets) != fit_manifest["tracklet_keys_sha256"]:
        raise RuntimeError("B1 calibration tracklet identity hash mismatch")
    if sha256_json(eval_tracklets) != eval_manifest["tracklet_keys_sha256"]:
        raise RuntimeError("B1 dev tracklet identity hash mismatch")
    overlap = sorted(set(fit_tracklets) & set(eval_tracklets))
    if overlap:
        raise RuntimeError(
            "B1 calibration/dev tracklet partitions overlap")
    result = fit_calibration(
        arrays, min_direction_speed=args.min_direction_speed,
        evaluation_arrays=evaluation_arrays,
        log_sigma_min=float(fit_manifest["b1_config"].get(
            "motion_v3_log_sigma_min", -2.302585092994046)),
        log_sigma_max=float(fit_manifest["b1_config"].get(
            "motion_v3_log_sigma_max", 2.5)))
    result = {
        "schema": "ct_seqtrack.b1_uncertainty_calibration.v3",
        "source_artifact": fit_manifest,
        "evaluation_artifact": eval_manifest,
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
