#!/usr/bin/env python3
"""Validate a zero-weight PFTC preflight and freeze its largest safe lambda."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


LAMBDA_CANDIDATES = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def read_scalar(version_dir: Path, directory_name: str) -> dict[int, float]:
    scalar_dir = version_dir / directory_name
    if not scalar_dir.is_dir():
        raise FileNotFoundError(f"missing TensorBoard scalar: {scalar_dir}")
    accumulator = event_accumulator.EventAccumulator(
        str(scalar_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if len(tags) != 1:
        raise ValueError(
            f"expected one scalar tag in {scalar_dir}, found {tags}")
    return {
        int(event.step): float(event.value)
        for event in accumulator.Scalars(tags[0])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version_dir",
        help="lightning_logs/version_N directory from a --preflight run")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--max-ratio", type=float, default=0.10)
    parser.add_argument("--min-valid-sample-rate", type=float, default=0.30)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")

    version_dir = Path(args.version_dir).resolve()
    supervised = read_scalar(version_dir, "loss_loss_total_sup")
    pftc_raw = read_scalar(version_dir, "loss_loss_pftc_raw")
    pftc_weighted = read_scalar(
        version_dir, "loss_loss_pftc_weighted")
    valid_rate = read_scalar(
        version_dir, "loss_pftc_valid_sample_ratio")
    common_steps = sorted(
        set(supervised)
        & set(pftc_raw)
        & set(pftc_weighted)
        & set(valid_rate))[:args.batches]
    if len(common_steps) < args.batches:
        raise RuntimeError(
            f"preflight is incomplete: {len(common_steps)}/{args.batches} "
            "common training batches")

    raw_ratios = []
    weighted_ratios = []
    for step in common_steps:
        denominator = supervised[step]
        if not math.isfinite(denominator) or denominator <= 0:
            raise RuntimeError(
                f"non-positive/non-finite supervised loss at step {step}")
        for name, numerator, target in (
                ("raw", pftc_raw[step], raw_ratios),
                ("weighted", pftc_weighted[step], weighted_ratios)):
            if not math.isfinite(numerator) or numerator < 0:
                raise RuntimeError(
                    f"negative/non-finite {name} PFTC loss at step {step}")
            target.append(numerator / denominator)

    mean_valid_rate = statistics.fmean(
        valid_rate[step] for step in common_steps)
    if mean_valid_rate < args.min_valid_sample_rate:
        recommendation = None
        status = "STOP_SPARSE_PFTC_SUPERVISION"
    else:
        recommendation = next((
            candidate for candidate in LAMBDA_CANDIDATES
            if max(
                statistics.median(
                    candidate * ratio for ratio in raw_ratios),
                statistics.median(
                    candidate * ratio for ratio in weighted_ratios),
            ) <= args.max_ratio + 1e-8
        ), None)
        status = (
            "PASS_PFTC_PREFLIGHT"
            if recommendation is not None
            else "STOP_PFTC_LOSS_TOO_LARGE")

    payload = {
        "schema": "ct_seqtrack_pftc_preflight_v1",
        "status": status,
        "version_dir": str(version_dir),
        "batch_count": len(common_steps),
        "step_first": common_steps[0],
        "step_last": common_steps[-1],
        "mean_valid_sample_rate": mean_valid_rate,
        "median_unscaled_raw_pftc_to_supervised": statistics.median(
            raw_ratios),
        "median_unscaled_weighted_pftc_to_supervised": statistics.median(
            weighted_ratios),
        "median_unscaled_pftc_to_supervised": max(
            statistics.median(raw_ratios),
            statistics.median(weighted_ratios)),
        "max_allowed_scaled_ratio": args.max_ratio,
        "lambda_candidates_descending": list(LAMBDA_CANDIDATES),
        "recommended_frozen_lambda": recommendation,
    }
    output_path = (
        Path(args.output).resolve()
        if args.output
        else version_dir / "pftc_preflight.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "PASS_PFTC_PREFLIGHT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
