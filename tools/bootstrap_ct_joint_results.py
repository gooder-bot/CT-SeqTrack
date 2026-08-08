#!/usr/bin/env python3
"""Tracklet-paired bootstrap for CT Joint Success/Precision endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_endpoints(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame_id", "final_iou", "final_distance"}
    if (not rows or not required.issubset(rows[0])
            or not ({"tracklet_id", "tracklet_key"} & set(rows[0]))):
        raise ValueError(
            f"{path} must contain {sorted(required)} and a tracklet id/key")
    result = {}
    for row in rows:
        tracklet = row.get("tracklet_key") or row.get("tracklet_id")
        key = (str(tracklet), int(row["frame_id"]))
        if key in result:
            raise ValueError(f"duplicate endpoint {key} in {path}")
        result[key] = (
            float(row["final_iou"]), float(row["final_distance"]))
    return result


def tracking_metrics(overlaps, distances):
    overlaps = np.asarray(overlaps, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    success_thresholds = np.linspace(0.0, 1.0, 21)
    precision_thresholds = np.linspace(0.0, 2.0, 21)
    success_curve = np.asarray([
        np.mean(overlaps >= threshold) for threshold in success_thresholds])
    precision_curve = np.asarray([
        np.mean(distances <= threshold)
        for threshold in precision_thresholds])
    return {
        "success": float(np.trapz(
            success_curve, x=success_thresholds) * 100.0),
        "precision": float(np.trapz(
            precision_curve, x=precision_thresholds) * 50.0),
    }


def paired_bootstrap(baseline, repaired, draws=20000, seed=42):
    if set(baseline) != set(repaired):
        missing = sorted(set(baseline).symmetric_difference(repaired))
        raise ValueError(
            "paired endpoint keys differ: " + repr(missing[:5]))
    tracklets = sorted({key[0] for key in baseline})
    if not tracklets:
        raise ValueError("paired bootstrap contains no tracklets")
    by_tracklet = {}
    for tracklet in tracklets:
        keys = sorted(
            (key for key in baseline if key[0] == tracklet),
            key=lambda key: key[1])
        # Legacy proposal diagnostics start at frame 1.  New metric exports
        # contain every endpoint, so add the identical initialization only
        # when frame 0 is genuinely absent.
        initial = [] if any(key[1] == 0 for key in keys) else [(1.0, 0.0)]
        baseline_values = initial + [baseline[key] for key in keys]
        repaired_values = initial + [repaired[key] for key in keys]
        by_tracklet[tracklet] = (
            np.asarray(baseline_values), np.asarray(repaired_values))

    def pooled(selection, branch):
        return np.concatenate([
            by_tracklet[tracklet][branch] for tracklet in selection], axis=0)

    baseline_all = pooled(tracklets, 0)
    repaired_all = pooled(tracklets, 1)
    baseline_metric = tracking_metrics(
        baseline_all[:, 0], baseline_all[:, 1])
    repaired_metric = tracking_metrics(
        repaired_all[:, 0], repaired_all[:, 1])
    rng = np.random.default_rng(int(seed))
    deltas = {"success": [], "precision": []}
    for _ in range(int(draws)):
        selection = rng.choice(tracklets, size=len(tracklets), replace=True)
        baseline_draw = pooled(selection, 0)
        repaired_draw = pooled(selection, 1)
        baseline_draw_metric = tracking_metrics(
            baseline_draw[:, 0], baseline_draw[:, 1])
        repaired_draw_metric = tracking_metrics(
            repaired_draw[:, 0], repaired_draw[:, 1])
        for metric in deltas:
            deltas[metric].append(
                repaired_draw_metric[metric] - baseline_draw_metric[metric])
    result = {
        "tracklet_count": len(tracklets),
        "endpoint_count_including_initial": int(len(baseline_all)),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "baseline": baseline_metric,
        "repaired": repaired_metric,
        "delta": {},
    }
    for metric, values in deltas.items():
        values = np.asarray(values, dtype=np.float64)
        result["delta"][metric] = {
            "point_estimate": repaired_metric[metric] - baseline_metric[metric],
            "bootstrap_mean": float(values.mean()),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "probability_positive": float(np.mean(values > 0.0)),
        }
    return result


def paired_bootstrap_multiseed(
        baselines, repaired_models, draws=20000, seed=42):
    """Bootstrap the mean metric across seeds by paired tracklet resampling."""
    if len(baselines) != len(repaired_models) or not baselines:
        raise ValueError("baseline/repaired seed lists must be non-empty/equal")
    endpoint_keys = set(baselines[0])
    for seed_index, (baseline, repaired) in enumerate(
            zip(baselines, repaired_models)):
        if set(baseline) != set(repaired):
            raise ValueError(
                f"paired endpoint keys differ for seed index {seed_index}")
        if set(baseline) != endpoint_keys:
            raise ValueError(
                "all seeds must evaluate the identical endpoint keys")
    tracklets = sorted({key[0] for key in endpoint_keys})
    if not tracklets:
        raise ValueError("paired bootstrap contains no tracklets")

    seed_rows = []
    for baseline, repaired in zip(baselines, repaired_models):
        by_tracklet = {}
        for tracklet in tracklets:
            keys = sorted(
                (key for key in endpoint_keys if key[0] == tracklet),
                key=lambda key: key[1])
            initial = (
                [] if any(key[1] == 0 for key in keys)
                else [(1.0, 0.0)])
            by_tracklet[tracklet] = (
                np.asarray(initial + [baseline[key] for key in keys]),
                np.asarray(initial + [repaired[key] for key in keys]),
            )
        seed_rows.append(by_tracklet)

    def metrics_for(seed_row, selection, branch):
        values = np.concatenate([
            seed_row[tracklet][branch] for tracklet in selection], axis=0)
        return tracking_metrics(values[:, 0], values[:, 1])

    per_seed = []
    for seed_row in seed_rows:
        baseline_metric = metrics_for(seed_row, tracklets, 0)
        repaired_metric = metrics_for(seed_row, tracklets, 1)
        per_seed.append({
            "baseline": baseline_metric,
            "repaired": repaired_metric,
            "delta": {
                metric: repaired_metric[metric] - baseline_metric[metric]
                for metric in ("success", "precision")},
        })
    baseline_mean = {
        metric: float(np.mean([
            row["baseline"][metric] for row in per_seed]))
        for metric in ("success", "precision")}
    repaired_mean = {
        metric: float(np.mean([
            row["repaired"][metric] for row in per_seed]))
        for metric in ("success", "precision")}

    rng = np.random.default_rng(int(seed))
    deltas = {"success": [], "precision": []}
    for _ in range(int(draws)):
        selection = rng.choice(tracklets, size=len(tracklets), replace=True)
        seed_deltas = {"success": [], "precision": []}
        for seed_row in seed_rows:
            baseline_metric = metrics_for(seed_row, selection, 0)
            repaired_metric = metrics_for(seed_row, selection, 1)
            for metric in seed_deltas:
                seed_deltas[metric].append(
                    repaired_metric[metric] - baseline_metric[metric])
        for metric in deltas:
            deltas[metric].append(float(np.mean(seed_deltas[metric])))

    result = {
        "seed_count": len(seed_rows),
        "tracklet_count": len(tracklets),
        "endpoint_count_per_seed_including_initial": int(sum(
            len(seed_rows[0][tracklet][0]) for tracklet in tracklets)),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "baseline_mean": baseline_mean,
        "repaired_mean": repaired_mean,
        "per_seed": per_seed,
        "delta": {},
    }
    for metric, values in deltas.items():
        values = np.asarray(values, dtype=np.float64)
        result["delta"][metric] = {
            "point_estimate": (
                repaired_mean[metric] - baseline_mean[metric]),
            "bootstrap_mean": float(values.mean()),
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "probability_positive": float(np.mean(values > 0.0)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path, nargs="+")
    parser.add_argument("--repaired", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    if args.output.exists():
        raise FileExistsError(f"bootstrap output already exists: {args.output}")
    if len(args.baseline) != len(args.repaired):
        raise ValueError("--baseline/--repaired must contain equal seed counts")
    baselines = [read_endpoints(path) for path in args.baseline]
    repaired_models = [read_endpoints(path) for path in args.repaired]
    result = (
        paired_bootstrap(
            baselines[0], repaired_models[0],
            draws=args.draws, seed=args.seed)
        if len(baselines) == 1 else
        paired_bootstrap_multiseed(
            baselines, repaired_models,
            draws=args.draws, seed=args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
