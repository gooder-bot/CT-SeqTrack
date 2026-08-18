"""Scene-paired tracking metrics and bootstrap intervals for v25."""

from __future__ import annotations

import numpy as np


def tracking_metrics(rows):
    if not rows:
        raise ValueError("tracking rows are empty")
    overlaps = np.asarray([float(row["final_iou"]) for row in rows])
    distances = np.asarray([float(row["final_distance"]) for row in rows])
    if not np.isfinite(overlaps).all() or not np.isfinite(distances).all():
        raise ValueError("tracking metrics must be finite")
    success_curve = np.asarray([
        np.mean(overlaps >= threshold)
        for threshold in np.linspace(0.0, 1.0, 21)])
    precision_curve = np.asarray([
        np.mean(distances <= threshold)
        for threshold in np.linspace(0.0, 2.0, 21)])
    return {
        "success": float(np.trapz(
            success_curve, x=np.linspace(0.0, 1.0, 21)) * 100.0),
        "precision": float(np.trapz(
            precision_curve, x=np.linspace(0.0, 2.0, 21)) * 50.0),
    }


def _index_rows(rows):
    indexed = {}
    for source in rows:
        row = dict(source)
        required = (
            "partition_group_key", "tracklet_key", "frame_id",
            "final_iou", "final_distance")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError("tracking row missing: " + ", ".join(missing))
        identity = (
            str(row["partition_group_key"]), str(row["tracklet_key"]),
            int(row["frame_id"]))
        if identity in indexed:
            raise ValueError(f"duplicate tracking endpoint: {identity}")
        indexed[identity] = row
    return indexed


def paired_scene_bootstrap(
        baseline_rows, method_rows, seed=42, resamples=20000):
    """Pair endpoints exactly, then resample physical scenes."""
    baseline = _index_rows(baseline_rows)
    method = _index_rows(method_rows)
    if set(baseline) != set(method):
        missing_baseline = len(set(method) - set(baseline))
        missing_method = len(set(baseline) - set(method))
        raise ValueError(
            "paired endpoint identity mismatch: "
            f"baseline_missing={missing_baseline}, "
            f"method_missing={missing_method}")
    scenes = sorted({key[0] for key in baseline})
    if not scenes:
        raise ValueError("paired tracking rows contain no scenes")
    identities_by_scene = {
        scene: [identity for identity in sorted(baseline)
                if identity[0] == scene]
        for scene in scenes}
    baseline_metrics = tracking_metrics(list(baseline.values()))
    method_metrics = tracking_metrics(list(method.values()))
    observed = {
        key: method_metrics[key] - baseline_metrics[key]
        for key in ("success", "precision")}
    rng = np.random.default_rng(int(seed))
    draws = {"success": [], "precision": []}
    for _ in range(int(resamples)):
        sampled_scenes = rng.choice(
            scenes, size=len(scenes), replace=True)
        baseline_sample = []
        method_sample = []
        for scene in sampled_scenes:
            for identity in identities_by_scene[str(scene)]:
                baseline_sample.append(baseline[identity])
                method_sample.append(method[identity])
        base_metric = tracking_metrics(baseline_sample)
        method_metric = tracking_metrics(method_sample)
        for key in draws:
            draws[key].append(method_metric[key] - base_metric[key])
    intervals = {
        key: {
            "delta": observed[key],
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in draws.items()}
    return {
        "schema": "ct_seqtrack.scene_paired_bootstrap.v1",
        "sampling_unit": "scene",
        "scene_count": len(scenes),
        "endpoint_count": len(baseline),
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "baseline": baseline_metrics,
        "method": method_metrics,
        "paired_delta": intervals,
    }
