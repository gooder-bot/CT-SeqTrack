"""Paired-tracklet promotion gate for optional CT-SeqTrack memory."""

import hashlib
import json

import numpy as np


SCHEMA = "ct_seqtrack.memory_promotion.v1"


def _sha256_json(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def evaluate_memory_promotion(rows, checkpoint_sha256, seed=42,
                              resamples=2000, min_tracklets=30):
    """Require real memory to beat both controls with paired 95% CIs."""
    required_metrics = ("success", "precision")
    controls = ("empty", "time_misaligned")
    normalized = []
    seen = set()
    for source in rows:
        row = dict(source)
        tracklet_id = str(row.get("tracklet_id", ""))
        if not tracklet_id or tracklet_id in seen:
            raise ValueError(
                "memory promotion requires unique non-empty tracklet_id")
        seen.add(tracklet_id)
        item = {"tracklet_id": tracklet_id}
        for mode in ("real",) + controls:
            for metric in required_metrics:
                key = f"{mode}_{metric}"
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"memory promotion row lacks numeric {key}") from exc
                if not np.isfinite(value):
                    raise ValueError("memory promotion rows must be finite")
                item[key] = value
        normalized.append(item)
    if len(normalized) < int(min_tracklets):
        passed = False
    else:
        passed = True
    rng = np.random.default_rng(int(seed))
    comparisons = {}
    for control in controls:
        for metric in required_metrics:
            deltas = np.asarray([
                row[f"real_{metric}"] - row[f"{control}_{metric}"]
                for row in normalized], dtype=np.float64)
            if deltas.size:
                means = np.empty(int(resamples), dtype=np.float64)
                for index in range(int(resamples)):
                    picked = rng.integers(0, deltas.size, size=deltas.size)
                    means[index] = deltas[picked].mean()
                lower = float(np.quantile(means, 0.025))
                upper = float(np.quantile(means, 0.975))
                mean = float(deltas.mean())
            else:
                mean, lower, upper = 0.0, -1e30, 1e30
            key = f"real_vs_{control}_{metric}"
            comparisons[key] = {
                "mean_delta": mean,
                "paired_ci_95": [lower, upper],
                "passed": lower > 0.0,
            }
            passed = passed and lower > 0.0
    checkpoint_sha256 = dict(checkpoint_sha256)
    if set(checkpoint_sha256) != {"real", "empty", "time_misaligned"}:
        raise ValueError(
            "memory promotion requires real/empty/time_misaligned checkpoints")
    artifact = {
        "schema": SCHEMA,
        "passed": bool(passed),
        "tracklets": len(normalized),
        "requirements": {
            "min_tracklets": int(min_tracklets),
            "paired_ci": 0.95,
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
        },
        "checkpoint_sha256": checkpoint_sha256,
        "comparisons": comparisons,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact
