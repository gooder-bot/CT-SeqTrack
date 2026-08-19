#!/usr/bin/env python3
"""Scene-held-out empirical RA-PMM quantile calibration (artifact only)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctseqtrack.runtime.calibration import sha256_file  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--level", type=float, default=0.95)
    return parser.parse_args()


def _manifest(path):
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"missing manifest for {path}")
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != result.get("artifact_sha256"):
        raise RuntimeError(f"artifact SHA mismatch for {path}")
    return result


def _aligned_absolute(error_xy, direction_xy):
    direction = np.asarray(direction_xy, dtype=np.float64)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    perpendicular = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
    error = np.asarray(error_xy, dtype=np.float64)
    return np.abs(
        np.stack(
            (
                np.sum(error * direction, axis=1),
                np.sum(error * perpendicular, axis=1),
            ),
            axis=1,
        )
    )


def _finite_sample_quantile(values, level):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        raise RuntimeError("calibration has no finite valid rows")
    rank = min(values.size - 1, max(0, math.ceil((values.size + 1) * level) - 1))
    return float(values[rank])


def _coverage(arrays, inflation):
    valid = np.asarray(arrays["valid"], dtype=bool)
    direction = arrays["direction_xy"]
    physical = _aligned_absolute(arrays["physical_error_xy"], direction)
    endpoint = _aligned_absolute(arrays["endpoint_error_xy"], direction)
    motion_q = np.asarray(arrays["motion_quantiles_pp"], dtype=np.float64) * inflation
    support_q = np.asarray(arrays["support_quantiles_pp"], dtype=np.float64) * inflation
    finite = (
        np.isfinite(physical).all(axis=1)
        & np.isfinite(endpoint).all(axis=1)
        & np.isfinite(motion_q).all(axis=(1, 2))
        & np.isfinite(support_q).all(axis=(1, 2))
    )
    valid &= finite
    if not np.any(valid):
        raise RuntimeError("audit has no valid finite rows")
    nominal = (0.50, 0.80, 0.95)
    result = {"rows": int(valid.sum()), "motion": {}, "support": {}}
    for index, label in enumerate(("50", "80", "95")):
        result["motion"][label] = float(
            np.mean(np.all(physical[valid] <= motion_q[valid, index], axis=1))
        )
        result["support"][label] = float(
            np.mean(np.all(endpoint[valid] <= support_q[valid, index], axis=1))
        )
    result["motion_ece"] = float(
        np.mean(
            [abs(result["motion"][str(int(level * 100))] - level) for level in nominal]
        )
    )
    result["support_ece"] = float(
        np.mean(
            [abs(result["support"][str(int(level * 100))] - level) for level in nominal]
        )
    )
    return result


def main():
    args = _arguments()
    if not 0.0 < args.level < 1.0:
        raise ValueError("level must lie in (0,1)")
    select_manifest = _manifest(args.select)
    audit_manifest = _manifest(args.audit)
    if select_manifest.get("partition") != "calibration_select":
        raise RuntimeError("selection artifact must use calibration_select scenes")
    if audit_manifest.get("partition") != "calibration_audit":
        raise RuntimeError("audit artifact must use calibration_audit scenes")
    if not select_manifest.get("scene_partition_identity_sha256") or not audit_manifest.get(
        "scene_partition_identity_sha256"
    ):
        raise RuntimeError("calibration artifacts lack scene partition identities")
    select = np.load(args.select, allow_pickle=False)
    audit = np.load(args.audit, allow_pickle=False)
    required = (
        "physical_error_xy",
        "endpoint_error_xy",
        "direction_xy",
        "motion_quantiles_pp",
        "support_quantiles_pp",
        "valid",
        "tracklet_key",
    )
    for name, arrays in (("select", select), ("audit", audit)):
        missing = [key for key in required if key not in arrays]
        if missing:
            raise RuntimeError(f"{name} artifact lacks: {', '.join(missing)}")
    select_scenes = set(map(str, select["tracklet_key"]))
    audit_scenes = set(map(str, audit["tracklet_key"]))
    overlap = select_scenes & audit_scenes
    if overlap:
        raise RuntimeError("calibration-select and audit populations overlap")

    valid = np.asarray(select["valid"], dtype=bool)
    physical = _aligned_absolute(select["physical_error_xy"], select["direction_xy"])
    endpoint = _aligned_absolute(select["endpoint_error_xy"], select["direction_xy"])
    motion_q95 = np.asarray(select["motion_quantiles_pp"], dtype=np.float64)[:, 2]
    support_q95 = np.asarray(select["support_quantiles_pp"], dtype=np.float64)[:, 2]
    nonconformity = np.maximum(
        np.max(physical / np.maximum(motion_q95, 1e-6), axis=1),
        np.max(endpoint / np.maximum(support_q95, 1e-6), axis=1),
    )
    valid &= np.isfinite(nonconformity)
    inflation = max(1.0, _finite_sample_quantile(nonconformity[valid], args.level))
    audit_metrics = _coverage(audit, inflation)
    result = {
        "schema": "ct_seqtrack.b1_quantile_calibration.v1",
        "method": "scene-held-out empirical calibration",
        "level": args.level,
        "global_inflation": inflation,
        "selection_rows": int(valid.sum()),
        "selection_artifact_sha256": sha256_file(args.select),
        "audit_artifact_sha256": sha256_file(args.audit),
        "selection_manifest": select_manifest,
        "audit_manifest": audit_manifest,
        "audit": audit_metrics,
        "promotion": {
            "passed": bool(
                0.90 <= audit_metrics["support"]["95"] <= 0.98
                and audit_metrics["support_ece"] <= 0.05
            )
        },
        "produces_checkpoint": False,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["promotion"], sort_keys=True))


if __name__ == "__main__":
    main()
