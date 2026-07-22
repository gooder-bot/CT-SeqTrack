#!/usr/bin/env python3
"""Freeze the first formal M2 alpha/radius rule from the existing M0-3 oracle.

This tool intentionally evaluates exactly one predeclared engineering rule.  It
does not expose a sweep or optimizer: using the same mini_train oracle to search
many bounds would turn the E6 freeze into retrospective hyper-parameter tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_ENDPOINTS_SHA256 = (
    "aa2e890a5fcc3e15964bb89d87dc8c7873b0c97a29f437c220b8cd00e406099b"
)
EXPECTED_SUMMARY_SHA256 = (
    "2ecd6e707ffee6e6551effadb7c896f974988064579174d48ad8e7686ecf367a"
)

FROZEN_ALPHA = 0.75
FROZEN_RADIUS_BASE_M = 0.5
FROZEN_RADIUS_PER_SECOND_M = 0.5
FROZEN_RADIUS_MAX_M = 2.0
FROZEN_WARMUP_EPOCHS = 5

MIN_SAMPLES = 100
MIN_TRACKLETS = 20
MIN_MEAN_GAIN_M = 0.05
USEFUL_GAIN_M = 0.05
MIN_USEFUL_GAIN_RATE = 0.15
MIN_LONG_GAP_SAMPLES = 30
LONG_GAP_THRESHOLD_S = 1.0
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "max": float(values.max()),
    }


def load_primary_rows(path: Path) -> list[dict[str, object]]:
    required = {
        "tracklet_id",
        "candidate_id",
        "resampled",
        "full_history",
        "crop_reachable",
        "dynamics_valid",
        "current_delta_t",
        "target_x",
        "target_y",
        "target_z",
        "observation_x",
        "observation_y",
        "observation_z",
        "dynamics_x",
        "dynamics_y",
        "dynamics_z",
        "observation_error",
        "innovation_norm",
        "alpha_oracle",
        "oracle_error",
    }
    selected: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Oracle CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            if int(row["candidate_id"]) != 0:
                continue
            if parse_bool(row["resampled"]):
                continue
            if not parse_bool(row["full_history"]):
                continue
            if not parse_bool(row["crop_reachable"]):
                continue
            if not parse_bool(row["dynamics_valid"]):
                continue
            selected.append(
                {
                    "tracklet_id": int(row["tracklet_id"]),
                    "dt": float(row["current_delta_t"]),
                    "target": np.asarray(
                        [row["target_x"], row["target_y"], row["target_z"]],
                        dtype=np.float64,
                    ),
                    "observation": np.asarray(
                        [
                            row["observation_x"],
                            row["observation_y"],
                            row["observation_z"],
                        ],
                        dtype=np.float64,
                    ),
                    "dynamics": np.asarray(
                        [row["dynamics_x"], row["dynamics_y"], row["dynamics_z"]],
                        dtype=np.float64,
                    ),
                    "stored_observation_error": float(row["observation_error"]),
                    "stored_innovation_norm": float(row["innovation_norm"]),
                    "alpha_oracle": float(row["alpha_oracle"]),
                    "oracle_error": float(row["oracle_error"]),
                }
            )
    if not selected:
        raise ValueError("No rows passed the frozen primary filter")
    return selected


def tracklet_bootstrap(
    tracklet_ids: np.ndarray,
    values: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for tracklet_id, value in zip(tracklet_ids, values):
        grouped[int(tracklet_id)].append(float(value))
    tracklet_means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        tracklet_means,
        size=(int(iterations), int(tracklet_means.size)),
        replace=True,
    )
    ci = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    return {
        "tracklet_count": int(tracklet_means.size),
        "equal_weighted_mean_gain_m": float(tracklet_means.mean()),
        "ci95_m": [float(ci[0]), float(ci[1])],
        "negative_mean_gain_rate": float(np.mean(tracklet_means < 0.0)),
        "mean_gain_distribution_m": distribution(tracklet_means),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def subset_metrics(
    mask: np.ndarray,
    observation_error: np.ndarray,
    rule_error: np.ndarray,
    gain: np.ndarray,
    clamp: np.ndarray,
) -> dict[str, object]:
    selected_gain = gain[mask]
    return {
        "sample_count": int(mask.sum()),
        "observation_error_m": distribution(observation_error[mask]),
        "frozen_rule_error_m": distribution(rule_error[mask]),
        "gain_m": distribution(selected_gain),
        "positive_gain_rate": float(np.mean(selected_gain > 0.0)),
        "useful_gain_rate": float(np.mean(selected_gain >= USEFUL_GAIN_M)),
        "clamp_rate": float(np.mean(clamp[mask])),
    }


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    tracklet_ids = np.asarray([row["tracklet_id"] for row in rows], dtype=np.int64)
    dt = np.asarray([row["dt"] for row in rows], dtype=np.float64)
    target = np.stack([row["target"] for row in rows]).astype(np.float64)
    observation = np.stack([row["observation"] for row in rows]).astype(np.float64)
    dynamics = np.stack([row["dynamics"] for row in rows]).astype(np.float64)
    alpha_oracle = np.asarray([row["alpha_oracle"] for row in rows], dtype=np.float64)
    oracle_error = np.asarray([row["oracle_error"] for row in rows], dtype=np.float64)

    arrays = (dt, target, observation, dynamics, alpha_oracle, oracle_error)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Primary oracle rows contain non-finite values")
    if np.any(dt <= 0.0):
        raise ValueError("Primary oracle rows contain non-positive current_delta_t")

    innovation = dynamics - observation
    innovation_norm = np.linalg.norm(innovation, axis=1)
    observation_error = np.linalg.norm(observation - target, axis=1)
    stored_observation_error = np.asarray(
        [row["stored_observation_error"] for row in rows], dtype=np.float64
    )
    stored_innovation_norm = np.asarray(
        [row["stored_innovation_norm"] for row in rows], dtype=np.float64
    )
    consistency = {
        "observation_error_max_abs_diff": float(
            np.max(np.abs(observation_error - stored_observation_error))
        ),
        "innovation_norm_max_abs_diff": float(
            np.max(np.abs(innovation_norm - stored_innovation_norm))
        ),
    }
    if consistency["observation_error_max_abs_diff"] > 1e-5:
        raise ValueError(f"Stored observation error mismatch: {consistency}")
    if consistency["innovation_norm_max_abs_diff"] > 1e-5:
        raise ValueError(f"Stored innovation norm mismatch: {consistency}")

    radius = np.minimum(
        FROZEN_RADIUS_BASE_M + FROZEN_RADIUS_PER_SECOND_M * dt,
        FROZEN_RADIUS_MAX_M,
    )
    scale = np.minimum(1.0, radius / np.maximum(innovation_norm, 1e-12))
    clamped_innovation = innovation * scale[:, None]
    frozen_prediction = observation + FROZEN_ALPHA * clamped_innovation
    frozen_error = np.linalg.norm(frozen_prediction - target, axis=1)
    gain = observation_error - frozen_error
    clamp = innovation_norm > radius + 1e-12

    unbounded_prediction = observation + FROZEN_ALPHA * innovation
    unbounded_error = np.linalg.norm(unbounded_prediction - target, axis=1)
    all_mask = np.ones(len(rows), dtype=bool)
    long_gap_mask = dt >= LONG_GAP_THRESHOLD_S
    bootstrap = tracklet_bootstrap(tracklet_ids, gain)

    primary = subset_metrics(
        all_mask, observation_error, frozen_error, gain, clamp
    )
    long_gap = subset_metrics(
        long_gap_mask, observation_error, frozen_error, gain, clamp
    )
    primary["tracklet_count"] = int(np.unique(tracklet_ids).size)
    long_gap["tracklet_count"] = int(np.unique(tracklet_ids[long_gap_mask]).size)

    checks = {
        "minimum_samples": primary["sample_count"] >= MIN_SAMPLES,
        "minimum_tracklets": primary["tracklet_count"] >= MIN_TRACKLETS,
        "mean_gain": primary["gain_m"]["mean"] >= MIN_MEAN_GAIN_M,
        "useful_gain_rate": primary["useful_gain_rate"] >= MIN_USEFUL_GAIN_RATE,
        "tracklet_bootstrap_lower_positive": bootstrap["ci95_m"][0] > 0.0,
        "long_gap_supported": long_gap["sample_count"] >= MIN_LONG_GAP_SAMPLES,
        "long_gap_mean_gain_positive": long_gap["gain_m"]["mean"] > 0.0,
    }
    return {
        "frozen_parameters": {
            "dynamics_innovation_alpha": FROZEN_ALPHA,
            "dynamics_innovation_radius_base": FROZEN_RADIUS_BASE_M,
            "dynamics_innovation_radius_per_second": FROZEN_RADIUS_PER_SECOND_M,
            "dynamics_innovation_radius_max": FROZEN_RADIUS_MAX_M,
            "physical_time_adapter_warmup_epoch": FROZEN_WARMUP_EPOCHS,
            "dynamics_innovation_warmup_epoch": FROZEN_WARMUP_EPOCHS,
            "radius_formula": "min(0.5 + 0.5 * current_delta_t, 2.0)",
        },
        "primary_filter": {
            "candidate_id": 0,
            "resampled": False,
            "full_history": True,
            "crop_reachable": True,
            "dynamics_valid": True,
        },
        "primary": primary,
        "long_gap_dt_ge_1s": long_gap,
        "tracklet_bootstrap_gain": bootstrap,
        "diagnostics_not_used_for_tuning": {
            "alpha_oracle": distribution(alpha_oracle),
            "innovation_norm_m": distribution(innovation_norm),
            "radius_m": distribution(radius),
            "applied_correction_bound_m": distribution(FROZEN_ALPHA * radius),
            "unbounded_alpha_0_75_error_m": distribution(unbounded_error),
            "oracle_error_m": distribution(oracle_error),
            "clipping_error_minus_unbounded_error_m": distribution(
                frozen_error - unbounded_error
            ),
        },
        "raw_vector_consistency": consistency,
        "preregistered_acceptance": {
            "min_samples": MIN_SAMPLES,
            "min_tracklets": MIN_TRACKLETS,
            "min_mean_gain_m": MIN_MEAN_GAIN_M,
            "useful_gain_m": USEFUL_GAIN_M,
            "min_useful_gain_rate": MIN_USEFUL_GAIN_RATE,
            "long_gap_threshold_s": LONG_GAP_THRESHOLD_S,
            "min_long_gap_samples": MIN_LONG_GAP_SAMPLES,
        },
        "checks": checks,
        "decision": (
            "FREEZE_M2_ALPHA_RADIUS" if all(checks.values()) else "HOLD_M2_ALPHA_RADIUS"
        ),
    }


def validate_m0_summary(path: Path, analysis: dict[str, object]) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema") != "ct_seqtrack_m0_proposal_oracle_v1":
        raise ValueError(f"Unexpected M0-3 summary schema: {summary.get('schema')!r}")
    if summary.get("decision") != "GO_M2_PROPOSAL_INNOVATION":
        raise ValueError(f"M0-3 decision is not GO: {summary.get('decision')!r}")
    expected_count = int(summary["primary"]["sample_count"])
    actual_count = int(analysis["primary"]["sample_count"])
    if expected_count != actual_count:
        raise ValueError(
            f"Primary count disagrees with M0-3 summary: {actual_count} != {expected_count}"
        )
    return {
        "schema": summary["schema"],
        "decision": summary["decision"],
        "primary_sample_count": expected_count,
        "primary_tracklet_count": int(summary["primary"]["tracklet_count"]),
    }


def render_markdown(payload: dict[str, object]) -> str:
    primary = payload["primary"]
    long_gap = payload["long_gap_dt_ge_1s"]
    bootstrap = payload["tracklet_bootstrap_gain"]
    params = payload["frozen_parameters"]
    diagnostics = payload["diagnostics_not_used_for_tuning"]
    checks = payload["checks"]
    lines = [
        "# M2 E6 formal 参数冻结",
        "",
        f"决定：**`{payload['decision']}`**",
        "",
        "本报告只验证进入工程门禁前已经声明的单一规则；未执行 alpha、scale、gate 或半径网格搜索。",
        "",
        "## 冻结值",
        "",
        "```yaml",
        f"dynamics_innovation_alpha: {params['dynamics_innovation_alpha']}",
        f"dynamics_innovation_radius_base: {params['dynamics_innovation_radius_base']}",
        "dynamics_innovation_radius_per_second: "
        f"{params['dynamics_innovation_radius_per_second']}",
        f"dynamics_innovation_radius_max: {params['dynamics_innovation_radius_max']}",
        "physical_time_adapter_warmup_epoch: "
        f"{params['physical_time_adapter_warmup_epoch']}",
        "dynamics_innovation_warmup_epoch: "
        f"{params['dynamics_innovation_warmup_epoch']}",
        "```",
        "",
        "## mini_train primary 复算",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| endpoint | {primary['sample_count']} |",
        f"| tracklet | {primary['tracklet_count']} |",
        f"| observation error mean | {primary['observation_error_m']['mean']:.6f} m |",
        f"| frozen-rule error mean | {primary['frozen_rule_error_m']['mean']:.6f} m |",
        f"| mean gain | {primary['gain_m']['mean']:.6f} m |",
        f"| median gain | {primary['gain_m']['p50']:.6f} m |",
        f"| gain >= 0.05 m | {primary['useful_gain_rate']:.6%} |",
        f"| positive gain | {primary['positive_gain_rate']:.6%} |",
        f"| clamp rate | {primary['clamp_rate']:.6%} |",
        "| tracklet-equal mean gain | "
        f"{bootstrap['equal_weighted_mean_gain_m']:.6f} m |",
        "| tracklet bootstrap 95% CI | "
        f"[{bootstrap['ci95_m'][0]:.6f}, {bootstrap['ci95_m'][1]:.6f}] m |",
        f"| long-gap endpoint | {long_gap['sample_count']} |",
        f"| long-gap mean gain | {long_gap['gain_m']['mean']:.6f} m |",
        "",
        "## 边界",
        "",
        "当前安全半径会牺牲一部分未裁剪 oracle 空间：",
        f"unbounded alpha=0.75 error mean 为 {diagnostics['unbounded_alpha_0_75_error_m']['mean']:.6f} m，",
        f"frozen-rule error mean 为 {primary['frozen_rule_error_m']['mean']:.6f} m。",
        "该差异只作为保守性诊断，不用于在同一 oracle 上放大半径。",
        "",
        "## 硬检查",
        "",
    ]
    lines.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in checks.items())
    lines.extend(
        [
            "",
            "输入 SHA256：",
            "",
            f"- endpoints: `{payload['inputs']['endpoints']['sha256']}`",
            f"- summary: `{payload['inputs']['summary']['sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    rows: list[dict[str, object]] = []
    for tracklet_id in range(24):
        for endpoint in range(5):
            target = np.asarray([1.0 + 0.1 * endpoint, 0.0, 0.0])
            observation = np.asarray([0.0, 0.0, 0.0])
            dynamics = target.copy()
            rows.append(
                {
                    "tracklet_id": tracklet_id,
                    "dt": 0.5 if endpoint < 3 else 1.5,
                    "target": target,
                    "observation": observation,
                    "dynamics": dynamics,
                    "stored_observation_error": float(np.linalg.norm(target)),
                    "stored_innovation_norm": float(np.linalg.norm(target)),
                    "alpha_oracle": 1.0,
                    "oracle_error": 0.0,
                }
            )
    result = analyze(rows)
    assert result["decision"] == "FREEZE_M2_ALPHA_RADIUS"
    assert result["primary"]["sample_count"] == 120
    assert result["long_gap_dt_ge_1s"]["sample_count"] == 48
    print("M2 formal parameter freeze self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints")
    parser.add_argument("--summary")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument(
        "--expected-endpoints-sha256", default=EXPECTED_ENDPOINTS_SHA256
    )
    parser.add_argument("--expected-summary-sha256", default=EXPECTED_SUMMARY_SHA256)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    required = (args.endpoints, args.summary, args.output_json, args.output_md)
    if any(value is None for value in required):
        parser.error(
            "--endpoints, --summary, --output-json and --output-md are required"
        )

    endpoints_path = Path(args.endpoints)
    summary_path = Path(args.summary)
    for path in (endpoints_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    endpoint_sha = sha256_file(endpoints_path)
    summary_sha = sha256_file(summary_path)
    if args.expected_endpoints_sha256 and endpoint_sha != args.expected_endpoints_sha256:
        raise RuntimeError(
            "M0-3 endpoint SHA256 mismatch: "
            f"expected {args.expected_endpoints_sha256}, got {endpoint_sha}"
        )
    if args.expected_summary_sha256 and summary_sha != args.expected_summary_sha256:
        raise RuntimeError(
            "M0-3 summary SHA256 mismatch: "
            f"expected {args.expected_summary_sha256}, got {summary_sha}"
        )

    analysis = analyze(load_primary_rows(endpoints_path))
    m0_summary = validate_m0_summary(summary_path, analysis)
    payload = {
        "schema": "ct_seqtrack.m2_formal_parameter_freeze",
        "schema_version": 1,
        "scope": "existing mini_train M0-3 primary rows; one predeclared rule; no sweep",
        "inputs": {
            "endpoints": {"path": endpoints_path.as_posix(), "sha256": endpoint_sha},
            "summary": {"path": summary_path.as_posix(), "sha256": summary_sha},
            "validated_m0_summary": m0_summary,
        },
        **analysis,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with output_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_markdown(payload))

    print(payload["decision"])
    print(f"json: {output_json}")
    print(f"markdown: {output_md}")
    if payload["decision"] != "FREEZE_M2_ALPHA_RADIUS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
