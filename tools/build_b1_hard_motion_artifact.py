#!/usr/bin/env python3
"""Build checkpoint-free GT-only acceleration-equivalent hard-motion stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctseqtrack.runtime.calibration import sha256_file, sha256_json  # noqa: E402
from datasets import get_dataset  # noqa: E402
from datasets.misc_utils import normalize_timestamp  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--path")
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--preloading", action="store_true")
    return parser.parse_args()


def _gap(first, second, default):
    first = normalize_timestamp(first)
    second = normalize_timestamp(second)
    if first is None or second is None:
        return float(default)
    value = float(first - second)
    return value if np.isfinite(value) and value > 0 else float(default)


def main():
    args = _arguments()
    if "train" not in args.split.lower() or any(
        token in args.split.lower() for token in ("dev", "val", "test")
    ):
        raise RuntimeError("hard-motion artifact may read only a training split")
    raw_config = load_yaml_config(args.config)
    raw_config.update({"test_split": args.split, "preloading": bool(args.preloading)})
    if args.path:
        raw_config["path"] = args.path
    config = EasyDict(raw_config)
    dataset = get_dataset(config, type="test", split=args.split, protocol_role="test")
    limit = len(dataset)
    if args.max_tracklets is not None:
        limit = min(limit, int(args.max_tracklets))
    default_dt = float(getattr(config, "default_time_step", 0.5))
    records = []
    gaps = []
    for tracklet_id in range(limit):
        sequence = dataset[tracklet_id]
        for frame_id in range(2, len(sequence)):
            current = sequence[frame_id]
            previous = sequence[frame_id - 1]
            older = sequence[frame_id - 2]
            current_dt = _gap(
                current.get("timestamp"), previous.get("timestamp"), default_dt
            )
            previous_dt = _gap(
                previous.get("timestamp"), older.get("timestamp"), default_dt
            )
            current_displacement = np.asarray(
                current["3d_bbox"].center[:2], dtype=np.float64
            ) - np.asarray(previous["3d_bbox"].center[:2], dtype=np.float64)
            previous_velocity = (
                np.asarray(previous["3d_bbox"].center[:2], dtype=np.float64)
                - np.asarray(older["3d_bbox"].center[:2], dtype=np.float64)
            ) / previous_dt
            cv_error = float(
                np.linalg.norm(current_displacement - previous_velocity * current_dt)
            )
            expert_errors = None
            if frame_id >= 3:
                oldest = sequence[frame_id - 3]
                older_dt = _gap(
                    older.get("timestamp"), oldest.get("timestamp"), default_dt
                )
                base_velocity = previous_velocity
                older_velocity = (
                    np.asarray(older["3d_bbox"].center[:2], dtype=np.float64)
                    - np.asarray(oldest["3d_bbox"].center[:2], dtype=np.float64)
                ) / older_dt
                acceleration = (base_velocity - older_velocity) / max(
                    0.5 * (previous_dt + older_dt), 1e-3
                )
                acceleration_norm = float(np.linalg.norm(acceleration))
                if acceleration_norm > 8.0:
                    acceleration *= 8.0 / acceleration_norm
                cv_endpoint = base_velocity * current_dt
                ca_endpoint = cv_endpoint + 0.5 * acceleration * current_dt**2
                base_speed = float(np.linalg.norm(base_velocity))
                if base_speed >= 0.2 and float(np.linalg.norm(older_velocity)) >= 0.2:
                    base_angle = float(np.arctan2(base_velocity[1], base_velocity[0]))
                    older_angle = float(
                        np.arctan2(older_velocity[1], older_velocity[0])
                    )
                    angle_delta = float(
                        np.arctan2(
                            np.sin(base_angle - older_angle),
                            np.cos(base_angle - older_angle),
                        )
                    )
                    turn_rate = float(
                        np.clip(
                            angle_delta / max(0.5 * (previous_dt + older_dt), 1e-3),
                            -1.0,
                            1.0,
                        )
                    )
                    theta = turn_rate * current_dt
                    direction = base_velocity / base_speed
                    perpendicular = np.asarray((-direction[1], direction[0]))
                    if abs(turn_rate) > 1e-6:
                        ctrv_endpoint = (
                            direction * base_speed * np.sin(theta) / turn_rate
                            + perpendicular
                            * base_speed
                            * (1.0 - np.cos(theta))
                            / turn_rate
                        )
                    else:
                        ctrv_endpoint = cv_endpoint
                else:
                    ctrv_endpoint = cv_endpoint
                fixed_endpoint = 0.5 * cv_endpoint + 0.5 * ca_endpoint
                errors = np.asarray(
                    [
                        np.linalg.norm(current_displacement - cv_endpoint),
                        np.linalg.norm(current_displacement - fixed_endpoint),
                        np.linalg.norm(current_displacement - ca_endpoint),
                        np.linalg.norm(current_displacement - ctrv_endpoint),
                    ]
                )
                expert_errors = np.concatenate((errors, [errors[[0, 2, 3]].min()]))
            records.append((cv_error, current_dt, expert_errors))
            gaps.append(current_dt)
    if not records:
        raise RuntimeError("hard-motion preflight produced no valid GT transitions")
    gaps = np.asarray(gaps, dtype=np.float64)
    dt_floor = max(float(np.quantile(gaps, 0.05)), 0.05)
    difficulty = np.asarray(
        [2.0 * error / max(dt, dt_floor) ** 2 for error, dt, _ in records],
        dtype=np.float64,
    )
    category = str(getattr(config, "category_name", "unknown"))
    expert_rows = np.asarray(
        [errors for _, _, errors in records if errors is not None], dtype=np.float64
    )
    expert_preflight = None
    if len(expert_rows):
        aligned_difficulty = difficulty[
            np.asarray([errors is not None for _, _, errors in records], dtype=bool)
        ]
        hard_mask = aligned_difficulty >= np.quantile(aligned_difficulty, 0.80)

        def rmse(values):
            return float(np.sqrt(np.mean(np.asarray(values) ** 2)))

        names = ("cv", "fixed_cv_ca", "ca", "ctrv", "oracle_cv_ca_ctrv")
        expert_preflight = {
            "overall_rmse": {
                name: rmse(expert_rows[:, index]) for index, name in enumerate(names)
            },
            "hard_q80_q100_rmse": {
                name: rmse(expert_rows[hard_mask, index])
                for index, name in enumerate(names)
            },
        }
        baseline = expert_preflight["hard_q80_q100_rmse"]["fixed_cv_ca"]
        oracle = expert_preflight["hard_q80_q100_rmse"]["oracle_cv_ca_ctrv"]
        expert_preflight["hard_oracle_relative_potential"] = float(
            (baseline - oracle) / max(baseline, 1e-12)
        )
        expert_preflight["promote_ctrv_top2"] = bool(
            expert_preflight["hard_oracle_relative_potential"] >= 0.10
        )
    artifact = {
        "schema": "ct_seqtrack.gt_hard_motion.v1",
        "source": "train_gt_only_cv_real_timestamp",
        "split": args.split,
        "category": category,
        "sample_count": int(difficulty.size),
        "tracklet_count": int(limit),
        "dt_floor": dt_floor,
        "timestamp_distribution": {
            "q05": float(np.quantile(gaps, 0.05)),
            "median": float(np.median(gaps)),
            "q95": float(np.quantile(gaps, 0.95)),
        },
        "difficulty": {
            "q50": float(np.quantile(difficulty, 0.50)),
            "q90": float(np.quantile(difficulty, 0.90)),
        },
        "expert_preflight": expert_preflight,
        "config_sha256": sha256_json(raw_config),
        "code_sha256": {
            "tool": sha256_file(Path(__file__)),
            "contract": sha256_file(ROOT / "utils" / "candidate_utils.py"),
        },
        "uses_checkpoint": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact["difficulty"], sort_keys=True))


if __name__ == "__main__":
    main()
