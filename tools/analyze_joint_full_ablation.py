#!/usr/bin/env python3
"""Analyze the 2026-08-05 CT joint-Full mini ablation runs.

The script reads the immutable run provenance and TensorBoard event streams,
then emits compact CSV/JSON artifacts used by the technical result report.
"""

from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
DATA_OUT = ROOT / "compare_results" / "data"

RUNS = {
    "Full": "20260805-2337-21_ct_joint_full-joint-full-mini-car-60ep-bs16-s42",
    "-B1": "20260805-2337-21_ct_joint_minus_b1-joint-minus-b1-mini-car-60ep-bs16-s42",
    "-B2": "20260805-2337-21_ct_joint_minus_b2-joint-minus-b2-mini-car-60ep-bs16-s42",
    "-B3": "20260805-2338-21_ct_joint_minus_b3-joint-minus-b3-mini-car-60ep-bs16-s42",
    "SeqTrack": "20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16",
}

TRAIN_METRICS = (
    "loss_total",
    "loss_center",
    "loss_center_aux",
    "loss_angle_aux",
    "loss_center_motion",
    "loss_center_ref",
    "loss_seg",
    "ct_candidate_valid_rate",
    "ct_foreground_points",
    "ct_query_gate_mean",
    "ct_query_shift_norm",
    "ct_motion_residual_saturation",
    "ct_router_gate_mean",
    "ct_observation_rmse",
    "ct_raw_search_rmse",
    "ct_final_rmse",
    "motion_v3_kinematic_rmse",
    "motion_v3_prior_rmse",
    "motion_v3_sigma_parallel_mean",
    "motion_v3_sigma_perpendicular_mean",
    "loss_motion_v3_prior",
    "loss_motion_v3_nll",
    "loss_ct_targetness",
    "loss_ct_vote",
    "loss_ct_raw_search",
    "loss_ct_query_gate",
    "loss_ct_router",
    "loss_ct_correction",
    "ct_search_baseline_points_mean",
    "ct_search_expansion_points_mean",
    "ct_search_expansion_ratio_mean",
    "ct_search_used_mean",
    "search_has_usable_points_mean",
    "trajectory_search_valid_mean",
)

STEPS_PER_EPOCH = 1262


def _main_event(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("events.out.tfevents.*.0"))
    if not candidates:
        candidates = sorted(log_dir.glob("events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file below {log_dir}")
    return candidates[0]


def _read_validation(label: str, run_name: str):
    log_dir = OUTPUT / run_name / "lightning_logs" / "version_0"
    accumulator = EventAccumulator(
        str(_main_event(log_dir)), size_guidance={"scalars": 0})
    accumulator.Reload()
    success = accumulator.Scalars("success/test")
    precision = accumulator.Scalars("precision/test")
    if len(success) != len(precision):
        raise RuntimeError(f"Mismatched validation streams for {label}")
    rows = []
    for index, (success_event, precision_event) in enumerate(
            zip(success, precision), start=1):
        if success_event.step != precision_event.step:
            raise RuntimeError(f"Mismatched validation steps for {label}")
        rows.append({
            "run": label,
            "epoch": index * 5,
            "global_step": int(success_event.step),
            "success": float(success_event.value),
            "precision": float(precision_event.value),
        })
    return rows


def _mean(values):
    return sum(values) / len(values) if values else None


def _read_train_metrics(item):
    label, run_name = item
    log_dir = OUTPUT / run_name / "lightning_logs" / "version_0"
    rows = []
    for metric in TRAIN_METRICS:
        metric_dir = log_dir / f"loss_{metric}"
        event_files = sorted(metric_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        accumulator = EventAccumulator(
            str(event_files[0]), size_guidance={"scalars": 0})
        accumulator.Reload()
        events = accumulator.Scalars("loss")
        by_epoch = {}
        for event in events:
            epoch = int(event.step) // STEPS_PER_EPOCH + 1
            by_epoch.setdefault(epoch, []).append(float(event.value))
        for epoch in sorted(by_epoch):
            values = by_epoch[epoch]
            rows.append({
                "run": label,
                "metric": metric,
                "epoch": epoch,
                "batch_count": len(values),
                "mean": _mean(values),
                "min": min(values),
                "max": max(values),
                "all_finite": all(math.isfinite(value) for value in values),
            })
    return label, rows


def _provenance_summary(label: str, run_name: str):
    path = OUTPUT / run_name / "run_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["resolved_config"]
    return {
        "run": label,
        "path": str(path.relative_to(ROOT)),
        "commit": payload["git"]["commit"],
        "dirty_tracked": payload["git"]["dirty_tracked"],
        "dirty_any": payload["git"]["dirty_any"],
        "seed": payload["seed"],
        "version": config.get("version"),
        "category": config.get("category_name"),
        "train_split": config.get("train_split"),
        "val_split": config.get("val_split"),
        "batch_size": config.get("batch_size"),
        "workers": config.get("workers"),
        "epochs": config.get("epoch"),
        "train_frames": payload["datasets"]["train"]["frames"],
        "val_frames": payload["datasets"]["val"]["frames"],
        "train_selection_sha256": payload["datasets"]["train"][
            "virtual_rate_selection_sha256"],
        "val_selection_sha256": payload["datasets"]["val"][
            "virtual_rate_selection_sha256"],
        "ct_enable_b1": config.get("ct_enable_b1"),
        "ct_enable_b2": config.get("ct_enable_b2"),
        "ct_enable_b3": config.get("ct_enable_b3"),
    }


def _write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    provenance = [
        _provenance_summary(label, run_name)
        for label, run_name in RUNS.items()
    ]
    validation = []
    for label, run_name in RUNS.items():
        validation.extend(_read_validation(label, run_name))

    train_rows = []
    joint_items = list(RUNS.items())
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_read_train_metrics, item)
                   for item in joint_items]
        for future in as_completed(futures):
            _, rows = future.result()
            train_rows.extend(rows)
    train_rows.sort(key=lambda row: (
        list(RUNS).index(row["run"]), row["metric"], row["epoch"]))

    validation_path = DATA_OUT / "joint_full_validation_curves_20260806.csv"
    train_path = DATA_OUT / "joint_full_training_diagnostics_20260806.csv"
    provenance_path = DATA_OUT / "joint_full_run_provenance_20260806.csv"
    _write_csv(validation_path, validation)
    _write_csv(train_path, train_rows)
    _write_csv(provenance_path, provenance)

    final = {row["run"]: row for row in validation if row["epoch"] == 60}
    summary = {
        "runs": provenance,
        "final": final,
        "deltas": {},
        "artifacts": {
            "validation": str(validation_path.relative_to(ROOT)),
            "training_diagnostics": str(train_path.relative_to(ROOT)),
            "provenance": str(provenance_path.relative_to(ROOT)),
        },
    }
    for label in ("Full", "-B1", "-B2", "-B3"):
        summary["deltas"][label] = {
            "vs_seqtrack_success": (
                final[label]["success"] - final["SeqTrack"]["success"]),
            "vs_seqtrack_precision": (
                final[label]["precision"] - final["SeqTrack"]["precision"]),
            "vs_minus_b2_success": (
                final[label]["success"] - final["-B2"]["success"]),
            "vs_minus_b2_precision": (
                final[label]["precision"] - final["-B2"]["precision"]),
        }
    summary_path = DATA_OUT / "joint_full_ablation_summary_20260806.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
