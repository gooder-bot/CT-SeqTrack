#!/usr/bin/env python3
"""Compare the current-commit 15-epoch B0 with Joint Full early training."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
DATA_OUT = ROOT / "compare_results" / "data"
STEPS_PER_EPOCH = 1262

RUNS = {
    "Current B0-15": "20260806-1357-01_seqtrack3d_baseline-current-baseline-mini-car-15ep-gpu3-s42",
    "Historical B0-60": "20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16",
    "Full-60": "20260805-2337-21_ct_joint_full-joint-full-mini-car-60ep-bs16-s42",
    "-B1-60": "20260805-2337-21_ct_joint_minus_b1-joint-minus-b1-mini-car-60ep-bs16-s42",
    "-B2-60": "20260805-2337-21_ct_joint_minus_b2-joint-minus-b2-mini-car-60ep-bs16-s42",
    "-B3-60": "20260805-2338-21_ct_joint_minus_b3-joint-minus-b3-mini-car-60ep-bs16-s42",
}

B0_METRICS = (
    "loss_center", "loss_center_aux", "loss_center_ref", "loss_seg",
    "loss_total",
)


def main_event(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("events.out.tfevents.*.0"))
    if not files:
        files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(log_dir)
    return files[0]


def accumulator(path: Path):
    result = EventAccumulator(str(path), size_guidance={"scalars": 0})
    result.Reload()
    return result


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    validation = []
    losses = []
    provenance = []
    lr_summary = {}
    for label, directory in RUNS.items():
        run = OUTPUT / directory
        log_dir = run / "lightning_logs" / "version_0"
        main = accumulator(main_event(log_dir))
        success = main.Scalars("success/test")
        precision = main.Scalars("precision/test")
        for index, (s_event, p_event) in enumerate(
                zip(success[:3], precision[:3]), start=1):
            validation.append({
                "run": label,
                "epoch": index * 5,
                "global_step": int(s_event.step),
                "success": float(s_event.value),
                "precision": float(p_event.value),
            })
        lr_events = main.Scalars("lr-Adam")
        early_lr_events = [
            event for event in lr_events if int(event.step) < 15 * STEPS_PER_EPOCH
        ]
        lr_summary[label] = {
            "first15_minimum": min(
                float(event.value) for event in early_lr_events),
            "first15_maximum": max(
                float(event.value) for event in early_lr_events),
        }
        for metric in B0_METRICS:
            files = sorted((log_dir / f"loss_{metric}").glob(
                "events.out.tfevents.*"))
            if not files:
                continue
            metric_events = accumulator(files[0]).Scalars("loss")
            by_epoch = {}
            for event in metric_events:
                epoch = int(event.step) // STEPS_PER_EPOCH + 1
                if epoch in (1, 5, 10, 15):
                    by_epoch.setdefault(epoch, []).append(float(event.value))
            for epoch, values in sorted(by_epoch.items()):
                losses.append({
                    "run": label,
                    "metric": metric,
                    "epoch": epoch,
                    "mean": sum(values) / len(values),
                    "batch_count": len(values),
                })
        payload = json.loads(
            (run / "run_provenance.json").read_text(encoding="utf-8"))
        config = payload["resolved_config"]
        provenance.append({
            "run": label,
            "commit": payload["git"]["commit"],
            "dirty_tracked": payload["git"]["dirty_tracked"],
            "seed": payload["seed"],
            "epochs": config["epoch"],
            "batch_size": config["batch_size"],
            "train_frames": payload["datasets"]["train"]["frames"],
            "val_frames": payload["datasets"]["val"]["frames"],
            "train_selection_sha256": payload["datasets"]["train"][
                "virtual_rate_selection_sha256"],
            "val_selection_sha256": payload["datasets"]["val"][
                "virtual_rate_selection_sha256"],
            "first15_lr_min": lr_summary[label]["first15_minimum"],
            "first15_lr_max": lr_summary[label]["first15_maximum"],
        })

    validation_path = DATA_OUT / "joint_full_baseline15_validation_20260806.csv"
    losses_path = DATA_OUT / "joint_full_baseline15_b0_losses_20260806.csv"
    provenance_path = DATA_OUT / "joint_full_baseline15_provenance_20260806.csv"
    write_csv(validation_path, validation)
    write_csv(losses_path, losses)
    write_csv(provenance_path, provenance)

    epoch15 = {
        row["run"]: row for row in validation if row["epoch"] == 15
    }
    summary = {
        "epoch15": epoch15,
        "deltas": {
            "current_vs_historical": {
                "success": epoch15["Current B0-15"]["success"]
                - epoch15["Historical B0-60"]["success"],
                "precision": epoch15["Current B0-15"]["precision"]
                - epoch15["Historical B0-60"]["precision"],
            },
            "current_vs_full": {
                "success": epoch15["Current B0-15"]["success"]
                - epoch15["Full-60"]["success"],
                "precision": epoch15["Current B0-15"]["precision"]
                - epoch15["Full-60"]["precision"],
            },
            "current_vs_minus_b2": {
                "success": epoch15["Current B0-15"]["success"]
                - epoch15["-B2-60"]["success"],
                "precision": epoch15["Current B0-15"]["precision"]
                - epoch15["-B2-60"]["precision"],
            },
        },
        "lr": lr_summary,
        "artifacts": {
            "validation": str(validation_path.relative_to(ROOT)),
            "losses": str(losses_path.relative_to(ROOT)),
            "provenance": str(provenance_path.relative_to(ROOT)),
        },
    }
    summary_path = DATA_OUT / "joint_full_baseline15_summary_20260806.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
