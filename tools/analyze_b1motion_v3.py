#!/usr/bin/env python3
"""Analyze the completed B1motion-v3 seed42 60-epoch experiment.

The analysis keeps three questions separate:

* did v3 repair the catastrophic v2 failure;
* did v3 beat the historical B0 guardrail at the preregistered final and
  late-3 checkpoints;
* if not, did the physical prior fail, or did reliability fusion fail to
  convert a learned prior into recursive tracking gains.

All report numbers are regenerated from TensorBoard scalars, provenance, and
checkpoint payloads.  Epoch-60 ``last.ckpt`` remains primary; best-checkpoint
values are diagnostics only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "compare_results" / "data"
REPORT_DIR = ROOT / "compare_results" / "reports"
STEM = "b1motion_v3_seed42_20260801"

RUNS = {
    "SEQTRACK": {
        "arm": "SeqTrack3D plain (original run)",
        "path": (
            "../seqtrack/output/"
            "20260528-1633-seqtrack3d_nuscenes_mini-"
            "seqtrack_mini_baseline_car_60ep_bs16"
        ),
    },
    "B0": {
        "arm": "B0 baseline (historical)",
        "path": (
            "output/20260725-2326-01_seqtrack3d_baseline-"
            "ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16"
        ),
    },
    "B1V2": {
        "arm": "B1motion-v2",
        "path": (
            "output/20260730-0305-02_ct_motion-"
            "ctv2_b1_motion_v2_car_seed42_60ep_bs16_"
            "gpu2_thread1_scratch"
        ),
    },
    "B1V3": {
        "arm": "B1motion-v3",
        "path": (
            "output/20260801-0117-02_ct_motion_v3-"
            "b1motion_v3_mini_car_60ep_bs16_seed42"
        ),
    },
}

TRAINING_LEAVES = {
    "candidate0_prior_rmse": "loss_motion_v3_prior_rmse_candidate0",
    "candidate0_cv_rmse": "loss_motion_v3_kinematic_rmse_candidate0",
    "candidate_nonzero_prior_rmse": (
        "loss_motion_v3_prior_rmse_candidate_nonzero"
    ),
    "candidate_nonzero_cv_rmse": (
        "loss_motion_v3_kinematic_rmse_candidate_nonzero"
    ),
    "candidate0_count": "loss_motion_v3_count_candidate0",
    "candidate_nonzero_count": "loss_motion_v3_count_candidate_nonzero",
    "gap2_prior_rmse": "loss_motion_v3_aux_prior_rmse_gap2",
    "gap2_cv_rmse": "loss_motion_v3_aux_kinematic_rmse_gap2",
    "gap4_prior_rmse": "loss_motion_v3_aux_prior_rmse_gap4",
    "gap4_cv_rmse": "loss_motion_v3_aux_kinematic_rmse_gap4",
    "gap2_count": "loss_motion_v3_aux_count_gap2",
    "gap4_count": "loss_motion_v3_aux_count_gap4",
    "alpha_utilization": "loss_motion_v3_gate_alpha_utilization",
    "gate_applied_rate": "loss_motion_v3_gate_applied_rate",
    "gate_precision": "loss_motion_v3_gate_precision",
    "helpful_rate": "loss_motion_v3_helpful_rate",
    "correction_norm_m": "loss_motion_v3_correction_norm",
    "clip_rate": "loss_motion_v3_clip_rate",
    "observation_error_m": "loss_motion_v3_observation_error",
    "prior_box_error_m": "loss_motion_v3_prior_box_error",
    "final_error_m": "loss_motion_v3_final_error",
    "prior_valid_rate": "loss_motion_v3_prior_valid_rate",
    "history_valid_ratio": "loss_motion_v3_history_valid_ratio",
    "aux_history_valid_ratio": "loss_motion_v3_aux_history_valid_ratio",
    "prior_loss": "loss_loss_motion_v3_prior",
    "aux_prior_loss": "loss_loss_motion_v3_aux_prior",
    "fused_loss": "loss_loss_motion_v3_fused",
    "gate_loss": "loss_loss_motion_v3_gate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_events(version_dir: Path, leaf: str) -> dict[int, float]:
    event_dir = version_dir / leaf
    if not event_dir.is_dir():
        return {}
    accumulator = EventAccumulator(
        str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        return {}
    tag = "loss" if "loss" in tags else tags[0]
    return {
        int(item.step): float(item.value)
        for item in accumulator.Scalars(tag)
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def install_easydict_fallback() -> None:
    try:
        __import__("easydict")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("easydict")

    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

        def __setattr__(self, key, value):
            self[key] = value

    EasyDict.__module__ = "easydict"
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module


def steps_per_epoch(provenance: dict[str, Any]) -> int:
    config = provenance["resolved_config"]
    frames = int(provenance["datasets"]["train"]["frames"])
    candidates = int(config.get("num_candidates", 1))
    return (frames * candidates) // int(config["batch_size"])


def collect_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    collected: dict[str, Any] = {}
    for run_id, spec in RUNS.items():
        run_dir = ROOT / spec["path"]
        version_dir = run_dir / "lightning_logs" / "version_0"
        success = scalar_events(version_dir, "metrics_test_success")
        precision = scalar_events(version_dir, "metrics_test_precision")
        shared_steps = sorted(set(success) & set(precision))
        if len(shared_steps) != 12:
            raise ValueError(
                f"{run_id} has {len(shared_steps)} paired validation points")
        run_rows = []
        for index, step in enumerate(shared_steps, start=1):
            row = {
                "run_id": run_id,
                "arm": spec["arm"],
                "epoch": index * 5,
                "step": step,
                "success": success[step],
                "precision": precision[step],
            }
            rows.append(row)
            run_rows.append(row)
        collected[run_id] = run_rows
    return rows, collected


def summarize_validation(
        validation: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    b0 = validation["B0"]
    b0_final = b0[-1]
    b0_late_s = float(np.mean([row["success"] for row in b0[-3:]]))
    b0_late_p = float(np.mean([row["precision"] for row in b0[-3:]]))
    rows = []
    for run_id in RUNS:
        values = validation[run_id]
        final = values[-1]
        best_success = max(values, key=lambda row: row["success"])
        best_precision = max(values, key=lambda row: row["precision"])
        late_success = float(np.mean(
            [row["success"] for row in values[-3:]]))
        late_precision = float(np.mean(
            [row["precision"] for row in values[-3:]]))
        rows.append({
            "run_id": run_id,
            "arm": RUNS[run_id]["arm"],
            "final_success": final["success"],
            "final_precision": final["precision"],
            "delta_final_success_vs_b0": (
                final["success"] - b0_final["success"]),
            "delta_final_precision_vs_b0": (
                final["precision"] - b0_final["precision"]),
            "late3_success": late_success,
            "late3_precision": late_precision,
            "delta_late3_success_vs_b0": late_success - b0_late_s,
            "delta_late3_precision_vs_b0": late_precision - b0_late_p,
            "best_success": best_success["success"],
            "best_success_epoch": best_success["epoch"],
            "best_precision": best_precision["precision"],
            "best_precision_epoch": best_precision["epoch"],
            "paired_success_at_best_precision": best_precision["success"],
        })
    return rows


def epoch_steps(
        series: dict[int, float], epoch: int, per_epoch: int
) -> set[int]:
    start = (epoch - 1) * per_epoch
    end = epoch * per_epoch
    return {step for step in series if start <= step < end}


def aligned_steps(
        series: dict[str, dict[int, float]], keys: Iterable[str],
        epoch: int, per_epoch: int
) -> list[int]:
    keys = list(keys)
    shared = epoch_steps(series[keys[0]], epoch, per_epoch)
    for key in keys[1:]:
        shared &= epoch_steps(series[key], epoch, per_epoch)
    return sorted(shared)


def epoch_mean(
        series: dict[str, dict[int, float]], key: str,
        epoch: int, per_epoch: int
) -> float | None:
    steps = aligned_steps(series, [key], epoch, per_epoch)
    if not steps:
        return None
    values = np.asarray([series[key][step] for step in steps], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else None


def epoch_rmse(
        series: dict[str, dict[int, float]], metric_key: str,
        count_key: str, epoch: int, per_epoch: int
) -> tuple[float, float]:
    steps = aligned_steps(
        series, [metric_key, count_key], epoch, per_epoch)
    if not steps:
        return math.nan, 0.0
    values = np.asarray(
        [series[metric_key][step] for step in steps], dtype=float)
    counts = np.asarray(
        [series[count_key][step] for step in steps], dtype=float)
    finite = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    if not np.any(finite):
        return math.nan, 0.0
    count = float(np.sum(counts[finite]))
    rmse = float(np.sqrt(
        np.sum(values[finite] ** 2 * counts[finite]) / count))
    return rmse, count


def pooled_rmse(parts: Iterable[tuple[float, float]]) -> float:
    parts = [(value, count) for value, count in parts if count > 0]
    if not parts:
        return math.nan
    return float(np.sqrt(
        sum(value * value * count for value, count in parts)
        / sum(count for _, count in parts)))


def percent_improvement(learned: float, reference: float) -> float:
    return 100.0 * (reference - learned) / reference


def collect_training() -> tuple[list[dict[str, Any]], int]:
    run_dir = ROOT / RUNS["B1V3"]["path"]
    provenance = read_json(run_dir / "run_provenance.json")
    per_epoch = steps_per_epoch(provenance)
    version_dir = run_dir / "lightning_logs" / "version_0"
    series = {
        key: scalar_events(version_dir, leaf)
        for key, leaf in TRAINING_LEAVES.items()
    }
    missing = [key for key, values in series.items() if not values]
    if missing:
        raise ValueError(f"missing v3 training scalars: {missing}")

    rows: list[dict[str, Any]] = []
    alpha_max = float(provenance["resolved_config"]["motion_v3_alpha_max"])
    for epoch in range(1, 61):
        c0_prior, c0_count = epoch_rmse(
            series, "candidate0_prior_rmse", "candidate0_count",
            epoch, per_epoch)
        c0_cv, _ = epoch_rmse(
            series, "candidate0_cv_rmse", "candidate0_count",
            epoch, per_epoch)
        cn_prior, cn_count = epoch_rmse(
            series, "candidate_nonzero_prior_rmse",
            "candidate_nonzero_count", epoch, per_epoch)
        cn_cv, _ = epoch_rmse(
            series, "candidate_nonzero_cv_rmse",
            "candidate_nonzero_count", epoch, per_epoch)
        main_prior = pooled_rmse(
            [(c0_prior, c0_count), (cn_prior, cn_count)])
        main_cv = pooled_rmse(
            [(c0_cv, c0_count), (cn_cv, cn_count)])

        gap2_prior, gap2_count = epoch_rmse(
            series, "gap2_prior_rmse", "gap2_count", epoch, per_epoch)
        gap2_cv, _ = epoch_rmse(
            series, "gap2_cv_rmse", "gap2_count", epoch, per_epoch)
        gap4_prior, gap4_count = epoch_rmse(
            series, "gap4_prior_rmse", "gap4_count", epoch, per_epoch)
        gap4_cv, _ = epoch_rmse(
            series, "gap4_cv_rmse", "gap4_count", epoch, per_epoch)
        aux_prior = pooled_rmse(
            [(gap2_prior, gap2_count), (gap4_prior, gap4_count)])
        aux_cv = pooled_rmse(
            [(gap2_cv, gap2_count), (gap4_cv, gap4_count)])

        row: dict[str, Any] = {
            "epoch": epoch,
            "main_prior_rmse_m": main_prior,
            "main_cv_rmse_m": main_cv,
            "main_improvement_pct": percent_improvement(
                main_prior, main_cv),
            "candidate0_prior_rmse_m": c0_prior,
            "candidate0_cv_rmse_m": c0_cv,
            "candidate0_improvement_pct": percent_improvement(
                c0_prior, c0_cv),
            "candidate_nonzero_prior_rmse_m": cn_prior,
            "candidate_nonzero_cv_rmse_m": cn_cv,
            "candidate_nonzero_improvement_pct": percent_improvement(
                cn_prior, cn_cv),
            "aux_prior_rmse_m": aux_prior,
            "aux_cv_rmse_m": aux_cv,
            "aux_improvement_pct": percent_improvement(aux_prior, aux_cv),
            "gap2_prior_rmse_m": gap2_prior,
            "gap2_cv_rmse_m": gap2_cv,
            "gap2_improvement_pct": percent_improvement(
                gap2_prior, gap2_cv),
            "gap4_prior_rmse_m": gap4_prior,
            "gap4_cv_rmse_m": gap4_cv,
            "gap4_improvement_pct": percent_improvement(
                gap4_prior, gap4_cv),
        }
        for key in (
                "alpha_utilization", "gate_applied_rate", "gate_precision",
                "helpful_rate", "correction_norm_m", "clip_rate",
                "observation_error_m", "prior_box_error_m", "final_error_m",
                "prior_valid_rate", "history_valid_ratio",
                "aux_history_valid_ratio", "prior_loss", "aux_prior_loss",
                "fused_loss", "gate_loss"):
            row[key] = epoch_mean(series, key, epoch, per_epoch)
        row["actual_alpha_mean"] = (
            row["alpha_utilization"] * alpha_max
            if row["alpha_utilization"] is not None else None)
        rows.append(row)
    return rows, per_epoch


def checkpoint_rows() -> list[dict[str, Any]]:
    install_easydict_fallback()
    checkpoint_dir = (
        ROOT / RUNS["B1V3"]["path"]
        / "lightning_logs" / "version_0" / "checkpoints"
    )
    rows = []
    for path in sorted(checkpoint_dir.glob("*.ckpt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]

        def find(suffix: str) -> torch.Tensor:
            matches = [
                tensor.detach().float()
                for key, tensor in state.items()
                if key.endswith(suffix)
            ]
            if len(matches) != 1:
                raise ValueError(f"{path.name}: expected one {suffix}")
            return matches[0]

        callbacks = payload.get("callbacks", {})
        best_score = None
        best_path = None
        for value in callbacks.values():
            if isinstance(value, dict) and "best_model_score" in value:
                score = value.get("best_model_score")
                best_score = float(score) if score is not None else None
                best_path = value.get("best_model_path")
                break
        residual_weight = find(
            "physical_motion_encoder.velocity_residual_head.weight")
        gate_weight = find("motion_v3_fusion.gate.2.weight")
        gate_bias = find("motion_v3_fusion.gate.2.bias")
        rows.append({
            "checkpoint": path.name,
            "epoch_zero_based": int(payload.get("epoch", -1)),
            "global_step": int(payload.get("global_step", -1)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "residual_head_weight_l2": float(torch.linalg.vector_norm(
                residual_weight)),
            "gate_last_weight_l2": float(torch.linalg.vector_norm(
                gate_weight)),
            "gate_last_bias": float(gate_bias.reshape(-1)[0]),
            "best_precision_score": best_score,
            "best_checkpoint_path": best_path,
        })
    return rows


def seqtrack_b0_diagnostic_rows(
        validation: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Build an evidence register for the cross-codebase baseline gap.

    This deliberately distinguishes numerical comparison from method
    attribution.  The original SeqTrack run predates run_provenance.json, so
    its exact source/data state cannot be reconstructed from the output folder.
    """
    install_easydict_fallback()
    seq_values = validation["SEQTRACK"]
    b0_values = validation["B0"]

    def late3(values: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([row[key] for row in values[-3:]]))

    def checkpoint_metadata(run_id: str) -> dict[str, Any]:
        checkpoint = (
            ROOT / RUNS[run_id]["path"] / "lightning_logs" / "version_0"
            / "checkpoints" / "last.ckpt"
        )
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        config = payload["hyper_parameters"]["config"]
        return {
            "keys": set(state),
            "shapes": {key: tuple(value.shape) for key, value in state.items()},
            "tensor_count": len(state),
            "element_count": int(sum(value.numel() for value in state.values())),
            "workers": int(config["workers"]),
            "seed": int(config["seed"]),
            "batch_size": int(config["batch_size"]),
            "global_step": int(payload["global_step"]),
            "lightning": str(payload.get("pytorch-lightning_version", "unknown")),
        }

    seq_meta = checkpoint_metadata("SEQTRACK")
    b0_meta = checkpoint_metadata("B0")
    same_topology = (
        seq_meta["keys"] == b0_meta["keys"]
        and seq_meta["shapes"] == b0_meta["shapes"]
    )

    per_epoch = 1262
    late_losses: dict[str, float] = {}
    for run_id in ("SEQTRACK", "B0"):
        version_dir = (
            ROOT / RUNS[run_id]["path"] / "lightning_logs" / "version_0")
        series = scalar_events(version_dir, "loss_loss_total")
        values = [
            value for step, value in series.items()
            if 55 * per_epoch <= step < 60 * per_epoch
        ]
        if not values:
            raise ValueError(f"{run_id} has no epoch56-60 total-loss values")
        late_losses[run_id] = float(np.mean(values))

    paired_success = [
        b0["success"] - seq["success"]
        for seq, b0 in zip(seq_values, b0_values)
    ]
    paired_precision = [
        b0["precision"] - seq["precision"]
        for seq, b0 in zip(seq_values, b0_values)
    ]

    replicate_values = []
    for relative in (
        "../seqtrack/output/20260629-1644-seqtrack3d_nuscenes_mini-"
        "seqtrack_mini_baseline_car_180ep_bs16_gpu1",
        "../seqtrack/output/20260702-0038-seqtrack3d_nuscenes_mini-"
        "seqtrack_mini_baseline_car_180ep_bs16_gpu3",
    ):
        version_dir = ROOT / relative / "lightning_logs" / "version_0"
        success = scalar_events(version_dir, "metrics_test_success")
        precision = scalar_events(version_dir, "metrics_test_precision")
        shared = sorted(set(success) & set(precision))
        if len(shared) >= 12:
            step = shared[11]
            replicate_values.append((success[step], precision[step]))

    rows = [
        {
            "item": "Epoch60 final",
            "seqtrack": (
                f"{seq_values[-1]['success']:.3f} S / "
                f"{seq_values[-1]['precision']:.3f} P"),
            "b0": (
                f"{b0_values[-1]['success']:.3f} S / "
                f"{b0_values[-1]['precision']:.3f} P"),
            "comparison": (
                f"{b0_values[-1]['success']-seq_values[-1]['success']:+.3f} S / "
                f"{b0_values[-1]['precision']-seq_values[-1]['precision']:+.3f} P"),
            "interpretation": "B0 is higher at the preregistered final checkpoint.",
        },
        {
            "item": "Late-3 mean",
            "seqtrack": (
                f"{late3(seq_values, 'success'):.3f} S / "
                f"{late3(seq_values, 'precision'):.3f} P"),
            "b0": (
                f"{late3(b0_values, 'success'):.3f} S / "
                f"{late3(b0_values, 'precision'):.3f} P"),
            "comparison": (
                f"{late3(b0_values, 'success')-late3(seq_values, 'success'):+.3f} S / "
                f"{late3(b0_values, 'precision')-late3(seq_values, 'precision'):+.3f} P"),
            "interpretation": "The positive gap is a late-training stability gap.",
        },
        {
            "item": "Best checkpoint by metric",
            "seqtrack": (
                f"{max(row['success'] for row in seq_values):.3f} best S / "
                f"{max(row['precision'] for row in seq_values):.3f} best P"),
            "b0": (
                f"{max(row['success'] for row in b0_values):.3f} best S / "
                f"{max(row['precision'] for row in b0_values):.3f} best P"),
            "comparison": "B0 best S is higher; SeqTrack best P is higher",
            "interpretation": "B0 is not uniformly superior over the whole trajectory.",
        },
        {
            "item": "All 12 validation points",
            "seqtrack": "Reference trajectory",
            "b0": (
                f"wins {sum(value > 0 for value in paired_success)}/12 S and "
                f"{sum(value > 0 for value in paired_precision)}/12 P"),
            "comparison": (
                f"mean {np.mean(paired_success):+.3f} S / "
                f"{np.mean(paired_precision):+.3f} P"),
            "interpretation": "Early B0 regressions precede the late reversal.",
        },
        {
            "item": "Epoch56-60 mean training loss",
            "seqtrack": f"{late_losses['SEQTRACK']:.5f}",
            "b0": f"{late_losses['B0']:.5f}",
            "comparison": f"{late_losses['B0']-late_losses['SEQTRACK']:+.5f}",
            "interpretation": "Optimization is nearly identical; no large objective change.",
        },
        {
            "item": "Checkpoint model topology",
            "seqtrack": (
                f"{seq_meta['tensor_count']} tensors / "
                f"{seq_meta['element_count']:,} elements"),
            "b0": (
                f"{b0_meta['tensor_count']} tensors / "
                f"{b0_meta['element_count']:,} elements"),
            "comparison": "identical keys and shapes" if same_topology else "different",
            "interpretation": "The gain is not caused by a larger CT model.",
        },
        {
            "item": "Training budget/framework",
            "seqtrack": (
                f"seed{seq_meta['seed']}, bs{seq_meta['batch_size']}, "
                f"{seq_meta['global_step']} steps, PL {seq_meta['lightning']}"),
            "b0": (
                f"seed{b0_meta['seed']}, bs{b0_meta['batch_size']}, "
                f"{b0_meta['global_step']} steps, PL {b0_meta['lightning']}"),
            "comparison": "matched",
            "interpretation": "Budget and top-level trainer version do not explain the gap.",
        },
        {
            "item": "DataLoader workers",
            "seqtrack": str(seq_meta["workers"]),
            "b0": str(b0_meta["workers"]),
            "comparison": f"{seq_meta['workers']} -> {b0_meta['workers']}",
            "interpretation": (
                "NumPy candidate/point-sampling streams change with worker assignment; "
                "seed42 is therefore not the same training sample stream."),
        },
        {
            "item": "Original-run provenance",
            "seqtrack": "no run_provenance.json / exact commit unknown",
            "b0": "clean d86990ce with dataset selection hashes",
            "comparison": "not attribution-safe",
            "interpretation": "Exact code and data identity cannot be proven retrospectively.",
        },
    ]
    if replicate_values:
        all_values = [
            (seq_values[-1]["success"], seq_values[-1]["precision"]),
            *replicate_values,
        ]
        rows.append({
            "item": "Same-labelled SeqTrack epoch60 spread",
            "seqtrack": "; ".join(
                f"{success:.3f}/{precision:.3f}"
                for success, precision in all_values),
            "b0": "single run 53.360/64.382",
            "comparison": (
                f"SeqTrack range {min(v[0] for v in all_values):.3f}-"
                f"{max(v[0] for v in all_values):.3f} S / "
                f"{min(v[1] for v in all_values):.3f}-"
                f"{max(v[1] for v in all_values):.3f} P"),
            "interpretation": (
                "These folders lack exact source provenance, so they diagnose "
                "repeatability risk rather than estimate clean seed variance."),
        })
    return rows


def integrity_rows(
        validation: dict[str, list[dict[str, Any]]],
        training_rows: list[dict[str, Any]], per_epoch: int,
        checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    b0_provenance = read_json(
        ROOT / RUNS["B0"]["path"] / "run_provenance.json")
    v3_run = ROOT / RUNS["B1V3"]["path"]
    v3_provenance = read_json(v3_run / "run_provenance.json")
    checkpoint_by_name = {row["checkpoint"]: row for row in checkpoints}
    checks = [
        ("training_steps", per_epoch * 60, 75720, per_epoch * 60 == 75720),
        ("validation_points", len(validation["B1V3"]), 12,
         len(validation["B1V3"]) == 12),
        ("training_epoch_rows", len(training_rows), 60,
         len(training_rows) == 60),
        ("epoch60_checkpoint", checkpoint_by_name["last.ckpt"]["global_step"],
         75720, checkpoint_by_name["last.ckpt"]["global_step"] == 75720),
        ("last_matches_epoch59_sha256",
         checkpoint_by_name["last.ckpt"]["sha256"],
         checkpoint_by_name["epoch=59-step=75720.ckpt"]["sha256"],
         checkpoint_by_name["last.ckpt"]["sha256"]
         == checkpoint_by_name["epoch=59-step=75720.ckpt"]["sha256"]),
        ("train_selection_matches_b0",
         v3_provenance["datasets"]["train"][
             "virtual_rate_selection_sha256"],
         b0_provenance["datasets"]["train"][
             "virtual_rate_selection_sha256"],
         v3_provenance["datasets"]["train"][
             "virtual_rate_selection_sha256"]
         == b0_provenance["datasets"]["train"][
             "virtual_rate_selection_sha256"]),
        ("val_selection_matches_b0",
         v3_provenance["datasets"]["val"][
             "virtual_rate_selection_sha256"],
         b0_provenance["datasets"]["val"][
             "virtual_rate_selection_sha256"],
         v3_provenance["datasets"]["val"][
             "virtual_rate_selection_sha256"]
         == b0_provenance["datasets"]["val"][
             "virtual_rate_selection_sha256"]),
        ("scratch_training", v3_provenance["init_checkpoint_path"], None,
         v3_provenance["init_checkpoint_path"] is None),
    ]
    for check, observed, expected, passed in checks:
        rows.append({
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
        })
    rows.extend([
        {
            "check": "v3_git_commit",
            "observed": v3_provenance["git"]["commit"],
            "expected": "recorded",
            "passed": bool(v3_provenance["git"]["commit"]),
        },
        {
            "check": "v3_tracked_source_clean",
            "observed": not v3_provenance["git"]["dirty_tracked"],
            "expected": True,
            "passed": not v3_provenance["git"]["dirty_tracked"],
        },
        {
            "check": "same_commit_as_historical_b0",
            "observed": v3_provenance["git"]["commit"],
            "expected": b0_provenance["git"]["commit"],
            "passed": (
                v3_provenance["git"]["commit"]
                == b0_provenance["git"]["commit"]),
        },
    ])
    return rows


def build_driver_rows(
        summary: list[dict[str, Any]],
        training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run = {row["run_id"]: row for row in summary}
    e11, e60 = training[10], training[59]
    return [
        {
            "priority": 1,
            "finding": "No stable standard-cadence gain",
            "evidence": (
                f"Final v3-B0 is {by_run['B1V3']['delta_final_success_vs_b0']:+.3f}"
                f"/{by_run['B1V3']['delta_final_precision_vs_b0']:+.3f}; "
                f"late-3 is {by_run['B1V3']['delta_late3_success_vs_b0']:+.3f}"
                f"/{by_run['B1V3']['delta_late3_precision_vs_b0']:+.3f}."
            ),
            "interpretation": (
                "Fails the preregistered +0.5/+1.0 final gate and the "
                "non-negative late-3 guardrail."
            ),
            "confidence": "High for No-Go; one seed cannot quantify variance",
        },
        {
            "priority": 2,
            "finding": "v3 is numerically above the original SeqTrack run",
            "evidence": (
                f"Final v3-SeqTrack is "
                f"{by_run['B1V3']['final_success']-by_run['SEQTRACK']['final_success']:+.3f}/"
                f"{by_run['B1V3']['final_precision']-by_run['SEQTRACK']['final_precision']:+.3f}; "
                f"current B0-SeqTrack is "
                f"{by_run['B0']['final_success']-by_run['SEQTRACK']['final_success']:+.3f}/"
                f"{by_run['B0']['final_precision']-by_run['SEQTRACK']['final_precision']:+.3f}."
            ),
            "interpretation": (
                "The paper-table comparison is positive, but the stronger current "
                "B0 shows that the gain cannot be attributed to motion."
            ),
            "confidence": "Medium; original run has no captured provenance",
        },
        {
            "priority": 3,
            "finding": "v3 repairs the catastrophic v2 architecture",
            "evidence": (
                f"Final v3-v2 is "
                f"{by_run['B1V3']['final_success']-by_run['B1V2']['final_success']:+.3f}/"
                f"{by_run['B1V3']['final_precision']-by_run['B1V2']['final_precision']:+.3f}."
            ),
            "interpretation": (
                "Protected B0 view, physical targets, and bounded post-transformer "
                "fusion removed v2's collapse, but safety repair is not a B0 gain."
            ),
            "confidence": "High",
        },
        {
            "priority": 4,
            "finding": "The physical prior learns beyond constant velocity",
            "evidence": (
                f"At epoch60 learned RMSE is lower by "
                f"{e60['main_improvement_pct']:.1f}% on main cadence, "
                f"{e60['gap2_improvement_pct']:.1f}% on gap2, and "
                f"{e60['gap4_improvement_pct']:.1f}% on gap4."
            ),
            "interpretation": (
                "GRU capacity and physical target identifiability are no longer "
                "the leading failure mode on the training distribution."
            ),
            "confidence": "High descriptively; training-only evidence",
        },
        {
            "priority": 5,
            "finding": "Observation improves while the prior plateaus",
            "evidence": (
                f"Observation/prior box error moves from "
                f"{e11['observation_error_m']:.3f}/{e11['prior_box_error_m']:.3f} m "
                f"at first active epoch to "
                f"{e60['observation_error_m']:.3f}/{e60['prior_box_error_m']:.3f} m; "
                f"helpful prevalence falls from {100*e11['helpful_rate']:.1f}% "
                f"to {100*e60['helpful_rate']:.1f}%."
            ),
            "interpretation": (
                "The gate must become more conservative late in training, but "
                "the current objective does not enforce that calibration."
            ),
            "confidence": "High on one-step training batches",
        },
        {
            "priority": 6,
            "finding": "The late gate remains half open and weakly selective",
            "evidence": (
                f"Epoch60 mean alpha is {e60['actual_alpha_mean']:.3f}; "
                f"application is {100*e60['gate_applied_rate']:.1f}% of decisive "
                f"samples with {100*e60['gate_precision']:.1f}% precision; mean "
                f"correction is {e60['correction_norm_m']:.3f} m."
            ),
            "interpretation": (
                "Class-balanced BCE calibrates class separation, not deployment "
                "prevalence or recursive cost. Nearly half of applied decisive "
                "corrections are not labeled helpful."
            ),
            "confidence": "Medium-high; decisive-count weighting is unavailable",
        },
        {
            "priority": 7,
            "finding": "One-step training improvement does not transfer",
            "evidence": (
                f"Epoch60 one-step final error is {e60['final_error_m']:.3f} m "
                f"versus observation {e60['observation_error_m']:.3f} m, yet normal "
                "recursive validation is below B0."
            ),
            "interpretation": (
                "Candidate-perturbed teacher-forced gate training is mismatched to "
                "recursive predicted-history deployment and tracklet metrics."
            ),
            "confidence": "Medium; fusion-off endpoint export is still missing",
        },
    ]


def build_next_rows() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "experiment": "Same-checkpoint standard on/off attribution",
            "change": (
                "Evaluate epoch30 and epoch60 v3 checkpoints with fusion on and "
                "motion_v3_fusion_scale=0 on identical mini_val endpoints."
            ),
            "decision": (
                "Separates protected observation quality from gate-induced closed-loop "
                "damage without retraining."
            ),
            "gpu_cost": "4 test passes; no training",
        },
        {
            "order": 2,
            "experiment": "Endpoint/tracklet motion attribution export",
            "change": (
                "Export observation, prior, final, GT, gate probability, correction, "
                "validity, candidate/history error, speed, and delta_t."
            ),
            "decision": (
                "Measures recursive helpful precision, cumulative drift, subgroup "
                "stability, and paired bootstrap uncertainty."
            ),
            "gpu_cost": "Reuse the 4 test passes",
        },
        {
            "order": 3,
            "experiment": "Branch on fusion-off result",
            "change": (
                "If off restores B0, freeze observation/prior and redesign only the "
                "gate. If off is also low, run one same-code scratch B0 before any "
                "motion change."
            ),
            "decision": (
                "Prevents a new gate design from masking a baseline/RNG/code-version "
                "regression."
            ),
            "gpu_cost": "No training unless off also fails",
        },
        {
            "order": 4,
            "experiment": "B1motion-v3.1 gate kill-test",
            "change": (
                "Train gate on frozen recursive mini_train rollouts; predict bounded "
                "correction benefit, remove class-balanced deployment bias, separate "
                "help probability from step size, and ramp fusion gradually."
            ),
            "decision": (
                "Continue past epoch15 only if observation identity holds, applied "
                "rate is conservative, helpful precision is high, and fused improves "
                "both standard metrics over fusion-off."
            ),
            "gpu_cost": "One short seed42 run after attribution",
        },
        {
            "order": 5,
            "experiment": "Strong cadence and causal-time controls",
            "change": (
                "Only after standard passes, evaluate gap1124/random20 and frozen "
                "true/fixed/shuffled time with per-tracklet bootstrap."
            ),
            "decision": (
                "Determines whether the paper may claim continuous-time benefit or "
                "only reliable trajectory fusion."
            ),
            "gpu_cost": "Blocked until standard pass",
        },
    ]


def rounded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if isinstance(value, float):
                converted[key] = round(value, 6) if math.isfinite(value) else None
            else:
                converted[key] = value
        result.append(converted)
    return result


def build_artifact(
        validation_rows: list[dict[str, Any]],
        summary_rows: list[dict[str, Any]],
        training_rows: list[dict[str, Any]],
        drivers: list[dict[str, Any]], next_rows: list[dict[str, Any]],
        integrity: list[dict[str, Any]], checkpoints: list[dict[str, Any]],
        seqtrack_b0_diagnostics: list[dict[str, Any]],
        generated_at: str) -> dict[str, Any]:
    summary_by_run = {row["run_id"]: row for row in summary_rows}
    seqtrack = summary_by_run["SEQTRACK"]
    b0 = summary_by_run["B0"]
    v3 = summary_by_run["B1V3"]
    v2 = summary_by_run["B1V2"]
    e60 = training_rows[-1]

    prior_e60 = [
        {
            "branch": "Main cadence",
            "learned_rmse_m": e60["main_prior_rmse_m"],
            "constant_velocity_rmse_m": e60["main_cv_rmse_m"],
            "rmse_improvement_pct": e60["main_improvement_pct"],
        },
        {
            "branch": "Aux gap2",
            "learned_rmse_m": e60["gap2_prior_rmse_m"],
            "constant_velocity_rmse_m": e60["gap2_cv_rmse_m"],
            "rmse_improvement_pct": e60["gap2_improvement_pct"],
        },
        {
            "branch": "Aux gap4",
            "learned_rmse_m": e60["gap4_prior_rmse_m"],
            "constant_velocity_rmse_m": e60["gap4_cv_rmse_m"],
            "rmse_improvement_pct": e60["gap4_improvement_pct"],
        },
    ]
    gate_epochs = [training_rows[index - 1] for index in (11, 15, 30, 45, 60)]

    validation_path = f"compare_results/data/{STEM}_validation.csv"
    training_path = f"compare_results/data/{STEM}_training_epochs.csv"
    summary_path = f"compare_results/data/{STEM}_summary.csv"
    integrity_path = f"compare_results/data/{STEM}_integrity.csv"
    checkpoint_path = f"compare_results/data/{STEM}_checkpoints.csv"
    driver_path = f"compare_results/data/{STEM}_drivers.csv"
    next_path = f"compare_results/data/{STEM}_next_experiments.csv"
    seqtrack_b0_path = (
        f"compare_results/data/{STEM}_seqtrack_b0_diagnostics.csv")
    sources = [
        {
            "id": "validation_source",
            "label": "Reviewed TensorBoard normal-validation metrics",
            "path": validation_path,
            "query": {
                "language": "python",
                "engine": "DuckDB",
                "sql": (
                    "SELECT run_id, arm, epoch, step, success, precision "
                    f"FROM read_csv_auto('{validation_path}', header=true) "
                    "ORDER BY run_id, epoch"
                ),
                "description": (
                    "tools/analyze_b1motion_v3.py extracts all twelve unsmoothed "
                    "five-epoch validation points from the original SeqTrack run, "
                    "current B0, B1motion-v2, and v3."
                ),
                "executed_at": "2026-08-01",
                "filters": [
                    "nuScenes v1.0-mini Car mini_val",
                    "normal cadence",
                    "seed=42",
                    "scratch 60 epochs",
                ],
                "metric_definitions": [
                    "Final is epoch60 and is the primary result.",
                    "Late-3 is the arithmetic mean at epochs 50, 55, and 60.",
                    "Best is diagnostic only and never replaces final.",
                ],
            },
        },
        {
            "id": "training_source",
            "label": "Reviewed B1motion-v3 training diagnostics",
            "path": training_path,
            "query": {
                "language": "python",
                "engine": "DuckDB",
                "sql": (
                    f"SELECT * FROM read_csv_auto('{training_path}', header=true) "
                    "ORDER BY epoch"
                ),
                "description": (
                    "Epoch diagnostics are regenerated from 75,720 TensorBoard "
                    "training steps. RMSE values are pooled using logged valid "
                    "sample counts; gate rates are unsmoothed batch means."
                ),
                "executed_at": "2026-08-01",
                "filters": [
                    "B1motion-v3 seed42 scratch",
                    "1,262 full-size batches per epoch",
                    "gate active from zero-based epoch10",
                ],
                "metric_definitions": [
                    "Prior improvement is (CV RMSE - learned RMSE) / CV RMSE.",
                    "Gate precision is helpful among probability>=0.5 decisive samples.",
                    "Applied rate uses decisive samples as denominator.",
                    "Actual alpha mean is alpha utilization times alpha_max=0.5.",
                ],
            },
        },
        {
            "id": "seqtrack_b0_source",
            "label": "SeqTrack-to-B0 attribution audit",
            "path": seqtrack_b0_path,
            "query": {
                "language": "sql",
                "engine": "DuckDB",
                "sql": (
                    "SELECT item, seqtrack, b0, comparison, interpretation "
                    f"FROM read_csv_auto('{seqtrack_b0_path}', header=true)"
                ),
                "description": (
                    "Recomputes validation-trajectory comparisons, late training "
                    "loss, checkpoint topology/config metadata, worker counts, and "
                    "available same-labelled SeqTrack epoch60 endpoints."),
                "executed_at": "2026-08-01",
                "metric_definitions": [
                    "Numerical gaps are descriptive and not method attribution.",
                    "The original SeqTrack run has no captured commit or dataset hashes.",
                ],
            },
        },
        {
            "id": "integrity_source",
            "label": "Run provenance and checkpoint integrity",
            "path": integrity_path,
            "query": {
                "language": "python",
                "description": (
                    "Checks run_provenance.json, training/validation counts, dataset "
                    "selection hashes, checkpoint steps, and SHA256 identity."
                ),
                "executed_at": "2026-08-01",
            },
        },
        {
            "id": "checkpoint_source",
            "label": "B1motion-v3 checkpoint parameter diagnostics",
            "path": checkpoint_path,
            "query": {
                "language": "python",
                "description": (
                    "Reads checkpoint metadata and selected physical-prior/gate "
                    "parameter norms without modifying checkpoints."
                ),
                "executed_at": "2026-08-01",
            },
        },
        {
            "id": "code_source",
            "label": "B1motion-v3 implementation audit",
            "path": "models/seqtrack3d.py",
            "query": {
                "language": "python",
                "description": (
                    "Code review covers models/seqtrack3d.py, models/ct_v2/fusion.py, "
                    "models/ct_v2/motion.py, datasets/sampler.py, and the v3 config."
                ),
                "executed_at": "2026-08-01",
                "filters": [
                    "physical xy target",
                    "detached observation/prior fusion",
                    "class-balanced helpful BCE",
                    "bounded correction",
                ],
            },
        },
        {
            "id": "driver_source",
            "label": "B1motion-v3 diagnostic evidence register",
            "path": driver_path,
            "query": {
                "language": "sql",
                "engine": "DuckDB",
                "sql": (
                    "SELECT priority, finding, evidence, interpretation, confidence "
                    f"FROM read_csv_auto('{driver_path}', header=true) "
                    "ORDER BY priority"
                ),
                "description": (
                    "Evidence register derived from reviewed validation, training, "
                    "checkpoint, and implementation sources."
                ),
                "executed_at": "2026-08-01",
            },
        },
        {
            "id": "next_source",
            "label": "Controlled B1motion follow-up register",
            "path": next_path,
            "query": {
                "language": "sql",
                "engine": "DuckDB",
                "sql": (
                    "SELECT \"order\", experiment, change, decision, gpu_cost "
                    f"FROM read_csv_auto('{next_path}', header=true) "
                    "ORDER BY \"order\""
                ),
                "description": (
                    "Ordered decision tree that requires inference-only attribution "
                    "before another training run."
                ),
                "executed_at": "2026-08-01",
            },
        },
    ]

    charts = [
        {
            "id": "success_curve",
            "title": "Normal validation Success by epoch",
            "subtitle": (
                "Twelve unsmoothed checkpoints; epoch60 is primary and best is "
                "diagnostic only."
            ),
            "type": "line",
            "dataset": "validation_metrics",
            "sourceId": "validation_source",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                "y": {"field": "success", "type": "quantitative", "label": "Success"},
                "color": {"field": "arm", "type": "nominal", "label": "Run"},
                "tooltip": [
                    {"field": "arm", "type": "nominal", "label": "Run"},
                    {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    {"field": "success", "type": "quantitative", "label": "Success"},
                ],
            },
            "xAxisTitle": "Epoch",
            "yAxisTitle": "Success",
        },
        {
            "id": "precision_curve",
            "title": "Normal validation Precision by epoch",
            "subtitle": (
                "v3 peaks at epoch30, then loses the transient precision advantage "
                "during late training."
            ),
            "type": "line",
            "dataset": "validation_metrics",
            "sourceId": "validation_source",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                "y": {"field": "precision", "type": "quantitative", "label": "Precision"},
                "color": {"field": "arm", "type": "nominal", "label": "Run"},
                "tooltip": [
                    {"field": "arm", "type": "nominal", "label": "Run"},
                    {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    {"field": "precision", "type": "quantitative", "label": "Precision"},
                ],
            },
            "xAxisTitle": "Epoch",
            "yAxisTitle": "Precision",
        },
        {
            "id": "prior_improvement",
            "title": "Epoch-60 prior RMSE improvement over constant velocity",
            "subtitle": (
                "Positive means lower learned-prior RMSE; these are training-distribution "
                "diagnostics, not recursive validation gains."
            ),
            "type": "bar",
            "dataset": "prior_e60",
            "sourceId": "training_source",
            "encodings": {
                "x": {"field": "branch", "type": "nominal", "label": "Branch"},
                "y": {
                    "field": "rmse_improvement_pct",
                    "type": "quantitative",
                    "label": "RMSE improvement (%)",
                },
                "tooltip": [
                    {"field": "branch", "type": "nominal", "label": "Branch"},
                    {
                        "field": "learned_rmse_m", "type": "quantitative",
                        "label": "Learned RMSE (m)",
                    },
                    {
                        "field": "constant_velocity_rmse_m", "type": "quantitative",
                        "label": "CV RMSE (m)",
                    },
                    {
                        "field": "rmse_improvement_pct", "type": "quantitative",
                        "label": "Improvement (%)",
                    },
                ],
            },
            "xAxisTitle": "Motion branch",
            "yAxisTitle": "RMSE improvement (%)",
        },
    ]

    tables = [
        {
            "id": "run_summary",
            "title": "Normal-validation result summary",
            "subtitle": (
                "Historical B0 and v2 context; final/late-3 are decision metrics."
            ),
            "dataset": "run_summary",
            "sourceId": "validation_source",
            "density": "compact",
            "defaultSort": {"field": "final_success", "direction": "desc"},
            "columns": [
                {"field": "arm", "label": "Run", "type": "text"},
                {"field": "final_success", "label": "Final S", "type": "number", "format": "number"},
                {"field": "final_precision", "label": "Final P", "type": "number", "format": "number"},
                {"field": "delta_final_success_vs_b0", "label": "ΔS vs B0", "type": "number", "format": "number", "semantic": "movement"},
                {"field": "delta_final_precision_vs_b0", "label": "ΔP vs B0", "type": "number", "format": "number", "semantic": "movement"},
                {"field": "late3_success", "label": "Late-3 S", "type": "number", "format": "number"},
                {"field": "late3_precision", "label": "Late-3 P", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "seqtrack_b0_table",
            "title": "Why the historical B0 endpoint is above SeqTrack",
            "subtitle": (
                "Verified matches, active stochastic differences, and provenance limits."
            ),
            "dataset": "seqtrack_b0_diagnostics",
            "sourceId": "seqtrack_b0_source",
            "density": "compact",
            "columns": [
                {"field": "item", "label": "Audit item", "type": "text"},
                {"field": "seqtrack", "label": "SeqTrack", "type": "text"},
                {"field": "b0", "label": "B0", "type": "text"},
                {"field": "comparison", "label": "Comparison", "type": "text"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "prior_table",
            "title": "Epoch-60 learned prior versus constant velocity",
            "subtitle": "RMSE in metres, pooled by logged valid sample counts.",
            "dataset": "prior_e60",
            "sourceId": "training_source",
            "density": "compact",
            "defaultSort": {"field": "rmse_improvement_pct", "direction": "desc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "learned_rmse_m", "label": "Learned RMSE", "type": "number", "format": "number"},
                {"field": "constant_velocity_rmse_m", "label": "CV RMSE", "type": "number", "format": "number"},
                {"field": "rmse_improvement_pct", "label": "Improvement %", "type": "number", "format": "number", "semantic": "movement"},
            ],
        },
        {
            "id": "gate_table",
            "title": "Gate calibration during active training",
            "subtitle": (
                "Rates are unsmoothed batch means; alpha is the actual applied mean."
            ),
            "dataset": "gate_epochs",
            "sourceId": "training_source",
            "density": "compact",
            "defaultSort": {"field": "epoch", "direction": "asc"},
            "columns": [
                {"field": "epoch", "label": "Epoch", "type": "number"},
                {"field": "observation_error_m", "label": "Obs err (m)", "type": "number", "format": "number"},
                {"field": "prior_box_error_m", "label": "Prior err (m)", "type": "number", "format": "number"},
                {"field": "helpful_rate", "label": "Helpful rate", "type": "number", "format": "percent"},
                {"field": "gate_applied_rate", "label": "Applied rate", "type": "number", "format": "percent"},
                {"field": "gate_precision", "label": "Gate precision", "type": "number", "format": "percent"},
                {"field": "actual_alpha_mean", "label": "Mean α", "type": "number", "format": "number"},
                {"field": "correction_norm_m", "label": "Correction (m)", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "driver_table",
            "title": "Failure-mechanism evidence register",
            "subtitle": "Observed evidence is separated from interpretation and confidence.",
            "dataset": "drivers",
            "sourceId": "driver_source",
            "density": "compact",
            "defaultSort": {"field": "priority", "direction": "asc"},
            "columns": [
                {"field": "priority", "label": "#", "type": "number"},
                {"field": "finding", "label": "Finding", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
                {"field": "confidence", "label": "Confidence", "type": "text"},
            ],
        },
        {
            "id": "next_table",
            "title": "Ordered next experiments",
            "subtitle": "Inference-only attribution comes before another training run.",
            "dataset": "next_experiments",
            "sourceId": "next_source",
            "density": "compact",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "#", "type": "number"},
                {"field": "experiment", "label": "Experiment", "type": "text"},
                {"field": "change", "label": "Change", "type": "text"},
                {"field": "decision", "label": "Decision use", "type": "text"},
                {"field": "gpu_cost", "label": "Cost", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# B1motion-v3 seed42 60-epoch 技术复核"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "validation_source",
            "body": (
                "## 技术结论：修复了 v2，但没有超过 B0\n\n"
                f"B1motion-v3 epoch60 为 **{v3['final_success']:.3f} Success / "
                f"{v3['final_precision']:.3f} Precision**，相对历史 B0 为 "
                f"**{v3['delta_final_success_vs_b0']:+.3f} / "
                f"{v3['delta_final_precision_vs_b0']:+.3f}**；late-3 为 "
                f"**{v3['delta_late3_success_vs_b0']:+.3f} / "
                f"{v3['delta_late3_precision_vs_b0']:+.3f}**。因此它未通过 "
                "final 的 +0.5/+1.0 门槛，也未通过 late-3 非负门槛。相对 v2 "
                f"则恢复 **{v3['final_success']-v2['final_success']:+.3f} / "
                f"{v3['final_precision']-v2['final_precision']:+.3f}**：v3 是成功的"
                "安全性重构，但当前不是相对 current B0 的涨点模块。若只和原始 "
                f"SeqTrack3D plain 的 **{seqtrack['final_success']:.3f}/"
                f"{seqtrack['final_precision']:.3f}** 比，v3 为 **"
                f"{v3['final_success']-seqtrack['final_success']:+.3f}/"
                f"{v3['final_precision']-seqtrack['final_precision']:+.3f}**；但 current "
                f"B0 相对原始 SeqTrack 已有 **"
                f"{b0['final_success']-seqtrack['final_success']:+.3f}/"
                f"{b0['final_precision']-seqtrack['final_precision']:+.3f}**，所以这部分"
                "正差不能归因给 motion。"
            ),
        },
        {"id": "summary_table_block", "type": "table", "tableId": "run_summary"},
        {
            "id": "seqtrack_b0_attribution",
            "type": "markdown",
            "sourceId": "seqtrack_b0_source",
            "body": (
                "## B0 为什么在 epoch60 高于原始 SeqTrack\n\n"
                "这不是 CT 模块或更大网络带来的提升：两个 last checkpoint 都是 "
                "**320 个同名同 shape tensors、3,718,065 个参数元素**，训练预算、"
                "损失权重、优化器、学习率计划、PL 2.0.2 和验证主路径均对齐。B0 "
                "只是在 final/late training 更高；原始 SeqTrack 的历史最好 Precision "
                "为 **65.214**，反而高于 B0 的 **64.382**，且 B0 前20轮多次明显"
                "落后。两者 epoch56–60 mean total loss 也仅为 **0.22286 vs "
                "0.22086**。\n\n"
                "当前最可能的直接驱动是随机训练流不同：原 run 使用 **12 workers**，"
                "B0 使用 **4 workers**；候选框扰动和点采样依赖 worker 内 NumPy "
                "随机流，所以同为 seed42 并不代表看到了同一批随机候选与点子集。"
                "这些小差异会通过递归 tracker history 被放大为数分的 endpoint "
                "差异。B0 sampler 还把 candidate offset 显式转为 float32，并计算"
                "额外但关闭的 CT 诊断字段，说明它与旧代码也不是 bitwise 同一路径。\n\n"
                "归因上仍有硬限制：原始 SeqTrack 输出没有 run_provenance.json，无法"
                "证明训练时的 exact commit、dirty state 和 dataset selection hash。"
                "因此正确表述是：**B0 的这次 scratch run 获得了更好的后期随机训练"
                "轨迹和稳定 endpoint；尚无证据表明 B0 方法本身优于 SeqTrack。**"
            ),
        },
        {"id": "seqtrack_b0_table_block", "type": "table", "tableId": "seqtrack_b0_table"},
        {
            "id": "curve_finding",
            "type": "markdown",
            "sourceId": "validation_source",
            "body": (
                "## 中期正信号没有保持到 late training\n\n"
                "v3 在 epoch20–30 出现明显的 Precision 正信号，epoch30 达到 "
                f"**{v3['best_precision']:.3f} Precision / "
                f"{v3['paired_success_at_best_precision']:.3f} Success**；但这不是"
                "联合涨点，且 epoch35 后回落。下面两条曲线应按同一规则阅读："
                "epoch60 与 late-3 决定晋级，best 只用于定位不稳定时点。"
            ),
        },
        {"id": "success_chart_block", "type": "chart", "chartId": "success_curve"},
        {"id": "precision_chart_block", "type": "chart", "chartId": "precision_curve"},
        {
            "id": "prior_finding",
            "type": "markdown",
            "sourceId": "training_source",
            "body": (
                "## Prior 已经学到东西，主要问题转向融合\n\n"
                f"epoch60 的 learned prior 相对 constant velocity 在主分支降低 "
                f"RMSE **{e60['main_improvement_pct']:.1f}%**，gap2 降低 "
                f"**{e60['gap2_improvement_pct']:.1f}%**，gap4 降低 "
                f"**{e60['gap4_improvement_pct']:.1f}%**。candidate0 与 nonzero "
                f"也分别改善 **{e60['candidate0_improvement_pct']:.1f}% / "
                f"{e60['candidate_nonzero_improvement_pct']:.1f}%**。这不证明"
                "递归验证有效，但足以排除“GRU 容量不足或 prior 完全没学会”作为"
                "当前首要解释。"
            ),
        },
        {"id": "prior_chart_block", "type": "chart", "chartId": "prior_improvement"},
        {"id": "prior_table_block", "type": "table", "tableId": "prior_table"},
        {
            "id": "gate_finding",
            "type": "markdown",
            "sourceId": "training_source",
            "body": (
                "## Gate 没有随 observation 变强而关闭\n\n"
                "第一轮 active gate 时 prior box error 低于 observation，helpful "
                "样本约占 70%；到 epoch60，observation error 已降至 "
                f"**{e60['observation_error_m']:.3f} m**，优于 prior 的 "
                f"**{e60['prior_box_error_m']:.3f} m**，helpful prevalence 只剩 "
                f"**{100*e60['helpful_rate']:.1f}%**。但 gate 仍对约 "
                f"**{100*e60['gate_applied_rate']:.1f}%** 的 decisive 样本应用"
                f"修正，precision 只有 **{100*e60['gate_precision']:.1f}%**，实际"
                f"平均 α 约 **{e60['actual_alpha_mean']:.3f}**。balanced BCE 将"
                "分类边界和部署频率混在一起，无法提供 conservative calibration。"
            ),
        },
        {"id": "gate_table_block", "type": "table", "tableId": "gate_table"},
        {
            "id": "mechanism_finding",
            "type": "markdown",
            "sourceId": "driver_source",
            "body": (
                "## 一步训练收益没有转成递归 tracking 收益\n\n"
                f"epoch60 训练 batch 上 final error 为 **{e60['final_error_m']:.3f} m**，"
                f"比 observation 的 **{e60['observation_error_m']:.3f} m** 略低；"
                "然而 recursive mini_val 低于 B0。最符合现有证据的解释是：gate "
                "在 candidate-perturbed、单步监督上优化，而部署在 predicted-history "
                "闭环；小而频繁的错误修正会沿 tracklet 累积。该判断仍需同 checkpoint "
                "fusion-off endpoint 结果确认。"
            ),
        },
        {"id": "driver_table_block", "type": "table", "tableId": "driver_table"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "integrity_source",
            "body": (
                "## 数据范围与指标口径\n\n"
                "实验为 nuScenes v1.0-mini Car、mini_train 274 tracklets/5,051 "
                "frames、mini_val 106 tracklets/2,285 frames、seed42、batch16、"
                "60 epoch、每 5 epoch 验证一次。Success/Precision 是递归 tracking "
                "聚合指标；final=epoch60，late-3=epoch50/55/60 均值。训练 RMSE "
                "使用 logged valid count 聚合；gate rate 是未平滑 batch mean。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "code_source",
            "body": (
                "## 复核方法\n\n"
                "分析脚本从三组 TensorBoard event 读取全部验证点，从 v3 的 75,720 "
                "training scalars 重算每轮 prior/CV 与 gate 指标，并检查 provenance、"
                "dataset selection hash、checkpoint global step 和 SHA256。物理 prior "
                "与 observation 在融合损失中 detach，xy correction 受 α 和半径限制；"
                "因此本文将 prior 学习质量与 gate 闭环效果分开判断。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 证据边界与稳健性\n\n"
                "当前只有 seed42、normal mini_val 聚合值；没有 per-tracklet bootstrap，"
                "不能把 -0.705/-2.547 解释为已量化的统计显著退化，但它明确不是预设"
                "意义上的涨点。历史 B0 来自不同 commit，v3 运行时 tracked source "
                "为 dirty；虽然 dataset selection 完全匹配且运行完整，仍缺少同代码 B0 "
                "或同 checkpoint fusion-off 对照。尚未运行 gap1124、random20、"
                "true/fixed/shuffled，因此不能声称连续时间带来因果收益。原始 "
                "SeqTrack run 的 checkpoint 可确认协议字段与训练步，但没有当时的 "
                "run_provenance.json，属于跨代码版本历史对照。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "next_source",
            "body": (
                "## 下一步：先做四次推理归因，不再直接开 60 轮\n\n"
                "立即评测 epoch30 与 epoch60 的 fusion on/off 四格，并在相同 endpoint "
                "导出 observation/prior/final/gate/GT。若 fusion-off 恢复 B0，只重做 "
                "gate：在 frozen recursive mini_train rollout 上学习 bounded correction "
                "的实际收益，取消 class-balanced 部署偏置，把 helpful probability 与 "
                "step size 分开，并使用渐进 ramp。若 fusion-off 也低，则先补 same-code "
                "B0，检查 main-view/RNG/代码版本，禁止用新 gate 掩盖基线问题。"
            ),
        },
        {"id": "next_table_block", "type": "table", "tableId": "next_table"},
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "1. v3 的 observation-only 是否与同代码 B0 一致？\n\n"
                "2. gate 的错误主要集中在哪些 tracklet、速度、history error 和 "
                "prior/observation disagreement 桶？\n\n"
                "3. e30 的 precision 峰值来自 observation、prior 还是 gate，是否能在 "
                "fusion-off 配对结果中复现？\n\n"
                "4. 只有 standard 同 checkpoint 同时提高两项后，true-time 才值得进入 "
                "gap1124 的固定/打乱时间因果控制。"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "B1motion-v3 seed42 60-epoch 技术复核",
            "description": (
                "Normal-cadence outcome, physical-prior learning, gate calibration, "
                "limitations, and the next attribution experiment."
            ),
            "generatedAt": generated_at,
            "sources": sources,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "validation_metrics": rounded_rows(validation_rows),
                "run_summary": rounded_rows(summary_rows),
                "prior_e60": rounded_rows(prior_e60),
                "gate_epochs": rounded_rows(gate_epochs),
                "drivers": drivers,
                "next_experiments": next_rows,
                "seqtrack_b0_diagnostics": seqtrack_b0_diagnostics,
                "integrity": integrity,
                "checkpoints": rounded_rows(checkpoints),
            },
        },
        "sources": sources,
        "package_info": {
            "generator": "tools/analyze_b1motion_v3.py",
            "generatedAt": generated_at,
        },
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    validation_rows, validation = collect_validation()
    summary_rows = summarize_validation(validation)
    training_rows, per_epoch = collect_training()
    checkpoints = checkpoint_rows()
    integrity = integrity_rows(
        validation, training_rows, per_epoch, checkpoints)
    drivers = build_driver_rows(summary_rows, training_rows)
    next_rows = build_next_rows()
    seqtrack_b0_diagnostics = seqtrack_b0_diagnostic_rows(validation)

    outputs = {
        "validation": validation_rows,
        "summary": summary_rows,
        "training_epochs": training_rows,
        "checkpoints": checkpoints,
        "integrity": integrity,
        "drivers": drivers,
        "next_experiments": next_rows,
        "seqtrack_b0_diagnostics": seqtrack_b0_diagnostics,
    }
    for suffix, rows in outputs.items():
        write_csv(DATA_DIR / f"{STEM}_{suffix}.csv", rounded_rows(rows))

    artifact = build_artifact(
        validation_rows, summary_rows, training_rows, drivers, next_rows,
        integrity, checkpoints, seqtrack_b0_diagnostics, generated_at)
    artifact_path = REPORT_DIR / f"{STEM}_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    by_run = {row["run_id"]: row for row in summary_rows}
    v3 = by_run["B1V3"]
    e60 = training_rows[-1]
    print(json.dumps({
        "artifact": str(artifact_path.relative_to(ROOT)),
        "steps_per_epoch": per_epoch,
        "final_v3": [v3["final_success"], v3["final_precision"]],
        "final_delta_vs_b0": [
            v3["delta_final_success_vs_b0"],
            v3["delta_final_precision_vs_b0"],
        ],
        "late3_delta_vs_b0": [
            v3["delta_late3_success_vs_b0"],
            v3["delta_late3_precision_vs_b0"],
        ],
        "epoch60_prior_improvement_pct": {
            "main": e60["main_improvement_pct"],
            "gap2": e60["gap2_improvement_pct"],
            "gap4": e60["gap4_improvement_pct"],
        },
        "epoch60_gate": {
            "actual_alpha": e60["actual_alpha_mean"],
            "applied_rate": e60["gate_applied_rate"],
            "precision": e60["gate_precision"],
            "helpful_rate": e60["helpful_rate"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
