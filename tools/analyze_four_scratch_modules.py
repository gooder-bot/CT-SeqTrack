#!/usr/bin/env python3
"""Audit the four pulled scratch-module runs and compare them with SeqTrack.

The output is deliberately marked partial when either M3 run is not a complete
60-epoch result.  TensorBoard scalars are treated as repeated observations from
one training run, not independent statistical replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import torch
from nbclient import NotebookClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ANALYSIS_DATE = "20260725"
GENERATED_AT = "2026-07-25T18:00:00+08:00"
STEPS_PER_EPOCH = 1262
EXPECTED_EPOCHS = 60
EXPECTED_STEPS = EXPECTED_EPOCHS * STEPS_PER_EPOCH
EXPECTED_VAL_POINTS = 12

RUNS = {
    "SEQ": {
        "label": "SeqTrack historical baseline",
        "short": "SeqTrack",
        "root": "../seqtrack/output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16",
        "role": "historical baseline",
        "module": "original SeqTrack",
    },
    "W0": {
        "label": "W0 shared-SE(2)",
        "short": "W0",
        "root": "output/scratch_w0_shared_se2_seed42",
        "role": "current-code control",
        "module": "shared-SE(2), M2 off, M3 off",
    },
    "M2": {
        "label": "M2 dynamics bundle",
        "short": "M2",
        "root": "output/scratch_m2_seed42",
        "role": "complete treatment",
        "module": "shared-SE(2) + dynamics/adapter/proposal innovation",
    },
    "B": {
        "label": "M2 + paired EMA path, weight 0",
        "short": "M3-w0",
        "root": "output/scratch_m2_paired_weight0_seed42",
        "role": "partial M3 compute-path control",
        "module": "M2 + paired/teacher path; distillation weight 0",
    },
    "C": {
        "label": "M2 + M3 endpoint distillation",
        "short": "M3-w.05",
        "root": "output/scratch_m2_m3_seed42",
        "role": "partial M3 treatment",
        "module": "M2 + endpoint distillation; target weight 0.05",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_dir(repo_root: Path, run_id: str) -> Path:
    return (repo_root / RUNS[run_id]["root"] / "lightning_logs/version_0").resolve()


def read_scalar(directory: Path, leaf: str) -> pd.DataFrame:
    event_dir = directory / leaf
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if len(tags) != 1:
        raise RuntimeError(f"Expected one scalar tag in {event_dir}; found {tags}")
    rows = [
        {"step": int(event.step), "value": float(event.value)}
        for event in accumulator.Scalars(tags[0])
    ]
    return pd.DataFrame(rows)


def install_easydict_pickle_shim() -> None:
    if "easydict" in sys.modules:
        return
    module = types.ModuleType("easydict")
    easy_dict = type("EasyDict", (dict,), {})
    easy_dict.__module__ = "easydict"
    module.EasyDict = easy_dict
    sys.modules["easydict"] = module


def load_checkpoint(path: Path) -> dict[str, Any]:
    install_easydict_pickle_shim()
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = load_checkpoint(path)
    state = checkpoint.get("state_dict", {})
    metadata = {
        "checkpoint_bytes": int(path.stat().st_size),
        "checkpoint_mib": path.stat().st_size / (1024 * 1024),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_epoch_zero_based": int(checkpoint.get("epoch", -1)),
        "checkpoint_human_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_tensor_count": len(state),
    }
    return metadata, state


def collect_metrics(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for run_id, meta in RUNS.items():
        directory = version_dir(repo_root, run_id)
        for metric, leaf in [
            ("Success", "metrics_test_success"),
            ("Precision", "metrics_test_precision"),
        ]:
            points = read_scalar(directory, leaf)
            for row in points.itertuples(index=False):
                point_rows.append(
                    {
                        "run_id": run_id,
                        "run": meta["short"],
                        "run_label": meta["label"],
                        "metric": metric,
                        "step": row.step,
                        "epoch": int(round(row.step / STEPS_PER_EPOCH)),
                        "value": row.value,
                    }
                )
    points_df = pd.DataFrame(point_rows).sort_values(
        ["run_id", "metric", "epoch"]
    )
    for run_id, meta in RUNS.items():
        subset = points_df[points_df["run_id"] == run_id]
        row: dict[str, Any] = {
            "run_id": run_id,
            "run": meta["short"],
            "run_label": meta["label"],
            "role": meta["role"],
            "module": meta["module"],
            "validation_points": int(
                (subset["metric"] == "Success").sum()
            ),
            "last_validation_epoch": int(subset["epoch"].max()),
        }
        for metric in ["Success", "Precision"]:
            series = subset[subset["metric"] == metric].sort_values("epoch")
            key = metric.lower()
            best = series.loc[series["value"].idxmax()]
            row[f"latest_{key}"] = float(series.iloc[-1]["value"])
            row[f"best_{key}"] = float(best["value"])
            row[f"best_{key}_epoch"] = int(best["epoch"])
            late = series[series["epoch"].between(40, 60)]
            row[f"late_mean_{key}"] = float(late["value"].mean())
            row[f"late_std_{key}"] = float(late["value"].std(ddof=0))
            row[f"all_mean_{key}"] = float(series["value"].mean())
        summary_rows.append(row)
    return points_df.reset_index(drop=True), pd.DataFrame(summary_rows)


def collect_integrity(
    repo_root: Path, metrics_summary: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    by_id = metrics_summary.set_index("run_id")
    for run_id, meta in RUNS.items():
        root = (repo_root / meta["root"]).resolve()
        checkpoint_path = version_dir(repo_root, run_id) / "checkpoints/last.ckpt"
        checkpoint, state = checkpoint_metadata(checkpoint_path)
        states[run_id] = state
        loss_points = read_scalar(
            version_dir(repo_root, run_id), "loss_loss_total"
        )
        provenance_path = root / "run_provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.exists()
            else None
        )
        complete = (
            checkpoint["checkpoint_epoch_zero_based"] == 59
            and checkpoint["checkpoint_global_step"] == EXPECTED_STEPS
            and by_id.loc[run_id, "validation_points"] == EXPECTED_VAL_POINTS
            and by_id.loc[run_id, "last_validation_epoch"] == EXPECTED_EPOCHS
        )
        rows.append(
            {
                "run_id": run_id,
                "run": meta["short"],
                "status": "COMPLETE" if complete else "PARTIAL",
                "checkpoint_epoch": checkpoint["checkpoint_human_epoch"],
                "checkpoint_step": checkpoint["checkpoint_global_step"],
                "latest_loss_step": int(loss_points["step"].max()),
                "estimated_latest_train_epoch": float(
                    loss_points["step"].max() / STEPS_PER_EPOCH
                ),
                "validation_points": int(by_id.loc[run_id, "validation_points"]),
                "last_validation_epoch": int(
                    by_id.loc[run_id, "last_validation_epoch"]
                ),
                "state_tensors": checkpoint["checkpoint_tensor_count"],
                "checkpoint_mib": checkpoint["checkpoint_mib"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "git_commit": (
                    provenance["git"]["commit"] if provenance else "unrecorded"
                ),
                "dirty_tracked": (
                    provenance["git"]["dirty_tracked"] if provenance else None
                ),
                "provenance": "current recorded" if provenance else "legacy",
            }
        )
    integrity_df = pd.DataFrame(rows)
    seq_shapes = {
        key: tuple(value.shape) for key, value in states["SEQ"].items()
    }
    w0_shapes = {
        key: tuple(value.shape) for key, value in states["W0"].items()
    }
    architecture_equal = seq_shapes == w0_shapes
    integrity_df["seqtrack_w0_state_schema_equal"] = ""
    integrity_df.loc[
        integrity_df["run_id"].isin(["SEQ", "W0"]),
        "seqtrack_w0_state_schema_equal",
    ] = str(architecture_equal)
    return integrity_df, states


def comparison_rows(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    by_id = metrics_summary.set_index("run_id")
    specs = [
        (
            "W0 − SeqTrack",
            "W0",
            "SEQ",
            "historical cross-code comparison",
            "Shared-SE(2)/current pipeline is confounded with code vintage.",
        ),
        (
            "M2 − W0",
            "M2",
            "W0",
            "matched current-code scratch bundle comparison",
            "Attributes the full M2 bundle, not its individual submodules.",
        ),
        (
            "M2 − SeqTrack",
            "M2",
            "SEQ",
            "historical reference comparison",
            "One seed; baseline lacks current provenance and exact code parity.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, treatment, control, evidence, caveat in specs:
        for scope, prefix in [
            ("latest/final", "latest"),
            ("late mean epoch40–60", "late_mean"),
            ("all validation points", "all_mean"),
        ]:
            for metric in ["success", "precision"]:
                treatment_value = float(
                    by_id.loc[treatment, f"{prefix}_{metric}"]
                )
                control_value = float(by_id.loc[control, f"{prefix}_{metric}"])
                rows.append(
                    {
                        "comparison": name,
                        "treatment": treatment,
                        "control": control,
                        "scope": scope,
                        "metric": metric.capitalize(),
                        "treatment_value": treatment_value,
                        "control_value": control_value,
                        "delta_points": treatment_value - control_value,
                        "evidence": evidence,
                        "caveat": caveat,
                    }
                )
    return pd.DataFrame(rows)


def paired_validation_comparisons(points_df: pd.DataFrame) -> pd.DataFrame:
    wide = points_df.pivot_table(
        index=["metric", "epoch"], columns="run_id", values="value"
    ).reset_index()
    rows: list[dict[str, Any]] = []
    specs = [
        ("M2 − W0", "M2", "W0", None),
        ("M2 − SeqTrack", "M2", "SEQ", None),
        ("M3-w0 − M2", "B", "M2", 40),
        ("M3-w.05 − M3-w0", "C", "B", 40),
        ("M3-w.05 − M2", "C", "M2", 45),
    ]
    for comparison, treatment, control, max_epoch in specs:
        subset = wide.dropna(subset=[treatment, control]).copy()
        if max_epoch is not None:
            subset = subset[subset["epoch"] <= max_epoch]
        for metric in ["Success", "Precision"]:
            series = subset[subset["metric"] == metric].copy()
            series["delta"] = series[treatment] - series[control]
            for epoch_floor, scope in [
                (0, "all shared validation points"),
                (25, "epoch25+ shared validation points"),
                (40, "epoch40+ shared validation points"),
            ]:
                scoped = series[series["epoch"] >= epoch_floor]
                if scoped.empty:
                    continue
                rows.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "scope": scope,
                        "n_points": int(scoped.shape[0]),
                        "first_epoch": int(scoped["epoch"].min()),
                        "last_epoch": int(scoped["epoch"].max()),
                        "mean_delta_points": float(scoped["delta"].mean()),
                        "positive_points": int((scoped["delta"] > 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def scalar_epoch_means(
    repo_root: Path, run_id: str, leaves: dict[str, str]
) -> pd.DataFrame:
    frames = []
    for metric, leaf in leaves.items():
        frame = read_scalar(version_dir(repo_root, run_id), leaf)
        frame["epoch"] = (frame["step"] // STEPS_PER_EPOCH) + 1
        grouped = (
            frame.groupby("epoch", as_index=False)
            .agg(value=("value", "mean"), points=("value", "size"))
        )
        grouped["metric"] = metric
        grouped["run_id"] = run_id
        grouped["run"] = RUNS[run_id]["short"]
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def collect_m3_diagnostics(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leaves = {
        "M3 path loss": "loss_loss_m3_path",
        "Irregular view B diagnostic GT loss": "loss_loss_total_b",
        "Canonical view A supervised loss": "loss_loss_total_a",
        "Center endpoint gap (m)": "loss_m3_center_gap",
        "Yaw endpoint gap (rad)": "loss_m3_yaw_gap",
        "Valid pair ratio": "loss_m3_valid_ratio",
        "Teacher confidence": "loss_m3_teacher_confidence",
        "Effective sample weight": "loss_m3_effective_sample_weight",
        "Effective M3 objective weight": "loss_m3_path_weight_effective",
    }
    epoch_means = pd.concat(
        [
            scalar_epoch_means(repo_root, run_id, leaves)
            for run_id in ["B", "C"]
        ],
        ignore_index=True,
    )
    late_rows: list[dict[str, Any]] = []
    for metric in leaves:
        subset = epoch_means[
            (epoch_means["metric"] == metric)
            & epoch_means["epoch"].between(31, 40)
        ]
        values = subset.pivot_table(
            index="epoch", columns="run_id", values="value"
        ).dropna()
        b_mean = float(values["B"].mean())
        c_mean = float(values["C"].mean())
        late_rows.append(
            {
                "metric": metric,
                "window": "epoch31–40",
                "M3_w0_mean": b_mean,
                "M3_w05_mean": c_mean,
                "absolute_change": c_mean - b_mean,
                "relative_change_pct": (
                    100.0 * (c_mean - b_mean) / b_mean
                    if abs(b_mean) > 1e-12
                    else np.nan
                ),
            }
        )
    late_effects = pd.DataFrame(late_rows)

    b_loss = read_scalar(version_dir(repo_root, "B"), "loss_loss_total")
    c_loss = read_scalar(version_dir(repo_root, "C"), "loss_loss_total")
    early = b_loss.merge(c_loss, on="step", suffixes=("_B", "_C"))
    early = early[early["step"] <= 10 * STEPS_PER_EPOCH].copy()
    early["abs_diff"] = (early["value_B"] - early["value_C"]).abs()
    tolerance = 1e-12
    differing = early["abs_diff"] > tolerance
    first_differing_step = (
        int(early.loc[differing, "step"].iloc[0]) if differing.any() else None
    )
    reproducibility = pd.DataFrame(
        [
            {
                "window": "first 10 training epochs (effective M3 weight = 0)",
                "shared_steps": int(early.shape[0]),
                "different_steps": int(differing.sum()),
                "different_fraction": float(differing.mean()),
                "mean_absolute_loss_difference": float(
                    early["abs_diff"].mean()
                ),
                "first_differing_step": first_differing_step,
                "interpretation": (
                    "Same nominal seed is not bitwise deterministic; "
                    "single-seed B−C is not a strict causal estimate."
                ),
            }
        ]
    )
    return epoch_means, late_effects, reproducibility


def config_and_provenance(repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = [
        "candidate_trajectory_mode",
        "use_dynamics_encoder",
        "use_physical_time_adapter",
        "dynamics_motion_mode",
        "m3_variant",
        "m3_path_weight",
        "m3_irregular_supervision_weight",
        "m3_warmup_epoch",
        "m3_ramp_epochs",
        "seed",
        "batch_size",
        "workers",
        "epoch",
        "check_val_every_n_epoch",
    ]
    configs: dict[str, dict[str, Any]] = {}
    for run_id in ["W0", "M2", "B", "C"]:
        root = (repo_root / RUNS[run_id]["root"]).resolve()
        provenance = json.loads(
            (root / "run_provenance.json").read_text(encoding="utf-8")
        )
        configs[run_id] = provenance["resolved_config"]
    for field in fields:
        row: dict[str, Any] = {"field": field}
        for run_id in ["W0", "M2", "B", "C"]:
            row[run_id] = configs[run_id].get(field, "field absent")
        rows.append(row)
    return pd.DataFrame(rows)


def tidy_validation_chart_data(points_df: pd.DataFrame) -> tuple[list, list]:
    success = (
        points_df[points_df["metric"] == "Success"][
            ["epoch", "run", "value"]
        ]
        .rename(columns={"value": "score"})
        .to_dict(orient="records")
    )
    precision = (
        points_df[points_df["metric"] == "Precision"][
            ["epoch", "run", "value"]
        ]
        .rename(columns={"value": "score"})
        .to_dict(orient="records")
    )
    return success, precision


def records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    cleaned = frame[columns].copy()
    cleaned = cleaned.replace({np.nan: None})
    result = cleaned.to_dict(orient="records")
    for row in result:
        for key, value in row.items():
            if isinstance(value, np.generic):
                row[key] = value.item()
    return result


def write_artifact(
    repo_root: Path,
    points_df: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    integrity_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    late_effects: pd.DataFrame,
    m3_epoch_means: pd.DataFrame,
    reproducibility: pd.DataFrame,
) -> Path:
    report_dir = (
        repo_root
        / "compare_results/reports/m_stage_seed42_partial_20260725"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    success_curve, precision_curve = tidy_validation_chart_data(points_df)

    summary_table = metrics_summary.copy()
    summary_table["latest_score_epoch"] = summary_table["last_validation_epoch"]
    summary_table["status"] = summary_table["run_id"].map(
        integrity_df.set_index("run_id")["status"]
    )
    summary_rows = records(
        summary_table,
        [
            "run",
            "status",
            "latest_score_epoch",
            "latest_success",
            "latest_precision",
            "best_success",
            "best_success_epoch",
            "best_precision",
            "best_precision_epoch",
        ],
    )
    integrity_rows = records(
        integrity_df,
        [
            "run",
            "status",
            "checkpoint_epoch",
            "latest_loss_step",
            "last_validation_epoch",
            "validation_points",
            "state_tensors",
            "checkpoint_mib",
            "provenance",
        ],
    )
    comparison_rows_snapshot = records(
        comparisons_df[
            comparisons_df["scope"].isin(
                ["latest/final", "late mean epoch40–60"]
            )
        ],
        [
            "comparison",
            "scope",
            "metric",
            "delta_points",
            "evidence",
        ],
    )
    selected_effects = late_effects[
        late_effects["metric"].isin(
            [
                "M3 path loss",
                "Irregular view B diagnostic GT loss",
                "Canonical view A supervised loss",
                "Center endpoint gap (m)",
                "Yaw endpoint gap (rad)",
            ]
        )
    ].copy()
    effect_rows = records(
        selected_effects,
        [
            "metric",
            "M3_w0_mean",
            "M3_w05_mean",
            "absolute_change",
            "relative_change_pct",
        ],
    )
    m3_path_curve = records(
        m3_epoch_means[
            (m3_epoch_means["metric"] == "M3 path loss")
            & (m3_epoch_means["epoch"] <= 40)
        ],
        ["epoch", "run", "value"],
    )
    m3_irregular_curve = records(
        m3_epoch_means[
            (
                m3_epoch_means["metric"]
                == "Irregular view B diagnostic GT loss"
            )
            & (m3_epoch_means["epoch"] <= 40)
        ],
        ["epoch", "run", "value"],
    )
    reproducibility_rows = records(
        reproducibility,
        [
            "window",
            "shared_steps",
            "different_steps",
            "different_fraction",
            "mean_absolute_loss_difference",
            "first_differing_step",
            "interpretation",
        ],
    )

    sources = [
        {
            "id": "validation_metrics_source",
            "label": "TensorBoard validation metric extraction",
            "path": "compare_results/data/four_scratch_validation_points_20260725.csv",
        },
        {
            "id": "integrity_source",
            "label": "Checkpoint and run completeness audit",
            "path": "compare_results/data/four_scratch_integrity_20260725.csv",
        },
        {
            "id": "m3_diagnostics_source",
            "label": "M3 TensorBoard diagnostic epoch means",
            "path": "compare_results/data/four_scratch_m3_epoch_diagnostics_20260725.csv",
        },
        {
            "id": "reproducibility_source",
            "label": "B/C early-trajectory reproducibility audit",
            "path": "compare_results/data/four_scratch_reproducibility_20260725.csv",
        },
        {
            "id": "notebook_source",
            "label": "Executed reproducible analysis notebook",
            "path": "compare_results/notebooks/four_scratch_module_analysis_20260725.ipynb",
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "CT-SeqTrack 四组模块实验阶段性分析（seed42）",
        "generatedAt": GENERATED_AT,
        "charts": [
            {
                "id": "success_curve_chart",
                "title": "标准 mini_val Success（每 5 epoch）",
                "type": "line",
                "dataset": "success_curve",
                "sourceId": "validation_metrics_source",
                "encodings": {
                    "x": {
                        "field": "epoch",
                        "type": "quantitative",
                        "label": "Epoch",
                    },
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "label": "Success (points)",
                    },
                    "color": {
                        "field": "run",
                        "type": "nominal",
                        "label": "Run",
                    },
                    "tooltip": [
                        {"field": "run", "type": "nominal", "label": "Run"},
                        {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        {
                            "field": "score",
                            "type": "quantitative",
                            "label": "Success",
                        },
                    ],
                },
            },
            {
                "id": "precision_curve_chart",
                "title": "标准 mini_val Precision（每 5 epoch）",
                "type": "line",
                "dataset": "precision_curve",
                "sourceId": "validation_metrics_source",
                "encodings": {
                    "x": {
                        "field": "epoch",
                        "type": "quantitative",
                        "label": "Epoch",
                    },
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "label": "Precision (points)",
                    },
                    "color": {
                        "field": "run",
                        "type": "nominal",
                        "label": "Run",
                    },
                },
            },
            {
                "id": "m3_effect_chart",
                "title": "M3-w.05 相对 M3-w0 的 epoch31–40 诊断变化",
                "type": "bar",
                "dataset": "m3_effects",
                "sourceId": "m3_diagnostics_source",
                "encodings": {
                    "x": {
                        "field": "metric",
                        "type": "nominal",
                        "label": "Diagnostic",
                    },
                    "y": {
                        "field": "relative_change_pct",
                        "type": "quantitative",
                        "label": "Relative change (%)",
                    },
                    "tooltip": [
                        {
                            "field": "metric",
                            "type": "nominal",
                            "label": "Diagnostic",
                        },
                        {
                            "field": "relative_change_pct",
                            "type": "quantitative",
                            "label": "Relative change (%)",
                        },
                    ],
                },
            },
            {
                "id": "m3_path_curve_chart",
                "title": "M3 path loss 训练期均值",
                "type": "line",
                "dataset": "m3_path_curve",
                "sourceId": "m3_diagnostics_source",
                "encodings": {
                    "x": {
                        "field": "epoch",
                        "type": "quantitative",
                        "label": "Epoch",
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative",
                        "label": "Path loss",
                    },
                    "color": {
                        "field": "run",
                        "type": "nominal",
                        "label": "Run",
                    },
                },
            },
        ],
        "tables": [
            {
                "id": "run_summary_table",
                "title": "验证指标摘要（partial 表示不是 60-epoch final）",
                "dataset": "run_summary",
                "sourceId": "validation_metrics_source",
                "defaultSort": {"field": "latest_score_epoch", "direction": "desc"},
                "columns": [
                    {"field": "run", "label": "Run"},
                    {"field": "status", "label": "Status"},
                    {"field": "latest_score_epoch", "label": "Latest epoch"},
                    {"field": "latest_success", "label": "Latest Success"},
                    {"field": "latest_precision", "label": "Latest Precision"},
                    {"field": "best_success", "label": "Best Success"},
                    {"field": "best_success_epoch", "label": "Best S epoch"},
                    {"field": "best_precision", "label": "Best Precision"},
                    {"field": "best_precision_epoch", "label": "Best P epoch"},
                ],
            },
            {
                "id": "integrity_table",
                "title": "结果完整性与 checkpoint 审计",
                "dataset": "integrity",
                "sourceId": "integrity_source",
                "defaultSort": {"field": "checkpoint_epoch", "direction": "desc"},
                "columns": [
                    {"field": "run", "label": "Run"},
                    {"field": "status", "label": "Status"},
                    {"field": "checkpoint_epoch", "label": "Ckpt epoch"},
                    {"field": "latest_loss_step", "label": "Latest loss step"},
                    {"field": "last_validation_epoch", "label": "Last val epoch"},
                    {"field": "validation_points", "label": "Val points"},
                    {"field": "state_tensors", "label": "State tensors"},
                    {"field": "checkpoint_mib", "label": "Ckpt MiB"},
                    {"field": "provenance", "label": "Provenance"},
                ],
            },
            {
                "id": "comparison_table",
                "title": "主要差值（points）",
                "dataset": "comparisons",
                "sourceId": "validation_metrics_source",
                "defaultSort": {"field": "comparison", "direction": "asc"},
                "columns": [
                    {"field": "comparison", "label": "Comparison"},
                    {"field": "scope", "label": "Scope"},
                    {"field": "metric", "label": "Metric"},
                    {"field": "delta_points", "label": "Delta"},
                    {"field": "evidence", "label": "Evidence class"},
                ],
            },
            {
                "id": "reproducibility_table",
                "title": "相同 seed 的早期轨迹一致性检查",
                "dataset": "reproducibility",
                "sourceId": "reproducibility_source",
                "defaultSort": {"field": "shared_steps", "direction": "desc"},
                "columns": [
                    {"field": "window", "label": "Window"},
                    {"field": "shared_steps", "label": "Shared steps"},
                    {"field": "different_steps", "label": "Different steps"},
                    {"field": "different_fraction", "label": "Different fraction"},
                    {
                        "field": "mean_absolute_loss_difference",
                        "label": "Mean |loss diff|",
                    },
                    {"field": "first_differing_step", "label": "First diff step"},
                    {"field": "interpretation", "label": "Interpretation"},
                ],
            },
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": (
                    "# CT-SeqTrack 四组模块实验阶段性分析（seed42）\n\n"
                    "SeqTrack historical reference + W0/M2/M3 scratch ablations · "
                    "nuScenes-mini Car · 2026-07-25"
                ),
                "sourceId": "notebook_source",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## 技术摘要\n\n"
                    "本地结果中只有 W0 与 M2 完成 60 epoch；M3-w0 的 checkpoint "
                    "停在 epoch40、日志到约 epoch43.4，M3-w.05 的 checkpoint 停在 "
                    "epoch45、日志到约 epoch45.2。因此 M3 只能作机制与阶段性分析，"
                    "不能报告成 final。完整结果显示，shared-SE(2) W0 相对历史 "
                    "SeqTrack 在 epoch60 下降 22.840 Success / 32.758 Precision，"
                    "而完整 M2 bundle 相对 W0 回升 24.433 / 34.889，最终相对历史 "
                    "SeqTrack 仅高 1.593 / 2.131 points。最稳健的当前结论是：M2 "
                    "使当前 shared-SE(2) pipeline 可训练且恢复竞争力；单 seed "
                    "尚不足以证明稳定超越 SeqTrack。"
                ),
                "sourceId": "validation_metrics_source",
            },
            {
                "id": "integrity_heading",
                "type": "markdown",
                "body": (
                    "## 先看数据完整性\n\n"
                    "下表按 checkpoint、global step、TensorBoard loss 与验证点数交叉"
                    "检查。M3 两组缺失 epoch60 final 是当前报告唯一必须补齐的数据访问问题。"
                ),
                "sourceId": "integrity_source",
            },
            {"id": "integrity_block", "type": "table", "tableId": "integrity_table"},
            {
                "id": "performance_heading",
                "type": "markdown",
                "body": (
                    "## 标准验证性能\n\n"
                    "W0 在全部 12 个验证点的 Success 与 Precision 都低于历史 "
                    "SeqTrack；M2 对 W0 的 Success 在 12/12 个点为正、Precision "
                    "在 11/12 个点为正。这比只看 epoch60 更能支持“完整 M2 bundle "
                    "修复了 shared-SE(2) 设置”的判断。M2 对历史 SeqTrack 则只是"
                    "晚期约 +1.524/+2.086 points 的方向性优势。"
                ),
                "sourceId": "validation_metrics_source",
            },
            {"id": "summary_block", "type": "table", "tableId": "run_summary_table"},
            {"id": "success_block", "type": "chart", "chartId": "success_curve_chart"},
            {"id": "precision_block", "type": "chart", "chartId": "precision_curve_chart"},
            {"id": "comparison_block", "type": "table", "tableId": "comparison_table"},
            {
                "id": "attribution",
                "type": "markdown",
                "body": (
                    "## 归因边界\n\n"
                    "M2 是 DynamicsEncoder、PhysicalTimeAdapter 与 "
                    "proposal-innovation 的组合实验，当前数据不能拆分三者贡献。"
                    "SeqTrack baseline 来自较早代码，虽然它与 W0 checkpoint 的 "
                    "320 个 state key/shape 完全一致，W0−SeqTrack 仍混合了 code path "
                    "与 shared-SE(2) 数据定义，不能写成纯模块因果效应。需要 current-code "
                    "independent-candidate W00 控制来隔离 shared-SE(2)。"
                ),
                "sourceId": "integrity_source",
            },
            {
                "id": "m3_heading",
                "type": "markdown",
                "body": (
                    "## M3 的机制证据\n\n"
                    "在共同可比的 epoch31–40，M3-w.05 相对 M3-w0 将 path loss "
                    "降低约 48.4%、不规则 view-B 诊断 GT loss 降低约 29.1%、"
                    "center endpoint gap 降低约 26.7%，而 canonical view-A "
                    "supervised loss 只变化约 −0.6%。这说明 M3 的优化方向与设计目标"
                    "一致，并未靠明显牺牲 canonical 训练目标换取诊断改善。"
                ),
                "sourceId": "m3_diagnostics_source",
            },
            {"id": "m3_effect_block", "type": "chart", "chartId": "m3_effect_chart"},
            {"id": "m3_path_block", "type": "chart", "chartId": "m3_path_curve_chart"},
            {
                "id": "m3_standard_val",
                "type": "markdown",
                "body": (
                    "## 为什么 M3 暂未在 standard val 上明显增益\n\n"
                    "截至共同 epoch40，M3-w.05−M3-w0 为 −0.915 Success / "
                    "−4.187 Precision；但在 epoch25–40 的均值是 +0.980/+1.455。"
                    "波动方向不一致，且 standard val 使用 canonical history，"
                    "并不直接测 M3 针对的 irregular [1,3,5] 路径。因此当前只能说"
                    "“机制生效，标准场景增益未确定”，关键验证应转向 matched "
                    "irregular/gap1124 endpoint evaluation。"
                ),
                "sourceId": "m3_diagnostics_source",
            },
            {
                "id": "repro_heading",
                "type": "markdown",
                "body": (
                    "## 可复现性警告\n\n"
                    "B/C 在前 10 epoch 的 effective M3 weight 都为 0，按严格 matched "
                    "设计应共享同一优化轨迹；实际 loss 很早即分叉。验证 epoch 是同一"
                    "训练过程的重复观测，不是独立样本，不能把 12 个点当作 n=12 做显著性"
                    "检验。应增加 seeds43/44，并固定 deterministic 配置与验证采样。"
                ),
                "sourceId": "reproducibility_source",
            },
            {
                "id": "repro_block",
                "type": "table",
                "tableId": "reproducibility_table",
            },
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## 范围、数据与指标\n\n"
                    "四个当前 CT run 均为 seed42、batch16、60-epoch target，使用相同 "
                    "mini_train 274 tracklets/5,051 frames 与 mini_val 106 "
                    "tracklets/2,285 frames；train/val selection SHA 一致。Success "
                    "和 Precision 来自每 5 epoch 的 TensorBoard aggregate。比较单位"
                    "为 points。没有 per-tracklet endpoint export，因此本报告不提供"
                    "置信区间。"
                ),
                "sourceId": "notebook_source",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## 方法\n\n"
                    "分析从原始 TensorBoard scalar 逐点重建验证曲线，并读取 last.ckpt "
                    "的 epoch、global_step、state tensor schema 与 SHA256。训练期 "
                    "diagnostic 先按 1,262 steps/epoch 聚合，再在预先选定的共同窗口 "
                    "epoch31–40 对 B/C 比较。所有差值均为描述性统计；没有把 epoch "
                    "当独立 replicate，也没有对未完成曲线外推。"
                ),
                "sourceId": "notebook_source",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 限制与不确定性\n\n"
                    "M3 两组未完成；全部模块实验仅一个 seed；nuScenes-mini 方差较大；"
                    "历史 SeqTrack 缺少当前 run_provenance；M2 是模块 bundle；standard "
                    "val 不直接覆盖 irregular-history 目标。B/C checkpoint 较大主要"
                    "因为训练期保存 EMA teacher，不能直接解释为部署开销。并发服务器"
                    "runtime 也不适合作为速度对比。"
                ),
                "sourceId": "integrity_source",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 建议的下一步实验\n\n"
                    "1. 先等 B/C 跑完并重新拉取完整 version_0、last.ckpt、console log "
                    "和 provenance；重做 epoch60 final。\n"
                    "2. 不重训即可用 M2、B、C final checkpoint 跑同一冻结 "
                    "irregular/gap1124 endpoint protocol，比较 standard 与 irregular "
                    "两种 history，并输出 per-tracklet 结果。\n"
                    "3. 补 current-code independent-candidate W00，从而把 shared-SE(2) "
                    "的影响与历史代码差异分离。\n"
                    "4. 拆 M2：Dynamics only、Dynamics+Adapter、full proposal-innovation，"
                    "否则无法写每个模块的消融结论。\n"
                    "5. 对最终候选组合补 seeds43/44；让 B/C 在 deterministic 设置下做"
                    "严格 matched rerun，再决定 M3 是否进入主表。"
                ),
                "sourceId": "notebook_source",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "- W0 崩溃来自 shared-SE(2) candidate diversity 减少，还是 current "
                    "code path 的其他变化？\n"
                    "- M2 的收益由 dynamics、adapter、innovation 中哪一项主导？\n"
                    "- M3 的明显机制改善能否转化为 irregular-history tracking 指标？\n"
                    "- M3 teacher 在推理时移除后，模型大小、吞吐与性能是否保持？\n"
                    "- 这些方向性提升能否跨 seed、full nuScenes 与独立 test split 保持？"
                ),
                "sourceId": "notebook_source",
            },
        ],
        "sources": sources,
    }
    # Portable report validation requires every rendered asset to carry the
    # exact snapshot query used to populate it, in addition to sourceId.
    for asset in [*manifest["charts"], *manifest["tables"]]:
        dataset = asset["dataset"]
        asset["source"] = {
            "id": f"{dataset}_snapshot_query",
            "label": f"Reviewed bounded snapshot: {dataset}",
            "query": {
                "engine": "snapshot",
                "id": f"select_{dataset}",
                "sql": f"SELECT * FROM snapshot.{dataset}",
                "description": (
                    f"Reads the reviewed {dataset} rows embedded in this "
                    "partial technical report."
                ),
            },
        }
    snapshot = {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "partial",
        "datasets": {
            "success_curve": success_curve,
            "precision_curve": precision_curve,
            "run_summary": summary_rows,
            "integrity": integrity_rows,
            "comparisons": comparison_rows_snapshot,
            "m3_effects": effect_rows,
            "m3_path_curve": m3_path_curve,
            "m3_irregular_curve": m3_irregular_curve,
            "reproducibility": reproducibility_rows,
        },
        "accessIssues": [
            {
                "id": "m3_weight0_final_missing",
                "scope": "M3-w0 60-epoch result",
                "dataset": "run_summary",
                "message": (
                    "The pulled local result ends around training epoch43.4; "
                    "last.ckpt is epoch40 and validation ends at epoch40."
                ),
            },
            {
                "id": "m3_weight005_final_missing",
                "scope": "M3-w.05 60-epoch result",
                "dataset": "run_summary",
                "message": (
                    "The pulled local result ends around training epoch45.2; "
                    "last.ckpt and validation end at epoch45."
                ),
            },
        ],
    }
    payload = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }
    artifact_path = report_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def write_and_execute_notebook(repo_root: Path, notebook_path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# CT-SeqTrack four-module scratch analysis\n\n"
            "Executed companion notebook. It reads the generated tidy CSV files "
            "and reproduces the main numerical claims without reading checkpoints."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd().resolve()\n"
            "DATA = ROOT / 'compare_results' / 'data'\n"
            "print('Repository:', ROOT)\n"
            "print('Data:', DATA)"
        ),
        nbformat.v4.new_code_cell(
            "summary = pd.read_csv(DATA / "
            "'four_scratch_validation_summary_20260725.csv')\n"
            "integrity = pd.read_csv(DATA / "
            "'four_scratch_integrity_20260725.csv')\n"
            "comparisons = pd.read_csv(DATA / "
            "'four_scratch_comparisons_20260725.csv')\n"
            "summary[['run','last_validation_epoch','latest_success',"
            "'latest_precision','best_success','best_precision']]"
        ),
        nbformat.v4.new_code_cell(
            "integrity[['run','status','checkpoint_epoch','latest_loss_step',"
            "'last_validation_epoch','validation_points','state_tensors']]"
        ),
        nbformat.v4.new_code_cell(
            "main = comparisons[(comparisons['scope'] == 'latest/final') & "
            "(comparisons['comparison'].isin(['W0 − SeqTrack','M2 − W0',"
            "'M2 − SeqTrack']))]\n"
            "main[['comparison','metric','delta_points','evidence']]"
        ),
        nbformat.v4.new_code_cell(
            "effects = pd.read_csv(DATA / "
            "'four_scratch_m3_late_effects_20260725.csv')\n"
            "effects[effects['metric'].isin(['M3 path loss',"
            "'Irregular view B diagnostic GT loss',"
            "'Canonical view A supervised loss',"
            "'Center endpoint gap (m)','Yaw endpoint gap (rad)'])]"
            "[['metric','M3_w0_mean','M3_w05_mean','relative_change_pct']]"
        ),
        nbformat.v4.new_code_cell(
            "repro = pd.read_csv(DATA / "
            "'four_scratch_reproducibility_20260725.csv')\n"
            "repro"
        ),
        nbformat.v4.new_code_cell(
            "assert set(integrity.query(\"status == 'COMPLETE'\")['run']) "
            "== {'SeqTrack','W0','M2'}\n"
            "assert set(integrity.query(\"status == 'PARTIAL'\")['run']) "
            "== {'M3-w0','M3-w.05'}\n"
            "m2_w0 = main[(main.comparison == 'M2 − W0')].set_index('metric')"
            ".delta_points\n"
            "assert m2_w0['Success'] > 24 and m2_w0['Precision'] > 34\n"
            "print('Integrity and headline-delta checks: PASS')"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation guardrails\n\n"
            "- M3 results are partial; no extrapolation to epoch60.\n"
            "- Epochs are dependent observations, not statistical replicates.\n"
            "- SeqTrack is a historical reference, not an exact current-code control.\n"
            "- The M2 comparison identifies a bundle, not individual submodules."
        ),
    ]
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)


def run(repo_root: Path) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    data_dir = repo_root / "compare_results/data"
    notebook_dir = repo_root / "compare_results/notebooks"
    data_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)

    points_df, metrics_summary = collect_metrics(repo_root)
    integrity_df, _ = collect_integrity(repo_root, metrics_summary)
    comparisons_df = comparison_rows(metrics_summary)
    paired_df = paired_validation_comparisons(points_df)
    m3_epoch_means, late_effects, reproducibility = collect_m3_diagnostics(
        repo_root
    )
    config_df = config_and_provenance(repo_root)

    outputs = {
        "points": data_dir / f"four_scratch_validation_points_{ANALYSIS_DATE}.csv",
        "summary": data_dir / f"four_scratch_validation_summary_{ANALYSIS_DATE}.csv",
        "integrity": data_dir / f"four_scratch_integrity_{ANALYSIS_DATE}.csv",
        "comparisons": data_dir / f"four_scratch_comparisons_{ANALYSIS_DATE}.csv",
        "paired": data_dir / f"four_scratch_paired_validation_{ANALYSIS_DATE}.csv",
        "m3_epoch": data_dir / f"four_scratch_m3_epoch_diagnostics_{ANALYSIS_DATE}.csv",
        "m3_effects": data_dir / f"four_scratch_m3_late_effects_{ANALYSIS_DATE}.csv",
        "reproducibility": data_dir / f"four_scratch_reproducibility_{ANALYSIS_DATE}.csv",
        "config": data_dir / f"four_scratch_config_provenance_{ANALYSIS_DATE}.csv",
    }
    for key, frame in [
        ("points", points_df),
        ("summary", metrics_summary),
        ("integrity", integrity_df),
        ("comparisons", comparisons_df),
        ("paired", paired_df),
        ("m3_epoch", m3_epoch_means),
        ("m3_effects", late_effects),
        ("reproducibility", reproducibility),
        ("config", config_df),
    ]:
        frame.to_csv(outputs[key], index=False, encoding="utf-8")

    outputs["artifact"] = write_artifact(
        repo_root,
        points_df,
        metrics_summary,
        integrity_df,
        comparisons_df,
        late_effects,
        m3_epoch_means,
        reproducibility,
    )
    outputs["notebook"] = (
        notebook_dir / f"four_scratch_module_analysis_{ANALYSIS_DATE}.ipynb"
    )
    write_and_execute_notebook(repo_root, outputs["notebook"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    outputs = run(args.repo_root)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
