#!/usr/bin/env python3
"""Reproducible audit and visualization for the 2026-07-22 M2 three-run batch.

The script treats the frozen A1 run as a historical reference and the three
newly imported runs as the primary evidence. It extracts TensorBoard scalars,
checks checkpoint metadata and provenance, writes tidy CSV files, and renders
publication-ready diagnostic figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator
from nbclient import NotebookClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ANALYSIS_DATE = "20260723"
STEPS_PER_EPOCH = 1262
EXPECTED_STEPS = 75720
EXPECTED_EPOCHS = 60
EXPECTED_EVAL_POINTS = 12

RUNS = {
    "A1": {
        "label": "A1 historical baseline",
        "short_label": "A1 baseline",
        "root": "output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1",
        "version": "lightning_logs/version_0",
        "initialization": "scratch",
        "method": "A1 / SeqTrack3D order-time",
        "evidence_role": "historical reference",
        "color": "#4B5563",
        "linestyle": "--",
    },
    "R1": {
        "label": "R1 A1-init M2",
        "short_label": "R1 A1-init M2",
        "root": "output/m2_formal_true_seed42_473738f_20260722_112536",
        "version": "lightning_logs/version_0",
        "initialization": "A1 last.ckpt",
        "method": "M2 proposal innovation",
        "evidence_role": "primary formal run",
        "color": "#2563EB",
        "linestyle": "-",
    },
    "R2": {
        "label": "R2 scratch M2",
        "short_label": "R2 scratch M2",
        "root": "output/scratch_full_m2_seed42_473738f_20260722_150418",
        "version": "lightning_logs/version_0",
        "initialization": "scratch",
        "method": "M2 proposal innovation",
        "evidence_role": "scratch method control",
        "color": "#EA580C",
        "linestyle": "-",
    },
    "R3": {
        "label": "R3 scratch W0",
        "short_label": "R3 scratch W0",
        "root": "output/scratch_w0_matched_seed42_473738f_20260722_150536",
        "version": "lightning_logs/version_0",
        "initialization": "scratch",
        "method": "W0 without M2",
        "evidence_role": "matched shared-SE(2) control",
        "color": "#9CA3AF",
        "linestyle": ":",
    },
}

CONFIG_FIELDS = [
    "candidate_trajectory_mode",
    "use_dynamics_encoder",
    "use_physical_time_adapter",
    "dynamics_motion_mode",
    "dynamics_time_mode",
    "dynamics_innovation_alpha",
    "dynamics_innovation_warmup_epoch",
    "velocity_weight",
    "dynamics_displacement_weight",
    "seed",
    "batch_size",
    "workers",
    "epoch",
    "check_val_every_n_epoch",
    "save_top_k",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scalar(version_dir: Path, leaf: str) -> list[tuple[int, float]]:
    event_dir = version_dir / leaf
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if len(tags) != 1:
        raise RuntimeError(f"Expected one scalar tag in {event_dir}, found {tags}")
    return [(int(event.step), float(event.value)) for event in accumulator.Scalars(tags[0])]


def install_easydict_pickle_shim() -> None:
    """Install the minimal class needed to read Lightning checkpoint metadata."""

    if "easydict" in sys.modules:
        return
    module = types.ModuleType("easydict")
    easy_dict = type("EasyDict", (dict,), {})
    easy_dict.__module__ = "easydict"
    module.EasyDict = easy_dict
    sys.modules["easydict"] = module


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    install_easydict_pickle_shim()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_epoch_zero_based": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_tensor_count": len(checkpoint.get("state_dict", {})),
    }


def manifest_audit(repo_root: Path, run_root: Path) -> dict[str, Any]:
    manifest = run_root / "artifact_manifest.sha256"
    if not manifest.exists():
        return {
            "manifest_present": False,
            "manifest_entries": 0,
            "manifest_matched": 0,
            "manifest_missing": 0,
            "manifest_mismatched": 0,
        }
    entries = []
    marker = f"/{run_root.name}/"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, server_path = line.split(maxsplit=1)
        normalized = server_path.strip().replace("\\", "/")
        if marker not in normalized:
            entries.append((expected, None))
            continue
        relative = normalized.split(marker, 1)[1]
        entries.append((expected, run_root / Path(relative)))
    matched = 0
    missing = 0
    mismatched = 0
    for expected, local_path in entries:
        if local_path is None or not local_path.exists():
            missing += 1
        elif sha256_file(local_path) == expected:
            matched += 1
        else:
            mismatched += 1
    return {
        "manifest_present": True,
        "manifest_entries": len(entries),
        "manifest_matched": matched,
        "manifest_missing": missing,
        "manifest_mismatched": mismatched,
    }


def parse_simple_hparams(path: Path) -> dict[str, Any]:
    """Read the first EasyDict block without depending on the legacy class."""

    values: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        if raw.startswith("  state:"):
            break
        if not raw.startswith("    ") or ":" not in raw:
            continue
        key, value = raw.strip().split(":", 1)
        if key not in CONFIG_FIELDS:
            continue
        text = value.strip().strip("'\"")
        if text.lower() in {"true", "false"}:
            parsed: Any = text.lower() == "true"
        elif text.lower() in {"null", "none", ""}:
            parsed = None
        else:
            try:
                parsed = int(text)
            except ValueError:
                try:
                    parsed = float(text)
                except ValueError:
                    parsed = text
        values[key] = parsed
    values.setdefault("candidate_trajectory_mode", "legacy per-candidate (field absent)")
    values.setdefault("use_physical_time_adapter", False)
    values.setdefault("dynamics_time_mode", "not available in legacy run")
    values.setdefault("dynamics_innovation_alpha", "n/a")
    values.setdefault("dynamics_innovation_warmup_epoch", "n/a")
    values.setdefault("save_top_k", "legacy diagnostic checkpoints")
    return values


def load_provenance(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "run_provenance.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metrics(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_id, meta in RUNS.items():
        version_dir = repo_root / meta["root"] / meta["version"]
        metric_points = {
            "Success": read_scalar(version_dir, "metrics_test_success"),
            "Precision": read_scalar(version_dir, "metrics_test_precision"),
        }
        for metric, points in metric_points.items():
            for step, value in points:
                rows.append(
                    {
                        "run_id": run_id,
                        "run_label": meta["label"],
                        "short_label": meta["short_label"],
                        "initialization": meta["initialization"],
                        "method": meta["method"],
                        "evidence_role": meta["evidence_role"],
                        "metric": metric,
                        "step": step,
                        "epoch": int(step / STEPS_PER_EPOCH),
                        "value": value,
                    }
                )
    points_df = pd.DataFrame(rows).sort_values(["run_id", "metric", "step"])
    summary_rows = []
    for run_id, meta in RUNS.items():
        run_points = points_df[points_df["run_id"] == run_id]
        row: dict[str, Any] = {
            "run_id": run_id,
            "run_label": meta["label"],
            "initialization": meta["initialization"],
            "method": meta["method"],
            "evidence_role": meta["evidence_role"],
            "validation_points": int(
                run_points[run_points["metric"] == "Success"].shape[0]
            ),
        }
        for metric in ["Success", "Precision"]:
            metric_key = metric.lower()
            series = run_points[run_points["metric"] == metric].sort_values("step")
            best_index = series["value"].idxmax()
            row[f"final_{metric_key}"] = float(series.iloc[-1]["value"])
            row[f"best_{metric_key}"] = float(series.loc[best_index, "value"])
            row[f"best_{metric_key}_epoch"] = int(series.loc[best_index, "epoch"])
            row[f"late_mean_{metric_key}"] = float(
                series[series["epoch"] >= 40]["value"].mean()
            )
            row[f"all_eval_mean_{metric_key}"] = float(series["value"].mean())
        summary_rows.append(row)
    return points_df.reset_index(drop=True), pd.DataFrame(summary_rows)


def collect_comparisons(summary_df: pd.DataFrame) -> pd.DataFrame:
    by_id = summary_df.set_index("run_id")
    specifications = [
        (
            "R1 minus A1",
            "R1",
            "A1",
            "descriptive_only",
            "M2 effect is confounded with another 60 epochs after A1 initialization.",
        ),
        (
            "R2 minus R3",
            "R2",
            "R3",
            "matched_scratch_control",
            "Same commit/seed/steps/data and shared-SE(2), but W0 collapses under this augmentation.",
        ),
        (
            "R2 minus A1",
            "R2",
            "A1",
            "historical_cross_run",
            "Promising reference comparison, not fully matched because A1 predates shared-SE(2).",
        ),
    ]
    rows = []
    for comparison, treatment, control, status, caveat in specifications:
        for scope in ["final", "late_mean"]:
            for metric in ["success", "precision"]:
                rows.append(
                    {
                        "comparison": comparison,
                        "treatment_run": treatment,
                        "control_run": control,
                        "scope": scope,
                        "metric": metric.capitalize(),
                        "treatment_value": float(
                            by_id.loc[treatment, f"{scope}_{metric}"]
                        ),
                        "control_value": float(by_id.loc[control, f"{scope}_{metric}"]),
                        "delta": float(
                            by_id.loc[treatment, f"{scope}_{metric}"]
                            - by_id.loc[control, f"{scope}_{metric}"]
                        ),
                        "interpretation_status": status,
                        "caveat": caveat,
                    }
                )
    return pd.DataFrame(rows)


def collect_loss(repo_root: Path) -> pd.DataFrame:
    rows = []
    for run_id, meta in RUNS.items():
        version_dir = repo_root / meta["root"] / meta["version"]
        points = read_scalar(version_dir, "loss_loss_total")
        frame = pd.DataFrame(points, columns=["step", "loss_total"])
        frame["epoch"] = (frame["step"] // STEPS_PER_EPOCH) + 1
        for epoch, group in frame.groupby("epoch", sort=True):
            rows.append(
                {
                    "run_id": run_id,
                    "run_label": meta["label"],
                    "epoch": int(epoch),
                    "batch_points": int(group.shape[0]),
                    "loss_mean": float(group["loss_total"].mean()),
                    "loss_median": float(group["loss_total"].median()),
                    "loss_p10": float(group["loss_total"].quantile(0.10)),
                    "loss_p90": float(group["loss_total"].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def collect_integrity(repo_root: Path) -> pd.DataFrame:
    rows = []
    for run_id, meta in RUNS.items():
        run_root = repo_root / meta["root"]
        version_dir = run_root / meta["version"]
        checkpoint_path = version_dir / "checkpoints/last.ckpt"
        checkpoint = checkpoint_metadata(checkpoint_path)
        success_points = read_scalar(version_dir, "metrics_test_success")
        precision_points = read_scalar(version_dir, "metrics_test_precision")
        loss_points = read_scalar(version_dir, "loss_loss_total")
        provenance = load_provenance(run_root)
        exit_path = run_root / "training_exit_code.txt"
        exit_code = int(exit_path.read_text(encoding="utf-8").strip()) if exit_path.exists() else None
        console_path = run_root / "console.log"
        console_finished = None
        if console_path.exists():
            with console_path.open("rb") as handle:
                handle.seek(max(0, console_path.stat().st_size - 100_000))
                tail = handle.read().decode("utf-8", errors="replace")
            console_finished = (
                "max_epochs=60" in tail
                and "reached" in tail
                and "Trainer.fit" in tail
            )
        manifest = manifest_audit(repo_root, run_root)
        git_commit = None
        git_dirty = None
        seed = 42
        if provenance:
            git_commit = provenance["git"]["commit"]
            git_dirty = provenance["git"]["dirty_any"]
            seed = provenance["seed"]
        complete = (
            checkpoint["checkpoint_epoch_zero_based"] == 59
            and checkpoint["checkpoint_global_step"] == EXPECTED_STEPS
            and len(success_points) == EXPECTED_EVAL_POINTS
            and len(precision_points) == EXPECTED_EVAL_POINTS
            and len(loss_points) == EXPECTED_STEPS
            and (exit_code in {None, 0})
            and (console_finished in {None, True})
            and manifest["manifest_missing"] == 0
            and manifest["manifest_mismatched"] == 0
        )
        rows.append(
            {
                "run_id": run_id,
                "run_label": meta["label"],
                "integrity_status": "PASS" if complete else "FAIL",
                "evidence_class": "clean provenance" if provenance else "legacy reference",
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "seed": seed,
                "training_exit_code": exit_code,
                "console_finished": console_finished,
                "success_points": len(success_points),
                "precision_points": len(precision_points),
                "loss_points": len(loss_points),
                **checkpoint,
                **manifest,
            }
        )
    return pd.DataFrame(rows)


def collect_config_diff(repo_root: Path) -> pd.DataFrame:
    run_configs: dict[str, dict[str, Any]] = {}
    for run_id, meta in RUNS.items():
        run_root = repo_root / meta["root"]
        provenance = load_provenance(run_root)
        if provenance:
            run_configs[run_id] = provenance["resolved_config"]
        else:
            run_configs[run_id] = parse_simple_hparams(
                run_root / meta["version"] / "hparams.yaml"
            )
    rows = []
    for field in CONFIG_FIELDS:
        row = {"field": field}
        for run_id in RUNS:
            value = run_configs[run_id].get(field, "field absent")
            row[run_id] = json.dumps(value) if isinstance(value, (list, dict)) else value
        rows.append(row)
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#374151")


def save_performance_curve(points_df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    for ax, metric in zip(axes, ["Success", "Precision"]):
        for run_id, meta in RUNS.items():
            subset = points_df[
                (points_df["run_id"] == run_id) & (points_df["metric"] == metric)
            ].sort_values("epoch")
            ax.plot(
                subset["epoch"],
                subset["value"],
                label=meta["short_label"],
                color=meta["color"],
                linestyle=meta["linestyle"],
                linewidth=2.25,
                marker="o",
                markersize=4.0,
            )
        ax.set_title(metric, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel(f"{metric} (points)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Standard mini_val performance across training",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.935,
        "R1 leads late training; R2 finishes above the historical A1 reference, while matched scratch W0 collapses.",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.text(
        0.99,
        0.004,
        "Source: TensorBoard metrics/test scalars; evaluations every 5 epochs.",
        ha="right",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0.03, 0.17, 0.99, 0.90))
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_deltas(comparisons_df: pd.DataFrame, output: Path) -> None:
    display = comparisons_df[comparisons_df["scope"] == "final"].copy()
    pivot = display.pivot(index="comparison", columns="metric", values="delta").loc[
        ["R1 minus A1", "R2 minus R3", "R2 minus A1"]
    ]
    x = np.arange(pivot.shape[0])
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    success = ax.bar(
        x - width / 2, pivot["Success"], width, label="Success", color="#2563EB"
    )
    precision = ax.bar(
        x + width / 2,
        pivot["Precision"],
        width,
        label="Precision",
        color="#EA580C",
    )
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.bar_label(success, fmt="%+.2f", padding=3, fontsize=9)
    ax.bar_label(precision, fmt="%+.2f", padding=3, fontsize=9)
    ax.set_xticks(
        x,
        [
            "R1 A1-init M2\n− historical A1",
            "R2 scratch M2\n− R3 scratch W0",
            "R2 scratch M2\n− historical A1",
        ],
    )
    ax.set_ylabel("Final-score difference (points)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_axes(ax)
    fig.suptitle(
        "Final-score comparisons and attribution limits",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.92,
        "All deltas are descriptive: R1 lacks an A1-init W0 continuation, and R2 vs A1 is not a fully matched code/augmentation comparison.",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.99, 0.88))
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_loss_curves(loss_df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    for run_id, meta in RUNS.items():
        subset = loss_df[loss_df["run_id"] == run_id].sort_values("epoch")
        ax.plot(
            subset["epoch"],
            subset["loss_mean"],
            label=meta["short_label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=2.1,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean batch loss_total (log scale)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_axes(ax)
    fig.suptitle(
        "Training loss convergence",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.92,
        "R1 starts from a much lower loss because it loads A1; all four runs complete 75,720 finite loss records.",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.text(
        0.99,
        0.01,
        "Each point is the mean of 1,262 batch-level loss_total scalars.",
        ha="right",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.88))
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_and_execute_notebook(repo_root: Path, notebook_path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# M2 三组训练结果复核\n\n"
            "该 notebook 从本地 TensorBoard、checkpoint 与 provenance 原始文件重建"
            "指标、完整性表、比较表和图表。A1 是历史参考；R1/R2/R3 是本批三个训练。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "from IPython.display import display, Image\n"
            "from tools.analyze_m2_three_runs import run_analysis\n\n"
            "repo_root = Path.cwd()\n"
            "outputs = run_analysis(repo_root, write_notebook=False)\n"
            "print('Analysis outputs regenerated.')"
        ),
        nbformat.v4.new_markdown_cell("## 完整性与最终指标"),
        nbformat.v4.new_code_cell(
            "display(outputs['integrity'][[\n"
            "    'run_id', 'integrity_status', 'git_commit', 'checkpoint_global_step',\n"
            "    'success_points', 'precision_points', 'loss_points', 'checkpoint_sha256'\n"
            "]])\n"
            "display(outputs['summary'].round(4))"
        ),
        nbformat.v4.new_markdown_cell("## 主要比较与可归因性限制"),
        nbformat.v4.new_code_cell(
            "display(outputs['comparisons'][outputs['comparisons']['scope'] == 'final'].round(4))\n"
            "display(outputs['config_diff'])"
        ),
        nbformat.v4.new_markdown_cell("## 图表"),
        nbformat.v4.new_code_cell(
            "for path in outputs['figure_paths']:\n"
            "    display(Image(filename=str(path)))"
        ),
    ]
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, notebook_path)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root)}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)


def write_artifact_json(
    repo_root: Path,
    points_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparisons_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
) -> Path:
    report_dir = (
        repo_root / "compare_results/reports/m2_three_run_analysis_20260723"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    run_order = {run_id: index + 1 for index, run_id in enumerate(RUNS)}

    final_scores = []
    summary_rows = []
    for row in summary_df.to_dict(orient="records"):
        run_id = row["run_id"]
        for metric in ["Success", "Precision"]:
            final_scores.append(
                {
                    "run_id": run_id,
                    "run_order": run_order[run_id],
                    "run_label": RUNS[run_id]["short_label"],
                    "metric": metric,
                    "score": round(float(row[f"final_{metric.lower()}"]), 6),
                }
            )
        summary_rows.append(
            {
                "run_id": run_id,
                "run_order": run_order[run_id],
                "run_label": RUNS[run_id]["short_label"],
                "initialization": row["initialization"],
                "method": row["method"],
                "final_success": round(float(row["final_success"]), 3),
                "final_precision": round(float(row["final_precision"]), 3),
                "late_success": round(float(row["late_mean_success"]), 3),
                "late_precision": round(float(row["late_mean_precision"]), 3),
            }
        )

    success_curve = []
    curve = points_df[points_df["metric"] == "Success"].copy()
    for row in curve.to_dict(orient="records"):
        success_curve.append(
            {
                "run_id": row["run_id"],
                "run_label": RUNS[row["run_id"]]["short_label"],
                "epoch": int(row["epoch"]),
                "success": round(float(row["value"]), 6),
            }
        )

    final_comparisons = []
    filtered = comparisons_df[comparisons_df["scope"] == "final"]
    for row in filtered.to_dict(orient="records"):
        final_comparisons.append(
            {
                "comparison": row["comparison"],
                "metric": row["metric"],
                "delta": round(float(row["delta"]), 3),
                "status": row["interpretation_status"],
                "caveat": row["caveat"],
            }
        )

    integrity_rows = []
    for row in integrity_df.to_dict(orient="records"):
        integrity_rows.append(
            {
                "run_id": row["run_id"],
                "run_order": run_order[row["run_id"]],
                "status": row["integrity_status"],
                "evidence_class": row["evidence_class"],
                "commit": (row["git_commit"][:7] if row["git_commit"] else "legacy"),
                "global_step": int(row["checkpoint_global_step"]),
                "eval_points": int(row["success_points"]),
                "loss_points": int(row["loss_points"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
        )

    next_actions = [
        {
            "priority": 1,
            "action": "R1 same-checkpoint true/fixed/shuffled",
            "purpose": "Test causal use of physical time on standard/gap1124/burst-drop without retraining.",
        },
        {
            "priority": 2,
            "action": "A1-init W0 continuation",
            "purpose": "Separate M2 from the extra 60-epoch continuation budget in R1.",
        },
        {
            "priority": 3,
            "action": "Current-code scratch legacy-candidate W0",
            "purpose": "Separate the shared-SE(2) effect from the M2 structural effect.",
        },
        {
            "priority": 4,
            "action": "Seeds 43/44 and full data",
            "purpose": "Run only if causal-time, matched-attribution and standard guardrails pass.",
        },
    ]

    title = "CT-SeqTrack M2 三组训练结果复核"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "R1 A1-init M2、R2 scratch M2、R3 scratch W0 与历史 A1 的完整性、指标、归因边界和下一步。",
        "generatedAt": "2026-07-23T12:00:00+08:00",
        "cards": [],
        "charts": [
            {
                "id": "final_scores_chart",
                "title": "Final standard mini_val scores",
                "subtitle": "R1 is strongest; R2 finishes above the historical A1 reference, while R3 collapses under shared-SE(2).",
                "type": "bar",
                "dataset": "final_scores",
                "sourceId": "metric_summary",
                "source": {
                    "id": "metric_summary",
                    "label": "Reviewed final metric materialization",
                    "path": "compare_results/data/m2_three_run_metric_summary_20260723.csv",
                    "query": {
                        "language": "sql",
                        "engine": "duckdb",
                        "sql": "SELECT run_id, run_order, run_label, metric, score FROM final_scores ORDER BY run_order, metric",
                        "description": "Select epoch-60 Success and Precision for each reviewed run.",
                        "tables_used": ["final_scores"],
                        "metric_definitions": [
                            "score: TensorBoard metrics/test scalar at global step 75720; unit is points."
                        ],
                    },
                },
                "encodings": {
                    "x": {
                        "field": "run_label",
                        "type": "nominal",
                        "label": "Run",
                    },
                    "y": {
                        "field": "score",
                        "type": "quantitative",
                        "label": "Final score",
                        "format": "number",
                    },
                    "color": {
                        "field": "metric",
                        "type": "nominal",
                        "label": "Metric",
                    },
                },
                "xAxisTitle": "Run",
                "yAxisTitle": "Final score (points)",
                "valueFormat": "number",
                "layout": "full",
                "maxRows": 20,
            },
            {
                "id": "success_curve_chart",
                "title": "Standard mini_val Success across training",
                "subtitle": "R1 leads late training; R2 recovers to the A1 range, and R3 remains far below it.",
                "type": "line",
                "dataset": "success_curve",
                "sourceId": "metric_points",
                "source": {
                    "id": "metric_points",
                    "label": "Reviewed Success curve materialization",
                    "path": "compare_results/data/m2_three_run_metric_points_20260723.csv",
                    "query": {
                        "language": "sql",
                        "engine": "duckdb",
                        "sql": "SELECT run_id, run_label, epoch, success FROM success_curve ORDER BY run_id, epoch",
                        "description": "Select standard mini_val Success at each five-epoch evaluation.",
                        "tables_used": ["success_curve"],
                        "metric_definitions": [
                            "success: TensorBoard metrics/test Success scalar; evaluations occur every five epochs."
                        ],
                    },
                },
                "encodings": {
                    "x": {
                        "field": "epoch",
                        "type": "quantitative",
                        "label": "Epoch",
                    },
                    "y": {
                        "field": "success",
                        "type": "quantitative",
                        "label": "Success",
                        "format": "number",
                    },
                    "color": {
                        "field": "run_label",
                        "type": "nominal",
                        "label": "Run",
                    },
                },
                "xAxisTitle": "Training epoch",
                "yAxisTitle": "Success (points)",
                "valueFormat": "number",
                "layout": "full",
                "maxRows": 100,
            },
        ],
        "tables": [
            {
                "id": "summary_table",
                "title": "Final and late-training metrics",
                "subtitle": "Final uses epoch-60 last.ckpt; late mean covers epochs 40/45/50/55/60.",
                "dataset": "summary",
                "sourceId": "metric_summary",
                "source": {
                    "id": "metric_summary",
                    "label": "Reviewed run metric summary",
                    "path": "compare_results/data/m2_three_run_metric_summary_20260723.csv",
                    "query": {
                        "language": "sql",
                        "engine": "duckdb",
                        "sql": "SELECT run_label, initialization, method, final_success, final_precision, late_success, late_precision FROM summary ORDER BY final_success DESC",
                        "description": "Select final and late-training metrics for all reviewed runs.",
                        "tables_used": ["summary"],
                        "metric_definitions": [
                            "final_*: epoch-60 last.ckpt metric in points.",
                            "late_*: mean of epochs 40, 45, 50, 55 and 60 in points.",
                        ],
                    },
                },
                "density": "spacious",
                "layout": "full",
                "defaultSort": {"field": "final_success", "direction": "desc"},
                "columns": [
                    {"field": "run_label", "label": "Run", "type": "text"},
                    {
                        "field": "initialization",
                        "label": "Initialization",
                        "type": "text",
                    },
                    {
                        "field": "final_success",
                        "label": "Final Success",
                        "type": "number",
                    },
                    {
                        "field": "final_precision",
                        "label": "Final Precision",
                        "type": "number",
                    },
                    {
                        "field": "late_success",
                        "label": "Late Success",
                        "type": "number",
                    },
                    {
                        "field": "late_precision",
                        "label": "Late Precision",
                        "type": "number",
                    },
                ],
            },
            {
                "id": "comparison_table",
                "title": "Final deltas and attribution limits",
                "subtitle": "Every delta is descriptive until the missing continuation and augmentation controls are completed.",
                "dataset": "comparisons",
                "sourceId": "comparisons_csv",
                "source": {
                    "id": "comparisons_csv",
                    "label": "Reviewed final comparisons",
                    "path": "compare_results/data/m2_three_run_comparisons_20260723.csv",
                    "query": {
                        "language": "sql",
                        "engine": "duckdb",
                        "sql": "SELECT comparison, metric, delta, status, caveat FROM comparisons ORDER BY delta DESC",
                        "description": "Select final-score deltas with their attribution status and caveat.",
                        "tables_used": ["comparisons"],
                        "metric_definitions": [
                            "delta: treatment final score minus control final score, in points."
                        ],
                    },
                },
                "density": "spacious",
                "layout": "full",
                "defaultSort": {"field": "delta", "direction": "desc"},
                "columns": [
                    {
                        "field": "comparison",
                        "label": "Comparison",
                        "type": "text",
                    },
                    {"field": "metric", "label": "Metric", "type": "text"},
                    {"field": "delta", "label": "Delta", "type": "number"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {"field": "caveat", "label": "Caveat", "type": "text"},
                ],
            },
            {
                "id": "integrity_table",
                "title": "Run and checkpoint integrity",
                "subtitle": "All three imported runs complete 60 epochs and pass checkpoint/event checks; A1 is a legacy reference.",
                "dataset": "integrity",
                "sourceId": "integrity_csv",
                "source": {
                    "id": "integrity_csv",
                    "label": "Checkpoint and event integrity audit",
                    "path": "compare_results/data/m2_three_run_integrity_20260723.csv",
                    "query": {
                        "language": "sql",
                        "engine": "duckdb",
                        "sql": "SELECT run_id, status, evidence_class, commit, global_step, eval_points, loss_points FROM integrity ORDER BY run_id",
                        "description": "Select completion, checkpoint and TensorBoard event audit fields.",
                        "tables_used": ["integrity"],
                        "metric_definitions": [
                            "global_step: Lightning checkpoint global step.",
                            "loss_points: count of batch-level loss_total scalars.",
                        ],
                    },
                },
                "density": "spacious",
                "layout": "full",
                "defaultSort": {"field": "run_id", "direction": "asc"},
                "columns": [
                    {"field": "run_id", "label": "Run", "type": "text"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {
                        "field": "evidence_class",
                        "label": "Evidence",
                        "type": "text",
                    },
                    {"field": "commit", "label": "Commit", "type": "text"},
                    {
                        "field": "global_step",
                        "label": "Global step",
                        "type": "number",
                    },
                    {
                        "field": "loss_points",
                        "label": "Loss points",
                        "type": "number",
                    },
                ],
            },
            {
                "id": "next_table",
                "title": "Frozen next-step order",
                "subtitle": "Do not start M3/M4 or multi-seed/full-data expansion before causal and attribution gates pass.",
                "dataset": "next_actions",
                "sourceId": "analysis_report",
                "source": {
                    "id": "analysis_report",
                    "label": "Frozen next-step plan",
                    "path": "compare_results/reports/m2_three_run_analysis_20260723.md",
                    "query": {
                        "language": "sql",
                        "engine": "duckdb",
                        "sql": "SELECT priority, action, purpose FROM next_actions ORDER BY priority",
                        "description": "Select the frozen next-step sequence from the reviewed analysis.",
                        "tables_used": ["next_actions"],
                        "metric_definitions": [
                            "priority: execution order; lower numbers run first."
                        ],
                    },
                },
                "density": "spacious",
                "layout": "full",
                "defaultSort": {"field": "priority", "direction": "asc"},
                "columns": [
                    {"field": "priority", "label": "Priority", "type": "number"},
                    {"field": "action", "label": "Action", "type": "text"},
                    {"field": "purpose", "label": "Purpose", "type": "text"},
                ],
            },
        ],
        "sources": [
            {
                "id": "metric_points",
                "label": "Reviewed TensorBoard metric points",
                "path": "compare_results/data/m2_three_run_metric_points_20260723.csv",
            },
            {
                "id": "metric_summary",
                "label": "Reviewed run metric summary",
                "path": "compare_results/data/m2_three_run_metric_summary_20260723.csv",
            },
            {
                "id": "comparisons_csv",
                "label": "Reviewed comparison table",
                "path": "compare_results/data/m2_three_run_comparisons_20260723.csv",
            },
            {
                "id": "integrity_csv",
                "label": "Checkpoint and event integrity audit",
                "path": "compare_results/data/m2_three_run_integrity_20260723.csv",
            },
            {
                "id": "analysis_report",
                "label": "Full technical report",
                "path": "compare_results/reports/m2_three_run_analysis_20260723.md",
            },
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": f"# {title}\n\n**2026-07-23｜R1/R2/R3 完整性 PASS｜standard 正信号，方法归因与 causal-time 仍 HOLD。**",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": "## 技术摘要\n\nR1 A1-init M2 最终达到 **55.303 Success / 67.182 Precision**，R2 scratch M2 达到 **53.318/62.503**，均高于历史 A1 的 **51.229/57.863**；R3 scratch W0 只有 **28.999/28.023**。三个新训练都来自 clean `473738f`，完成 60 epoch/75720 step，checkpoint 与事件完整。\n\n当前结论是 **M2 standard signal positive / evaluation Conditional-Go**，不是 Method GO。R1 缺少同预算 A1-init W0 continuation，R2/R3 又暴露了 shared-SE(2) 与 M2 的强交互；所有新训练都只有 true-time standard cadence，尚不能证明正确物理时间有效。",
            },
            {
                "id": "key_findings",
                "type": "markdown",
                "body": "## 关键发现\n\n- R1 相对历史 A1 final 为 **+4.074 Success / +9.318 Precision**，late mean 为 **+3.702/+8.391**。\n- R2 相对历史 A1 final 为 **+2.090/+4.640**，说明 full M2 可以从头训练，不依赖 A1 才能工作。\n- R2 相对 R3 为 **+24.319/+34.480**，但这主要证明 M2 对当前 shared-SE(2) pipeline 至关重要，不能直接写成“超过 SeqTrack3D”。",
            },
            {
                "id": "final_scores_block",
                "type": "chart",
                "chartId": "final_scores_chart",
            },
            {
                "id": "success_curve_block",
                "type": "chart",
                "chartId": "success_curve_chart",
            },
            {
                "id": "attribution",
                "type": "markdown",
                "body": "## 归因边界\n\nR1 从已经训练 60 epoch 的 A1 权重继续训练 60 epoch，因此 R1−A1 混合了额外训练预算、shared-SE(2)、adapter 和 innovation。R3 虽与 R2 在 commit/seed/data/steps 上匹配，但它是 shared-SE(2) 下的 W0，不是历史 A1 的精确复制；其塌陷意味着当前数据定义与模型结构存在强交互。",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": "## 方法与完整性\n\n指标从 TensorBoard `metrics_test_success`、`metrics_test_precision` 和 `loss_loss_total` 重建。所有 final 数字固定使用 epoch60 `last.ckpt`；best epoch 仅作诊断。checkpoint 在 CPU 读取 epoch/global_step/tensor count 并计算 SHA256；R1 的 35 项服务器 artifact manifest 全部本地匹配。",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## 限制与稳健性\n\n当前只有 seed42、nuScenes-mini、standard aggregate 曲线，无法估计多 seed 方差或做 per-tracklet paired bootstrap。standard 的 `delta_t` 波动很小，只能作为性能 guardrail。mini_val 已参与多轮开发，最终论文还需要独立 held-out 测试。训练 loss 定义在 M2 与 W0 间不同，不能用 loss 高低替代 tracking 指标。",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": "## 下一步\n\n第一优先级是不重训：用 R1 final checkpoint 跑 standard/gap1124/burst-drop 的 true/fixed/shuffled 和 matched A1，输出 endpoint/per-tracklet 并做 paired bootstrap。随后补 A1-init W0 continuation 以排除 extra-training confound，再补 current-code legacy-candidate W0 解释 R3 塌陷。只有三个 gate 同时通过，才做 seeds43/44、full data 和 M3；M4 继续锁定。",
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": "## 仍需回答的问题\n\n- R1 的提升有多少来自额外 60 epoch，有多少来自 M2？\n- R3 塌陷是 shared-SE(2) candidate diversity 问题还是结构不匹配？\n- 同一 R1 checkpoint 的 true time 能否同时胜过 fixed 与 shuffled？\n- strong cadence 的收益能否在 standard 不退化、多个 seed 和独立测试上保持？",
            },
        ],
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": "2026-07-23T12:00:00+08:00",
        "datasets": {
            "final_scores": final_scores,
            "success_curve": success_curve,
            "summary": summary_rows,
            "comparisons": final_comparisons,
            "integrity": integrity_rows,
            "next_actions": next_actions,
        },
    }
    payload = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": manifest["sources"],
    }
    artifact_path = report_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def run_analysis(repo_root: Path, write_notebook: bool = True) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    data_dir = repo_root / "compare_results/data"
    line_dir = repo_root / "compare_results/figures/line_charts"
    delta_dir = repo_root / "compare_results/figures/delta_charts"
    notebook_dir = repo_root / "compare_results/notebooks"
    for directory in [data_dir, line_dir, delta_dir, notebook_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    points_df, summary_df = collect_metrics(repo_root)
    comparisons_df = collect_comparisons(summary_df)
    loss_df = collect_loss(repo_root)
    integrity_df = collect_integrity(repo_root)
    config_diff_df = collect_config_diff(repo_root)

    csv_paths = {
        "points": data_dir / f"m2_three_run_metric_points_{ANALYSIS_DATE}.csv",
        "summary": data_dir / f"m2_three_run_metric_summary_{ANALYSIS_DATE}.csv",
        "comparisons": data_dir / f"m2_three_run_comparisons_{ANALYSIS_DATE}.csv",
        "loss": data_dir / f"m2_three_run_loss_epoch_{ANALYSIS_DATE}.csv",
        "integrity": data_dir / f"m2_three_run_integrity_{ANALYSIS_DATE}.csv",
        "config_diff": data_dir / f"m2_three_run_config_diff_{ANALYSIS_DATE}.csv",
    }
    for key, frame in [
        ("points", points_df),
        ("summary", summary_df),
        ("comparisons", comparisons_df),
        ("loss", loss_df),
        ("integrity", integrity_df),
        ("config_diff", config_diff_df),
    ]:
        frame.to_csv(csv_paths[key], index=False, encoding="utf-8")

    figure_paths = [
        line_dir / f"m2_three_run_standard_curves_{ANALYSIS_DATE}.png",
        delta_dir / f"m2_three_run_final_deltas_{ANALYSIS_DATE}.png",
        line_dir / f"m2_three_run_loss_curves_{ANALYSIS_DATE}.png",
    ]
    save_performance_curve(points_df, figure_paths[0])
    save_deltas(comparisons_df, figure_paths[1])
    save_loss_curves(loss_df, figure_paths[2])

    artifact_path = write_artifact_json(
        repo_root,
        points_df,
        summary_df,
        comparisons_df,
        integrity_df,
    )

    if write_notebook:
        write_and_execute_notebook(
            repo_root,
            notebook_dir / f"m2_three_run_analysis_{ANALYSIS_DATE}.ipynb",
        )

    return {
        "points": points_df,
        "summary": summary_df,
        "comparisons": comparisons_df,
        "loss": loss_df,
        "integrity": integrity_df,
        "config_diff": config_diff_df,
        "csv_paths": csv_paths,
        "figure_paths": figure_paths,
        "artifact_path": artifact_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--skip-notebook", action="store_true")
    args = parser.parse_args()
    outputs = run_analysis(args.repo_root, write_notebook=not args.skip_notebook)
    print(outputs["summary"].to_string(index=False))
    print()
    print(outputs["integrity"].to_string(index=False))


if __name__ == "__main__":
    main()
