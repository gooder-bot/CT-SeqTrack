#!/usr/bin/env python3
"""Summarize the same-commit TWC A/B/C seed42 experiment.

The experiment separates single-view training (A), paired-view training with
zero consistency weight (B), and paired-view training with corrected TWC (C).
This script validates the local run provenance, parses TensorBoard scalar
events, writes analysis-ready CSV files, and renders publication-oriented
PNG/SVG figures under ``compare_results``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from summarize_latest_5runs import scalar_rows


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output" / "paper_twc_abc_20260720_183711"
OUT = ROOT / "compare_results"
DATA_DIR = OUT / "data"
LINE_DIR = OUT / "figures" / "line_charts"
DELTA_DIR = OUT / "figures" / "delta_charts"
DIAG_DIR = OUT / "figures" / "diagnostics"

PREFIX = "twc_abc_seed42"
STEPS_PER_EPOCH = 1262
LATE_START_EPOCH = 40
DIAG_BLOCK_SIZE = 1000

RUNS = [
    {
        "key": "A",
        "condition": "Single view",
        "directory": "A_single_seed42",
        "color": "#4E79A7",
    },
    {
        "key": "B",
        "condition": "Paired views, TWC weight 0",
        "directory": "B_paired_weight0_seed42",
        "color": "#F28E2B",
    },
    {
        "key": "C",
        "condition": "Paired views + corrected TWC",
        "directory": "C_corrected_twc_seed42",
        "color": "#59A14F",
    },
]

METRICS = {
    "success/test": ("metrics_test_success", "metrics/test", "Success"),
    "precision/test": ("metrics_test_precision", "metrics/test", "Precision"),
}

DIAGNOSTICS = {
    "loss_twc": "loss_loss_twc",
    "twc_valid_ratio": "loss_twc_valid_ratio",
    "twc_center_gap": "loss_twc_center_gap",
    "twc_angle_gap": "loss_twc_angle_gap",
    "twc_anchor_gap_max": "loss_twc_anchor_gap_max",
    "twc_current_point_gap_max": "loss_twc_current_point_gap_max",
    "loss_total": "loss_loss_total",
    "loss_total_sup": "loss_loss_total_sup",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def version_dir(run: dict[str, str]) -> Path:
    return RUN_ROOT / run["directory"] / "lightning_logs" / "version_0"


def load_provenance() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        path = RUN_ROOT / run["directory"] / "run_provenance.json"
        provenance = json.loads(path.read_text(encoding="utf-8"))
        raw[run["key"]] = provenance
        cfg = provenance["resolved_config"]
        train = provenance["datasets"]["train"]
        val = provenance["datasets"]["val"]
        last_checkpoint = version_dir(run) / "checkpoints" / "last.ckpt"
        last_checkpoint_sha256 = sha256(last_checkpoint)
        explicit_final = (
            version_dir(run)
            / "checkpoints"
            / f"epoch={cfg['epoch'] - 1}-step={cfg['epoch'] * STEPS_PER_EPOCH}.ckpt"
        )
        if explicit_final.exists() and sha256(explicit_final) != last_checkpoint_sha256:
            raise AssertionError(
                f"Explicit final checkpoint differs from last.ckpt for {run['key']}"
            )
        rows.append(
            {
                "run_key": run["key"],
                "condition": run["condition"],
                "git_commit": provenance["git"]["commit"],
                "dirty_any": provenance["git"]["dirty_any"],
                "dirty_tracked": provenance["git"]["dirty_tracked"],
                "seed": provenance["seed"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epoch"],
                "check_val_every_n_epoch": cfg["check_val_every_n_epoch"],
                "num_candidates": cfg["num_candidates"],
                "use_twc": cfg["use_twc"],
                "twc_weight": cfg["twc_weight"],
                "train_tracklets": train["tracklets"],
                "train_frames": train["frames"],
                "train_selection_sha256": train["virtual_rate_selection_sha256"],
                "val_tracklets": val["tracklets"],
                "val_frames": val["frames"],
                "val_selection_sha256": val["virtual_rate_selection_sha256"],
                "config_sha256": provenance["config_sha256"],
                "resolved_config_sha256": provenance["resolved_config_sha256"],
                "last_checkpoint_sha256": last_checkpoint_sha256,
                "last_checkpoint_bytes": last_checkpoint.stat().st_size,
            }
        )
    return raw, rows


def assert_comparability(provenance: dict[str, dict[str, Any]]) -> None:
    comparable_fields = [
        ("git", "commit"),
        ("seed",),
        ("resolved_config", "batch_size"),
        ("resolved_config", "epoch"),
        ("resolved_config", "check_val_every_n_epoch"),
        ("resolved_config", "num_candidates"),
        ("datasets", "train", "tracklets"),
        ("datasets", "train", "frames"),
        ("datasets", "train", "virtual_rate_selection_sha256"),
        ("datasets", "val", "tracklets"),
        ("datasets", "val", "frames"),
        ("datasets", "val", "virtual_rate_selection_sha256"),
    ]

    def nested(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
        value: Any = item
        for key in keys:
            value = value[key]
        return value

    for keys in comparable_fields:
        values = {nested(provenance[key], keys) for key in ("A", "B", "C")}
        if len(values) != 1:
            raise AssertionError(f"A/B/C provenance mismatch at {'.'.join(keys)}: {values}")

    for key in ("A", "B", "C"):
        if provenance[key]["git"]["dirty_tracked"]:
            raise AssertionError(f"{key} was run with tracked source modifications")

    b_cfg = provenance["B"]["resolved_config"]
    c_cfg = provenance["C"]["resolved_config"]
    ignored = {"cfg", "log_dir", "tag", "twc_weight"}
    material_differences = {
        key
        for key in set(b_cfg) | set(c_cfg)
        if key not in ignored and b_cfg.get(key) != c_cfg.get(key)
    }
    if material_differences:
        raise AssertionError(
            f"B/C differ beyond twc_weight and run metadata: {sorted(material_differences)}"
        )
    if b_cfg["twc_weight"] != 0.0 or c_cfg["twc_weight"] != 0.05:
        raise AssertionError("Unexpected B/C TWC weights")


def read_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    expected_steps: list[int] | None = None
    for run in RUNS:
        for metric, (scalar_dir, tag, _) in METRICS.items():
            rows = scalar_rows(version_dir(run), scalar_dir, tag)
            steps = [int(row["step"]) for row in rows]
            if expected_steps is None:
                expected_steps = steps
            elif steps != expected_steps:
                raise AssertionError(f"Evaluation steps differ for {run['key']} {metric}")
            if len(rows) != 12 or steps[-1] != 60 * STEPS_PER_EPOCH:
                raise AssertionError(
                    f"Incomplete metric series for {run['key']} {metric}: "
                    f"{len(rows)} points, last step {steps[-1] if steps else 'missing'}"
                )
            for row in rows:
                points.append(
                    {
                        "run_key": run["key"],
                        "condition": run["condition"],
                        "metric": metric,
                        "epoch": int(row["epoch"]),
                        "step": int(row["step"]),
                        "value": float(row["value"]),
                    }
                )
            values = [float(row["value"]) for row in rows]
            late = [
                float(row["value"])
                for row in rows
                if int(row["epoch"]) >= LATE_START_EPOCH
            ]
            best = max(rows, key=lambda row: float(row["value"]))
            final = rows[-1]
            summaries.append(
                {
                    "run_key": run["key"],
                    "condition": run["condition"],
                    "metric": metric,
                    "final": float(final["value"]),
                    "final_epoch": int(final["epoch"]),
                    "final_step": int(final["step"]),
                    "best": float(best["value"]),
                    "best_epoch": int(best["epoch"]),
                    "best_step": int(best["step"]),
                    "best_final_gap": float(best["value"]) - float(final["value"]),
                    "late_mean_40_60": mean(late),
                    "late_std_40_60": std(late),
                    "mean_all_eval_points": mean(values),
                    "std_all_eval_points": std(values),
                    "num_eval_points": len(rows),
                }
            )
    return points, summaries


def compute_deltas(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["run_key"], row["metric"]): row for row in summaries}
    comparisons = [
        ("B-A", "B", "A", "paired-view augmentation effect"),
        ("C-B", "C", "B", "net corrected-TWC effect"),
        ("C-A", "C", "A", "end-to-end effect vs single-view"),
    ]
    statistics = [
        ("final", "Final"),
        ("best", "Best"),
        ("late_mean_40_60", "Late mean 40-60"),
        ("mean_all_eval_points", "Mean all eval points"),
    ]
    output: list[dict[str, Any]] = []
    for comparison, minuend, subtrahend, interpretation in comparisons:
        for metric in METRICS:
            for field, label in statistics:
                left = float(lookup[(minuend, metric)][field])
                right = float(lookup[(subtrahend, metric)][field])
                recovery = ""
                if comparison == "C-B":
                    a = float(lookup[("A", metric)][field])
                    b = float(lookup[("B", metric)][field])
                    recovery = (left - right) / (a - b) if a != b else float("nan")
                output.append(
                    {
                        "comparison": comparison,
                        "interpretation": interpretation,
                        "metric": metric,
                        "statistic": label,
                        "minuend_value": left,
                        "subtrahend_value": right,
                        "delta": left - right,
                        "paired_view_loss_recovered_fraction": recovery,
                    }
                )
    return output


def read_diagnostics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for run in RUNS[1:]:
        for diagnostic, scalar_dir in DIAGNOSTICS.items():
            path = version_dir(run) / scalar_dir
            if not path.exists():
                continue
            rows = scalar_rows(version_dir(run), scalar_dir, "loss")
            values = [float(row["value"]) for row in rows]
            if len(values) != 60 * STEPS_PER_EPOCH:
                raise AssertionError(
                    f"Incomplete diagnostic {run['key']} {diagnostic}: {len(values)}"
                )
            summaries.append(
                {
                    "run_key": run["key"],
                    "condition": run["condition"],
                    "diagnostic": diagnostic,
                    "final": values[-1],
                    "mean": mean(values),
                    "std": std(values),
                    "tail1000_mean": mean(values[-1000:]),
                    "tail1000_std": std(values[-1000:]),
                    "min": min(values),
                    "max": max(values),
                    "num_points": len(values),
                }
            )
            for start in range(0, len(rows), DIAG_BLOCK_SIZE):
                block = rows[start : start + DIAG_BLOCK_SIZE]
                blocks.append(
                    {
                        "run_key": run["key"],
                        "condition": run["condition"],
                        "diagnostic": diagnostic,
                        "end_step": int(block[-1]["step"]),
                        "end_epoch": float(block[-1]["step"]) / STEPS_PER_EPOCH,
                        "block_mean": mean([float(row["value"]) for row in block]),
                        "block_size": len(block),
                    }
                )
    return summaries, blocks


def assert_diagnostic_integrity(summaries: list[dict[str, Any]]) -> None:
    lookup = {(row["run_key"], row["diagnostic"]): row for row in summaries}
    for key in ("B", "C"):
        for diagnostic in ("twc_anchor_gap_max", "twc_current_point_gap_max"):
            if float(lookup[(key, diagnostic)]["max"]) != 0.0:
                raise AssertionError(f"Non-zero coordinate mismatch: {key} {diagnostic}")
    b_valid = scalar_rows(version_dir(RUNS[1]), DIAGNOSTICS["twc_valid_ratio"], "loss")
    c_valid = scalar_rows(version_dir(RUNS[2]), DIAGNOSTICS["twc_valid_ratio"], "loss")
    b_series = [(int(row["step"]), float(row["value"])) for row in b_valid]
    c_series = [(int(row["step"]), float(row["value"])) for row in c_valid]
    if b_series != c_series:
        raise AssertionError("B/C TWC valid-pair sequences differ")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(points: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharex=True)
    for ax, (metric, (_, _, label)) in zip(axes, METRICS.items()):
        metric_points = [row for row in points if row["metric"] == metric]
        values = [float(row["value"]) for row in metric_points]
        for run in RUNS:
            rows = [row for row in metric_points if row["run_key"] == run["key"]]
            ax.plot(
                [int(row["epoch"]) for row in rows],
                [float(row["value"]) for row in rows],
                color=run["color"],
                marker="o",
                markersize=4,
                linewidth=2.2,
                label=f"{run['key']}: {run['condition']}",
            )
            final = rows[-1]
            ax.annotate(
                f"{float(final['value']):.2f}",
                (int(final["epoch"]), float(final["value"])),
                xytext=(-4, 7),
                textcoords="offset points",
                ha="right",
                color=run["color"],
                fontsize=8,
                fontweight="bold",
            )
        margin = max(2.0, (max(values) - min(values)) * 0.10)
        ax.set_ylim(min(values) - margin, max(values) + margin)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Evaluation epoch")
        ax.set_ylabel(label)
        ax.set_xticks(range(5, 61, 5))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("TWC A/B/C seed42 on standard mini_val", y=1.02, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.965,
        "Same commit, seed, split, optimizer steps and 5-epoch evaluation schedule",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(fig, LINE_DIR, f"{PREFIX}_metric_curves")


def plot_deltas(deltas: list[dict[str, Any]]) -> None:
    shown_stats = ["Final", "Best", "Late mean 40-60"]
    colors = {"Final": "#4E79A7", "Best": "#B07AA1", "Late mean 40-60": "#59A14F"}
    comparisons = ["B-A", "C-B", "C-A"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), sharex=True)
    x = np.arange(len(comparisons))
    width = 0.24
    for ax, (metric, (_, _, label)) in zip(axes, METRICS.items()):
        relevant = {
            (row["comparison"], row["statistic"]): float(row["delta"])
            for row in deltas
            if row["metric"] == metric and row["statistic"] in shown_stats
        }
        all_values: list[float] = []
        for idx, statistic in enumerate(shown_stats):
            values = [relevant[(comparison, statistic)] for comparison in comparisons]
            all_values.extend(values)
            bars = ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                color=colors[statistic],
                label=statistic.replace(" 40-60", ""),
            )
            ax.bar_label(
                bars,
                labels=[f"{value:+.2f}" for value in values],
                padding=3,
                fontsize=8,
            )
        bound = max(abs(value) for value in all_values) * 1.24
        ax.set_ylim(-bound, bound)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(x, comparisons)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Delta (percentage points)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Decomposing paired-view and corrected-TWC effects", y=1.02, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.965,
        "B-A = paired-view effect; C-B = net TWC effect; C-A = end-to-end effect",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(fig, DELTA_DIR, f"{PREFIX}_effect_deltas")


def plot_diagnostics(blocks: list[dict[str, Any]]) -> None:
    panels = [
        ("loss_twc", "TWC loss", True),
        ("twc_center_gap", "Center gap", True),
        ("twc_angle_gap", "Angle gap", True),
        ("twc_valid_ratio", "Valid-pair ratio", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 7.5), sharex=True)
    for ax, (diagnostic, label, log_scale) in zip(axes.flat, panels):
        # Draw C first and the dashed B line second so exact overlaps remain visible.
        for run in reversed(RUNS[1:]):
            rows = [
                row
                for row in blocks
                if row["run_key"] == run["key"] and row["diagnostic"] == diagnostic
            ]
            ax.plot(
                [float(row["end_epoch"]) for row in rows],
                [float(row["block_mean"]) for row in rows],
                color=run["color"],
                linewidth=2.0,
                linestyle="--" if run["key"] == "B" else "-",
                label=f"{run['key']}: {run['condition']}",
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch (1,000-step block end)")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if diagnostic == "twc_valid_ratio":
            ax.text(
                0.03,
                0.06,
                "B = C at all 75,720 steps",
                transform=ax.transAxes,
                fontsize=9,
                color="#555555",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Paired-view training diagnostics: passive B vs optimized C", y=1.01, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.955,
        "Both runs compute the same diagnostic; only C applies a 0.05 consistency weight",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    save_figure(fig, DIAG_DIR, f"{PREFIX}_training_diagnostics")


def main() -> None:
    provenance, provenance_rows = load_provenance()
    assert_comparability(provenance)
    metric_points, metric_summaries = read_metrics()
    deltas = compute_deltas(metric_summaries)
    diagnostic_summaries, diagnostic_blocks = read_diagnostics()
    assert_diagnostic_integrity(diagnostic_summaries)

    write_csv(DATA_DIR / f"{PREFIX}_provenance.csv", provenance_rows)
    write_csv(DATA_DIR / f"{PREFIX}_metrics_points.csv", metric_points)
    write_csv(DATA_DIR / f"{PREFIX}_metrics_summary.csv", metric_summaries)
    write_csv(DATA_DIR / f"{PREFIX}_deltas.csv", deltas)
    write_csv(DATA_DIR / f"{PREFIX}_diagnostics_summary.csv", diagnostic_summaries)
    write_csv(DATA_DIR / f"{PREFIX}_diagnostics_block_points.csv", diagnostic_blocks)

    plot_metric_curves(metric_points)
    plot_deltas(deltas)
    plot_diagnostics(diagnostic_blocks)

    print("A/B/C provenance and event-series validation: PASS")
    print(f"Data written under: {DATA_DIR}")
    print(f"Figures written under: {OUT / 'figures'}")


if __name__ == "__main__":
    main()
