#!/usr/bin/env python3
"""Compare corrected-TWC seed42 runs with their aligned baseline families.

The script reads TensorBoard TFRecord files and writes focused comparison
tables plus three chart types: absolute bars, epoch curves, and summary gaps.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from summarize_latest_5runs import scalar_rows


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "compare_results"
DATA_DIR = OUT / "data"
REPORT_DIR = OUT / "reports"
LINE_DIR = OUT / "figures" / "line_charts"
BAR_DIR = OUT / "figures" / "bar_charts"
DELTA_DIR = OUT / "figures" / "delta_charts"
DIAG_DIR = OUT / "figures" / "diagnostics"

PREFIX = "corrected_twc_seed42"
COMPARISON_PREFIX = f"{PREFIX}_baseline_vs_twc"
STEPS_PER_EPOCH = 1262
LATE_START_EPOCH = 40

RUNS = [
    {
        "key": "a1_baseline",
        "model": "A1-order",
        "family": "A1",
        "condition": "Baseline",
        "version": ROOT
        / "output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1/lightning_logs/version_0",
    },
    {
        "key": "a1_corrected_twc",
        "model": "A1-order + corrected-TWC",
        "family": "A1",
        "condition": "corrected-TWC",
        "version": ROOT
        / "output/corrected_twc_gpu3_seed42_20260714_234729/a1_order_twc_w005_seed42/lightning_logs/version_0",
    },
    {
        "key": "a2_baseline",
        "model": "A2-order-dyn",
        "family": "A2",
        "condition": "Baseline",
        "version": ROOT
        / "output/20260531-2322-seqtrack3d_nuscenes_a2_order_dyn-ct_a2_order_dyn_car_60ep_bs16_gpu2/lightning_logs/version_0",
    },
    {
        "key": "a2_corrected_twc",
        "model": "A2-order-dyn + corrected-TWC",
        "family": "A2",
        "condition": "corrected-TWC",
        "version": ROOT
        / "output/corrected_twc_gpu3_seed42_20260714_234729/a2_order_dyn_twc_w005_seed42/lightning_logs/version_0",
    },
]

METRICS = {
    "success/test": ("metrics_test_success", "Success"),
    "precision/test": ("metrics_test_precision", "Precision"),
}

DIAGNOSTICS = {
    "loss_twc": "loss_loss_twc",
    "twc_valid_ratio": "loss_twc_valid_ratio",
    "twc_center_gap": "loss_twc_center_gap",
    "twc_angle_gap": "loss_twc_angle_gap",
    "twc_anchor_gap_max": "loss_twc_anchor_gap_max",
    "twc_current_point_gap_max": "loss_twc_current_point_gap_max",
}

COLORS = {
    "A1-order": "#4E79A7",
    "A1-order + corrected-TWC": "#59A14F",
    "A2-order-dyn": "#E15759",
    "A2-order-dyn + corrected-TWC": "#F28E2B",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_metric_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for run in RUNS:
        version = Path(run["version"])
        if not version.exists():
            raise FileNotFoundError(version)
        for metric, (scalar_dir, _) in METRICS.items():
            for row in scalar_rows(version, scalar_dir, "metrics/test"):
                points.append(
                    {
                        "run_key": run["key"],
                        "model": run["model"],
                        "family": run["family"],
                        "condition": run["condition"],
                        "metric": metric,
                        "epoch": int(row["epoch"]),
                        "step": int(row["step"]),
                        "value": float(row["value"]),
                    }
                )
    return points


def summarize_metrics(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run in RUNS:
        for metric in METRICS:
            rows = [
                row
                for row in points
                if row["run_key"] == run["key"] and row["metric"] == metric
            ]
            rows.sort(key=lambda row: int(row["step"]))
            best = max(rows, key=lambda row: float(row["value"]))
            final = rows[-1]
            late = [
                float(row["value"])
                for row in rows
                if int(row["epoch"]) >= LATE_START_EPOCH
            ]
            values = [float(row["value"]) for row in rows]
            summaries.append(
                {
                    "run_key": run["key"],
                    "model": run["model"],
                    "family": run["family"],
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
                    "mean_all": mean(values),
                    "std_all": std(values),
                    "num_eval_points": len(rows),
                }
            )
    return summaries


def compute_gaps(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["family"], row["condition"], row["metric"]): row
        for row in summaries
    }
    gaps: list[dict[str, Any]] = []
    statistics = [
        ("final", "Final"),
        ("best", "Best"),
        ("late_mean_40_60", "Late mean 40-60"),
    ]
    for family in ("A1", "A2"):
        for metric in METRICS:
            baseline = lookup[(family, "Baseline", metric)]
            corrected = lookup[(family, "corrected-TWC", metric)]
            for field, label in statistics:
                baseline_value = float(baseline[field])
                corrected_value = float(corrected[field])
                gaps.append(
                    {
                        "family": family,
                        "metric": metric,
                        "statistic": label,
                        "baseline": baseline_value,
                        "corrected_twc": corrected_value,
                        "gap_corrected_minus_baseline": corrected_value
                        - baseline_value,
                    }
                )
    return gaps


def diagnostic_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    block_points: list[dict[str, Any]] = []
    for run in RUNS:
        if run["condition"] != "corrected-TWC":
            continue
        for name, scalar_dir in DIAGNOSTICS.items():
            rows = scalar_rows(Path(run["version"]), scalar_dir, "loss")
            values = [float(row["value"]) for row in rows]
            summaries.append(
                {
                    "run_key": run["key"],
                    "model": run["model"],
                    "diagnostic": name,
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
            block_size = 500
            for start in range(0, len(rows), block_size):
                block = rows[start : start + block_size]
                block_points.append(
                    {
                        "run_key": run["key"],
                        "model": run["model"],
                        "diagnostic": name,
                        "step": int(block[-1]["step"]),
                        "epoch": float(block[-1]["step"]) / STEPS_PER_EPOCH,
                        "block_mean": mean([float(row["value"]) for row in block]),
                    }
                )
    return summaries, block_points


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(points: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharex=True, sharey="col")
    condition_styles = {
        "Baseline": {"color": "#4E79A7", "linestyle": "--", "marker": "x"},
        "corrected-TWC": {"color": "#59A14F", "linestyle": "-", "marker": "o"},
    }
    for row_idx, family in enumerate(("A1", "A2")):
        for col_idx, (metric, (_, metric_label)) in enumerate(METRICS.items()):
            ax = axes[row_idx, col_idx]
            for condition in ("Baseline", "corrected-TWC"):
                run = next(
                    item
                    for item in RUNS
                    if item["family"] == family and item["condition"] == condition
                )
                rows = sorted(
                    (
                        row
                        for row in points
                        if row["run_key"] == run["key"] and row["metric"] == metric
                    ),
                    key=lambda row: int(row["epoch"]),
                )
                ax.plot(
                    [int(row["epoch"]) for row in rows],
                    [float(row["value"]) for row in rows],
                    label=condition,
                    linewidth=2,
                    markersize=4,
                    **condition_styles[condition],
                )
            ax.set_title(f"{family} · {metric_label}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_label)
            ax.set_xticks(range(5, 61, 5))
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Corrected-TWC seed42 vs baseline: epoch curves", y=0.99, fontsize=13)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, LINE_DIR, f"{COMPARISON_PREFIX}_curves")


def plot_summary_bars(summaries: list[dict[str, Any]]) -> None:
    lookup = {
        (row["family"], row["condition"], row["metric"]): row
        for row in summaries
    }
    statistics = [
        ("final", "Final"),
        ("best", "Best"),
        ("late_mean_40_60", "Late mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharey="col")
    x = np.arange(len(statistics))
    width = 0.36
    condition_styles = [
        ("Baseline", "#4E79A7", -width / 2),
        ("corrected-TWC", "#59A14F", width / 2),
    ]
    column_maxima = [0.0, 0.0]
    for row_idx, family in enumerate(("A1", "A2")):
        for col_idx, (metric, (_, metric_label)) in enumerate(METRICS.items()):
            ax = axes[row_idx, col_idx]
            for condition, color, offset in condition_styles:
                summary = lookup[(family, condition, metric)]
                values = [float(summary[field]) for field, _ in statistics]
                column_maxima[col_idx] = max(column_maxima[col_idx], *values)
                bars = ax.bar(
                    x + offset,
                    values,
                    width,
                    label=condition,
                    color=color,
                )
                ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
            ax.set_title(f"{family} · {metric_label}")
            ax.set_xticks(x, [label for _, label in statistics])
            ax.set_ylabel(metric_label)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    for col_idx, maximum in enumerate(column_maxima):
        axes[0, col_idx].set_ylim(0, maximum * 1.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Corrected-TWC seed42 vs baseline: absolute metrics", y=0.99, fontsize=13)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, BAR_DIR, f"{COMPARISON_PREFIX}_final_best_late")


def plot_summary_gaps(gaps: list[dict[str, Any]]) -> None:
    lookup = {
        (row["family"], row["metric"], row["statistic"]): float(
            row["gap_corrected_minus_baseline"]
        )
        for row in gaps
    }
    statistics = [
        ("Final", "#59A14F", "o"),
        ("Best", "#4E79A7", "s"),
        ("Late mean 40-60", "#F28E2B", "^"),
    ]
    row_labels = [
        f"{family} · {statistic}"
        for family in ("A1", "A2")
        for statistic, _, _ in statistics
    ]
    y = np.arange(len(row_labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True, sharey=True)
    for ax, (metric, (_, metric_label)) in zip(axes, METRICS.items()):
        point_idx = 0
        for family in ("A1", "A2"):
            for statistic, color, marker in statistics:
                value = lookup[(family, metric, statistic)]
                ypos = y[point_idx]
                ax.hlines(ypos, min(0, value), max(0, value), color=color, linewidth=3)
                ax.scatter(value, ypos, color=color, marker=marker, s=55, zorder=3)
                ax.annotate(
                    f"{value:+.2f}",
                    (value, ypos),
                    xytext=(5 if value >= 0 else -5, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8,
                )
                point_idx += 1
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(metric_label)
        ax.set_xlabel("corrected-TWC − baseline")
        ax.set_yticks(y, row_labels)
        ax.set_xlim(-3.4, 5.9)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linewidth=3,
            label=statistic.replace(" 40-60", ""),
        )
        for statistic, color, marker in statistics
    ]
    fig.suptitle("Corrected-TWC seed42 vs baseline: summary gaps", y=0.99, fontsize=13)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save_figure(fig, DELTA_DIR, f"{COMPARISON_PREFIX}_summary_gaps")


def plot_diagnostics(block_points: list[dict[str, Any]]) -> None:
    panels = [
        ("loss_twc", "TWC loss", True),
        ("twc_valid_ratio", "Valid ratio", False),
        ("twc_center_gap", "Center gap", True),
        ("twc_angle_gap", "Angle gap", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), sharex=True)
    for ax, (name, label, log_scale) in zip(axes.flat, panels):
        for run in RUNS:
            if run["condition"] != "corrected-TWC":
                continue
            rows = [
                row
                for row in block_points
                if row["run_key"] == run["key"] and row["diagnostic"] == name
            ]
            ax.plot(
                [float(row["epoch"]) for row in rows],
                [float(row["block_mean"]) for row in rows],
                label=run["model"],
                color=COLORS[run["model"]],
                linewidth=1.7,
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Corrected-TWC training diagnostics (500-step block means)", y=0.985, fontsize=13)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, DIAG_DIR, f"{PREFIX}_diagnostics")


def fmt(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def write_report(
    summaries: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> Path:
    summary_lookup = {
        (row["family"], row["condition"], row["metric"]): row
        for row in summaries
    }
    gap_lookup = {
        (row["family"], row["metric"], row["statistic"]): row
        for row in gaps
    }
    diag_lookup = {
        (row["run_key"], row["diagnostic"]): row for row in diagnostics
    }
    report = [
        "# Corrected-TWC seed42 与 Baseline 对比",
        "",
        "## 实验完整性与口径",
        "",
        "- corrected-TWC 两组均有 12 个评测点、epoch-59 checkpoint 和 75720 optimizer steps。",
        "- A1/A2 baseline 与 corrected-TWC 均为 seed42、60 epoch、batch16、candidate4、每 5 epoch 评测，外层 DataLoader 均为 1262 steps/epoch。",
        "- baseline 是 2026-05-31 的旧 run，hparams 未记录 git commit；当前只能确认关键配置对齐。因此下列差距是配置级参考，不视为严格同代码提交的因果配对实验。",
        "- 指标来自 mini_val，但日志命名为 `metrics/test`，属于开发集证据。",
        "- 差距统一定义为 `corrected-TWC - baseline`：正值表示 corrected-TWC 更好，负值表示更差。",
        "- `Final` 是 epoch60 评测，`Best` 是各 run 自己的最佳评测点，`Late mean` 是 epoch40-60 的均值。",
        "",
        "## Tracking 指标",
        "",
        "| family | condition | metric | final | best | best epoch | late mean 40-60 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for family in ("A1", "A2"):
        for condition in ("Baseline", "corrected-TWC"):
            for metric in METRICS:
                row = summary_lookup[(family, condition, metric)]
                report.append(
                    f"| {family} | {condition} | {METRICS[metric][1]} | {fmt(row['final'])} | {fmt(row['best'])} | {row['best_epoch']} | {fmt(row['late_mean_40_60'])} |"
                )
    report.extend(
        [
            "",
            "## corrected-TWC 相对 baseline 的差距",
            "",
            "| family | metric | final delta | best delta | late-mean delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for family in ("A1", "A2"):
        for metric in METRICS:
            report.append(
                f"| {family} | {METRICS[metric][1]} | "
                f"{fmt(gap_lookup[(family, metric, 'Final')]['gap_corrected_minus_baseline'])} | "
                f"{fmt(gap_lookup[(family, metric, 'Best')]['gap_corrected_minus_baseline'])} | "
                f"{fmt(gap_lookup[(family, metric, 'Late mean 40-60')]['gap_corrected_minus_baseline'])} |"
            )
    report.extend(
        [
            "",
            "## 坐标修复后的 TWC 诊断",
            "",
            "| model | valid mean | TWC loss tail1000 | center gap tail1000 | angle gap tail1000 | anchor max | current XYZ max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in (
        ("a1_corrected_twc", "A1 + corrected-TWC"),
        ("a2_corrected_twc", "A2 + corrected-TWC"),
    ):
        report.append(
            f"| {label} | {fmt(diag_lookup[(key, 'twc_valid_ratio')]['mean'], 4)} | "
            f"{fmt(diag_lookup[(key, 'loss_twc')]['tail1000_mean'], 5)} | "
            f"{fmt(diag_lookup[(key, 'twc_center_gap')]['tail1000_mean'], 5)} | "
            f"{fmt(diag_lookup[(key, 'twc_angle_gap')]['tail1000_mean'], 5)} | "
            f"{fmt(diag_lookup[(key, 'twc_anchor_gap_max')]['max'], 1)} | "
            f"{fmt(diag_lookup[(key, 'twc_current_point_gap_max')]['max'], 1)} |"
        )
    report.extend(
        [
            "",
            "## 结论",
            "",
            "1. 以关键配置对齐的 baseline 为参考，corrected-TWC 在 A1 上形成 seed42 的正信号：final Success/Precision 分别提升 1.49/5.03，late mean 提升 0.99/2.67。",
            "2. corrected-TWC 没有给 A2 带来 tracking 收益：final Success/Precision 分别下降 0.93/2.07，late mean 下降 1.33/2.53。",
            "3. 两组 corrected-TWC 的 anchor gap max 和 current-point XYZ gap max 均为 0，说明坐标修正路径已正确生效；但更低的 TWC loss 不等价于更高的 tracking 指标。",
            "4. 当前只有 seed42。A1 的正信号需要 seed43/44 才能升级为稳定结论；A2 暂不建议接入 TWC 主线。",
            "",
            "## 图表目录",
            "",
            "| 图表类型 | 文件夹 | 内容 |",
            "| --- | --- | --- |",
            f"| 柱状图 | `../figures/bar_charts/` | baseline 与 corrected-TWC 的 Final/Best/Late mean 绝对指标 |",
            f"| 线性图 | `../figures/line_charts/` | 两组方法随 epoch 的 Success/Precision 曲线 |",
            f"| 差距图 | `../figures/delta_charts/` | `corrected-TWC - baseline` 的 Final/Best/Late mean 差距 |",
            "",
            "### 柱状图",
            "",
            f"![absolute comparison](../figures/bar_charts/{COMPARISON_PREFIX}_final_best_late.png)",
            "",
            "### 线性图",
            "",
            f"![epoch curves](../figures/line_charts/{COMPARISON_PREFIX}_curves.png)",
            "",
            "### 差距图",
            "",
            f"![summary gaps](../figures/delta_charts/{COMPARISON_PREFIX}_summary_gaps.png)",
            "",
            "### corrected-TWC 训练诊断（补充）",
            "",
            f"![diagnostics](../figures/diagnostics/{PREFIX}_diagnostics.png)",
            "",
            "## 数据文件",
            "",
            f"- `../data/{COMPARISON_PREFIX}_points.csv`：逐 epoch 原始评测点。",
            f"- `../data/{COMPARISON_PREFIX}_summary.csv`：Final/Best/Late mean 汇总。",
            f"- `../data/{COMPARISON_PREFIX}_gaps.csv`：corrected-TWC 与 baseline 的长表差距数据。",
            f"- `../data/{PREFIX}_diagnostics_summary.csv`",
            f"- `../data/{PREFIX}_diagnostics_block_points.csv`",
            "",
            "## 独立扩展参考",
            "",
            "- [TrajTrack GT-assisted 与 plain SeqTrack3D 参考对比](trajtrack_gt_assisted_vs_plain_seqtrack_reference.md)：单独保存，不与 corrected-TWC 的公平差距图混合。",
        ]
    )
    path = REPORT_DIR / f"{PREFIX}_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def main() -> None:
    points = read_metric_points()
    summaries = summarize_metrics(points)
    gaps = compute_gaps(summaries)
    diagnostics, diag_points = diagnostic_data()

    write_csv(DATA_DIR / f"{COMPARISON_PREFIX}_points.csv", points)
    write_csv(DATA_DIR / f"{COMPARISON_PREFIX}_summary.csv", summaries)
    write_csv(DATA_DIR / f"{COMPARISON_PREFIX}_gaps.csv", gaps)
    write_csv(DATA_DIR / f"{PREFIX}_diagnostics_summary.csv", diagnostics)
    write_csv(DATA_DIR / f"{PREFIX}_diagnostics_block_points.csv", diag_points)

    plot_metric_curves(points)
    plot_summary_bars(summaries)
    plot_summary_gaps(gaps)
    plot_diagnostics(diag_points)
    report = write_report(summaries, gaps, diagnostics)

    print(f"Wrote {report}")
    print(f"Charts: {BAR_DIR}, {LINE_DIR}, {DELTA_DIR}")


if __name__ == "__main__":
    main()
