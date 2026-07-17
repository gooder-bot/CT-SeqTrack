#!/usr/bin/env python3
"""Summarize the aligned TrajTrack run against the plain SeqTrack3D baseline.

TrajTrack values come from the full-precision scalar rows pasted from the
completed server run.  The comparison is deliberately labelled diagnostic:
TrajTrack's current evaluator uses current-frame ground truth during proposal
refinement, so its values are not a fair online ranking against SeqTrack3D.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from summarize_latest_5runs import scalar_rows


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "compare_results"
DATA_DIR = OUT / "data"
REPORT_DIR = OUT / "reports"
BAR_DIR = OUT / "figures" / "bar_charts"
LINE_DIR = OUT / "figures" / "line_charts"

PREFIX = "trajtrack_gt_assisted_vs_plain_seqtrack"
LATE_START_EPOCH = 40
STEPS_PER_EPOCH = 1262

SEQTRACK_VERSION = (
    WORKSPACE
    / "seqtrack/output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16/lightning_logs/version_0"
)
SEQTRACK_SOURCE = str(SEQTRACK_VERSION)

TRAJTRACK_RUN = "trajtrack_car_mini_seqtrack_protocol_seed42_gpu0_20260714-175518"
TRAJTRACK_REMOTE_ROOT = "/home/lishengjie/study/lcyu/trajtrack"
TRAJTRACK_SCALARS = (
    f"{TRAJTRACK_REMOTE_ROOT}/work_dirs/{TRAJTRACK_RUN}/"
    "20260714_175547/vis_data/scalars.json"
)
TRAJTRACK_LOG = f"{TRAJTRACK_REMOTE_ROOT}/logs/seqtrack_protocol/{TRAJTRACK_RUN}.log"
TRAJTRACK_CHECKPOINT = (
    f"{TRAJTRACK_REMOTE_ROOT}/work_dirs/{TRAJTRACK_RUN}/epoch_60.pth"
)
TRAJTRACK_LAST_CHECKPOINT = (
    f"{TRAJTRACK_REMOTE_ROOT}/work_dirs/{TRAJTRACK_RUN}/last_checkpoint"
)
TRAJTRACK_CHECKPOINT_SIZE_BYTES = 492_332_184
LOCAL_TERMINAL_EVIDENCE = (
    r"C:\Users\25227\.codex\attachments\036135a9-87df-421f-afab-20c87599ad0d\pasted-text.txt"
)

METRICS = {
    "success": ("metrics_test_success", "Success"),
    "precision": ("metrics_test_precision", "Precision"),
}

METHODS = [
    {
        "run_key": "seqtrack_plain_seed42",
        "method": "SeqTrack3D plain",
        "evaluation_mode": "online_gt_free",
        "online_comparable": True,
        "source_path": SEQTRACK_SOURCE,
        "color": "#4E79A7",
    },
    {
        "run_key": "trajtrack_gt_assisted_seed42",
        "method": "TrajTrack",
        "evaluation_mode": "gt_assisted_refinement",
        "online_comparable": False,
        "source_path": TRAJTRACK_SCALARS,
        "color": "#F28E2B",
    },
]

# Full-precision rows extracted from the completed server scalars.json.
TRAJTRACK_POINTS = [
    (5, 65.22682189941406, 76.48267364501953),
    (10, 58.359622955322266, 66.54205322265625),
    (15, 64.9990234375, 78.05198669433594),
    (20, 68.13375854492188, 80.78271484375),
    (25, 66.01732635498047, 78.69839477539062),
    (30, 64.98929595947266, 77.35105895996094),
    (35, 66.6092300415039, 78.8746109008789),
    (40, 64.86858367919922, 76.88571166992188),
    (45, 63.235008239746094, 77.56230163574219),
    (50, 65.08858489990234, 78.30899810791016),
    (55, 63.97098922729492, 77.05997467041016),
    (60, 64.94256591796875, 79.0722427368164),
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def fmt(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    baseline = METHODS[0]
    for metric, (scalar_dir, _) in METRICS.items():
        rows = scalar_rows(SEQTRACK_VERSION, scalar_dir, "metrics/test")
        for row in rows:
            points.append(
                {
                    "run_key": baseline["run_key"],
                    "method": baseline["method"],
                    "evaluation_mode": baseline["evaluation_mode"],
                    "online_comparable": baseline["online_comparable"],
                    "metric": metric,
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "value": float(row["value"]),
                    "source_tag": "metrics/test",
                    "source_path": baseline["source_path"],
                }
            )

    trajtrack = METHODS[1]
    for epoch, success, precision in TRAJTRACK_POINTS:
        for metric, value in (("success", success), ("precision", precision)):
            points.append(
                {
                    "run_key": trajtrack["run_key"],
                    "method": trajtrack["method"],
                    "evaluation_mode": trajtrack["evaluation_mode"],
                    "online_comparable": trajtrack["online_comparable"],
                    "metric": metric,
                    "epoch": epoch,
                    "step": epoch * STEPS_PER_EPOCH,
                    "value": value,
                    "source_tag": metric,
                    "source_path": trajtrack["source_path"],
                }
            )
    return points


def summarize(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        for metric in METRICS:
            rows = sorted(
                (
                    row
                    for row in points
                    if row["run_key"] == method["run_key"]
                    and row["metric"] == metric
                ),
                key=lambda row: int(row["epoch"]),
            )
            final = rows[-1]
            best = max(rows, key=lambda row: float(row["value"]))
            late_values = [
                float(row["value"])
                for row in rows
                if int(row["epoch"]) >= LATE_START_EPOCH
            ]
            summaries.append(
                {
                    "run_key": method["run_key"],
                    "method": method["method"],
                    "evaluation_mode": method["evaluation_mode"],
                    "online_comparable": method["online_comparable"],
                    "metric": metric,
                    "final": float(final["value"]),
                    "final_epoch": int(final["epoch"]),
                    "best_observed": float(best["value"]),
                    "best_epoch": int(best["epoch"]),
                    "best_final_gap": float(best["value"]) - float(final["value"]),
                    "late_mean_40_60": mean(late_values),
                    "late_std_40_60": std(late_values),
                    "mean_all": mean([float(row["value"]) for row in rows]),
                    "num_eval_points": len(rows),
                    "checkpoint_policy": (
                        "final epoch60 reference"
                        if method["online_comparable"]
                        else "fixed epoch60; do not select by GT-assisted metric"
                    ),
                    "source_path": method["source_path"],
                    "note": (
                        "GT-free online reference"
                        if method["online_comparable"]
                        else "Diagnostic only: evaluator uses current-frame GT during refinement"
                    ),
                }
            )
    return summaries


def run_manifest() -> list[dict[str, Any]]:
    """Preserve the completed-run identity and audit evidence in one row."""
    return [
        {
            "run_key": "trajtrack_gt_assisted_seed42",
            "run_name": TRAJTRACK_RUN,
            "dataset": "nuScenes-mini",
            "category": "Car",
            "train_split": "mini_train",
            "val_split": "mini_val",
            "seed": 42,
            "epochs": 60,
            "batch_size": 16,
            "num_candidates": 4,
            "num_workers": 12,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "validation_interval_epochs": 5,
            "unique_validation_points": len(TRAJTRACK_POINTS),
            "validation_epochs": ";".join(str(row[0]) for row in TRAJTRACK_POINTS),
            "checkpoint_path": TRAJTRACK_CHECKPOINT,
            "checkpoint_size_bytes": TRAJTRACK_CHECKPOINT_SIZE_BYTES,
            "last_checkpoint_path": TRAJTRACK_LAST_CHECKPOINT,
            "last_checkpoint_target": TRAJTRACK_CHECKPOINT,
            "scalars_path": TRAJTRACK_SCALARS,
            "log_path": TRAJTRACK_LOG,
            "training_complete": True,
            "completion_evidence": (
                "Saving checkpoint at 60 epochs; Epoch(val) [60][116/116]"
            ),
            "evaluation_mode": "gt_assisted_refinement",
            "online_comparable": False,
            "local_terminal_evidence": LOCAL_TERMINAL_EVIDENCE,
        }
    ]


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_final_reference(summaries: list[dict[str, Any]]) -> None:
    lookup = {
        (row["run_key"], row["metric"]): row
        for row in summaries
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for ax, (metric, (_, label)) in zip(axes, METRICS.items()):
        values = [
            float(lookup[(method["run_key"], metric)]["final"])
            for method in METHODS
        ]
        bars = ax.bar(
            np.arange(2),
            values,
            color=[METHODS[0]["color"], METHODS[1]["color"]],
            edgecolor=[METHODS[0]["color"], "#555555"],
            linewidth=[0.8, 1.2],
        )
        bars[1].set_hatch("///")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(label)
        ax.set_xticks(
            np.arange(2),
            ["SeqTrack3D plain\nonline", "TrajTrack\nGT-assisted*"],
        )
        ax.set_ylabel(label)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Final scores: diagnostic reference only\n"
        "TrajTrack uses GT-assisted refinement — not a fair online ranking",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, BAR_DIR, f"{PREFIX}_final_reference")


def plot_curves(points: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    for ax, (metric, (_, label)) in zip(axes, METRICS.items()):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in points
                    if row["run_key"] == method["run_key"]
                    and row["metric"] == metric
                ),
                key=lambda row: int(row["epoch"]),
            )
            is_trajtrack = not method["online_comparable"]
            ax.plot(
                [int(row["epoch"]) for row in rows],
                [float(row["value"]) for row in rows],
                color=method["color"],
                linestyle="--" if is_trajtrack else "-",
                marker="o" if is_trajtrack else "x",
                markerfacecolor="white" if is_trajtrack else method["color"],
                linewidth=2,
                markersize=5,
                label=(
                    "TrajTrack (GT-assisted)*"
                    if is_trajtrack
                    else "SeqTrack3D plain (online)"
                ),
            )
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_xticks(range(5, 61, 5))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Validation trajectories: protocol-aligned training, different evaluators\n"
        "TrajTrack curve is GT-assisted diagnostic evidence",
        y=1.02,
        fontsize=12,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save_figure(fig, LINE_DIR, f"{PREFIX}_curves")


def write_report(summaries: list[dict[str, Any]]) -> Path:
    lookup = {
        (row["run_key"], row["metric"]): row
        for row in summaries
    }
    baseline = METHODS[0]["run_key"]
    trajtrack = METHODS[1]["run_key"]
    report = [
        "# TrajTrack GT-assisted 与 Plain SeqTrack3D 参考对比",
        "",
        "> **边界：这不是公平在线排名。** TrajTrack 当前 evaluator 使用当前帧 GT overlap 触发 refinement，并用 GT overlap 从 proposals 中选择结果。下面数值只能作为实现诊断和带 oracle 辅助的参考。",
        "",
        "## 运行完整性",
        "",
        f"- run: `{TRAJTRACK_RUN}`",
        "- `epoch_60.pth` 已生成（492,332,184 bytes；`ls -lh` 显示 470 MB），`last_checkpoint` 指向该文件。",
        "- 日志包含 `Saving checkpoint at 60 epochs`，随后完成 epoch60 的 `116/116` validation。",
        "- scalars 中有 12 个唯一验证点（epoch5-60，每 5 epoch 一次），日志四位小数与全精度 scalars 一致。",
        "",
        "## 协议与 evaluator",
        "",
        "| 项目 | Plain SeqTrack3D | TrajTrack |",
        "| --- | --- | --- |",
        "| dataset/category | nuScenes-mini / Car | nuScenes-mini / Car |",
        "| split | mini_train / mini_val | mini_train / mini_val |",
        "| seed / epochs / batch | 42 / 60 / 16 | 42 / 60 / 16 |",
        "| candidates / workers | 4 / 12 | 4 / 12 |",
        "| steps per epoch / val interval | 1262 / 5 | 1262 / 5 |",
        "| evaluator | GT-free online | GT-assisted refinement |",
        "| fair online ranking | reference | **no** |",
        "",
        "## Final 指标",
        "",
        "| method | evaluation mode | Success | Precision | fair online ranking |",
        "| --- | --- | ---: | ---: | --- |",
        f"| SeqTrack3D plain | GT-free online | {fmt(lookup[(baseline, 'success')]['final'])} | {fmt(lookup[(baseline, 'precision')]['final'])} | reference |",
        f"| TrajTrack | GT-assisted refinement | {fmt(lookup[(trajtrack, 'success')]['final'])} | {fmt(lookup[(trajtrack, 'precision')]['final'])} | no |",
        "",
        "## 完整汇总",
        "",
        "| method | metric | final | best observed | best epoch | late mean 40-60 | late std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        for metric, (_, label) in METRICS.items():
            row = lookup[(method["run_key"], metric)]
            report.append(
                f"| {method['method']} | {label} | {fmt(row['final'])} | "
                f"{fmt(row['best_observed'])} | {row['best_epoch']} | "
                f"{fmt(row['late_mean_40_60'])} | {fmt(row['late_std_40_60'])} |"
            )
    report.extend(
        [
            "",
            "## 算术差值（仅描述，不代表方法增益）",
            "",
            "| metric | final difference | best-observed difference | late-mean difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric, (_, label) in METRICS.items():
        base = lookup[(baseline, metric)]
        traj = lookup[(trajtrack, metric)]
        report.append(
            f"| {label} | {fmt(float(traj['final']) - float(base['final']))} | "
            f"{fmt(float(traj['best_observed']) - float(base['best_observed']))} | "
            f"{fmt(float(traj['late_mean_40_60']) - float(base['late_mean_40_60']))} |"
        )
    report.extend(
        [
            "",
            "这些差值混合了模型差异与 evaluator 的 GT oracle 信息，不能写成‘TrajTrack 提升了 X 点’。因此没有生成 `delta_charts` 性能增益图。",
            "",
            "## 结论",
            "",
            "1. TrajTrack aligned seed42 run 已完整训练到 60 epoch，训练预算与 plain SeqTrack3D 基本对齐。",
            "2. TrajTrack 的 Success 和 Precision 都在 epoch20 达到最高 observed value；epoch40-60 的波动较小。",
            "3. 当前数值较高，但 evaluator 使用 GT-assisted refinement，只能作为实现诊断，不能支持 TrajTrack 优于 SeqTrack3D 的论文结论。",
            "4. 公平比较需要改用 `pre_wo_refine()` 或单独实现不读取当前帧 GT 的 evaluator，并用固定 epoch60 checkpoint 重新评测。",
            "",
            "## 图表",
            "",
            f"![final reference](../figures/bar_charts/{PREFIX}_final_reference.png)",
            "",
            f"![validation curves](../figures/line_charts/{PREFIX}_curves.png)",
            "",
            "## 数据文件",
            "",
            f"- `../data/{PREFIX}_points.csv`",
            f"- `../data/{PREFIX}_summary.csv`",
            f"- `../data/{PREFIX}_run_manifest.csv`",
            "",
            "## 数据来源",
            "",
            f"- SeqTrack3D events: `{SEQTRACK_SOURCE}`",
            f"- TrajTrack scalars: `{TRAJTRACK_SCALARS}`",
            f"- TrajTrack log: `{TRAJTRACK_LOG}`",
            f"- TrajTrack final checkpoint: `{TRAJTRACK_CHECKPOINT}`",
            f"- Local terminal evidence: `{LOCAL_TERMINAL_EVIDENCE}`",
        ]
    )
    path = REPORT_DIR / f"{PREFIX}_reference.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def main() -> None:
    points = read_points()
    summaries = summarize(points)
    write_csv(DATA_DIR / f"{PREFIX}_points.csv", points)
    write_csv(DATA_DIR / f"{PREFIX}_summary.csv", summaries)
    write_csv(DATA_DIR / f"{PREFIX}_run_manifest.csv", run_manifest())
    plot_final_reference(summaries)
    plot_curves(points)
    report = write_report(summaries)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
