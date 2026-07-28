#!/usr/bin/env python3
"""Summarize the paired six-run nuScenes-mini HTV experiment.

The script discovers the three A1/A2 protocol pairs under ``output/``, reads
their TensorBoard scalar files, validates within-protocol step alignment, and
writes CSV tables, a Markdown report, and publication-ready PNG figures.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_180ep_baseline_a2_a3 import read_scalar_events


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
COMPARE_DIR = ROOT / "compare_results"
DATA_DIR = COMPARE_DIR / "data"
REPORT_DIR = COMPARE_DIR / "reports"
LINE_DIR = COMPARE_DIR / "figures" / "line_charts"
BAR_DIR = COMPARE_DIR / "figures" / "bar_charts"
DELTA_DIR = COMPARE_DIR / "figures" / "delta_charts"

PREFIX = "htv_6runs"
PROTOCOLS = ["gap1124", "burst_drop", "random20"]
PROTOCOL_LABELS = {
    "gap1124": "Gap 1-1-2-4",
    "burst_drop": "Burst drop",
    "random20": "Random drop 20%",
}
MODELS = ["A1-order", "A2-order-dyn"]
METRICS = ["success/test", "precision/test"]
METRIC_LABELS = {"success/test": "Success", "precision/test": "Precision"}
COLORS = {"A1-order": "#4E79A7", "A2-order-dyn": "#E15759"}
MARKERS = {"A1-order": "o", "A2-order-dyn": "s"}
LATE_START_EPOCH = 40

RUN_PATTERN = re.compile(
    r"htv_(?P<protocol>gap1124|burst_drop|random20)_"
    r"(?P<model>a1_order|a2_order_dyn)_seed42"
)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def hparam(text: str, key: str, default: str = "") -> str:
    match = re.search(rf"^    {re.escape(key)}:\s*(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def root_hparam(text: str, key: str, default: str = "") -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def discover_runs() -> dict[tuple[str, str], dict[str, object]]:
    runs: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(OUTPUT_DIR.iterdir()):
        if not path.is_dir():
            continue
        match = RUN_PATTERN.search(path.name)
        if not match:
            continue
        protocol = match.group("protocol")
        model = "A1-order" if match.group("model") == "a1_order" else "A2-order-dyn"
        log_name = "lighting_logs" if (path / "lighting_logs").exists() else "lightning_logs"
        version_dir = path / log_name / "version_0"
        hp_path = version_dir / "hparams.yaml"
        if not hp_path.is_file():
            raise FileNotFoundError(f"Missing hparams.yaml: {hp_path}")
        text = hp_path.read_text(encoding="utf-8")
        train_length = int(root_hparam(text, "train_dataloader_length", "0"))
        if train_length <= 0:
            raise ValueError(f"Invalid train_dataloader_length in {hp_path}")
        runs[(protocol, model)] = {
            "protocol": protocol,
            "model": model,
            "run_dir": path,
            "version_dir": version_dir,
            "train_length": train_length,
            "cfg": hparam(text, "cfg"),
            "seed": int(hparam(text, "seed", "-1")),
            "epochs": int(hparam(text, "epoch", "0")),
            "batch_size": int(hparam(text, "batch_size", "0")),
            "workers": int(hparam(text, "workers", "0")),
            "num_candidates": int(hparam(text, "num_candidates", "0")),
            "check_every": int(hparam(text, "check_val_every_n_epoch", "0")),
            "use_dynamics": hparam(text, "use_dynamics_encoder").lower() == "true",
            "virtual_rate_mode": hparam(text, "virtual_rate_mode"),
            "virtual_rate_seed": int(hparam(text, "virtual_rate_seed", "-1")),
            "virtual_rate_manifest": hparam(text, "virtual_rate_manifest"),
        }

    expected = {(protocol, model) for protocol in PROTOCOLS for model in MODELS}
    missing = sorted(expected - set(runs))
    extra = sorted(set(runs) - expected)
    if missing or extra:
        raise RuntimeError(f"Expected exactly six HTV runs; missing={missing}, extra={extra}")

    for protocol in PROTOCOLS:
        a1 = runs[(protocol, "A1-order")]
        a2 = runs[(protocol, "A2-order-dyn")]
        comparable_keys = [
            "train_length",
            "seed",
            "epochs",
            "batch_size",
            "workers",
            "num_candidates",
            "check_every",
            "virtual_rate_mode",
            "virtual_rate_seed",
            "virtual_rate_manifest",
        ]
        mismatches = [key for key in comparable_keys if a1[key] != a2[key]]
        if mismatches:
            raise RuntimeError(f"{protocol} A1/A2 mismatch in {mismatches}")
        if a1["use_dynamics"] or not a2["use_dynamics"]:
            raise RuntimeError(f"Unexpected dynamics flags for {protocol}")
    return runs


def load_metric(run: dict[str, object], metric: str) -> list[dict[str, object]]:
    metric_dir = "metrics_test_success" if metric == "success/test" else "metrics_test_precision"
    event_files = sorted((Path(run["version_dir"]) / metric_dir).glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No event file under {Path(run['version_dir']) / metric_dir}")
    by_step: dict[int, float] = {}
    for event_file in event_files:
        for step, value in read_scalar_events(event_file, "metrics/test"):
            by_step[int(step)] = float(value)
    rows = []
    train_length = int(run["train_length"])
    for point_idx, step in enumerate(sorted(by_step)):
        epoch = int(round(step / train_length))
        rows.append(
            {
                "protocol": run["protocol"],
                "model": run["model"],
                "metric": metric,
                "point_idx": point_idx,
                "epoch": epoch,
                "step": step,
                "value": by_step[step],
                "run_dir": Path(run["run_dir"]).name,
            }
        )
    expected_epochs = list(range(int(run["check_every"]), int(run["epochs"]) + 1, int(run["check_every"])))
    actual_epochs = [int(row["epoch"]) for row in rows]
    if actual_epochs != expected_epochs:
        raise RuntimeError(
            f"Unexpected evaluation epochs for {run['protocol']} {run['model']} {metric}: "
            f"{actual_epochs} != {expected_epochs}"
        )
    return rows


def summarize(run: dict[str, object], metric: str, rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["value"]) for row in rows]
    best_row = max(rows, key=lambda row: float(row["value"]))
    final_row = rows[-1]
    late_values = [float(row["value"]) for row in rows if int(row["epoch"]) >= LATE_START_EPOCH]
    return {
        "protocol": run["protocol"],
        "model": run["model"],
        "metric": metric,
        "final": float(final_row["value"]),
        "final_epoch": int(final_row["epoch"]),
        "final_step": int(final_row["step"]),
        "best": float(best_row["value"]),
        "best_epoch": int(best_row["epoch"]),
        "best_step": int(best_row["step"]),
        "best_final_gap": float(best_row["value"]) - float(final_row["value"]),
        "mean_all": mean(values),
        "std_all": std(values),
        "late_mean_40_60": mean(late_values),
        "late_std_40_60": std(late_values),
        "train_dataloader_length": int(run["train_length"]),
        "seed": int(run["seed"]),
        "cfg": run["cfg"],
        "run_dir": Path(run["run_dir"]).name,
    }


def build_deltas(
    summaries: list[dict[str, object]], points: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_map = {
        (str(row["protocol"]), str(row["model"]), str(row["metric"])): row
        for row in summaries
    }
    delta_summary = []
    delta_points = []
    for protocol in PROTOCOLS:
        for metric in METRICS:
            a1 = summary_map[(protocol, "A1-order", metric)]
            a2 = summary_map[(protocol, "A2-order-dyn", metric)]
            delta_summary.append(
                {
                    "protocol": protocol,
                    "metric": metric,
                    "final_delta_a2_minus_a1": float(a2["final"]) - float(a1["final"]),
                    "best_delta_a2_minus_a1": float(a2["best"]) - float(a1["best"]),
                    "late_mean_delta_a2_minus_a1": float(a2["late_mean_40_60"])
                    - float(a1["late_mean_40_60"]),
                    "mean_all_delta_a2_minus_a1": float(a2["mean_all"]) - float(a1["mean_all"]),
                }
            )
            for epoch in range(5, 61, 5):
                a1_point = next(
                    row
                    for row in points
                    if row["protocol"] == protocol
                    and row["model"] == "A1-order"
                    and row["metric"] == metric
                    and row["epoch"] == epoch
                )
                a2_point = next(
                    row
                    for row in points
                    if row["protocol"] == protocol
                    and row["model"] == "A2-order-dyn"
                    and row["metric"] == metric
                    and row["epoch"] == epoch
                )
                delta_points.append(
                    {
                        "protocol": protocol,
                        "metric": metric,
                        "epoch": epoch,
                        "a1_value": a1_point["value"],
                        "a2_value": a2_point["value"],
                        "delta_a2_minus_a1": float(a2_point["value"]) - float(a1_point["value"]),
                    }
                )
    return delta_summary, delta_points


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curves(points: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.4), sharex=True)
    for col, protocol in enumerate(PROTOCOLS):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for model in MODELS:
                series = [
                    item
                    for item in points
                    if item["protocol"] == protocol
                    and item["model"] == model
                    and item["metric"] == metric
                ]
                x = [int(item["epoch"]) for item in series]
                y = [float(item["value"]) for item in series]
                ax.plot(
                    x,
                    y,
                    label=model,
                    color=COLORS[model],
                    marker=MARKERS[model],
                    linewidth=2,
                    markersize=4,
                )
                ax.annotate(
                    f"{y[-1]:.1f}",
                    (x[-1], y[-1]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=COLORS[model],
                )
            ax.set_title(PROTOCOL_LABELS[protocol] if row == 0 else "")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_xlabel("Epoch" if row == 1 else "")
            ax.set_xticks(range(5, 61, 10))
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("nuScenes-mini HTV: A1-order vs A2-order-dyn", y=0.985, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, LINE_DIR, f"{PREFIX}_metric_curves")


def plot_summary(summaries: list[dict[str, object]]) -> None:
    summary_map = {
        (str(row["protocol"]), str(row["model"]), str(row["metric"])): row
        for row in summaries
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = np.arange(len(PROTOCOLS))
    width = 0.34
    for ax, metric in zip(axes, METRICS):
        for model_idx, model in enumerate(MODELS):
            offset = (-0.5 if model_idx == 0 else 0.5) * width
            rows = [summary_map[(protocol, model, metric)] for protocol in PROTOCOLS]
            finals = [float(row["final"]) for row in rows]
            bests = [float(row["best"]) for row in rows]
            lates = [float(row["late_mean_40_60"]) for row in rows]
            bars = ax.bar(x + offset, finals, width, color=COLORS[model], alpha=0.82, label=f"{model} final")
            ax.scatter(x + offset, bests, marker="^", s=52, facecolors="none", edgecolors=COLORS[model], linewidths=1.8, label=f"{model} best")
            ax.scatter(x + offset, lates, marker="_", s=150, color=COLORS[model], linewidths=2.5, label=f"{model} late mean")
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x, [PROTOCOL_LABELS[p] for p in PROTOCOLS])
        ax.set_ylabel("Metric value")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Final bars, best triangles, and epoch 40-60 late means",
        y=0.985,
        fontsize=13,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save_figure(fig, BAR_DIR, f"{PREFIX}_final_best_late_summary")


def plot_deltas(delta_summary: list[dict[str, object]]) -> None:
    delta_map = {
        (str(row["protocol"]), str(row["metric"])): row for row in delta_summary
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    x = np.arange(len(PROTOCOLS))
    width = 0.36
    for ax, metric in zip(axes, METRICS):
        rows = [delta_map[(protocol, metric)] for protocol in PROTOCOLS]
        finals = [float(row["final_delta_a2_minus_a1"]) for row in rows]
        lates = [float(row["late_mean_delta_a2_minus_a1"]) for row in rows]
        bars_final = ax.bar(x - width / 2, finals, width, color="#59A14F", label="Final delta")
        bars_late = ax.bar(x + width / 2, lates, width, color="#F28E2B", label="Late-mean delta")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.bar_label(bars_final, fmt="%+.1f", padding=2, fontsize=8)
        ax.bar_label(bars_late, fmt="%+.1f", padding=2, fontsize=8)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x, [PROTOCOL_LABELS[p] for p in PROTOCOLS])
        ax.set_ylabel("A2 - A1")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Dynamics effect under HTV protocols", y=0.985, fontsize=13)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save_figure(fig, DELTA_DIR, f"{PREFIX}_a2_minus_a1_deltas")


def fmt(value: object, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def write_report(
    summaries: list[dict[str, object]], delta_summary: list[dict[str, object]]
) -> None:
    summary_map = {
        (str(row["protocol"]), str(row["model"]), str(row["metric"])): row
        for row in summaries
    }
    delta_map = {
        (str(row["protocol"]), str(row["metric"])): row for row in delta_summary
    }
    lines = [
        "# nuScenes-mini HTV 六组实验总结",
        "",
        "## 协议与公平性",
        "",
        "六组实验均为 seed 42、60 epoch、batch size 16、candidate 4、每 5 epoch 评测一次。",
        "同一协议内 A1/A2 的 DataLoader 长度和总 optimizer steps 一致；三种协议之间样本数不同，因此只做协议内 A2-A1 配对比较。",
        "`virtual_rate_manifest` 为空，但 virtual-rate seed 固定为 42；本轮是确定性配置配对，不是冻结 manifest 配对。",
        "当前指标来自 `mini_val` 上记录为 `metrics/test` 的开发评测，不应写成正式 held-out test 结果。",
        "",
        "| protocol | train batches/epoch | total steps |",
        "| --- | ---: | ---: |",
    ]
    for protocol in PROTOCOLS:
        row = summary_map[(protocol, "A1-order", "success/test")]
        lines.append(
            f"| {PROTOCOL_LABELS[protocol]} | {row['train_dataloader_length']} | {row['final_step']} |"
        )
    lines += [
        "",
        "## 模型指标",
        "",
        "| protocol | model | metric | final | best | best epoch | best-final gap | late mean 40-60 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for protocol in PROTOCOLS:
        for model in MODELS:
            for metric in METRICS:
                row = summary_map[(protocol, model, metric)]
                lines.append(
                    "| {protocol} | {model} | {metric} | {final} | {best} | {best_epoch} | {gap} | {late} |".format(
                        protocol=PROTOCOL_LABELS[protocol],
                        model=model,
                        metric=METRIC_LABELS[metric],
                        final=fmt(row["final"]),
                        best=fmt(row["best"]),
                        best_epoch=row["best_epoch"],
                        gap=fmt(row["best_final_gap"]),
                        late=fmt(row["late_mean_40_60"]),
                    )
                )
    lines += [
        "",
        "## A2-order-dyn 相对 A1-order",
        "",
        "| protocol | metric | final delta | best delta | late-mean delta | all-eval mean delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for protocol in PROTOCOLS:
        for metric in METRICS:
            row = delta_map[(protocol, metric)]
            lines.append(
                "| {protocol} | {metric} | {final} | {best} | {late} | {mean_delta} |".format(
                    protocol=PROTOCOL_LABELS[protocol],
                    metric=METRIC_LABELS[metric],
                    final=fmt(row["final_delta_a2_minus_a1"]),
                    best=fmt(row["best_delta_a2_minus_a1"]),
                    late=fmt(row["late_mean_delta_a2_minus_a1"]),
                    mean_delta=fmt(row["mean_all_delta_a2_minus_a1"]),
                )
            )

    gap_precision = summary_map[("gap1124", "A2-order-dyn", "precision/test")]
    random_success_delta = delta_map[("random20", "success/test")]
    random_precision_delta = delta_map[("random20", "precision/test")]
    burst_success_delta = delta_map[("burst_drop", "success/test")]
    burst_precision_delta = delta_map[("burst_drop", "precision/test")]
    gap_success_delta = delta_map[("gap1124", "success/test")]
    gap_precision_delta = delta_map[("gap1124", "precision/test")]
    lines += [
        "",
        "## 结论",
        "",
        "1. **A2 dynamics 只在 random20 上形成一致的 final 正收益。** "
        f"Success {fmt(random_success_delta['final_delta_a2_minus_a1'])}，"
        f"Precision {fmt(random_precision_delta['final_delta_a2_minus_a1'])}；late mean 也为正。",
        "2. **在更强的 gap1124 和 burst-drop 上，A2 明显低于 A1。** "
        f"gap1124 final 为 {fmt(gap_success_delta['final_delta_a2_minus_a1'])} / "
        f"{fmt(gap_precision_delta['final_delta_a2_minus_a1'])}，burst-drop 为 "
        f"{fmt(burst_success_delta['final_delta_a2_minus_a1'])} / "
        f"{fmt(burst_precision_delta['final_delta_a2_minus_a1'])}。",
        "3. **gap1124 的 A2 存在明显早期高点和后期回落。** "
        f"Precision best={fmt(gap_precision['best'])}（epoch {gap_precision['best_epoch']}），"
        f"final={fmt(gap_precision['final'])}，best-final gap={fmt(gap_precision['best_final_gap'])}。"
        "这更像训练/监督稳定性问题，而不是稳定的时间建模收益。",
        "4. **当前结果不支持‘时间间隔越不规则，feature-concat dynamics 越有效’。** "
        "相反，旧 A2 feature-concat 只在温和 random20 上受益，在强 gap/burst 下退化，"
        "支持继续验证 observation-first bounded residual，而不支持把旧 A2 直接作为主方法。",
        "5. **这仍是单 seed、mini_val 筛选证据。** 尚不能形成统计结论，也没有 true-dt/fixed-dt/shuffled-dt 因果对照；"
        "下一步应优先冻结 manifest，运行 residual 的三 seed 配对矩阵和困难分桶。",
        "",
        "## 图表",
        "",
        f"![metric curves](../figures/line_charts/{PREFIX}_metric_curves.png)",
        "",
        f"![final best late](../figures/bar_charts/{PREFIX}_final_best_late_summary.png)",
        "",
        f"![A2 minus A1](../figures/delta_charts/{PREFIX}_a2_minus_a1_deltas.png)",
        "",
        "## 数据文件",
        "",
        f"- `../data/{PREFIX}_metrics_points.csv`",
        f"- `../data/{PREFIX}_metrics_summary.csv`",
        f"- `../data/{PREFIX}_paired_deltas.csv`",
        f"- `../data/{PREFIX}_paired_delta_points.csv`",
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{PREFIX}_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    runs = discover_runs()
    all_points: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for model in MODELS:
            run = runs[(protocol, model)]
            for metric in METRICS:
                rows = load_metric(run, metric)
                all_points.extend(rows)
                summaries.append(summarize(run, metric, rows))

    delta_summary, delta_points = build_deltas(summaries, all_points)

    write_csv(
        DATA_DIR / f"{PREFIX}_metrics_points.csv",
        all_points,
        ["protocol", "model", "metric", "point_idx", "epoch", "step", "value", "run_dir"],
    )
    write_csv(
        DATA_DIR / f"{PREFIX}_metrics_summary.csv",
        summaries,
        [
            "protocol",
            "model",
            "metric",
            "final",
            "final_epoch",
            "final_step",
            "best",
            "best_epoch",
            "best_step",
            "best_final_gap",
            "mean_all",
            "std_all",
            "late_mean_40_60",
            "late_std_40_60",
            "train_dataloader_length",
            "seed",
            "cfg",
            "run_dir",
        ],
    )
    write_csv(
        DATA_DIR / f"{PREFIX}_paired_deltas.csv",
        delta_summary,
        [
            "protocol",
            "metric",
            "final_delta_a2_minus_a1",
            "best_delta_a2_minus_a1",
            "late_mean_delta_a2_minus_a1",
            "mean_all_delta_a2_minus_a1",
        ],
    )
    write_csv(
        DATA_DIR / f"{PREFIX}_paired_delta_points.csv",
        delta_points,
        ["protocol", "metric", "epoch", "a1_value", "a2_value", "delta_a2_minus_a1"],
    )

    plot_curves(all_points)
    plot_summary(summaries)
    plot_deltas(delta_summary)
    write_report(summaries, delta_summary)

    print(f"Wrote {REPORT_DIR / f'{PREFIX}_comparison.md'}")
    print(f"Wrote CSV summaries under {DATA_DIR}")
    print(f"Wrote PNG figures under {COMPARE_DIR / 'figures'}")


if __name__ == "__main__":
    main()
