#!/usr/bin/env python3
"""Build related CT-SeqTrack comparison tables and SVG charts.

This script reads the already exported CSV files under compare_results/data.
It does not parse TensorBoard again. Each comparison group includes a
baseline row so the final/best deltas are recomputed within the group.
"""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "compare_results"
DATA_DIR = OUT / "data"
REPORT_DIR = OUT / "reports"
BAR_DIR = OUT / "figures" / "bar_charts"
DELTA_DIR = OUT / "figures" / "delta_charts"
LINE_DIR = OUT / "figures" / "line_charts"

PREFIX = "related_comparisons"
METRICS = ["success/test", "precision/test"]
METRIC_SHORT = {"success/test": "success", "precision/test": "precision"}
METRIC_COLOR = {"success/test": "#2f5f9f", "precision/test": "#c85a3a"}

SOURCES = {
    "main": ("main_ablation_metrics_summary.csv", "main_ablation_metrics_points.csv"),
    "a1_time": ("a1_time_encoding_metrics_summary.csv", "a1_time_encoding_metrics_points.csv"),
    "scaled": ("scaled_a1_a2_baseline_metrics_summary.csv", "scaled_a1_a2_baseline_metrics_points.csv"),
    "order": ("order_pseudo_baseline_metrics_summary.csv", "order_pseudo_baseline_metrics_points.csv"),
    "cand_disp": ("cand1_disp_dynamics_metrics_summary.csv", "cand1_disp_dynamics_metrics_points.csv"),
    "twc_gate": ("twc_gate_ablation_metrics_summary.csv", "twc_gate_ablation_metrics_points.csv"),
    "active_twc": ("active_twc_vs_baseline_metrics_summary.csv", "active_twc_vs_baseline_metrics_points.csv"),
    "gate": ("gate_variants_vs_baseline_metrics_summary.csv", "gate_variants_vs_baseline_metrics_points.csv"),
    "latest": ("latest_5runs_metrics_summary.csv", "latest_5runs_metrics_points.csv"),
    "long180": ("baseline_a2_a3_180ep_metrics_summary.csv", "baseline_a2_a3_180ep_metrics_points.csv"),
}


GROUPS = [
    {
        "key": "main_a1_a2_p5_60ep",
        "name": "Main A1/A2/P5 Progression (60ep)",
        "protocol": "60ep seed42 nuScenes-mini",
        "note": "Shows the raw real-time path, dynamics recovery, and old full P5 collapse.",
        "items": [
            ("main", "SeqTrack baseline", "SeqTrack baseline", "baseline", "baseline"),
            ("main", "A1 CT-base", "A1 raw real-time", "variant", "raw real timestamp main branch"),
            ("main", "A2 Dynamics", "A2 raw-dyn", "variant", "raw real-time + dynamics"),
            ("main", "CT P5 full", "P5 full", "variant", "raw real-time + dynamics + gate"),
        ],
    },
    {
        "key": "a1_time_variants_60ep",
        "name": "A1 Time Encoding / Main-Branch Variants (60ep)",
        "protocol": "60ep seed42 nuScenes-mini",
        "note": "Collects A1 variants that modify the main time token semantics.",
        "items": [
            ("a1_time", "SeqTrack baseline", "SeqTrack baseline", "baseline", "baseline"),
            ("a1_time", "A1-raw", "A1 raw", "variant", "real seconds in main branch"),
            ("a1_time", "A1-pseudo", "A1 pseudo", "variant", "pseudo time sanity check"),
            ("a1_time", "A1-MLP", "A1 MLP", "variant", "scalar-preserving MLP time encoding"),
            ("a1_time", "A1-Fourier", "A1 Fourier", "variant", "scalar-preserving Fourier time encoding"),
            ("scaled", "A1-scaled", "A1 scaled", "variant", "real time rescaled near pseudo range"),
            ("order", "A1-order", "A1 order", "variant", "restore SeqTrack order-time semantics"),
            ("twc_gate", "A1-order+TWC", "A1 order+TWC", "variant", "active TWC on A1-order"),
        ],
    },
    {
        "key": "a2_dynamics_variants_60ep",
        "name": "A2 Dynamics Variants (60ep)",
        "protocol": "60ep nuScenes-mini; cand1 has fewer optimizer steps",
        "note": "Compares dynamics injection choices and diagnostics against the same baseline.",
        "items": [
            ("main", "SeqTrack baseline", "SeqTrack baseline", "baseline", "baseline"),
            ("main", "A2 Dynamics", "A2 raw-dyn", "variant", "raw real-time + dynamics"),
            ("scaled", "A2-scaled-dyn", "A2 scaled-dyn", "variant", "scaled real time + dynamics"),
            ("order", "A2-order-dyn", "A2 order-dyn seed42", "variant", "order main branch + dynamics"),
            ("cand_disp", "A2-order-dyn-cand1", "A2 cand1", "variant", "num_candidates=1, not step-aligned"),
            ("cand_disp", "A2-order-dyn-disp", "A2 dyn+disp", "variant", "small dynamics displacement loss"),
            ("twc_gate", "A2-order-dyn+TWC", "A2 dyn+TWC .05", "variant", "active TWC weight 0.05"),
            ("latest", "A2-order-dyn+TWC w0.01 seed42", "A2 dyn+TWC .01", "variant", "active TWC weight 0.01"),
        ],
    },
    {
        "key": "twc_related_60ep",
        "name": "TWC-Related Runs (60ep)",
        "protocol": "60ep nuScenes-mini; active TWC validity fixed",
        "note": "Separates TWC on A1 from the unstable A2+dynamics combination.",
        "items": [
            ("twc_gate", "SeqTrack baseline", "SeqTrack baseline", "baseline", "baseline"),
            ("twc_gate", "A1-order", "A1 order", "variant", "parent for A1+TWC"),
            ("twc_gate", "A1-order+TWC", "A1 order+TWC", "variant", "active TWC"),
            ("twc_gate", "A2-order-dyn", "A2 order-dyn", "variant", "parent for A2+TWC"),
            ("twc_gate", "A2-order-dyn+TWC", "A2 dyn+TWC .05", "variant", "active TWC weight 0.05"),
            ("latest", "A2-order-dyn+TWC w0.01 seed42", "A2 dyn+TWC .01", "variant", "active TWC weight 0.01"),
        ],
    },
    {
        "key": "a3_gate_variants_60ep",
        "name": "A3 / Gate Variants (60ep)",
        "protocol": "60ep nuScenes-mini plus latest retests",
        "note": "Compares gate variants and latest conf-res retests to baseline and A2.",
        "items": [
            ("main", "SeqTrack baseline", "SeqTrack baseline", "baseline", "baseline"),
            ("order", "A2-order-dyn", "A2 order-dyn seed42", "variant", "cleaner A2 parent"),
            ("main", "CT P5 full", "P5 full", "variant", "old full model with raw real-time path"),
            ("gate", "A3-order-gate-safe", "A3 gate-safe", "variant", "observation-biased feature gate"),
            ("gate", "A3-order-conf-res-gate", "A3 conf-res old", "variant", "old run; high best not reproduced"),
            ("latest", "A3-conf-res best-e14 retest", "A3 best-e14 retest", "variant", "single checkpoint retest"),
            ("latest", "A3-conf-res rerun seed42", "A3 conf-res rerun", "variant", "latest seed42 rerun"),
        ],
    },
    {
        "key": "a2_seed_stability_60ep",
        "name": "A2 Seed Stability (60ep)",
        "protocol": "60ep nuScenes-mini",
        "note": "Puts seed42, seed43, and seed44 against the same SeqTrack baseline.",
        "items": [
            ("order", "SeqTrack baseline", "SeqTrack baseline", "baseline", "baseline"),
            ("order", "A2-order-dyn", "A2 seed42", "variant", "old positive seed42 signal"),
            ("latest", "A2-order-dyn seed43", "A2 seed43", "variant", "latest seed43 collapse"),
            ("latest", "A2-order-dyn seed44", "A2 seed44", "variant", "latest seed44 partial recovery"),
        ],
    },
    {
        "key": "long_training_180ep",
        "name": "Long Training Stability (180ep)",
        "protocol": "180ep nuScenes-mini",
        "note": "Keeps the 180ep stability evidence separate from 60ep ablations.",
        "items": [
            ("long180", "SeqTrack baseline 180ep", "Baseline 180ep", "baseline", "baseline"),
            ("long180", "CT-SeqTrack A2-order-dyn 180ep", "A2 180ep", "variant", "A2 long training"),
            ("long180", "CT-SeqTrack A3-conf-res-gate 180ep", "A3 conf-res 180ep", "variant", "A3 long training"),
        ],
    },
]

PALETTE = [
    "#2f5f9f", "#c85a3a", "#3d8b62", "#7a5195", "#d9862c", "#4d7c8a",
    "#b24c63", "#6b6ecf", "#8c6d31", "#637939", "#843c39", "#17becf",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 2) -> str:
    v = fnum(value)
    if math.isnan(v):
        return ""
    return f"{v:.{digits}f}"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def load_sources():
    summaries = {}
    points = {}
    for key, (summary_file, points_file) in SOURCES.items():
        summary_path = DATA_DIR / summary_file
        points_path = DATA_DIR / points_file
        for row in read_csv(summary_path):
            summaries[(key, row["model"], row["metric"])] = row
        for row in read_csv(points_path):
            points.setdefault((key, row["model"], row["metric"]), []).append(row)
    return summaries, points


def normalize_summary_row(
    group: dict[str, object],
    item_index: int,
    source: str,
    model: str,
    display: str,
    role: str,
    note: str,
    metric: str,
    row: dict[str, str],
    baseline: dict[str, float],
) -> dict[str, object]:
    final = fnum(row.get("final"))
    best = fnum(row.get("best"))
    baseline_final = baseline[metric + "_final"]
    baseline_best = baseline[metric + "_best"]
    return {
        "group_key": group["key"],
        "group_name": group["name"],
        "protocol": group["protocol"],
        "item_order": item_index,
        "source": source,
        "source_model": model,
        "display_model": display,
        "role": role,
        "metric": metric,
        "final": final,
        "final_epoch": row.get("final_epoch", ""),
        "final_step": row.get("final_step", ""),
        "best": best,
        "best_epoch": row.get("best_epoch", ""),
        "best_step": row.get("best_step", ""),
        "best_final_gap": best - final,
        "mean": fnum(row.get("mean")),
        "std": fnum(row.get("std")),
        "baseline_final": baseline_final,
        "baseline_best": baseline_best,
        "final_delta_vs_group_baseline": final - baseline_final,
        "best_delta_vs_group_baseline": best - baseline_best,
        "note": note,
    }


def build_related_rows(summaries, point_lookup):
    summary_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for group in GROUPS:
        items = group["items"]  # type: ignore[index]
        baseline_item = next((item for item in items if item[3] == "baseline"), items[0])
        baseline = {}
        for metric in METRICS:
            key = (baseline_item[0], baseline_item[1], metric)
            row = summaries.get(key)
            if row is None:
                warnings.append(f"Missing baseline summary: {key}")
                baseline[metric + "_final"] = float("nan")
                baseline[metric + "_best"] = float("nan")
            else:
                baseline[metric + "_final"] = fnum(row.get("final"))
                baseline[metric + "_best"] = fnum(row.get("best"))

        for idx, item in enumerate(items):
            source, model, display, role, note = item
            for metric in METRICS:
                row = summaries.get((source, model, metric))
                if row is None:
                    warnings.append(f"Missing summary: {group['key']} {source}/{model}/{metric}")
                    continue
                summary_rows.append(
                    normalize_summary_row(group, idx, source, model, display, role, note, metric, row, baseline)
                )
                for point in point_lookup.get((source, model, metric), []):
                    point_rows.append({
                        "group_key": group["key"],
                        "group_name": group["name"],
                        "protocol": group["protocol"],
                        "item_order": idx,
                        "source": source,
                        "source_model": model,
                        "display_model": display,
                        "role": role,
                        "metric": metric,
                        "epoch": point.get("epoch", ""),
                        "step": point.get("step", ""),
                        "value": fnum(point.get("value")),
                    })
    return summary_rows, point_rows, warnings


def group_rows(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    return [r for r in rows if r["group_key"] == group_key]


def by_metric(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    return [r for r in rows if r["metric"] == metric]


def model_order(rows: list[dict[str, object]]) -> list[str]:
    pairs = {}
    for row in rows:
        pairs[str(row["display_model"])] = int(row["item_order"])
    return [name for name, _ in sorted(pairs.items(), key=lambda kv: kv[1])]


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def axis_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    if math.isnan(vmin) or math.isnan(vmax):
        return [0.0]
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    raw_step = (vmax - vmin) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(abs(raw_step))) if raw_step else 1
    norm = raw_step / mag
    if norm <= 1:
        step = mag
    elif norm <= 2:
        step = 2 * mag
    elif norm <= 5:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def write_score_chart(path: Path, group: dict[str, object], rows: list[dict[str, object]]) -> None:
    models = model_order(rows)
    pivots = {}
    for row in rows:
        pivots.setdefault(row["display_model"], {})[row["metric"]] = row
    max_value = max(fnum(row["final"]) for row in rows if not math.isnan(fnum(row["final"])))
    x_max = max(70.0, math.ceil((max_value + 5) / 10) * 10)
    width = 1120
    left = 230
    right = 40
    top = 82
    row_h = 42
    bottom = 60
    height = top + bottom + row_h * len(models)
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + (value / x_max) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="22" font-family="Arial" font-weight="700">{esc(group["name"])}</text>',
        f'<text x="24" y="58" font-size="13" font-family="Arial" fill="#555">{esc(group["protocol"])} - final scores, baseline included</text>',
    ]
    for tick in axis_ticks(0, x_max, 8):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top-8}" x2="{tx:.1f}" y2="{height-bottom+8}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height-24}" font-size="11" font-family="Arial" text-anchor="middle" fill="#666">{tick:.0f}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom+8}" x2="{width-right}" y2="{height-bottom+8}" stroke="#999"/>')

    baseline_by_metric = {}
    for row in rows:
        if row["role"] == "baseline":
            baseline_by_metric[row["metric"]] = fnum(row["final"])
    for metric, value in baseline_by_metric.items():
        bx = x(value)
        color = METRIC_COLOR[metric]
        parts.append(f'<line x1="{bx:.1f}" y1="{top-8}" x2="{bx:.1f}" y2="{height-bottom+8}" stroke="{color}" stroke-dasharray="5 4" stroke-width="1.4"/>')
    parts.append(f'<rect x="{left}" y="64" width="12" height="12" fill="{METRIC_COLOR["success/test"]}"/><text x="{left+18}" y="75" font-size="12" font-family="Arial">success final</text>')
    parts.append(f'<rect x="{left+140}" y="64" width="12" height="12" fill="{METRIC_COLOR["precision/test"]}"/><text x="{left+158}" y="75" font-size="12" font-family="Arial">precision final</text>')
    parts.append(f'<line x1="{left+300}" y1="70" x2="{left+328}" y2="70" stroke="#777" stroke-dasharray="5 4"/><text x="{left+336}" y="75" font-size="12" font-family="Arial">baseline line</text>')

    for i, model in enumerate(models):
        y0 = top + i * row_h
        parts.append(f'<text x="{left-12}" y="{y0+24}" font-size="12" font-family="Arial" text-anchor="end">{esc(model)}</text>')
        for j, metric in enumerate(METRICS):
            row = pivots.get(model, {}).get(metric)
            if not row:
                continue
            value = fnum(row["final"])
            bar_y = y0 + 7 + j * 15
            bar_h = 12
            parts.append(f'<rect x="{left}" y="{bar_y}" width="{max(0, x(value)-left):.1f}" height="{bar_h}" fill="{METRIC_COLOR[metric]}" opacity="0.88"/>')
            parts.append(f'<text x="{x(value)+5:.1f}" y="{bar_y+10}" font-size="11" font-family="Arial" fill="#333">{value:.2f}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_delta_chart(path: Path, group: dict[str, object], rows: list[dict[str, object]]) -> None:
    models = model_order(rows)
    pivots = {}
    deltas = []
    for row in rows:
        pivots.setdefault(row["display_model"], {})[row["metric"]] = row
        deltas.append(fnum(row["final_delta_vs_group_baseline"]))
    abs_max = max(5.0, max(abs(v) for v in deltas if not math.isnan(v)) + 2)
    x_min, x_max = -math.ceil(abs_max / 5) * 5, math.ceil(abs_max / 5) * 5
    width = 1120
    left = 230
    right = 40
    top = 82
    row_h = 42
    bottom = 60
    height = top + bottom + row_h * len(models)
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_w

    zero_x = x(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="22" font-family="Arial" font-weight="700">{esc(group["name"])}</text>',
        f'<text x="24" y="58" font-size="13" font-family="Arial" fill="#555">{esc(group["protocol"])} - final delta vs group baseline</text>',
    ]
    for tick in axis_ticks(x_min, x_max, 9):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top-8}" x2="{tx:.1f}" y2="{height-bottom+8}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height-24}" font-size="11" font-family="Arial" text-anchor="middle" fill="#666">{tick:.0f}</text>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top-8}" x2="{zero_x:.1f}" y2="{height-bottom+8}" stroke="#333" stroke-width="1.2"/>')
    parts.append(f'<rect x="{left}" y="64" width="12" height="12" fill="{METRIC_COLOR["success/test"]}"/><text x="{left+18}" y="75" font-size="12" font-family="Arial">success delta</text>')
    parts.append(f'<rect x="{left+140}" y="64" width="12" height="12" fill="{METRIC_COLOR["precision/test"]}"/><text x="{left+158}" y="75" font-size="12" font-family="Arial">precision delta</text>')

    for i, model in enumerate(models):
        y0 = top + i * row_h
        parts.append(f'<text x="{left-12}" y="{y0+24}" font-size="12" font-family="Arial" text-anchor="end">{esc(model)}</text>')
        for j, metric in enumerate(METRICS):
            row = pivots.get(model, {}).get(metric)
            if not row:
                continue
            value = fnum(row["final_delta_vs_group_baseline"])
            bar_y = y0 + 7 + j * 15
            x1, x2 = sorted((zero_x, x(value)))
            parts.append(f'<rect x="{x1:.1f}" y="{bar_y}" width="{max(1, x2-x1):.1f}" height="12" fill="{METRIC_COLOR[metric]}" opacity="0.88"/>')
            label_x = x(value) + (5 if value >= 0 else -5)
            anchor = "start" if value >= 0 else "end"
            parts.append(f'<text x="{label_x:.1f}" y="{bar_y+10}" font-size="11" font-family="Arial" text-anchor="{anchor}" fill="#333">{value:+.2f}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_line_chart(path: Path, group: dict[str, object], point_rows: list[dict[str, object]], metric: str) -> None:
    rows = [r for r in point_rows if r["metric"] == metric]
    if not rows:
        return
    models = model_order(rows)
    series = {}
    for row in rows:
        series.setdefault(row["display_model"], []).append(row)
    for values in series.values():
        values.sort(key=lambda r: (fnum(r["epoch"]), fnum(r["step"])))

    xs = [fnum(r["epoch"]) for r in rows]
    ys = [fnum(r["value"]) for r in rows]
    min_x, max_x = min(xs), max(xs)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    y_ticks = axis_ticks(min(ys) - 2, max(ys) + 2, 7)
    min_y, max_y = y_ticks[0], y_ticks[-1]
    width = 1120
    height = 600
    left = 70
    right = 260
    top = 82
    bottom = 74
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x(value: float) -> float:
        return left + ((value - min_x) / (max_x - min_x)) * plot_w

    def y(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_h

    metric_label = METRIC_SHORT[metric]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="22" font-family="Arial" font-weight="700">{esc(group["name"])}</text>',
        f'<text x="24" y="58" font-size="13" font-family="Arial" fill="#555">{esc(group["protocol"])} - {esc(metric_label)} curve</text>',
    ]
    for tick in y_ticks:
        ty = y(tick)
        parts.append(f'<line x1="{left}" y1="{ty:.1f}" x2="{width-right}" y2="{ty:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{left-10}" y="{ty+4:.1f}" font-size="11" font-family="Arial" text-anchor="end" fill="#666">{tick:.0f}</text>')
    for tick in axis_ticks(min_x, max_x, 7):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{height-bottom}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height-38}" font-size="11" font-family="Arial" text-anchor="middle" fill="#666">{tick:.0f}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#999"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#999"/>')
    parts.append(f'<text x="{(left + width - right)/2:.1f}" y="{height-12}" font-size="12" font-family="Arial" text-anchor="middle">epoch</text>')

    for idx, model in enumerate(models):
        color = color_for(idx)
        values = series[model]
        coords = [(x(fnum(v["epoch"])), y(fnum(v["value"]))) for v in values]
        if len(coords) > 1:
            path_d = " ".join(f"{px:.1f},{py:.1f}" for px, py in coords)
            parts.append(f'<polyline points="{path_d}" fill="none" stroke="{color}" stroke-width="2.1"/>')
        for px, py in coords:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}" opacity="0.9"/>')
        legend_y = top + idx * 22
        if legend_y < height - bottom + 16:
            parts.append(f'<line x1="{width-right+24}" y1="{legend_y}" x2="{width-right+48}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>')
            parts.append(f'<text x="{width-right+56}" y="{legend_y+4}" font-size="12" font-family="Arial">{esc(model)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def summary_lookup(rows: list[dict[str, object]], group_key: str):
    out = {}
    for row in group_rows(rows, group_key):
        out.setdefault(row["display_model"], {})[row["metric"]] = row
    return out


def report_table(rows: list[dict[str, object]], group_key: str) -> list[str]:
    pivot = summary_lookup(rows, group_key)
    ordered = model_order(group_rows(rows, group_key))
    lines = [
        "| model | succ final | succ delta | succ best | prec final | prec delta | prec best | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in ordered:
        s = pivot.get(model, {}).get("success/test", {})
        p = pivot.get(model, {}).get("precision/test", {})
        note = s.get("note") or p.get("note") or ""
        lines.append(
            "| "
            + " | ".join([
                str(model),
                fmt(s.get("final")),
                fmt(s.get("final_delta_vs_group_baseline")),
                fmt(s.get("best")),
                fmt(p.get("final")),
                fmt(p.get("final_delta_vs_group_baseline")),
                fmt(p.get("best")),
                str(note),
            ])
            + " |"
        )
    return lines


def rel(path: Path) -> str:
    return str(path.relative_to(OUT)).replace("\\", "/")


def write_report(summary_rows: list[dict[str, object]], warnings: list[str], generated: dict[str, dict[str, Path]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{PREFIX}.md"
    lines = [
        "# Related CT-SeqTrack Comparisons",
        "",
        "This report is generated from existing CSV exports under `compare_results/data`.",
        "Each group recomputes deltas against the baseline row included in that group.",
        "",
        "## Generated data",
        "",
        f"- `../data/{PREFIX}_metrics_summary.csv`",
        f"- `../data/{PREFIX}_metrics_points.csv`",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")

    for group in GROUPS:
        key = group["key"]
        files = generated[key]
        lines.extend([
            f"## {group['name']}",
            "",
            f"Protocol: {group['protocol']}",
            "",
            str(group["note"]),
            "",
            f"![final scores](../{rel(files['score'])})",
            "",
            f"![final delta](../{rel(files['delta'])})",
            "",
            f"![success curve](../{rel(files['success_line'])})",
            "",
            f"![precision curve](../{rel(files['precision_line'])})",
            "",
        ])
        lines.extend(report_table(summary_rows, key))
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summaries, points = load_sources()
    summary_rows, point_rows, warnings = build_related_rows(summaries, points)

    summary_path = DATA_DIR / f"{PREFIX}_metrics_summary.csv"
    points_path = DATA_DIR / f"{PREFIX}_metrics_points.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "group_key", "group_name", "protocol", "item_order", "source", "source_model",
            "display_model", "role", "metric", "final", "final_epoch", "final_step",
            "best", "best_epoch", "best_step", "best_final_gap", "mean", "std",
            "baseline_final", "baseline_best", "final_delta_vs_group_baseline",
            "best_delta_vs_group_baseline", "note",
        ],
    )
    write_csv(
        points_path,
        point_rows,
        [
            "group_key", "group_name", "protocol", "item_order", "source", "source_model",
            "display_model", "role", "metric", "epoch", "step", "value",
        ],
    )

    generated = {}
    for group in GROUPS:
        key = group["key"]
        rows = group_rows(summary_rows, key)
        prows = group_rows(point_rows, key)
        score = BAR_DIR / f"{key}_final_scores.svg"
        delta = DELTA_DIR / f"{key}_final_delta_vs_baseline.svg"
        success_line = LINE_DIR / f"{key}_success_curve.svg"
        precision_line = LINE_DIR / f"{key}_precision_curve.svg"
        write_score_chart(score, group, rows)
        write_delta_chart(delta, group, rows)
        write_line_chart(success_line, group, prows, "success/test")
        write_line_chart(precision_line, group, prows, "precision/test")
        generated[key] = {
            "score": score,
            "delta": delta,
            "success_line": success_line,
            "precision_line": precision_line,
        }
    write_report(summary_rows, warnings, generated)

    print(f"Wrote {summary_path}")
    print(f"Wrote {points_path}")
    print(f"Wrote {REPORT_DIR / (PREFIX + '.md')}")
    print(f"Wrote related comparison figures under {OUT / 'figures'}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
