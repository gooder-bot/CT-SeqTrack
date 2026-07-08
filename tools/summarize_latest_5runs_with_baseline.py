#!/usr/bin/env python3
"""Combine the latest five CT-SeqTrack runs with the 60ep SeqTrack baseline.

Inputs are existing CSV exports under compare_results/data. The baseline is
the 60-epoch SeqTrack baseline from twc_gate_ablation metrics, matching the
latest five runs' 60ep protocol better than the separate 180ep baseline.
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

PREFIX = "latest_5runs_with_baseline"
METRICS = ["success/test", "precision/test"]
METRIC_COLOR = {"success/test": "#2f5f9f", "precision/test": "#c85a3a"}
PALETTE = ["#2f5f9f", "#ef5675", "#2f4b7c", "#ffa600", "#3d8b62", "#7a5195"]

RUN_ORDER = [
    ("baseline_60ep", "SeqTrack baseline", "SeqTrack baseline", "60ep baseline"),
    ("a3_conf_res_best_e14_retest", "A3-conf-res best-e14 retest", "A3 best-e14 retest", "single checkpoint test"),
    ("a2_order_dyn_seed43", "A2-order-dyn seed43", "A2 seed43", "large seed regression"),
    ("a2_order_dyn_seed44", "A2-order-dyn seed44", "A2 seed44", "partial recovery"),
    ("a2_order_dyn_twc_w001_seed42", "A2-order-dyn+TWC w0.01 seed42", "A2+TWC .01 seed42", "low TWC weight still collapses"),
    ("a3_conf_res_rerun_seed42", "A3-conf-res rerun seed42", "A3 conf-res rerun", "rerun remains low"),
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
    value = fnum(value)
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def axis_ticks(vmin: float, vmax: float, count: int = 7) -> list[float]:
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    raw = (vmax - vmin) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(abs(raw))) if raw else 1
    norm = raw / mag
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


def load_combined_rows():
    baseline_summary = {
        (row["model"], row["metric"]): row
        for row in read_csv(DATA_DIR / "twc_gate_ablation_metrics_summary.csv")
        if row["model"] == "SeqTrack baseline"
    }
    baseline_points = [
        row for row in read_csv(DATA_DIR / "twc_gate_ablation_metrics_points.csv")
        if row["model"] == "SeqTrack baseline"
    ]
    latest_summary = {
        (row["run_key"], row["model"], row["metric"]): row
        for row in read_csv(DATA_DIR / "latest_5runs_metrics_summary.csv")
    }
    latest_points = read_csv(DATA_DIR / "latest_5runs_metrics_points.csv")

    baseline_values = {
        metric: {
            "final": fnum(baseline_summary[("SeqTrack baseline", metric)]["final"]),
            "best": fnum(baseline_summary[("SeqTrack baseline", metric)]["best"]),
        }
        for metric in METRICS
    }

    summary_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []

    for order, (run_key, source_model, display_model, note) in enumerate(RUN_ORDER):
        for metric in METRICS:
            if run_key == "baseline_60ep":
                src = baseline_summary[(source_model, metric)]
            else:
                src = latest_summary[(run_key, source_model, metric)]
            final = fnum(src.get("final"))
            best = fnum(src.get("best"))
            summary_rows.append({
                "run_order": order,
                "run_key": run_key,
                "source_model": source_model,
                "display_model": display_model,
                "metric": metric,
                "final": final,
                "final_epoch": src.get("final_epoch", ""),
                "final_step": src.get("final_step", ""),
                "best": best,
                "best_epoch": src.get("best_epoch", ""),
                "best_step": src.get("best_step", ""),
                "best_final_gap": best - final,
                "mean": fnum(src.get("mean")),
                "std": fnum(src.get("std")),
                "baseline_final": baseline_values[metric]["final"],
                "baseline_best": baseline_values[metric]["best"],
                "final_delta_vs_baseline": final - baseline_values[metric]["final"],
                "best_delta_vs_baseline": best - baseline_values[metric]["best"],
                "note": note,
            })

    for point in baseline_points:
        point_rows.append({
            "run_order": 0,
            "run_key": "baseline_60ep",
            "source_model": "SeqTrack baseline",
            "display_model": "SeqTrack baseline",
            "metric": point["metric"],
            "epoch": point.get("epoch", ""),
            "step": point.get("step", ""),
            "value": fnum(point.get("value")),
        })
    display_lookup = {source: (run_key, display) for run_key, source, display, _ in RUN_ORDER}
    order_lookup = {run_key: idx for idx, (run_key, _, _, _) in enumerate(RUN_ORDER)}
    for point in latest_points:
        run_key, display = display_lookup[point["model"]]
        point_rows.append({
            "run_order": order_lookup[run_key],
            "run_key": run_key,
            "source_model": point["model"],
            "display_model": display,
            "metric": point["metric"],
            "epoch": point.get("epoch", ""),
            "step": point.get("step", ""),
            "value": fnum(point.get("value")),
        })

    return summary_rows, point_rows


def ordered_models(rows: list[dict[str, object]]) -> list[str]:
    order = {}
    for row in rows:
        order[str(row["display_model"])] = int(row["run_order"])
    return [model for model, _ in sorted(order.items(), key=lambda item: item[1])]


def pivot_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row["display_model"]), {})[str(row["metric"])] = row
    return out


def write_score_chart(path: Path, rows: list[dict[str, object]]) -> None:
    models = ordered_models(rows)
    pivot = pivot_summary(rows)
    max_value = max(fnum(row["final"]) for row in rows)
    x_max = max(70.0, math.ceil((max_value + 5) / 10) * 10)
    width, left, right, top, row_h, bottom = 1120, 230, 40, 82, 42, 60
    height = top + bottom + row_h * len(models)
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + value / x_max * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-size="22" font-family="Arial" font-weight="700">Latest 5 Runs + SeqTrack Baseline</text>',
        '<text x="24" y="58" font-size="13" font-family="Arial" fill="#555">final scores; baseline included</text>',
    ]
    for tick in axis_ticks(0, x_max, 8):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top-8}" x2="{tx:.1f}" y2="{height-bottom+8}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height-24}" font-size="11" font-family="Arial" text-anchor="middle" fill="#666">{tick:.0f}</text>')
    for metric in METRICS:
        value = fnum(pivot["SeqTrack baseline"][metric]["final"])
        bx = x(value)
        parts.append(f'<line x1="{bx:.1f}" y1="{top-8}" x2="{bx:.1f}" y2="{height-bottom+8}" stroke="{METRIC_COLOR[metric]}" stroke-dasharray="5 4" stroke-width="1.4"/>')
    parts.append(f'<rect x="{left}" y="64" width="12" height="12" fill="{METRIC_COLOR["success/test"]}"/><text x="{left+18}" y="75" font-size="12" font-family="Arial">success final</text>')
    parts.append(f'<rect x="{left+140}" y="64" width="12" height="12" fill="{METRIC_COLOR["precision/test"]}"/><text x="{left+158}" y="75" font-size="12" font-family="Arial">precision final</text>')
    for i, model in enumerate(models):
        y0 = top + i * row_h
        parts.append(f'<text x="{left-12}" y="{y0+24}" font-size="12" font-family="Arial" text-anchor="end">{esc(model)}</text>')
        for j, metric in enumerate(METRICS):
            value = fnum(pivot[model][metric]["final"])
            y = y0 + 7 + j * 15
            parts.append(f'<rect x="{left}" y="{y}" width="{max(0, x(value)-left):.1f}" height="12" fill="{METRIC_COLOR[metric]}" opacity="0.9"/>')
            parts.append(f'<text x="{x(value)+5:.1f}" y="{y+10}" font-size="11" font-family="Arial">{value:.2f}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_delta_chart(path: Path, rows: list[dict[str, object]]) -> None:
    models = ordered_models(rows)
    pivot = pivot_summary(rows)
    deltas = [fnum(row["final_delta_vs_baseline"]) for row in rows]
    abs_max = max(5.0, max(abs(v) for v in deltas) + 2)
    x_min, x_max = -math.ceil(abs_max / 5) * 5, math.ceil(abs_max / 5) * 5
    width, left, right, top, row_h, bottom = 1120, 230, 40, 82, 42, 60
    height = top + bottom + row_h * len(models)
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    zero_x = x(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-size="22" font-family="Arial" font-weight="700">Latest 5 Runs + SeqTrack Baseline</text>',
        '<text x="24" y="58" font-size="13" font-family="Arial" fill="#555">final delta vs SeqTrack baseline</text>',
    ]
    for tick in axis_ticks(x_min, x_max, 9):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" y1="{top-8}" x2="{tx:.1f}" y2="{height-bottom+8}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height-24}" font-size="11" font-family="Arial" text-anchor="middle" fill="#666">{tick:.0f}</text>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top-8}" x2="{zero_x:.1f}" y2="{height-bottom+8}" stroke="#333"/>')
    parts.append(f'<rect x="{left}" y="64" width="12" height="12" fill="{METRIC_COLOR["success/test"]}"/><text x="{left+18}" y="75" font-size="12" font-family="Arial">success delta</text>')
    parts.append(f'<rect x="{left+140}" y="64" width="12" height="12" fill="{METRIC_COLOR["precision/test"]}"/><text x="{left+158}" y="75" font-size="12" font-family="Arial">precision delta</text>')
    for i, model in enumerate(models):
        y0 = top + i * row_h
        parts.append(f'<text x="{left-12}" y="{y0+24}" font-size="12" font-family="Arial" text-anchor="end">{esc(model)}</text>')
        for j, metric in enumerate(METRICS):
            value = fnum(pivot[model][metric]["final_delta_vs_baseline"])
            y = y0 + 7 + j * 15
            x1, x2 = sorted((zero_x, x(value)))
            parts.append(f'<rect x="{x1:.1f}" y="{y}" width="{max(1, x2-x1):.1f}" height="12" fill="{METRIC_COLOR[metric]}" opacity="0.9"/>')
            label_x = x(value) + (5 if value >= 0 else -5)
            anchor = "start" if value >= 0 else "end"
            parts.append(f'<text x="{label_x:.1f}" y="{y+10}" font-size="11" font-family="Arial" text-anchor="{anchor}">{value:+.2f}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_line_chart(path: Path, rows: list[dict[str, object]], metric: str) -> None:
    metric_rows = [row for row in rows if row["metric"] == metric]
    models = ordered_models(metric_rows)
    series: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        series.setdefault(str(row["display_model"]), []).append(row)
    for values in series.values():
        values.sort(key=lambda row: (fnum(row["epoch"]), fnum(row["step"])))

    xs = [fnum(row["epoch"]) for row in metric_rows]
    ys = [fnum(row["value"]) for row in metric_rows]
    min_x, max_x = min(xs), max(xs)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    y_ticks = axis_ticks(min(ys) - 2, max(ys) + 2, 7)
    min_y, max_y = y_ticks[0], y_ticks[-1]
    width, height, left, right, top, bottom = 1120, 600, 70, 260, 82, 74
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def y(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_h

    short = "success" if metric == "success/test" else "precision"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="34" font-size="22" font-family="Arial" font-weight="700">Latest 5 Runs + SeqTrack Baseline</text>',
        f'<text x="24" y="58" font-size="13" font-family="Arial" fill="#555">{short} curve</text>',
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
    for idx, model in enumerate(models):
        color = PALETTE[idx % len(PALETTE)]
        coords = [(x(fnum(row["epoch"])), y(fnum(row["value"]))) for row in series[model]]
        if len(coords) > 1:
            parts.append(f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in coords)}" fill="none" stroke="{color}" stroke-width="2.1"/>')
        for px, py in coords:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')
        legend_y = top + idx * 22
        parts.append(f'<line x1="{width-right+24}" y1="{legend_y}" x2="{width-right+48}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<text x="{width-right+56}" y="{legend_y+4}" font-size="12" font-family="Arial">{esc(model)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(summary_rows: list[dict[str, object]]) -> None:
    pivot = pivot_summary(summary_rows)
    models = ordered_models(summary_rows)
    lines = [
        "# Latest 5 Runs + SeqTrack Baseline",
        "",
        "This report combines the latest five local runs with the 60ep SeqTrack baseline.",
        "The baseline row is taken from `twc_gate_ablation_metrics_*` to match the 60ep protocol.",
        "",
        "## Figures",
        "",
        "![final scores](../figures/bar_charts/latest_5runs_with_baseline_final_scores.svg)",
        "",
        "![final delta](../figures/delta_charts/latest_5runs_with_baseline_final_delta_vs_baseline.svg)",
        "",
        "![success curve](../figures/line_charts/latest_5runs_with_baseline_success_curve.svg)",
        "",
        "![precision curve](../figures/line_charts/latest_5runs_with_baseline_precision_curve.svg)",
        "",
        "## Summary",
        "",
        "| model | succ final | succ delta | succ best | prec final | prec delta | prec best | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in models:
        s = pivot[model]["success/test"]
        p = pivot[model]["precision/test"]
        lines.append(
            "| "
            + " | ".join([
                model,
                fmt(s["final"]),
                fmt(s["final_delta_vs_baseline"]),
                fmt(s["best"]),
                fmt(p["final"]),
                fmt(p["final_delta_vs_baseline"]),
                fmt(p["best"]),
                str(s["note"]),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Generated files",
        "",
        f"- `../data/{PREFIX}_metrics_summary.csv`",
        f"- `../data/{PREFIX}_metrics_points.csv`",
        "- `../figures/bar_charts/latest_5runs_with_baseline_final_scores.svg`",
        "- `../figures/delta_charts/latest_5runs_with_baseline_final_delta_vs_baseline.svg`",
        "- `../figures/line_charts/latest_5runs_with_baseline_success_curve.svg`",
        "- `../figures/line_charts/latest_5runs_with_baseline_precision_curve.svg`",
    ])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{PREFIX}_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary_rows, point_rows = load_combined_rows()
    write_csv(
        DATA_DIR / f"{PREFIX}_metrics_summary.csv",
        summary_rows,
        [
            "run_order", "run_key", "source_model", "display_model", "metric",
            "final", "final_epoch", "final_step", "best", "best_epoch", "best_step",
            "best_final_gap", "mean", "std", "baseline_final", "baseline_best",
            "final_delta_vs_baseline", "best_delta_vs_baseline", "note",
        ],
    )
    write_csv(
        DATA_DIR / f"{PREFIX}_metrics_points.csv",
        point_rows,
        ["run_order", "run_key", "source_model", "display_model", "metric", "epoch", "step", "value"],
    )
    write_score_chart(BAR_DIR / "latest_5runs_with_baseline_final_scores.svg", summary_rows)
    write_delta_chart(DELTA_DIR / "latest_5runs_with_baseline_final_delta_vs_baseline.svg", summary_rows)
    write_line_chart(LINE_DIR / "latest_5runs_with_baseline_success_curve.svg", point_rows, "success/test")
    write_line_chart(LINE_DIR / "latest_5runs_with_baseline_precision_curve.svg", point_rows, "precision/test")
    write_report(summary_rows)

    print(f"Wrote {DATA_DIR / (PREFIX + '_metrics_summary.csv')}")
    print(f"Wrote {DATA_DIR / (PREFIX + '_metrics_points.csv')}")
    print(f"Wrote {REPORT_DIR / (PREFIX + '_comparison.md')}")
    print(f"Wrote figures under {OUT / 'figures'}")


if __name__ == "__main__":
    main()
