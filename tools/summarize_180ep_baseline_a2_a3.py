#!/usr/bin/env python3
"""Summarize the latest 180-epoch baseline/A2/A3 TensorBoard metrics.

This script intentionally uses only the Python standard library because the
local analysis environment may not have tensorboard, pandas, or matplotlib.
It reads TensorBoard event files as TFRecord streams and extracts scalar
Summary.Value.simple_value entries.
"""

from __future__ import annotations

import csv
import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "compare_results"
DATA_DIR = OUT / "data"
REPORT_DIR = OUT / "reports"
LINE_DIR = OUT / "figures" / "line_charts"
BAR_DIR = OUT / "figures" / "bar_charts"

PREFIX = "baseline_a2_a3_180ep"
LATE_START_EPOCH = 120

RUNS = [
    {
        "model": "SeqTrack baseline 180ep",
        "short": "Baseline",
        "version": WORKSPACE
        / "seqtrack/output/20260702-0038-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_180ep_bs16_gpu3/lightning_logs/version_0",
    },
    {
        "model": "CT-SeqTrack A2-order-dyn 180ep",
        "short": "A2-order-dyn",
        "version": ROOT
        / "output/20260702-0038-seqtrack3d_nuscenes_a2_order_dyn-ct_a2_order_dyn_car_180ep_bs16_gpu3/lightning_logs/version_0",
    },
    {
        "model": "CT-SeqTrack A3-conf-res-gate 180ep",
        "short": "A3-conf-res",
        "version": ROOT
        / "output/20260702-0043-seqtrack3d_nuscenes_a3_order_conf_res_gate-ct_a3_order_conf_res_gate_car_180ep_bs16_gpu3/lightning_logs/version_0",
    },
]

METRICS = [
    ("success/test", "metrics_test_success", "metrics/test"),
    ("precision/test", "metrics_test_precision", "metrics/test"),
]

COLORS = {
    "SeqTrack baseline 180ep": "#2f5f9f",
    "CT-SeqTrack A2-order-dyn 180ep": "#c85a3a",
    "CT-SeqTrack A3-conf-res-gate 180ep": "#3d8b62",
}


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7


def skip_field(buf: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:
        _, pos = read_varint(buf, pos)
        return pos
    if wire_type == 1:
        return pos + 8
    if wire_type == 2:
        n, pos = read_varint(buf, pos)
        return pos + n
    if wire_type == 5:
        return pos + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def parse_value(buf: bytes) -> tuple[str | None, float | None]:
    pos = 0
    tag = None
    simple_value = None
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_num, wire_type = key >> 3, key & 7
        if field_num == 1 and wire_type == 2:
            n, pos = read_varint(buf, pos)
            tag = buf[pos : pos + n].decode("utf-8", errors="replace")
            pos += n
        elif field_num == 2 and wire_type == 5:
            simple_value = struct.unpack("<f", buf[pos : pos + 4])[0]
            pos += 4
        else:
            pos = skip_field(buf, pos, wire_type)
    return tag, simple_value


def parse_summary(buf: bytes) -> list[tuple[str, float]]:
    pos = 0
    values = []
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_num, wire_type = key >> 3, key & 7
        if field_num == 1 and wire_type == 2:
            n, pos = read_varint(buf, pos)
            tag, simple_value = parse_value(buf[pos : pos + n])
            pos += n
            if tag is not None and simple_value is not None:
                values.append((tag, simple_value))
        else:
            pos = skip_field(buf, pos, wire_type)
    return values


def parse_event(buf: bytes) -> tuple[int | None, list[tuple[str, float]]]:
    pos = 0
    step = None
    scalars = []
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_num, wire_type = key >> 3, key & 7
        if field_num == 2 and wire_type == 0:
            step, pos = read_varint(buf, pos)
        elif field_num in (3, 5) and wire_type == 2:
            n, pos = read_varint(buf, pos)
            scalars.extend(parse_summary(buf[pos : pos + n]))
            pos += n
        else:
            pos = skip_field(buf, pos, wire_type)
    return step, scalars


def read_scalar_events(event_file: Path, wanted_tag: str) -> list[tuple[int, float]]:
    points = []
    with event_file.open("rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError(f"Truncated TFRecord header: {event_file}")
            length = struct.unpack("<Q", header)[0]
            f.read(4)  # length CRC
            payload = f.read(length)
            f.read(4)  # payload CRC
            if len(payload) != length:
                raise ValueError(f"Truncated TFRecord payload: {event_file}")
            step, scalars = parse_event(payload)
            if step is None:
                continue
            for tag, value in scalars:
                if tag == wanted_tag:
                    points.append((step, value))
    return points


def metric_points(version_dir: Path, output_metric: str, metric_dir: str, event_tag: str) -> list[dict[str, object]]:
    event_files = sorted((version_dir / metric_dir).glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {version_dir / metric_dir}")
    rows = []
    for event_file in event_files:
        for step, value in read_scalar_events(event_file, event_tag):
            epoch = round(step / 1262)
            rows.append({"epoch": epoch, "step": step, "value": value})
    rows.sort(key=lambda r: (int(r["step"]), int(r["epoch"])))
    # Keep last value per step in case an event file contains duplicates.
    dedup = {}
    for row in rows:
        dedup[int(row["step"])] = row
    return [dedup[k] for k in sorted(dedup)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def summarize(model: str, metric: str, rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(r["value"]) for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: float(rows[i]["value"]))
    final = rows[-1]
    best = rows[best_idx]
    late_values = [float(r["value"]) for r in rows if int(r["epoch"]) >= LATE_START_EPOCH]
    return {
        "model": model,
        "metric": metric,
        "final": float(final["value"]),
        "final_epoch": int(final["epoch"]),
        "final_step": int(final["step"]),
        "best": float(best["value"]),
        "best_epoch": int(best["epoch"]),
        "best_step": int(best["step"]),
        "best_final_gap": float(best["value"]) - float(final["value"]),
        "mean": mean(values),
        "std": std(values),
        f"late_mean_epoch{LATE_START_EPOCH}_180": mean(late_values),
        f"late_std_epoch{LATE_START_EPOCH}_180": std(late_values),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def points_to_xy(rows: list[dict[str, object]], metric: str, width: int, height: int, pad: int):
    metric_rows = [r for r in rows if r["metric"] == metric]
    epochs = [int(r["epoch"]) for r in metric_rows]
    values = [float(r["value"]) for r in metric_rows]
    min_x, max_x = min(epochs), max(epochs)
    min_y = math.floor((min(values) - 2) / 5) * 5
    max_y = math.ceil((max(values) + 2) / 5) * 5
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    def sx(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / (max_y - min_y) * plot_h

    return metric_rows, sx, sy, min_x, max_x, min_y, max_y


def svg_line_chart(path: Path, rows: list[dict[str, object]], metric: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, pad = 920, 480, 64
    metric_rows, sx, sy, min_x, max_x, min_y, max_y = points_to_xy(rows, metric, width, height, pad)
    models = list(dict.fromkeys(r["model"] for r in metric_rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    for y in range(int(min_y), int(max_y) + 1, 5):
        yy = sy(y)
        parts.append(f'<line x1="{pad}" y1="{yy:.1f}" x2="{width-pad}" y2="{yy:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{pad-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y}</text>')
    for x in range(int(min_x), int(max_x) + 1, 20):
        xx = sx(x)
        parts.append(f'<line x1="{xx:.1f}" y1="{height-pad}" x2="{xx:.1f}" y2="{height-pad+5}" stroke="#333"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-pad+22}" text-anchor="middle" font-family="Arial" font-size="12">{x}</text>')
    parts.append(f'<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="13">Epoch</text>')
    parts.append(f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="13">{metric}</text>')

    for i, model in enumerate(models):
        model_rows = [r for r in metric_rows if r["model"] == model]
        pts = " ".join(f'{sx(int(r["epoch"])):.1f},{sy(float(r["value"])):.1f}' for r in model_rows)
        color = COLORS.get(model, "#444")
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r in model_rows:
            if int(r["epoch"]) in (5, 45, 90, 135, 180):
                parts.append(f'<circle cx="{sx(int(r["epoch"])):.1f}" cy="{sy(float(r["value"])):.1f}" r="3" fill="{color}"/>')
        lx, ly = width - 270, 58 + i * 24
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+34}" y="{ly+4}" font-family="Arial" font-size="13">{model}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_bar_chart(path: Path, summary_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 520
    pad_l, pad_b, pad_t = 72, 90, 56
    metrics = ["success/test", "precision/test"]
    models = list(dict.fromkeys(r["model"] for r in summary_rows))
    max_y = math.ceil(max(float(r["best"]) for r in summary_rows) / 5) * 5 + 5
    plot_h = height - pad_t - pad_b
    plot_w = width - pad_l - 40

    def sy(y: float) -> float:
        return height - pad_b - y / max_y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">180ep Best / Final Summary</text>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-40}" y2="{height-pad_b}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#333"/>',
    ]
    for y in range(0, int(max_y) + 1, 10):
        yy = sy(y)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-40}" y2="{yy:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{pad_l-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y}</text>')
    group_w = plot_w / len(metrics)
    bar_w = 18
    for mi, metric in enumerate(metrics):
        group_x = pad_l + mi * group_w + group_w / 2
        parts.append(f'<text x="{group_x}" y="{height-34}" text-anchor="middle" font-family="Arial" font-size="13">{metric}</text>')
        for si, stat in enumerate(["final", "best"]):
            for j, model in enumerate(models):
                row = next(r for r in summary_rows if r["metric"] == metric and r["model"] == model)
                value = float(row[stat])
                x = group_x - 105 + si * 115 + j * (bar_w + 4)
                y = sy(value)
                color = COLORS.get(model, "#444")
                opacity = "1.0" if stat == "final" else "0.55"
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{height-pad_b-y:.1f}" fill="{color}" opacity="{opacity}"/>')
                parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.1f}</text>')
            parts.append(f'<text x="{group_x - 95 + si * 115}" y="{height-58}" text-anchor="start" font-family="Arial" font-size="11">{stat}</text>')
    for i, model in enumerate(models):
        lx, ly = 650, 62 + i * 24
        color = COLORS.get(model, "#444")
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="18" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx+26}" y="{ly}" font-family="Arial" font-size="13">{model}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(summary_rows: list[dict[str, object]]) -> None:
    report_path = REPORT_DIR / f"{PREFIX}_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline vs A2 vs A3 180ep",
        "",
        "## Summary",
        "",
        "| model | metric | final | best | best epoch | late mean 120-180 | final delta vs baseline | best delta vs baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {metric} | {final} | {best} | {best_epoch} | {late} | {fd} | {bd} |".format(
                model=row["model"],
                metric=row["metric"],
                final=fmt(row["final"], 2),
                best=fmt(row["best"], 2),
                best_epoch=row["best_epoch"],
                late=fmt(row[f"late_mean_epoch{LATE_START_EPOCH}_180"], 2),
                fd=fmt(row["final_delta_vs_baseline_180ep"], 2),
                bd=fmt(row["best_delta_vs_baseline_180ep"], 2),
            )
        )
    lines += [
        "",
        "## Figures",
        "",
        f"![curves](../figures/line_charts/{PREFIX}_curves.svg)",
        "",
        f"![success](../figures/line_charts/{PREFIX}_success_curve.svg)",
        "",
        f"![precision](../figures/line_charts/{PREFIX}_precision_curve.svg)",
        "",
        f"![best final](../figures/bar_charts/{PREFIX}_best_final_summary.svg)",
        "",
        "## Files",
        "",
        f"- `../data/{PREFIX}_metrics_points.csv`",
        f"- `../data/{PREFIX}_metrics_summary.csv`",
        f"- `../figures/line_charts/{PREFIX}_curves.svg`",
        f"- `../figures/line_charts/{PREFIX}_success_curve.svg`",
        f"- `../figures/line_charts/{PREFIX}_precision_curve.svg`",
        f"- `../figures/bar_charts/{PREFIX}_best_final_summary.svg`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def svg_combined(path: Path, rows: list[dict[str, object]]) -> None:
    success_path = LINE_DIR / f"{PREFIX}_success_curve.svg"
    precision_path = LINE_DIR / f"{PREFIX}_precision_curve.svg"
    svg_line_chart(success_path, rows, "success/test", "180ep Success")
    svg_line_chart(precision_path, rows, "precision/test", "180ep Precision")
    success = success_path.read_text(encoding="utf-8")
    precision = precision_path.read_text(encoding="utf-8")
    # SVG nesting keeps the two hand-drawn charts intact.
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="960" viewBox="0 0 920 960">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f'<g transform="translate(0 0)">{success}</g>\n'
        f'<g transform="translate(0 480)">{precision}</g>\n'
        "</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    all_points = []
    summaries = []
    for run in RUNS:
        for metric_tag, metric_dir, event_tag in METRICS:
            rows = metric_points(run["version"], metric_tag, metric_dir, event_tag)
            for idx, row in enumerate(rows):
                all_points.append(
                    {
                        "model": run["model"],
                        "metric": metric_tag,
                        "point_idx": idx,
                        "epoch": row["epoch"],
                        "step": row["step"],
                        "value": row["value"],
                    }
                )
            summaries.append(summarize(run["model"], metric_tag, rows))

    baseline_by_metric = {
        r["metric"]: r for r in summaries if r["model"] == "SeqTrack baseline 180ep"
    }
    for row in summaries:
        base = baseline_by_metric[row["metric"]]
        row["final_delta_vs_baseline_180ep"] = float(row["final"]) - float(base["final"])
        row["best_delta_vs_baseline_180ep"] = float(row["best"]) - float(base["best"])
        row["late_mean_delta_vs_baseline_180ep"] = (
            float(row[f"late_mean_epoch{LATE_START_EPOCH}_180"])
            - float(base[f"late_mean_epoch{LATE_START_EPOCH}_180"])
        )

    point_fields = ["model", "metric", "point_idx", "epoch", "step", "value"]
    summary_fields = [
        "model",
        "metric",
        "final",
        "final_epoch",
        "final_step",
        "best",
        "best_epoch",
        "best_step",
        "best_final_gap",
        "mean",
        "std",
        f"late_mean_epoch{LATE_START_EPOCH}_180",
        f"late_std_epoch{LATE_START_EPOCH}_180",
        "final_delta_vs_baseline_180ep",
        "best_delta_vs_baseline_180ep",
        "late_mean_delta_vs_baseline_180ep",
    ]
    write_csv(DATA_DIR / f"{PREFIX}_metrics_points.csv", all_points, point_fields)
    write_csv(DATA_DIR / f"{PREFIX}_metrics_summary.csv", summaries, summary_fields)
    svg_line_chart(LINE_DIR / f"{PREFIX}_success_curve.svg", all_points, "success/test", "180ep Success")
    svg_line_chart(LINE_DIR / f"{PREFIX}_precision_curve.svg", all_points, "precision/test", "180ep Precision")
    svg_combined(LINE_DIR / f"{PREFIX}_curves.svg", all_points)
    svg_bar_chart(BAR_DIR / f"{PREFIX}_best_final_summary.svg", summaries)
    write_report(summaries)

    print(f"Wrote {DATA_DIR / f'{PREFIX}_metrics_summary.csv'}")
    print(f"Wrote {DATA_DIR / f'{PREFIX}_metrics_points.csv'}")
    print(f"Wrote {REPORT_DIR / f'{PREFIX}_comparison.md'}")
    print(f"Wrote SVG figures under {LINE_DIR} and {BAR_DIR}")


if __name__ == "__main__":
    main()
