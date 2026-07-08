#!/usr/bin/env python3
"""Summarize the five newest CT-SeqTrack experiment outputs.

The script mirrors the lightweight style of the existing comparison scripts:
it parses TensorBoard scalar events directly, writes CSV tables, SVG charts,
and a short Markdown report under compare_results/.
"""

from __future__ import annotations

import csv
import math
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "compare_results"
DATA_DIR = OUT / "data"
REPORT_DIR = OUT / "reports"
LINE_DIR = OUT / "figures" / "line_charts"
BAR_DIR = OUT / "figures" / "bar_charts"
DIAG_DIR = OUT / "figures" / "diagnostics"

PREFIX = "latest_5runs"
STEPS_PER_EPOCH = 1262

RUNS = [
    {
        "key": "a3_conf_res_best_e14_retest",
        "model": "A3-conf-res best-e14 retest",
        "short": "A3 best retest",
        "type": "test",
        "version": ROOT / "output/20260707-1351-seqtrack3d_nuscenes_a3_order_conf_res_gate-retest_a3_conf_res_precision_best_e14/lightning_logs/version_0",
        "metrics": {
            "success/test": ("metrics_test_current_success", "metrics/test/current"),
            "precision/test": ("metrics_test_current_precision", "metrics/test/current"),
        },
    },
    {
        "key": "a2_order_dyn_seed43",
        "model": "A2-order-dyn seed43",
        "short": "A2 seed43",
        "type": "train",
        "version": ROOT / "output/20260707-1351-seqtrack3d_nuscenes_a2_order_dyn-ct_a2_order_dyn_car_60ep_bs16_seed43/lightning_logs/version_0",
    },
    {
        "key": "a2_order_dyn_seed44",
        "model": "A2-order-dyn seed44",
        "short": "A2 seed44",
        "type": "train",
        "version": ROOT / "output/20260707-1351-seqtrack3d_nuscenes_a2_order_dyn-ct_a2_order_dyn_car_60ep_bs16_seed44/lightning_logs/version_0",
    },
    {
        "key": "a2_order_dyn_twc_w001_seed42",
        "model": "A2-order-dyn+TWC w0.01 seed42",
        "short": "A2+TWC .01",
        "type": "train",
        "version": ROOT / "output/20260707-1714-seqtrack3d_nuscenes_a2_order_dyn_twc_w001-ct_a2_order_dyn_twc_w001_seed42_car_60ep_gpu0_nowpreload/lightning_logs/version_0",
        "diagnostics": {
            "loss_twc": "loss_loss_twc",
            "twc_valid_ratio": "loss_twc_valid_ratio",
            "twc_center_gap": "loss_twc_center_gap",
            "twc_angle_gap": "loss_twc_angle_gap",
        },
    },
    {
        "key": "a3_conf_res_rerun_seed42",
        "model": "A3-conf-res rerun seed42",
        "short": "A3 rerun",
        "type": "train",
        "version": ROOT / "output/20260707-1715-seqtrack3d_nuscenes_a3_order_conf_res_gate-ct_a3_order_conf_res_gate_rerun_seed42_car_60ep_gpu0_nowpreload/lightning_logs/version_0",
        "diagnostics": {
            "obs_alpha_dyn_mean": "loss_obs_alpha_dyn_mean",
            "obs_alpha_dyn_clamped_mean": "loss_obs_alpha_dyn_clamped_mean",
            "obs_dyn_residual_norm": "loss_obs_dyn_residual_norm",
        },
    },
]

DEFAULT_METRICS = {
    "success/test": ("metrics_test_success", "metrics/test"),
    "precision/test": ("metrics_test_precision", "metrics/test"),
}

COLORS = {
    "A3-conf-res best-e14 retest": "#7a5195",
    "A2-order-dyn seed43": "#ef5675",
    "A2-order-dyn seed44": "#2f4b7c",
    "A2-order-dyn+TWC w0.01 seed42": "#ffa600",
    "A3-conf-res rerun seed42": "#3d8b62",
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
            f.read(4)
            payload = f.read(length)
            f.read(4)
            if len(payload) != length:
                raise ValueError(f"Truncated TFRecord payload: {event_file}")
            step, scalars = parse_event(payload)
            if step is None and scalars:
                step = 0
            if step is None:
                continue
            for tag, value in scalars:
                if tag == wanted_tag:
                    points.append((step, value))
    return points


def hparams(version_dir: Path) -> dict[str, str]:
    path = version_dir / "hparams.yaml"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in values:
            values[key] = value
    return values


def checkpoint_epoch_step(params: dict[str, str]) -> tuple[int | None, int | None]:
    checkpoint = params.get("checkpoint", "")
    match = re.search(r"epoch=(\d+)-step=(\d+)", checkpoint)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def scalar_rows(version_dir: Path, scalar_dir: str, event_tag: str) -> list[dict[str, object]]:
    event_files = sorted((version_dir / scalar_dir).glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {version_dir / scalar_dir}")
    rows = []
    for event_file in event_files:
        for step, value in read_scalar_events(event_file, event_tag):
            rows.append({"step": step, "epoch": round(step / STEPS_PER_EPOCH), "value": value})
    rows.sort(key=lambda r: (int(r["step"]), int(r["epoch"])))
    dedup = {}
    for row in rows:
        dedup[int(row["step"])] = row
    return [dedup[k] for k in sorted(dedup)]


def metric_rows(run: dict[str, object], metric: str, params: dict[str, str]) -> list[dict[str, object]]:
    metrics = run.get("metrics", DEFAULT_METRICS)
    scalar_dir, event_tag = metrics[metric]  # type: ignore[index]
    rows = scalar_rows(run["version"], scalar_dir, event_tag)  # type: ignore[arg-type]
    if run["type"] == "test":
        epoch, step = checkpoint_epoch_step(params)
        for row in rows:
            if epoch is not None:
                row["epoch"] = epoch
            if step is not None:
                row["step"] = step
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def tail(values: list[dict[str, object]], n: int = 1000) -> list[float]:
    return [float(r["value"]) for r in values[-n:]]


def summarize_metric(run: dict[str, object], metric: str, rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(r["value"]) for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: float(rows[i]["value"]))
    final = rows[-1]
    best = rows[best_idx]
    return {
        "run_key": run["key"],
        "model": run["model"],
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
        "num_points": len(rows),
    }


def summarize_diagnostic(run: dict[str, object], name: str, scalar_dir: str) -> dict[str, object]:
    rows = scalar_rows(run["version"], scalar_dir, "loss")  # type: ignore[arg-type]
    values = [float(r["value"]) for r in rows]
    tail_values = tail(rows, 1000)
    return {
        "run_key": run["key"],
        "model": run["model"],
        "diagnostic": name,
        "final": values[-1],
        "mean": mean(values),
        "std": std(values),
        "tail1000_mean": mean(tail_values),
        "tail1000_std": std(tail_values),
        "min": min(values),
        "max": max(values),
        "num_points": len(rows),
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


def metric_plot_scale(rows: list[dict[str, object]], metric: str, width: int, height: int, pad: int):
    metric_rows = [r for r in rows if r["metric"] == metric]
    epochs = [int(r["epoch"]) for r in metric_rows]
    values = [float(r["value"]) for r in metric_rows]
    min_x, max_x = min(epochs), max(epochs)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    min_y = math.floor((min(values) - 2) / 5) * 5
    max_y = math.ceil((max(values) + 2) / 5) * 5
    if min_y == max_y:
        min_y -= 5
        max_y += 5
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad

    def sx(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / (max_y - min_y) * plot_h

    return metric_rows, sx, sy, min_x, max_x, min_y, max_y


def svg_line_chart(path: Path, rows: list[dict[str, object]], metric: str, title: str) -> None:
    width, height, pad = 940, 480, 64
    metric_rows, sx, sy, min_x, max_x, min_y, max_y = metric_plot_scale(rows, metric, width, height, pad)
    models = list(dict.fromkeys(r["model"] for r in metric_rows))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    for i in range(6):
        yv = min_y + (max_y - min_y) * i / 5
        y = sy(yv)
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{pad-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.0f}</text>')
    for i in range(0, 7):
        xv = min_x + (max_x - min_x) * i / 6
        x = sx(xv)
        parts.append(f'<text x="{x:.1f}" y="{height-pad+24}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.0f}</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="13">epoch</text>')
    parts.append(f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="13">{metric}</text>')

    for idx, model in enumerate(models):
        model_rows = [r for r in metric_rows if r["model"] == model]
        color = COLORS.get(model, "#555555")
        points = " ".join(f'{sx(int(r["epoch"])):.1f},{sy(float(r["value"])):.1f}' for r in model_rows)
        if len(model_rows) > 1:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')
        for row in model_rows:
            parts.append(
                f'<circle cx="{sx(int(row["epoch"])):.1f}" cy="{sy(float(row["value"])):.1f}" r="4" fill="{color}"/>'
            )
        lx, ly = width - pad - 260, pad + idx * 22
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="14" height="3" fill="{color}"/>')
        parts.append(f'<text x="{lx+20}" y="{ly-5}" font-family="Arial" font-size="12">{model}</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_best_final_bar(path: Path, summary_rows: list[dict[str, object]]) -> None:
    width, height, pad = 1080, 560, 76
    models = list(dict.fromkeys(r["model"] for r in summary_rows))
    metrics = ["success/test", "precision/test"]
    rows_by = {(r["model"], r["metric"]): r for r in summary_rows}
    max_y = math.ceil(max(float(r["best"]) for r in summary_rows) / 10) * 10
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    group_w = plot_w / len(models)
    bar_w = min(28, group_w / 6)

    def sy(y: float) -> float:
        return height - pad - y / max_y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Latest 5 Runs: Best vs Final</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    for i in range(6):
        yv = max_y * i / 5
        y = sy(yv)
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{pad-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.0f}</text>')

    color_map = {
        ("success/test", "final"): "#4e79a7",
        ("success/test", "best"): "#a0cbe8",
        ("precision/test", "final"): "#f28e2b",
        ("precision/test", "best"): "#ffbe7d",
    }
    for i, model in enumerate(models):
        center = pad + group_w * (i + 0.5)
        offsets = [-1.5, -0.5, 0.5, 1.5]
        labels = [
            ("success/test", "final"),
            ("success/test", "best"),
            ("precision/test", "final"),
            ("precision/test", "best"),
        ]
        for offset, (metric, kind) in zip(offsets, labels):
            row = rows_by[(model, metric)]
            value = float(row[kind])
            x = center + offset * bar_w
            y = sy(value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{height-pad-y:.1f}" fill="{color_map[(metric, kind)]}"/>')
            parts.append(f'<text x="{x+(bar_w-2)/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-family="Arial" font-size="9">{value:.1f}</text>')
        label = model.replace(" ", "\n")
        parts.append(f'<text x="{center:.1f}" y="{height-pad+22}" text-anchor="middle" font-family="Arial" font-size="10">{model}</text>')

    legend = [
        ("success final", color_map[("success/test", "final")]),
        ("success best", color_map[("success/test", "best")]),
        ("precision final", color_map[("precision/test", "final")]),
        ("precision best", color_map[("precision/test", "best")]),
    ]
    for i, (label, color) in enumerate(legend):
        x = pad + i * 170
        y = 54
        parts.append(f'<rect x="{x}" y="{y}" width="14" height="10" fill="{color}"/>')
        parts.append(f'<text x="{x+20}" y="{y+10}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_diag_bar(path: Path, diag_rows: list[dict[str, object]]) -> None:
    width, height, pad = 880, 420, 70
    rows = diag_rows
    max_y = max(float(r["tail1000_mean"]) for r in rows)
    max_y = max_y * 1.2 if max_y > 0 else 1.0
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    bar_w = plot_w / max(len(rows), 1) * 0.68

    def sy(y: float) -> float:
        return height - pad - y / max_y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Latest 5 Runs: Diagnostics Tail Mean</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    for i, row in enumerate(rows):
        x = pad + (i + 0.5) * plot_w / len(rows) - bar_w / 2
        value = float(row["tail1000_mean"])
        y = sy(value)
        color = COLORS.get(str(row["model"]), "#666666")
        label = f'{row["model"]} {row["diagnostic"]}'
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height-pad-y:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.4f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-pad+16}" transform="rotate(22 {x+bar_w/2:.1f} {height-pad+16})" text-anchor="start" font-family="Arial" font-size="9">{label}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(summary_rows: list[dict[str, object]], diag_rows: list[dict[str, object]], hparams_rows: list[dict[str, object]]) -> None:
    by_model = {}
    for row in summary_rows:
        by_model.setdefault(row["model"], {})[row["metric"]] = row

    lines = [
        "# Latest 5 Runs Comparison",
        "",
        "## Summary",
        "",
        "| model | success final | success best | precision final | precision best | note |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    notes = {
        "A3-conf-res best-e14 retest": "single checkpoint test; does not reproduce old 62/76 best signal",
        "A2-order-dyn seed43": "large seed regression",
        "A2-order-dyn seed44": "better than seed43 but still below old seed42 60ep report",
        "A2-order-dyn+TWC w0.01 seed42": "lower TWC weight still collapses",
        "A3-conf-res rerun seed42": "rerun remains low and unstable",
    }
    for model, metrics in by_model.items():
        s = metrics["success/test"]
        p = metrics["precision/test"]
        lines.append(
            f"| {model} | {fmt(s['final'])} | {fmt(s['best'])} | {fmt(p['final'])} | {fmt(p['best'])} | {notes.get(model, '')} |"
        )

    lines.extend([
        "",
        "## Diagnostics",
        "",
        "| model | diagnostic | final | mean | tail1000 mean | min | max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in diag_rows:
        lines.append(
            f"| {row['model']} | {row['diagnostic']} | {fmt(row['final'], 4)} | {fmt(row['mean'], 4)} | {fmt(row['tail1000_mean'], 4)} | {fmt(row['min'], 4)} | {fmt(row['max'], 4)} |"
        )

    lines.extend([
        "",
        "## Readout",
        "",
        "1. `A3-conf-res best-e14 retest` only reaches 28.06 / 37.70 on the tested checkpoint, so the earlier 62.04 / 76.30 best point should no longer be treated as confirmed until its exact evaluation path is reconciled.",
        "2. `A2-order-dyn` now shows high seed sensitivity: seed43 collapses to 23.64 / 23.77 while seed44 is 46.90 / 52.62.",
        "3. Reducing TWC from 0.05 to 0.01 does not rescue the A2+dynamics combination: the final result is 22.88 / 24.27, despite valid TWC diagnostics.",
        "4. The A3 conf-res rerun remains low at 32.11 / 31.87; gate/conf-res should move to diagnostic analysis before more structure changes.",
        "",
        "## Generated files",
        "",
        f"- `../data/{PREFIX}_metrics_points.csv`",
        f"- `../data/{PREFIX}_metrics_summary.csv`",
        f"- `../data/{PREFIX}_diagnostics_summary.csv`",
        f"- `../data/{PREFIX}_hparams_summary.csv`",
        f"- `../figures/line_charts/{PREFIX}_success_curve.svg`",
        f"- `../figures/line_charts/{PREFIX}_precision_curve.svg`",
        f"- `../figures/bar_charts/{PREFIX}_best_final_summary.svg`",
        f"- `../figures/diagnostics/{PREFIX}_diagnostics_tail_mean.svg`",
    ])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{PREFIX}_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metric_points: list[dict[str, object]] = []
    metric_summary: list[dict[str, object]] = []
    diag_summary: list[dict[str, object]] = []
    hparams_summary: list[dict[str, object]] = []

    for run in RUNS:
        params = hparams(run["version"])  # type: ignore[arg-type]
        hparams_summary.append({
            "run_key": run["key"],
            "model": run["model"],
            "cfg": params.get("cfg", ""),
            "tag": params.get("tag", ""),
            "seed": params.get("seed", ""),
            "checkpoint": params.get("checkpoint", ""),
            "test": params.get("test", ""),
            "epoch": params.get("epoch", ""),
            "check_val_every_n_epoch": params.get("check_val_every_n_epoch", ""),
        })
        for metric in ("success/test", "precision/test"):
            rows = metric_rows(run, metric, params)
            for row in rows:
                metric_points.append({
                    "run_key": run["key"],
                    "model": run["model"],
                    "metric": metric,
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "value": float(row["value"]),
                })
            metric_summary.append(summarize_metric(run, metric, rows))

        for name, scalar_dir in run.get("diagnostics", {}).items():  # type: ignore[union-attr]
            diag_summary.append(summarize_diagnostic(run, name, scalar_dir))

    write_csv(
        DATA_DIR / f"{PREFIX}_metrics_points.csv",
        metric_points,
        ["run_key", "model", "metric", "epoch", "step", "value"],
    )
    write_csv(
        DATA_DIR / f"{PREFIX}_metrics_summary.csv",
        metric_summary,
        [
            "run_key", "model", "metric", "final", "final_epoch", "final_step",
            "best", "best_epoch", "best_step", "best_final_gap", "mean", "std", "num_points",
        ],
    )
    write_csv(
        DATA_DIR / f"{PREFIX}_diagnostics_summary.csv",
        diag_summary,
        ["run_key", "model", "diagnostic", "final", "mean", "std", "tail1000_mean", "tail1000_std", "min", "max", "num_points"],
    )
    write_csv(
        DATA_DIR / f"{PREFIX}_hparams_summary.csv",
        hparams_summary,
        ["run_key", "model", "cfg", "tag", "seed", "checkpoint", "test", "epoch", "check_val_every_n_epoch"],
    )

    svg_line_chart(LINE_DIR / f"{PREFIX}_success_curve.svg", metric_points, "success/test", "Latest 5 Runs: Success")
    svg_line_chart(LINE_DIR / f"{PREFIX}_precision_curve.svg", metric_points, "precision/test", "Latest 5 Runs: Precision")
    svg_best_final_bar(BAR_DIR / f"{PREFIX}_best_final_summary.svg", metric_summary)
    svg_diag_bar(DIAG_DIR / f"{PREFIX}_diagnostics_tail_mean.svg", diag_summary)
    write_report(metric_summary, diag_summary, hparams_summary)

    print(f"Wrote {REPORT_DIR / f'{PREFIX}_comparison.md'}")


if __name__ == "__main__":
    main()
