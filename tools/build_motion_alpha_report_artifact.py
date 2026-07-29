#!/usr/bin/env python3
"""Build the portable-report artifact for the CT-v2 motion alpha sweep."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "compare_results" / "data"
REPORT_DIR = ROOT / "compare_results" / "reports"
STEM = "ct_motion_alpha_sweep_seed42_20260730"

METRICS_PATH = DATA_DIR / f"{STEM}_metrics.csv"
SUMMARY_PATH = DATA_DIR / f"{STEM}_summary.csv"
DIAGNOSTICS_PATH = DATA_DIR / f"{STEM}_diagnostics.csv"
OUTPUT_PATH = REPORT_DIR / f"{STEM}_artifact.json"

RUN_LABELS = {
    "B0": "B0 baseline",
    "A0": "B1 motion α=0",
    "A025": "B1 motion α=0.25",
    "A075": "B1 motion α=0.75 (historical)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str) -> float | None:
    return None if value == "" else float(value)


def maybe_int(value: str) -> int | None:
    return None if value == "" else int(float(value))


def main() -> None:
    metrics_rows = read_csv(METRICS_PATH)
    summary_rows = read_csv(SUMMARY_PATH)

    validation_metrics = [
        {
            "run_id": row["run_id"],
            "arm": RUN_LABELS[row["run_id"]],
            "epoch": int(row["epoch"]),
            "step": int(row["step"]),
            "success": round(float(row["success"]), 6),
            "precision": round(float(row["precision"]), 6),
        }
        for row in metrics_rows
    ]
    summary = [
        {
            "run_id": row["run_id"],
            "arm": RUN_LABELS[row["run_id"]],
            "alpha": maybe_float(row["alpha"]),
            "final_success": round(float(row["final_success"]), 6),
            "final_precision": round(float(row["final_precision"]), 6),
            "best_success": round(float(row["best_success"]), 6),
            "best_success_epoch": int(row["best_success_epoch"]),
            "best_precision": round(float(row["best_precision"]), 6),
            "best_precision_epoch": int(row["best_precision_epoch"]),
            "late3_success": round(float(row["late3_success"]), 6),
            "late3_precision": round(float(row["late3_precision"]), 6),
        }
        for row in summary_rows
    ]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sources = [
        {
            "id": "validation_metrics_source",
            "label": "TensorBoard validation metrics and provenance review",
            "path": f"compare_results/data/{STEM}_metrics.csv",
            "query": {
                "language": "python",
                "engine": "DuckDB",
                "sql": (
                    "SELECT run_id, epoch, step, success, precision "
                    f"FROM read_csv_auto('compare_results/data/{STEM}_metrics.csv', "
                    "header = true) ORDER BY run_id, epoch"
                ),
                "description": (
                    "tools/analyze_motion_alpha_sweep.py reads the four reviewed "
                    "TensorBoard runs, validates completion/provenance, and extracts "
                    "Success and Precision at twelve five-epoch validation checkpoints."
                ),
                "executed_at": "2026-07-30",
                "filters": [
                    "nuScenes v1.0-mini Car validation",
                    "seed=42",
                    "60 training epochs",
                    "batch size=16",
                    "12 validation checkpoints at epochs 5..60",
                ],
                "metric_definitions": [
                    "Final Success/Precision: validation metric at epoch 60.",
                    "Best Success/Precision: maximum of the 12 validation checkpoints.",
                    "Late-3 mean: arithmetic mean at epochs 50, 55, and 60.",
                ],
            },
        },
        {
            "id": "run_summary_source",
            "label": "Run-level validation summary",
            "path": f"compare_results/data/{STEM}_summary.csv",
            "query": {
                "language": "python",
                "engine": "DuckDB",
                "sql": (
                    "SELECT run_id, label, alpha, final_success, final_precision, "
                    "best_success, best_success_epoch, best_precision, "
                    "best_precision_epoch, late3_success, late3_precision "
                    f"FROM read_csv_auto('compare_results/data/{STEM}_summary.csv', "
                    "header = true) ORDER BY final_success DESC"
                ),
                "description": (
                    "Reads the reviewed run-level final, best, and late-window "
                    "validation summary produced by tools/analyze_motion_alpha_sweep.py."
                ),
                "executed_at": "2026-07-30",
                "filters": [
                    "nuScenes v1.0-mini Car validation",
                    "seed=42",
                    "60 training epochs",
                ],
                "metric_definitions": [
                    "Final Success/Precision: validation metric at epoch 60.",
                    "Best Success/Precision: maximum of the 12 validation checkpoints.",
                    "Late-3 mean: arithmetic mean at epochs 50, 55, and 60.",
                ],
            },
        },
        {
            "id": "diagnostics_source",
            "label": "Motion intervention diagnostics",
            "path": f"compare_results/data/{STEM}_diagnostics.csv",
            "query": {
                "language": "python",
                "description": (
                    "Aggregated training diagnostics for nominal/effective alpha, "
                    "application ratio, correction norm, clamp ratio, and losses."
                ),
                "executed_at": "2026-07-30",
                "filters": [
                    "Same four reviewed runs",
                    "Post-warmup aggregates exclude epochs 0..4",
                ],
                "metric_definitions": [
                    "Applied ratio: share of samples where a non-zero motion innovation is applied.",
                    "Correction norm: Euclidean magnitude in metres of the applied proposal correction.",
                    "Clamp ratio: share of raw motion residuals clipped by the safety bound.",
                ],
            },
        },
        {
            "id": "implementation_source",
            "label": "CT-v2 motion implementation and data path",
            "path": "models/seqtrack3d.py",
            "query": {
                "language": "python",
                "description": (
                    "Code-path audit of proposal innovation in models/seqtrack3d.py "
                    "and the training reference construction in datasets/sampler.py."
                ),
                "executed_at": "2026-07-30",
                "filters": [
                    "CT-v2 dynamics path",
                    "Training versus recursive evaluation reference boxes",
                ],
            },
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "CT-v2 Motion Fixed-Alpha 复核",
            "description": (
                "B1 motion α=0/0.25 新跑结果与 B0、历史 α=0.75 的完整性、"
                "效果和失效机制复核。"
            ),
            "generatedAt": generated_at,
            "sources": sources,
            "charts": [
                {
                    "id": "success_by_epoch",
                    "title": "Validation Success by epoch",
                    "subtitle": (
                        "α=0.25 虽优于历史 α=0.75，但从 epoch 25 起始终低于 α=0，"
                        "最终也显著低于 B0。"
                    ),
                    "type": "line",
                    "dataset": "validation_metrics",
                    "sourceId": "validation_metrics_source",
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
                        },
                        "color": {
                            "field": "arm",
                            "type": "nominal",
                            "label": "Run",
                        },
                        "tooltip": [
                            {"field": "arm", "type": "nominal", "label": "Run"},
                            {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                            {
                                "field": "success",
                                "type": "quantitative",
                                "label": "Success",
                            },
                        ],
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Success",
                },
                {
                    "id": "precision_by_epoch",
                    "title": "Validation Precision by epoch",
                    "subtitle": (
                        "正 alpha 的 Precision 降幅比 Success 更大，符合错误平移"
                        "直接破坏中心定位并经递归传播的特征。"
                    ),
                    "type": "line",
                    "dataset": "validation_metrics",
                    "sourceId": "validation_metrics_source",
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        "y": {
                            "field": "precision",
                            "type": "quantitative",
                            "label": "Precision",
                        },
                        "color": {
                            "field": "arm",
                            "type": "nominal",
                            "label": "Run",
                        },
                        "tooltip": [
                            {"field": "arm", "type": "nominal", "label": "Run"},
                            {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                            {
                                "field": "precision",
                                "type": "quantitative",
                                "label": "Precision",
                            },
                        ],
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Precision",
                },
            ],
            "tables": [
                {
                    "id": "run_summary",
                    "title": "Run-level validation summary",
                    "subtitle": (
                        "B0 与 α=0/0.25 为本轮主比较；α=0.75 来自历史提交，仅作方向性上下文。"
                    ),
                    "dataset": "run_summary",
                    "sourceId": "run_summary_source",
                    "density": "compact",
                    "defaultSort": {"field": "final_success", "direction": "desc"},
                    "columns": [
                        {"field": "arm", "label": "Run", "type": "text"},
                        {
                            "field": "alpha",
                            "label": "Alpha",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "final_success",
                            "label": "Final Success",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "final_precision",
                            "label": "Final Precision",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "best_success",
                            "label": "Best Success",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "best_precision",
                            "label": "Best Precision",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "late3_success",
                            "label": "Late-3 Success",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "late3_precision",
                            "label": "Late-3 Precision",
                            "type": "number",
                            "format": "number",
                        },
                    ],
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# CT-v2 Motion Fixed-Alpha 复核",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "validation_metrics_source",
                    "body": (
                        "## 技术结论\n\n"
                        "**当前这版“固定全局 alpha 的 motion proposal innovation”不能涨点，"
                        "应停止继续做长周期 alpha 扫描。** α=0.25 最终为 "
                        "**29.581 / 28.862**，相对 B0 的 **53.360 / 64.382** 分别下降 "
                        "**23.779 / 35.520**；相对同构 α=0 也下降 **17.468 / 20.322**。"
                        "α=0.25 比历史 α=0.75 略好，只说明减弱干预能减轻伤害，"
                        "并不构成正收益。"
                    ),
                },
                {"id": "summary_table_block", "type": "table", "tableId": "run_summary"},
                {
                    "id": "evidence",
                    "type": "markdown",
                    "sourceId": "validation_metrics_source",
                    "body": (
                        "## 关键证据\n\n"
                        "四条曲线均完成 60 epoch、12 次验证。正 alpha 的任何验证点都没有"
                        "同时超过同 epoch B0 的 Success 和 Precision。α=0.25 从 epoch 25 "
                        "到 epoch 60 在两个指标上连续 8/8 个检查点低于 α=0。α=0 的最佳点"
                        "（epoch 35：49.876 / 58.691）也低于同 epoch B0"
                        "（51.539 / 63.763），但这个差异不能直接归因于 motion 干预。"
                    ),
                },
                {"id": "success_chart_block", "type": "chart", "chartId": "success_by_epoch"},
                {
                    "id": "precision_chart_block",
                    "type": "chart",
                    "chartId": "precision_by_epoch",
                },
                {
                    "id": "mechanism",
                    "type": "markdown",
                    "sourceId": "diagnostics_source",
                    "body": (
                        "## 为什么不涨点\n\n"
                        "α=0.25 在 warmup 后的平均有效 alpha 为 **0.184**，约 **73.7%** "
                        "样本实际施加修正，平均修正量 **0.083 m**，且约 **33.0%** 的原始"
                        "残差触发 clamp。它不是“没有接上”，而是频繁、持续地接入了一个"
                        "方向尚未被在线可靠性验证的校正。更大的 alpha 同时带来更低的局部"
                        "训练 loss 和更差的递归验证，说明核心矛盾是训练目标/状态分布与"
                        "在线递归跟踪不匹配，而不是训练不充分。"
                    ),
                },
                {
                    "id": "integration",
                    "type": "markdown",
                    "sourceId": "implementation_source",
                    "body": (
                        "## 接入审计\n\n"
                        "训练时 dynamics 看到由 GT 历史构造的 `ct_motion_ref_boxs` / "
                        "`canonical_ref_boxs`，评估时却接收模型上一帧的递归预测；"
                        "`dynamics_valid` 只表示历史 transition 存在，不表示方向准确。"
                        "motion 修正发生在 `aux_box` 和 transformer box-corner query 之前，"
                        "因此一个小的错误平移会同时改变搜索几何和下一帧状态，并被递归放大。"
                        "这解释了 Precision 比 Success 掉得更重。"
                    ),
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## 解释边界\n\n"
                        "α=0 是严格的零干预 fallback，但不是与 B0 完全同构的因果对照："
                        "新增 DynamicsEncoder 会改变后续共享层的随机初始化顺序；两次新跑"
                        "也未声明完全确定性。因此，对“固定正 alpha 有害”的判断为高置信，"
                        "对“整个 motion prior 思路无价值”则证据不足。历史 α=0.75 来自"
                        "不同提交，只用于观察剂量方向。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 下一步\n\n"
                        "先做同一 checkpoint 的无训练 2×2 互换评估：A0 checkpoint 分别"
                        "以 α=0/0.25 推理，A025 checkpoint 分别以 α=0.25/0 推理。随后导出"
                        "逐帧 observation proposal、dynamics proposal、GT、上一帧误差、"
                        "分歧量、历史有效性、点数、速度和 Δt，计算 helpful rate、oracle "
                        "alpha、修正方向与 GT residual 的余弦，以及首次失控帧和漂移长度。"
                        "只有在独立切分上出现稳定、可由 GT-free 特征识别的受益子群后，"
                        "才值得实现条件式 alpha；否则关闭该分支。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 决策问题\n\n"
                        "1. 关闭 α 后，A025 checkpoint 能否立即恢复？\n"
                        "2. dynamics proposal 的 helpful rate 是否在任何 GT-free bucket "
                        "稳定高于 50%？\n"
                        "3. 首次失控前，是否存在足够早且可泛化的可靠性信号？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "validation_metrics": validation_metrics,
                "run_summary": summary,
            },
        },
        "sources": sources,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
