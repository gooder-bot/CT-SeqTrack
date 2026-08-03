#!/usr/bin/env python3
"""Build the canonical portable report for the B2-v2.1 diagnosis."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "compare_results" / "data"
REPORTS = ROOT / "compare_results" / "reports"
STEM = "b2_search_v21_seed42_20260803"

TEXT_FIELDS = {
    "run_id", "arm", "role", "treatment", "baseline",
    "comparison_basis", "question", "criterion", "observed", "decision",
    "finding", "evidence", "interpretation", "confidence", "action",
    "change", "decision_unlocked", "gpu_cost", "file", "symbol",
    "verified_behavior", "risk", "checkpoint", "git_commit", "git_dirty",
    "config_path", "init_checkpoint", "status",
}


def rows(name: str) -> list[dict[str, Any]]:
    path = DATA / f"{STEM}_{name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        output = list(csv.DictReader(handle))
    for row in output:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
                continue
            if key in TEXT_FIELDS:
                continue
            if value in {"True", "False"}:
                row[key] = value == "True"
                continue
            try:
                numeric = float(value)
                row[key] = int(numeric) if numeric.is_integer() else numeric
            except ValueError:
                pass
    return output


def source(
        source_id: str,
        label: str,
        path: str,
        description: str,
        definitions: list[str] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "language": "python",
        "description": description,
        "executed_at": "2026-08-03",
    }
    if path.endswith(".csv"):
        query.update({
            "engine": "DuckDB",
            "sql": f"SELECT * FROM read_csv_auto('{path}', header=true)",
        })
    if definitions:
        query["metric_definitions"] = definitions
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": query,
    }


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    validation = rows("validation")
    summary = rows("summary")
    comparisons = rows("comparisons")
    training = rows("v21_training_epochs")
    v2_training = rows("v2_training_epochs")
    diagnostics = rows("derived_diagnostics")
    integrity = rows("integrity")
    checkpoints = rows("checkpoint_diagnostics")
    drivers = rows("drivers")
    decisions = rows("decision")
    next_steps = rows("next_steps")
    code_audit = rows("code_audit")
    cause_timeline = rows("cause_timeline")
    base_loss_comparison = rows("base_loss_comparison")
    seqtrack_checkpoint_audit = rows("seqtrack_checkpoint_audit")
    retrain_decision = rows("retrain_decision")

    short_names = {
        "SEQTRACK_HIST": "Historical SeqTrack",
        "B0_HIST": "Historical B0",
        "B2V2_FULL": "B2-v2 full",
        "V21_SEARCH": "V2.1 Search-only",
        "V21_FULL": "V2.1 full",
    }
    validation_chart = []
    for row in validation:
        if row["run_id"] not in short_names:
            continue
        current = dict(row)
        current["series"] = short_names[row["run_id"]]
        validation_chart.append(current)

    summary_visible = [
        row for row in summary
        if row["run_id"] in {
            "SEQTRACK_NEW", "SEQTRACK_HIST", "B0_HIST", "LEGACY_SEARCH",
            "B2V2_FULL", "V21_SEARCH", "V21_FULL",
        }
    ]
    key_comparisons = [
        row for row in comparisons
        if (row["treatment"], row["baseline"]) in {
            ("V21_SEARCH", "LEGACY_SEARCH"),
            ("V21_SEARCH", "SEQTRACK_HIST"),
            ("V21_SEARCH", "B0_HIST"),
            ("V21_FULL", "B2V2_FULL"),
            ("V21_FULL", "SEQTRACK_NEW"),
        }
    ]

    gate_rows = []
    gate_names = {
        "search_valid_rate": "Search valid",
        "search_material_selection_rate": "Applied weight ≥ 0.1",
        "search_helpful_prevalence_valid": "Helpful prevalence",
        "search_helpful_precision_selected": "Helpful precision",
    }
    for row in diagnostics[:2]:
        for field, label in gate_names.items():
            gate_rows.append({
                "run": (
                    "Search-only" if row["run_id"] == "V21_SEARCH"
                    else "Motion + Search"),
                "metric": label,
                "rate": row[field],
                "epoch": 60,
            })

    search_evidence_rows = []
    v2_e60 = next(row for row in v2_training if row["epoch"] == 60)
    for run_id, label in (
            ("V21_SEARCH", "V2.1 Search-only"),
            ("V21_FULL", "V2.1 full")):
        row = next(
            item for item in training
            if item["run_id"] == run_id and item["epoch"] == 60)
        search_evidence_rows.append({
            "run": label,
            "valid_rate": row["search_candidate_valid_rate"],
            "foreground_points": row["search_foreground_points"],
            "targetness_loss": row["search_targetness_loss"],
            "vote_loss": row["search_vote_loss"],
            "proposal_loss": row["search_proposal_loss"],
        })
    search_evidence_rows.insert(0, {
        "run": "B2-v2 full",
        "valid_rate": v2_e60["search_candidate_valid_rate"],
        "foreground_points": v2_e60["search_foreground_points"],
        "targetness_loss": v2_e60["search_targetness_loss"],
        "vote_loss": v2_e60["search_vote_loss"],
        "proposal_loss": v2_e60["search_proposal_loss"],
    })

    training_anchor_rows = [
        row for row in training
        if row["run_id"] in {"V21_SEARCH", "V21_FULL"}
        and row["epoch"] in {10, 15, 20, 40, 60}
    ]
    integrity_visible = [
        row for row in integrity
        if row["run_id"] in {
            "SEQTRACK_NEW", "B0_HIST", "B2V2_FULL",
            "V21_SEARCH", "V21_FULL",
        }
    ]
    cause_timeline_visible = [
        row for row in cause_timeline
        if row["epoch"] in {10, 15, 60}
    ]
    base_loss_visible = [
        row for row in base_loss_comparison
        if row["epoch"] == 60
    ]

    sources = [
        source(
            "validation_source",
            "Reviewed normal-validation TensorBoard scalars",
            f"compare_results/data/{STEM}_validation.csv",
            "tools/analyze_b2_search_v21.py extracts all twelve unsmoothed validation points from each run's raw TensorBoard event files.",
            [
                "Final is epoch60 and is the primary decision point.",
                "Late-3 is the arithmetic mean of epochs 50, 55, and 60.",
                "Best checkpoints are diagnostic only.",
            ],
        ),
        source(
            "summary_source",
            "Reviewed final, late-3, and best checkpoint summary",
            f"compare_results/data/{STEM}_summary.csv",
            "Final, late-3, and best values are recomputed from all twelve paired validation points.",
        ),
        source(
            "comparison_source",
            "Reviewed comparison register",
            f"compare_results/data/{STEM}_comparisons.csv",
            "Pairwise score differences preserve the comparator's provenance limitations and never substitute a best checkpoint for epoch60.",
        ),
        source(
            "training_source",
            "B2-v2.1 epoch-level training diagnostics",
            f"compare_results/data/{STEM}_v21_training_epochs.csv",
            "Epoch means are regenerated from 75,720 TensorBoard training steps using 1,262 batches per epoch for both V2.1 runs.",
        ),
        source(
            "diagnostic_source",
            "Derived gate and evidence diagnostics",
            f"compare_results/data/{STEM}_derived_diagnostics.csv",
            "Conditional rates and weights are recomputed from raw epoch60 means; the ESS valid-row estimate removes the mathematically verified 1e6 invalid-row sentinel.",
        ),
        source(
            "driver_source",
            "B2-v2.1 evidence and root-cause register",
            f"compare_results/data/{STEM}_drivers.csv",
            "Findings distinguish directly verified evidence from closed-loop causal inference and unresolved attribution.",
        ),
        source(
            "decision_source",
            "Frozen decision checks",
            f"compare_results/data/{STEM}_decision.csv",
            "Applies the stated epoch60, late-3, and helpful-precision gates without best-checkpoint substitution.",
        ),
        source(
            "integrity_source",
            "Run integrity and comparability audit",
            f"compare_results/data/{STEM}_integrity.csv",
            "Checks provenance, clean commit, config, seed, training length, validation coverage, and final checkpoint presence.",
        ),
        source(
            "code_source",
            "B2-v2.1 implementation audit",
            f"compare_results/data/{STEM}_code_audit.csv",
            "Code review covers the evidence encoder, advantage fusion, one-step loss, and source-aware sampler.",
        ),
        source(
            "next_source",
            "Controlled follow-up register",
            f"compare_results/data/{STEM}_next_steps.csv",
            "Orders inference-only attribution and verified bug repair before any new full training.",
        ),
        source(
            "cause_source",
            "Regression timing and intervention audit",
            f"compare_results/data/{STEM}_cause_timeline.csv",
            "Aligns both SeqTrack controls and both V2.1 runs at epochs 5, 10, 15, 20, and 60; the expected full-fusion ramp is recorded explicitly.",
        ),
        source(
            "base_loss_source",
            "Base supervised-loss comparison",
            f"compare_results/data/{STEM}_base_loss_comparison.csv",
            "Recomputes epoch means for total, center, angle, segmentation, and box-cloud losses at seven fixed training anchors for four runs.",
        ),
        source(
            "seqtrack_audit_source",
            "SeqTrack checkpoint divergence audit",
            f"compare_results/data/{STEM}_seqtrack_checkpoint_audit.csv",
            "Compares every state tensor in the two epoch10 SeqTrack checkpoints and records the largest BatchNorm running-stat differences and missing provenance.",
        ),
        source(
            "retrain_source",
            "Retraining decision register",
            f"compare_results/data/{STEM}_retrain_decision.csv",
            "Separates unchanged retraining, inference-only attribution, and corrected-model retraining decisions with confidence labels.",
        ),
    ]

    charts = [
        {
            "id": "success_curve",
            "title": "Normal validation Success by epoch",
            "subtitle": "nuScenes-mini Car, seed42; 12 unsmoothed validation checkpoints; epoch60 is primary.",
            "type": "line",
            "dataset": "validation_chart",
            "sourceId": "validation_source",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                "y": {"field": "success", "type": "quantitative", "label": "Success"},
                "color": {"field": "series", "type": "nominal", "label": "Run"},
                "tooltip": [
                    {"field": "series", "type": "nominal", "label": "Run"},
                    {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    {"field": "success", "type": "quantitative", "label": "Success"},
                ],
            },
            "xAxisTitle": "Epoch",
            "yAxisTitle": "Success",
        },
        {
            "id": "precision_curve",
            "title": "Normal validation Precision by epoch",
            "subtitle": "Motion+Search V2.1 collapses as the epoch10-19 fusion ramp activates.",
            "type": "line",
            "dataset": "validation_chart",
            "sourceId": "validation_source",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                "y": {"field": "precision", "type": "quantitative", "label": "Precision"},
                "color": {"field": "series", "type": "nominal", "label": "Run"},
                "tooltip": [
                    {"field": "series", "type": "nominal", "label": "Run"},
                    {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    {"field": "precision", "type": "quantitative", "label": "Precision"},
                ],
            },
            "xAxisTitle": "Epoch",
            "yAxisTitle": "Precision",
        },
        {
            "id": "gate_chart",
            "title": "Epoch-60 Search availability and gate reliability",
            "subtitle": "Rates are fractions of all rows except prevalence and precision, which condition on valid or selected Search rows.",
            "type": "bar",
            "dataset": "gate_rows",
            "sourceId": "diagnostic_source",
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                "y": {"field": "rate", "type": "quantitative", "label": "Rate"},
                "color": {"field": "run", "type": "nominal", "label": "Run"},
                "tooltip": [
                    {"field": "run", "type": "nominal", "label": "Run"},
                    {"field": "metric", "type": "nominal", "label": "Metric"},
                    {"field": "rate", "type": "quantitative", "label": "Rate"},
                ],
            },
            "xAxisTitle": "Metric",
            "yAxisTitle": "Rate",
        },
    ]

    tables = [
        {
            "id": "summary_table",
            "title": "Normal-validation result summary",
            "subtitle": "Final and late-3 drive the decision; best is shown only for diagnosis.",
            "dataset": "summary_visible",
            "sourceId": "summary_source",
            "density": "compact",
            "defaultSort": {"field": "final_success", "direction": "desc"},
            "columns": [
                {"field": "arm", "label": "Run", "type": "text"},
                {"field": "final_success", "label": "Final S", "type": "number", "format": "number"},
                {"field": "final_precision", "label": "Final P", "type": "number", "format": "number"},
                {"field": "late3_success", "label": "Late-3 S", "type": "number", "format": "number"},
                {"field": "late3_precision", "label": "Late-3 P", "type": "number", "format": "number"},
                {"field": "best_success", "label": "Best S", "type": "number", "format": "number"},
                {"field": "best_success_epoch", "label": "Best-S epoch", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "comparison_table",
            "title": "Key epoch-60 and late-3 deltas",
            "subtitle": "Positive favors V2.1; comparison basis records non-matched controls.",
            "dataset": "key_comparisons",
            "sourceId": "comparison_source",
            "density": "compact",
            "columns": [
                {"field": "treatment", "label": "Treatment", "type": "text"},
                {"field": "baseline", "label": "Comparator", "type": "text"},
                {"field": "delta_final_success", "label": "Δ Final S", "type": "number", "format": "number", "semantic": "movement"},
                {"field": "delta_final_precision", "label": "Δ Final P", "type": "number", "format": "number", "semantic": "movement"},
                {"field": "delta_late3_success", "label": "Δ Late-3 S", "type": "number", "format": "number", "semantic": "movement"},
                {"field": "delta_late3_precision", "label": "Δ Late-3 P", "type": "number", "format": "number", "semantic": "movement"},
                {"field": "comparison_basis", "label": "Basis", "type": "text"},
            ],
        },
        {
            "id": "search_evidence_table",
            "title": "Epoch-60 Search evidence quality",
            "subtitle": "Training means; lower losses are better, valid rate uses all training rows.",
            "dataset": "search_evidence_rows",
            "sourceId": "training_source",
            "density": "compact",
            "defaultSort": {"field": "valid_rate", "direction": "desc"},
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "valid_rate", "label": "Valid rate", "type": "number", "format": "percent"},
                {"field": "foreground_points", "label": "FG points", "type": "number", "format": "number"},
                {"field": "targetness_loss", "label": "Targetness loss", "type": "number", "format": "number"},
                {"field": "vote_loss", "label": "Vote loss", "type": "number", "format": "number"},
                {"field": "proposal_loss", "label": "Proposal loss", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "driver_table",
            "title": "Root-cause evidence register",
            "subtitle": "Confidence explicitly separates verified metrics, verified code behavior, and likely closed-loop causality.",
            "dataset": "drivers",
            "sourceId": "driver_source",
            "density": "spacious",
            "defaultSort": {"field": "priority", "direction": "asc"},
            "columns": [
                {"field": "priority", "label": "Priority", "type": "number", "format": "number"},
                {"field": "finding", "label": "Finding", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
                {"field": "confidence", "label": "Confidence", "type": "text"},
            ],
        },
        {
            "id": "decision_table",
            "title": "Decision checks",
            "subtitle": "The anomalous new SeqTrack control is shown but cannot establish method attribution.",
            "dataset": "decisions",
            "sourceId": "decision_source",
            "density": "spacious",
            "columns": [
                {"field": "question", "label": "Question", "type": "text"},
                {"field": "criterion", "label": "Criterion", "type": "text"},
                {"field": "observed", "label": "Observed", "type": "text"},
                {"field": "decision", "label": "Decision", "type": "text"},
            ],
        },
        {
            "id": "integrity_table",
            "title": "Run integrity and comparability",
            "subtitle": "Both V2.1 runs are complete, clean, same-commit scratch runs; the new SeqTrack control lacks provenance.",
            "dataset": "integrity_visible",
            "sourceId": "integrity_source",
            "density": "compact",
            "columns": [
                {"field": "run_id", "label": "Run", "type": "text"},
                {"field": "provenance_present", "label": "Provenance", "type": "boolean"},
                {"field": "git_commit", "label": "Commit", "type": "text"},
                {"field": "git_dirty", "label": "Dirty", "type": "text"},
                {"field": "seed", "label": "Seed", "type": "number", "format": "number"},
                {"field": "workers", "label": "Workers", "type": "number", "format": "number"},
                {"field": "validation_points", "label": "Val points", "type": "number", "format": "number"},
                {"field": "last_checkpoint_present", "label": "Last ckpt", "type": "boolean"},
            ],
        },
        {
            "id": "code_table",
            "title": "Implementation audit",
            "subtitle": "Direct code behavior relevant to the observed failure.",
            "dataset": "code_audit",
            "sourceId": "code_source",
            "density": "spacious",
            "columns": [
                {"field": "file", "label": "File", "type": "text"},
                {"field": "symbol", "label": "Symbol", "type": "text"},
                {"field": "verified_behavior", "label": "Verified behavior", "type": "text"},
                {"field": "risk", "label": "Risk", "type": "text"},
            ],
        },
        {
            "id": "next_table",
            "title": "Controlled next steps",
            "subtitle": "Inference attribution and the verified numerical bug come before another 60-epoch full run.",
            "dataset": "next_steps",
            "sourceId": "next_source",
            "density": "spacious",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "Order", "type": "number", "format": "number"},
                {"field": "action", "label": "Action", "type": "text"},
                {"field": "change", "label": "Change", "type": "text"},
                {"field": "decision_unlocked", "label": "Decision unlocked", "type": "text"},
                {"field": "gpu_cost", "label": "GPU cost", "type": "text"},
            ],
        },
        {
            "id": "cause_timeline_table",
            "title": "两次退步的时序与 intervention 对齐",
            "subtitle": "Epoch10/15/60；两组 V2.1 使用同一 ramp，但只有 Motion+Search 崩塌。",
            "dataset": "cause_timeline_visible",
            "sourceId": "cause_source",
            "density": "compact",
            "columns": [
                {"field": "run_id", "label": "Run", "type": "text"},
                {"field": "epoch", "label": "Epoch", "type": "number", "format": "number"},
                {"field": "success", "label": "Success", "type": "number", "format": "number"},
                {"field": "precision", "label": "Precision", "type": "number", "format": "number"},
                {"field": "advantage_fusion_ramp_expected", "label": "Fusion ramp", "type": "number", "format": "percent"},
                {"field": "interpretation", "label": "State", "type": "text"},
            ],
        },
        {
            "id": "base_loss_table",
            "title": "Epoch60 基础监督损失",
            "subtitle": "V2.1 total 包含额外分支损失，因此判断 B0 时以 center/angle/seg/box-cloud 为主。",
            "dataset": "base_loss_visible",
            "sourceId": "base_loss_source",
            "density": "compact",
            "columns": [
                {"field": "run_id", "label": "Run", "type": "text"},
                {"field": "loss_total", "label": "Total", "type": "number", "format": "number"},
                {"field": "loss_center", "label": "Center", "type": "number", "format": "number"},
                {"field": "loss_angle", "label": "Angle", "type": "number", "format": "number"},
                {"field": "loss_seg", "label": "Seg", "type": "number", "format": "number"},
                {"field": "loss_bc", "label": "Box-cloud", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "seqtrack_audit_table",
            "title": "SeqTrack epoch10 checkpoint 分叉",
            "subtitle": "同一 state key 集，但多数 tensor 已分叉；最大差异集中在 BatchNorm running stats。",
            "dataset": "seqtrack_checkpoint_audit",
            "sourceId": "seqtrack_audit_source",
            "density": "compact",
            "columns": [
                {"field": "shared_tensor_count", "label": "Shared", "type": "number", "format": "number"},
                {"field": "exact_tensor_count", "label": "Exact", "type": "number", "format": "number"},
                {"field": "changed_tensor_count", "label": "Changed", "type": "number", "format": "number"},
                {"field": "bn_rank", "label": "BN rank", "type": "number", "format": "number"},
                {"field": "bn_tensor", "label": "BN tensor", "type": "text"},
                {"field": "bn_max_abs_difference", "label": "Max |Δ|", "type": "number", "format": "number"},
                {"field": "interpretation", "label": "Interpretation", "type": "text"},
            ],
        },
        {
            "id": "retrain_table",
            "title": "是否重训的分层决策",
            "subtitle": "区分原样重训、归因测试和修复后重训。",
            "dataset": "retrain_decision",
            "sourceId": "retrain_source",
            "density": "spacious",
            "columns": [
                {"field": "question", "label": "Question", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "decision", "label": "Decision", "type": "text"},
                {"field": "confidence", "label": "Confidence", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# B2-v2.1 seed42：Observation-Queried Search 与 Advantage Fusion 复核"},
        {"id": "technical_summary", "type": "markdown", "sourceId": "summary_source", "body": "## 技术结论：两次退步共享递归放大器，但不是同一直接原因\n\n**Search-only** epoch60 为 **52.743 Success / 63.046 Precision**，比 legacy search-only 高 **3.088 / 6.654**。**Motion+Search full** 只有 **25.848 / 25.177**，并在 fusion ramp 开启时突降。异常 SeqTrack control 则从首次验证就低，且基础训练损失与历史 SeqTrack 几乎相同、缺少运行 provenance。结论是：两者都会被递归历史放大，但 V2.1 full 有明确的 fusion/Motion 结构触发；**不要原样重训 current full，先做四模式归因并修复 gate，修复后的 full 需要从零重训。**"},
        {"id": "summary_block", "type": "table", "tableId": "summary_table"},
        {"id": "comparison_finding", "type": "markdown", "sourceId": "comparison_source", "body": "## Search-only 的正向结论取决于基线口径，full 在所有口径下失败\n\n本次新 SeqTrack control final 只有 **31.684/31.337**，且没有 `run_provenance.json`；Search-only 相对它的 +21.059/+31.709 是异常 control 放大的数值，不能写成论文收益。历史 SeqTrack 给出数值上涨，但更强的历史 B0 给出负值。full 即使对这个异常低 control 也仍是 **−5.836/−6.160**。"},
        {"id": "comparison_block", "type": "table", "tableId": "comparison_table"},
        {"id": "curve_finding", "type": "markdown", "sourceId": "validation_source", "body": "## full 的崩塌与 fusion ramp 同步，不是 epoch60 偶然波动\n\nfull 在 fusion 关闭的 epoch10 尚为 **32.406/31.001**；ramp 进入中段的 epoch15 立即降到 **22.896/21.687**，之后一直维持约 25 分。Search-only 在相同 ramp 下 epoch15 已达到 **51.741/63.872**。时间对齐把故障范围缩小到 Motion 与 advantage fusion 的闭环交互。"},
        {"id": "success_block", "type": "chart", "chartId": "success_curve"},
        {"id": "precision_block", "type": "chart", "chartId": "precision_curve"},
        {"id": "search_finding", "type": "markdown", "sourceId": "training_source", "body": "## Observation-query 与 overlap 证据确实改善了 Search 分支\n\n相对 B2-v2，V2.1 full 的 epoch60 Search valid rate 从 **23.29%** 升到 **30.71%**，平均目标点从 **3.68** 增到 **7.51**，proposal loss 从 **0.01361** 降到 **0.01034**（约 **24.0%**）。这证明 Search encoder/crop 改造方向有效；最终没有涨过 B0，主要是候选覆盖与 gate 选择仍不合格。"},
        {"id": "search_evidence_block", "type": "table", "tableId": "search_evidence_table"},
        {"id": "gate_finding", "type": "markdown", "sourceId": "diagnostic_source", "body": "## Advantage gate 实际退化为 availability gate\n\nSearch-only 在 epoch60 对约 **97.8%** 的 valid Search 行施加了至少 0.1 权重，valid 行条件平均权重约 **0.486**，几乎顶到 normal 0.5 上限。被选 Search 的 helpful precision 只有 **55.8%**，而自然 helpful prevalence 已有 **54.7%**，提升仅 **1.1 个百分点**，远低于预设 70%。full 中也选择了约 **97.1%** valid Search 行。"},
        {"id": "gate_block", "type": "chart", "chartId": "gate_chart"},
        {"id": "root_finding", "type": "markdown", "sourceId": "driver_source", "body": "## full 崩塌由 Motion 过度使用和一个无效行数值缺陷共同驱动\n\nfull 的 Motion candidate 在 **90.4%** 样本有效，却只有 **16.9%** 比 observation 至少好 0.05 m；仍获得约 **0.272** 的 valid-row 条件权重。与此同时，Search 无效行的 pool weight 全零，`1 / clamp(sum(w²), 1e-6)` 会生成 **1e6** 的 effective sample size；该值未按 `search_valid` 屏蔽就进入共享 gate。epoch60 Search 无效率约 69.3%，所以这一异常特征可以在多数 Motion 决策中出现。代码缺陷与时序证据已验证；它在总崩塌中的独立因果份额仍需四模式推理确认。"},
        {"id": "driver_block", "type": "table", "tableId": "driver_table"},
        {"id": "same_cause_finding", "type": "markdown", "sourceId": "cause_source", "body": "## 与异常 SeqTrack 不是同一直接原因\n\n异常 SeqTrack 在没有任何 fusion intervention 的情况下从 epoch5 起就只有 **32.303/32.136**，之后始终约 27–32 分；V2.1 full 在 ramp 关闭的 epoch10 为 **32.406/31.001**，到 ramp≈0.5 的 epoch15 才突降至 **22.896/21.687**，而 Search-only 同期升至 **51.741/63.872**。因此共同点是 recursive tracking 会放大小的模型差异；不同点是 V2.1 full 的直接触发被定位到 Motion/fusion 闭环，SeqTrack 的触发仍是缺 provenance 的训练流/代码数据状态不确定性。"},
        {"id": "cause_timeline_block", "type": "table", "tableId": "cause_timeline_table"},
        {"id": "base_loss_finding", "type": "markdown", "sourceId": "base_loss_source", "body": "## 两次退步都不是普通的 supervised underfit\n\n新旧 SeqTrack epoch60 total loss 为 **0.22027 vs 0.21873**，center loss 为 **0.02147 vs 0.02105**，但验证相差 **19.302 Success / 28.625 Precision**。V2.1 Search-only/full 的 center loss为 **0.02220/0.02127**，full 反而略低；angle、seg、box-cloud 也同量级。这排除了‘再用同一配置多训一次就会自然修好’的主要依据，但 observation-only recursive 表现仍应由同 checkpoint 反事实确认。"},
        {"id": "base_loss_block", "type": "table", "tableId": "base_loss_table"},
        {"id": "seqtrack_divergence_finding", "type": "markdown", "sourceId": "seqtrack_audit_source", "body": "## SeqTrack 更像随机训练流与 BatchNorm 路径分叉后被递归放大\n\n两次 SeqTrack 的前三个 total-loss step 完全一致，epoch10 checkpoint 的 **320** 个 state tensor key 也完全对应，但只有 **28** 个 tensor 逐值相等、**292** 个已经分叉，最大差异集中在 BatchNorm running mean/variance。新跑使用 workers4、历史跑使用 workers12，且两者都没有 `run_provenance.json`，所以可以确认 stochastic/BN path 已分叉，却不能把根因唯一归结为 workers 数、代码版本或数据缓存。"},
        {"id": "seqtrack_audit_block", "type": "table", "tableId": "seqtrack_audit_table"},
        {"id": "decision_finding", "type": "markdown", "sourceId": "decision_source", "body": "## 决策：保留 Search-v2.1 思路，否决当前 full 与当前 gate\n\nSearch-v2.1 通过“优于 legacy Search”的修复目标，也通过“相对历史 SeqTrack 数值上涨”的弱口径；但未通过最强 B0 guardrail，更没有同 commit matched B0。full 同时失败于 V2 对比、SeqTrack 对比和闭环稳定性。"},
        {"id": "decision_block", "type": "table", "tableId": "decision_table"},
        {"id": "scope", "type": "markdown", "sourceId": "integrity_source", "body": "## 范围、运行合同与数据质量\n\n两组 V2.1 都是 commit `16c2b8b` 的 clean scratch run：nuScenes v1.0-mini Car、seed42、batch16、workers4、60 epoch、每 5 epoch 验证，共 12 个验证点和 epoch59/last checkpoint；训练/验证分别为 5051/2285 frames。V2 与历史 B0 来自其他 clean commit；新 SeqTrack control 完整但缺 provenance。因此结果足以否决 current full，尚不足以正式宣布 Search-only 相对 matched B0 涨点。"},
        {"id": "integrity_block", "type": "table", "tableId": "integrity_table"},
        {"id": "method", "type": "markdown", "sourceId": "code_source", "body": "## 方法与实现核验\n\n分析从原始 TensorBoard event 重算 final、late-3、best 和 75,720-step epoch means，并读取两份 last checkpoint 的 gate 参数。实现审计确认：B0 1024 点未被 Search 分支改写；Search point/query/context 在新分支中学习；candidate 和 B0 context 在 joint loss 前 detach。full epoch60 的 B0 center loss与 Search-only 同量级，故当前证据不支持“B0 被辅助梯度改坏”。"},
        {"id": "code_block", "type": "table", "tableId": "code_table"},
        {"id": "limitations", "type": "markdown", "body": "## 限制与未决归因\n\n本轮只有一个 seed，没有 commit-16c2b8b matched B0、四模式 inference、endpoint/tracklet 级导出、AUPRC 或 paired bootstrap。helpful precision 是训练 batch 比率的 epoch mean，不是 validation endpoint precision。两组 V2.1 还曾并行占用同一物理 GPU2，runtime 标量不能用于 15% runtime 晋级判断。这些限制不会改变 full 的失败结论，但会限制 Search-only 的论文归因。"},
        {"id": "next_finding", "type": "markdown", "sourceId": "next_source", "body": "## 下一步先归因和修 bug，不要直接再跑 60 epoch\n\n先用现有 checkpoint 跑 `obs / obs_motion / obs_search / full`；随后把 invalid Search 的 ESS/统计量置零并归一化，收紧 Motion 介入和 gate abstention，再补同 commit B0。只有 corrected full 的 Search helpful precision 达到 70%、epoch60 相对 matched B0 同时达到 +0.5/+1.0，才值得进入 seed43/44。"},
        {"id": "next_block", "type": "table", "tableId": "next_table"},
        {"id": "retrain_finding", "type": "markdown", "sourceId": "retrain_source", "body": "## 重训结论分三层\n\n**current full 不要原样重训**：已存在 verified numerical defect 和不选择性的 gate，换 seed 只能把结构风险伪装成随机波动。**当前 Search-only 不必立即重训**：保留 checkpoint，先跑 `obs/obs_search`。**修复后的 full 必须从零重训**：gate 输入、abstention 与监督分布已经改变；正式 60 epoch 前先在 ramp 跨越点做短诊断，确认 epoch15 不再崩塌。"},
        {"id": "retrain_block", "type": "table", "tableId": "retrain_table"},
        {"id": "questions", "type": "markdown", "body": "## 仍需回答的问题\n\n- full checkpoint 的 `obs` 模式能否恢复到约 52–53 分，从而直接证明 B0 没坏？\n- `obs_motion` 是否独立复现 25 分崩塌，还是只有 Search-invalid ESS 与 Motion gate 交互才触发？\n- valid Search 的 validation AUPRC 是否明显高于 55% prevalence？\n- 修复 invalid ESS 并加入 abstention 后，Search-only 能否超过 commit-matched B0，而不是只超过 legacy Search？"},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "B2-v2.1 seed42：Observation-Queried Search 与 Advantage Fusion 复核",
            "description": "Normal-mini outcome, V2 and baseline comparisons, Search evidence quality, gate calibration, closed-loop failure drivers, and controlled next steps.",
            "generatedAt": generated,
            "sources": sources,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated,
            "datasets": {
                "validation_chart": validation_chart,
                "summary_visible": summary_visible,
                "key_comparisons": key_comparisons,
                "gate_rows": gate_rows,
                "search_evidence_rows": search_evidence_rows,
                "training_anchor_rows": training_anchor_rows,
                "drivers": drivers,
                "decisions": decisions,
                "integrity_visible": integrity_visible,
                "checkpoints": checkpoints,
                "code_audit": code_audit,
                "next_steps": next_steps,
                "cause_timeline_visible": cause_timeline_visible,
                "base_loss_visible": base_loss_visible,
                "seqtrack_checkpoint_audit": seqtrack_checkpoint_audit,
                "retrain_decision": retrain_decision,
            },
        },
        "sources": [],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    artifact_path = REPORTS / f"{STEM}_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    notes = {
        "audience": "technical",
        "delivery_mode": "portable_html",
        "required_structure_mapping": {
            "technical_summary": "technical_summary",
            "key_findings_with_visual_evidence": [
                "comparison_finding", "curve_finding", "search_finding",
                "gate_finding", "root_finding", "same_cause_finding",
                "base_loss_finding", "seqtrack_divergence_finding",
                "decision_finding", "retrain_finding",
            ],
            "scope_data_metric_definitions": "scope",
            "methodology": "method",
            "limitations_uncertainty_robustness": "limitations",
            "recommended_next_steps": "next_finding",
            "further_questions": "questions",
        },
        "chart_map": [
            {
                "section": "validation outcome",
                "question": "How do Success trajectories compare through epoch60?",
                "family": "trend",
                "type": "line",
                "fields": ["epoch", "success", "series"],
                "point_count": len(validation_chart),
                "takeaway": "Search-only repairs the legacy path; full collapses after fusion activation.",
                "palette": "relaxed multi-category native report palette; legend and line geometry provide non-color distinction",
                "delivery": f"compare_results/reports/{STEM}.html",
            },
            {
                "section": "validation outcome",
                "question": "How do Precision trajectories compare through epoch60?",
                "family": "trend",
                "type": "line",
                "fields": ["epoch", "precision", "series"],
                "point_count": len(validation_chart),
                "takeaway": "Full precision remains collapsed while Search-only stays near the historical B0.",
                "palette": "same mapping as Success for cross-chart consistency",
                "delivery": f"compare_results/reports/{STEM}.html",
            },
            {
                "section": "gate reliability",
                "question": "Does applied Search selection enrich helpful candidates?",
                "family": "comparison",
                "type": "grouped bar",
                "fields": ["metric", "rate", "run"],
                "category_count": 4,
                "takeaway": "Material selection nearly equals availability and helpful precision barely exceeds prevalence.",
                "palette": "hard two-root cap with direct metric axis and visible run legend",
                "delivery": f"compare_results/reports/{STEM}.html",
            },
        ],
        "source_notes": [
            "The new SeqTrack control lacks run_provenance.json and is anomalously low.",
            "V2, historical B0, and V2.1 are clean but not same-commit runs.",
            "Runtime is omitted from promotion because the V2.1 runs contended on physical GPU2.",
            "No endpoint-level validation export exists, so AUPRC and paired bootstrap remain unavailable.",
            "The two SeqTrack controls lack run provenance and differ in worker count, so the exact stochastic-divergence initiator is unresolved.",
        ],
    }
    (DATA / f"{STEM}_report_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
