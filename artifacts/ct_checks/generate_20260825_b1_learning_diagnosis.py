"""Append a focused B1 learning diagnosis to the latest verified report.

The experiment output trees remain read-only.  This script only writes derived
tables, a SQLite evidence snapshot, and a report artifact under
``artifacts/ct_checks/reports``.
"""

from __future__ import annotations

import copy
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "artifacts/ct_checks/reports/20260825_rerun_and_short_term_plan"
REPORT_DIR = ROOT / "artifacts/ct_checks/reports/20260825_b1_learning_diagnosis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "B1-only retryfix epoch30": ROOT
    / "output/20260824-0220-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_retryfix",
    "B1-only rerun epoch30": ROOT
    / "output/20260825-0057-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_rerun_20260825",
}


def scalar_tail(run: Path, directory: str, count: int = 213) -> float:
    accumulator = EventAccumulator(
        str(run / "lightning_logs/version_0" / directory),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    values = [row.value for row in accumulator.Scalars("loss")]
    if not values:
        raise RuntimeError(f"missing scalar values: {run} / {directory}")
    return float(np.mean(values[-count:]))


rows = []
for run_name, run in RUNS.items():
    frame = pd.read_csv(
        run / "lightning_logs/version_0/candidate_diagnostics/epoch_30.csv"
    )
    valid = (
        (frame["b1_valid"] > 0)
        & np.isfinite(frame["learned_motion_error"])
        & np.isfinite(frame["kinematic_error"])
        & np.isfinite(frame["b1_nll"])
    )
    frame = frame.loc[valid].copy()
    paired_delta = frame["learned_motion_error"] - frame["kinematic_error"]
    tail_threshold = frame["kinematic_error"].quantile(0.99)
    tail = frame["kinematic_error"] > tail_threshold
    prior_loss = scalar_tail(run, "loss_loss_motion_v3_prior")
    nll_loss = scalar_tail(run, "loss_loss_motion_v3_nll")
    weighted_prior = 0.1 * prior_loss
    weighted_nll = 0.05 * nll_loss
    rows.append(
        {
            "run": run_name,
            "valid_rows": int(len(frame)),
            "learned_rmse": float(np.sqrt(np.mean(frame["learned_motion_error"] ** 2))),
            "cv_rmse": float(np.sqrt(np.mean(frame["kinematic_error"] ** 2))),
            "learned_minus_cv_rmse": float(
                np.sqrt(np.mean(frame["learned_motion_error"] ** 2))
                - np.sqrt(np.mean(frame["kinematic_error"] ** 2))
            ),
            "paired_help_rate": float((paired_delta < 0).mean()),
            "mean_nll": float(frame["b1_nll"].mean()),
            "coverage_95": float(frame["b1_coverage_95"].mean()),
            "sigma_parallel_median": float(frame["sigma_parallel"].median()),
            "sigma_perpendicular_median": float(frame["sigma_perpendicular"].median()),
            "residual_parallel_median": float(frame["residual_unit_parallel"].median()),
            "top1pct_nll_share": float(
                frame.loc[tail, "b1_nll"].sum() / frame["b1_nll"].sum()
            ),
            "tail213_prior_loss": prior_loss,
            "tail213_nll_loss": nll_loss,
            "weighted_prior_contribution": weighted_prior,
            "weighted_nll_contribution": weighted_nll,
            "nll_to_prior_contribution_ratio": float(weighted_nll / weighted_prior),
        }
    )

diagnosis_rows = [
    {
        "component": "B1 mean / GRU residual",
        "status": "FAIL: weak and path-dependent",
        "evidence": (
            "Retryfix learned−CV RMSE = −0.091 m, but rerun = +0.107 m; "
            "rerun helps only 4.2% of valid rows and residual_parallel median is +0.942."
        ),
        "causal_role": "Direct B1 effectiveness failure; not explained by coverage alone.",
        "next_action": (
            "Keep GRU and fixed search geometry; train mean with robust SmoothL1, "
            "and stop NLL gradients from steering the mean in the diagnostic arm."
        ),
    },
    {
        "component": "NLL objective / loss scale",
        "status": "FAIL: dominates B1 transaction",
        "evidence": (
            "Rerun tail-213 weighted contributions are about 1.266 for NLL versus "
            "0.068 for mean SmoothL1, a ratio of 18.6×."
        ),
        "causal_role": "Likely optimization driver of residual-head collapse.",
        "next_action": (
            "Use detached-residual sigma NLL and select its weight from measured "
            "gradient norms, not the raw metric magnitude."
        ),
    },
    {
        "component": "Sigma parameterization",
        "status": "FAIL: under-dispersed and tail-infeasible",
        "evidence": (
            "Rerun median sigma is about 0.323 m per axis while errors reach about "
            "70 m; shared-anchor sigma is capped by 4 m / 3 m envelopes."
        ),
        "causal_role": "Creates very large NLL gradients on recursive outliers.",
        "next_action": (
            "Decouple uncertainty bounds from the fixed crop geometry, initialize "
            "from train-residual scale, and use a robust/censored likelihood for tails."
        ),
    },
    {
        "component": "Coverage",
        "status": "FAIL as a diagnostic, not a cause",
        "evidence": "Rerun 95% empirical coverage is 31.1%; retryfix is 45.6%.",
        "causal_role": "Reports sigma miscalibration; it does not backpropagate in current code.",
        "next_action": (
            "After the mean passes CV, fit a held-out tracklet calibration scale and "
            "gate on 50/80/95% coverage plus NLL."
        ),
    },
    {
        "component": "Upstream recursive B0 path",
        "status": "UNRESOLVED confounder",
        "evidence": (
            "Matched B1 runs share initial state and observation fingerprints but "
            "their B0 hashes diverge after step1."
        ),
        "causal_role": "Changes the online history/error distribution presented to B1.",
        "next_action": "Complete the sequential step1 gradient/Adam-state audit before formal B1 claims.",
    },
]

artifact = copy.deepcopy(
    json.loads((BASE_DIR / "artifact.json").read_text(encoding="utf-8"))
)
source_id = "src_b1_learning_diagnosis_20260825"
source = {
    "id": source_id,
    "label": "B1 epoch30 candidate diagnostics and TensorBoard loss-scale audit",
    "path": "artifacts/ct_checks/reports/20260825_b1_learning_diagnosis/b1_learning_diagnosis.sqlite",
    "query": {
        "engine": "sqlite",
        "sql": "SELECT snapshot_json FROM b1_diagnosis_snapshot WHERE snapshot_id = 'b1_learning_20260825'",
        "tablesUsed": ["b1_diagnosis_snapshot"],
    },
}
artifact["manifest"]["title"] = "CT-SeqTrack B1 学习失败诊断与修复门槛（2026-08-25）"
artifact["manifest"]["description"] = (
    "在既有复跑审计上追加B1均值、NLL、sigma与coverage的因果拆分。"
)
artifact["manifest"]["sources"].append(source)
artifact["sources"].append(source)
artifact["snapshot"]["datasets"]["b1_loss_scale_audit"] = rows
artifact["snapshot"]["datasets"]["b1_causal_diagnosis"] = diagnosis_rows

artifact["manifest"]["tables"].extend(
    [
        {
            "id": "table_b1_loss_scale_audit",
            "title": "两次 B1-only 的均值、sigma 与损失尺度",
            "subtitle": "epoch30逐帧诊断；loss贡献使用训练末213个B1记录及正式权重",
            "dataset": "b1_loss_scale_audit",
            "sourceId": source_id,
            "defaultSort": {"field": "run", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "learned_minus_cv_rmse", "label": "Learned−CV RMSE", "format": "number", "movement": True},
                {"field": "paired_help_rate", "label": "Help rate", "format": "percent"},
                {"field": "coverage_95", "label": "95% coverage", "format": "percent"},
                {"field": "sigma_parallel_median", "label": "Median sigma∥", "format": "number"},
                {"field": "residual_parallel_median", "label": "Median residual unit∥", "format": "number"},
                {"field": "top1pct_nll_share", "label": "Top1% NLL share", "format": "percent"},
                {"field": "nll_to_prior_contribution_ratio", "label": "Weighted NLL / mean", "format": "number"},
            ],
        },
        {
            "id": "table_b1_causal_diagnosis",
            "title": "B1 问题的因果拆分",
            "subtitle": "coverage是结果指标；NLL/sigma设计与递归路径才是优化原因或混杂因素",
            "dataset": "b1_causal_diagnosis",
            "sourceId": source_id,
            "defaultSort": {"field": "component", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "component", "label": "Component", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "causal_role", "label": "Causal role", "type": "text"},
                {"field": "next_action", "label": "Next action", "type": "text"},
            ],
        },
    ]
)
artifact["manifest"]["blocks"].extend(
    [
        {
            "id": "b1_learning_conclusion_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## B1结论：不是完全学不到，而是当前联合目标会把均值头推入路径相关的近常数解\n\n"
                "旧run在epoch30仅比CV好0.091 m，新run反而差0.107 m；新run逐帧只有4.2%优于CV，"
                "平行残差单元的中位数为+0.942，表现为接近全局常数的补偿。"
                "这已经构成B1有效性失败，但不能归因于coverage本身。当前NLL对B1损失的加权贡献约为均值SmoothL1的19倍；"
                "sigma又被压到约0.32 m，而递归长尾误差可达约70 m且sigma受4/3 m envelope上界限制。"
                "因此高NLL与低coverage共同证明不确定性失配，NLL的梯度耦合和不可覆盖长尾则很可能正在破坏mean学习。"
            ),
        },
        {"id": "b1_loss_scale_table_block", "type": "table", "layout": "full", "tableId": "table_b1_loss_scale_audit"},
        {"id": "b1_causal_table_block", "type": "table", "layout": "full", "tableId": "table_b1_causal_diagnosis"},
        {
            "id": "b1_repair_sequence_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## 推荐修复顺序（保持GRU、固定geometry、全部从零训练）\n\n"
                "1. 先完成B0 step1 gradient/Adam-state顺序审计，避免把不同递归轨迹误判为B1改进。\n"
                "2. 为B1增加mean/sigma分支梯度与输出审计；保留GRU和CV anchor，固定search crop。\n"
                "3. 将均值优化固定为robust SmoothL1；sigma NLL使用detach后的残差训练，使两个head都从零参与，但NLL不再牵引mean。\n"
                "4. sigma不再受crop envelope的4/3 m上界约束；初始尺度由mini_train残差统计给出，并对递归长尾使用robust/censored likelihood。\n"
                "5. mean在validation上稳定优于CV后，才用独立calibration tracklets拟合后验尺度。正式门槛为paired CI支持learned RMSE<CV、NLL优于固定sigma基线、95% coverage接近名义值。"
            ),
        },
        {
            "id": "b1_diagnosis_method_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## 本次追加的口径与限制\n\n"
                "逐帧指标来自两次B1-only的epoch30 candidate diagnostics；损失尺度取各run训练末213个B1 scalar记录，"
                "并使用配置中的prior_weight=0.1、nll_weight=0.05换算。只有两个匹配端点，不新增比较图；"
                "现有曲线与全部旧数据源原样保留。当前结论用于定位机制失败，不构成论文增益声明。"
            ),
        },
    ]
)

snapshot_payload = {"loss_scale_audit": rows, "causal_diagnosis": diagnosis_rows}
db_path = REPORT_DIR / "b1_learning_diagnosis.sqlite"
with sqlite3.connect(db_path) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS b1_diagnosis_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO b1_diagnosis_snapshot(snapshot_id, snapshot_json) VALUES (?, ?)",
        ("b1_learning_20260825", json.dumps(snapshot_payload, ensure_ascii=False)),
    )

for filename, data in (
    ("b1_loss_scale_audit.csv", rows),
    ("b1_causal_diagnosis.csv", diagnosis_rows),
):
    with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

(REPORT_DIR / "analysis_summary.json").write_text(
    json.dumps(snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(REPORT_DIR / "artifact.json")
