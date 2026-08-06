#!/usr/bin/env python3
"""Build the canonical CT joint-Full ablation technical report artifact."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "compare_results" / "data"
REPORTS = ROOT / "compare_results" / "reports"


def read_csv(name: str):
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value):
    return float(value)


def metric_at(rows, run, metric, epoch=60):
    for row in rows:
        if (row["run"] == run and row["metric"] == metric
                and int(row["epoch"]) == epoch):
            return f(row["mean"])
    raise KeyError((run, metric, epoch))


def source(source_id, label, path, sql, description):
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "snapshot",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": [f"snapshot.{source_id}"],
            "executed_at": "2026-08-06T00:00:00Z",
        },
    }


def main():
    validation = read_csv("joint_full_validation_curves_20260806.csv")
    diagnostics = read_csv("joint_full_training_diagnostics_20260806.csv")
    provenance = read_csv("joint_full_run_provenance_20260806.csv")
    baseline15_validation = read_csv(
        "joint_full_baseline15_validation_20260806.csv")
    baseline15_losses = read_csv(
        "joint_full_baseline15_b0_losses_20260806.csv")

    run_order = ["Full", "-B1", "-B2", "-B3", "SeqTrack"]
    final_by_run = {
        row["run"]: row for row in validation if int(row["epoch"]) == 60
    }
    final_metrics = []
    curve_summary = []
    for run in run_order:
        rows = sorted(
            (row for row in validation if row["run"] == run),
            key=lambda row: int(row["epoch"]),
        )
        final = final_by_run[run]
        baseline = final_by_run["SeqTrack"]
        final_metrics.append({
            "run": run,
            "success": f(final["success"]),
            "precision": f(final["precision"]),
            "success_delta_vs_seqtrack": (
                f(final["success"]) - f(baseline["success"])),
            "precision_delta_vs_seqtrack": (
                f(final["precision"]) - f(baseline["precision"])),
            "commit": next(row["commit"] for row in provenance
                           if row["run"] == run)[:8],
        })
        last_three = rows[-3:]
        curve_summary.append({
            "run": run,
            "best_success": max(f(row["success"]) for row in rows),
            "best_precision": max(f(row["precision"]) for row in rows),
            "last3_success": sum(f(row["success"]) for row in last_three) / 3,
            "last3_precision": sum(f(row["precision"]) for row in last_three) / 3,
            "curve_mean_success": sum(f(row["success"]) for row in rows) / len(rows),
            "curve_mean_precision": sum(f(row["precision"]) for row in rows) / len(rows),
        })

    validation_rows = [{
        "run": row["run"],
        "epoch": int(row["epoch"]),
        "success": f(row["success"]),
        "precision": f(row["precision"]),
    } for row in validation]

    early_chart_runs = {
        "Current B0-15", "Historical B0-60", "Full-60", "-B2-60"
    }
    early_validation_rows = [{
        "run": row["run"],
        "epoch": int(row["epoch"]),
        "success": f(row["success"]),
        "precision": f(row["precision"]),
    } for row in baseline15_validation if row["run"] in early_chart_runs]
    early_epoch15_rows = [{
        "run": row["run"],
        "success": f(row["success"]),
        "precision": f(row["precision"]),
        "success_delta_vs_current_b0": (
            f(row["success"])
            - next(f(candidate["success"])
                   for candidate in baseline15_validation
                   if candidate["run"] == "Current B0-15"
                   and int(candidate["epoch"]) == 15)),
        "precision_delta_vs_current_b0": (
            f(row["precision"])
            - next(f(candidate["precision"])
                   for candidate in baseline15_validation
                   if candidate["run"] == "Current B0-15"
                   and int(candidate["epoch"]) == 15)),
    } for row in baseline15_validation if int(row["epoch"]) == 15]

    def early_loss(run, metric):
        return next(
            f(row["mean"]) for row in baseline15_losses
            if row["run"] == run and row["metric"] == metric
            and int(row["epoch"]) == 15)

    early_b0_loss_rows = [{
        "run": run,
        "center": early_loss(run, "loss_center"),
        "center_aux": early_loss(run, "loss_center_aux"),
        "center_ref": early_loss(run, "loss_center_ref"),
        "seg": early_loss(run, "loss_seg"),
    } for run in ("Current B0-15", "Historical B0-60", "Full-60", "-B2-60")]

    rmse_rows = []
    for run in ["Full", "-B1", "-B3"]:
        for proposal, metric in (
                ("Observation", "ct_observation_rmse"),
                ("Raw Search", "ct_raw_search_rmse"),
                ("Final", "ct_final_rmse")):
            rmse_rows.append({
                "run": run,
                "proposal": proposal,
                "rmse_m": metric_at(diagnostics, run, metric),
            })

    full_obs = metric_at(diagnostics, "Full", "ct_observation_rmse")
    full_raw = metric_at(diagnostics, "Full", "ct_raw_search_rmse")
    full_final = metric_at(diagnostics, "Full", "ct_final_rmse")
    kin = metric_at(diagnostics, "Full", "motion_v3_kinematic_rmse")
    b1 = metric_at(diagnostics, "Full", "motion_v3_prior_rmse")
    valid_rate = metric_at(diagnostics, "Full", "ct_candidate_valid_rate")
    mechanism_checks = [
        {
            "component": "Shared-anchor B1",
            "metric": "B1 prior RMSE",
            "value": b1,
            "reference": kin,
            "delta": b1 - kin,
            "interpretation": "Internally useful: 7.1% lower than the kinematic anchor.",
        },
        {
            "component": "Dynamic residual bound",
            "metric": "Residual saturation rate",
            "value": metric_at(
                diagnostics, "Full", "ct_motion_residual_saturation"),
            "reference": 0.05,
            "delta": metric_at(
                diagnostics, "Full", "ct_motion_residual_saturation") - 0.05,
            "interpretation": "Stable; the bounded B1 residual is not exploding.",
        },
        {
            "component": "Query reliability gate",
            "metric": "Gate mean / BCE",
            "value": metric_at(diagnostics, "Full", "ct_query_gate_mean"),
            "reference": metric_at(
                diagnostics, "Full", "loss_ct_query_gate"),
            "delta": metric_at(
                diagnostics, "Full", "loss_ct_query_gate") - 0.693147,
            "interpretation": "Not discriminative: BCE remains near log(2).",
        },
        {
            "component": "Expansion support",
            "metric": "Candidate-valid rate",
            "value": valid_rate,
            "reference": 0.95,
            "delta": valid_rate - 0.95,
            "interpretation": "Fails the planned 95% availability target.",
        },
        {
            "component": "B2 raw Search",
            "metric": "Raw/observation RMSE ratio",
            "value": full_raw / full_obs,
            "reference": 1.0,
            "delta": full_raw / full_obs - 1.0,
            "interpretation": "Unsafe: raw Search error is 2.21x observation error.",
        },
        {
            "component": "B3 router",
            "metric": "Final/observation RMSE ratio",
            "value": full_final / full_obs,
            "reference": 1.0,
            "delta": full_final / full_obs - 1.0,
            "interpretation": "Helps teacher-forced batches, but not recursive tracking.",
        },
    ]

    b0_loss_rows = []
    for run in run_order:
        b0_loss_rows.append({
            "run": run,
            "center": metric_at(diagnostics, run, "loss_center"),
            "center_aux": metric_at(diagnostics, run, "loss_center_aux"),
            "center_ref": metric_at(diagnostics, run, "loss_center_ref"),
            "seg": metric_at(diagnostics, run, "loss_seg"),
        })

    provenance_rows = []
    for row in provenance:
        provenance_rows.append({
            "run": row["run"],
            "commit": row["commit"][:8],
            "clean": not (row["dirty_any"].lower() == "true"),
            "seed": int(row["seed"]),
            "train_frames": int(row["train_frames"]),
            "val_frames": int(row["val_frames"]),
            "train_selection": row["train_selection_sha256"][:10],
            "val_selection": row["val_selection_sha256"][:10],
        })

    next_experiments = [
        {
            "priority": "DONE",
            "experiment": "Current-commit SeqTrack B0, seed42, 15 epochs",
            "question": "Does commit 835f911 preserve the recursive B0 baseline?",
            "decision": "Yes at early training: 46.76/53.65 at epoch 15; global B0 collapse is rejected.",
        },
        {
            "priority": "P0",
            "experiment": "Same-B0-checkpoint baseline vs -B2 recursive cross-evaluation",
            "question": "Does the common Joint inference path alter B0 inputs or predictions?",
            "decision": "Equal low scores imply training/RNG trajectory; unequal scores identify inference transmission.",
        },
        {
            "priority": "P0",
            "experiment": "First-batch and 100-step baseline vs -B2 equivalence audit",
            "question": "Are B0 inputs, outputs, gradients, optimizer updates and RNG streams identical?",
            "decision": "Locate the first divergent B0 tensor/step before another full training run.",
        },
        {
            "priority": "P1",
            "experiment": "Standalone recursive endpoint export for Full and -B2",
            "question": "At which frames do observation/raw/final error streaks begin?",
            "decision": "Only after the common Joint path passes the B0 equivalence checks.",
        },
        {
            "priority": "P2",
            "experiment": "Repair query-gate supervision and alpha=0 identity",
            "question": "Does B1 improve B2 only on confidently helpful samples?",
            "decision": "Promote B1 coupling only if Full beats -B1 and -B2 on both metrics.",
        },
    ]

    sources = [
        source(
            "validation_metrics",
            "TensorBoard validation metrics",
            "compare_results/data/joint_full_validation_curves_20260806.csv",
            "SELECT run, epoch, success, precision FROM snapshot.validation_metrics ORDER BY epoch, run",
            "Reviewed 5-epoch validation points parsed from the five TensorBoard event streams.",
        ),
        source(
            "training_diagnostics",
            "Epoch-level training diagnostics",
            "compare_results/data/joint_full_training_diagnostics_20260806.csv",
            "SELECT run, metric, epoch, mean, min, max, all_finite FROM snapshot.training_diagnostics WHERE epoch = 60",
            "Reviewed epoch-60 means parsed from per-scalar TensorBoard event streams.",
        ),
        source(
            "run_provenance",
            "Run provenance",
            "compare_results/data/joint_full_run_provenance_20260806.csv",
            "SELECT run, commit, seed, train_frames, val_frames, train_selection_sha256, val_selection_sha256 FROM snapshot.run_provenance",
            "Reviewed configuration, git, dataset and selection fingerprints for the five runs.",
        ),
        source(
            "baseline15_validation",
            "Current-commit B0 and Joint early validation",
            "compare_results/data/joint_full_baseline15_validation_20260806.csv",
            "SELECT run, epoch, success, precision FROM snapshot.baseline15_validation ORDER BY epoch, run",
            "Reviewed epoch-5/10/15 recursive validation metrics for current B0, historical B0 and four Joint runs.",
        ),
        source(
            "baseline15_losses",
            "Current-commit B0 and Joint early B0 losses",
            "compare_results/data/joint_full_baseline15_b0_losses_20260806.csv",
            "SELECT run, metric, epoch, mean, batch_count FROM snapshot.baseline15_losses WHERE epoch = 15",
            "Reviewed B0 training-loss means through epoch 15 for current B0 and Joint comparisons.",
        ),
        source(
            "baseline15_provenance",
            "Current B0 and Joint early-run provenance",
            "compare_results/data/joint_full_baseline15_provenance_20260806.csv",
            "SELECT run, commit, seed, epochs, batch_size, train_frames, val_frames, train_selection_sha256, val_selection_sha256, first15_lr_min, first15_lr_max FROM snapshot.baseline15_provenance",
            "Reviewed commit, configuration and dataset fingerprints for the current B0 and early-run comparisons.",
        ),
        {
            "id": "query_code",
            "label": "Joint query and Search implementation",
            "path": "models/ct_v2/joint_full.py",
        },
        {
            "id": "loss_code",
            "label": "Joint loss and router supervision implementation",
            "path": "models/seqtrack3d.py",
        },
        {
            "id": "analysis_code",
            "label": "Reproducible analysis script",
            "path": "tools/analyze_joint_full_ablation.py",
            "query": {
                "engine": "snapshot",
                "language": "sql",
                "sql": "SELECT priority, experiment, question, decision FROM snapshot.next_experiments ORDER BY priority",
                "description": "Prioritized decision gates produced from the reviewed validation and diagnostic evidence.",
                "tables_used": ["snapshot.next_experiments"],
                "executed_at": "2026-08-06T00:00:00Z",
            },
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "CT-SeqTrack Joint Full 消融与15轮基线诊断",
            "description": "nuScenes mini Car seed42 Joint ablation and current-commit 15-epoch B0 diagnosis.",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "cards": [
                {
                    "id": "full_success",
                    "description": "Full 的 epoch-60 Success，后项为相对历史 SeqTrack 的百分点差。",
                    "dataset": "headline",
                    "sourceId": "validation_metrics",
                    "metrics": [
                        {"label": "Full Success", "field": "full_success", "format": "number", "unit": "%"},
                        {"label": "vs SeqTrack", "field": "full_success_delta", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
                {
                    "id": "full_precision",
                    "description": "Full 的 epoch-60 Precision，后项为相对历史 SeqTrack 的百分点差。",
                    "dataset": "headline",
                    "sourceId": "validation_metrics",
                    "metrics": [
                        {"label": "Full Precision", "field": "full_precision", "format": "number", "unit": "%"},
                        {"label": "vs SeqTrack", "field": "full_precision_delta", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
                {
                    "id": "best_ablation",
                    "description": "四组联合实验中，-B2 的 epoch-60 Success 最高。",
                    "dataset": "headline",
                    "sourceId": "validation_metrics",
                    "metrics": [
                        {"label": "Best joint run: -B2 Success", "field": "minus_b2_success", "format": "number", "unit": "%"},
                        {"label": "Full − -B2", "field": "full_vs_minus_b2_success", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
                {
                    "id": "raw_ratio",
                    "description": "Full epoch-60 teacher-forced训练批次上的 raw Search/observation 中心误差比。",
                    "dataset": "headline",
                    "sourceId": "training_diagnostics",
                    "metrics": [
                        {"label": "Raw Search / observation RMSE", "field": "raw_obs_ratio", "format": "number", "unit": "×"},
                    ],
                },
                {
                    "id": "current_b0_early",
                    "description": "当前 commit 纯 B0 的 epoch-15 Success；后项为相对同 epoch -B2 的差值。",
                    "dataset": "headline",
                    "sourceId": "baseline15_validation",
                    "metrics": [
                        {"label": "Current B0 epoch-15 Success", "field": "current_b0_e15_success", "format": "number", "unit": "%"},
                        {"label": "vs -B2 at epoch 15", "field": "current_b0_vs_minus_b2_e15", "format": "number", "unit": "pp", "signed": True},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "success_curve",
                    "title": "Validation Success by epoch",
                    "subtitle": "所有 Joint 变体在 60 epochs 内均明显低于历史 SeqTrack。",
                    "type": "line",
                    "dataset": "validation_curve",
                    "sourceId": "validation_metrics",
                    "encodings": {
                        "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                        "y": {"field": "success", "type": "quantitative", "label": "Success (%)"},
                        "color": {"field": "run", "type": "nominal", "label": "Run"},
                    },
                    "yAxisTitle": "Success (%)",
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "rmse_chart",
                    "title": "Full/-B1/-B3 proposal RMSE at epoch 60",
                    "subtitle": "Raw Search 明显差于 observation；B3 只能在 teacher-forced 批次中部分收回误差。",
                    "type": "bar",
                    "dataset": "rmse",
                    "sourceId": "training_diagnostics",
                    "encodings": {
                        "x": {"field": "run", "type": "nominal", "label": "Run"},
                        "y": {"field": "rmse_m", "type": "quantitative", "label": "Mean center error (m)"},
                        "color": {"field": "proposal", "type": "nominal", "label": "Proposal"},
                    },
                    "yAxisTitle": "Mean center error (m)",
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "early_success_chart",
                    "title": "Validation Success at epochs 5, 10 and 15",
                    "subtitle": "Same seed, split, batch size, global steps and first-15 learning rate; current B0 separates from Joint after epoch 5.",
                    "type": "bar",
                    "dataset": "early_validation",
                    "sourceId": "baseline15_validation",
                    "encodings": {
                        "x": {"field": "epoch", "type": "ordinal", "label": "Epoch"},
                        "y": {"field": "success", "type": "quantitative", "label": "Success (%)"},
                        "color": {"field": "run", "type": "nominal", "label": "Run"},
                    },
                    "yAxisTitle": "Success (%)",
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "early_epoch15_table",
                    "title": "Epoch-15 recursive validation comparison",
                    "dataset": "early_epoch15",
                    "sourceId": "baseline15_validation",
                    "defaultSort": {"field": "success", "direction": "desc"},
                    "columns": [
                        {"field": "run", "label": "Run", "type": "text"},
                        {"field": "success", "label": "Success (%)", "format": "number"},
                        {"field": "precision", "label": "Precision (%)", "format": "number"},
                        {"field": "success_delta_vs_current_b0", "label": "Δ Success vs current B0 (pp)", "format": "number", "movement": True},
                        {"field": "precision_delta_vs_current_b0", "label": "Δ Precision vs current B0 (pp)", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "early_b0_loss_table",
                    "title": "Epoch-15 B0 training-loss comparison",
                    "dataset": "early_b0_losses",
                    "sourceId": "baseline15_losses",
                    "defaultSort": {"field": "center_aux", "direction": "asc"},
                    "columns": [
                        {"field": "run", "label": "Run", "type": "text"},
                        {"field": "center", "label": "Center loss", "format": "number"},
                        {"field": "center_aux", "label": "Observation center loss", "format": "number"},
                        {"field": "center_ref", "label": "Reference center loss", "format": "number"},
                        {"field": "seg", "label": "Segmentation loss", "format": "number"},
                    ],
                },
                {
                    "id": "final_table",
                    "title": "Epoch-60 standard validation comparison",
                    "dataset": "final_metrics",
                    "sourceId": "validation_metrics",
                    "defaultSort": {"field": "success", "direction": "desc"},
                    "columns": [
                        {"field": "run", "label": "Run", "type": "text"},
                        {"field": "success", "label": "Success (%)", "format": "number"},
                        {"field": "precision", "label": "Precision (%)", "format": "number"},
                        {"field": "success_delta_vs_seqtrack", "label": "Δ Success vs SeqTrack (pp)", "format": "number", "movement": True},
                        {"field": "precision_delta_vs_seqtrack", "label": "Δ Precision vs SeqTrack (pp)", "format": "number", "movement": True},
                        {"field": "commit", "label": "Commit", "type": "text"},
                    ],
                },
                {
                    "id": "curve_table",
                    "title": "Best and late-training stability",
                    "dataset": "curve_summary",
                    "sourceId": "validation_metrics",
                    "defaultSort": {"field": "best_success", "direction": "desc"},
                    "columns": [
                        {"field": "run", "label": "Run", "type": "text"},
                        {"field": "best_success", "label": "Best Success", "format": "number"},
                        {"field": "best_precision", "label": "Best Precision", "format": "number"},
                        {"field": "last3_success", "label": "Last-3 Success mean", "format": "number"},
                        {"field": "last3_precision", "label": "Last-3 Precision mean", "format": "number"},
                    ],
                },
                {
                    "id": "mechanism_table",
                    "title": "Mechanism acceptance checks",
                    "dataset": "mechanism_checks",
                    "sourceId": "training_diagnostics",
                    "defaultSort": {"field": "component", "direction": "asc"},
                    "columns": [
                        {"field": "component", "label": "Component", "type": "text"},
                        {"field": "metric", "label": "Metric", "type": "text"},
                        {"field": "value", "label": "Value", "format": "number"},
                        {"field": "reference", "label": "Reference", "format": "number"},
                        {"field": "delta", "label": "Delta", "format": "number", "movement": True},
                        {"field": "interpretation", "label": "Interpretation", "type": "text"},
                    ],
                },
                {
                    "id": "b0_loss_table",
                    "title": "Epoch-60 B0 training losses",
                    "dataset": "b0_losses",
                    "sourceId": "training_diagnostics",
                    "defaultSort": {"field": "center_aux", "direction": "asc"},
                    "columns": [
                        {"field": "run", "label": "Run", "type": "text"},
                        {"field": "center", "label": "Center loss", "format": "number"},
                        {"field": "center_aux", "label": "Observation center loss", "format": "number"},
                        {"field": "center_ref", "label": "Reference center loss", "format": "number"},
                        {"field": "seg", "label": "Segmentation loss", "format": "number"},
                    ],
                },
                {
                    "id": "provenance_table",
                    "title": "Run comparability and provenance",
                    "dataset": "provenance",
                    "sourceId": "run_provenance",
                    "defaultSort": {"field": "run", "direction": "asc"},
                    "columns": [
                        {"field": "run", "label": "Run", "type": "text"},
                        {"field": "commit", "label": "Commit", "type": "text"},
                        {"field": "clean", "label": "Clean", "type": "boolean"},
                        {"field": "seed", "label": "Seed", "format": "number"},
                        {"field": "train_frames", "label": "Train frames", "format": "number"},
                        {"field": "val_frames", "label": "Val frames", "format": "number"},
                        {"field": "train_selection", "label": "Train selection", "type": "text"},
                        {"field": "val_selection", "label": "Val selection", "type": "text"},
                    ],
                },
                {
                    "id": "next_table",
                    "title": "Minimum next-experiment sequence",
                    "dataset": "next_experiments",
                    "sourceId": "analysis_code",
                    "defaultSort": {"field": "priority", "direction": "asc"},
                    "columns": [
                        {"field": "priority", "label": "Priority", "type": "text"},
                        {"field": "experiment", "label": "Experiment", "type": "text"},
                        {"field": "question", "label": "Question", "type": "text"},
                        {"field": "decision", "label": "Decision gate", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# CT-SeqTrack Joint Full 消融与15轮基线诊断\n\nnuScenes v1.0-mini · Car · seed42 · 2026-08-06"},
                {"id": "verdict", "type": "markdown", "sourceId": "baseline15_validation", "body": "## 结论：纯 B0 没有从一开始崩坏，退化由 Joint 公共路径引入\n\n当前 commit 的纯 B0 在 epoch 15 达到 **46.76 Success / 53.65 Precision**，分别比同 epoch Full 高 **20.38 / 26.99 个百分点**，比 −B2 高 **16.77 / 21.54 个百分点**。因此可以排除“当前代码的基础 SeqTrack 从初始化起整体坏掉”。但 −B2 已绕过 B2/B3 最终校正仍然低分，所以退化也不能归因于某一个 B1→B2→B3 传输节点；它位于所有 Joint 变体共享的训练/推理基础设施，并在递归验证中被放大。"},
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["current_b0_early", "full_success", "full_precision", "best_ablation", "raw_ratio"]},
                {"id": "early_onset", "type": "markdown", "sourceId": "baseline15_validation", "body": "## 分叉发生在 epoch 5–10，而不是初始化阶段\n\nEpoch 5 时，当前 B0、Full 和 −B2 的 Success 分别为 **29.39、29.29、32.81**，尚处在同一量级。到 epoch 10，纯 B0 上升至 **41.79/55.66**，Full 和 −B2 却降至 **25.03/25.16** 与 **24.54/22.18**。这证明 Joint 训练路径在早期若干轮后进入了对 teacher-forced 损失仍友好、对闭环递归跟踪不友好的参数区域。"},
                {"id": "early_success_block", "type": "chart", "chartId": "early_success_chart", "layout": "full"},
                {"id": "early_epoch15_block", "type": "table", "tableId": "early_epoch15_table", "layout": "full"},
                {"id": "loss_mismatch", "type": "markdown", "sourceId": "baseline15_losses", "body": "## 单步训练损失看不见这次崩坏\n\nEpoch 15 的 observation center loss 为：当前 B0 **0.04577**、历史 B0 **0.04534**、Full **0.04389**、−B2 **0.04268**。Joint 的单步 B0 损失甚至略低，但递归验证低二十多个百分点。这将故障位置进一步缩小到闭环鲁棒性、Joint 公共随机流/优化轨迹，或训练—推理历史契约，而不是普通监督回归能力。"},
                {"id": "early_loss_block", "type": "table", "tableId": "early_b0_loss_table", "layout": "full"},
                {"id": "final_results", "type": "table", "tableId": "final_table", "layout": "full"},
                {"id": "curve_intro", "type": "markdown", "sourceId": "validation_metrics", "body": "## 不是最后一个 checkpoint 选坏了\n\n即使取每组各自最好的验证点，Full 最好 Success 仅 **29.86**，−B2 最好 **33.36**，仍远低于 SeqTrack 的 **54.13**。末三次验证均值也保持同样排序，所以差距不是单个 epoch 抖动造成。"},
                {"id": "success_curve_block", "type": "chart", "chartId": "success_curve", "layout": "full"},
                {"id": "curve_table_block", "type": "table", "tableId": "curve_table", "layout": "full"},
                {"id": "architecture_read", "type": "markdown", "sourceId": "training_diagnostics", "body": "## 模块读数：B1 没爆炸，B2/B3 没把内部信号变成跟踪收益\n\nB1 将共享运动锚点 RMSE 从 **0.313 m** 降到 **0.291 m**，残差边界饱和率仅 **0.19%**，说明共享锚点与动态边界在训练样本内是稳定的。可是 raw Search RMSE 为 **0.546 m**，是 observation 的 **2.21 倍**；B1 虽把 −B1 的 raw Search 误差从 **0.583 m** 降到 **0.546 m**，Full 与 −B1 的最终验证却基本持平，说明有效的 B1 信号没有被可靠地转化为匹配和最终框收益。"},
                {"id": "rmse_chart_block", "type": "chart", "chartId": "rmse_chart", "layout": "full"},
                {"id": "mechanism_table_block", "type": "table", "tableId": "mechanism_table", "layout": "full"},
                {"id": "router_finding", "type": "markdown", "sourceId": "training_diagnostics", "body": "## 可靠性门控没有学成“可靠性”\n\nFull 的 query gate 均值从 0.05 初始化升到 **0.424**，但 query-gate BCE 仍为 **0.691**，接近随机二分类的 log(2)=0.693。B3 相比 −B3 明显更安全，但 Full 仍输给完全关闭 Search 的 −B2；现有门控只能减少损害，没有产生净收益。"},
                {"id": "router_label_cause", "type": "markdown", "sourceId": "loss_code", "body": "### 门控标签为何趋向模糊\n\n当前 query 软标签是 `sigmoid((kinematic_error - learned_error) / 0.25)`。当 B1 只比运动锚点好几厘米时，标签自然聚集在 0.5 附近，BCE 会鼓励一个接近常数的门，而不是清晰区分“应使用/不应使用 B1”。"},
                {"id": "b0_contract", "type": "markdown", "sourceId": "training_diagnostics", "body": "## 首要阻断项是 matched B0 没有复现\n\n−B2 按定义严格输出 observation，却只有 **30.51 / 30.93**。与此同时，epoch-60 的 B0 observation center loss（−B2 0.02038；历史 SeqTrack 0.02086）以及 center/ref/seg 损失并未恶化。这种“teacher-forced 训练损失正常、递归验证大幅坍塌”的组合，更符合当前 commit 下的递归推理契约、随机流或训练—推理历史分布失配，而不是 B0 根本没学会。"},
                {"id": "b0_loss_block", "type": "table", "tableId": "b0_loss_table", "layout": "full"},
                {"id": "code_findings", "type": "markdown", "sourceId": "query_code", "body": "## 已定位的查询回退问题\n\n`alpha_q=0` 时实现仍会再经过一个带可学习仿射参数的 LayerNorm，因此训练后不能保证 `q_search=q_obs`。不过当前 B2 匹配分数实际使用 `observation_score + alpha_q * residual_score`，所以这个问题违反了严格回退契约，但不太可能独自解释全部分数坍塌。"},
                {"id": "query_fallback_measurement", "type": "markdown", "sourceId": "training_diagnostics", "body": "### 回退违约已经出现在训练指标中\n\n−B1 的 `alpha_q` 被强制为 0，但 epoch-60 的 query shift norm 仍为 **0.110**，与“关闭 B1 时 q_search 严格等于 q_obs”的验收条件不一致。"},
                {"id": "candidate_valid_issue", "type": "markdown", "sourceId": "loss_code", "body": "### Candidate-valid 与候选可信度混在一起\n\n当前 candidate-valid 主要表达点集可计算性，router 与 correction 又直接使用它作为有效掩码。这样“存在有限点”会被当作“Search 候选值得校正”，却没有要求足够的目标证据或经过校准的候选置信度。"},
                {"id": "comparability", "type": "markdown", "sourceId": "baseline15_provenance", "body": "## 可比性与证据边界\n\n当前 B0 与四个 Joint 实验均来自 commit `835f911`，使用相同 seed、mini_train/mini_val、batch size、5051/2285 帧及相同 selection hash；前15轮记录的学习率也均保持 `1e-4`，因此早期分叉可直接比较。历史 SeqTrack 来自较早 commit `d86990c`，只作为外部参考。当前 B0 尚未跑到60轮，所以它足以否定“全局 B0 初始崩坏”，但还不能替代正式 epoch-60 matched baseline。"},
                {"id": "provenance_block", "type": "table", "tableId": "provenance_table", "layout": "full"},
                {"id": "next_steps_intro", "type": "markdown", "body": "## 下一轮：不要立刻重训，先做同权重交叉推理\n\n最有信息量的下一步不是继续60轮，而是把同一份 B0 权重分别放进 baseline 与 −B2 推理路径。如果两条路径输出相同且都低，问题在 Joint 训练/RNG轨迹；如果同权重在 baseline 路径恢复、在 −B2 路径下降，问题就在递归输入构造或推理传输。随后再做100-step参数等价审计，定位第一处偏离。"},
                {"id": "next_steps_block", "type": "table", "tableId": "next_table", "layout": "full"},
                {"id": "limitations", "type": "markdown", "body": "## 局限与尚未解决的问题\n\n当前证据只有 nuScenes mini Car、单 seed；当前纯 B0 只运行15轮，没有正式60轮终值或 tracklet-level paired bootstrap。现有数据能确定故障不是“基础模型从初始化起完全坏掉”，也能排除 B2/B3 最终校正是唯一原因，但尚不能区分三种共同路径因素：额外模块消耗随机数导致 B0 dropout/优化轨迹变化、Joint 数据构造意外改变 B0 输入、或同权重下的递归推理分支不等价。"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
            "datasets": {
                "headline": [{
                    "full_success": f(final_by_run["Full"]["success"]),
                    "full_success_delta": f(final_by_run["Full"]["success"]) - f(final_by_run["SeqTrack"]["success"]),
                    "full_precision": f(final_by_run["Full"]["precision"]),
                    "full_precision_delta": f(final_by_run["Full"]["precision"]) - f(final_by_run["SeqTrack"]["precision"]),
                    "minus_b2_success": f(final_by_run["-B2"]["success"]),
                    "full_vs_minus_b2_success": f(final_by_run["Full"]["success"]) - f(final_by_run["-B2"]["success"]),
                    "raw_obs_ratio": full_raw / full_obs,
                    "current_b0_e15_success": next(
                        f(row["success"]) for row in baseline15_validation
                        if row["run"] == "Current B0-15"
                        and int(row["epoch"]) == 15),
                    "current_b0_vs_minus_b2_e15": (
                        next(f(row["success"])
                             for row in baseline15_validation
                             if row["run"] == "Current B0-15"
                             and int(row["epoch"]) == 15)
                        - next(f(row["success"])
                               for row in baseline15_validation
                               if row["run"] == "-B2-60"
                               and int(row["epoch"]) == 15)),
                }],
                "final_metrics": final_metrics,
                "curve_summary": curve_summary,
                "validation_curve": validation_rows,
                "rmse": rmse_rows,
                "mechanism_checks": mechanism_checks,
                "b0_losses": b0_loss_rows,
                "provenance": provenance_rows,
                "next_experiments": next_experiments,
                "early_validation": early_validation_rows,
                "early_epoch15": early_epoch15_rows,
                "early_b0_losses": early_b0_loss_rows,
            },
        },
        "sources": sources,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "joint_full_ablation_diagnosis_20260806_artifact.json"
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
