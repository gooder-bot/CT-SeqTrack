#!/usr/bin/env python3
"""Build the canonical portable-report artifact for the B2-v2 diagnosis."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "compare_results" / "data"
REPORTS = ROOT / "compare_results" / "reports"
STEM = "b2_search_v2_seed42_20260802"


def rows(name: str) -> list[dict[str, Any]]:
    path = DATA / f"{STEM}_{name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        output = list(csv.DictReader(handle))
    for row in output:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
                continue
            if key in {"run_id", "arm", "role", "treatment", "baseline", "comparison_basis", "question", "criterion", "observed", "decision", "finding", "evidence", "interpretation", "confidence", "experiment", "change", "gpu_cost", "checkpoint"}:
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
        sql: str | None = None,
        definitions: list[str] | None = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "language": "python",
        "description": description,
        "executed_at": "2026-08-02",
    }
    if sql:
        query.update({"engine": "DuckDB", "sql": sql})
    if definitions:
        query["metric_definitions"] = definitions
    return {"id": source_id, "label": label, "path": path, "query": query}


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    validation = rows("validation")
    summary = rows("summary")
    comparisons = rows("comparisons")
    training = rows("full_training_epochs")
    control = rows("control_stability_audit")
    checkpoint = rows("checkpoint_diagnostics")
    drivers = rows("drivers")
    decisions = rows("decision")
    next_steps = rows("next_experiments")

    short_names = {
        "SEQTRACK_NEW": "New control (anomalous)",
        "SEARCH_LEGACY": "Legacy search-only",
        "B2V2_FULL": "B2-v2 full",
        "SEQTRACK_HIST": "Historical SeqTrack",
        "B0_HIST": "Historical B0",
    }
    validation_chart = []
    for row in validation:
        if row["run_id"] not in short_names:
            continue
        item = dict(row)
        item["series"] = short_names[row["run_id"]]
        validation_chart.append(item)

    summary_visible = [r for r in summary if r["run_id"] in short_names]
    e60 = training[-1]
    motion_e60 = []
    for branch, learned_key, cv_key in (
        ("candidate0 main", "motion_candidate0_prior_rmse", "motion_candidate0_cv_rmse"),
        ("nonzero candidates", "motion_nonzero_prior_rmse", "motion_nonzero_cv_rmse"),
        ("gap2 auxiliary", "motion_gap2_prior_rmse", "motion_gap2_cv_rmse"),
        ("gap4 auxiliary", "motion_gap4_prior_rmse", "motion_gap4_cv_rmse"),
    ):
        learned = float(e60[learned_key])
        cv = float(e60[cv_key])
        motion_e60.append({
            "branch": branch,
            "learned_rmse_m": learned,
            "constant_velocity_rmse_m": cv,
            "improvement_pct": 100.0 * (cv - learned) / cv,
        })
    gate_epochs = [r for r in training if r["epoch"] in {20, 30, 40, 50, 60}]

    sources = [
        source(
            "validation_source",
            "Reviewed normal-validation TensorBoard scalars",
            f"compare_results/data/{STEM}_validation.csv",
            "tools/analyze_b2_search_v2.py extracts all twelve unsmoothed validation points for the three requested runs and historical references.",
            f"SELECT * FROM read_csv_auto('compare_results/data/{STEM}_validation.csv', header=true) ORDER BY run_id, epoch",
            [
                "Final is epoch60 and is the primary decision point.",
                "Late-3 is the arithmetic mean of epochs 50, 55, and 60.",
                "Best checkpoints are diagnostic only.",
            ],
        ),
        source(
            "summary_source",
            "Reviewed run summary and comparison register",
            f"compare_results/data/{STEM}_summary.csv",
            "Final, late-3, and best values are recomputed from the twelve paired validation points; pairwise deltas are stored in the companion comparisons CSV.",
            f"SELECT * FROM read_csv_auto('compare_results/data/{STEM}_summary.csv', header=true)",
        ),
        source(
            "training_source",
            "B2-v2 epoch-level training diagnostics",
            f"compare_results/data/{STEM}_full_training_epochs.csv",
            "Epoch means are regenerated from 75,720 TensorBoard training steps using 1,262 full batches per epoch.",
            f"SELECT * FROM read_csv_auto('compare_results/data/{STEM}_full_training_epochs.csv', header=true) ORDER BY epoch",
        ),
        source(
            "driver_source",
            "B2-v2 diagnostic evidence register",
            f"compare_results/data/{STEM}_drivers.csv",
            "Evidence register derived from reviewed validation, training, checkpoint, sampler, and fusion sources.",
            f"SELECT * FROM read_csv_auto('compare_results/data/{STEM}_drivers.csv', header=true) ORDER BY priority",
        ),
        source(
            "control_source",
            "SeqTrack control stability audit",
            f"compare_results/data/{STEM}_control_stability_audit.csv",
            "Compares exact early loss_total steps between the new anomalous control and the historical original SeqTrack run.",
            f"SELECT * FROM read_csv_auto('compare_results/data/{STEM}_control_stability_audit.csv', header=true) ORDER BY step",
        ),
        source(
            "decision_source",
            "B2-v2 preregistered decision checks",
            f"compare_results/data/{STEM}_decision.csv",
            "Applies the frozen epoch60 and late-3 gates without substituting best checkpoints.",
            f"SELECT * FROM read_csv_auto('compare_results/data/{STEM}_decision.csv', header=true)",
        ),
        source(
            "next_source",
            "Controlled B2-v2 follow-up register",
            f"compare_results/data/{STEM}_next_experiments.csv",
            "Orders inference-only attribution before a matched B0 and any B2-v2.1 training.",
            f"SELECT * FROM read_csv_auto('compare_results/data/{STEM}_next_experiments.csv', header=true) ORDER BY \"order\"",
        ),
        source(
            "code_source",
            "B2-v2 implementation audit",
            "models/ct_v2/motion.py",
            "Code review covers models/ct_v2/motion.py, models/seqtrack3d.py, datasets/sampler.py, utils/ct_search.py, and cfgs/ct_v2/03_ct_motion_search_v2.yaml.",
        ),
    ]

    charts = [
        {
            "id": "success_curve",
            "title": "Normal validation Success by epoch",
            "subtitle": "Twelve unsmoothed checkpoints; epoch60 is primary and late-3 covers epochs 50/55/60.",
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
            "subtitle": "B2-v2 full stays above the historical B0 at all three late checkpoints but misses the final +1.0 gate.",
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
    ]

    tables = [
        {
            "id": "summary_table",
            "title": "Normal-validation result summary",
            "subtitle": "Requested runs plus historical references; final and late-3 are the decision metrics.",
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
            ],
        },
        {
            "id": "decision_table",
            "title": "Preregistered decision checks",
            "subtitle": "Historical B0 is the strongest available guardrail, but it is not a same-commit matched baseline.",
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
            "id": "motion_table",
            "title": "Epoch-60 physical prior versus constant velocity",
            "subtitle": "Training RMSE in metres; all four branches favor the learned prior.",
            "dataset": "motion_e60",
            "sourceId": "training_source",
            "density": "compact",
            "defaultSort": {"field": "improvement_pct", "direction": "desc"},
            "columns": [
                {"field": "branch", "label": "Branch", "type": "text"},
                {"field": "learned_rmse_m", "label": "Learned RMSE", "type": "number", "format": "number"},
                {"field": "constant_velocity_rmse_m", "label": "CV RMSE", "type": "number", "format": "number"},
                {"field": "improvement_pct", "label": "Improvement %", "type": "number", "format": "number", "semantic": "movement"},
            ],
        },
        {
            "id": "gate_table",
            "title": "Search availability and gate use after warmup",
            "subtitle": "Epoch means; selected rate is the fraction whose applied gate argmax is search.",
            "dataset": "gate_epochs",
            "sourceId": "training_source",
            "density": "compact",
            "defaultSort": {"field": "epoch", "direction": "asc"},
            "columns": [
                {"field": "epoch", "label": "Epoch", "type": "number", "format": "number"},
                {"field": "search_candidate_valid_rate", "label": "Search valid rate", "type": "number", "format": "percent"},
                {"field": "search_confidence_mean", "label": "Mean confidence", "type": "number", "format": "number"},
                {"field": "joint_search_selected_rate", "label": "Search selected rate", "type": "number", "format": "percent"},
            ],
        },
        {
            "id": "driver_table",
            "title": "Root-cause evidence register",
            "subtitle": "Findings are diagnostic; module causality still requires same-checkpoint inference.",
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
            "id": "next_table",
            "title": "Controlled follow-up sequence",
            "subtitle": "Inference attribution precedes any new architecture training.",
            "dataset": "next_steps",
            "sourceId": "next_source",
            "density": "spacious",
            "defaultSort": {"field": "order", "direction": "asc"},
            "columns": [
                {"field": "order", "label": "Order", "type": "number", "format": "number"},
                {"field": "experiment", "label": "Experiment", "type": "text"},
                {"field": "change", "label": "What changes", "type": "text"},
                {"field": "decision", "label": "Decision unlocked", "type": "text"},
                {"field": "gpu_cost", "label": "GPU cost", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# B2-v2 seed42：Search Evidence 与联合融合技术复核"},
        {"id": "technical_summary", "type": "markdown", "sourceId": "summary_source", "body": "## 技术结论：有正信号，但未达到正式晋级条件\n\nB2-v2 full epoch60 为 **54.132 Success / 64.755 Precision**。相对最强可用历史 B0 为 **+0.772 / +0.373**：Success 通过 +0.5，Precision 未达到 +1.0。late-3 为 **+1.557 / +2.909**，说明后期信号稳定；但同 commit matched B0 缺失，正式状态是 **HOLD_B2V2_PROMOTION**。旧 search-only 相对历史 B0 为 **−3.705 / −7.990**，继续 No-Go，而且它是 legacy tube/75–25 设计，不是新版 Search Evidence-only 消融。"},
        {"id": "summary_block", "type": "table", "tableId": "summary_table"},
        {"id": "decision_finding", "type": "markdown", "sourceId": "decision_source", "body": "## full 只部分通过 epoch60 门槛\n\n严格判定不能采用异常低的新 control，也不能用 best checkpoint 替换 final。历史 B0 是当前最强 guardrail：full 的 Success 通过，Precision 还差 **0.627 分**才达到 +1.0 门槛；late-3 两项均为正，所以应保留归因，而不是直接废弃。"},
        {"id": "decision_block", "type": "table", "tableId": "decision_table"},
        {"id": "curve_finding", "type": "markdown", "sourceId": "validation_source", "body": "## full 的 late-3 正信号稳定，旧 search-only 始终落后 B0\n\n两条曲线使用全部 12 个无平滑验证点。重点看 epoch50/55/60：B2-v2 full 在这三个时点的 Success 和 Precision 都高于历史 B0；旧 search-only 的后期两项仍持续落后。新 control 全程位于约 30 分，显示它是异常训练轨迹而非可信 matched baseline。"},
        {"id": "success_block", "type": "chart", "chartId": "success_curve"},
        {"id": "precision_block", "type": "chart", "chartId": "precision_curve"},
        {"id": "control_finding", "type": "markdown", "sourceId": "control_source", "body": "## 新 SeqTrack control 不能承担方法归因\n\n新 control 完整且有限值，但 final 只有 **31.684/31.337**。它和历史 SeqTrack 的前 3 个 batch loss 逐 float 相等，从 step3 开始分叉，支持“同初始化后随机训练流被递归评测放大”的解释。该 run 没有 run_provenance.json，且位于独立 dirty 仓库；因此 **full−control 的 +22.449/+33.418 不能写成 B2 净增益**。"},
        {"id": "search_finding", "type": "markdown", "sourceId": "driver_source", "body": "## Search 的首要瓶颈是证据饥饿与 gate 双重抑制\n\nendpoint crop 经过 extension-only 删除和去重后，epoch60 只有 **23.29%** 样本形成有效 search candidate；约 **76.71%** 样本完全没有候选。有效候选还要同时克服 observation bias 和 `log(confidence)` 惩罚，最终 search argmax 率只有 **0.104%**。这说明当前 gate 学会了避免风险，却没有学会在少数有用场景消费搜索证据。"},
        {"id": "gate_block", "type": "table", "tableId": "gate_table"},
        {"id": "motion_finding", "type": "markdown", "sourceId": "training_source", "body": "## Physical prior 学到了，正信号更可能来自 motion 与保守融合\n\nlearned prior 在 candidate0、非零 candidate、gap2、gap4 上分别比 constant velocity 降低 RMSE **6.60% / 12.81% / 16.83% / 21.58%**。epoch60 一步训练 xy error 从 observation 的 **0.247 m** 降到 joint final 的 **0.236 m**。这支持存在有用 correction signal，但不证明它能稳定迁移到 recursive tracking。"},
        {"id": "motion_block", "type": "table", "tableId": "motion_table"},
        {"id": "drivers_finding", "type": "markdown", "sourceId": "driver_source", "body": "## 当前证据不支持把 full 的增益归因给 Search Evidence\n\nsearch auxiliary loss 收敛只能说明有效 crop 上的局部监督可学；candidate 覆盖率和实际 gate 使用率决定它几乎不影响最终 proposal。当前可辩护的表述是“B2-v2 full 出现正信号，motion prior 学习有效，search 贡献尚未建立”。"},
        {"id": "drivers_block", "type": "table", "tableId": "driver_table"},
        {"id": "scope", "type": "markdown", "sourceId": "validation_source", "body": "## 范围、数据与指标定义\n\n范围是 nuScenes v1.0-mini Car normal cadence、seed42、scratch 60 epoch。训练集 5051 frames/274 tracklets，验证集 2285 frames/106 tracklets；batch16、workers4、candidate4，每 5 epoch 验证。主指标是 epoch60 Success/Precision；late-3 是 epoch50/55/60 算术平均；best 只作诊断。"},
        {"id": "methodology", "type": "markdown", "sourceId": "code_source", "body": "## 方法与实现核验\n\n分析从 TensorBoard 原始 scalar 重新抽取 12 个验证点和 75,720 个训练 step，并审计 sampler、Search Evidence encoder、loss 与 JointProposalFusion。B2-v2 保留 B0 的 1024 点，使用独立 128 点 endpoint branch；legacy search-only 则在触发时使用 75/25 token 分配。两者不是同一搜索模块，不能做差获得净贡献。"},
        {"id": "limitations", "type": "markdown", "body": "## 限制、稳健性与诊断盲区\n\n本轮只有一个 seed，没有 same-commit matched B0、endpoint 级 paired bootstrap 或置信区间。`joint_search_error` 未 mask invalid 行；训练日志中的 helpful precision 是分母 clamp 后的 batch mean，不等于 validation endpoint precision。两个 B2 run 还并行占用同一 GPU2，因此 FPS 不能用于 runtime 晋级判定。以上限制阻止因果模块归因，但不改变 legacy search 的负结果。"},
        {"id": "next_finding", "type": "markdown", "sourceId": "next_source", "body": "## 下一步先做无重训归因，再补 matched B0\n\n优先用 full epoch60 同 checkpoint 跑 full、observation-only、motion-only、search-only 四模式并导出 endpoint 诊断；随后补 commit-a486a36 matched B0。只有 final 同时满足 +0.5/+1.0 且 late-3 不退化，才进入 seed43/44。若 search 被确认无贡献，再以 endpoint 全点 + overlap flag、availability/utility 分离和去除未校准 `log(confidence)` 惩罚做一次 bounded kill-test。"},
        {"id": "next_block", "type": "table", "tableId": "next_table"},
        {"id": "questions", "type": "markdown", "body": "## 仍需回答的问题\n\n- valid search 行中，proposal 比 observation 好 0.05 m 的真实比例是多少？\n- 禁用 search 后 full checkpoint 是否保持指标？\n- `_extension_only` 是否删掉了 endpoint crop 中最稳定的目标表面点？\n- same-commit B0 能否复现历史 53.360/64.382？"},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "B2-v2 seed42：Search Evidence 与联合融合技术复核",
            "description": "Normal-mini outcome, baseline comparability, component attribution, failure drivers, and controlled next steps.",
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
                "comparisons": comparisons,
                "motion_e60": motion_e60,
                "gate_epochs": gate_epochs,
                "control_early": control[:8],
                "checkpoint": checkpoint,
                "drivers": drivers,
                "decisions": decisions,
                "next_steps": next_steps,
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
        "delivery_mode": "html",
        "required_structure_mapping": {
            "technical_summary": "technical_summary",
            "key_findings_with_visual_evidence": ["decision_finding", "curve_finding", "search_finding", "motion_finding"],
            "scope_data_metric_definitions": "scope",
            "methodology": "methodology",
            "limitations_uncertainty_robustness": "limitations",
            "recommended_next_steps": "next_finding",
            "further_questions": "questions",
        },
        "chart_map": [
            {
                "section": "late-stage validation",
                "question": "Does full retain its advantage through epoch60?",
                "family": "trend",
                "type": "line",
                "fields": ["epoch", "success", "series"],
                "takeaway": "full remains above historical B0 at epochs 50/55/60",
                "palette": "relaxed multi-category native report palette; legend plus line geometry prevents color-only interpretation",
                "delivery": f"compare_results/reports/{STEM}.html",
            },
            {
                "section": "late-stage validation",
                "question": "Does full pass the final Precision gate and stay stable?",
                "family": "trend",
                "type": "line",
                "fields": ["epoch", "precision", "series"],
                "takeaway": "late-3 is positive but final improvement misses +1.0",
                "palette": "same series mapping as Success for cross-chart consistency",
                "delivery": f"compare_results/reports/{STEM}.html",
            },
        ],
        "visual_omissions": [
            "Search gate diagnostics use a table because only five selected epoch anchors are needed for exact low-rate lookup.",
            "Motion prior diagnostics use a table because exact four-branch RMSE comparisons are the audit target.",
        ],
        "source_notes": [
            "New SeqTrack control lacks run_provenance.json and comes from a separate dirty working tree.",
            "Legacy search-only is not a new Search Evidence-only ablation.",
            "Runtime is omitted because both B2 runs contended on physical GPU2.",
        ],
    }
    (DATA / f"{STEM}_report_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(artifact_path)


if __name__ == "__main__":
    main()
