"""Append uncertainty/path diagnosis and CfC placement evidence to the report.

Only derived artifacts are written.  Experiment outputs and historical Git
objects are read-only evidence.
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
BASE_DIR = ROOT / "artifacts/ct_checks/reports/20260825_b1_learning_diagnosis"
REPORT_DIR = ROOT / "artifacts/ct_checks/reports/20260825_uncertainty_path_cfc_integration"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_RUNS = {
    "B1 retryfix epoch30": ROOT
    / "output/20260824-0220-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_retryfix",
    "B1 rerun epoch30": ROOT
    / "output/20260825-0057-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_rerun_20260825",
}

CFC_RUNS = {
    "Plain GRU": ROOT
    / "output/20260819-1401-b1_gru_mini_seed42-ct25_b1_gru_mini_seed42_60ep_bs16_20260819-140102",
    "Plain CfC": ROOT
    / "output/20260819-1401-b1_cfc_mini_seed42-ct25_b1_cfc_mini_seed42_60ep_bs16_20260819-140108",
    "RA-PMM GRU": ROOT
    / "output/20260819-2043-b1_gru_mini_seed42-ct25_b1_ra_pmm_gru_mini_seed42_60ep_bs16_fixed_20260819-204347",
    "RA-PMM CfC": ROOT
    / "output/20260819-2043-b1_cfc_mini_seed42-ct25_b1_ra_pmm_cfc_mini_seed42_60ep_bs16_fixed_20260819-204347",
}


def valid_frame(run: Path, epoch: int) -> pd.DataFrame:
    frame = pd.read_csv(
        run / "lightning_logs/version_0/candidate_diagnostics" / f"epoch_{epoch:02d}.csv"
    )
    valid = (
        (frame["b1_valid"] > 0)
        & np.isfinite(frame["learned_motion_error"])
        & np.isfinite(frame["kinematic_error"])
    )
    return frame.loc[valid].copy()


def scalar_tail(run: Path, directory: str, count: int = 213) -> float:
    accumulator = EventAccumulator(
        str(run / "lightning_logs/version_0" / directory),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    values = [row.value for row in accumulator.Scalars("loss")]
    return float(np.mean(values[-count:]))


current_rows = []
for name, run in CURRENT_RUNS.items():
    frame = valid_frame(run, 30)
    learned_rmse = float(np.sqrt(np.mean(frame["learned_motion_error"] ** 2)))
    cv_rmse = float(np.sqrt(np.mean(frame["kinematic_error"] ** 2)))
    current_rows.append(
        {
            "run": name,
            "valid_rows": len(frame),
            "learned_minus_cv_rmse": learned_rmse - cv_rmse,
            "help_rate": float(
                (frame["learned_motion_error"] < frame["kinematic_error"]).mean()
            ),
            "coverage_95": float(frame["b1_coverage_95"].mean()),
            "sigma_parallel_p50": float(frame["sigma_parallel"].median()),
            "sigma_parallel_p95": float(frame["sigma_parallel"].quantile(0.95)),
            "sigma_perpendicular_p50": float(frame["sigma_perpendicular"].median()),
            "sigma_perpendicular_p95": float(
                frame["sigma_perpendicular"].quantile(0.95)
            ),
            "residual_parallel_p50": float(frame["residual_unit_parallel"].median()),
            "residual_parallel_p05": float(
                frame["residual_unit_parallel"].quantile(0.05)
            ),
            "residual_parallel_p95": float(
                frame["residual_unit_parallel"].quantile(0.95)
            ),
            "query_delta_t_mean": float(frame["query_delta_t"].mean()),
            "query_delta_t_std": float(frame["query_delta_t"].std()),
            "recursive_age_unique": int(frame["recursive_age"].nunique()),
            "gap2_learned_rmse": scalar_tail(
                run, "loss_motion_v3_aux_prior_rmse_gap2"
            ),
            "gap2_cv_rmse": scalar_tail(
                run, "loss_motion_v3_aux_kinematic_rmse_gap2"
            ),
            "gap4_learned_rmse": scalar_tail(
                run, "loss_motion_v3_aux_prior_rmse_gap4"
            ),
            "gap4_cv_rmse": scalar_tail(
                run, "loss_motion_v3_aux_kinematic_rmse_gap4"
            ),
        }
    )

cfc_rows = []
for name, run in CFC_RUNS.items():
    frame = valid_frame(run, 60)
    learned_rmse = float(np.sqrt(np.mean(frame["learned_motion_error"] ** 2)))
    cv_rmse = float(np.sqrt(np.mean(frame["kinematic_error"] ** 2)))
    cfc_rows.append(
        {
            "run": name,
            "family": "plain fixed-geometry" if name.startswith("Plain") else "RA-PMM multi-change",
            "backend": "CfC" if name.endswith("CfC") else "GRU",
            "valid_rows": len(frame),
            "learned_rmse": learned_rmse,
            "cv_rmse": cv_rmse,
            "learned_minus_cv_rmse": learned_rmse - cv_rmse,
            "mean_nll": float(frame["b1_nll"].mean()),
            "coverage_95": float(frame["b1_coverage_95"].mean()),
            "help_rate": float(
                (frame["learned_motion_error"] < frame["kinematic_error"]).mean()
            ),
            "query_delta_t_mean": float(frame["query_delta_t"].mean()),
            "query_delta_t_std": float(frame["query_delta_t"].std()),
            "interpretation": (
                "Pure backbone comparison, but old B0 path/protocol is not a formal v25 control."
                if name.startswith("Plain")
                else "Not a CfC-only comparison: uncertainty objective and dynamic geometry also changed."
            ),
        }
    )

issue_rows = [
    {
        "issue": "Effective uncertainty collapse",
        "status": "Verified in the current rerun",
        "included_before": "Yes: the earlier sigma/NLL/coverage failure is the aggregate symptom.",
        "evidence": (
            "Rerun sigma p50/p95 is 0.323/0.375 m parallel and 0.323/0.340 m perpendicular; "
            "95% coverage is 31.1%."
        ),
        "boundary": "Not literal zero variance; it is severe under-dispersion with weak sample adaptation.",
    },
    {
        "issue": "B1 endpoint/path inaccuracy",
        "status": "Verified for the endpoint; full path is not yet measured",
        "included_before": "Yes: learned-vs-CV RMSE and residual-head collapse already captured endpoint failure.",
        "evidence": (
            "Rerun learned RMSE is 0.107 m worse than CV and helps only 4.2% of rows; "
            "gap2/gap4 train-tail gains are only −0.003/+0.039 m relative to CV."
        ),
        "boundary": "The current contract predicts one xy endpoint, not a dense continuous trajectory.",
    },
    {
        "issue": "B0 recursive training-path divergence",
        "status": "Verified upstream confounder",
        "included_before": "Yes, but it is separate from B1 internal collapse.",
        "evidence": "Matched runs share input fingerprints yet B0 parameters diverge after step1.",
        "boundary": "It changes B1 inputs/targets and can trigger different B1 collapse directions.",
    },
    {
        "issue": "Recursive-age stratification",
        "status": "Not measurable with the current export",
        "included_before": "The plan requested it, but the current epoch30 CSV contains only age=0.",
        "evidence": "Both current B1 epoch30 exports have recursive_age_unique=1 and value 0.",
        "boundary": "Fix the diagnostic field before claiming robustness to recursive drift.",
    },
]

placement_rows = [
    {
        "location": "B1 temporal aggregation (recommended)",
        "decision": "GO after B1 mean/sigma repair",
        "change": "Replace only the GRU over chronological transition embeddings with a parameter-matched CfC cell.",
        "preserved": "B0, step projection, CV anchor, residual/sigma heads, fixed crop, B2/B3 contracts and observation-only writer.",
        "reason": "This is the only location where explicit pairwise delta-t is both available and causally relevant.",
    },
    {
        "location": "B0 backbone/decoder",
        "decision": "NO-GO",
        "change": "Would inject motion features into the Safe-SeqTrack observation baseline.",
        "preserved": "No",
        "reason": "Breaks B0 parity and late coupling; confounds the paper baseline.",
    },
    {
        "location": "B2 evidence network",
        "decision": "NO-GO for first integration",
        "change": "Would make CfC responsible for evidence supply rather than physical motion.",
        "preserved": "Extension-only could remain, but the attribution becomes unclear.",
        "reason": "B2 is bottlenecked by target-bearing supply, not temporal recurrence capacity.",
    },
    {
        "location": "B3 selective router",
        "decision": "NO-GO",
        "change": "Would add recurrence after detached evidence/action features.",
        "preserved": "Fail-closed semantics might remain, but calibration data are currently absent.",
        "reason": "B3 currently has zero deployment action coverage; CfC cannot create calibration evidence.",
    },
]

artifact = copy.deepcopy(
    json.loads((BASE_DIR / "artifact.json").read_text(encoding="utf-8"))
)
source_id = "src_uncertainty_path_cfc_20260825"
source = {
    "id": source_id,
    "label": "Current B1 diagnostics, historical matched GRU/CfC runs, and read-only Git implementation audit",
    "path": "artifacts/ct_checks/reports/20260825_uncertainty_path_cfc_integration/uncertainty_path_cfc.sqlite",
    "query": {
        "engine": "sqlite",
        "sql": "SELECT snapshot_json FROM cfc_diagnosis_snapshot WHERE snapshot_id = 'uncertainty_path_cfc_20260825'",
        "tablesUsed": ["cfc_diagnosis_snapshot"],
    },
}
manifest = artifact["manifest"]
manifest["title"] = "CT-SeqTrack 不确定度、路径预测与 CfC 融合诊断（2026-08-25）"
manifest["description"] = "在B1学习失败报告上追加不确定度塌陷、路径测量边界和CfC最佳融合位置。"
manifest["sources"].append(source)
artifact["sources"].append(source)

for block in manifest["blocks"]:
    if block.get("id") == "title":
        block["body"] = "# CT-SeqTrack 不确定度、路径预测与 CfC 融合诊断（2026-08-25）"
    if block.get("id") == "executive_summary":
        block["body"] += (
            "\n\n**本次追加：** 当前仍存在有效不确定度塌陷和B1 endpoint预测失效；"
            "二者属于此前mean/NLL/coverage问题的内部机制，但B0随机轨迹是独立上游混杂因素。"
            "CfC最契合的位置是B1内部的时间聚合器，不应进入B0、B2或B3。"
        )

datasets = artifact["snapshot"]["datasets"]
datasets["current_uncertainty_path"] = current_rows
datasets["historical_cfc_comparison"] = cfc_rows
datasets["issue_scope_map"] = issue_rows
datasets["cfc_placement_decision"] = placement_rows

manifest["tables"].extend(
    [
        {
            "id": "table_current_uncertainty_path",
            "title": "当前B1的sigma、endpoint与多时域辅助指标",
            "subtitle": "epoch30验证endpoint与训练末213条gap2/gap4标量；两者用途不同",
            "dataset": "current_uncertainty_path",
            "sourceId": source_id,
            "defaultSort": {"field": "run", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "learned_minus_cv_rmse", "label": "Endpoint learned−CV", "format": "number", "movement": True},
                {"field": "help_rate", "label": "Help rate", "format": "percent"},
                {"field": "coverage_95", "label": "95% coverage", "format": "percent"},
                {"field": "sigma_parallel_p50", "label": "sigma∥ p50", "format": "number"},
                {"field": "sigma_parallel_p95", "label": "sigma∥ p95", "format": "number"},
                {"field": "gap2_learned_rmse", "label": "gap2 learned", "format": "number"},
                {"field": "gap2_cv_rmse", "label": "gap2 CV", "format": "number"},
                {"field": "gap4_learned_rmse", "label": "gap4 learned", "format": "number"},
                {"field": "gap4_cv_rmse", "label": "gap4 CV", "format": "number"},
            ],
        },
        {
            "id": "table_issue_scope_map",
            "title": "这些问题与此前诊断的关系",
            "subtitle": "区分内部机制、上游混杂因素和当前不可测项",
            "dataset": "issue_scope_map",
            "sourceId": source_id,
            "defaultSort": {"field": "issue", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "issue", "label": "Issue", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "included_before", "label": "Included before", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "boundary", "label": "Boundary", "type": "text"},
            ],
        },
        {
            "id": "table_historical_cfc_comparison",
            "title": "历史GRU/CfC端点证据",
            "subtitle": "epoch60逐帧诊断；旧协议且B0路径未匹配，只能判断CfC未自动修复mean/sigma",
            "dataset": "historical_cfc_comparison",
            "sourceId": source_id,
            "defaultSort": {"field": "run", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "run", "label": "Run", "type": "text"},
                {"field": "valid_rows", "label": "Rows", "format": "number"},
                {"field": "learned_rmse", "label": "Learned RMSE", "format": "number"},
                {"field": "cv_rmse", "label": "CV RMSE", "format": "number"},
                {"field": "learned_minus_cv_rmse", "label": "Learned−CV", "format": "number", "movement": True},
                {"field": "mean_nll", "label": "NLL", "format": "number"},
                {"field": "coverage_95", "label": "95% coverage", "format": "percent"},
                {"field": "help_rate", "label": "Help rate", "format": "percent"},
            ],
        },
        {
            "id": "table_cfc_placement_decision",
            "title": "CfC融合位置决策",
            "subtitle": "以不破坏Safe B0、固定geometry和晚耦合为硬约束",
            "dataset": "cfc_placement_decision",
            "sourceId": source_id,
            "defaultSort": {"field": "location", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "location", "label": "Location", "type": "text"},
                {"field": "decision", "label": "Decision", "type": "text"},
                {"field": "change", "label": "Change", "type": "text"},
                {"field": "preserved", "label": "Preserved", "type": "text"},
                {"field": "reason", "label": "Reason", "type": "text"},
            ],
        },
    ]
)

manifest["blocks"].extend(
    [
        {
            "id": "uncertainty_path_summary_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## sigma仍然有效塌陷，endpoint也没有稳定学准\n\n"
                "这里的塌陷不是sigma严格等于零，而是输出集中在很窄且明显偏小的区间。"
                "新run的平行/垂直sigma中位数约0.323 m，95分位也只有0.375/0.340 m，"
                "对应95%经验coverage仅31.1%。同一run的learned endpoint比CV差0.107 m，"
                "逐帧只有4.2%更好。二者属于此前B1 mean/NLL/coverage失败的更细机制；"
                "B0从step1分叉则是独立上游混杂因素。"
            ),
        },
        {"id": "current_uncertainty_path_table_block", "type": "table", "layout": "full", "tableId": "table_current_uncertainty_path"},
        {"id": "issue_scope_table_block", "type": "table", "layout": "full", "tableId": "table_issue_scope_map"},
        {
            "id": "path_measurement_boundary_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## 当前所谓‘路径’实际上只是单个endpoint\n\n"
                "B1读取有序历史转移并输出下一时刻xy位移、方向与sigma；它没有导出连续轨迹点。"
                "gap2/gap4辅助训练可以测试不同预测时域，但现有两次run相对CV只出现毫米到厘米级且方向不一致的差值，"
                "不足以证明轨迹建模有效。另一个测量缺口是epoch30导出的recursive_age全部为0，"
                "当前无法按递归漂移年龄验证路径误差。"
            ),
        },
        {
            "id": "historical_cfc_result_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## 历史CfC没有自动解决均值或不确定度\n\n"
                "参数量匹配、固定geometry的旧对照中，CfC只比自身CV好0.019 m，但NLL为11988.8、"
                "95% coverage仅5.4%；GRU则比CV差0.014 m、coverage为4.9%。"
                "RA-PMM对照里CfC均值略好，但NLL和coverage反而差于GRU，而且该组同时改变了不确定度目标和动态geometry。"
                "这些旧run的B0路径/协议不满足当前正式控制，因此只能得出CfC不是sigma/NLL修复器，不能得出CfC优于GRU。"
            ),
        },
        {"id": "historical_cfc_table_block", "type": "table", "layout": "full", "tableId": "table_historical_cfc_comparison"},
        {
            "id": "cfc_best_placement_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## CfC最契合B1内部的时间聚合器\n\n"
                "推荐数据流是：历史预测框和真实pair gap → 9维转移特征 → step projection → "
                "按时间顺序的CfC隐状态更新 → query-gap context → CV anchor上的bounded residual mean和独立sigma head。"
                "CfC只替换GRU；无效转移必须是hidden-state精确no-op，elapsed time使用pair_gap/time_scale。"
                "历史实现用105个backbone units将CfC参数量匹配到GRU的0.1%以内，并由zero-init输出头保证cold start都精确等于CV。"
            ),
        },
        {"id": "cfc_placement_table_block", "type": "table", "layout": "full", "tableId": "table_cfc_placement_decision"},
        {
            "id": "cfc_experiment_gate_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## 正确顺序是先修B1，再做matched CfC\n\n"
                "1. 先完成B0 step1/step100审计，并修复mean/NLL/sigma解耦；CfC和GRU必须共享同一 repaired B1 objective。\n"
                "2. 新增配置而不覆盖25_b1；GRU/CfC均从epoch0训练，全部参数requires_grad=true，不共享checkpoint。\n"
                "3. 第一阶段只做参数匹配的backend replacement；不要并行GRU+CfC融合，也不要打开dynamic sigma geometry。\n"
                "4. 第二阶段才增加query-time CfC step和gap1/2/4的共享权重辅助预测，外部contract仍只导出当前endpoint。\n"
                "5. 做backend×time-mode的GRU/CfC × true/fixed/shuffled对照，并加入dropped-frame或held-out cadence。"
                "只有CfC在true time和长gap上同时优于GRU、CV以及fixed/shuffled，且NLL/coverage不退化，才能作为连续时间贡献；否则只作backbone消融。"
            ),
        },
        {
            "id": "cfc_report_limit_20260825",
            "type": "markdown",
            "layout": "full",
            "sourceId": source_id,
            "body": (
                "## 证据限制与待补指标\n\n"
                "历史CfC对照来自旧协议且没有可接受的matched B0 hash，因此不作因果排序。"
                "nuScenes mini的query delta-t均值约0.499 s、标准差约0.024 s，时间变化很窄，"
                "即使CfC有效也可能难以显现；需要dropped-frame、held-out cadence或后续高时间变化数据集。"
                "本次只有少量匹配端点，使用表格而不新增图表；原报告的全部图表与数据源均保留。"
            ),
        },
    ]
)

snapshot = {
    "current_uncertainty_path": current_rows,
    "historical_cfc_comparison": cfc_rows,
    "issue_scope_map": issue_rows,
    "cfc_placement_decision": placement_rows,
}
with sqlite3.connect(REPORT_DIR / "uncertainty_path_cfc.sqlite") as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS cfc_diagnosis_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO cfc_diagnosis_snapshot(snapshot_id, snapshot_json) VALUES (?, ?)",
        ("uncertainty_path_cfc_20260825", json.dumps(snapshot, ensure_ascii=False)),
    )

for filename, rows in (
    ("current_uncertainty_path.csv", current_rows),
    ("historical_cfc_comparison.csv", cfc_rows),
    ("issue_scope_map.csv", issue_rows),
    ("cfc_placement_decision.csv", placement_rows),
):
    with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

(REPORT_DIR / "analysis_summary.json").write_text(
    json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
)
(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(REPORT_DIR / "artifact.json")
