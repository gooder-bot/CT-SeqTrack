"""Build the read-only v26 five-arm mini experiment audit.

The experiment outputs are never modified.  Derived tables, a SQLite snapshot,
and the portable-report input are written below ``artifacts/ct_checks/reports``.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT / "artifacts" / "ct_checks" / "reports" /
    "20260905_v26_mini_five_arm"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "B0": ROOT / "output/20260903-2301-26_b0-ct26_b0_mini_car_seed42_60ep_bs16_20260903_225946",
    "B1-GRU": ROOT / "output/20260903-2301-26_b1_gru-ct26_b1_gru_mini_car_seed42_60ep_bs16_20260903_225946",
    "B1-CfC": ROOT / "output/20260903-2301-26_b1_cfc-ct26_b1_cfc_mini_car_seed42_60ep_bs16_20260903_225946",
    "Full-minus-B3": ROOT / "output/20260903-2301-26_full_minus_b3-ct26_full_b3_mini_car_seed42_60ep_bs16_20260903_225946",
    "Full": ROOT / "output/20260903-2301-26_full-ct26_full_mini_car_seed42_60ep_bs16_20260903_225946",
}
EVENTS = {name: path / "lightning_logs/version_0" for name, path in RUNS.items()}


def scalar_values(root: Path, folder: str):
    accumulator = EventAccumulator(str(root / folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if len(tags) != 1:
        raise RuntimeError(f"{root / folder}: expected one scalar tag, got {tags}")
    return accumulator.Scalars(tags[0])


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


validation_rows = []
tracking_rows = []
for arm, root in EVENTS.items():
    success = scalar_values(root, "metrics_mini_val_success")
    precision = scalar_values(root, "metrics_mini_val_precision")
    runtime = scalar_values(root, "runtime_runtime")
    if not (len(success) == len(precision) == len(runtime)):
        raise RuntimeError(f"{arm}: metric series lengths disagree")
    expected = 30 if arm != "Full-minus-B3" else 4
    if len(success) != expected:
        raise RuntimeError(f"{arm}: expected {expected} validation points, got {len(success)}")
    for index, (s_item, p_item, r_item) in enumerate(
        zip(success, precision, runtime), start=1
    ):
        epoch = index * 2
        validation_rows.append({
            "arm": arm,
            "epoch": epoch,
            "global_step": int(s_item.step),
            "success": float(s_item.value),
            "precision": float(p_item.value),
            "runtime_fps": float(r_item.value),
            "run_status": "partial" if arm == "Full-minus-B3" else "complete",
        })
    tracking_rows.append({
        "arm": arm,
        "status": "partial: crash in epoch-10 validation" if arm == "Full-minus-B3" else "complete",
        "last_complete_epoch": len(success) * 2,
        "success": float(success[-1].value),
        "precision": float(precision[-1].value),
        "tail3_observed_success": float(np.mean([item.value for item in success[-3:]])),
        "tail3_observed_precision": float(np.mean([item.value for item in precision[-3:]])),
        "runtime_fps": float(runtime[-1].value),
        "validation_points": len(success),
        "formal_late3_available": False,
    })

b0_final = next(row for row in tracking_rows if row["arm"] == "B0")
for row in tracking_rows:
    row["descriptive_delta_success_vs_b0"] = row["success"] - b0_final["success"]
    row["descriptive_delta_precision_vs_b0"] = row["precision"] - b0_final["precision"]

tracking_bar_rows = [
    {
        "arm": row["arm"],
        "metric": metric,
        "value": row[field],
        "epoch": row["last_complete_epoch"],
        "status": row["status"],
    }
    for row in tracking_rows
    for metric, field in (("Success", "success"), ("Precision", "precision"))
]

# These B1 summaries are computed from the corresponding final complete
# candidate CSV (epoch 60), except the explicitly partial arm (epoch 8).
b1_rows = [
    {
        "arm": "B1-GRU", "epoch": 60, "valid_rows": 1844,
        "learned_rmse_m": 6.314978, "cv_rmse_m": 6.302485,
        "learned_minus_cv_m": 0.012494,
        "bootstrap_ci_low_m": -0.000166, "bootstrap_ci_high_m": 0.026820,
        "target_in_support": 0.7115, "coverage_50": 0.9154,
        "coverage_80": 0.9295, "coverage_95": 0.9376,
        "promotion": "FAIL",
    },
    {
        "arm": "B1-CfC", "epoch": 60, "valid_rows": 1831,
        "learned_rmse_m": 12.491975, "cv_rmse_m": 12.758745,
        "learned_minus_cv_m": -0.266769,
        "bootstrap_ci_low_m": -0.303133, "bootstrap_ci_high_m": -0.229701,
        "target_in_support": 0.4194, "coverage_50": 0.4304,
        "coverage_80": 0.5270, "coverage_95": 0.6029,
        "promotion": "UNPAIRED / under-calibrated",
    },
    {
        "arm": "Full", "epoch": 60, "valid_rows": 1846,
        "learned_rmse_m": 6.455751, "cv_rmse_m": 6.446156,
        "learned_minus_cv_m": 0.009595,
        "bootstrap_ci_low_m": -0.004154, "bootstrap_ci_high_m": 0.026046,
        "target_in_support": 0.7210, "coverage_50": 0.8456,
        "coverage_80": 0.8987, "coverage_95": 0.9247,
        "promotion": "FAIL",
    },
    {
        "arm": "Full-minus-B3", "epoch": 8, "valid_rows": 1870,
        "learned_rmse_m": 12.010901, "cv_rmse_m": 12.330110,
        "learned_minus_cv_m": -0.319209,
        "bootstrap_ci_low_m": -0.369420, "bootstrap_ci_high_m": -0.271992,
        "target_in_support": 0.4102, "coverage_50": 0.4818,
        "coverage_80": 0.5663, "coverage_95": 0.6508,
        "promotion": "PARTIAL / under-calibrated",
    },
]
b1_chart_rows = [
    {
        "arm": row["arm"], "metric": label, "rmse_m": row[field],
        "epoch": row["epoch"], "promotion": row["promotion"],
    }
    for row in b1_rows
    for label, field in (("Learned", "learned_rmse_m"), ("CV", "cv_rmse_m"))
]


def b2_summary(arm: str, epoch: int, report_name: str):
    report = json.loads((REPORT_DIR / report_name).read_text(encoding="utf-8"))
    csv_path = EVENTS[arm] / "candidate_diagnostics" / f"epoch_{epoch:02d}.csv"
    frame = pd.read_csv(csv_path)
    available = frame["available"] > 0
    target_selected = frame["selected_target_count"] > 0
    gain = frame["observation_error"] - frame["raw_search_error"]
    iou_gain = frame["raw_search_iou"] - frame["observation_iou"]
    strict_need = (
        (frame["global_target_count_label"] > 0)
        & (frame["base_raw_target_count"] == 0)
    )
    strict_recovered = strict_need & (frame["support_novel_target_count"] > 0)
    available_gain = gain[available]
    available_iou = iou_gain[available]
    no_target = available & ~target_selected
    return {
        "arm": arm,
        "epoch": epoch,
        "status": "partial" if arm == "Full-minus-B3" else "complete",
        "csv_rows": int(len(frame)),
        "csv_tracklets": int(frame["tracklet_id"].nunique()),
        "missing_nonfirst_endpoints": int(2179 - len(frame)),
        "global_need_rows": int(report["globally_observable_need"]["rows"]),
        "novel_pool_target_recall": float(
            report["globally_observable_need"]["novel_pool_target_bearing"]
        ),
        "mechanism_threshold": 0.15,
        "mechanism_passed": bool(report["mechanism_thresholds"]["passed"]),
        "selection_eligible_rows": int(report["selection"]["eligible_rows"]),
        "selection_row_recall": float(report["selection"]["row_recall"]),
        "selection_point_recall": float(report["selection"]["point_recall"]),
        "relation_auroc_row_mean": float(report["selection"]["relation_auroc_mean"]),
        "relation_ap_row_mean": float(report["selection"]["relation_ap_mean"]),
        "available_rows": int(available.sum()),
        "available_rate": float(available.mean()),
        "available_center_gain_m": float(available_gain.mean()),
        "available_iou_gain": float(available_iou.mean()),
        "available_harm_rate": float((available_gain < -0.1).mean()),
        "no_target_available_rows": int(no_target.sum()),
        "no_target_center_gain_m": float(gain[no_target].mean()),
        "no_target_harm_rate": float((gain[no_target] < -0.1).mean()),
        "strict_miss_rows": int(strict_need.sum()),
        "strict_novel_recovered_rows": int(strict_recovered.sum()),
        "strict_novel_recall": float(strict_recovered.sum() / max(strict_need.sum(), 1)),
        "strict_observation_error_median_m": float(frame.loc[strict_need, "observation_error"].median()),
        "strict_observation_error_p90_m": float(frame.loc[strict_need, "observation_error"].quantile(0.9)),
        "active_b1_prior_rows": int((frame["active_prior_source"] == "b1").sum()),
        "fallback_cv_rows": int((frame["active_prior_source"] == "fallback_cv").sum()),
        "coverage_need_rate": float((frame["search_coverage_need"] > 0).mean()),
        "corridor_valid_rate": float((frame["corridor_valid"] > 0).mean()),
        "prepool_points_median": float(frame["prepool_point_count"].median()),
        "b3_calibrated_rate": float((frame["b3_calibrated"] > 0).mean()),
        "b3_action_coverage": float((frame["router_applied_gate"] > 0).mean()),
    }


b2_rows = [
    b2_summary("Full", 60, "full_b2_epoch60.json"),
    b2_summary("Full-minus-B3", 8, "full_b3_b2_epoch08.json"),
]

b2_chart_rows = [
    {
        "arm": row["arm"],
        "epoch": row["epoch"],
        "novel_pool_target_recall": row["novel_pool_target_recall"],
        "strict_novel_recall": row["strict_novel_recall"],
        "selection_row_recall": row["selection_row_recall"],
        "selection_point_recall": row["selection_point_recall"],
    }
    for row in b2_rows
]

audit_rows = [
    {"arm": "B0", "initial_equal": True, "input100_equal": True, "step1": "126580d7", "step100": "4231eff7", "b0_updates": 75720, "plugin_updates": 0, "frozen": False},
    {"arm": "B1-GRU", "initial_equal": True, "input100_equal": True, "step1": "6b901e2f", "step100": "ab98b21c", "b0_updates": 75720, "plugin_updates": 12780, "frozen": False},
    {"arm": "B1-CfC", "initial_equal": True, "input100_equal": True, "step1": "726e7516", "step100": "dc9467bf", "b0_updates": 75720, "plugin_updates": 12780, "frozen": False},
    {"arm": "Full-minus-B3", "initial_equal": True, "input100_equal": True, "step1": "b875c2ab", "step100": "8e6910b0", "b0_updates": 10096, "plugin_updates": 1704, "frozen": False},
    {"arm": "Full", "initial_equal": True, "input100_equal": True, "step1": "2d036f20", "step100": "d94bb0c3", "b0_updates": 75720, "plugin_updates": 12780, "frozen": False},
]

issue_rows = [
    {"priority": "P0", "area": "B1 acquisition", "finding": "motion prepass omits acquisition_margin_parallel_perp; all online B1 priors are invalid", "impact": "learned mean/margin never controls the B2 crop; coverage-need is always on", "required_action": "return and unbatch the margin field; add online source/margin contract test"},
    {"priority": "P0", "area": "Hybrid FPS", "finding": "identical-XY candidates can repeat index 0", "impact": "Full-minus-B3 crashes with selected rows are not unique", "required_action": "mask selected indices and test identical XY / different Z plus quota borrowing"},
    {"priority": "P0", "area": "Matched B0", "finding": "B0 parameter and Adam hashes diverge at step 1 despite identical inputs", "impact": "cross-arm deltas have no module-level causal attribution", "required_action": "sequential same-GPU step1/100 gradient and optimizer-state audit"},
    {"priority": "P0", "area": "Empty crop", "finding": "empty B0 crop returns before B2 and before writing diagnostics", "impact": "recovery is disabled on the hardest frames; CSV misses 10–12% endpoints", "required_action": "run extension-only recovery and always emit a diagnostic endpoint row"},
    {"priority": "P1", "area": "Counterfactual v3", "finding": "float32 local B0 keys are compared with re-transformed float64 global points at 1e-6", "impact": "counterfactual raw and novel target counts become identical", "required_action": "reuse exact online crop arrays/key path and assert row-wise equivalence"},
    {"priority": "P1", "area": "Acquisition", "finding": "novel target recall is 6.32% (Full) / 8.40% (partial), below 15%", "impact": "selection cannot recover evidence that never enters the bounded support", "required_action": "repair B1 path first, then reevaluate bounded support; do not enlarge blindly"},
    {"priority": "P1", "area": "Consensus voting", "finding": "coherent background often gets high consistency/inlier scores", "impact": "Full available candidates have negative mean gain and high harm", "required_action": "penalize sparse effective support and expose count/mass to held-out B3 calibration"},
    {"priority": "P1", "area": "B3", "finding": "no held-out calibration/dev artifact", "impact": "action coverage is zero and Full score is observation output", "required_action": "calibrate each late-3 checkpoint only after upstream retraining"},
    {"priority": "P2", "area": "Formal evaluation", "finding": "epoch59 is not independently evaluated; SeqTrack-strict v26 is absent; one seed only", "impact": "no formal late-3, paired CI, external comparison, or stability claim", "required_action": "test e58/e59/e60, add SeqTrack-strict, then full dataset"},
]

headline = [{
    "completed_arms": 4,
    "requested_arms": 5,
    "full_novel_recall": b2_rows[0]["novel_pool_target_recall"],
    "required_novel_recall": 0.15,
    "full_b2_harm": b2_rows[0]["available_harm_rate"],
    "b3_action_coverage": b2_rows[0]["b3_action_coverage"],
}]

snapshot = {
    "headline": headline,
    "validation_curves": validation_rows,
    "tracking_summary": tracking_rows,
    "tracking_bars": tracking_bar_rows,
    "b1_summary": b1_rows,
    "b1_chart": b1_chart_rows,
    "b2_summary": b2_rows,
    "b2_chart": b2_chart_rows,
    "contract_audit": audit_rows,
    "issues": issue_rows,
}

db_path = REPORT_DIR / "report_data.sqlite"
query = "SELECT snapshot_json FROM report_snapshot WHERE snapshot_id='v26_five_arm'"
with sqlite3.connect(db_path) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS report_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO report_snapshot VALUES (?, ?)",
        ("v26_five_arm", json.dumps(snapshot, ensure_ascii=False, allow_nan=False)),
    )
    checked = connection.execute(query).fetchone()
if checked is None or json.loads(checked[0]) != snapshot:
    raise RuntimeError("SQLite report snapshot verification failed")

source = {
    "id": "src_v26_five_arm",
    "label": "v26 mini five-arm TensorBoard, checkpoints and candidate diagnostics",
    "path": "artifacts/ct_checks/reports/20260905_v26_mini_five_arm/report_data.sqlite",
    "query": {
        "engine": "sqlite",
        "sql": query,
        "tables_used": ["report_snapshot"],
        "executed_at": "2026-09-05T12:00:00+08:00",
    },
}

title = "CT-SeqTrack v26 五组 mini 实验审计"
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "完成状态、跟踪分数、B0 公平性、B1/B2/B3 机制漏斗及 full nuScenes 准入判断",
        "generatedAt": "2026-09-05T12:00:00+08:00",
        "sources": [source],
        "cards": [
            {"id": "completion", "dataset": "headline", "sourceId": source["id"], "description": "五组中只有四组完成 60 epoch；Full-minus-B3 在 epoch-10 validation 中止。", "metrics": [{"label": "完成臂", "field": "completed_arms", "format": "number"}, {"label": "计划臂", "field": "requested_arms", "format": "number"}]},
            {"id": "acquisition", "dataset": "headline", "sourceId": source["id"], "description": "globally-observable need rows 中 novel pre-pool 的 target-bearing 比例。", "metrics": [{"label": "Full novel recall", "field": "full_novel_recall", "format": "percent"}, {"label": "最低门槛", "field": "required_novel_recall", "format": "percent"}]},
            {"id": "harm", "dataset": "headline", "sourceId": source["id"], "description": "Full epoch60 中 available raw B2 candidate 比 observation 恶化超过 0.1 m 的比例。", "metrics": [{"label": "B2 harmful", "field": "full_b2_harm", "format": "percent"}]},
            {"id": "calibration", "dataset": "headline", "sourceId": source["id"], "description": "缺少 held-out calibration artifact，B3 按合同 fail closed。", "metrics": [{"label": "B3 action coverage", "field": "b3_action_coverage", "format": "percent"}]},
        ],
        "charts": [
            {"id": "success_curve", "title": "mini_val Success 曲线", "subtitle": "每两 epoch 验证；Full-minus-B3 曲线在 epoch8 结束", "intent": "trend", "question": "五组训练各自达到什么水平，哪一组未跑完？", "rationale": "时间序列同时展示训练波动和中止位置；跨臂差值只作描述。", "type": "line", "dataset": "validation_curves", "sourceId": source["id"], "encodings": {"x": {"field": "epoch", "type": "quantitative", "label": "Epoch"}, "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"}, "color": {"field": "arm", "type": "nominal", "label": "Arm"}, "tooltip": [{"field": "precision", "type": "quantitative", "label": "Precision"}, {"field": "runtime_fps", "type": "quantitative", "label": "FPS"}, {"field": "run_status", "type": "text", "label": "Status"}]}, "layout": "full"},
            {"id": "tracking_bar", "title": "最后完整验证点的 Success / Precision", "subtitle": "四个完整臂为 epoch60；Full-minus-B3 为 partial epoch8，不能视为 final", "intent": "comparison", "question": "最后可读分数是多少？", "rationale": "分组柱图仅比较已观测读数，并在 tooltip 标明 epoch/status。", "type": "bar", "dataset": "tracking_bars", "sourceId": source["id"], "encodings": {"x": {"field": "arm", "type": "nominal", "label": "Arm"}, "y": {"field": "value", "type": "quantitative", "label": "Score", "unit": "points"}, "color": {"field": "metric", "type": "nominal", "label": "Metric"}, "tooltip": [{"field": "epoch", "type": "quantitative", "label": "Epoch"}, {"field": "status", "type": "text", "label": "Status"}]}, "layout": "full"},
            {"id": "b1_rmse", "title": "B1 learned prior 与 CV 的中心 RMSE", "subtitle": "验证 candidate rows；越低越好，单位 m", "intent": "comparison", "question": "学习先验是否稳定优于 CV？", "rationale": "同臂同 endpoint 的 learned/CV 并列比较最直接。", "type": "bar", "dataset": "b1_chart", "sourceId": source["id"], "encodings": {"x": {"field": "arm", "type": "nominal", "label": "Arm"}, "y": {"field": "rmse_m", "type": "quantitative", "label": "RMSE", "unit": "m"}, "color": {"field": "metric", "type": "nominal", "label": "Prior"}, "tooltip": [{"field": "epoch", "type": "quantitative", "label": "Epoch"}, {"field": "promotion", "type": "text", "label": "Promotion"}]}, "layout": "full"},
            {"id": "b2_recall", "title": "B2 novel evidence acquisition", "subtitle": "globally-observable need rows；15% 为注册最低门槛", "intent": "comparison", "question": "目标是否真正进入 novel pre-pool？", "rationale": "采样 row/point recall 只有在 acquisition 成功后才有意义，因此主图只显示最上游 novel recall。", "type": "bar", "dataset": "b2_chart", "sourceId": source["id"], "encodings": {"x": {"field": "arm", "type": "nominal", "label": "Arm"}, "y": {"field": "novel_pool_target_recall", "type": "quantitative", "format": "percent", "label": "Novel target recall"}, "tooltip": [{"field": "strict_novel_recall", "type": "quantitative", "format": "percent", "label": "Strict miss recovery"}, {"field": "selection_row_recall", "type": "quantitative", "format": "percent", "label": "Selection row recall"}, {"field": "selection_point_recall", "type": "quantitative", "format": "percent", "label": "Selection point recall"}]}, "referenceLines": [{"axis": "y", "value": 0.15, "label": "registered minimum 15%", "lineStyle": "dashed", "color": "danger"}], "layout": "full"},
        ],
        "tables": [
            {"id": "tracking_table", "title": "运行完成状态与最后可读指标", "subtitle": "tail-3 observed 是最后三个已记录偶数 epoch，不能替代正式 e58/e59/e60 late-3", "dataset": "tracking_summary", "sourceId": source["id"], "density": "dense", "layout": "full", "columns": [{"field": "arm", "label": "Arm", "type": "text"}, {"field": "status", "label": "Status", "type": "text"}, {"field": "last_complete_epoch", "label": "Epoch", "format": "number"}, {"field": "success", "label": "Success", "format": "number"}, {"field": "precision", "label": "Precision", "format": "number"}, {"field": "tail3_observed_success", "label": "Tail-3 obs S", "format": "number"}, {"field": "tail3_observed_precision", "label": "Tail-3 obs P", "format": "number"}, {"field": "runtime_fps", "label": "FPS", "format": "number"}]},
            {"id": "b2_table", "title": "B2 acquisition 与 raw candidate", "subtitle": "Full-minus-B3 仅为 epoch8；harm/gain 只统计 available 行", "dataset": "b2_summary", "sourceId": source["id"], "density": "dense", "layout": "full", "columns": [{"field": "arm", "label": "Arm", "type": "text"}, {"field": "epoch", "label": "Epoch", "format": "number"}, {"field": "novel_pool_target_recall", "label": "Novel recall", "format": "percent"}, {"field": "strict_novel_recall", "label": "Strict recovery", "format": "percent"}, {"field": "selection_row_recall", "label": "Selection row", "format": "percent"}, {"field": "selection_point_recall", "label": "Selection point", "format": "percent"}, {"field": "available_rate", "label": "Available", "format": "percent"}, {"field": "available_center_gain_m", "label": "Center gain m", "format": "number"}, {"field": "available_harm_rate", "label": "Harm", "format": "percent"}, {"field": "active_b1_prior_rows", "label": "Online B1 rows", "format": "number"}]},
            {"id": "audit_table", "title": "共享 B0 前缀审计", "subtitle": "initial 和输入指纹一致，但 step1/step100 参数 hash 全部分叉", "dataset": "contract_audit", "sourceId": source["id"], "density": "dense", "layout": "full", "columns": [{"field": "arm", "label": "Arm", "type": "text"}, {"field": "initial_equal", "label": "Initial", "type": "text"}, {"field": "input100_equal", "label": "Input100", "type": "text"}, {"field": "step1", "label": "Step1", "type": "text"}, {"field": "step100", "label": "Step100", "type": "text"}, {"field": "b0_updates", "label": "B0 updates", "format": "number"}, {"field": "plugin_updates", "label": "Plugin updates", "format": "number"}, {"field": "frozen", "label": "Frozen", "type": "text"}]},
            {"id": "issues_table", "title": "未解决问题与修复顺序", "subtitle": "P0 会阻止正确训练/公平归因；P1 阻止机制和论文结论；P2 是正式报告缺口", "dataset": "issues", "sourceId": source["id"], "density": "spacious", "layout": "full", "defaultSort": {"field": "priority", "direction": "asc"}, "columns": [{"field": "priority", "label": "Priority", "type": "text"}, {"field": "area", "label": "Area", "type": "text"}, {"field": "finding", "label": "Finding", "type": "text"}, {"field": "impact", "label": "Impact", "type": "text"}, {"field": "required_action", "label": "Required action", "type": "text"}]},
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
            {"id": "notice", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "> **准入结论：目前不能开始正式 full nuScenes。** 四臂完成 60 epoch，但 Full-minus-B3 在第 10 轮验证中崩溃；更关键的是，在线 B1 acquisition prepass 恒为 invalid，五臂 B0 又从 step1 起分叉。因此当前结果适合定位工程问题，不支持模块涨分或论文因果结论。"},
            {"id": "metrics", "type": "metric-strip", "layout": "full", "cardIds": ["completion", "acquisition", "harm", "calibration"]},
            {"id": "summary", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "## 1. 一句话结果\n\nB1-GRU 的最后读数最高（51.973 Success / 61.953 Precision），Full 为 48.802 / 54.888，B0 为 26.903 / 25.601；但 matched-prefix 失败意味着这些差值是不同 B0 优化轨迹的混合结果。Full 的 B3 没有 calibration artifact，action coverage=0，所以 Full 指标其实是该 checkpoint 的 observation 输出。"},
            {"id": "tracking", "type": "table", "layout": "full", "tableId": "tracking_table"},
            {"id": "curve", "type": "chart", "layout": "full", "chartId": "success_curve"},
            {"id": "bar", "type": "chart", "layout": "full", "chartId": "tracking_bar"},
            {"id": "contract", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "## 2. 训练合同\n\n五组都从随机初始化开始，没有 init checkpoint；启用模块均有非零有限梯度和更新计数，未发现冻结。共同 initial hash 和前 100 个 observation batch 指纹一致，但第一次 Adam 更新后 B0 参数与 Adam hash 已全部不同。按项目实验协议，跨臂分数只能描述，不能归因。"},
            {"id": "audit", "type": "table", "layout": "full", "tableId": "audit_table"},
            {"id": "b1_text", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "## 3. B1 没有真正驱动在线搜索\n\n`predict_motion_from_history()` 没有返回 acquisition margin，但 unbatch prepass 把它当作必需字段，因此在线 prior 被全部降级为 fallback-CV/base-only。CSV 中的 `b1_valid=1` 是 crop 之后的 B1 forward，不能证明 acquisition 使用了 B1。结果是 learned mean/margin 从未控制 shell，coverage-need 变成 100%，corridor 近乎常开。CfC 在自身 survivor rows 上优于 CV，但与 GRU endpoint 身份不一致且 coverage 严重偏低，不能据此正式选择后端。"},
            {"id": "b1chart", "type": "chart", "layout": "full", "chartId": "b1_rmse"},
            {"id": "b2_text", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "## 4. B2 瓶颈仍是 acquisition，不是后端 selection\n\nFull 的 globally-observable need novel recall 只有 6.32%，partial Full-minus-B3 为 8.40%，均未过 15% 门槛。目标一旦进入 pre-pool，selection row recall 为 100%，Full point recall 为 97.38%；但严格 B0 miss 的恢复率仅约 0.8%（Full），且 median observation error 接近 19 m。说明 bounded evidence 区域仍抓不到灾难长尾。Full available 行的 raw candidate 平均让中心误差恶化约 0.53 m，背景-only available 行恶化更明显；高 vote consistency 不能排除一致背景。"},
            {"id": "b2chart", "type": "chart", "layout": "full", "chartId": "b2_recall"},
            {"id": "b2table", "type": "table", "layout": "full", "tableId": "b2_table"},
            {"id": "quality", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "## 5. 诊断数据限制\n\n空 B0 crop 会在模型和诊断前直接 fallback，导致 epoch60 CSV 只覆盖 102/106 tracklets，并漏掉约 10%–12% 非首帧 endpoint；机制指标因此有 survivorship bias。反事实 v3 的 B0 novelty subtraction 又混用了 float32 local keys 与重新变换的 float64 global points，`raw` 与 `novel` 计数异常相同，所以当前 fixed/adaptive/dual counterfactual novel recall 不可引用。所有 CSV/TensorBoard 数值已检查为有限值，这不等于统计口径完整。"},
            {"id": "issues", "type": "table", "layout": "full", "tableId": "issues_table"},
            {"id": "next", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "## 6. 下一轮准入顺序\n\n1. 修复 B1 prepass margin 接口与 FPS 唯一性，并补相应回归测试。\n2. 让 empty-crop 帧仍能进入 extension-only B2，并总是写完整 endpoint diagnostic。\n3. 复用在线 crop/key 路径修复 counterfactual；补 recursive-age 诊断。\n4. 在同一 GPU 顺序跑 step1/100，审计 B0 gradient、Adam state 与参数 hash，直到五臂 matched-prefix 通过。\n5. 核心路径改变后五组 mini 必须从 epoch0 重训，不能续用当前 Full-minus-B3 checkpoint。\n6. mini 过机制门槛后，再给 e58/e59/e60 各自做互斥 calibration/dev promotion、正式 test、paired CI 和 SeqTrack-strict；最后才启动 full nuScenes。"},
            {"id": "method", "type": "markdown", "layout": "full", "sourceId": source["id"], "body": "## 7. 口径\n\n数据来自 commit `b445ecd` 下五个 2026-09-03 v26 mini run 的 TensorBoard scalars、`last.ckpt`/formal checkpoint audit、candidate diagnostics 与 acquisition report。Success/Precision 为所有 endpoint 的 micro-average；frame0 GT 也计入。所谓 tail-3 observed 是 epoch56/58/60 的训练期验证均值，不是正式 late-3；正式口径必须分别测试 epoch58/59/60。当前只有 seed42，不作跨 seed 稳定性声明。"},
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": "2026-09-05T12:00:00+08:00",
        "status": "ready",
        "datasets": snapshot,
    },
    "sources": [source],
}

analysis_summary = {
    "scope": {"runs": {name: rel(path) for name, path in RUNS.items()}},
    "tracking": tracking_rows,
    "b1": b1_rows,
    "b2": b2_rows,
    "contract_audit": audit_rows,
    "issues": issue_rows,
    "decision": {
        "ready_for_full_nuscenes": False,
        "reason": "P0 acquisition, uniqueness, empty-crop and matched-prefix failures",
    },
}

(REPORT_DIR / "analysis_summary.json").write_text(
    json.dumps(analysis_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
for filename, rows in (
    ("validation_curves.csv", validation_rows),
    ("tracking_summary.csv", tracking_rows),
    ("b1_summary.csv", b1_rows),
    ("b2_summary.csv", b2_rows),
    ("contract_audit.csv", audit_rows),
    ("issues.csv", issue_rows),
):
    with (REPORT_DIR / filename).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print(REPORT_DIR)
