"""Build the 2026-08-27 B0/B1 rerun and B2 diagnostic handoff report.

The source experiment directories are read-only.  This script materializes a
bounded, reviewed snapshot below artifacts/ct_checks/reports.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT / "artifacts" / "ct_checks" / "reports"
    / "20260827_b0_b1_b2_rerun_analysis"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


RUNS = {
    "new_b0": ROOT / "output/20260826-1534-25_b0-b0_seed42_diag_v2",
    "new_gru": ROOT / "output/20260826-1534-25_b1-b1_gru_seed42_diag_v2",
    "new_cfc": ROOT / "output/20260826-1535-25_b1-b1_cfc_seed42_diag_v2",
    "old_low_b0": ROOT / "output/20260825-1931-25_b0-ct25_b0_mini_car_seed42_60ep_bs16_val5",
    "seqtrack_jul": ROOT / "output/20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16",
    "seqtrack_aug": ROOT / "output/20260813-0116-01_seqtrack3d_baseline-scratch_ct21_b0_car_60ep_bs16_s42",
}

for name, path in RUNS.items():
    if not path.exists():
        raise FileNotFoundError(f"{name}: missing source run {path}")


# Values were recomputed from all TensorBoard scalar events (no reservoir
# sampling), epoch-60 CSVs, checkpoint audit hashes, and run provenance.
point_summary = [
    {
        "run": "New B0",
        "event_count": 75720,
        "raw_mean": 344.828966,
        "raw_median": 278.093750,
        "raw_p05": 54.125000,
        "raw_p95": 892.756250,
        "raw_zero_rate": 0.0,
        "estimated_fg_mean": 111.651789,
        "soft_fg_mean": 446.781499,
        "comparability": "B0 observation stream; directly comparable",
    },
    {
        "run": "Old low-score B0",
        "event_count": 75720,
        "raw_mean": 344.828966,
        "raw_median": 278.093750,
        "raw_p05": 54.125000,
        "raw_p95": 892.756250,
        "raw_zero_rate": 0.0,
        "estimated_fg_mean": 111.377579,
        "soft_fg_mean": 446.645653,
        "comparability": "Same observation point-count path as New B0",
    },
    {
        "run": "New B1-GRU",
        "event_count": 88500,
        "raw_mean": 333.087879,
        "raw_median": 261.437500,
        "raw_p05": 46.937500,
        "raw_p95": 896.693750,
        "raw_zero_rate": 0.0,
        "estimated_fg_mean": 104.998466,
        "soft_fg_mean": 467.623781,
        "comparability": "Mixed 1262 observation + 213 mechanism steps/epoch",
    },
    {
        "run": "New B1-CfC",
        "event_count": 88500,
        "raw_mean": 334.407871,
        "raw_median": 261.812500,
        "raw_p05": 47.625000,
        "raw_p95": 899.690625,
        "raw_zero_rate": 0.0,
        "estimated_fg_mean": 104.397145,
        "soft_fg_mean": 465.224521,
        "comparability": "Mixed 1262 observation + 213 mechanism steps/epoch",
    },
    {
        "run": "SeqTrack Jul",
        "event_count": 75720,
        "raw_mean": 344.713781,
        "raw_median": 278.031250,
        "raw_p05": 54.312500,
        "raw_p95": 895.315625,
        "raw_zero_rate": 0.0,
        "estimated_fg_mean": 116.190904,
        "soft_fg_mean": 456.450858,
        "comparability": "Historical SeqTrack descriptive reference",
    },
    {
        "run": "SeqTrack Aug",
        "event_count": 75720,
        "raw_mean": 345.198383,
        "raw_median": 276.937500,
        "raw_p05": 54.312500,
        "raw_p95": 893.378125,
        "raw_zero_rate": 0.0,
        "estimated_fg_mean": 117.327389,
        "soft_fg_mean": 458.889564,
        "comparability": "Historical SeqTrack descriptive reference",
    },
]

point_chart = []
for row in point_summary:
    if row["run"] == "Old low-score B0":
        continue
    point_chart.extend([
        {"run": row["run"], "metric": "Raw search points", "value": row["raw_mean"]},
        {"run": row["run"], "metric": "Estimated foreground", "value": row["estimated_fg_mean"]},
    ])

tracking_summary = [
    {
        "run": "New B0",
        "final_success": 54.826038,
        "late3_success": 54.109049,
        "final_precision": 66.340263,
        "late3_precision": 64.023705,
        "validation_points": 60,
        "role": "Safe-SeqTrack-derived B0 host",
    },
    {
        "run": "New B1-GRU",
        "final_success": 30.734137,
        "late3_success": 31.735596,
        "final_precision": 29.625820,
        "late3_precision": 30.924872,
        "validation_points": 60,
        "role": "B1-only; deployed output remains B0 observation",
    },
    {
        "run": "New B1-CfC",
        "final_success": 35.318382,
        "late3_success": 35.315465,
        "final_precision": 37.696938,
        "late3_precision": 37.410650,
        "validation_points": 60,
        "role": "B1 ablation; not a promoted backend",
    },
    {
        "run": "SeqTrack Jul",
        "final_success": 53.359955,
        "late3_success": 52.904814,
        "final_precision": 64.381836,
        "late3_precision": 63.104301,
        "validation_points": 12,
        "role": "Historical reference; validation every 5 epochs",
    },
    {
        "run": "SeqTrack Aug",
        "final_success": 51.001095,
        "late3_success": 48.413933,
        "final_precision": 60.892784,
        "late3_precision": 54.816923,
        "validation_points": 12,
        "role": "Historical reference; different evaluator/contract",
    },
]

fairness_audit = [
    {
        "stage": "Initial B0 parameters",
        "B0": "798a8def3e82",
        "GRU": "798a8def3e82",
        "CfC": "798a8def3e82",
        "all_equal": "Yes",
        "meaning": "Same seeded B0 initialization",
    },
    {
        "stage": "Initial B0 Adam state",
        "B0": "57063755cb70",
        "GRU": "57063755cb70",
        "CfC": "57063755cb70",
        "all_equal": "Yes",
        "meaning": "Same initial optimizer state",
    },
    {
        "stage": "First 100 full observation-batch fingerprints",
        "B0": "100/100",
        "GRU": "100/100",
        "CfC": "100/100",
        "all_equal": "Yes",
        "meaning": "Input/candidate/point-sampling path matches",
    },
    {
        "stage": "First B0 loss",
        "B0": "18.006416321",
        "GRU": "18.006416321",
        "CfC": "18.006416321",
        "all_equal": "Yes",
        "meaning": "First forward and B0 loss agree",
    },
    {
        "stage": "B0 parameters after step 1",
        "B0": "850c94343bdd",
        "GRU": "0578255bea00",
        "CfC": "7b9edc83f6cc",
        "all_equal": "No",
        "meaning": "Strict fairness audit fails at first optimizer update",
    },
    {
        "stage": "B0 parameters after step 100",
        "B0": "54c5080699d3",
        "GRU": "5a994a39eb1e",
        "CfC": "90908dc24738",
        "all_equal": "No",
        "meaning": "Early numerical/update split persists",
    },
]

b1_mechanism = [
    {
        "arm": "B1-GRU",
        "rows": 1923,
        "observation_error_mean": 6.0827,
        "observation_iou_mean": 0.2468,
        "learned_error_mean": 6.0599,
        "kinematic_error_mean": 6.2349,
        "nll_mean": 8.7970,
        "coverage50": 0.562,
        "coverage80": 0.652,
        "coverage95": 0.733,
        "base_target_median": 2.0,
        "expansion_target_median": 0.0,
        "pool_target_mean": 0.1856,
    },
    {
        "arm": "B1-CfC",
        "rows": 1927,
        "observation_error_mean": 5.2406,
        "observation_iou_mean": 0.3052,
        "learned_error_mean": 5.1750,
        "kinematic_error_mean": 5.3345,
        "nll_mean": 8.9420,
        "coverage50": 0.589,
        "coverage80": 0.659,
        "coverage95": 0.717,
        "base_target_median": 2.0,
        "expansion_target_median": 0.0,
        "pool_target_mean": 0.0529,
    },
]

legacy_funnel = [
    {
        "arm": "B1-GRU",
        "rows": 1923,
        "base_zero_rows": 806,
        "base_zero_rate": 806 / 1923,
        "strict_geometry_bearing_rows": 1,
        "strict_geometry_bearing_rate": 1 / 806,
        "strict_pool_bearing_rows": 1,
        "strict_sample_bearing_rows": 1,
        "weak_base_le2_rows": 1010,
        "weak_geometry_bearing_rows": 119,
        "weak_pool_bearing_rows": 1,
    },
    {
        "arm": "B1-CfC",
        "rows": 1927,
        "base_zero_rows": 759,
        "base_zero_rate": 759 / 1927,
        "strict_geometry_bearing_rows": 1,
        "strict_geometry_bearing_rate": 1 / 759,
        "strict_pool_bearing_rows": 1,
        "strict_sample_bearing_rows": 1,
        "weak_base_le2_rows": 967,
        "weak_geometry_bearing_rows": 120,
        "weak_pool_bearing_rows": 2,
    },
]

diagnostic_plan = [
    {
        "order": 1,
        "action": "Update evaluation code to d384282 and verify schema marker",
        "gate": "rg finds acquisition_schema_version and learned_2_1_z0",
        "reason": "Training runs were clean e9a2d6d and epoch_60.csv is schema v1",
    },
    {
        "order": 2,
        "action": "Evaluate B0, B1-GRU and B1-CfC last.ckpt on the same dev manifest",
        "gate": "All finish with identical partition identity; B1 CSV schema=2",
        "reason": "Existing checkpoints are reusable; diagnostics do not alter forward",
    },
    {
        "order": 3,
        "action": "Generate GRU funnel and CfC stable-key paired reference report",
        "gate": "Report invariants pass; paired coverage is reported",
        "reason": "Do not interpret unpaired recursive endpoints as backend promotion",
    },
    {
        "order": 4,
        "action": "Classify misses by observability, XY, z, geometry, novelty and sampling",
        "gate": "One dominant failure stage is identified on label-visible rows",
        "reason": "This selects the smallest B2 change that addresses the actual bottleneck",
    },
    {
        "order": 5,
        "action": "Implement one bounded B2 intervention only",
        "gate": "Counterfactual recovery is material and background/support cost is acceptable",
        "reason": "Avoid mixing z margin, shell, dual support and sampler changes",
    },
]

headline = [{
    "b0_raw_vs_seqtrack_jul": 344.828966 / 344.713781 - 1.0,
    "b0_final_success": 54.826038,
    "b0_success_delta_vs_seqtrack_jul": 54.826038 - 53.359955,
    "strict_fairness_audit": "FAIL",
    "b2_schema_v2_available": "NOT YET RUN",
}]

snapshot = {
    "headline": headline,
    "point_summary": point_summary,
    "point_chart": point_chart,
    "tracking_summary": tracking_summary,
    "fairness_audit": fairness_audit,
    "b1_mechanism": b1_mechanism,
    "legacy_funnel": legacy_funnel,
    "diagnostic_plan": diagnostic_plan,
}

report_db = REPORT_DIR / "report_data.sqlite"
report_query = (
    "SELECT snapshot_json FROM report_snapshot "
    "WHERE snapshot_id = 'b0_b1_b2_rerun_20260827'"
)
with sqlite3.connect(report_db) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS report_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO report_snapshot(snapshot_id, snapshot_json) "
        "VALUES (?, ?)",
        ("b0_b1_b2_rerun_20260827", json.dumps(
            snapshot, ensure_ascii=False, allow_nan=False)),
    )
    selected = connection.execute(report_query).fetchone()
if selected is None or json.loads(selected[0]) != snapshot:
    raise RuntimeError("SQLite snapshot verification failed")

source = {
    "id": "src_b0_b1_b2_rerun_20260827",
    "label": "CT-SeqTrack 三组 60-epoch 重跑日志、checkpoint 审计与候选诊断快照",
    "path": relative(report_db),
    "query": {
        "engine": "sqlite",
        "sql": report_query,
        "tablesUsed": ["report_snapshot"],
        "description": "读取经全量日志复算并人工复核的有界诊断快照",
    },
}

title = "CT-SeqTrack B0/B1 重跑结果与 B2 诊断入口（2026-08-27）"
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "判断点数、SeqTrack 数值比较、随机路径假设及下一轮 B2 acquisition diagnostics。",
        "generatedAt": "2026-08-27T12:00:00+08:00",
        "sources": [source],
        "cards": [
            {
                "id": "card_points",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "New B0 raw search point 均值相对 SeqTrack Jul 的比例差。",
                "metrics": [{"label": "B0 raw points vs SeqTrack", "field": "b0_raw_vs_seqtrack_jul", "format": "percent"}],
            },
            {
                "id": "card_success",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "New B0 epoch60 mini_val Success。",
                "metrics": [{"label": "New B0 final Success", "field": "b0_final_success", "format": "number"}],
            },
            {
                "id": "card_delta",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "只作描述性比较；协议并非 transaction-equivalent。",
                "metrics": [{"label": "vs SeqTrack Jul", "field": "b0_success_delta_vs_seqtrack_jul", "format": "number", "unit": "points"}],
            },
            {
                "id": "card_audit",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "要求 initial、step1、step100 的 B0 参数与 Adam 状态一致。",
                "metrics": [{"label": "Strict fairness audit", "field": "strict_fairness_audit"}],
            },
            {
                "id": "card_b2",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "当前训练 CSV 是 v1，不能回答 v2 的几何归因问题。",
                "metrics": [{"label": "B2 acquisition schema v2", "field": "b2_schema_v2_available"}],
            },
        ],
        "charts": [
            {
                "id": "chart_points",
                "title": "各 run 的训练点数均值",
                "subtitle": "B1 两臂包含额外 mechanism stream，因此只用于健康检查，不作为严格同分布比较",
                "intent": "comparison",
                "question": "新 B0 的搜索点数是否回到 SeqTrack 数量级？",
                "rationale": "Raw 与 estimated foreground 同单位并列，可同时观察裁剪输入和前景估计是否塌陷。",
                "type": "bar",
                "dataset": "point_chart",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "run", "type": "nominal", "label": "Run"},
                    "y": {"field": "value", "type": "quantitative", "label": "Mean points"},
                    "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                },
                "layout": "full",
            },
            {
                "id": "chart_tracking",
                "title": "Epoch60 mini_val Success 的描述性比较",
                "subtitle": "New B0 高于两条历史 SeqTrack 数值；B1-only 两臂明显低于 B0",
                "intent": "comparison",
                "question": "新结果在数值上是否高于历史 SeqTrack？",
                "rationale": "Final 是协议规定的主比较点；late-3 在表中作为稳定性上下文。",
                "type": "bar",
                "dataset": "tracking_summary",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "run", "type": "nominal", "label": "Run"},
                    "y": {"field": "final_success", "type": "quantitative", "label": "Final Success", "unit": "points"},
                    "tooltip": [
                        {"field": "late3_success", "type": "quantitative", "label": "Late-3 Success"},
                        {"field": "final_precision", "type": "quantitative", "label": "Final Precision"}
                    ],
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "table_points",
                "title": "点数分布审计",
                "subtitle": "全量 scalar events；零点率均为 0",
                "dataset": "point_summary",
                "sourceId": source["id"],
                "defaultSort": {"field": "raw_mean", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "run", "label": "Run", "type": "text"},
                    {"field": "event_count", "label": "Events", "format": "number"},
                    {"field": "raw_mean", "label": "Raw mean", "format": "number"},
                    {"field": "raw_median", "label": "Median", "format": "number"},
                    {"field": "raw_p05", "label": "P05", "format": "number"},
                    {"field": "raw_p95", "label": "P95", "format": "number"},
                    {"field": "raw_zero_rate", "label": "Zero rate", "format": "percent"},
                    {"field": "estimated_fg_mean", "label": "Estimated FG", "format": "number"},
                    {"field": "comparability", "label": "Comparability", "type": "text"},
                ],
            },
            {
                "id": "table_tracking",
                "title": "Final / late-3 tracking 汇总",
                "subtitle": "历史 SeqTrack 每 5 epoch 验证一次；late-3 是最后三个已记录点",
                "dataset": "tracking_summary",
                "sourceId": source["id"],
                "defaultSort": {"field": "final_success", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "run", "label": "Run", "type": "text"},
                    {"field": "final_success", "label": "Final S", "format": "number"},
                    {"field": "late3_success", "label": "Late-3 S", "format": "number"},
                    {"field": "final_precision", "label": "Final P", "format": "number"},
                    {"field": "late3_precision", "label": "Late-3 P", "format": "number"},
                    {"field": "role", "label": "Interpretation", "type": "text"},
                ],
            },
            {
                "id": "table_audit",
                "title": "三臂公平性分叉链",
                "subtitle": "前100个输入完全一致，但第一次 optimizer update 后已不一致",
                "dataset": "fairness_audit",
                "sourceId": source["id"],
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "stage", "label": "Stage", "type": "text"},
                    {"field": "B0", "label": "B0", "type": "text"},
                    {"field": "GRU", "label": "GRU", "type": "text"},
                    {"field": "CfC", "label": "CfC", "type": "text"},
                    {"field": "all_equal", "label": "Equal", "type": "text"},
                    {"field": "meaning", "label": "Meaning", "type": "text"},
                ],
            },
            {
                "id": "table_b1",
                "title": "Epoch60 B1 / acquisition 旧字段",
                "subtitle": "这是 mini_val schema v1，只能提示问题，不能定位 observability/XY/z/dedup/sampling",
                "dataset": "b1_mechanism",
                "sourceId": source["id"],
                "defaultSort": {"field": "observation_error_mean", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "arm", "label": "Arm", "type": "text"},
                    {"field": "rows", "label": "Rows", "format": "number"},
                    {"field": "observation_error_mean", "label": "Obs error", "format": "number"},
                    {"field": "learned_error_mean", "label": "Learned error", "format": "number"},
                    {"field": "kinematic_error_mean", "label": "CV error", "format": "number"},
                    {"field": "nll_mean", "label": "NLL", "format": "number"},
                    {"field": "coverage95", "label": "Coverage95", "format": "percent"},
                    {"field": "base_target_median", "label": "Base target median", "format": "number"},
                    {"field": "expansion_target_median", "label": "Expansion median", "format": "number"},
                    {"field": "pool_target_mean", "label": "Pool mean", "format": "number"},
                ],
            },
            {
                "id": "table_funnel",
                "title": "旧 acquisition funnel 的警报",
                "subtitle": "Strict 指 base_target_count=0；当前字段不能解释为何未恢复",
                "dataset": "legacy_funnel",
                "sourceId": source["id"],
                "defaultSort": {"field": "base_zero_rate", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "arm", "label": "Arm", "type": "text"},
                    {"field": "base_zero_rows", "label": "Base=0 rows", "format": "number"},
                    {"field": "base_zero_rate", "label": "Base=0 rate", "format": "percent"},
                    {"field": "strict_geometry_bearing_rows", "label": "Geometry bearing", "format": "number"},
                    {"field": "strict_geometry_bearing_rate", "label": "Geometry recall", "format": "percent"},
                    {"field": "strict_pool_bearing_rows", "label": "Pool bearing", "format": "number"},
                    {"field": "strict_sample_bearing_rows", "label": "Sample bearing", "format": "number"},
                    {"field": "weak_base_le2_rows", "label": "Base≤2 rows", "format": "number"},
                    {"field": "weak_geometry_bearing_rows", "label": "Weak geometry", "format": "number"},
                    {"field": "weak_pool_bearing_rows", "label": "Weak pool", "format": "number"},
                ],
            },
            {
                "id": "table_plan",
                "title": "下一轮 B2 诊断顺序",
                "subtitle": "先获得 v2 归因，再决定最小改动",
                "dataset": "diagnostic_plan",
                "sourceId": source["id"],
                "defaultSort": {"field": "order", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "order", "label": "#", "format": "number"},
                    {"field": "action", "label": "Action", "type": "text"},
                    {"field": "gate", "label": "Pass gate", "type": "text"},
                    {"field": "reason", "label": "Why", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
            {
                "id": "summary",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 结论先行\n\n"
                    "**点数正常，但现有数据还不能解释 B2 为什么搜不到目标。** New B0 的 raw search point "
                    "均值为 344.829，SeqTrack Jul 为 344.714，差 0.033%；零点率均为 0。更关键的是，"
                    "旧低分 B0 与 New B0 的 75,720 个 raw point scalar 分布完全相同，所以分数从 29.87 "
                    "恢复到 54.83 不是由点数变化造成。New B0 的 final Success 在数值上比 SeqTrack Jul "
                    "高 1.466 点，但训练目标、采样事务和 evaluator 不等价，只能称为描述性更高，不能称为因果增益。\n\n"
                    "随机输入路径也已基本排除：三臂初始 B0、初始 Adam、前100个完整 observation batch 指纹和"
                    "首个 B0 loss 一致；但 step1 后 B0 参数与 Adam 状态已经分叉。由于三臂在不同物理 GPU 并发运行，"
                    "当前无法区分 CUDA/PointNet2 数值非确定性与跨模块更新耦合，严格公平性审计仍是 FAIL。"
                ),
            },
            {"id": "cards", "type": "metric-strip", "layout": "full", "cardIds": ["card_points", "card_success", "card_delta", "card_audit", "card_b2"]},
            {
                "id": "points_text",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 1. 点数没有塌陷，也不是这次分数恢复的原因\n\n"
                    "B0 的均值、median、P05、P95 都与两条 SeqTrack 参考重合，且每个训练 batch 都有点。"
                    "B1-GRU/CfC 的均值约低 3%，但它们每 epoch 记录 1,262 个 observation step 加 213 个 mechanism step，"
                    "共 88,500 条事件，不是和 B0/SeqTrack 完全同分布的直接比较。两臂 P05 仍约 47 点且 zero rate=0，"
                    "所以也没有输入点数塌陷。"
                ),
            },
            {"id": "points_chart", "type": "chart", "layout": "full", "chartId": "chart_points"},
            {"id": "points_table", "type": "table", "layout": "full", "tableId": "table_points"},
            {
                "id": "tracking_text",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 2. B0 数值高于 SeqTrack；B1 两臂没有高于\n\n"
                    "New B0 final/late-3 Success 为 54.826/54.109；SeqTrack Jul 为 53.360/52.905。"
                    "这是好迹象，说明 Safe-SeqTrack-derived B0 已重新进入正常盆地。"
                    "但 New GRU/CfC final Success 只有 30.734/35.318，明显低于 B0。CfC 在这次递归轨迹上数值高于 GRU，"
                    "可两者行集仅部分配对，且 B0 从 step1 就不同，不能据此晋升 CfC。"
                ),
            },
            {"id": "tracking_chart", "type": "chart", "layout": "full", "chartId": "chart_tracking"},
            {"id": "tracking_table", "type": "table", "layout": "full", "tableId": "table_tracking"},
            {
                "id": "random_text",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 3. 不是 DataLoader/点采样随机路径；是首次更新后的路径分叉\n\n"
                    "checkpoint 记录的 observation fingerprint 覆盖完整 batch 内容，不只是样本 ID。前100个三臂逐项相同，"
                    "因此数据顺序、candidate 和点采样不是首要嫌疑。step1 参数 SHA 与 Adam SHA 已不同，而日志中的 B0 loss "
                    "到 step3 才出现可见差异。这符合 CUDA backward/atomic 运算产生微小差异、再被非凸优化和递归裁剪放大的模式。"
                    "但由于没有按原计划在同一张 GPU 顺序跑 100-step，仍不能把跨模块耦合完全排除。"
                ),
            },
            {"id": "audit_table", "type": "table", "layout": "full", "tableId": "table_audit"},
            {
                "id": "b2_text",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 4. 当前旧诊断只能确认 B2 acquisition 严重，不足以决定怎么改\n\n"
                    "GRU/CfC epoch60 CSV 的 header 没有 `acquisition_schema_version`、global observability 或 counterfactual 字段，"
                    "所以它们仍是 schema v1。旧 funnel 中约 42%/39% 行的 base target=0，而严格行中当前 expansion 只恢复 1 行；"
                    "这提示固定 2m/1m support 的新目标供给极弱，却无法判断是传感器不可观测、XY 漏、z 裁剪、extension-only 去重，"
                    "还是 sampler 丢失。`acquisition_supply/epoch_60.json` 全零也不能解释点数，它属于未启用的训练侧 B2 supply。"
                ),
            },
            {"id": "b1_table", "type": "table", "layout": "full", "tableId": "table_b1"},
            {"id": "funnel_table", "type": "table", "layout": "full", "tableId": "table_funnel"},
            {
                "id": "next_text",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 5. 下一步：不重训，直接用 d384282 对现有 checkpoint 做 dev 侧车评估\n\n"
                    "三条训练 provenance 都是 clean tracked commit `e9a2d6d`，而完整 acquisition diagnostics v2 在后续 commit "
                    "`d384282`。诊断代码不增加模型参数、不改变 forward 和实际 extension tensor，因此现有 epoch60 checkpoint 可以直接复用。"
                    "先在服务器更新到 d384282，确认 schema marker；再对 B0、GRU、CfC 使用同一 dev partition。"
                    "报告阶段以 GRU 为 reference 对 CfC 做稳定行键配对。"
                ),
            },
            {"id": "plan_table", "type": "table", "layout": "full", "tableId": "table_plan"},
            {
                "id": "decision_rules",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 6. 拿到 v2 JSON 后的修改决策\n\n"
                    "- `global_target_count_label=0` 占主导：数据/传感器不可观测，不扩大 B2。\n"
                    "- XY recall 高、XYZ 低，且 z05/z10 恢复：只加最小有效 z margin。\n"
                    "- learned shell 加宽恢复且背景可控：做 bounded adaptive shell。\n"
                    "- CV 或 learned+CV 明显更好：做紧凑双 support。\n"
                    "- support 有目标而 pool=0：修 extension-only 构造/去重。\n"
                    "- pool 有目标而 sample=0：再做 relation-aware sampling。\n"
                    "- 宽 shell 与 CV 都失败但 global 有目标：先查坐标、方向和裁剪实现，不训练 B2。"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 7. 限制\n\n"
                    "本报告只使用 nuScenes mini、seed42 和历史单次 run；SeqTrack 对照的训练事务/evaluator 不完全等价。"
                    "三条新 run 在不同物理 GPU 并发，未通过同卡 step1/step100 parity gate。现有 acquisition 行来自递归轨迹，"
                    "GRU/CfC 行集不完全一致。因此所有‘更高/更低’都是描述性结果，不构成论文增益、backend 晋升或 B2 几何选择证据。"
                ),
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": "2026-08-27T12:00:00+08:00",
        "status": "ready",
        "datasets": snapshot,
    },
    "sources": [source],
}

source_paths = []
for run in RUNS.values():
    source_paths.append(relative(run / "run_provenance.json"))
for key in ("new_b0", "new_gru", "new_cfc"):
    source_paths.append(relative(
        RUNS[key] / "lightning_logs/version_0/checkpoints/last.ckpt"))
for key in ("new_gru", "new_cfc"):
    source_paths.append(relative(
        RUNS[key] / "lightning_logs/version_0/candidate_diagnostics/epoch_60.csv"))

analysis_summary = {
    "verdict": {
        "b0_point_count_normal": True,
        "point_count_caused_score_recovery": False,
        "b0_descriptively_above_seqtrack_jul": True,
        "causal_gain_claim_allowed": False,
        "input_random_path_is_primary_cause": False,
        "strict_cross_arm_fairness_passed": False,
        "b2_schema_v2_evaluation_complete": False,
        "retraining_required_for_v2_diagnostics": False,
    },
    "headline": headline[0],
    "point_summary": point_summary,
    "tracking_summary": tracking_summary,
    "fairness_audit": fairness_audit,
    "b1_mechanism": b1_mechanism,
    "legacy_funnel": legacy_funnel,
    "diagnostic_plan": diagnostic_plan,
    "source_paths": source_paths,
    "generator": relative(Path(__file__)),
    "chart_contract_notes": {
        "chart_points": "Grouped bar; shared point-count unit; B1 stream-composition caveat is visible.",
        "chart_tracking": "Single-series bar; zero baseline; exact final and late-3 values remain in the table.",
    },
}

(REPORT_DIR / "analysis_summary.json").write_text(
    json.dumps(analysis_summary, indent=2, ensure_ascii=False, allow_nan=False)
    + "\n",
    encoding="utf-8",
)
(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(json.dumps({
    "report_dir": relative(REPORT_DIR),
    "artifact": relative(REPORT_DIR / "artifact.json"),
    "summary": relative(REPORT_DIR / "analysis_summary.json"),
    "source_db": relative(report_db),
}, ensure_ascii=False))
