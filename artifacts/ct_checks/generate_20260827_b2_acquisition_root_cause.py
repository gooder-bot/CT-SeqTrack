"""Build the bounded B2 acquisition root-cause report from imported dev CSVs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IMPORT = ROOT / "artifacts/ct_checks/imported_server_diagnostics/b2_diag_v2_export_20260827_140008"
OUT = ROOT / "artifacts/ct_checks/reports/20260827_b2_acquisition_root_cause_and_literature"
GENERATED_AT = "2026-08-27T20:00:00+08:00"


def finite_quantile(series: pd.Series, quantile: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return None if values.empty else float(values.quantile(quantile))


def summarize_arm(name: str, path: Path) -> tuple[dict, list[dict]]:
    data = pd.read_csv(path)
    visible = data["global_target_count_label"] > 0
    exact_visible = data["global_target_count_exact"] > 0
    strict = visible & (data["base_raw_target_count"] == 0)
    base_sampling_loss = (
        visible
        & (data["base_raw_target_count"] > 0)
        & (data["base_sampled_target_count"] == 0)
    )
    strict_rows = data.loc[strict]
    endpoint = data["endpoint_raw_target_count"] > 0
    tube = data["tube_raw_target_count"] > 0
    union = data["support_union_target_count"] > 0
    key = ["tracklet_key", "frame_id", "candidate_id"]

    summary = {
        "arm": name,
        "rows": int(len(data)),
        "tracklets": int(data["tracklet_key"].nunique()),
        "schema_v2_rows": int((data["acquisition_schema_version"] == 2).sum()),
        "duplicate_stable_keys": int(data.duplicated(key).sum()),
        "nonfinite_numeric_values": int(
            (~np.isfinite(data.select_dtypes(include=[np.number]))).sum().sum()
        ),
        "exact_visible_rows": int(exact_visible.sum()),
        "label_visible_rows": int(visible.sum()),
        "sensor_unobservable_rows": int((~visible).sum()),
        "boundary_only_rows": int((visible & ~exact_visible).sum()),
        "base_captured_rows": int((visible & (data["base_raw_target_count"] > 0)).sum()),
        "base_strict_miss_rows": int(strict.sum()),
        "base_strict_miss_rate": float(strict.sum() / visible.sum()),
        "base_sampling_loss_rows": int(base_sampling_loss.sum()),
        "strict_actual_geometry_bearing_rows": int(
            (strict & (data["expansion_target_count"] > 0)).sum()
        ),
        "strict_pool_bearing_rows": int(
            (strict & (data["pool_target_count"] > 0)).sum()
        ),
        "strict_sample_bearing_rows": int(
            (strict & (data["sampled_target_count"] > 0)).sum()
        ),
        "pool_target_bearing_rows_all": int((data["pool_target_count"] > 0).sum()),
        "sample_target_bearing_rows_all": int((data["sampled_target_count"] > 0).sum()),
        "extension_pool_nonempty_rows": int((data["extension_pool_count"] > 0).sum()),
        "extension_pool_background_only_rows": int(
            ((data["extension_pool_count"] > 0) & (data["pool_target_count"] == 0)).sum()
        ),
        "endpoint_target_bearing_rows": int(endpoint.sum()),
        "tube_target_bearing_rows": int(tube.sum()),
        "union_target_bearing_rows": int(union.sum()),
        "endpoint_only_bearing_rows": int((endpoint & ~tube).sum()),
        "tube_only_bearing_rows": int((~endpoint & tube).sum()),
        "both_branch_bearing_rows": int((endpoint & tube).sum()),
        "strict_support_bearing_rows": int((strict & union).sum()),
        "support_target_subset_of_base": bool(
            (data["support_union_target_count"] <= data["base_raw_target_count"]).all()
        ),
        "observable_strict_miss_global_ge6": int(
            (strict & (data["global_target_count_label"] >= 6)).sum()
        ),
        "strict_observation_error_median": finite_quantile(strict_rows["observation_error"], 0.5),
        "strict_learned_error_mean": float(strict_rows["learned_motion_error"].mean()),
        "strict_learned_error_median": finite_quantile(strict_rows["learned_motion_error"], 0.5),
        "strict_learned_error_p95": finite_quantile(strict_rows["learned_motion_error"], 0.95),
        "strict_cv_error_median": finite_quantile(strict_rows["kinematic_error"], 0.5),
        "strict_abs_parallel_median": finite_quantile(strict_rows["learned_error_parallel"].abs(), 0.5),
        "strict_abs_parallel_p95": finite_quantile(strict_rows["learned_error_parallel"].abs(), 0.95),
        "strict_abs_perpendicular_median": finite_quantile(strict_rows["learned_error_perpendicular"].abs(), 0.5),
        "strict_abs_perpendicular_p95": finite_quantile(strict_rows["learned_error_perpendicular"].abs(), 0.95),
        "strict_z_error_median": finite_quantile(strict_rows["active_endpoint_error_z"], 0.5),
        "strict_z_error_p95": finite_quantile(strict_rows["active_endpoint_error_z"], 0.95),
        "strict_learned_cv_disagreement_median": finite_quantile(strict_rows["learned_cv_disagreement"], 0.5),
        "strict_learned_cv_disagreement_p95": finite_quantile(strict_rows["learned_cv_disagreement"], 0.95),
        "strict_gap_ratio_median": finite_quantile(strict_rows["gap_ratio"], 0.5),
        "strict_gap_ratio_p95": finite_quantile(strict_rows["gap_ratio"], 0.95),
        "strict_parallel_only_exceed_rows": int(
            ((strict_rows["learned_error_parallel"].abs() > 2.0)
             & (strict_rows["learned_error_perpendicular"].abs() <= 1.0)).sum()
        ),
        "strict_perpendicular_only_exceed_rows": int(
            ((strict_rows["learned_error_parallel"].abs() <= 2.0)
             & (strict_rows["learned_error_perpendicular"].abs() > 1.0)).sum()
        ),
        "strict_both_axes_exceed_rows": int(
            ((strict_rows["learned_error_parallel"].abs() > 2.0)
             & (strict_rows["learned_error_perpendicular"].abs() > 1.0)).sum()
        ),
        "strict_inside_current_margins_rows": int(
            ((strict_rows["learned_error_parallel"].abs() <= 2.0)
             & (strict_rows["learned_error_perpendicular"].abs() <= 1.0)).sum()
        ),
        "strict_endpoint_width_median": finite_quantile(strict_rows["active_endpoint_width"], 0.5),
        "strict_endpoint_length_median": finite_quantile(strict_rows["active_endpoint_length"], 0.5),
        "strict_tube_extra_length_median": finite_quantile(
            strict_rows["active_tube_length"] - strict_rows["active_endpoint_length"], 0.5
        ),
        "strict_derived_base_width_median": finite_quantile(
            (strict_rows["active_endpoint_width"] - 2.0) * 1.25 + 4.0, 0.5
        ),
        "strict_derived_base_length_median": finite_quantile(
            (strict_rows["active_endpoint_length"] - 4.0) * 1.25 + 4.0, 0.5
        ),
        "helper_mismatch_rows": int(
            (data["support_xyz_target_count"] != data["expansion_target_count"]).sum()
        ),
        "recursive_age_valid_rows": int((data["recursive_age_valid"] > 0).sum()),
    }

    grouped = (
        data.assign(_visible=visible.astype(int), _strict=strict.astype(int))
        .groupby("tracklet_key", as_index=False)
        .agg(rows=("frame_id", "size"), visible_rows=("_visible", "sum"), strict_misses=("_strict", "sum"))
        .sort_values(["strict_misses", "visible_rows"], ascending=False)
    )
    grouped["strict_miss_rate"] = grouped["strict_misses"] / grouped["visible_rows"].replace(0, np.nan)
    grouped["arm"] = name
    grouped["tracklet"] = grouped["tracklet_key"].str.rsplit("/", n=1).str[-1]
    top = grouped.head(6).copy()
    top["cumulative_miss_share"] = top["strict_misses"].cumsum() / max(int(strict.sum()), 1)
    return summary, top[[
        "arm", "tracklet", "rows", "visible_rows", "strict_misses",
        "strict_miss_rate", "cumulative_miss_share"
    ]].to_dict("records")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "GRU": IMPORT / "gru_proposal_endpoints_v2.csv",
        "CfC": IMPORT / "cfc_proposal_endpoints_v2.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    summaries = []
    tracklets = []
    for name, path in paths.items():
        summary, top = summarize_arm(name, path)
        summaries.append(summary)
        tracklets.extend(top)
    by_arm = {row["arm"]: row for row in summaries}

    common_keys = ["tracklet_key", "frame_id", "candidate_id"]
    paired = frames["GRU"].merge(
        frames["CfC"], on=common_keys, suffixes=("_gru", "_cfc"), how="inner"
    )
    missing_gru = frames["GRU"].merge(
        frames["CfC"][common_keys], on=common_keys, how="left", indicator=True
    )
    missing_keys = missing_gru.loc[missing_gru["_merge"] == "left_only", common_keys]

    funnel = []
    for row in summaries:
        visible = row["label_visible_rows"]
        stages = [
            ("Label-visible", visible),
            ("B0 raw captured", row["base_captured_rows"]),
            ("B2 recovered strict miss", row["strict_actual_geometry_bearing_rows"]),
            ("Novel pool target-bearing", row["pool_target_bearing_rows_all"]),
            ("Sample target-bearing", row["sample_target_bearing_rows_all"]),
        ]
        for order, (stage, count) in enumerate(stages, 1):
            funnel.append({
                "arm": row["arm"], "stage_order": order, "stage": stage,
                "rows": count, "rate_of_label_visible": count / visible,
                "label_visible_denominator": visible,
            })

    errors = []
    error_fields = [
        ("Observation center", "strict_observation_error_median"),
        ("Learned endpoint", "strict_learned_error_median"),
        ("Absolute parallel", "strict_abs_parallel_median"),
        ("Absolute perpendicular", "strict_abs_perpendicular_median"),
        ("Z", "strict_z_error_median"),
        ("Learned–CV disagreement", "strict_learned_cv_disagreement_median"),
    ]
    for row in summaries:
        for label, field in error_fields:
            errors.append({
                "arm": row["arm"], "component": label,
                "median_m": row[field], "strict_miss_rows": row["base_strict_miss_rows"],
            })

    acquisition = []
    for row in summaries:
        acquisition.append({
            "arm": row["arm"],
            "label_visible": row["label_visible_rows"],
            "sensor_unobservable": row["sensor_unobservable_rows"],
            "boundary_only": row["boundary_only_rows"],
            "base_strict_miss": row["base_strict_miss_rows"],
            "base_strict_miss_rate": row["base_strict_miss_rate"],
            "base_sampling_loss": row["base_sampling_loss_rows"],
            "strict_support_target": row["strict_actual_geometry_bearing_rows"],
            "strict_pool_target": row["strict_pool_bearing_rows"],
            "strict_sample_target": row["strict_sample_bearing_rows"],
            "pool_nonempty": row["extension_pool_nonempty_rows"],
            "pool_background_only": row["extension_pool_background_only_rows"],
            "global_ge6_strict_miss": row["observable_strict_miss_global_ge6"],
        })

    branch = []
    for row in summaries:
        branch.append({
            "arm": row["arm"],
            "endpoint_bearing": row["endpoint_target_bearing_rows"],
            "tube_bearing": row["tube_target_bearing_rows"],
            "union_bearing": row["union_target_bearing_rows"],
            "endpoint_only": row["endpoint_only_bearing_rows"],
            "tube_only": row["tube_only_bearing_rows"],
            "both": row["both_branch_bearing_rows"],
            "strict_union_bearing": row["strict_support_bearing_rows"],
            "all_support_targets_already_in_base": row["support_target_subset_of_base"],
        })

    geometry = []
    for row in summaries:
        geometry.append({
            "arm": row["arm"],
            "endpoint_width_median": row["strict_endpoint_width_median"],
            "endpoint_length_median": row["strict_endpoint_length_median"],
            "derived_b0_width_median": row["strict_derived_base_width_median"],
            "derived_b0_length_median": row["strict_derived_base_length_median"],
            "tube_extra_length_median": row["strict_tube_extra_length_median"],
            "parallel_only_exceed": row["strict_parallel_only_exceed_rows"],
            "perpendicular_only_exceed": row["strict_perpendicular_only_exceed_rows"],
            "both_exceed": row["strict_both_axes_exceed_rows"],
            "inside_current_margins": row["strict_inside_current_margins_rows"],
        })

    tracking = [
        {"run": "B0 epoch60 mini_val", "success": 54.826038, "precision": 66.340263,
         "role": "健康宿主对照；训练期 final"},
        {"run": "B1-GRU epoch60 mini_val", "success": 30.734137, "precision": 29.625820,
         "role": "B1-only 递归轨迹；训练期 final"},
        {"run": "B1-CfC epoch60 mini_val", "success": 35.318382, "precision": 37.696938,
         "role": "CfC 消融；训练期 final"},
        {"run": "B1-GRU dev diagnostic", "success": 28.800577, "precision": 28.236994,
         "role": "本次 v2 funnel 的实际轨迹"},
        {"run": "B1-CfC dev diagnostic", "success": 32.153175, "precision": 38.179188,
         "role": "本次 v2 funnel 的实际轨迹"},
        {"run": "SeqTrack Jul reference", "success": 53.359955, "precision": 64.381836,
         "role": "历史描述性参考，协议并非严格等价"},
    ]

    quality = [
        {"check": "Schema v2 / stable key / finite", "GRU": "311/311, 0 duplicate, 0 nonfinite",
         "CfC": "310/310, 0 duplicate, 0 nonfinite", "status": "PASS",
         "interpretation": "原始 CSV 身份与基础数值质量可用"},
        {"check": "Actual expansion vs pure helper XYZ", "GRU": "124 mismatch rows",
         "CfC": "129 mismatch rows", "status": "FAIL CLOSED",
         "interpretation": "长宽轴交换；所有 support_xy/xyz、center-inside 与 cf_* 暂停解释"},
        {"check": "Actual support targets subset of B0 targets", "GRU": "True",
         "CfC": "True", "status": "PASS",
         "interpretation": "实际 support 没有带来任何新增目标点"},
        {"check": "Recursive age", "GRU": "0 valid rows", "CfC": "0 valid rows",
         "status": "UNAVAILABLE", "interpretation": "本批不能做 recursive-age 分层"},
        {"check": "Paired row coverage", "GRU": "310/311 in intersection",
         "CfC": "310/310 in intersection", "status": "PARTIAL",
         "interpretation": "global counts 配对一致；不可据未匹配 B0 轨迹晋升 backend"},
    ]

    literature = [
        {"theme": "搜索中心是上游硬边界", "papers": "SC3D, P2B, BAT, M²-Track, DMT",
         "evidence": "大多在上一预测附近做局部 crop；目标在区域外时下游无法恢复",
         "decision": "先修中心/覆盖，再修采样"},
        {"theme": "关系采样只保留池内目标", "papers": "PTTR, SyncTrack",
         "evidence": "RAS/APST 改善区域内稀疏点保留，同时受错误模板影响",
         "decision": "pool_target>0 且 sample_target=0 后才启用"},
        {"theme": "运动先验需要当前证据纠正", "papers": "M²-Track, DMT",
         "evidence": "粗运动中心有效，但局部精修存在有限捕获半径",
         "decision": "B1 定位 acquisition；B2 只用实际点证据修正"},
        {"theme": "历史有益但会污染", "papers": "SeqTrack3D, M3SOT, MBPTrack, StreamTrack",
         "evidence": "有限历史最佳；更长或错误历史会退化",
         "decision": "必要时只加少量、可靠、可诊断备用锚点"},
        {"theme": "上下文必须有界", "papers": "CXTrack, CorpNet, STNet, MLVSNet",
         "evidence": "上下文和多层保留改善区域内判别，但不能替代 crop recall",
         "decision": "stable base + bounded shell，不做无约束全帧膨胀"},
    ]

    next_steps = [
        {"order": 1, "action": "修复纯诊断几何的 wlh→local xyz 映射",
         "implementation": "x 使用 length=wlh[1]，y 使用 width=wlh[0]；union volume 同步修复；增加非正方形旋转框测试",
         "gate": "learned_2_1_z0 与 actual expansion 逐行一致", "changes_model": "No"},
        {"order": 2, "action": "只重跑 GRU/CfC epoch60 checkpoint 的同一 dev 评估",
         "implementation": "不重训，不改 25_* YAML；重新生成 v2 CSV/JSON",
         "gate": "所有 v2 invariants 通过且 report 不再 fail closed", "changes_model": "No"},
        {"order": 3, "action": "先判断 bounded local shell 是否有真实恢复",
         "implementation": "比较 corrected 2/1、3/1.5、4/2、6/3、z05/z10 与 learned+CV；同时看背景 p95/volume",
         "gate": "若 3–6 m shell 能恢复且背景可控，进入单独的 B2-G1 配置", "changes_model": "Later"},
        {"order": 4, "action": "若 6/3 仍低召回，修改搜索中心而非继续扩大",
         "implementation": "保留 B0 stable base；增加因果且有界的最后可靠锚点/短历史 corridor 反事实，再决定是否实现备用 support",
         "gate": "备用中心显著优于同中心加宽；learned/CV 双中心因当前分歧很小不优先", "changes_model": "Later"},
        {"order": 5, "action": "目标进入 raw extension pool 后再修采样",
         "implementation": "relation top-k + spatial coverage + 少量随机探索；endpoint/tube 空余配额可借用",
         "gate": "真实出现 pool_target>0 且 sampled_target=0 的非偶发损失", "changes_model": "Later"},
        {"order": 6, "action": "采样稳定后再做 vote consistency / top-k 与 B3",
         "implementation": "不在零 supply 阶段加入 decoder、voting 或 calibration",
         "gate": "B2 target-bearing supply 与 retention 已通过", "changes_model": "Later"},
    ]

    headline = [{
        "diagnosis": "SEARCH CENTER / SUPPORT GEOMETRY",
        "gru_strict_miss_rate": by_arm["GRU"]["base_strict_miss_rate"],
        "cfc_strict_miss_rate": by_arm["CfC"]["base_strict_miss_rate"],
        "strict_support_recovered_rows": (
            by_arm["GRU"]["strict_actual_geometry_bearing_rows"]
            + by_arm["CfC"]["strict_actual_geometry_bearing_rows"]
        ),
        "pool_target_rows": (
            by_arm["GRU"]["pool_target_bearing_rows_all"]
            + by_arm["CfC"]["pool_target_bearing_rows_all"]
        ),
        "base_sampling_loss_rows": (
            by_arm["GRU"]["base_sampling_loss_rows"]
            + by_arm["CfC"]["base_sampling_loss_rows"]
        ),
        "helper_mismatch_rows": (
            by_arm["GRU"]["helper_mismatch_rows"]
            + by_arm["CfC"]["helper_mismatch_rows"]
        ),
    }]

    summary = {
        "verdict": {
            "primary_bottleneck": "search_center_and_support_geometry",
            "sampling_is_current_bottleneck": False,
            "dataset_unobservability_is_primary": False,
            "z_clip_is_primary": False,
            "simple_learned_cv_dual_center_is_promising_on_current_misses": False,
            "counterfactual_margin_selection_valid": False,
            "retraining_needed_after_diagnostic_fix": False,
        },
        "headline": headline[0],
        "arm_summary": summaries,
        "paired": {
            "intersection_rows": int(len(paired)),
            "gru_rows": int(len(frames["GRU"])),
            "cfc_rows": int(len(frames["CfC"])),
            "global_label_counts_equal": bool(
                (paired["global_target_count_label_gru"] == paired["global_target_count_label_cfc"]).all()
            ),
            "pool_target_counts_equal": bool(
                (paired["pool_target_count_gru"] == paired["pool_target_count_cfc"]).all()
            ),
            "missing_gru_keys": missing_keys.to_dict("records"),
        },
        "tracking_summary": tracking,
        "next_steps": next_steps,
        "source_paths": [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths.values()],
        "generator": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    datasets = {
        "headline": headline,
        "tracking": tracking,
        "funnel": funnel,
        "acquisition": acquisition,
        "errors": errors,
        "arm_summary": summaries,
        "branch": branch,
        "geometry": geometry,
        "tracklets": tracklets,
        "quality": quality,
        "literature": literature,
        "next_steps": next_steps,
    }
    db_path = OUT / "report_data.sqlite"
    with sqlite3.connect(db_path) as connection:
        for dataset_id, rows in datasets.items():
            pd.DataFrame(rows).to_sql(dataset_id, connection, if_exists="replace", index=False)
        pd.DataFrame([{
            "snapshot_id": "b2_acquisition_root_cause_20260827",
            "snapshot_json": json.dumps(summary, ensure_ascii=False),
        }]).to_sql("report_snapshot", connection, if_exists="replace", index=False)

    source = {
        "id": "src_b2_analysis_20260827",
        "label": "CT-SeqTrack GRU/CfC dev acquisition diagnostics v2 复算快照",
        "path": "artifacts/ct_checks/reports/20260827_b2_acquisition_root_cause_and_literature/report_data.sqlite",
        "query": {
            "engine": "sqlite",
            "sql": "SELECT snapshot_json FROM report_snapshot WHERE snapshot_id = 'b2_acquisition_root_cause_20260827'",
            "tablesUsed": ["report_snapshot"],
            "description": "从已校验导入 CSV 计算 observability、实际 acquisition funnel、误差分解与 tracklet 集中度",
            "metricDefinitions": {
                "label_visible": "global_target_count_label > 0；完整当前帧按 bb_scale=1.25 至少有一个目标点",
                "observable_strict_miss": "label_visible 且 base_raw_target_count = 0",
                "geometry_recovered": "observable_strict_miss 且 actual expansion_target_count > 0",
                "sampling_loss": "pool_target_count > 0 且 sampled_target_count = 0",
            },
        },
    }
    literature_source = {
        "id": "src_literature_20260827",
        "label": "15 篇 3D SOT 搜索与采样论文的系统化综述",
        "path": "artifacts/ct_checks/reports/20260827_b2_acquisition_root_cause_and_literature/literature_review.md",
    }

    cards = [
        {"id": "card_diagnosis", "dataset": "headline", "sourceId": source["id"],
         "description": "当前漏斗首先在搜索中心/几何覆盖处中断。",
         "metrics": [{"label": "Primary bottleneck", "field": "diagnosis"}]},
        {"id": "card_gru_miss", "dataset": "headline", "sourceId": source["id"],
         "description": "分母为 278 个 label-visible GRU dev 行。",
         "metrics": [{"label": "GRU observable strict miss", "field": "gru_strict_miss_rate", "format": "percent"}]},
        {"id": "card_cfc_miss", "dataset": "headline", "sourceId": source["id"],
         "description": "分母为 277 个 label-visible CfC dev 行。",
         "metrics": [{"label": "CfC observable strict miss", "field": "cfc_strict_miss_rate", "format": "percent"}]},
        {"id": "card_recovery", "dataset": "headline", "sourceId": source["id"],
         "description": "101+88 个 observable strict miss 中，实际 endpoint∪tube 获得目标的行数。",
         "metrics": [{"label": "Strict misses recovered", "field": "strict_support_recovered_rows", "format": "number"}]},
        {"id": "card_pool", "dataset": "headline", "sourceId": source["id"],
         "description": "两臂全部 621 行中 raw extension pool 含目标的行数。",
         "metrics": [{"label": "Target-bearing pools", "field": "pool_target_rows", "format": "number"}]},
        {"id": "card_bug", "dataset": "headline", "sourceId": source["id"],
         "description": "纯诊断 helper 与实际 crop 计数不一致；反事实 margin 暂不可解释。",
         "metrics": [{"label": "Helper mismatch rows", "field": "helper_mismatch_rows", "format": "number"}]},
    ]

    charts = [
        {
            "id": "chart_funnel", "title": "Label-visible 行的 acquisition 阶段保留率",
            "subtitle": "B0 已捕获约 64%–68%；B0 严格 miss 中 B2 实际 support 恢复率为 0",
            "intent": "progression", "question": "漏斗在哪一阶段首先归零？",
            "rationale": "以各臂 label-visible 行为共同分母，直接比较 B0 捕获、B2 几何恢复与 pool/sample 保留。",
            "type": "bar", "dataset": "funnel", "sourceId": source["id"], "layout": "full",
            "encodings": {
                "x": {"field": "stage", "type": "nominal", "label": "Stage"},
                "y": {"field": "rate_of_label_visible", "type": "quantitative", "label": "Rate of label-visible rows"},
                "color": {"field": "arm", "type": "nominal", "label": "Arm"},
                "tooltip": [
                    {"field": "rows", "type": "quantitative", "label": "Rows"},
                    {"field": "label_visible_denominator", "type": "quantitative", "label": "Visible denominator"},
                ],
            },
        },
        {
            "id": "chart_errors", "title": "Observable strict miss 的中心误差分解",
            "subtitle": "单位为米，中位数；主误差沿运动平行方向，z 与 learned–CV 分歧都很小",
            "intent": "comparison", "question": "当前搜索具体错在中心、方向还是 z？",
            "rationale": "同一批严格 miss 上比较 observation、learned endpoint 与方向分量。",
            "type": "bar", "dataset": "errors", "sourceId": source["id"], "layout": "full",
            "encodings": {
                "x": {"field": "component", "type": "nominal", "label": "Error component"},
                "y": {"field": "median_m", "type": "quantitative", "label": "Median error", "unit": "m"},
                "color": {"field": "arm", "type": "nominal", "label": "Arm"},
                "tooltip": [{"field": "strict_miss_rows", "type": "quantitative", "label": "Strict miss rows"}],
            },
        },
    ]

    def table(table_id, title, subtitle, dataset, columns, sort_field=None, direction="desc"):
        value = {
            "id": table_id, "title": title, "subtitle": subtitle,
            "dataset": dataset, "sourceId": source["id"], "layout": "full",
            "density": "dense", "columns": columns,
        }
        if sort_field:
            value["defaultSort"] = {"field": sort_field, "direction": direction}
        return value

    tables = [
        table("table_tracking", "实验结果口径汇总", "训练期 final、dev diagnostics 与历史 SeqTrack 只作各自口径内的描述",
              "tracking", [
                  {"field": "run", "label": "Run", "type": "text"},
                  {"field": "success", "label": "Success", "format": "number"},
                  {"field": "precision", "label": "Precision", "format": "number"},
                  {"field": "role", "label": "Role / caveat", "type": "text"},
              ], "success"),
        table("table_acquisition", "B2 acquisition funnel 精确计数", "strict miss 分母仅含全帧 label-visible 行",
              "acquisition", [
                  {"field": "arm", "label": "Arm", "type": "text"},
                  {"field": "label_visible", "label": "Visible", "format": "number"},
                  {"field": "sensor_unobservable", "label": "Unobservable", "format": "number"},
                  {"field": "base_strict_miss", "label": "B0 strict miss", "format": "number"},
                  {"field": "base_strict_miss_rate", "label": "Miss rate", "format": "percent"},
                  {"field": "global_ge6_strict_miss", "label": "Miss with ≥6 global target pts", "format": "number"},
                  {"field": "strict_support_target", "label": "Support recovered", "format": "number"},
                  {"field": "strict_pool_target", "label": "Pool target", "format": "number"},
                  {"field": "strict_sample_target", "label": "Sample target", "format": "number"},
                  {"field": "base_sampling_loss", "label": "B0 sample loss", "format": "number"},
              ], "base_strict_miss"),
        table("table_branch", "Endpoint 与 tube 的实际互补性", "全部 target-bearing support 行都已被 B0 crop 覆盖",
              "branch", [
                  {"field": "arm", "label": "Arm", "type": "text"},
                  {"field": "endpoint_bearing", "label": "Endpoint", "format": "number"},
                  {"field": "tube_bearing", "label": "Tube", "format": "number"},
                  {"field": "union_bearing", "label": "Union", "format": "number"},
                  {"field": "endpoint_only", "label": "Endpoint only", "format": "number"},
                  {"field": "tube_only", "label": "Tube only", "format": "number"},
                  {"field": "both", "label": "Both", "format": "number"},
                  {"field": "strict_union_bearing", "label": "Strict union", "format": "number"},
                  {"field": "all_support_targets_already_in_base", "label": "Subset of B0", "type": "boolean"},
              ], "union_bearing"),
        table("table_geometry", "当前 support 尺寸与方向超界", "B0 尺寸由同一车辆尺寸和 bb_scale=1.25、bb_offset=2 反推",
              "geometry", [
                  {"field": "arm", "label": "Arm", "type": "text"},
                  {"field": "endpoint_width_median", "label": "Endpoint W", "format": "number"},
                  {"field": "endpoint_length_median", "label": "Endpoint L", "format": "number"},
                  {"field": "derived_b0_width_median", "label": "B0 W", "format": "number"},
                  {"field": "derived_b0_length_median", "label": "B0 L", "format": "number"},
                  {"field": "tube_extra_length_median", "label": "Tube extra L", "format": "number"},
                  {"field": "parallel_only_exceed", "label": "Parallel only exceed", "format": "number"},
                  {"field": "both_exceed", "label": "Both exceed", "format": "number"},
                  {"field": "inside_current_margins", "label": "Inside 2/1", "format": "number"},
              ], "parallel_only_exceed"),
        table("table_tracklets", "严格 miss 的 tracklet 集中度", "每臂按 strict miss 数取前 6；mini dev 仅 14 条 tracklet",
              "tracklets", [
                  {"field": "arm", "label": "Arm", "type": "text"},
                  {"field": "tracklet", "label": "Tracklet", "type": "text"},
                  {"field": "visible_rows", "label": "Visible", "format": "number"},
                  {"field": "strict_misses", "label": "Strict misses", "format": "number"},
                  {"field": "strict_miss_rate", "label": "Miss rate", "format": "percent"},
                  {"field": "cumulative_miss_share", "label": "Cumulative share", "format": "percent"},
              ], "strict_misses"),
        table("table_quality", "数据质量与诊断可信度", "反事实几何在修复长宽轴之前必须 fail closed",
              "quality", [
                  {"field": "check", "label": "Check", "type": "text"},
                  {"field": "GRU", "label": "GRU", "type": "text"},
                  {"field": "CfC", "label": "CfC", "type": "text"},
                  {"field": "status", "label": "Status", "type": "text"},
                  {"field": "interpretation", "label": "Interpretation", "type": "text"},
              ], "check", "asc"),
        table("table_literature", "文献结论到 B2 决策的映射", "15 篇论文的详细注释与参考文献见 supporting review",
              "literature", [
                  {"field": "theme", "label": "Theme", "type": "text"},
                  {"field": "papers", "label": "Representative papers", "type": "text"},
                  {"field": "evidence", "label": "Shared evidence", "type": "text"},
                  {"field": "decision", "label": "B2 decision", "type": "text"},
              ], "theme", "asc"),
        table("table_next", "按短期计划推进的修改顺序", "每一步只改变一个 acquisition 层级；25_* 正式配置保持不动",
              "next_steps", [
                  {"field": "order", "label": "#", "format": "number"},
                  {"field": "action", "label": "Action", "type": "text"},
                  {"field": "implementation", "label": "Implementation", "type": "text"},
                  {"field": "gate", "label": "Gate", "type": "text"},
                  {"field": "changes_model", "label": "Model change", "type": "text"},
              ], "order", "asc"),
    ]

    blocks = [
        {"id": "title", "type": "markdown", "layout": "full",
         "body": "# CT-SeqTrack B2 acquisition funnel：根因判断与修改顺序"},
        {"id": "summary", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## 技术结论\n\n"
             "**当前主要问题是搜索中心/支持几何，不是采样，也不是数据集整体不可观测。** "
             "GRU/CfC 的完整当前帧分别有 278/277 个 label-visible 行，其中 B0 raw crop 严格遗漏 "
             "101/88 行；实际 endpoint∪tube 对这些遗漏恢复均为 0。更下游的 raw extension pool 在全部 "
             "621 行中没有一行含目标，所以 sampler 没有正点可保留。\n\n"
             "严格 miss 的 learned endpoint XY 误差中位数为 14.27 m（GRU）和 15.21 m（CfC），"
             "绝对平行误差中位数为 13.85/14.97 m，而 z 误差中位数仅 0.21/0.19 m；learned–CV "
             "分歧也只有 0.77/0.48 m。问题本质上是递归 B0 锚点已经漂走，B1 和 CV 从同一个错误锚点做"
             "局部外推。简单换 sampler、加 vote、或把 learned 与 CV 两个相近中心并起来都不会解决。"
         )},
        {"id": "cards", "type": "metric-strip", "layout": "full",
         "cardIds": [card["id"] for card in cards]},
        {"id": "results", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## 实验数据说明：B0 已恢复，但本次 B1 dev 轨迹是长尾压力路径\n\n"
             "B0 epoch60 mini_val 为 54.83/66.34（Success/Precision），数值上略高于历史 SeqTrack Jul，"
             "但协议不是 transaction-equivalent，只能作描述性参照。当前 v2 funnel 对应的 GRU/CfC dev "
             "结果为 28.80/28.24 与 32.15/38.18，明显低于 B0；两臂的递归 B0 轨迹也没有通过严格的"
             "同 GPU optimizer parity。因此这些 checkpoint 适合定位失踪后的 stress failure，不宜直接用来"
             "选择正常轨迹的最终 shell 尺寸或宣称 CfC 晋升。"
         )},
        {"id": "tracking_table", "type": "table", "layout": "full", "tableId": "table_tracking"},
        {"id": "funnel_text", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## 漏斗在实际搜索几何处归零\n\n"
             "完整帧无 label 点只占 33/311 与 33/310（约 10.6%），不足以解释主要失败。更重要的是，"
             "GRU 的 101 个严格 miss 中有 75 个、CfC 的 88 个中有 71 个在完整帧里至少有 6 个目标点；"
             "因此不是只有 1–2 个点的极端传感器稀疏。B0 采样只各丢失 1 行，而 B2 support 对所有"
             "observable strict miss 都没有取得目标。"
         )},
        {"id": "funnel_chart", "type": "chart", "layout": "full", "chartId": "chart_funnel"},
        {"id": "acquisition_table", "type": "table", "layout": "full", "tableId": "table_acquisition"},
        {"id": "center_text", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## 具体错在递归锚点与纵向中心，而不是 z\n\n"
             "严格 miss 上 observation 本身的中心误差已经是 14.58/15.21 m；B1 learned endpoint "
             "几乎没有改变这个量级。tube 相对 endpoint 的长度增量中位数只有 0.06/0.03 m，说明局部"
             "运动外推很短，而历史锚点已经远离真实目标。99/101 个 GRU miss 与全部 88 个 CfC miss "
             "都越过 2 m 平行 margin；70/101 与 44/88 只在平行方向越界。"
         )},
        {"id": "errors_chart", "type": "chart", "layout": "full", "chartId": "chart_errors"},
        {"id": "geometry_table", "type": "table", "layout": "full", "tableId": "table_geometry"},
        {"id": "novelty_text", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## extension-only 不是采样失败，而是没有新增目标证据\n\n"
             "实际 endpoint/tube 在全部行中有目标时，这些目标点始终是 B0 crop 已包含的子集；去重后"
             "没有新增目标点。GRU/CfC 只有 27/11 行得到非空 extension pool，而且全部只含背景。"
             "tube 仅比 endpoint 多覆盖 4/2 个 target-bearing 行，且这些行也不是 strict miss。"
             "因此 relation-aware sampling、分支配额借用和 top-k voting 都应后置。"
         )},
        {"id": "branch_table", "type": "table", "layout": "full", "tableId": "table_branch"},
        {"id": "tracklet_text", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## 长尾集中在少数 tracklet，说明是递归漂移而非统一 crop 尺寸不足\n\n"
             "GRU 前 4 条 tracklet 占 78.2% 严格 miss，CfC 前 4 条占 92.1%。同一 mini dev 中也有多条"
             "长 tracklet 保持接近零 miss。这种集中结构更符合某些轨迹进入错误递归盆地，而不是所有车辆"
             "都缺少同一个固定 margin。"
         )},
        {"id": "tracklet_table", "type": "table", "layout": "full", "tableId": "table_tracklets"},
        {"id": "quality_text", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## 反事实 margin 目前不能用于选型：诊断 helper 交换了 width/length\n\n"
             "正式 crop 通过 `Box.corners()` 使用 local x=length、local y=width；纯诊断 helper 却按"
             " `wlh[0], wlh[1]` 直接构造 x/y 半轴。车辆框非正方形，因此 GRU/CfC 分别有 124/129 行"
             "的 helper XYZ target count 与 actual expansion 不一致。报告工具的异常是正确的 fail-closed "
             "行为。修复前不要解释 z05/z10、3/1.5、4/2、6/3、CV 或 learned+CV 的 recall/背景。"
         )},
        {"id": "quality_table", "type": "table", "layout": "full", "tableId": "table_quality"},
        {"id": "literature_text", "type": "markdown", "layout": "full", "sourceId": literature_source["id"],
         "body": (
             "## 文献结论支持“几何先于采样”\n\n"
             "P2B 的搜索中心消融显示递归预测中心相较 GT 中心可带来远大于多数下游模块的性能损失；"
             "PTTR 与 SyncTrack 的关系采样则明确作用于已经进入搜索区的点。M²-Track、DMT、SeqTrack3D、"
             "MBPTrack 和 StreamTrack 共同说明运动或历史可以改善局部跟踪，但错误递归状态仍会污染后续。"
             "所以你的短期计划方向正确，但 3A 不能只理解成围绕同一 learned center 加宽：若 corrected 6/3 "
             "仍失败，就必须转向有界备用中心/历史 corridor。"
         )},
        {"id": "literature_table", "type": "table", "layout": "full", "tableId": "table_literature"},
        {"id": "next_text", "type": "markdown", "layout": "full",
         "body": (
             "## 推荐修改顺序\n\n"
             "第一轮只修诊断并重跑 checkpoint 评估；第二轮只做 B2-G1 的 bounded geometry；第三轮只有在"
             "同中心加宽不足时才加入 causal alternate center；relation-aware sampling 必须等真实 pool "
             "出现目标后再做。B0 observation crop、25_* YAML、B1 backend、B3 和递归 writer 本轮都不动。"
         )},
        {"id": "next_table", "type": "table", "layout": "full", "tableId": "table_next"},
        {"id": "limitations", "type": "markdown", "layout": "full", "sourceId": source["id"],
         "body": (
             "## 限制与进一步问题\n\n"
             "数据仅来自 nuScenes mini dev 的 14 条 tracklet 与两个未配对的递归 checkpoint；结论是机制诊断，"
             "不是正式性能因果结论。recursive_age 在本批没有有效行。修复 helper 后，仍需用同一稳定行键比较"
             " corrected counterfactual，并在健康 B0/B1 轨迹上复核正常 shell；如果宽 shell 和备用中心都无法"
             "从完整帧恢复目标，下一步应查坐标变换、方向定义和 crop 实现，而不是训练 B2。"
         )},
    ]

    manifest = {
        "version": 1, "surface": "report",
        "title": "CT-SeqTrack B2 acquisition funnel：根因判断与修改顺序",
        "description": "基于 GRU/CfC dev diagnostics v2、短期计划、本地代码和 15 篇 3D SOT 论文判断 B2 搜索与采样瓶颈。",
        "generatedAt": GENERATED_AT,
        "sources": [source, literature_source],
        "cards": cards, "charts": charts, "tables": tables, "blocks": blocks,
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {"version": 1, "generatedAt": GENERATED_AT, "status": "ready", "datasets": datasets},
        "sources": [source, literature_source],
    }
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
