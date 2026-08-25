"""Build a reproducible diagnostic snapshot for the 2026-08-24 v25 runs.

The script is intentionally read-only with respect to ``output/``.  It reads
only the four registered retryfix arms and writes derived report artifacts
under ``artifacts/ct_checks/reports``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "artifacts" / "ct_checks" / "reports" / "20260824_v25_four_arm"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "B0": ROOT / "output/20260824-0219-25_b0-ct25_b0_mini_car_60ep_bs16_seed42_retryfix",
    "B1-only": ROOT / "output/20260824-0220-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_retryfix",
    "B1+B2": ROOT / "output/20260824-0220-25_full_minus_b3-ct25_b1_b2_mini_car_60ep_bs16_seed42_retryfix",
    "Full": ROOT / "output/20260824-0220-25_full-ct25_full_mini_car_60ep_bs16_seed42_retryfix",
}
EVENT_ROOTS = {
    name: path / "lightning_logs/version_0" for name, path in RUNS.items()
}


def event_values(root: Path, folder: str):
    accumulator = EventAccumulator(str(root / folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if len(tags) != 1:
        raise RuntimeError(f"{root / folder}: expected one scalar tag, got {tags}")
    return accumulator.Scalars(tags[0])


def success_auc(values):
    values = np.asarray(values, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, 21)
    curve = np.asarray([(values >= threshold).mean() for threshold in thresholds])
    return float(np.trapz(curve, x=thresholds) * 100.0)


def average_precision(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = targets[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def install_easydict_pickle_stub():
    module = types.ModuleType("easydict")

    class EasyDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    EasyDict.__module__ = "easydict"
    module.EasyDict = EasyDict
    sys.modules.setdefault("easydict", module)


def checkpoint(path: Path):
    install_easydict_pickle_stub()
    return torch.load(path, map_location="cpu", weights_only=False)


validation_rows = []
summary_rows = []
for arm, root in EVENT_ROOTS.items():
    success = event_values(root, "metrics_mini_val_success")
    precision = event_values(root, "metrics_mini_val_precision")
    runtime = event_values(root, "runtime_runtime")
    if not (len(success) == len(precision) == len(runtime) == 60):
        raise RuntimeError(f"{arm}: validation series is incomplete")
    for epoch, (success_item, precision_item, runtime_item) in enumerate(
        zip(success, precision, runtime), start=1
    ):
        validation_rows.append({
            "epoch": epoch,
            "global_step": int(success_item.step),
            "arm": arm,
            "success": float(success_item.value),
            "precision": float(precision_item.value),
            "runtime_fps": float(runtime_item.value),
            "late3": epoch >= 58,
            "final": epoch == 60,
        })
    summary_rows.append({
        "arm": arm,
        "final_success": float(success[-1].value),
        "late3_success": float(np.mean([item.value for item in success[-3:]])),
        "final_precision": float(precision[-1].value),
        "late3_precision": float(np.mean([item.value for item in precision[-3:]])),
        "runtime_fps_final": float(runtime[-1].value),
        "runtime_fps_mean": float(np.mean([item.value for item in runtime])),
        "validation_points": len(success),
    })


checkpoints = {}
audit_rows = []
for arm, run in RUNS.items():
    ckpt_path = run / "lightning_logs/version_0/checkpoints/last.ckpt"
    payload = checkpoint(ckpt_path)
    checkpoints[arm] = payload
    module_audit = payload["ct_module_audit"]
    prefixes = payload["ct_b0_prefix_hashes"]
    audit_rows.append({
        "arm": arm,
        "epoch": int(payload["epoch"]) + 1,
        "global_step": int(payload["global_step"]),
        "initial_hash": prefixes["initial"][:12],
        "step1_hash": prefixes["step_1"][:12],
        "step100_hash": prefixes["step_100"][:12],
        "final_hash": module_audit["parameter_sha256"]["b0"][:12],
        "b0_updates": int(module_audit["update_steps"]["b0"]),
        "plugin_updates": int(max(
            [value for name, value in module_audit["update_steps"].items() if name != "b0"],
            default=0,
        )),
        "input_fingerprints": len(payload["ct_observation_batch_fingerprints"]),
        "peak_allocated_mb": float(
            max(stage["peak_allocated_mb"] for stage in payload["ct_cuda_stage_audit"].values())
        ),
    })

base_prefixes = checkpoints["B0"]["ct_b0_prefix_hashes"]
base_fingerprints = checkpoints["B0"]["ct_observation_batch_fingerprints"][:100]
prefix_checks = {
    key: len({payload["ct_b0_prefix_hashes"][key] for payload in checkpoints.values()}) == 1
    for key in ("initial", "step_1", "step_100", "epoch_060")
}
fingerprints_equal = all(
    payload["ct_observation_batch_fingerprints"][:100] == base_fingerprints
    for payload in checkpoints.values()
)

cpu_rng_hashes = []
cuda_rng_hashes = []
for payload in checkpoints.values():
    rng = payload["ct_global_rng_state"]
    cpu_rng_hashes.append(hashlib.sha256(rng["torch_cpu"].numpy().tobytes()).hexdigest())
    cuda_rng_hashes.append(tuple(
        hashlib.sha256(item.numpy().tobytes()).hexdigest()
        for item in rng["torch_cuda"]
    ))

loss_divergence_rows = []
base_losses = event_values(EVENT_ROOTS["B0"], "loss_loss_b0_view0")
for arm, root in EVENT_ROOTS.items():
    values = event_values(root, "loss_loss_b0_view0")
    first_diff = next((
        int(left.step)
        for left, right in zip(base_losses, values)
        if left.step != right.step or left.value != right.value
    ), None)
    loss_divergence_rows.append({
        "arm": arm,
        "first_view0_loss_difference_step": first_diff if first_diff is not None else -1,
        "view0_event_count": len(values),
    })


diagnostic_frames = {}
b1_rows = []
b2_rows = []
for arm in ("B1-only", "B1+B2", "Full"):
    csv_path = EVENT_ROOTS[arm] / "candidate_diagnostics/epoch_60.csv"
    frame = pd.read_csv(csv_path)
    diagnostic_frames[arm] = frame
    valid = (
        (frame["b1_valid"] > 0)
        & np.isfinite(frame["learned_motion_error"])
        & np.isfinite(frame["kinematic_error"])
    )
    learned_rmse = float(np.sqrt(np.mean(frame.loc[valid, "learned_motion_error"] ** 2)))
    cv_rmse = float(np.sqrt(np.mean(frame.loc[valid, "kinematic_error"] ** 2)))
    b1_rows.append({
        "arm": arm,
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "learned_rmse": learned_rmse,
        "cv_rmse": cv_rmse,
        "learned_minus_cv": learned_rmse - cv_rmse,
        "mean_nll": float(frame.loc[valid, "b1_nll"].mean()),
        "coverage_50": float(frame.loc[valid, "b1_coverage_50"].mean()),
        "coverage_80": float(frame.loc[valid, "b1_coverage_80"].mean()),
        "coverage_95": float(frame.loc[valid, "b1_coverage_95"].mean()),
    })

    if arm == "B1-only":
        continue
    available = frame["available"] > 0
    presence = frame["presence_target"] > 0
    available_rows = frame.loc[available]
    gain = available_rows["observation_error"] - available_rows["raw_search_error"]
    oracle_iou = np.maximum(frame["observation_iou"], frame["raw_search_iou"])
    acquisition = json.loads((
        EVENT_ROOTS[arm] / "acquisition_supply/epoch_60.json"
    ).read_text(encoding="utf-8"))["populations"]["candidate0"]
    training_presence_ap = event_values(
        EVENT_ROOTS[arm], "ct_epoch_calibration_presence_ap"
    )[-1].value
    b2_rows.append({
        "arm": arm,
        "diagnostic_rows": int(len(frame)),
        "available_rows": int(available.sum()),
        "available_rate": float(available.mean()),
        "target_bearing_rows": int(((frame["extension_foreground_count"] > 0) & available).sum()),
        "target_bearing_given_available": float(
            ((frame.loc[available, "extension_foreground_count"] > 0).mean())
            if available.any() else 0.0
        ),
        "validation_presence_prior": float(presence[available].mean()) if available.any() else 0.0,
        "validation_presence_ap": average_precision(
            frame.loc[available, "presence_probability"], presence[available]
        ) if available.any() else float("nan"),
        "training_presence_ap": float(training_presence_ap),
        "raw_center_gain_m": float(gain.mean()),
        "raw_helpful_rate": float((gain > 0.1).mean()),
        "raw_harmful_rate": float((gain < -0.1).mean()),
        "raw_iou_gain": float((
            available_rows["raw_search_iou"] - available_rows["observation_iou"]
        ).mean()),
        "oracle_success_headroom_points": (
            success_auc(oracle_iou) - success_auc(frame["observation_iou"])
        ),
        "deployment_change_rate": float((
            np.abs(frame["final_error"] - frame["observation_error"]) > 1e-9
        ).mean()),
        "training_row_recall": float(acquisition["row_recall"]),
        "training_point_recall": float(acquisition["point_recall"]),
        "training_eligible_rows": int(acquisition["eligible_rows"]),
        "training_sampled_targets": int(acquisition["sampled_targets"]),
    })


full_frame = diagnostic_frames["Full"]
evidence_valid = full_frame["router_evidence_valid"] > 0
utility = evidence_valid & (
    (full_frame["center_gain"] > 0.1)
    | (full_frame["center_gain"] < -0.1)
    | (full_frame["iou_gain"] < 0)
)
helpful = utility & (full_frame["center_gain"] > 0.1) & (full_frame["iou_gain"] >= 0)
b3_row = {
    "validation_rows": int(len(full_frame)),
    "evidence_valid_rows": int(evidence_valid.sum()),
    "evidence_valid_rate": float(evidence_valid.mean()),
    "utility_rows": int(utility.sum()),
    "helpful_prior": float(helpful[utility].mean()) if utility.any() else 0.0,
    "validation_action_ap": average_precision(
        full_frame.loc[utility, "action_score"], helpful[utility]
    ) if utility.any() else float("nan"),
    "training_action_ap": float(event_values(
        EVENT_ROOTS["Full"], "ct_epoch_calibration_alpha_ap"
    )[-1].value),
    "training_action_auroc": float(event_values(
        EVENT_ROOTS["Full"], "ct_epoch_calibration_alpha_auroc"
    )[-1].value),
    "training_action_ece": float(event_values(
        EVENT_ROOTS["Full"], "ct_epoch_calibration_alpha_ece"
    )[-1].value),
    "calibrated_rate": float(full_frame["b3_calibrated"].mean()),
    "router_applied_rate": float(full_frame["router_applied_gate"].mean()),
    "deployment_change_rate": float((
        np.abs(full_frame["final_error"] - full_frame["observation_error"]) > 1e-9
    ).mean()),
}

b1_chart_rows = []
for row in b1_rows:
    for metric, field in (("Learned", "learned_rmse"), ("CV", "cv_rmse")):
        b1_chart_rows.append({
            "arm": row["arm"],
            "metric": metric,
            "rmse": row[field],
            "valid_rows": row["valid_rows"],
            "coverage_95": row["coverage_95"],
            "mean_nll": row["mean_nll"],
            "learned_minus_cv": row["learned_minus_cv"],
        })


summary_by_arm = {row["arm"]: row for row in summary_rows}
headline = [{
    "b0_final_success": summary_by_arm["B0"]["final_success"],
    "b0_late3_success": summary_by_arm["B0"]["late3_success"],
    "b0_parity_status": "FAIL",
    "input_fingerprint_status": "PASS",
    "b12_raw_harmful_rate": next(
        row["raw_harmful_rate"] for row in b2_rows if row["arm"] == "B1+B2"
    ),
    "full_raw_harmful_rate": next(
        row["raw_harmful_rate"] for row in b2_rows if row["arm"] == "Full"
    ),
    "b3_action_coverage": b3_row["router_applied_rate"],
    "b3_calibrated_rate": b3_row["calibrated_rate"],
}]

gate_rows = [
    {
        "module": "B0",
        "criterion": "四臂 initial/step1/step100/final B0 哈希一致",
        "result": "仅 initial 一致；step1 起失配",
        "verdict": "FAIL",
    },
    {
        "module": "B1",
        "criterion": "mini_val learned RMSE < CV 且 coverage 有限/接近名义值",
        "result": "B1+B2/Full learned 分别差 0.168/0.107 m；95% coverage 37.5%/55.5%",
        "verdict": "FAIL",
    },
    {
        "module": "B2",
        "criterion": "正 oracle headroom、有效 target-bearing、presence AP 高于先验",
        "result": "headroom 0.005–0.015 点；可用行 93%–97% 有害；验证正例仅 1–2 行",
        "verdict": "FAIL",
    },
    {
        "module": "B3",
        "criterion": "action coverage > 0 且相对 raw 降低 harmful rate",
        "result": "无校准 artifact，action coverage=0",
        "verdict": "NOT READY",
    },
]


def relative(path: Path):
    return path.relative_to(ROOT).as_posix()


source_paths = []
for run in RUNS.values():
    source_paths.extend([
        relative(run / "run_provenance.json"),
        relative(run / "lightning_logs/version_0/hparams.yaml"),
        relative(run / "lightning_logs/version_0/checkpoints/last.ckpt"),
    ])
for arm in ("B1-only", "B1+B2", "Full"):
    source_paths.append(relative(
        EVENT_ROOTS[arm] / "candidate_diagnostics/epoch_60.csv"
    ))
for arm in ("B1+B2", "Full"):
    source_paths.append(relative(
        EVENT_ROOTS[arm] / "acquisition_supply/epoch_60.json"
    ))

# Materialize the bounded, derived widget snapshot in a local SQLite source so
# the packaged report retains an executable provenance query.  The protected
# experiment outputs remain read-only; this database lives with the report.
report_snapshot = {
    "headline": headline,
    "validation_curves": validation_rows,
    "arm_summary": summary_rows,
    "b0_audit": audit_rows,
    "loss_divergence": loss_divergence_rows,
    "b1_validation": b1_rows,
    "b1_rmse_chart": b1_chart_rows,
    "b2_validation": b2_rows,
    "b3_validation": [b3_row],
    "gate_status": gate_rows,
}
report_db = REPORT_DIR / "report_data.sqlite"
report_query = (
    "SELECT snapshot_json FROM report_snapshot "
    "WHERE snapshot_id = 'v25_four_arm'"
)
with sqlite3.connect(report_db) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS report_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO report_snapshot(snapshot_id, snapshot_json) "
        "VALUES (?, ?)",
        ("v25_four_arm", json.dumps(report_snapshot, ensure_ascii=False, allow_nan=False)),
    )
    selected = connection.execute(report_query).fetchone()
if selected is None or json.loads(selected[0]) != report_snapshot:
    raise RuntimeError("SQLite report snapshot verification failed")

source = {
    "id": "src_v25_four_arm",
    "label": "v25 retryfix 四臂 TensorBoard、checkpoint 与逐帧诊断",
    "path": "artifacts/ct_checks/reports/20260824_v25_four_arm/report_data.sqlite",
    "query": {
        "engine": "sqlite",
        "sql": report_query,
        "tablesUsed": ["report_snapshot"],
    },
}


title = "CT-SeqTrack v25 四臂实验诊断（2026-08-24）"
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Safe-SeqTrack B0 对齐、B1/B2/B3 机制门槛与下一步实验判断。",
        "generatedAt": "2026-08-24T00:00:00+08:00",
        "sources": [source],
        "cards": [
            {
                "id": "card_b0_score",
                "dataset": "headline",
                "sourceId": "src_v25_four_arm",
                "description": "安全 evaluator 下 B0 的第60轮 Success；late-3 作为稳定性上下文。",
                "metrics": [
                    {"label": "B0 final Success", "field": "b0_final_success", "format": "number"},
                    {"label": "late-3", "field": "b0_late3_success", "format": "number"},
                ],
            },
            {
                "id": "card_b0_parity",
                "dataset": "headline",
                "sourceId": "src_v25_four_arm",
                "description": "跨臂 B0 前缀哈希；输入指纹通过不等于梯度/更新通过。",
                "metrics": [
                    {"label": "B0 跨臂哈希", "field": "b0_parity_status"},
                    {"label": "前100批输入", "field": "input_fingerprint_status"},
                ],
            },
            {
                "id": "card_b2_harm",
                "dataset": "headline",
                "sourceId": "src_v25_four_arm",
                "description": "epoch60 mini_val 中 available B2 raw candidate 的中心误差有害率。",
                "metrics": [
                    {"label": "B1+B2 harmful", "field": "b12_raw_harmful_rate", "format": "percent"},
                    {"label": "Full harmful", "field": "full_raw_harmful_rate", "format": "percent"},
                ],
            },
            {
                "id": "card_b3_coverage",
                "dataset": "headline",
                "sourceId": "src_v25_four_arm",
                "description": "Full 未安装 held-out calibration artifact，因此保持 fail-closed observation。",
                "metrics": [
                    {"label": "B3 action coverage", "field": "b3_action_coverage", "format": "percent"},
                    {"label": "calibrated rows", "field": "b3_calibrated_rate", "format": "percent"},
                ],
            },
        ],
        "charts": [
            {
                "id": "chart_validation_success",
                "title": "四臂 mini_val Success 曲线",
                "subtitle": "nuScenes mini car，seed42，60 epochs；每轮验证，单位为点",
                "intent": "trend",
                "question": "共享 B0 的四臂是否沿相同验证轨迹训练？",
                "rationale": "60个有序验证点足以显示从早期开始的持续分叉，并保留 final/late-3 上下文。",
                "type": "line",
                "dataset": "validation_curves",
                "sourceId": "src_v25_four_arm",
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"},
                    "color": {"field": "arm", "type": "nominal", "label": "Arm"},
                    "tooltip": [
                        {"field": "precision", "type": "quantitative", "label": "Precision"},
                        {"field": "runtime_fps", "type": "quantitative", "label": "FPS"},
                        {"field": "global_step", "type": "quantitative", "label": "Global step"},
                    ],
                },
                "layout": "full",
            },
            {
                "id": "chart_b1_rmse",
                "title": "epoch60 mini_val B1 learned 与 CV RMSE",
                "subtitle": "仅 b1_valid=1 的逐帧中心误差；越低越好，单位 m",
                "intent": "comparison",
                "question": "B1 learned prior 是否在验证分布上优于运动学 CV？",
                "rationale": "三个机制臂各有两项同单位离散比较，分组柱图能直接显示方向和幅度。",
                "type": "bar",
                "dataset": "b1_rmse_chart",
                "sourceId": "src_v25_four_arm",
                "encodings": {
                    "x": {"field": "arm", "type": "nominal", "label": "Arm"},
                    "y": {"field": "rmse", "type": "quantitative", "label": "RMSE", "unit": "m"},
                    "color": {"field": "metric", "type": "nominal", "label": "Method"},
                    "tooltip": [
                        {"field": "valid_rows", "type": "quantitative", "label": "Valid rows"},
                        {"field": "coverage_95", "type": "quantitative", "format": "percent", "label": "95% coverage"},
                        {"field": "mean_nll", "type": "quantitative", "label": "Mean NLL"},
                    ],
                },
                "layout": "full",
            },
            {
                "id": "chart_b2_harmful",
                "title": "epoch60 mini_val B2 raw candidate 有害率",
                "subtitle": "available=1 且中心误差比 observation 恶化超过0.1 m",
                "intent": "comparison",
                "question": "B2 有效候选进入决策前有多大比例会伤害跟踪？",
                "rationale": "两个模块臂使用同一比率定义，简单柱图比混合 oracle headroom 的双轴图更诚实。",
                "type": "bar",
                "dataset": "b2_validation",
                "sourceId": "src_v25_four_arm",
                "encodings": {
                    "x": {"field": "arm", "type": "nominal", "label": "Arm"},
                    "y": {"field": "raw_harmful_rate", "type": "quantitative", "format": "percent", "label": "Harmful rate"},
                    "tooltip": [
                        {"field": "available_rows", "type": "quantitative", "label": "Available rows"},
                        {"field": "raw_center_gain_m", "type": "quantitative", "label": "Mean center gain", "unit": "m"},
                        {"field": "oracle_success_headroom_points", "type": "quantitative", "label": "Oracle headroom", "unit": "points"},
                    ],
                },
                "valueFormat": "percent",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "table_arm_summary",
                "title": "四臂 final 与 late-3",
                "subtitle": "严格使用第60轮与58–60轮均值；不选 best epoch",
                "dataset": "arm_summary",
                "sourceId": "src_v25_four_arm",
                "defaultSort": {"field": "arm", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "arm", "label": "Arm", "type": "text"},
                    {"field": "final_success", "label": "Final S", "format": "number"},
                    {"field": "late3_success", "label": "Late-3 S", "format": "number"},
                    {"field": "final_precision", "label": "Final P", "format": "number"},
                    {"field": "late3_precision", "label": "Late-3 P", "format": "number"},
                    {"field": "runtime_fps_final", "label": "Final FPS", "format": "number"},
                ],
            },
            {
                "id": "table_b0_audit",
                "title": "B0 跨臂审计",
                "subtitle": "短哈希仅用于显示；完整 SHA256 保留在 checkpoint",
                "dataset": "b0_audit",
                "sourceId": "src_v25_four_arm",
                "defaultSort": {"field": "arm", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "arm", "label": "Arm", "type": "text"},
                    {"field": "initial_hash", "label": "Initial", "type": "text"},
                    {"field": "step1_hash", "label": "Step1", "type": "text"},
                    {"field": "step100_hash", "label": "Step100", "type": "text"},
                    {"field": "final_hash", "label": "Final", "type": "text"},
                    {"field": "b0_updates", "label": "B0 updates", "format": "number"},
                    {"field": "plugin_updates", "label": "Plugin updates", "format": "number"},
                ],
            },
            {
                "id": "table_gates",
                "title": "论文实验进入门槛",
                "subtitle": "按 SAFE_SEQTRACK_V25_PROTOCOL 与既定实验计划逐项判断",
                "dataset": "gate_status",
                "sourceId": "src_v25_four_arm",
                "defaultSort": {"field": "module", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "module", "label": "Module", "type": "text"},
                    {"field": "criterion", "label": "Criterion", "type": "text"},
                    {"field": "result", "label": "Observed", "type": "text"},
                    {"field": "verdict", "label": "Verdict", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## Executive Summary / 执行摘要\n\n"
                    "**结论：B0 在分数上出现明显恢复迹象，但没有通过 Safe-SeqTrack 的正式对齐门槛，当前不能宣称已经恢复到 SeqTrack 水平，也不能把 Full 相对 B0 的差值解释为模块增益。** 独立 B0 在第60轮达到 50.690 Success / 59.280 Precision，late-3 为 50.185 / 57.038；然而四臂虽然初始 B0、前100批输入指纹和最终全局 RNG 状态一致，却从 step1 起 B0 参数哈希失配。B1-only 的 observation 只剩 33.443 / 35.138，而 Full 的 fail-closed observation 为 52.553 / 61.194，证明跨臂随机训练轨迹仍未受控。\n\n"
                    "模块侧也尚未过论文门槛：B1 在验证流上不稳定且不校准；B2 的 raw candidate 几乎没有 oracle headroom，并在可用行中大多有害；B3 尚无 held-out calibration artifact，所以 action coverage 为0。建议停止 full-nuScenes 扩展，先完成 step1 梯度/Adam 事务审计和顺序100-step复现。"
                ),
            },
            {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["card_b0_score", "card_b0_parity", "card_b2_harm", "card_b3_coverage"]},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## 1. 范围、数据与指标定义\n\n"
                    "本报告只读取四个 v25 `retryfix` 运行：B0、B1-only、B1+B2、Full。四组均为 nuScenes mini car、seed42、batch16、60 epochs、每轮验证；checkpoint 均为 epoch60 / global_step 75,720。四组 provenance 的代码提交、mini_train/mini_val 清单、安全 evaluator、候选权重与采样协议一致，训练代码没有 tracked dirty。旧 v24、B02x2、历史高分和 `trajtrack` 均未进入计算。\n\n"
                    "跟踪指标只报告 final 与 late-3。逐帧 candidate CSV 用于同 checkpoint 内的 B1/B2/B3 机制诊断；由于 CSV 未覆盖全部2285个验证帧，它不替代 TensorBoard 的绝对 Success/Precision，只用于成对方向与机制供给分析。"
                ),
            },
            {"id": "arm_summary_block", "type": "table", "layout": "full", "tableId": "table_arm_summary"},
            {
                "id": "b0_result",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## 2. B0：分数恢复，但事务未恢复\n\n"
                    "独立 B0 的 final/late-3 都维持在约50 Success，说明先前31分的单条坏轨迹没有再次出现；训练也完整走完60轮。因此可以说 **B0 的性能表象基本回到可用区间**。但正式问题不是某一条 run 是否到50，而是相同 observation 事务在四臂中是否一致。该条件明确失败：initial hash 一致，step1、step100、epoch60 全部失配。\n\n"
                    "曲线进一步说明这不是末期微小浮点差：B1-only 长期停留在约33 Success，而 B0、B1+B2 与 Full 位于约48–53区间。Full 的配置是 `proposal_inference_mode: observation` 且无校准 artifact，其52.553主要是另一条 B0 轨迹，不是 B3 涨点。"
                ),
            },
            {"id": "validation_chart_block", "type": "chart", "layout": "full", "chartId": "chart_validation_success"},
            {
                "id": "root_cause",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## 3. B0 分叉定位\n\n"
                    "现有证据把问题定位到 **第一次 backward/Adam 更新**，而不是 DataLoader 或 validation cadence：四臂 step0 的 view0 loss 完全相同，前100批 observation 指纹完全相同，最终 Python/NumPy/Torch CPU/CUDA RNG 状态也完全相同；但 step1 参数哈希已经不同。B1-only 首个可见 view0 loss 差异出现在 step6，B1+B2/Full 出现在 step3，说明极小的第一次更新差异随后被 mini 训练放大。\n\n"
                    "最可能的两类原因是 PointNet2/自定义 CUDA backward 的非确定性，或 Adam foreach/多参数组在不同拓扑下对 B0 更新产生位级差异。仅凭现有 checkpoint 不能二选一；需要在服务器保存 step1 的 B0 grad hash 和 Adam state hash。若 grad 已不同，根因在 backward；若 grad 相同而参数不同，根因在 optimizer。并行运行（其中 B0 与 B1-only 共用物理GPU）会增加调度扰动，后续验收应同卡顺序执行。"
                ),
            },
            {"id": "b0_audit_block", "type": "table", "layout": "full", "tableId": "table_b0_audit"},
            {
                "id": "b1_result",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## 4. B1：平均位置先验增益弱，验证不稳且不校准\n\n"
                    "训练机制流的 epoch60 平均值里，learned RMSE 比 CV 只好约0.05–0.20 m；到 mini_val epoch60，B1-only 仍仅好0.101 m，但 B1+B2 与 Full 分别反而差0.168 m和0.107 m。更严重的是不确定性失配：B1-only/B1+B2/Full 的95%经验 coverage 仅45.0%/37.5%/55.5%，mean NLL 为120.5/144.3/150.0。\n\n"
                    "因此 B1 目前不能作为可靠的 physical prior gate。优先问题不是扩大网络，而是让训练机制流与 online validation 的误差分布一致，并重新检查 sigma 下界、NLL尺度和递归漂移下的 calibration。"
                ),
            },
            {"id": "b1_chart_block", "type": "chart", "layout": "full", "chartId": "chart_b1_rmse"},
            {
                "id": "b2_b3_result",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## 5. B2 与 B3：证据供给不足，选择器尚未进入部署态\n\n"
                    "B2 的训练采样供应不是空的：epoch60 eligible row recall 为100%，point recall 为40.7%–44.5%，训练 presence AP 为0.808–0.838。但 online mini_val 发生明显分布坍缩：B2 available 仅6.3%–7.6%，其中真正含 extension target 的只有1–2行；presence AP 只有0.042–0.125。可用 raw candidate 的平均中心增益为 -2.71 到 -4.03 m，93.4%–96.7% 会恶化超过0.1 m，oracle Success headroom 仅0.005–0.015点。\n\n"
                    "B3 在训练行上 action AP=0.620、AUROC=0.769，但 epoch60 验证只有17个 evidence-valid 行、仅2个 helpful 行，validation action AP=0.226。更关键的是 `ct_action_calibration_path=null` 且 `ct_require_action_calibration=true`，所以所有验证行都 fail closed：`b3_calibrated=0`、`router_applied_gate=0`。当前 Full 分数不能检验 B3 是否降低 harmful rate。"
                ),
            },
            {"id": "b2_chart_block", "type": "chart", "layout": "full", "chartId": "chart_b2_harmful"},
            {"id": "gate_table_block", "type": "table", "layout": "full", "tableId": "table_gates"},
            {
                "id": "recommendations",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## 6. 推荐的下一步\n\n"
                    "1. **先不要启动 full nuScenes。** 当前四臂不满足已注册的 B0 hash gate，跨臂差值没有因果解释力。\n"
                    "2. **做顺序而非并行的 step1/100 审计。** 同一张空闲A40上依次跑 B0→B1→B1+B2→Full；保存 observation loss、每个 B0 参数的 grad SHA256、Adam `exp_avg/exp_avg_sq/step` SHA256、更新后参数哈希。\n"
                    "3. **先区分 backward 与 optimizer。** 若 grad hash 不同，启用确定性诊断设置并排查 PointNet2 CUDA 原子操作；若 grad 相同，固定 Adam `foreach=False` 做一次100-step对照。所有 smoke checkpoint 继续丢弃。\n"
                    "4. **B0 parity 通过后只重跑 mini 四臂。** B1 必须在 validation 上 learned RMSE < CV 且 coverage 接近名义值；否则先校准 B1，不进入 B2。\n"
                    "5. **B2 优先修供给，不改耦合。** 保持 extension-only 与晚耦合，重点提高 online validation 的 target-bearing retention，并在 candidate 应用前要求正 oracle headroom；当前不应训练第五臂。\n"
                    "6. **B3 最后校准。** 只有 B2 过门槛后，使用独立 calibration tracklets 生成与 checkpoint/config/manifest 绑定的 artifact，再报告 action coverage、risk–coverage 和 harmful-rate reduction。"
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "layout": "full",
                "sourceId": "src_v25_four_arm",
                "body": (
                    "## 7. 方法、限制与可复现性\n\n"
                    "本次结论来自固定目录的 TensorBoard scalar、last.ckpt 审计字段、epoch60 candidate diagnostics 与 acquisition supply。Success/Precision 的绝对值使用框架已记录标量；逐帧 CSV 只做同臂内成对机制分析。报告未使用旧高分作为阈值，也未用冻结的官方 SeqTrack 数值直接对比，因为该参考实现的验证路径会读取当前帧 GT `wlh`，不满足本项目的 safe evaluator 合同。\n\n"
                    "当前只有一个 seed，且四臂 B0 已分叉，所以不能估计稳定增益、paired CI 或论文效果。显存 audit 的 lifetime peak 为 B0 3.894 GiB、插件臂约4.447 GiB，但插件臂峰值包含 mechanism 事务，不能替代“同一 observation batch 对 Safe SeqTrack control”的分阶段显存验收。"
                ),
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": "2026-08-24T00:00:00+08:00",
        "status": "ready",
        "datasets": report_snapshot,
    },
    "sources": [source],
}


analysis_summary = {
    "scope": {"runs": {name: relative(path) for name, path in RUNS.items()}},
    "validation_summary": summary_rows,
    "b0_audit": {
        "rows": audit_rows,
        "prefix_checks": prefix_checks,
        "first_100_fingerprints_equal": fingerprints_equal,
        "final_cpu_rng_equal": len(set(cpu_rng_hashes)) == 1,
        "final_cuda_rng_equal": len(set(cuda_rng_hashes)) == 1,
        "loss_divergence": loss_divergence_rows,
    },
    "b1_validation": b1_rows,
    "b2_validation": b2_rows,
    "b3_validation": b3_row,
    "gate_status": gate_rows,
}

(REPORT_DIR / "analysis_summary.json").write_text(
    json.dumps(analysis_summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)
for name, rows in (
    ("validation_curves.csv", validation_rows),
    ("arm_summary.csv", summary_rows),
    ("b0_audit.csv", audit_rows),
    ("b1_validation.csv", b1_rows),
    ("b2_validation.csv", b2_rows),
):
    with (REPORT_DIR / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print(REPORT_DIR)
