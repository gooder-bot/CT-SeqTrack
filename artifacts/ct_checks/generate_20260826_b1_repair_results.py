"""Build the reproducible report for the 2026-08-25 B1 repair runs.

The script is read-only with respect to ``output/``.  It compares the three
registered v25 runs (B0, B1-GRU and B1-CfC) and uses the newest matching
60-epoch SeqTrack run only as a descriptive historical reference.  Derived
files are written below ``artifacts/ct_checks/reports``.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.report_ct_b1 import load_rows, summarize


REPORT_DIR = (
    ROOT / "artifacts" / "ct_checks" / "reports"
    / "20260826_b1_repair_results"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "B0": {
        "path": ROOT / "output/20260825-1931-25_b0-ct25_b0_mini_car_seed42_60ep_bs16_val5",
        "success_folder": "metrics_mini_val_success",
        "precision_folder": "metrics_mini_val_precision",
        "kind": "current",
    },
    "B1-GRU": {
        "path": ROOT / "output/20260825-1932-25_b1-ct25_b1only_gru_mini_car_seed42_60ep_bs16_val5",
        "success_folder": "metrics_mini_val_success",
        "precision_folder": "metrics_mini_val_precision",
        "kind": "current",
    },
    "B1-CfC": {
        "path": ROOT / "output/20260825-1932-25_b1-ct25_b1only_cfc_mini_car_seed42_60ep_bs16_val5",
        "success_folder": "metrics_mini_val_success",
        "precision_folder": "metrics_mini_val_precision",
        "kind": "current",
    },
    "SeqTrack reference": {
        "path": ROOT / "output/20260813-0116-01_seqtrack3d_baseline-scratch_ct21_b0_car_60ep_bs16_s42",
        "success_folder": "metrics_test_success",
        "precision_folder": "metrics_test_precision",
        "kind": "historical_reference",
    },
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def event_values(event_root: Path, folder: str):
    accumulator = EventAccumulator(
        str(event_root / folder), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if len(tags) != 1:
        raise RuntimeError(
            f"{event_root / folder}: expected one scalar tag, got {tags}")
    return accumulator.Scalars(tags[0])


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


def finite_mean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise RuntimeError("metric series contains non-finite values")
    return float(array.mean())


def ece_three(coverage: dict[str, float]) -> float:
    nominal = {"50": 0.50, "80": 0.80, "95": 0.95}
    return float(np.mean([
        abs(float(coverage[level]) - value)
        for level, value in nominal.items()
    ]))


# ---------------------------------------------------------------------------
# Tracking curves and final/late-3 summaries.
# ---------------------------------------------------------------------------

tracking_curve = []
tracking_summary = []
provenance = {}
for arm, spec in RUNS.items():
    run = spec["path"]
    event_root = run / "lightning_logs/version_0"
    success = event_values(event_root, spec["success_folder"])
    precision = event_values(event_root, spec["precision_folder"])
    runtime = event_values(event_root, "runtime_runtime")
    if not (len(success) == len(precision) == len(runtime) == 12):
        raise RuntimeError(
            f"{arm}: expected 12 validation points, got "
            f"{len(success)}/{len(precision)}/{len(runtime)}")
    for index, (success_item, precision_item, runtime_item) in enumerate(
            zip(success, precision, runtime), start=1):
        epoch = index * 5
        tracking_curve.append({
            "arm": arm,
            "epoch": epoch,
            "global_step": int(success_item.step),
            "success": float(success_item.value),
            "precision": float(precision_item.value),
            "runtime_fps": float(runtime_item.value),
            "final": epoch == 60,
            "late3": epoch >= 50,
        })
    tracking_summary.append({
        "arm": arm,
        "comparison_role": (
            "matched-current arm" if spec["kind"] == "current"
            else "historical descriptive reference"),
        "final_success": float(success[-1].value),
        "late3_success": finite_mean([item.value for item in success[-3:]]),
        "final_precision": float(precision[-1].value),
        "late3_precision": finite_mean([item.value for item in precision[-3:]]),
        "final_runtime_fps": float(runtime[-1].value),
        "validation_points": len(success),
    })
    provenance[arm] = json.loads(
        (run / "run_provenance.json").read_text(encoding="utf-8"))

summary_by_arm = {row["arm"]: row for row in tracking_summary}
for row in tracking_summary:
    for reference, suffix in (("B0", "vs_b0"),
                              ("SeqTrack reference", "vs_seqtrack")):
        row[f"final_success_{suffix}"] = (
            row["final_success"] - summary_by_arm[reference]["final_success"])
        row[f"late3_success_{suffix}"] = (
            row["late3_success"] - summary_by_arm[reference]["late3_success"])
        row[f"final_precision_{suffix}"] = (
            row["final_precision"]
            - summary_by_arm[reference]["final_precision"])
        row[f"late3_precision_{suffix}"] = (
            row["late3_precision"]
            - summary_by_arm[reference]["late3_precision"])


# ---------------------------------------------------------------------------
# Run identity, optimizer and enabled-module audit.
# ---------------------------------------------------------------------------

current_payloads = {}
audit_rows = []
for arm in ("B0", "B1-GRU", "B1-CfC"):
    spec = RUNS[arm]
    payload = checkpoint(
        spec["path"] / "lightning_logs/version_0/checkpoints/last.ckpt")
    current_payloads[arm] = payload
    prefix = payload["ct_b0_prefix_hashes"]
    optimizer_prefix = payload["ct_b0_optimizer_state_hashes"]
    module_audit = payload["ct_module_audit"]
    gradient_max = module_audit.get("max_gradient_norm", {})
    audit_rows.append({
        "arm": arm,
        "epoch": int(payload["epoch"]) + 1,
        "global_step": int(payload["global_step"]),
        "initial_hash": prefix["initial"][:12],
        "step1_hash": prefix["step_1"][:12],
        "step100_hash": prefix["step_100"][:12],
        "adam_initial_hash": optimizer_prefix["initial"][:12],
        "adam_step1_hash": optimizer_prefix["step_1"][:12],
        "adam_step100_hash": optimizer_prefix["step_100"][:12],
        "optimizer_groups": ", ".join(module_audit["parameter_groups"]),
        "frozen_parameters": len(module_audit["active_frozen_parameters"]),
        "b0_max_grad": float(gradient_max.get("b0", 0.0)),
        "b1_max_grad": float(gradient_max.get("b1", 0.0)),
        "fingerprints": len(payload["ct_observation_batch_fingerprints"]),
    })

base_fingerprints = current_payloads["B0"][
    "ct_observation_batch_fingerprints"][:100]
fingerprints_equal = all(
    payload["ct_observation_batch_fingerprints"][:100] == base_fingerprints
    for payload in current_payloads.values())
prefix_equal = {
    key: len({
        payload["ct_b0_prefix_hashes"][key]
        for payload in current_payloads.values()
    }) == 1
    for key in ("initial", "step_1", "step_100")
}
optimizer_prefix_equal = {
    key: len({
        payload["ct_b0_optimizer_state_hashes"][key]
        for payload in current_payloads.values()
    }) == 1
    for key in ("initial", "step_1", "step_100")
}

new_provenance = [provenance[arm] for arm in ("B0", "B1-GRU", "B1-CfC")]
identity_audit = {
    "same_git_commit": len({item["git"]["commit"] for item in new_provenance}) == 1,
    "all_clean_tracked": all(not item["git"]["dirty_tracked"] for item in new_provenance),
    "same_train_selection": len({
        item["datasets"]["train"]["virtual_rate_selection_sha256"]
        for item in new_provenance
    }) == 1,
    "same_val_selection": len({
        item["datasets"]["val"]["virtual_rate_selection_sha256"]
        for item in new_provenance
    }) == 1,
    "same_seed": len({item["resolved_config"]["seed"] for item in new_provenance}) == 1,
    "same_batch_size": len({
        item["resolved_config"]["batch_size"] for item in new_provenance
    }) == 1,
    "same_epochs": len({item["resolved_config"]["epoch"] for item in new_provenance}) == 1,
    "same_validation_cadence": len({
        item["resolved_config"]["check_val_every_n_epoch"]
        for item in new_provenance
    }) == 1,
    "scratch_only": all(
        item["init_checkpoint_path"] is None
        and item["checkpoint_path"] is None
        for item in new_provenance),
    "fingerprints_first100_equal": fingerprints_equal,
    "b0_prefix_equal": prefix_equal,
    "b0_optimizer_prefix_equal": optimizer_prefix_equal,
}


# ---------------------------------------------------------------------------
# B1 epoch-60 mechanism results, coverage and promotion evidence.
# ---------------------------------------------------------------------------

b1_diagnostics = []
b1_rmse_chart = []
coverage_curve = []
endpoint_sets = {}
age_valid_counts = {}
for arm in ("B1-GRU", "B1-CfC"):
    csv_path = (
        RUNS[arm]["path"] / "lightning_logs/version_0"
        / "candidate_diagnostics/epoch_60.csv")
    rows = load_rows(csv_path)
    result = summarize(rows)
    coverage_ece = ece_three(result["coverage"])
    endpoint_sets[arm] = {
        (
            str(row.get("tracklet_key", row.get("tracklet_id"))),
            int(float(row["frame_id"])),
            int(float(row.get("candidate_id", 0))),
        )
        for row in rows
    }
    age_valid_counts[arm] = sum(
        float(row.get("recursive_age_valid", 1.0)) > 0 for row in rows)
    bootstrap = result["learned_vs_cv_paired_bootstrap"]
    quantiles = result["quantiles"]
    b1_diagnostics.append({
        "arm": arm,
        "csv_rows": len(rows),
        "valid_rows": int(result["count"]),
        "learned_rmse": float(result["learned_rmse"]),
        "cv_rmse": float(result["cv_rmse"]),
        "learned_minus_cv_rmse": float(result["learned_minus_cv_rmse"]),
        "relative_mse_improvement": float(
            1.0 - result["learned_rmse"] ** 2 / result["cv_rmse"] ** 2),
        "paired_ci_low": float(bootstrap["ci95"][0]),
        "paired_ci_high": float(bootstrap["ci95"][1]),
        "paired_ci_pass": bool(bootstrap["upper_lt_zero"]),
        "tracklets": int(bootstrap["tracklets"]),
        "help_rate": float(result["paired_help_rate"]),
        "nll": float(result["nll"]),
        "coverage_50": float(result["coverage"]["50"]),
        "coverage_80": float(result["coverage"]["80"]),
        "coverage_95": float(result["coverage"]["95"]),
        "coverage_ece": coverage_ece,
        "coverage_gate_pass": bool(
            coverage_ece <= 0.05 and result["coverage"]["95"] >= 0.90),
        "top1pct_nll_share": float(result["top1pct_nll_share"]),
        "recoverable_saturation_rate": float(
            result["recoverable_saturation_rate"]),
        "tail_axis_fraction": float(result["tail_axis_fraction"]),
        "sigma_parallel_p50": float(
            quantiles["sigma_parallel"]["p50"]),
        "sigma_perpendicular_p50": float(
            quantiles["sigma_perpendicular"]["p50"]),
        "residual_parallel_p50": float(
            quantiles["residual_unit_parallel"]["p50"]),
        "residual_perpendicular_p50": float(
            quantiles["residual_unit_perpendicular"]["p50"]),
        "recursive_age_valid_rows": int(age_valid_counts[arm]),
    })
    for metric, value in (("Learned", result["learned_rmse"]),
                          ("CV", result["cv_rmse"])):
        b1_rmse_chart.append({
            "arm": arm,
            "metric": metric,
            "rmse": float(value),
        })
    for point in result.get("coverage_curve", []):
        coverage_curve.append({
            "arm": arm,
            "nominal": float(point["nominal"]),
            "observed": float(point["observed"]),
        })

endpoints_identical = endpoint_sets["B1-GRU"] == endpoint_sets["B1-CfC"]


# ---------------------------------------------------------------------------
# Final-epoch training mechanism scalars.  There are 213 mechanism batches per
# epoch; the final 213 records therefore form the registered epoch-60 slice.
# ---------------------------------------------------------------------------

TRAINING_FOLDERS = {
    "main_learned": "loss_motion_v3_prior_rmse",
    "main_cv": "loss_motion_v3_kinematic_rmse",
    "gap2_learned": "loss_motion_v3_aux_prior_rmse_gap2",
    "gap2_cv": "loss_motion_v3_aux_kinematic_rmse_gap2",
    "gap4_learned": "loss_motion_v3_aux_prior_rmse_gap4",
    "gap4_cv": "loss_motion_v3_aux_kinematic_rmse_gap4",
    "coverage_95": "loss_motion_v3_coverage_95",
    "coverage_ece": "loss_motion_v3_coverage_ece",
    "sigma_parallel": "loss_motion_v3_sigma_parallel_mean",
    "sigma_perpendicular": "loss_motion_v3_sigma_perpendicular_mean",
    "saturation": "loss_motion_v3_recoverable_saturation_rate",
    "tail_axis_fraction": "loss_motion_v3_tail_axis_fraction",
}

training_epoch60 = []
training_gap_chart = []
for arm in ("B1-GRU", "B1-CfC"):
    event_root = RUNS[arm]["path"] / "lightning_logs/version_0"
    values = {}
    for key, folder in TRAINING_FOLDERS.items():
        events = event_values(event_root, folder)
        if len(events) != 12780:
            raise RuntimeError(f"{arm}/{folder}: expected 12780 events")
        values[key] = finite_mean([item.value for item in events[-213:]])
    training_epoch60.append({"arm": arm, **values})
    for gap in ("main", "gap2", "gap4"):
        for metric in ("learned", "cv"):
            training_gap_chart.append({
                "backend_gap": f"{arm.removeprefix('B1-')} {gap}",
                "metric": metric.upper() if metric == "cv" else "Learned",
                "rmse": values[f"{gap}_{metric}"],
            })


# ---------------------------------------------------------------------------
# Gate verdicts and reproducible snapshot.
# ---------------------------------------------------------------------------

diag = {row["arm"]: row for row in b1_diagnostics}
gru_track = summary_by_arm["B1-GRU"]
b0_track = summary_by_arm["B0"]
seqtrack = summary_by_arm["SeqTrack reference"]

gate_status = [
    {
        "gate": "Run integrity",
        "criterion": "同 commit/data/seed/cadence，scratch，所有启用模块有梯度且无冻结",
        "observed": "PASS；三组 epoch60/global_step75720，B0/B1 optimizer groups 均参与",
        "verdict": "PASS",
    },
    {
        "gate": "B0 matched-prefix",
        "criterion": "initial、step1、step100 B0 参数与 Adam 状态跨臂一致",
        "observed": "仅 initial 一致；step1 起参数与 Adam hash 均分叉",
        "verdict": "FAIL",
    },
    {
        "gate": "Mean residual",
        "criterion": "learned−CV tracklet bootstrap 95% CI 上界 < 0",
        "observed": (
            f"GRU {diag['B1-GRU']['learned_minus_cv_rmse']:.3f} m, "
            f"CI [{diag['B1-GRU']['paired_ci_low']:.3f}, "
            f"{diag['B1-GRU']['paired_ci_high']:.3f}]；"
            f"CfC {diag['B1-CfC']['learned_minus_cv_rmse']:.3f} m, "
            f"CI [{diag['B1-CfC']['paired_ci_low']:.3f}, "
            f"{diag['B1-CfC']['paired_ci_high']:.3f}]"),
        "verdict": "PASS",
    },
    {
        "gate": "Sigma/coverage",
        "criterion": "独立校准后 ECE≤5%、coverage95≥90%、NLL 优于 fixed sigma",
        "observed": (
            f"raw GRU ECE {diag['B1-GRU']['coverage_ece']:.1%}/C95 "
            f"{diag['B1-GRU']['coverage_95']:.1%}；raw CfC ECE "
            f"{diag['B1-CfC']['coverage_ece']:.1%}/C95 "
            f"{diag['B1-CfC']['coverage_95']:.1%}；无 calibration/dev artifact"),
        "verdict": "NOT PASSED",
    },
    {
        "gate": "Long-tail supervision",
        "criterion": "gap2/gap4 有效训练，并可按真实 recursive_age 分层验收",
        "observed": "gap2/gap4 epoch60 learned RMSE 均优于 CV；验证 CSV 的 age-valid 行为 0",
        "verdict": "PARTIAL",
    },
    {
        "gate": "CfC promotion",
        "criterion": "相同 endpoint 的 CfC−GRU RMSE CI<0，且 NLL/coverage 不劣",
        "observed": (
            f"endpoint 集不一致（GRU {len(endpoint_sets['B1-GRU'])} / "
            f"CfC {len(endpoint_sets['B1-CfC'])}），且 B0 step1 分叉"),
        "verdict": "FAIL / KEEP GRU",
    },
]

headline = [{
    "gru_final_success_delta_b0": (
        gru_track["final_success"] - b0_track["final_success"]),
    "gru_mean_delta_cv_m": diag["B1-GRU"]["learned_minus_cv_rmse"],
    "gru_coverage95": diag["B1-GRU"]["coverage_95"],
    "b0_prefix_gate": "FAIL",
    "overall_verdict": "部分成功，尚未达到论文实验晋升门槛",
}]

report_snapshot = {
    "headline": headline,
    "tracking_curve": tracking_curve,
    "tracking_summary": tracking_summary,
    "b1_diagnostics": b1_diagnostics,
    "b1_rmse_chart": b1_rmse_chart,
    "coverage_curve": coverage_curve,
    "training_epoch60": training_epoch60,
    "training_gap_chart": training_gap_chart,
    "audit_rows": audit_rows,
    "gate_status": gate_status,
}

report_db = REPORT_DIR / "report_data.sqlite"
report_query = (
    "SELECT snapshot_json FROM report_snapshot "
    "WHERE snapshot_id = 'b1_repair_20260826'")
with sqlite3.connect(report_db) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS report_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)")
    connection.execute(
        "INSERT OR REPLACE INTO report_snapshot(snapshot_id, snapshot_json) "
        "VALUES (?, ?)",
        ("b1_repair_20260826", json.dumps(
            report_snapshot, ensure_ascii=False, allow_nan=False)))
    selected = connection.execute(report_query).fetchone()
if selected is None or json.loads(selected[0]) != report_snapshot:
    raise RuntimeError("SQLite report snapshot verification failed")

source_paths = []
for spec in RUNS.values():
    source_paths.extend([
        relative(spec["path"] / "run_provenance.json"),
        relative(spec["path"] / "lightning_logs/version_0/hparams.yaml"),
    ])
for arm in ("B0", "B1-GRU", "B1-CfC"):
    source_paths.append(relative(
        RUNS[arm]["path"]
        / "lightning_logs/version_0/checkpoints/last.ckpt"))
for arm in ("B1-GRU", "B1-CfC"):
    source_paths.append(relative(
        RUNS[arm]["path"] / "lightning_logs/version_0"
        / "candidate_diagnostics/epoch_60.csv"))

source = {
    "id": "src_b1_repair_20260826",
    "label": "三组 v25 B1 修复实验与 SeqTrack 历史参考的可复核快照",
    "path": relative(report_db),
    "query": {
        "engine": "sqlite",
        "sql": report_query,
        "tablesUsed": ["report_snapshot"],
    },
}

title = "CT-SeqTrack B1 修复实验结果（2026-08-26）"
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "B0/B1-GRU/B1-CfC、正常 SeqTrack、B1 均值与 uncertainty/long-tail 门槛审计。",
        "generatedAt": "2026-08-26T00:00:00+08:00",
        "sources": [source],
        "cards": [
            {
                "id": "card_gru_delta",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "仅为跨独立训练轨迹的描述性差值，不能解释为 B1 因果增益。",
                "metrics": [{
                    "label": "GRU final S vs B0",
                    "field": "gru_final_success_delta_b0",
                    "format": "number",
                }],
            },
            {
                "id": "card_gru_mean",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "epoch60 mini_val 逐 tracklet paired bootstrap 的点估计。",
                "metrics": [{
                    "label": "GRU learned−CV RMSE",
                    "field": "gru_mean_delta_cv_m",
                    "format": "number",
                }],
            },
            {
                "id": "card_coverage",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "未经独立 post-hoc calibration 的 raw 95% coverage。",
                "metrics": [{
                    "label": "GRU raw coverage95",
                    "field": "gru_coverage95",
                    "format": "percent",
                }],
            },
            {
                "id": "card_parity",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "B0 参数和 Adam state 在 step1 已经跨臂失配。",
                "metrics": [
                    {"label": "Matched-prefix", "field": "b0_prefix_gate"},
                    {"label": "Overall", "field": "overall_verdict"},
                ],
            },
        ],
        "charts": [
            {
                "id": "chart_success",
                "title": "mini_val Success 曲线",
                "subtitle": "每 5 epoch 验证；SeqTrack 为旧提交/旧配置的描述性参考",
                "intent": "trend",
                "question": "三条新运行与正常 SeqTrack 的收敛轨迹如何？",
                "rationale": "12 个同 cadence 验证点显示末期差值是否只是单点波动。",
                "type": "line",
                "dataset": "tracking_curve",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"},
                    "color": {"field": "arm", "type": "nominal", "label": "Run"},
                    "tooltip": [
                        {"field": "precision", "type": "quantitative", "label": "Precision"},
                        {"field": "runtime_fps", "type": "quantitative", "label": "FPS"},
                    ],
                },
                "layout": "full",
            },
            {
                "id": "chart_precision",
                "title": "mini_val Precision 曲线",
                "subtitle": "每 5 epoch 验证；单位为点",
                "intent": "trend",
                "question": "Precision 是否支持与 Success 相同的方向判断？",
                "rationale": "Success 与 Precision 同向时，描述性差值更不易由单一指标定义造成。",
                "type": "line",
                "dataset": "tracking_curve",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "precision", "type": "quantitative", "label": "Precision", "unit": "points"},
                    "color": {"field": "arm", "type": "nominal", "label": "Run"},
                    "tooltip": [{"field": "success", "type": "quantitative", "label": "Success"}],
                },
                "layout": "full",
            },
            {
                "id": "chart_b1_rmse",
                "title": "epoch60 B1 learned prior 与 CV RMSE",
                "subtitle": "仅 b1_valid=1 的中心误差；越低越好，单位 m",
                "intent": "comparison",
                "question": "修复后的均值头是否真正优于其固定运动学 anchor？",
                "rationale": "同一 backend 内 learned/CV 是有效的逐 endpoint 配对比较。",
                "type": "bar",
                "dataset": "b1_rmse_chart",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "arm", "type": "nominal", "label": "Backend"},
                    "y": {"field": "rmse", "type": "quantitative", "label": "RMSE", "unit": "m"},
                    "color": {"field": "metric", "type": "nominal", "label": "Method"},
                },
                "layout": "full",
            },
            {
                "id": "chart_coverage",
                "title": "epoch60 raw coverage 曲线",
                "subtitle": "虚线理想关系由 nominal=observed 表示；当前数据尚未独立校准",
                "intent": "trend",
                "question": "sigma 给出的置信域是否与经验覆盖率匹配？",
                "rationale": "多置信水平曲线比单个 coverage95 更清楚地显示欠/过离散。",
                "type": "line",
                "dataset": "coverage_curve",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "nominal", "type": "quantitative", "format": "percent", "label": "Nominal coverage"},
                    "y": {"field": "observed", "type": "quantitative", "format": "percent", "label": "Observed coverage"},
                    "color": {"field": "arm", "type": "nominal", "label": "Backend"},
                },
                "layout": "full",
            },
            {
                "id": "chart_training_gaps",
                "title": "epoch60 训练流 main/gap2/gap4 RMSE",
                "subtitle": "每项为最后 213 个 mechanism batch 均值；越低越好",
                "intent": "comparison",
                "question": "gap2/gap4 辅助监督是否实际参与并产生可学习信号？",
                "rationale": "分组柱图同时显示两种 backend 在三个 gap 上的 learned/CV 差值。",
                "type": "bar",
                "dataset": "training_gap_chart",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "backend_gap", "type": "nominal", "label": "Backend / gap"},
                    "y": {"field": "rmse", "type": "quantitative", "label": "RMSE", "unit": "m"},
                    "color": {"field": "metric", "type": "nominal", "label": "Method"},
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "table_tracking",
                "title": "Tracking final 与 late-3",
                "subtitle": "final=epoch60；late-3=epoch50/55/60 均值；不选 best epoch",
                "dataset": "tracking_summary",
                "sourceId": source["id"],
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "arm", "label": "Run", "type": "text"},
                    {"field": "final_success", "label": "Final S", "format": "number"},
                    {"field": "late3_success", "label": "Late-3 S", "format": "number"},
                    {"field": "final_precision", "label": "Final P", "format": "number"},
                    {"field": "late3_precision", "label": "Late-3 P", "format": "number"},
                    {"field": "final_success_vs_b0", "label": "Final S ΔB0", "format": "number"},
                    {"field": "final_success_vs_seqtrack", "label": "Final S ΔSeqTrack", "format": "number"},
                ],
            },
            {
                "id": "table_b1",
                "title": "B1 epoch60 机制与 uncertainty 指标",
                "subtitle": "paired CI 以 tracklet 为 bootstrap 单位；coverage 均为未校准 raw 输出",
                "dataset": "b1_diagnostics",
                "sourceId": source["id"],
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "arm", "label": "Backend", "type": "text"},
                    {"field": "learned_rmse", "label": "Learned RMSE", "format": "number"},
                    {"field": "cv_rmse", "label": "CV RMSE", "format": "number"},
                    {"field": "learned_minus_cv_rmse", "label": "ΔRMSE", "format": "number"},
                    {"field": "paired_ci_low", "label": "CI low", "format": "number"},
                    {"field": "paired_ci_high", "label": "CI high", "format": "number"},
                    {"field": "help_rate", "label": "Help rate", "format": "percent"},
                    {"field": "nll", "label": "Raw NLL", "format": "number"},
                    {"field": "coverage_95", "label": "Coverage95", "format": "percent"},
                    {"field": "coverage_ece", "label": "ECE", "format": "percent"},
                    {"field": "recursive_age_valid_rows", "label": "Age-valid", "format": "number"},
                ],
            },
            {
                "id": "table_audit",
                "title": "B0/optimizer 前缀与训练参与审计",
                "subtitle": "短 hash 仅用于可视化；完整 SHA256 保留在 last.ckpt",
                "dataset": "audit_rows",
                "sourceId": source["id"],
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "arm", "label": "Run", "type": "text"},
                    {"field": "initial_hash", "label": "B0 initial", "type": "text"},
                    {"field": "step1_hash", "label": "B0 step1", "type": "text"},
                    {"field": "step100_hash", "label": "B0 step100", "type": "text"},
                    {"field": "adam_step1_hash", "label": "Adam step1", "type": "text"},
                    {"field": "optimizer_groups", "label": "Groups", "type": "text"},
                    {"field": "frozen_parameters", "label": "Frozen", "format": "number"},
                    {"field": "b0_max_grad", "label": "B0 max grad", "format": "number"},
                    {"field": "b1_max_grad", "label": "B1 max grad", "format": "number"},
                ],
            },
            {
                "id": "table_gates",
                "title": "本轮修改验收门槛",
                "subtitle": "按预注册计划逐项判断，不以单个 tracking 分数替代机制门槛",
                "dataset": "gate_status",
                "sourceId": source["id"],
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "gate", "label": "Gate", "type": "text"},
                    {"field": "criterion", "label": "Criterion", "type": "text"},
                    {"field": "observed", "label": "Observed", "type": "text"},
                    {"field": "verdict", "label": "Verdict", "type": "text"},
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
                    "## 技术结论\n\n"
                    "**这次修改是部分成功，不是完整成功。** 均值残差修复已得到明确正证据：GRU 与 CfC 在各自 epoch60 mini_val 上都显著优于固定 CV anchor，tracklet paired-bootstrap 的 95% CI 全部低于 0；gap2/gap4 在训练末期也都由 learned prior 优于 CV，说明辅助长间隔损失已真正进入训练。\n\n"
                    "但还不能宣称 B1 给 tracking 涨分。GRU 相对独立 B0 的 final Success/Precision 表面上为正，但三组 B0 虽有相同初始化和前100批输入指纹，第一次 Adam 更新后的参数与 optimizer-state hash 已不同；而且 B1-only 的最终跟踪合同仍是 observation 输出。因此该差值只是独立训练轨迹的描述，不能归因于 B1。Sigma/coverage 也尚未达标，recursive-age 分层数据缺失。当前正式选择应继续保留 **GRU**，CfC 不晋升。"),
            },
            {"id": "cards", "type": "metric-strip", "layout": "full", "cardIds": ["card_gru_delta", "card_gru_mean", "card_coverage", "card_parity"]},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 1. 范围、数据与比较边界\n\n"
                    "三组新实验均来自 commit `b8222bb`，nuScenes mini Car，seed42，batch16，60 epochs，每5轮验证，mini_train/mini_val 分别为274/106条 tracklet，均从 epoch0 随机初始化；没有加载 checkpoint，没有 frozen 参数。正常 SeqTrack 参考为 2026-08-13 的60-epoch、batch16、seed42 scratch run，数据清单相同，但代码提交与配置族不同，所以只用于描述当前分数距离，不作为匹配因果 control。逐帧 B1 CSV 只覆盖导出的 candidate endpoint，不替代官方 TensorBoard tracking 指标。"),
            },
            {"id": "tracking_table", "type": "table", "layout": "full", "tableId": "table_tracking"},
            {"id": "success_chart", "type": "chart", "layout": "full", "chartId": "chart_success"},
            {"id": "precision_chart", "type": "chart", "layout": "full", "chartId": "chart_precision"},
            {
                "id": "tracking_interpretation",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 2. B1 相对 B0 是否提升\n\n"
                    f"按 final 看，GRU 比 B0 高 {gru_track['final_success'] - b0_track['final_success']:.3f} Success / {gru_track['final_precision'] - b0_track['final_precision']:.3f} Precision；late-3 分别高 {gru_track['late3_success'] - b0_track['late3_success']:.3f} / {gru_track['late3_precision'] - b0_track['late3_precision']:.3f}。CfC 则在 final 比 B0 低 {abs(summary_by_arm['B1-CfC']['final_success'] - b0_track['final_success']):.3f} / {abs(summary_by_arm['B1-CfC']['final_precision'] - b0_track['final_precision']):.3f}。这些数值只说明三条 run 的结果排序，不是 B1 的因果涨分。\n\n"
                    f"与正常 SeqTrack 参考相比，当前 B0/GRU/CfC 的 final Success 分别低 {abs(b0_track['final_success'] - seqtrack['final_success']):.3f}、{abs(gru_track['final_success'] - seqtrack['final_success']):.3f}、{abs(summary_by_arm['B1-CfC']['final_success'] - seqtrack['final_success']):.3f} 点。当前主要异常仍是新代码线三条 B0 训练轨迹整体偏低且跨臂失配，而不是已经验证的 B1 涨分。"),
            },
            {
                "id": "mean_result",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 3. 均值残差：修复成功\n\n"
                    f"GRU learned RMSE={diag['B1-GRU']['learned_rmse']:.3f} m，CV={diag['B1-GRU']['cv_rmse']:.3f} m，差值 {diag['B1-GRU']['learned_minus_cv_rmse']:.3f} m，95% CI [{diag['B1-GRU']['paired_ci_low']:.3f}, {diag['B1-GRU']['paired_ci_high']:.3f}]；CfC 为 {diag['B1-CfC']['learned_rmse']:.3f} vs {diag['B1-CfC']['cv_rmse']:.3f} m，差值 {diag['B1-CfC']['learned_minus_cv_rmse']:.3f} m，95% CI [{diag['B1-CfC']['paired_ci_low']:.3f}, {diag['B1-CfC']['paired_ci_high']:.3f}]。两者 CI 上界都小于0，达到预注册 mean gate。可恢复样本饱和率均为0，预测残差中位数也没有整体卡死在 ±1，支持“均值残差塌陷已被修复”的判断。"),
            },
            {"id": "b1_chart", "type": "chart", "layout": "full", "chartId": "chart_b1_rmse"},
            {"id": "b1_table", "type": "table", "layout": "full", "tableId": "table_b1"},
            {
                "id": "uncertainty_result",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 4. Sigma / NLL / coverage：明显改善，但尚未过门槛\n\n"
                    f"GRU raw NLL={diag['B1-GRU']['nll']:.3f}、coverage95={diag['B1-GRU']['coverage_95']:.1%}、三点 coverage ECE={diag['B1-GRU']['coverage_ece']:.1%}；CfC 分别为 {diag['B1-CfC']['nll']:.3f}、{diag['B1-CfC']['coverage_95']:.1%}、{diag['B1-CfC']['coverage_ece']:.1%}。相较修复前约31%–46%的 coverage95，方向上已有大幅改善，NLL 也不再出现历史级长尾爆炸；但 raw 输出仍低于 coverage95≥90%、ECE≤5%的门槛。\n\n"
                    "本批结果没有独立 calibration/dev tracklet 产物，也没有 calibrated NLL 与 fixed-sigma NLL 的同集外比较，所以不能把 sigma/NLL/coverage 记为完成。正确下一步是对这两个 scratch checkpoint 分别做 held-out post-hoc calibration，再在独立 dev 上验收；校准 checkpoint 只用于评估，不能回流训练。"),
            },
            {"id": "coverage_chart", "type": "chart", "layout": "full", "chartId": "chart_coverage"},
            {
                "id": "tail_result",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 5. 递归长尾监督：loss 接线成功，age 验收未完成\n\n"
                    "epoch60 训练流中，GRU 的 main/gap2/gap4 learned RMSE 为3.161/6.747/10.911 m，对应 CV 为3.458/7.512/11.133 m；CfC 为3.212/7.188/11.442 m，对应 CV 为3.678/8.178/11.916 m。说明 gap2/gap4 不再只是日志项，而是获得了有效训练信号。\n\n"
                    "但是两个 epoch60 candidate CSV 的 `recursive_age_valid` 都没有一个有效行，官方 B1 report 因空 age stratum 无法生成。因此“真实递归年龄分组等权归约”和 age 分层 RMSE/help-rate 仍无法在验证流中证实。本项只能判为部分完成。"),
            },
            {"id": "gap_chart", "type": "chart", "layout": "full", "chartId": "chart_training_gaps"},
            {
                "id": "backend_result",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 6. GRU 与 CfC：当前保留 GRU\n\n"
                    "绝对 epoch60 指标上，GRU 的 learned RMSE、coverage95、ECE、Success 和 Precision 都优于 CfC；CfC 只在 raw NLL、相对自身 CV 的改善量、help rate 与 top-1% NLL 集中度上略好。更重要的是，两条 backend run 的 B0 从 step1 起不同，导出的 endpoint 集也不同，无法执行计划要求的同 endpoint `CfC−GRU` paired CI。\n\n"
                    "因此这批数据不支持 CfC 晋升。最稳妥且符合预注册规则的选择是：**GRU 继续作为默认正式方案，CfC 保留为可切换消融插件**。"),
            },
            {"id": "audit_table", "type": "table", "layout": "full", "tableId": "table_audit"},
            {"id": "gate_table", "type": "table", "layout": "full", "tableId": "table_gates"},
            {
                "id": "next_steps",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 7. 下一步建议\n\n"
                    "1. 先不要进入 full nuScenes 或 B1+B2 涨分实验；先解决 B0 step1/Adam 跨臂分叉。建议同一张 GPU 顺序运行三臂的 2-step 与 100-step audit，并同时保存每个 B0 参数的 grad hash、Adam `step/exp_avg/exp_avg_sq` hash 和更新后参数 hash。\n"
                    "2. 修复验证数据中 `recursive_age_valid=0` 的传播或导出，并让 `report_ct_b1.py` 对空 age stratum 给出明确 fail-closed 报告，而不是 IndexError。\n"
                    "3. 对 GRU/CfC 当前 scratch checkpoint 分别运行独立 calibration，再用独立 dev 检查 ECE、coverage95、calibrated NLL vs fixed sigma。\n"
                    "4. 完成前两项后，仍以 GRU 为主、CfC 为消融，重新从 epoch0 做匹配 mini 三臂。只有 B0 prefix gate 通过，tracking 差值和 backend paired promotion 才可解释。\n"
                    "5. mini 全部门槛通过后，再从头训练 full-minus-B3；不要迁移本次 B1-only 权重，也不要冻结任何模块。"),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 8. 方法、限制与可复现性\n\n"
                    "Tracking 取 TensorBoard 的第60轮和最后三个验证点（epoch50/55/60），没有选择 best epoch。B1 RMSE/NLL/coverage 来自同 checkpoint 的 epoch60 candidate CSV；mean gate 使用 tracklet-cluster bootstrap 2,000次。训练 gap 指标为最后一个 epoch 的213个 mechanism batch 均值。Checkpoint 用于核对初始/step1/step100 B0 与 Adam hash、optimizer groups、梯度和 frozen 参数。\n\n"
                    "主要限制是单 seed、mini 数据、B0 matched-prefix 失败、GRU/CfC endpoint 不一致、没有 held-out calibration/dev、recursive-age 字段无效，以及正常 SeqTrack 参考来自不同 commit/config。因此本报告不支持涨分、SOTA、CfC 优越或完整 calibration 成功的论文声明。"),
            },
            {
                "id": "questions",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 9. 仍需回答的问题\n\n"
                    "- B0 的第一次梯度已经不同，还是梯度相同但 Adam 多参数组更新不同？\n"
                    "- `recursive_age_valid` 是在 online validation 状态构造、batch contract 还是 CSV 导出阶段丢失？\n"
                    "- held-out parallel/perpendicular log-scale 校准后，coverage95 与 ECE 能否同时过门槛，并且 calibrated NLL 胜过 fixed sigma？\n"
                    "- B0 prefix gate 通过后，GRU 的小幅 tracking 正差是否仍然存在？"),
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": "2026-08-26T00:00:00+08:00",
        "status": "ready",
        "datasets": report_snapshot,
    },
    "sources": [source],
}

analysis_summary = {
    "scope": {
        "runs": {name: relative(spec["path"]) for name, spec in RUNS.items()},
        "source_paths": source_paths,
    },
    "identity_audit": identity_audit,
    "tracking_summary": tracking_summary,
    "b1_diagnostics": b1_diagnostics,
    "training_epoch60": training_epoch60,
    "backend_comparison": {
        "endpoint_sets_identical": endpoints_identical,
        "endpoint_counts": {
            arm: len(endpoints) for arm, endpoints in endpoint_sets.items()},
        "promotion_computable": bool(
            endpoints_identical and prefix_equal["step_1"]),
        "recommended_backend": "gru",
    },
    "gate_status": gate_status,
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
