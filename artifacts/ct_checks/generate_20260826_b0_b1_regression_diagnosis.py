"""Diagnose the B0/B1 score regression after the 2026-08-25 B1 repair.

Experiment outputs are read-only.  The script writes a bounded technical
report and machine-readable evidence below ``artifacts/ct_checks/reports``.
"""

from __future__ import annotations

import json
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
    / "20260826_b0_b1_regression_diagnosis"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "SeqTrack reference": ROOT / "output/20260813-0116-01_seqtrack3d_baseline-scratch_ct21_b0_car_60ep_bs16_s42",
    "v25 B0 prior-high": ROOT / "output/20260824-0219-25_b0-ct25_b0_mini_car_60ep_bs16_seed42_retryfix",
    "v25 B1 prior-high@30": ROOT / "output/20260825-0057-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_rerun_20260825",
    "v25 B0 current-low": ROOT / "output/20260825-1931-25_b0-ct25_b0_mini_car_seed42_60ep_bs16_val5",
    "v25 B1-GRU current": ROOT / "output/20260825-1932-25_b1-ct25_b1only_gru_mini_car_seed42_60ep_bs16_val5",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def event_values(run: Path, folder: str):
    root = run / "lightning_logs/version_0" / folder
    accumulator = EventAccumulator(str(root), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if len(tags) != 1:
        raise RuntimeError(f"{root}: expected one scalar tag, got {tags}")
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


def load_checkpoint(run: Path):
    install_easydict_pickle_stub()
    return torch.load(
        run / "lightning_logs/version_0/checkpoints/last.ckpt",
        map_location="cpu", weights_only=False)


def provenance(run: Path):
    return json.loads((run / "run_provenance.json").read_text(encoding="utf-8"))


def tracking(run: Path, success_folder: str, precision_folder: str, cadence: int):
    success = event_values(run, success_folder)
    precision = event_values(run, precision_folder)
    if len(success) != len(precision):
        raise RuntimeError(f"{run}: Success/Precision event counts differ")
    return [{
        "epoch": (index + 1) * cadence,
        "success": float(success_item.value),
        "precision": float(precision_item.value),
        "global_step": int(success_item.step),
    } for index, (success_item, precision_item) in enumerate(
        zip(success, precision))]


def epoch_loss(run: Path):
    events = event_values(run, "loss_loss_b0_transaction")
    if len(events) % 1262:
        raise RuntimeError(f"{run}: B0 loss event count is not epoch-aligned")
    values = np.asarray([item.value for item in events], dtype=np.float64)
    return [{
        "epoch": epoch + 1,
        "mean_b0_loss": float(values[epoch * 1262:(epoch + 1) * 1262].mean()),
    } for epoch in range(len(events) // 1262)]


prov = {name: provenance(path) for name, path in RUNS.items()}
payloads = {
    name: load_checkpoint(path)
    for name, path in RUNS.items()
    if name != "SeqTrack reference"
}

curves = {
    "SeqTrack reference": tracking(
        RUNS["SeqTrack reference"], "metrics_test_success",
        "metrics_test_precision", 5),
    "v25 B0 prior-high": tracking(
        RUNS["v25 B0 prior-high"], "metrics_mini_val_success",
        "metrics_mini_val_precision", 1),
    "v25 B1 prior-high@30": tracking(
        RUNS["v25 B1 prior-high@30"], "metrics_mini_val_success",
        "metrics_mini_val_precision", 1),
    "v25 B0 current-low": tracking(
        RUNS["v25 B0 current-low"], "metrics_mini_val_success",
        "metrics_mini_val_precision", 5),
    "v25 B1-GRU current": tracking(
        RUNS["v25 B1-GRU current"], "metrics_mini_val_success",
        "metrics_mini_val_precision", 5),
}


def at_epoch(name: str, epoch: int):
    return next(row for row in curves[name] if row["epoch"] == epoch)


tracking_curve = []
for name in ("SeqTrack reference", "v25 B0 prior-high", "v25 B0 current-low"):
    for row in curves[name]:
        if row["epoch"] % 5 == 0:
            tracking_curve.append({"run": name, **row})

b1_tracking_curve = []
for name in ("v25 B1 prior-high@30", "v25 B1-GRU current"):
    for row in curves[name]:
        if row["epoch"] % 5 == 0:
            b1_tracking_curve.append({"run": name, **row})

b0_loss_curve = []
for name in ("v25 B0 prior-high", "v25 B0 current-low"):
    for row in epoch_loss(RUNS[name]):
        b0_loss_curve.append({"run": name, **row})


def final_late3(name: str):
    rows = curves[name]
    tail = rows[-3:]
    return {
        "run": name,
        "final_epoch": rows[-1]["epoch"],
        "final_success": rows[-1]["success"],
        "final_precision": rows[-1]["precision"],
        "late3_success": float(np.mean([row["success"] for row in tail])),
        "late3_precision": float(np.mean([row["precision"] for row in tail])),
        "validation_points": len(rows),
    }


tracking_summary = [final_late3(name) for name in curves]
summary_by_name = {row["run"]: row for row in tracking_summary}


# Initial state/input/loss equality and the exact first visible divergence.
old_b0 = payloads["v25 B0 prior-high"]
new_b0 = payloads["v25 B0 current-low"]
old_b1 = payloads["v25 B1 prior-high@30"]
new_b1 = payloads["v25 B1-GRU current"]


def prefix(payload, key):
    return payload["ct_b0_prefix_hashes"][key]


old_losses = event_values(RUNS["v25 B0 prior-high"], "loss_loss_b0_transaction")
new_losses = event_values(RUNS["v25 B0 current-low"], "loss_loss_b0_transaction")
first_loss_diff = next(
    (left, right) for left, right in zip(old_losses, new_losses)
    if left.step != right.step or left.value != right.value)

b0_chain = [
    {
        "stage": "Initialization",
        "prior_high": prefix(old_b0, "initial")[:12],
        "current_low": prefix(new_b0, "initial")[:12],
        "equal": prefix(old_b0, "initial") == prefix(new_b0, "initial"),
        "meaning": "B0 architecture and seeded initialization match",
    },
    {
        "stage": "First 100 observation fingerprints",
        "prior_high": str(len(old_b0["ct_observation_batch_fingerprints"])),
        "current_low": str(len(new_b0["ct_observation_batch_fingerprints"])),
        "equal": (
            old_b0["ct_observation_batch_fingerprints"][:100]
            == new_b0["ct_observation_batch_fingerprints"][:100]),
        "meaning": "Candidate IDs, point samples and observation batches match",
    },
    {
        "stage": "First logged B0 loss (step0)",
        "prior_high": f"{old_losses[0].value:.9f}",
        "current_low": f"{new_losses[0].value:.9f}",
        "equal": old_losses[0].value == new_losses[0].value,
        "meaning": "The first forward and loss reduction match",
    },
    {
        "stage": "B0 parameters after optimizer step1",
        "prior_high": prefix(old_b0, "step_1")[:12],
        "current_low": prefix(new_b0, "step_1")[:12],
        "equal": prefix(old_b0, "step_1") == prefix(new_b0, "step_1"),
        "meaning": "The regression begins in backward/Adam, before validation",
    },
    {
        "stage": f"First visible loss difference (step{first_loss_diff[0].step})",
        "prior_high": f"{first_loss_diff[0].value:.9f}",
        "current_low": f"{first_loss_diff[1].value:.9f}",
        "equal": False,
        "meaning": "A tiny update difference becomes visible three batches later",
    },
    {
        "stage": "mini_val Success at epoch5",
        "prior_high": f"{at_epoch('v25 B0 prior-high', 5)['success']:.3f}",
        "current_low": f"{at_epoch('v25 B0 current-low', 5)['success']:.3f}",
        "equal": False,
        "meaning": "Recursive validation amplifies the early numerical split",
    },
]


# Resolved-config comparison.  The YAML content hash stayed the same, but CLI
# validation cadence and newly introduced dormant B1 identity fields changed.
old_cfg = prov["v25 B0 prior-high"]["resolved_config"]
new_cfg = prov["v25 B0 current-low"]["resolved_config"]
resolved_diffs = [{
    "field": key,
    "prior_high": old_cfg.get(key),
    "current_low": new_cfg.get(key),
    "active_for_b0": key == "check_val_every_n_epoch",
} for key in sorted(set(old_cfg) | set(new_cfg))
    if old_cfg.get(key) != new_cfg.get(key)]


baseline_contract = [
    {
        "aspect": "B0 network implementation",
        "SeqTrack_reference": "SEQTRACK3D",
        "current_v25_B0": "CTSEQTRACK subclass of SEQTRACK3D; B1/B2/B3 disabled",
        "equivalent": "Mostly yes",
        "impact": "Core observation model is inherited",
    },
    {
        "aspect": "Training candidate objective",
        "SeqTrack_reference": "4 candidates through the ordinary batch mean",
        "current_v25_B0": "balanced candidate batches; 0.5*view0 + (view1+view2+view3)/6",
        "equivalent": "No",
        "impact": "Different optimization target",
    },
    {
        "aspect": "Sampling/RNG",
        "SeqTrack_reference": "ordinary shuffled DataLoader and global RNG",
        "current_v25_B0": "stateless candidate IDs and point-sampling fingerprints",
        "equivalent": "No",
        "impact": "Safer auditability, but not the original training transaction",
    },
    {
        "aspect": "Validation box size",
        "SeqTrack_reference": "current-frame GT size when safe flag is absent",
        "current_v25_B0": "first-frame size propagated through recursive predictions",
        "equivalent": "No",
        "impact": "v25 removes a GT-size dependency; scores are not the same metric",
    },
    {
        "aspect": "Runtime/optimizer envelope",
        "SeqTrack_reference": "standard Lightning path",
        "current_v25_B0": "safe_seqtrack_auto_v1, named unified Adam group, dual-stream envelope",
        "equivalent": "No",
        "impact": "B0-only still has one group, but the experiment contract differs",
    },
]


def basic_b1(run_name: str, epoch: int):
    rows = load_rows(
        RUNS[run_name] / "lightning_logs/version_0/candidate_diagnostics"
        / f"epoch_{epoch:02d}.csv")
    result = summarize(rows)
    return {
        "run": run_name,
        "epoch": epoch,
        "tracking_success": at_epoch(run_name, epoch)["success"],
        "tracking_precision": at_epoch(run_name, epoch)["precision"],
        "learned_rmse": float(result["learned_rmse"]),
        "cv_rmse": float(result["cv_rmse"]),
        "learned_minus_cv": float(result["learned_minus_cv_rmse"]),
        "help_rate": float(result["paired_help_rate"]),
        "nll": float(result["nll"]),
        "coverage95": float(result["coverage"]["95"]),
        "deployment_output": "B0 observation",
    }


b1_mechanism = [
    basic_b1("v25 B1 prior-high@30", 30),
    basic_b1("v25 B1-GRU current", 30),
    basic_b1("v25 B1-GRU current", 60),
]


run_identity = []
for name in RUNS:
    p = prov[name]
    cfg = p["resolved_config"]
    payload = payloads.get(name)
    run_identity.append({
        "run": name,
        "commit": p["git"]["commit"][:8],
        "config": p["config_path"],
        "resolved_sha": p["resolved_config_sha256"][:12],
        "seed": int(cfg["seed"]),
        "workers": int(cfg["workers"]),
        "validation_cadence": int(cfg["check_val_every_n_epoch"]),
        "evaluator": str(cfg.get("ct_evaluator_identity") or "legacy/current-frame-size"),
        "initial_b0_hash": (
            prefix(payload, "initial")[:12] if payload is not None else "not recorded"),
        "step1_b0_hash": (
            prefix(payload, "step_1")[:12] if payload is not None else "not recorded"),
    })


loss_old = np.asarray(
    [row["mean_b0_loss"] for row in epoch_loss(RUNS["v25 B0 prior-high"])])
loss_new = np.asarray(
    [row["mean_b0_loss"] for row in epoch_loss(RUNS["v25 B0 current-low"])])
b0_loss_summary = [{
    "run": "v25 B0 prior-high",
    "epoch1_mean": float(loss_old[0]),
    "epoch60_mean": float(loss_old[-1]),
    "last1000_mean": float(np.mean(
        [item.value for item in old_losses[-1000:]])),
}, {
    "run": "v25 B0 current-low",
    "epoch1_mean": float(loss_new[0]),
    "epoch60_mean": float(loss_new[-1]),
    "last1000_mean": float(np.mean(
        [item.value for item in new_losses[-1000:]])),
}]


root_cause_ladder = [
    {
        "status": "Verified",
        "finding": "Not a data/init/first-forward mismatch",
        "evidence": "Same train/val manifests, initial B0 SHA, first 100 fingerprints and step0 loss",
        "confidence": "High",
    },
    {
        "status": "Verified",
        "finding": "The trajectories split at the first optimizer update",
        "evidence": "step1 B0 hashes differ; first visible loss difference is step3",
        "confidence": "High",
    },
    {
        "status": "Verified",
        "finding": "B1 repair is not the direct cause of the independent B0 low run",
        "evidence": "B0 has B1 disabled; reviewed commit diff does not change the active B0 forward/loss definition",
        "confidence": "High",
    },
    {
        "status": "Likely",
        "finding": "Uncontrolled CUDA backward/Adam numerical path selects different basins",
        "evidence": "Trainer deterministic mode is absent; custom PointNet2 CUDA is active; separate GPUs were used",
        "confidence": "Medium-high",
    },
    {
        "status": "Unresolved",
        "finding": "Whether divergence first appears in gradients or only in Adam update",
        "evidence": "Checkpoints contain post-step parameters/Adam hashes but no pre-step per-parameter gradient hash",
        "confidence": "Requires a new 2-step audit",
    },
]


other_issues = [
    {
        "severity": "Critical",
        "issue": "B0 baseline is not reproducible",
        "evidence": "One v25 B0 run reaches 50.690, the next reaches 29.870 from the same seed/init/data",
        "impact": "No cross-arm causal attribution or stable baseline claim",
    },
    {
        "severity": "High",
        "issue": "Current v25 B0 is SeqTrack-derived, not transaction-equivalent",
        "evidence": "Candidate weighting, RNG/sampler, wrapper and evaluator differ",
        "impact": "Numerical parity with a historical SeqTrack score is descriptive only",
    },
    {
        "severity": "High",
        "issue": "Old-high and current-low resolved configs are not identical",
        "evidence": "Validation cadence is 1 vs 5; YAML SHA matches but resolved SHA does not",
        "impact": "The historical 50-point run is not a formally matched control",
    },
    {
        "severity": "High",
        "issue": "B1-only tracking score does not measure deployed B1 benefit",
        "evidence": "Both prior and current B1-only runs output B0 observation",
        "impact": "A high/low B1-only Success mainly reports its B0 trajectory",
    },
    {
        "severity": "High",
        "issue": "Uncertainty promotion gate remains open",
        "evidence": "Current GRU raw coverage95=82.7%, ECE=10.8%; no held-out calibration/dev artifact",
        "impact": "Sigma/NLL/coverage repair cannot be called complete",
    },
    {
        "severity": "Medium",
        "issue": "Recursive-age validation diagnostics are incomplete",
        "evidence": "epoch60 CSV has zero age-valid rows; official B1 report crashes on the empty stratum",
        "impact": "Age-stratified long-tail generalization cannot be audited",
    },
    {
        "severity": "Medium",
        "issue": "Environment identity is missing from run provenance",
        "evidence": "GPU model/index, CUDA/cuDNN, torch and PointNet2 build are not recorded",
        "impact": "Cross-GPU numerical divergence cannot be reconciled after the run",
    },
    {
        "severity": "Medium",
        "issue": "Evidence is still one seed on mini",
        "evidence": "All new arms use seed42 and nuScenes mini",
        "impact": "No variance estimate or full-dataset paper claim",
    },
]


headline = [{
    "current_b0_success": summary_by_name["v25 B0 current-low"]["final_success"],
    "b0_regression_points": (
        summary_by_name["v25 B0 current-low"]["final_success"]
        - summary_by_name["v25 B0 prior-high"]["final_success"]),
    "step1_parity": "FAIL",
    "seqtrack_baseline_status": "NOT RESTORED",
    "current_b1_mean_delta_cv": b1_mechanism[-1]["learned_minus_cv"],
}]

snapshot = {
    "headline": headline,
    "tracking_curve": tracking_curve,
    "b1_tracking_curve": b1_tracking_curve,
    "tracking_summary": tracking_summary,
    "b0_loss_curve": b0_loss_curve,
    "b0_loss_summary": b0_loss_summary,
    "b0_chain": b0_chain,
    "run_identity": run_identity,
    "resolved_diffs": resolved_diffs,
    "baseline_contract": baseline_contract,
    "b1_mechanism": b1_mechanism,
    "root_cause_ladder": root_cause_ladder,
    "other_issues": other_issues,
}

report_db = REPORT_DIR / "report_data.sqlite"
report_query = (
    "SELECT snapshot_json FROM report_snapshot "
    "WHERE snapshot_id = 'b0_b1_regression_20260826'")
with sqlite3.connect(report_db) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS report_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)")
    connection.execute(
        "INSERT OR REPLACE INTO report_snapshot(snapshot_id, snapshot_json) "
        "VALUES (?, ?)",
        ("b0_b1_regression_20260826", json.dumps(
            snapshot, ensure_ascii=False, allow_nan=False)))
    selected = connection.execute(report_query).fetchone()
if selected is None or json.loads(selected[0]) != snapshot:
    raise RuntimeError("SQLite report snapshot verification failed")

source = {
    "id": "src_b0_b1_regression_20260826",
    "label": "CT-SeqTrack B0/B1 TensorBoard、checkpoint、provenance 与代码合同派生快照",
    "path": relative(report_db),
    "query": {
        "engine": "sqlite",
        "sql": report_query,
        "tablesUsed": ["report_snapshot"],
    },
}

title = "CT-SeqTrack B0/B1 再次退化诊断（2026-08-26）"
artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "定位 v25 B0/B1 从约50分回落到约30分的训练事务、SeqTrack基线等价性与剩余风险。",
        "generatedAt": "2026-08-26T00:00:00+08:00",
        "sources": [source],
        "cards": [
            {
                "id": "card_current_b0",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "当前 v25 B0 的 epoch60 mini_val Success。",
                "metrics": [{"label": "Current B0 Success", "field": "current_b0_success", "format": "number"}],
            },
            {
                "id": "card_regression",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "当前 B0 相对 2026-08-24 单条高分 B0 的 final Success 差值。",
                "metrics": [{"label": "B0 regression", "field": "b0_regression_points", "format": "number"}],
            },
            {
                "id": "card_step1",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "相同初始化和首批输入下，第一次 Adam 更新后的 B0 参数一致性。",
                "metrics": [{"label": "Step1 parity", "field": "step1_parity"}],
            },
            {
                "id": "card_baseline",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "要求稳定复现且训练/评估合同清楚后，才可称为恢复 SeqTrack B0 基线。",
                "metrics": [{"label": "SeqTrack baseline", "field": "seqtrack_baseline_status"}],
            },
            {
                "id": "card_b1_mean",
                "dataset": "headline",
                "sourceId": source["id"],
                "description": "当前 GRU epoch60 learned prior 相对自身 CV anchor 的 RMSE 差值。",
                "metrics": [{"label": "B1 learned−CV", "field": "current_b1_mean_delta_cv", "format": "number"}],
            },
        ],
        "charts": [
            {
                "id": "chart_b0_success",
                "title": "B0 与 SeqTrack mini_val Success",
                "subtitle": "每5 epoch 对齐；v25 使用 safe evaluator，SeqTrack 参考使用旧 evaluator",
                "intent": "trend",
                "question": "之前的50分是否形成了可复现的 B0 收敛轨迹？",
                "rationale": "12个对齐验证点显示回退从早期开始，而非最后一个 checkpoint 波动。",
                "type": "line",
                "dataset": "tracking_curve",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"},
                    "color": {"field": "run", "type": "nominal", "label": "Run"},
                    "tooltip": [{"field": "precision", "type": "quantitative", "label": "Precision"}],
                },
                "layout": "full",
            },
            {
                "id": "chart_b0_loss",
                "title": "v25 B0 训练目标的 epoch 均值",
                "subtitle": "loss_b0_transaction；两条轨迹损失接近，但递归 tracking 分数相差约21点",
                "intent": "trend",
                "question": "低分是否由明显的训练 loss 发散解释？",
                "rationale": "60个 epoch 均值显示 surrogate loss 的轻微差异不能直接解释 tracking 崩落。",
                "type": "line",
                "dataset": "b0_loss_curve",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "mean_b0_loss", "type": "quantitative", "label": "Mean B0 loss"},
                    "color": {"field": "run", "type": "nominal", "label": "Run"},
                },
                "layout": "full",
            },
            {
                "id": "chart_b1_tracking",
                "title": "B1-only observation Success",
                "subtitle": "旧高轨迹只运行到epoch30；两条实验的部署输出都仍是B0 observation",
                "intent": "trend",
                "question": "B1-only 的高低分是否等同于 B1 学习质量？",
                "rationale": "同 epoch 对照表明 top-line tracking 与 B1 mechanism 指标可以反向变化。",
                "type": "line",
                "dataset": "b1_tracking_curve",
                "sourceId": source["id"],
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                    "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"},
                    "color": {"field": "run", "type": "nominal", "label": "Run"},
                    "tooltip": [{"field": "precision", "type": "quantitative", "label": "Precision"}],
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "table_tracking",
                "title": "Tracking final/late-3 汇总",
                "subtitle": "每条 run 按其实际最后验证点计算；旧 B1 high 仅到 epoch30",
                "dataset": "tracking_summary",
                "sourceId": source["id"],
                "defaultSort": {"field": "final_success", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "run", "label": "Run", "type": "text"},
                    {"field": "final_epoch", "label": "Final epoch", "format": "number"},
                    {"field": "final_success", "label": "Final S", "format": "number"},
                    {"field": "late3_success", "label": "Late-3 S", "format": "number"},
                    {"field": "final_precision", "label": "Final P", "format": "number"},
                    {"field": "late3_precision", "label": "Late-3 P", "format": "number"},
                ],
            },
            {
                "id": "table_chain",
                "title": "B0 分叉链路",
                "subtitle": "从初始化、输入、forward 到第一次 optimizer update 的证据顺序",
                "dataset": "b0_chain",
                "sourceId": source["id"],
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "stage", "label": "Stage", "type": "text"},
                    {"field": "prior_high", "label": "Prior high", "type": "text"},
                    {"field": "current_low", "label": "Current low", "type": "text"},
                    {"field": "equal", "label": "Equal", "type": "text"},
                    {"field": "meaning", "label": "Interpretation", "type": "text"},
                ],
            },
            {
                "id": "table_baseline",
                "title": "当前 v25 B0 与正常 SeqTrack 的合同差异",
                "subtitle": "架构继承不等于训练事务和 evaluator 等价",
                "dataset": "baseline_contract",
                "sourceId": source["id"],
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "aspect", "label": "Aspect", "type": "text"},
                    {"field": "SeqTrack_reference", "label": "SeqTrack", "type": "text"},
                    {"field": "current_v25_B0", "label": "Current v25 B0", "type": "text"},
                    {"field": "equivalent", "label": "Equivalent", "type": "text"},
                    {"field": "impact", "label": "Impact", "type": "text"},
                ],
            },
            {
                "id": "table_b1",
                "title": "B1 top-line 与 mechanism 指标",
                "subtitle": "旧高分 B1 的 mean/coverage 实际更差；当前修复改善了 mechanism，而不是 observation 轨迹",
                "dataset": "b1_mechanism",
                "sourceId": source["id"],
                "defaultSort": {"field": "epoch", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "run", "label": "Run", "type": "text"},
                    {"field": "epoch", "label": "Epoch", "format": "number"},
                    {"field": "tracking_success", "label": "Tracking S", "format": "number"},
                    {"field": "learned_rmse", "label": "Learned RMSE", "format": "number"},
                    {"field": "cv_rmse", "label": "CV RMSE", "format": "number"},
                    {"field": "learned_minus_cv", "label": "Learned−CV", "format": "number"},
                    {"field": "nll", "label": "NLL", "format": "number"},
                    {"field": "coverage95", "label": "Coverage95", "format": "percent"},
                ],
            },
            {
                "id": "table_causes",
                "title": "根因证据分级",
                "subtitle": "Verified 与 Likely/Unresolved 明确分开",
                "dataset": "root_cause_ladder",
                "sourceId": source["id"],
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "status", "label": "Status", "type": "text"},
                    {"field": "finding", "label": "Finding", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                    {"field": "confidence", "label": "Confidence", "type": "text"},
                ],
            },
            {
                "id": "table_issues",
                "title": "仍存在的问题",
                "subtitle": "按对论文因果解释和服务器实验决策的影响排序",
                "dataset": "other_issues",
                "sourceId": source["id"],
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "severity", "label": "Severity", "type": "text"},
                    {"field": "issue", "label": "Issue", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                    {"field": "impact", "label": "Impact", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 技术结论\n\n"
                    "**当前没有恢复到稳定的 SeqTrack B0 基线。** 2026-08-24 的 v25 B0 达到 50.690 Success，但 2026-08-25 的同 seed scratch B0 只有 29.870，下降20.821点。两者数据清单、B0 初始化、前100批 observation 指纹和第一个 forward/loss 相同；第一次 Adam 更新后的 B0 参数已经不同，第3个训练 step 才出现可见 loss 差。回退因此起源于 backward/optimizer 数值路径，被递归验证放大，而不是 B1 新 loss 直接污染了独立 B0。\n\n"
                    "旧50分证明当前 SeqTrack-derived B0 有能力进入高分盆地，但单条 run 不能证明基线已恢复或固定。当前 B1-GRU 的 tracking 也主要跟随低分 B0 observation 轨迹；它的 mean prior 相比旧高分 B1 反而学得更好。因此应把问题称为 **B0 事务不可复现 + B1-only 指标口径混淆**，而不是“B1 修复失败导致整体掉分”。"),
            },
            {"id": "metrics", "type": "metric-strip", "layout": "full", "cardIds": ["card_current_b0", "card_regression", "card_step1", "card_baseline", "card_b1_mean"]},
            {
                "id": "key_finding_b0",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 1. 50分是一次高分轨迹，不是已锁定的 B0\n\n"
                    "旧 v25 B0 在 epoch45 后稳定在约50 Success；当前 B0 从 epoch5 起就只有20.791，之后长期停在约30。两条 run 的 epoch60 B0 loss 均值仅为0.2425与0.2494，最后1000步均值也仅差约0.0066，但 final Success 相差20.821点。这说明当前 observation surrogate loss 对递归 tracking 盆地非常敏感，单看 loss 收敛无法发现回归。"),
            },
            {"id": "b0_chart", "type": "chart", "layout": "full", "chartId": "chart_b0_success"},
            {"id": "loss_chart", "type": "chart", "layout": "full", "chartId": "chart_b0_loss"},
            {"id": "tracking_table", "type": "table", "layout": "full", "tableId": "table_tracking"},
            {
                "id": "key_finding_chain",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 2. 分叉发生在第一次 backward/Adam 更新\n\n"
                    "initial SHA、首100批输入和 step0 loss 都一致，排除了数据划分、初始化和首个 forward 口径。step1 参数 SHA 已经不同，但日志 loss 到 step3 才出现约0.0013的差值；到 epoch5，Success 已相差13.880点。当前 checkpoint 只保存 post-step 参数与 Adam 状态，没有保存 pre-step gradient SHA，因此尚不能判断是 CUDA backward 先不同，还是梯度相同而 Adam 更新不同。"),
            },
            {"id": "chain_table", "type": "table", "layout": "full", "tableId": "table_chain"},
            {"id": "causes_table", "type": "table", "layout": "full", "tableId": "table_causes"},
            {
                "id": "seqtrack_equivalence",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 3. 架构继承了 SeqTrack，但实验基线并不等价\n\n"
                    "当前 `CTSEQTRACK` 确实继承 `SEQTRACK3D`，B0 arm 也关闭 B1/B2/B3，因此 observation 网络核心仍来自 SeqTrack。但 v25 改成候选均衡的 stateless sampler、`0.5 + 3×1/6` 加权目标、safe dual-stream envelope 和 first-frame-size recursive evaluator；正常 SeqTrack 使用普通 batch mean/global RNG，且旧 evaluator 在 safe flag 缺失时读取当前帧 GT 尺寸。两者训练目标和评估口径都不同。\n\n"
                    "所以旧 v25 B0 的50.690与 SeqTrack 的约51分只能叫数值接近，不能叫 transaction-equivalent restoration。当前29.870进一步说明数值基线也没有稳定恢复。"),
            },
            {"id": "baseline_table", "type": "table", "layout": "full", "tableId": "table_baseline"},
            {
                "id": "b1_interpretation",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 4. B1 top-line 回退不等于 B1 又学坏了\n\n"
                    "旧 B1-only 高轨迹在 epoch30 有49.275 Success，但 learned RMSE=5.700 m，比 CV=5.592 m 更差，coverage95仅31.1%。当前 GRU 在同为 epoch30 时只有30.415 Success，却已经做到 learned 10.244 m < CV 10.463 m，coverage95=78.0%；到 epoch60，learned−CV=-0.208 m 且 coverage95=82.7%。旧高分来自其 B0 observation 轨迹，不是 B1 mechanism 成功；当前则相反，B1 mechanism 改善但上游 B0 轨迹低。"),
            },
            {"id": "b1_chart", "type": "chart", "layout": "full", "chartId": "chart_b1_tracking"},
            {"id": "b1_table", "type": "table", "layout": "full", "tableId": "table_b1"},
            {
                "id": "scope_definitions",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 5. 范围、数据与指标定义\n\n"
                    "比较使用 nuScenes mini Car、seed42、batch16、60-epoch scratch runs；旧 B1 high 只完成30轮。Tracking 使用各 run 的 TensorBoard Success/Precision，B0 曲线按每5 epoch 对齐。B0 loss 为每1262个 observation optimizer step 的 epoch 均值。输入一致性来自 checkpoint 的前100批 observation fingerprint；参数一致性来自 B0 SHA256。B1 mechanism 只在同 checkpoint 的 `b1_valid` endpoint 上计算 RMSE/NLL/coverage。"),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 6. 诊断方法\n\n"
                    "诊断顺序是：先复算 final/late-3 与完整曲线，再核对 provenance 的数据清单、commit、resolved config 和 evaluator；随后沿 initialization→fingerprint→step0 loss→step1 hash→step3 loss→epoch5 validation 建立分叉链。Git diff 用于确认 B1 repair commit 没有修改启用状态下的 B0 forward/损失定义。当前代码门禁为148 passed、1 skipped，compileall 通过；slimming verify 仅因 HEAD 不等于固定 baseline commit `001951a` 按设计失败。"),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 7. 限制、不确定性与稳健性\n\n"
                    "最主要的不确定性是缺少 step1 更新前的逐参数 gradient SHA，无法在 backward 与 Adam 之间二选一。旧高分与当前低分使用不同 commit，且 validation cadence 为1与5；虽然两项都不能解释发生在首次 validation 前的 step1 分叉，但它们阻止了严格单变量复现。运行 provenance 也没有记录 GPU型号、CUDA/cuDNN、PyTorch和PointNet2 build。三个新 arm 并行放在不同GPU上，不能满足同物理GPU、顺序执行的位级公平性验收。"),
            },
            {"id": "issues_table", "type": "table", "layout": "full", "tableId": "table_issues"},
            {
                "id": "next_steps",
                "type": "markdown",
                "layout": "full",
                "sourceId": source["id"],
                "body": (
                    "## 8. 建议的下一步\n\n"
                    "1. 暂停 full nuScenes 和 B1+B2 涨分实验。先在同一张空闲GPU上顺序跑 B0→B1-GRU→B1-CfC 的2-step与100-step audit，三条都从 epoch0 开始。\n"
                    "2. 在 optimizer step 前保存每个 B0 参数的 gradient SHA/范数，step 后保存 Adam `step/exp_avg/exp_avg_sq` 与参数 SHA。若 grad 已不同，排查 PointNet2/CUDA backward；若 grad 相同而更新不同，固定 Adam `foreach=False` 做诊断对照。\n"
                    "3. 固定同一 resolved config，包括 validation cadence；同时记录 GPU UUID/model、driver、CUDA/cuDNN、torch、Lightning和PointNet2 build。\n"
                    "4. 要么把论文 B0 明确定义为 Safe-SeqTrack-derived control，并用相同 evaluator 重新建立多seed基线；要么增加一个完全忠实于原始 SeqTrack 训练事务的 reference arm。不要再用单条50分 run 同时承担这两个含义。\n"
                    "5. B0 parity 通过后再重跑 mini；B1-GRU 保留为主方案。修复 recursive-age validation 导出并完成 held-out uncertainty calibration 后，才进入 full-minus-B3。"),
            },
            {
                "id": "questions",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## 9. 仍需回答的问题\n\n"
                    "- step1 的 B0 gradient 是否已经不同？\n"
                    "- 同一物理GPU顺序运行时，B0 能否重复得到相同 step1/step100 SHA？\n"
                    "- 论文中的 B0 要定义为原始 SeqTrack，还是安全评估下的 SeqTrack-derived control？\n"
                    "- 多seed下高/低盆地的概率和方差有多大？"),
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": "2026-08-26T00:00:00+08:00",
        "status": "ready",
        "datasets": snapshot,
    },
    "sources": [source],
}

source_paths = []
for run in RUNS.values():
    source_paths.extend([
        relative(run / "run_provenance.json"),
        relative(run / "lightning_logs/version_0/hparams.yaml"),
    ])
for name, run in RUNS.items():
    if name != "SeqTrack reference":
        source_paths.append(relative(
            run / "lightning_logs/version_0/checkpoints/last.ckpt"))
for name, epoch in (("v25 B1 prior-high@30", 30),
                    ("v25 B1-GRU current", 30),
                    ("v25 B1-GRU current", 60)):
    source_paths.append(relative(
        RUNS[name] / "lightning_logs/version_0/candidate_diagnostics"
        / f"epoch_{epoch:02d}.csv"))

analysis_summary = {
    "verdict": {
        "stable_seqtrack_b0_restored": False,
        "b0_core_inherits_seqtrack": True,
        "b0_training_and_evaluator_equivalent_to_seqtrack": False,
        "regression_origin": "first backward/Adam update",
        "b1_mechanism_regressed": False,
        "ready_for_full_nuscenes": False,
    },
    "source_paths": source_paths,
    "headline": headline[0],
    "tracking_summary": tracking_summary,
    "b0_chain": b0_chain,
    "b0_loss_summary": b0_loss_summary,
    "resolved_config_differences": resolved_diffs,
    "baseline_contract": baseline_contract,
    "b1_mechanism": b1_mechanism,
    "root_cause_ladder": root_cause_ladder,
    "other_issues": other_issues,
    "verification": {
        "pytest": "148 passed, 1 skipped",
        "compileall": "passed",
        "slimming_verify": "expected failure: HEAD e9a2d6d != pinned 001951a",
    },
}

(REPORT_DIR / "analysis_summary.json").write_text(
    json.dumps(analysis_summary, indent=2, ensure_ascii=False, allow_nan=False)
    + "\n", encoding="utf-8")
(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8")

print(json.dumps({
    "report_dir": relative(REPORT_DIR),
    "artifact": relative(REPORT_DIR / "artifact.json"),
    "summary": relative(REPORT_DIR / "analysis_summary.json"),
    "source_db": relative(report_db),
}, ensure_ascii=False))
