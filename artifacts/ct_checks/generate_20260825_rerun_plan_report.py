"""Append the 2026-08-25 rerun and short-term-plan audit to the prior report.

All experiment ``output/`` trees and frozen baseline repositories are read-only.
Only derived material under ``artifacts/ct_checks/reports`` is produced.  The
analysis intentionally excludes trajtrack and the abandoned B0 2x2 protocol.
"""

from __future__ import annotations

import copy
import csv
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
RESEARCH_ROOT = ROOT.parent
SEQ_ROOT = RESEARCH_ROOT / "seqtrack"
BASE_REPORT = ROOT / "artifacts/ct_checks/reports/20260825_seqtrack_comparison"
REPORT_DIR = ROOT / "artifacts/ct_checks/reports/20260825_rerun_and_short_term_plan"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

B1_NEW = (
    ROOT
    / "output/20260825-0057-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_rerun_20260825"
)
B1_OLD = (
    ROOT
    / "output/20260824-0220-25_b1-ct25_b1_only_mini_car_60ep_bs16_seed42_retryfix"
)
B0_FORMAL = (
    ROOT
    / "output/20260824-0219-25_b0-ct25_b0_mini_car_60ep_bs16_seed42_retryfix"
)
SEQ_NEW = (
    SEQ_ROOT
    / "output/20260825-0057-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16_seed42_rerun_20260825"
)
SEQ_LOW = (
    SEQ_ROOT
    / "output/20260801-2155-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16_seed42"
)
SEQ_HIGH = (
    SEQ_ROOT
    / "output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16"
)


def accumulator(run: Path) -> EventAccumulator:
    value = EventAccumulator(str(run / "lightning_logs/version_0"), size_guidance={"scalars": 0})
    value.Reload()
    return value


def scalar_rows(run: Path, tag: str):
    return accumulator(run).Scalars(tag)


def metrics_at(run: Path, success_tag: str, precision_tag: str, count: int):
    success = scalar_rows(run, success_tag)
    precision = scalar_rows(run, precision_tag)
    if len(success) < count or len(success) != len(precision):
        raise RuntimeError(f"invalid metric series for {run}")
    index = count - 1
    return {
        "epoch": count,
        "success": float(success[index].value),
        "precision": float(precision[index].value),
        "late3_success": float(np.mean([row.value for row in success[index - 2:index + 1]])),
        "late3_precision": float(np.mean([row.value for row in precision[index - 2:index + 1]])),
        "validation_points": len(success),
    }


def loss_series(run: Path, leaf: str = "loss_loss_total"):
    value = EventAccumulator(
        str(run / "lightning_logs/version_0" / leaf),
        size_guidance={"scalars": 0},
    )
    value.Reload()
    return value.Scalars("loss")


def first_loss_difference(left: Path, right: Path) -> int:
    left_rows = loss_series(left)
    right_rows = loss_series(right)
    return next(
        index
        for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows))
        if left_row.step != right_row.step or left_row.value != right_row.value
    )


def load_checkpoint(path: Path):
    # Lightning serialized EasyDict in hparams; a read-only stub is sufficient
    # for extracting the plain audit dictionaries without adding a dependency.
    if "easydict" not in sys.modules:
        module = types.ModuleType("easydict")
        easy_dict = type("EasyDict", (dict,), {})
        easy_dict.__module__ = "easydict"
        module.EasyDict = easy_dict
        sys.modules["easydict"] = module
    return torch.load(path, map_location="cpu", weights_only=False)


def b1_diagnostic(run: Path, epoch: int):
    frame = pd.read_csv(
        run
        / "lightning_logs/version_0/candidate_diagnostics"
        / f"epoch_{epoch:02d}.csv"
    )
    valid = (
        (frame["b1_valid"] > 0)
        & np.isfinite(frame["learned_motion_error"])
        & np.isfinite(frame["kinematic_error"])
    )
    learned_rmse = float(np.sqrt(np.mean(frame.loc[valid, "learned_motion_error"] ** 2)))
    cv_rmse = float(np.sqrt(np.mean(frame.loc[valid, "kinematic_error"] ** 2)))
    return {
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
    }


new_b1_30 = metrics_at(B1_NEW, "success/mini_val", "precision/mini_val", 30)
old_b1_30 = metrics_at(B1_OLD, "success/mini_val", "precision/mini_val", 30)
b0_30 = metrics_at(B0_FORMAL, "success/mini_val", "precision/mini_val", 30)
b0_60 = metrics_at(B0_FORMAL, "success/mini_val", "precision/mini_val", 60)
seq_new_60 = metrics_at(SEQ_NEW, "success/test", "precision/test", 12)
seq_low_60 = metrics_at(SEQ_LOW, "success/test", "precision/test", 12)
seq_high_60 = metrics_at(SEQ_HIGH, "success/test", "precision/test", 12)

# SeqTrack logs every five epochs.  Correct the display epoch from validation
# point index while retaining the final/late-3 values computed above.
for metric in (seq_new_60, seq_low_60, seq_high_60):
    metric["epoch"] = 60

new_provenance = json.loads((B1_NEW / "run_provenance.json").read_text(encoding="utf-8"))
old_provenance = json.loads((B1_OLD / "run_provenance.json").read_text(encoding="utf-8"))
new_config = copy.deepcopy(new_provenance["resolved_config"])
old_config = copy.deepcopy(old_provenance["resolved_config"])
new_config.pop("tag", None)
old_config.pop("tag", None)
if new_config != old_config:
    raise RuntimeError("B1 rerun config differs beyond tag")
if new_provenance["git"]["commit"] != old_provenance["git"]["commit"]:
    raise RuntimeError("B1 rerun used a different tracked commit")
if new_provenance["config_sha256"] != old_provenance["config_sha256"]:
    raise RuntimeError("B1 rerun used a different config file")
if new_provenance["datasets"] != old_provenance["datasets"]:
    raise RuntimeError("B1 rerun used a different dataset manifest")

new_checkpoint = load_checkpoint(
    B1_NEW / "lightning_logs/version_0/checkpoints/last.ckpt"
)
old_checkpoint = load_checkpoint(
    B1_OLD / "lightning_logs/version_0/checkpoints/last.ckpt"
)
new_hashes = new_checkpoint["ct_b0_prefix_hashes"]
old_hashes = old_checkpoint["ct_b0_prefix_hashes"]
fingerprints_equal = (
    new_checkpoint["ct_observation_batch_fingerprints"]
    == old_checkpoint["ct_observation_batch_fingerprints"]
)
if not fingerprints_equal:
    raise RuntimeError("B1 rerun observation fingerprints unexpectedly differ")
if new_hashes["initial"] != old_hashes["initial"]:
    raise RuntimeError("B1 rerun initial B0 parameters unexpectedly differ")

new_b1_diag = b1_diagnostic(B1_NEW, 30)
old_b1_diag = b1_diagnostic(B1_OLD, 30)


def normalize_seq_hparams(text: str):
    return [
        (f"{' ' * (len(line) - len(line.lstrip()))}tag: <TAG>"
         if line.strip().startswith("tag:") else line)
        for line in text.splitlines()
    ]


new_seq_hparams = (
    SEQ_NEW / "lightning_logs/version_0/hparams.yaml"
).read_text(encoding="utf-8")
low_seq_hparams = (
    SEQ_LOW / "lightning_logs/version_0/hparams.yaml"
).read_text(encoding="utf-8")
if normalize_seq_hparams(new_seq_hparams) != normalize_seq_hparams(low_seq_hparams):
    raise RuntimeError("SeqTrack workers=4 rerun hparams differ beyond tag")

b1_first_loss_diff = first_loss_difference(B1_NEW, B1_OLD)
seq_first_loss_diff = first_loss_difference(SEQ_NEW, SEQ_LOW)

rerun_rows = [
    {
        "run": "v25 B1-only retryfix（昨天）",
        "comparison_epoch": 30,
        "success": old_b1_30["success"],
        "precision": old_b1_30["precision"],
        "late3_success": old_b1_30["late3_success"],
        "late3_precision": old_b1_30["late3_precision"],
        "evaluator": "safe recursive",
        "interpretation": "部署输出为observation；低B0轨迹",
    },
    {
        "run": "v25 B1-only rerun（今天）",
        "comparison_epoch": 30,
        "success": new_b1_30["success"],
        "precision": new_b1_30["precision"],
        "late3_success": new_b1_30["late3_success"],
        "late3_precision": new_b1_30["late3_precision"],
        "evaluator": "safe recursive",
        "interpretation": "同配置同臂；高B0轨迹；仅完成30轮",
    },
    {
        "run": "v25 B0 formal",
        "comparison_epoch": 30,
        "success": b0_30["success"],
        "precision": b0_30["precision"],
        "late3_success": b0_30["late3_success"],
        "late3_precision": b0_30["late3_precision"],
        "evaluator": "safe recursive",
        "interpretation": "正式安全B0在同轮次的描述性参照",
    },
    {
        "run": "v25 B0 formal final",
        "comparison_epoch": 60,
        "success": b0_60["success"],
        "precision": b0_60["precision"],
        "late3_success": b0_60["late3_success"],
        "late3_precision": b0_60["late3_precision"],
        "evaluator": "safe recursive",
        "interpretation": "当前安全B0数值候选基线",
    },
    {
        "run": "SeqTrack workers4 rerun",
        "comparison_epoch": 60,
        "success": seq_new_60["success"],
        "precision": seq_new_60["precision"],
        "late3_success": seq_new_60["late3_success"],
        "late3_precision": seq_new_60["late3_precision"],
        "evaluator": "原版（当前帧GT尺寸）",
        "interpretation": "与旧workers4日志仅tag不同，落入更低轨迹",
    },
    {
        "run": "SeqTrack workers4 old-low",
        "comparison_epoch": 60,
        "success": seq_low_60["success"],
        "precision": seq_low_60["precision"],
        "late3_success": seq_low_60["late3_success"],
        "late3_precision": seq_low_60["late3_precision"],
        "evaluator": "原版（当前帧GT尺寸）",
        "interpretation": "旧低分轨迹",
    },
    {
        "run": "SeqTrack workers12 old-high",
        "comparison_epoch": 60,
        "success": seq_high_60["success"],
        "precision": seq_high_60["precision"],
        "late3_success": seq_high_60["late3_success"],
        "late3_precision": seq_high_60["late3_precision"],
        "evaluator": "原版（当前帧GT尺寸）",
        "interpretation": "旧高分轨迹；旧run无commit身份",
    },
]

transaction_rows = [
    {
        "check": "tracked git commit",
        "result": new_provenance["git"]["commit"],
        "equal": "是",
        "meaning": "不是代码版本变化",
    },
    {
        "check": "formal config file SHA256",
        "result": new_provenance["config_sha256"],
        "equal": "是",
        "meaning": "resolved配置去掉tag后逐项相同",
    },
    {
        "check": "dataset manifests",
        "result": "train/mechanism/val逐项相同",
        "equal": "是",
        "meaning": "不是数据划分或manifest变化",
    },
    {
        "check": "initial B0 parameter SHA256",
        "result": new_hashes["initial"],
        "equal": "是",
        "meaning": "随机初始化完全相同",
    },
    {
        "check": "first 100 observation fingerprints",
        "result": "100/100相同",
        "equal": "是",
        "meaning": "输入、候选与点采样事务已对齐",
    },
    {
        "check": "B0 parameter SHA256 after step1",
        "result": f"new={new_hashes['step_1'][:12]}… / old={old_hashes['step_1'][:12]}…",
        "equal": "否",
        "meaning": "第一次backward/Adam更新已经分叉",
    },
    {
        "check": "first visible total-loss difference",
        "result": f"logged step {b1_first_loss_diff}",
        "equal": "否",
        "meaning": "微小参数差在数步后转成可见loss差",
    },
    {
        "check": "epoch30 Success",
        "result": f"{new_b1_30['success']:.3f} vs {old_b1_30['success']:.3f}",
        "equal": "否",
        "meaning": "相差约14.589点，形成高低训练盆地",
    },
]

b1_rows = [
    {"run": "B1-only retryfix epoch30", **old_b1_diag},
    {"run": "B1-only rerun epoch30", **new_b1_diag},
]

plan_rows = [
    {
        "order": 0,
        "plan_item": "恢复并固定B0",
        "status": "数值恢复；尚未固定",
        "evidence": "safe B0 final S=50.690，接近SeqTrack旧高轨迹50.986；但同臂step1不可复现",
        "next_gate": "同GPU空闲、顺序重复的step1/step100 B0参数hash必须一致",
    },
    {
        "order": 1,
        "plan_item": "修复B1 sigma/NLL/coverage；保持GRU、固定geometry",
        "status": "未通过",
        "evidence": f"新run learned−CV={new_b1_diag['learned_minus_cv']:+.3f}m，NLL={new_b1_diag['mean_nll']:.1f}，95% coverage={new_b1_diag['coverage_95']:.1%}",
        "next_gate": "B0锁定后再做box-only mean/NLL与held-out校准；95% coverage≥90%、ECE≤0.05",
    },
    {
        "order": 2,
        "plan_item": "修复B2 acquisition funnel",
        "status": "未通过",
        "evidence": "旧正式run available约6–8%，target-bearing仅1–2行，harmful约93–97%，oracle headroom近零",
        "next_gate": "先修extension supply/target retention；presence AUROC≥0.65且foreground-valid≥15%",
    },
    {
        "order": 3,
        "plan_item": "3A bounded adaptive shell；3B GRU vs CfC matched",
        "status": "等待",
        "evidence": "B0未固定、B1未校准，当前比较会把训练轨迹误当架构差",
        "next_gate": "仅在B1 gate通过后从epoch0做matched短跑",
    },
    {
        "order": 4,
        "plan_item": "relation-aware sampling / robust voting / optional decoder",
        "status": "等待",
        "evidence": "当前瓶颈是没有有效target-bearing evidence，不是decoder容量不足",
        "next_gate": "按4A→4B顺序；4C只在前两者产生oracle headroom后启用",
    },
    {
        "order": 5,
        "plan_item": "最后恢复B3 calibration",
        "status": "禁止提前",
        "evidence": "现有Full无artifact、applied rate=0，最终分数来自其B0轨迹",
        "next_gate": "B2有正headroom后，在独立calibration tracklets生成绑定artifact",
    },
]

headline = [{
    "b1_ep30_success_delta": new_b1_30["success"] - old_b1_30["success"],
    "b1_ep30_late3_delta": new_b1_30["late3_success"] - old_b1_30["late3_success"],
    "seq_rerun_success": seq_new_60["success"],
    "b0_vs_seq_high": b0_60["success"] - seq_high_60["success"],
    "b1_new_learned_minus_cv": new_b1_diag["learned_minus_cv"],
    "b1_new_coverage95": new_b1_diag["coverage_95"],
}]

b1_curve_rows = []
for label, run in (("B1 retryfix", B1_OLD), ("B1 rerun", B1_NEW)):
    success = scalar_rows(run, "success/mini_val")[:30]
    precision = scalar_rows(run, "precision/mini_val")[:30]
    for epoch, (success_row, precision_row) in enumerate(zip(success, precision), start=1):
        b1_curve_rows.append({
            "run": label,
            "epoch": epoch,
            "success": float(success_row.value),
            "precision": float(precision_row.value),
        })

seq_curve_rows = []
for label, run in (
    ("Seq old-high workers12", SEQ_HIGH),
    ("Seq old-low workers4", SEQ_LOW),
    ("Seq rerun workers4", SEQ_NEW),
):
    success = scalar_rows(run, "success/test")
    precision = scalar_rows(run, "precision/test")
    for index, (success_row, precision_row) in enumerate(zip(success, precision), start=1):
        seq_curve_rows.append({
            "run": label,
            "epoch": index * 5,
            "success": float(success_row.value),
            "precision": float(precision_row.value),
        })

snapshot = {
    "rerun_headline": headline,
    "rerun_comparison": rerun_rows,
    "b1_transaction_audit": transaction_rows,
    "b1_epoch30_diagnostics": b1_rows,
    "short_term_plan_status": plan_rows,
    "b1_rerun_curves": b1_curve_rows,
    "seqtrack_three_curves": seq_curve_rows,
}

database_path = REPORT_DIR / "rerun_plan_data.sqlite"
query = (
    "SELECT snapshot_json FROM rerun_plan_snapshot "
    "WHERE snapshot_id = 'rerun_plan_20260825'"
)
with sqlite3.connect(database_path) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS rerun_plan_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO rerun_plan_snapshot(snapshot_id, snapshot_json) "
        "VALUES (?, ?)",
        ("rerun_plan_20260825", json.dumps(snapshot, ensure_ascii=False)),
    )
    selected = connection.execute(query).fetchone()
if selected is None or json.loads(selected[0]) != snapshot:
    raise RuntimeError("rerun-plan SQLite snapshot verification failed")

source = {
    "id": "src_rerun_plan_20260825",
    "label": "B1/SeqTrack复跑TensorBoard、v25 checkpoint审计与短期计划派生快照",
    "path": "artifacts/ct_checks/reports/20260825_rerun_and_short_term_plan/rerun_plan_data.sqlite",
    "query": {
        "engine": "sqlite",
        "sql": query,
        "tables_used": ["rerun_plan_snapshot"],
        "description": "读取本报告对复跑、事务审计和计划门槛的有界快照",
    },
}

artifact = copy.deepcopy(json.loads(
    (BASE_REPORT / "artifact.json").read_text(encoding="utf-8")
))
manifest = artifact["manifest"]
manifest["title"] = "CT-SeqTrack B1/SeqTrack复跑与短期计划决策（2026-08-25）"
manifest["description"] = "在既有四臂与历史基线报告上追加同臂复跑、随机路径归因及短期计划门槛。"
manifest["generatedAt"] = "2026-08-25T18:00:00+08:00"
manifest["sources"].append(source)
artifact["sources"].append(source)
artifact["snapshot"]["generatedAt"] = "2026-08-25T18:00:00+08:00"
artifact["snapshot"]["datasets"].update(snapshot)
manifest["blocks"][0]["body"] = f"# {manifest['title']}"

executive = next(block for block in manifest["blocks"] if block["id"] == "executive_summary")
executive["sourceId"] = source["id"]
executive["body"] = (
    "## 技术摘要\n\n"
    "**这次复跑把‘随机训练路径’从跨臂推测提升为同臂直接证据。** "
    "两次B1-only使用同一tracked commit、同一配置文件、同一数据manifest、相同初始化和100/100相同的observation指纹；"
    "但B0参数在step1就不同。到epoch30，今天/昨天Success为49.275/34.686，差14.589点。"
    "由于proposal仍为observation，这不是B1部署增益，而是B1臂内的B0轨迹分叉。\n\n"
    "**B0可以表述为‘数值上恢复到SeqTrack高分量级’，不能表述为‘已经稳定复现SeqTrack’。** "
    "v25 safe B0 final Success=50.690，与旧SeqTrack高轨迹50.986只差-0.295；但同workers4的SeqTrack复跑从31.684进一步变成27.997，"
    "且原版验证读取当前帧GT尺寸。因此50分量级已恢复，事务稳定性和严格安全等价仍未完成。\n\n"
    "**短期计划必须停在第0步。** 新B1在epoch30的learned RMSE仍比CV差0.107m，NLL=148.6，95% coverage=31.1%；"
    "B2仍缺target-bearing evidence，B3仍没有可执行校准动作。下一步应先做空闲单GPU顺序step1梯度/Adam审计，锁定B0后再修B1校准。"
)

manifest["cards"].extend([
    {
        "id": "card_b1_rerun_delta",
        "dataset": "rerun_headline",
        "sourceId": source["id"],
        "description": "同一B1-only配置在epoch30的Success差；部署均为observation。",
        "metrics": [
            {"label": "B1 rerun ΔS@30", "field": "b1_ep30_success_delta", "format": "number", "signed": True},
        ],
    },
    {
        "id": "card_seq_rerun",
        "dataset": "rerun_headline",
        "sourceId": source["id"],
        "description": "SeqTrack同workers4、同seed、同日志配置（tag除外）的新复跑。",
        "metrics": [
            {"label": "Seq rerun Final S", "field": "seq_rerun_success", "format": "number"},
        ],
    },
    {
        "id": "card_b0_near_seq",
        "dataset": "rerun_headline",
        "sourceId": source["id"],
        "description": "仅为数值量级比较；SeqTrack旧评估器不是safe evaluator。",
        "metrics": [
            {"label": "safe B0 − Seq-high", "field": "b0_vs_seq_high", "format": "number", "signed": True},
        ],
    },
    {
        "id": "card_b1_calibration",
        "dataset": "rerun_headline",
        "sourceId": source["id"],
        "description": "新B1 epoch30；正数表示learned motion RMSE差于CV。",
        "metrics": [
            {"label": "B1 learned−CV", "field": "b1_new_learned_minus_cv", "format": "number", "signed": True},
            {"label": "B1 95% coverage", "field": "b1_new_coverage95", "format": "percent"},
        ],
    },
])

manifest["charts"].extend([
    {
        "id": "chart_b1_same_arm_rerun",
        "title": "B1-only同臂复跑：前30轮Success轨迹",
        "subtitle": "同commit、同配置、同数据、同初始化与相同observation指纹；tracking输出均为observation",
        "intent": "trend",
        "question": "B1-only低分是否是可重复的模块效应？",
        "rationale": "逐epoch曲线直接显示同一arm如何进入约34分与约49分的不同B0轨迹。",
        "type": "line",
        "dataset": "b1_rerun_curves",
        "sourceId": source["id"],
        "encodings": {
            "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
            "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"},
            "color": {"field": "run", "type": "nominal", "label": "Run"},
            "tooltip": [
                {"field": "precision", "type": "quantitative", "label": "Precision"},
            ],
        },
        "layout": "full",
    },
    {
        "id": "chart_seqtrack_three_runs",
        "title": "SeqTrack三条60轮mini轨迹",
        "subtitle": "新旧workers4配置仅tag不同；workers12旧高轨迹仅作描述性参照",
        "intent": "trend",
        "question": "SeqTrack低分能否由同一命令稳定复现？",
        "rationale": "三条每5轮验证曲线显示workers4两次复跑也不完全一致，但都落入低分盆地。",
        "type": "line",
        "dataset": "seqtrack_three_curves",
        "sourceId": source["id"],
        "encodings": {
            "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
            "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"},
            "color": {"field": "run", "type": "nominal", "label": "Run"},
            "tooltip": [
                {"field": "precision", "type": "quantitative", "label": "Precision"},
            ],
        },
        "layout": "full",
    },
])

manifest["tables"].extend([
    {
        "id": "table_rerun_comparison",
        "title": "复跑与既有实验的同口径读数",
        "subtitle": "B1按epoch30比较；SeqTrack/B0 final另列；不同评估器不可做严格排名",
        "dataset": "rerun_comparison",
        "sourceId": source["id"],
        "defaultSort": {"field": "success", "direction": "desc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "run", "label": "Run", "type": "text"},
            {"field": "comparison_epoch", "label": "Epoch", "format": "number"},
            {"field": "success", "label": "Success", "format": "number"},
            {"field": "precision", "label": "Precision", "format": "number"},
            {"field": "late3_success", "label": "Late-3 S", "format": "number"},
            {"field": "evaluator", "label": "Evaluator", "type": "text"},
            {"field": "interpretation", "label": "Interpretation", "type": "text"},
        ],
    },
    {
        "id": "table_b1_transaction_rerun",
        "title": "B1同臂复跑的B0事务审计",
        "subtitle": "从代码/数据/输入逐层排除，到第一次参数更新定位分叉",
        "dataset": "b1_transaction_audit",
        "sourceId": source["id"],
        "defaultSort": {"field": "check", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "check", "label": "Check", "type": "text"},
            {"field": "result", "label": "Observed", "type": "text"},
            {"field": "equal", "label": "Equal", "type": "text"},
            {"field": "meaning", "label": "Meaning", "type": "text"},
        ],
    },
    {
        "id": "table_b1_epoch30_diagnostics",
        "title": "两次B1-only在epoch30的机制指标",
        "subtitle": "tracking高分不代表B1 mean/sigma通过；误差受各自递归B0轨迹影响",
        "dataset": "b1_epoch30_diagnostics",
        "sourceId": source["id"],
        "defaultSort": {"field": "run", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "run", "label": "Run", "type": "text"},
            {"field": "learned_rmse", "label": "Learned RMSE", "format": "number"},
            {"field": "cv_rmse", "label": "CV RMSE", "format": "number"},
            {"field": "learned_minus_cv", "label": "Learned−CV", "format": "number", "movement": True},
            {"field": "mean_nll", "label": "Mean NLL", "format": "number"},
            {"field": "coverage_95", "label": "95% coverage", "format": "percent"},
            {"field": "valid_rate", "label": "Valid rate", "format": "percent"},
        ],
    },
    {
        "id": "table_short_term_plan",
        "title": "《短期计划》逐项状态与下一门槛",
        "subtitle": "顺序不变；未通过上游门槛时不提前启动下游模块",
        "dataset": "short_term_plan_status",
        "sourceId": source["id"],
        "defaultSort": {"field": "order", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "order", "label": "Order", "format": "number"},
            {"field": "plan_item", "label": "Plan item", "type": "text"},
            {"field": "status", "label": "Status", "type": "text"},
            {"field": "evidence", "label": "Evidence", "type": "text"},
            {"field": "next_gate", "label": "Next gate", "type": "text"},
        ],
    },
])

manifest["blocks"].extend([
    {
        "id": "rerun_decision_20260825",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 结论：确实是训练路径问题，但‘随机’已定位到输入之后\n\n"
            "今天与昨天的B1-only不是普通超参数差异：tracked commit、配置文件、dataset manifests、initial B0 hash和前100个observation batch fingerprints都相同。"
            "参数hash却在step1不同，总loss到logged step5才出现可见差异，epoch30 Success相差14.589点。"
            "因此stateless shuffle/candidate/point sampling已经基本排除；剩余首要嫌疑是PointNet2自定义CUDA backward的归约次序，其次是Adam多参数组执行。"
            "当前checkpoint只记录更新后参数hash，尚缺step1逐参数gradient hash与Adam state hash，所以不能把具体算子写成已证实根因。"
        ),
    },
    {
        "id": "rerun_metric_strip_20260825",
        "type": "metric-strip",
        "layout": "full",
        "cardIds": ["card_b1_rerun_delta", "card_seq_rerun", "card_b0_near_seq", "card_b1_calibration"],
    },
    {"id": "b1_rerun_curve_block", "type": "chart", "layout": "full", "chartId": "chart_b1_same_arm_rerun"},
    {"id": "b1_transaction_table_block", "type": "table", "layout": "full", "tableId": "table_b1_transaction_rerun"},
    {
        "id": "b0_status_20260825",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## B0状态：恢复了分数量级，尚未固定为可复现论文基线\n\n"
            "safe B0 final Success/Precision=50.690/59.280，与SeqTrack旧高轨迹50.986/59.962非常接近。"
            "所以可以说取消泄露后B0仍能达到SeqTrack附近的性能量级。不能说已经严格复现SeqTrack：原版评估器读取当前帧GT尺寸，"
            "且同workers4的SeqTrack两次final Success为31.684和27.997；v25同臂B1的B0前缀也不能复现。"
            "论文正式基线应继续使用safe evaluator，但必须先消除step1 hash分叉或将其量化为三种子方差。"
        ),
    },
    {"id": "seqtrack_three_curve_block", "type": "chart", "layout": "full", "chartId": "chart_seqtrack_three_runs"},
    {"id": "rerun_comparison_table_block", "type": "table", "layout": "full", "tableId": "table_rerun_comparison"},
    {
        "id": "module_status_20260825",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 后续模块仍未通过各自科学门槛\n\n"
            "B1新run虽然tracking接近50分，但输出是observation；其learned RMSE比CV差0.107m、NLL=148.6、95% coverage仅31.1%，"
            "所以不能把高分归给B1，sigma也不能交给B2/B3使用。B2此前extension available仅约6–8%，target-bearing只有1–2行，"
            "raw candidate在可用行约93–97%有害且oracle headroom近零；应先修supply/retention，不应扩decoder。"
            "B3缺少held-out calibration artifact、action coverage为0；Full的52.553是该臂B0轨迹，不是选择器收益。"
        ),
    },
    {"id": "b1_epoch30_table_block", "type": "table", "layout": "full", "tableId": "table_b1_epoch30_diagnostics"},
    {
        "id": "next_action_20260825",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 下一步：先完成短期计划第0步，再进入B1\n\n"
            "1. 在同一张空闲GPU上顺序运行两次B0和两次B1的100-step audit，禁止并发；记录step1的loss、逐参数gradient hash、Adam state hash和更新后parameter hash。\n"
            "2. gradient先分叉时，优先审计PointNet2的group/gather CUDA backward；gradient一致而Adam state/parameter分叉时，将Adam显式设为foreach=False并核对参数组顺序。\n"
            "3. 只有step1与step100 B0 hash可复现，才从epoch0重跑mini四臂。然后按短期计划第1步单独修B1 mean/NLL/coverage，保持GRU和固定geometry。\n"
            "4. B1通过校准门槛后再修B2 acquisition funnel；3A/3B、4A/4B/4C和B3均不得提前。当前不应启动full nuScenes或再做无审计的60轮碰运气复跑。"
        ),
    },
    {"id": "short_term_plan_table_block", "type": "table", "layout": "full", "tableId": "table_short_term_plan"},
    {
        "id": "rerun_methodology_20260825",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 本次追加分析口径\n\n"
            "B1只比较共同完成的epoch30及epochs28–30 late-3，不用今天未完成的epoch31–60推断final。"
            "复跑一致性来自run_provenance、last.ckpt审计字段、TensorBoard和candidate diagnostics；"
            f"B1首次可见总loss差在logged step {b1_first_loss_diff}，SeqTrack新旧workers4日志的首次可见总loss差在logged step {seq_first_loss_diff}。"
            "SeqTrack历史run缺run-level commit且验证使用当前帧GT尺寸，所有SeqTrack数值仅作描述性量级比较。"
        ),
    },
])

analysis_summary = {
    "generated_at": "2026-08-25T18:00:00+08:00",
    "decision": {
        "random_path": "confirmed within the same B1-only arm after identical input transactions",
        "b0_numeric_scale_restored": True,
        "b0_transaction_reproducible": False,
        "next_plan_step": 0,
    },
    "metrics": {
        "new_b1_epoch30": new_b1_30,
        "old_b1_epoch30": old_b1_30,
        "b0_epoch60": b0_60,
        "seqtrack_rerun_epoch60": seq_new_60,
        "seqtrack_old_low_epoch60": seq_low_60,
        "seqtrack_old_high_epoch60": seq_high_60,
    },
    "b1_diagnostics": {"new": new_b1_diag, "old": old_b1_diag},
    "transaction": {
        "same_git_commit": True,
        "same_config_except_tag": True,
        "same_dataset_manifests": True,
        "same_initial_b0_hash": True,
        "same_first_100_observation_fingerprints": True,
        "same_step1_b0_hash": new_hashes["step_1"] == old_hashes["step_1"],
        "same_step100_b0_hash": new_hashes["step_100"] == old_hashes["step_100"],
        "first_visible_b1_total_loss_difference_step": b1_first_loss_diff,
        "first_visible_seq_total_loss_difference_step": seq_first_loss_diff,
    },
    "short_term_plan": plan_rows,
    "limitations": [
        "The rerun changed known server scheduling: yesterday B1 shared physical GPU1 with B0; today it ran alone.",
        "The checkpoint does not contain pre-step gradient or Adam-state hashes, so the exact CUDA/optimizer source remains unproven.",
        "The new B1 run ended cleanly at epoch 30 and is not a 60-epoch final result.",
        "SeqTrack historical runs lack run-level git provenance and use an unsafe current-frame-size evaluator.",
    ],
    "excluded": ["trajtrack", "B0 2x2", "historical leaked-score threshold"],
}

(REPORT_DIR / "analysis_summary.json").write_text(
    json.dumps(analysis_summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
(REPORT_DIR / "artifact.json").write_text(
    json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)

for filename, rows in (
    ("rerun_comparison.csv", rerun_rows),
    ("b1_transaction_audit.csv", transaction_rows),
    ("b1_epoch30_diagnostics.csv", b1_rows),
    ("short_term_plan_status.csv", plan_rows),
    ("b1_rerun_curves.csv", b1_curve_rows),
    ("seqtrack_three_curves.csv", seq_curve_rows),
):
    with (REPORT_DIR / filename).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print(REPORT_DIR)
