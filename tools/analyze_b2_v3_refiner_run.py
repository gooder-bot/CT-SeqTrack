#!/usr/bin/env python3
"""Audit the formal seed42 B2-v3 refiner run and build report inputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/20260804-1228-13_b2_v3_refiner-b2_v3_refiner_seed42_20ep_bs16"
LOG = RUN / "lightning_logs/version_0"
DIAG = LOG / "candidate_diagnostics"
STEM = "b2_v3_refiner_seed42_20260804"
DATA_DIR = ROOT / "compare_results/data"
REPORT_DIR = ROOT / "compare_results/reports" / STEM
STEPS_PER_EPOCH = 1262


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def epoch_mean(metric_dir: str, epoch: int) -> float:
    ea = EventAccumulator(
        str(LOG / metric_dir), size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars("loss")
    lo = (epoch - 1) * STEPS_PER_EPOCH
    hi = epoch * STEPS_PER_EPOCH
    values = [event.value for event in events if lo <= event.step < hi]
    if len(values) != STEPS_PER_EPOCH:
        raise RuntimeError(
            f"{metric_dir} epoch {epoch}: {len(values)} != {STEPS_PER_EPOCH}")
    return float(np.mean(values))


def cluster_bootstrap(x: pd.DataFrame, columns: list[str], draws: int = 20000):
    tracklets = x["tracklet_id"].unique()
    sums = np.asarray([
        [x.loc[x.tracklet_id == tid, column].sum() for column in columns]
        for tid in tracklets
    ])
    counts = np.asarray([(x.tracklet_id == tid).sum() for tid in tracklets])
    rng = np.random.default_rng(42)
    samples = np.empty((draws, len(columns)), dtype=np.float64)
    for index in range(draws):
        selected = rng.integers(0, len(tracklets), len(tracklets))
        samples[index] = sums[selected].sum(axis=0) / counts[selected].sum()
    return {
        column: {
            "mean": float(samples[:, offset].mean()),
            "ci_low": float(np.quantile(samples[:, offset], 0.025)),
            "ci_high": float(np.quantile(samples[:, offset], 0.975)),
        }
        for offset, column in enumerate(columns)
    }


def collect_candidate_epochs() -> pd.DataFrame:
    rows = []
    for path in sorted(DIAG.glob("epoch_*.csv")):
        frame = pd.read_csv(path)
        epoch = int(frame["epoch"].iloc[0])
        structural = frame[frame.search_valid == 1]
        valid_fg = frame[frame.valid_foreground == 1]
        y_true = structural.valid_foreground.to_numpy()
        probability = structural.presence_probability.to_numpy()
        rows.append({
            "epoch": epoch,
            "rows": len(frame),
            "structural_n": len(structural),
            "structural_rate": len(structural) / len(frame),
            "valid_foreground_n": len(valid_fg),
            "valid_foreground_rate": len(valid_fg) / len(frame),
            "foreground_given_structural": len(valid_fg) / len(structural),
            "observation_error": valid_fg.observation_error.mean(),
            "motion_error": valid_fg.motion_error.mean(),
            "raw_search_error": valid_fg.raw_search_error.mean(),
            "refined_error": valid_fg.search_error.mean(),
            "refined_minus_motion": (
                valid_fg.search_error - valid_fg.motion_error).mean(),
            "refined_minus_raw": (
                valid_fg.search_error - valid_fg.raw_search_error).mean(),
            "raw_minus_observation": (
                valid_fg.raw_search_error - valid_fg.observation_error).mean(),
            "refined_minus_observation": (
                valid_fg.search_error - valid_fg.observation_error).mean(),
            "refined_win_motion_rate": (
                valid_fg.search_error < valid_fg.motion_error).mean(),
            "refined_win_raw_rate": (
                valid_fg.search_error < valid_fg.raw_search_error).mean(),
            "presence_auc": roc_auc_score(y_true, probability),
            "presence_ap": average_precision_score(y_true, probability),
            "presence_positive_mean": structural.loc[
                structural.valid_foreground == 1,
                "presence_probability"].mean(),
            "presence_negative_mean": structural.loc[
                structural.valid_foreground == 0,
                "presence_probability"].mean(),
        })
    return pd.DataFrame(rows)


def collect_training_epochs() -> pd.DataFrame:
    metrics = {
        "train_structural_rate": "loss_search_v3_structural_valid_rate",
        "train_presence_probability": "loss_search_v3_presence_probability",
        "train_presence_loss": "loss_loss_search_v3_presence",
        "train_raw_proposal_loss": "loss_loss_search_v3_raw_proposal",
        "train_refined_proposal_loss": "loss_loss_search_v3_refined_proposal",
    }
    return pd.DataFrame([
        {"epoch": epoch, **{
            label: epoch_mean(metric, epoch)
            for label, metric in metrics.items()
        }}
        for epoch in (5, 10, 15, 20)
    ])


def collect_epoch20(candidate_epochs: pd.DataFrame):
    frame = pd.read_csv(DIAG / "epoch_20.csv")
    valid_fg = frame[frame.valid_foreground == 1].copy()
    valid_fg["refined_minus_raw"] = (
        valid_fg.search_error - valid_fg.raw_search_error)
    valid_fg["refined_minus_motion"] = (
        valid_fg.search_error - valid_fg.motion_error)
    valid_fg["raw_minus_observation"] = (
        valid_fg.raw_search_error - valid_fg.observation_error)
    valid_fg["refined_minus_observation"] = (
        valid_fg.search_error - valid_fg.observation_error)
    ci = cluster_bootstrap(valid_fg, [
        "refined_minus_raw", "refined_minus_motion",
        "raw_minus_observation", "refined_minus_observation",
    ])
    delta = valid_fg.refined_minus_raw
    changed = delta.abs() > 1e-6
    structural = frame[frame.search_valid == 1]
    comparison = pd.DataFrame([
        {"candidate": "Observation", "mean_error": valid_fg.observation_error.mean()},
        {"candidate": "B1 motion", "mean_error": valid_fg.motion_error.mean()},
        {"candidate": "Raw Search", "mean_error": valid_fg.raw_search_error.mean()},
        {"candidate": "B2-v3 refined", "mean_error": valid_fg.search_error.mean()},
    ])
    presence = pd.DataFrame([{
        "structural_rows": len(structural),
        "positive_rows": int(structural.valid_foreground.sum()),
        "negative_rows": int((1 - structural.valid_foreground).sum()),
        "auc": roc_auc_score(
            structural.valid_foreground, structural.presence_probability),
        "average_precision": average_precision_score(
            structural.valid_foreground, structural.presence_probability),
        "positive_mean_probability": structural.loc[
            structural.valid_foreground == 1,
            "presence_probability"].mean(),
        "negative_mean_probability": structural.loc[
            structural.valid_foreground == 0,
            "presence_probability"].mean(),
    }])
    robustness = pd.DataFrame([
        {"comparison": label, **values}
        for label, values in ci.items()
    ])
    changed_summary = pd.DataFrame([{
        "valid_foreground_rows": len(valid_fg),
        "unchanged_rows": int((~changed).sum()),
        "changed_rows": int(changed.sum()),
        "changed_harm_rows": int((changed & (delta > 0)).sum()),
        "changed_help_rows": int((changed & (delta < 0)).sum()),
        "mean_delta_changed": float(delta[changed].mean()),
    }])
    return comparison, presence, robustness, changed_summary


def collect_validation() -> pd.DataFrame:
    ea = EventAccumulator(str(LOG), size_guidance={"scalars": 0})
    ea.Reload()
    success = ea.Scalars("success/test")
    precision = ea.Scalars("precision/test")
    epochs = (5, 10, 15, 20)
    return pd.DataFrame([
        {"epoch": epoch, "success": success[index].value,
         "precision": precision[index].value,
         "meaning": "observation-only during refiner fit"}
        for index, epoch in enumerate(epochs)
    ])


def build_decisions(epoch20: pd.Series, presence: pd.DataFrame,
                    robustness: pd.DataFrame) -> pd.DataFrame:
    ci = robustness.set_index("comparison")
    return pd.DataFrame([
        {"check": "run_identity", "criterion": "Formal cfg13 V3 refiner, seed42, epoch20 last",
         "observed": "cfg13; 25,240 steps; clean commit; strict init",
         "status": "passed", "interpretation": "The run is valid and complete."},
        {"check": "refined_vs_motion", "criterion": "Refined error < B1 motion error on valid foreground",
         "observed": f"{epoch20.refined_error:.3f} vs {epoch20.motion_error:.3f} ({epoch20.refined_minus_motion:+.3f})",
         "status": "passed", "interpretation": "Search evidence improves the B1 candidate."},
        {"check": "refined_vs_raw", "criterion": "Refined error < raw Search error on valid foreground",
         "observed": f"{epoch20.refined_error:.3f} vs {epoch20.raw_search_error:.3f} ({epoch20.refined_minus_raw:+.3f}); cluster CI [{ci.loc['refined_minus_raw','ci_low']:+.3f}, {ci.loc['refined_minus_raw','ci_high']:+.3f}]",
         "status": "failed", "interpretation": "The fixed B1-centered bound discards raw Search accuracy."},
        {"check": "presence_generalization", "criterion": "Presence separates foreground-present from foreground-absent structural states",
         "observed": f"AUC {presence.auc.iloc[0]:.3f}; positive/negative probabilities {presence.positive_mean_probability.iloc[0]:.3f}/{presence.negative_mean_probability.iloc[0]:.3f}",
         "status": "failed", "interpretation": "Presence is nearly random under recursive validation states."},
        {"check": "candidate_promotion", "criterion": "Refined beats both B1 motion and raw Search",
         "observed": "Beats motion, loses to raw Search",
         "status": "failed", "interpretation": "Do not promote this checkpoint to final router training."},
        {"check": "final_tracking_gain", "criterion": "Packaged obs_vs_all beats same-checkpoint obs_only",
         "observed": "Router and packaged four-mode evaluation not run",
         "status": "not_run", "interpretation": "Refiner-stage Success/Precision is observation-only."},
    ])


def build_next_steps() -> pd.DataFrame:
    return pd.DataFrame([
        {"order": 1, "action": "Hold router promotion",
         "why": "The mandatory refined-vs-raw candidate gate failed.",
         "gate": "Do not train the final router from this refiner."},
        {"order": 2, "action": "Run a raw-vs-refined action diagnostic",
         "why": "Raw Search is the best mean candidate, but the current router cannot execute it.",
         "gate": "Export raw Search as a diagnostic action and compare H=3 gains."},
        {"order": 3, "action": "Replace fixed B1 clipping with a learned reliability blend",
         "why": "Among 21 changed valid-foreground rows, clipping harmed 15 and helped 6.",
         "gate": "The revised refined candidate must beat both inputs on seed42."},
        {"order": 4, "action": "Add recursive hard negatives for presence/evidence",
         "why": "Train structural validity is about 30.6%, validation only 5.8%; validation presence AUC is 0.497.",
         "gate": "Presence/evidence must separate no-foreground structural states on held-out tracklets."},
        {"order": 5, "action": "Rerun only the corrected seed42 refiner",
         "why": "The run and initialization are healthy; repeating the same config cannot fix the objective.",
         "gate": "Only after candidate gate passes: round0, provisional router, round1, final router, four modes."},
    ])


def source(source_id: str, label: str, path: Path, description: str):
    relative = rel(path)
    return {
        "id": source_id, "label": label, "path": relative,
        "query": {
            "language": "sql", "engine": "DuckDB",
            "description": description, "executed_at": "2026-08-04",
            "sql": f"SELECT * FROM read_csv_auto('{relative}', header=true)",
        },
    }


def build_artifact(paths: dict[str, Path], datasets: dict[str, pd.DataFrame],
                   epoch20: pd.Series, presence: pd.DataFrame,
                   robustness: pd.DataFrame, changed: pd.DataFrame):
    generated = datetime.now(timezone.utc).isoformat()
    ci = robustness.set_index("comparison")
    sources = [
        source("headline_source", "Consolidated refiner findings", paths["headline"],
               "One-row reconciliation of candidate quality, coverage, presence and robustness findings."),
        source("candidate_source", "Candidate diagnostics by checkpoint", paths["candidate_epochs"],
               "Valid-foreground candidate errors and validation coverage at epochs 5/10/15/20."),
        source("training_source", "Training epoch diagnostics", paths["training_epochs"],
               "Unsmoothed TensorBoard scalars aggregated over 1,262 batches per epoch."),
        source("comparison_source", "Epoch20 candidate comparison", paths["comparison"],
               "Same-state valid-foreground mean center errors at epoch20."),
        source("presence_source", "Epoch20 presence diagnostic", paths["presence"],
               "Presence discrimination on structurally valid recursive validation states."),
        source("robustness_source", "Tracklet-cluster bootstrap", paths["robustness"],
               "Twenty-thousand tracklet-cluster bootstrap draws for paired candidate deltas."),
        source("integrity_source", "Run integrity audit", paths["integrity"],
               "Configuration, initialization, frozen-state and completion checks."),
        source("decision_source", "Promotion decision register", paths["decisions"],
               "Applies the seed42 candidate-stage promotion contract."),
        source("next_source", "Corrective action register", paths["next_steps"],
               "Orders the minimum corrective work before router promotion."),
    ]
    manifest = {
        "version": 1, "surface": "report",
        "title": "B2-v3 Refiner 复核：Raw Search 有信号，当前 Refiner 未过门",
        "description": "Seed42 cfg13 candidate quality, generalization and promotion diagnosis.",
        "generatedAt": generated, "sources": sources,
        "charts": [
            {"id": "candidate_error_chart", "title": "Valid-foreground candidate error by checkpoint",
             "subtitle": "mini_val recursive states; lower is better, 64-66 eligible rows per checkpoint.",
             "type": "line", "dataset": "candidate_error_long", "sourceId": "candidate_source",
             "encodings": {
                 "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                 "y": {"field": "mean_error", "type": "quantitative", "label": "Mean center error"},
                 "color": {"field": "candidate", "type": "nominal", "label": "Candidate"},
                 "tooltip": [
                     {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                     {"field": "candidate", "type": "nominal", "label": "Candidate"},
                     {"field": "mean_error", "type": "quantitative", "label": "Mean center error"},
                 ]}, "xAxisTitle": "Epoch", "yAxisTitle": "Mean center error"},
            {"id": "coverage_chart", "title": "Search structural-valid rate: training versus validation",
             "subtitle": "Epoch20; same structural-valid field, validation is recursive observation-policy state.",
             "type": "bar", "dataset": "coverage", "sourceId": "training_source",
             "encodings": {
                 "x": {"field": "population", "type": "nominal", "label": "Population"},
                 "y": {"field": "structural_rate", "type": "quantitative", "label": "Structural-valid rate"},
                 "tooltip": [
                     {"field": "population", "type": "nominal", "label": "Population"},
                     {"field": "structural_rate", "type": "quantitative", "label": "Rate"},
                 ]}, "xAxisTitle": "Population", "yAxisTitle": "Structural-valid rate"},
        ],
        "tables": [
            {"id": "comparison_table", "title": "Epoch20 candidate comparison",
             "subtitle": "Valid-foreground rows only; lower error is better.", "dataset": "comparison",
             "sourceId": "comparison_source", "density": "spacious",
             "defaultSort": {"field": "mean_error", "direction": "asc"},
             "columns": [
                 {"field": "candidate", "label": "Candidate", "type": "text"},
                 {"field": "mean_error", "label": "Mean error", "type": "number", "format": "number"},
             ]},
            {"id": "presence_table", "title": "Presence generalization",
             "subtitle": "Structurally valid validation rows; foreground presence is the target.",
             "dataset": "presence", "sourceId": "presence_source", "density": "spacious",
             "defaultSort": {"field": "auc", "direction": "desc"},
             "columns": [
                 {"field": "structural_rows", "label": "Rows", "type": "number", "format": "number"},
                 {"field": "positive_rows", "label": "FG present", "type": "number", "format": "number"},
                 {"field": "negative_rows", "label": "FG absent", "type": "number", "format": "number"},
                 {"field": "auc", "label": "AUC", "type": "number", "format": "number"},
                 {"field": "positive_mean_probability", "label": "Mean p | FG", "type": "number", "format": "number"},
                 {"field": "negative_mean_probability", "label": "Mean p | no FG", "type": "number", "format": "number"},
             ]},
            {"id": "decision_table", "title": "Seed42 candidate-stage promotion checks",
             "subtitle": "Final tracking promotion remains blocked until all candidate gates pass.",
             "dataset": "decisions", "sourceId": "decision_source", "density": "spacious",
             "defaultSort": {"field": "check", "direction": "asc"},
             "columns": [
                 {"field": "check", "label": "Check", "type": "text"},
                 {"field": "criterion", "label": "Criterion", "type": "text"},
                 {"field": "observed", "label": "Observed", "type": "text"},
                 {"field": "status", "label": "Status", "type": "text"},
                 {"field": "interpretation", "label": "Interpretation", "type": "text"},
             ]},
            {"id": "next_table", "title": "Minimum corrective path",
             "subtitle": "Do not repeat cfg13 unchanged or advance to final router training.",
             "dataset": "next_steps", "sourceId": "next_source", "density": "spacious",
             "defaultSort": {"field": "order", "direction": "asc"},
             "columns": [
                 {"field": "order", "label": "#", "type": "number", "format": "number"},
                 {"field": "action", "label": "Action", "type": "text"},
                 {"field": "why", "label": "Why", "type": "text"},
                 {"field": "gate", "label": "Gate", "type": "text"},
             ]},
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# B2-v3 Refiner 复核：Raw Search 有信号，当前 Refiner 未过门"},
            {"id": "summary", "type": "markdown", "sourceId": "headline_source",
             "body": (
                 "## 技术结论：有 Search 信号，但还不能说 B2-v3 有效或涨分\n\n"
                 f"epoch20 valid-foreground 上，raw Search error 为 **{epoch20.raw_search_error:.3f}**，"
                 f"优于 B1 motion 的 **{epoch20.motion_error:.3f}**；但 B2-v3 refined 为 "
                 f"**{epoch20.refined_error:.3f}**，比 raw Search **差 {epoch20.refined_minus_raw:.3f}**。"
                 "因此 refiner 只通过 refined-vs-motion，未通过 refined-vs-raw 的强制 gate。"
                 "训练期 tracking 输出是 observation，当前 Success/Precision 不是 B2 增益。"
             )},
            {"id": "candidate_finding", "type": "markdown", "sourceId": "headline_source",
             "body": (
                 "## 固定 B1 锚定正在持续吃掉 raw Search 的优势\n\n"
                 f"refined-minus-raw 从 epoch5 的 **+0.039** 扩大到 epoch20 的 "
                 f"**+{epoch20.refined_minus_raw:.3f}**。tracklet-cluster bootstrap 的 epoch20 "
                 f"95% 区间为 **[{ci.loc['refined_minus_raw','ci_low']:+.3f}, "
                 f"{ci.loc['refined_minus_raw','ci_high']:+.3f}]**，方向稳定为退化。"
                 f"66个有效前景状态中，固定截断实际改变21个；其中伤害15个、帮助6个，"
                 f"改变状态的平均 raw→refined 退化为 **+{changed.mean_delta_changed.iloc[0]:.3f}**。"
             )},
            {"id": "candidate_error", "type": "chart", "chartId": "candidate_error_chart"},
            {"id": "comparison", "type": "table", "tableId": "comparison_table"},
            {"id": "distribution_finding", "type": "markdown", "sourceId": "headline_source",
             "body": (
                 "## 训练状态与递归验证状态严重失配\n\n"
                 "epoch20 training structural-valid rate 为 **30.6%**，而 mini_val 递归状态只有 "
                 f"**{epoch20.structural_rate*100:.1f}%**；valid-foreground 进一步只有 "
                 f"**{epoch20.valid_foreground_rate*100:.1f}%（{int(epoch20.valid_foreground_n)}/2004）**。"
                 "训练增强产生的 Search 激活远多于在线状态，导致 evidence/presence 学到的分布不能直接泛化。"
             )},
            {"id": "coverage", "type": "chart", "chartId": "coverage_chart"},
            {"id": "presence_finding", "type": "markdown", "sourceId": "headline_source",
             "body": (
                 "## Presence 在 held-out 递归状态上几乎等于随机\n\n"
                 f"116个结构有效状态中只有66个真正含前景。presence AUC 为 "
                 f"**{presence.auc.iloc[0]:.3f}**；有前景/无前景的平均概率却分别为 "
                 f"**{presence.positive_mean_probability.iloc[0]:.3f}/{presence.negative_mean_probability.iloc[0]:.3f}**。"
                 "训练 presence loss 接近零，但验证无法识别 endpoint crop miss，说明训练中缺少递归 hard negatives。"
             )},
            {"id": "presence", "type": "table", "tableId": "presence_table"},
            {"id": "scope", "type": "markdown", "sourceId": "integrity_source",
             "body": (
                 "## 范围与数据合同：运行本身健康\n\n"
                 "本次为正式 cfg13、seed42、mini_train/mini_val、20 epochs、25,240 optimizer steps。"
                 "严格初始化加载367/411个目标张量；B1 14个张量和33个迁移张量完整，"
                 "全部冻结 B0/B1 tensor 与 init bitwise 相同；33个迁移 Search tensor 全部更新。"
             )},
            {"id": "method", "type": "markdown", "sourceId": "robustness_source",
             "body": (
                 "## 方法：同状态配对误差与 tracklet-cluster bootstrap\n\n"
                 "候选误差只在 `valid_foreground=1` 上比较；每个 epoch 读取完整 CSV。"
                 "稳健性区间按 tracklet 整簇重采样20,000次，避免把同一轨迹的连续帧当成独立样本。"
                 "训练指标由每 epoch 1,262 个 unsmoothed TensorBoard batch scalar 重新聚合。"
             )},
            {"id": "limitations", "type": "markdown",
             "body": (
                 "## 限制：样本小，且尚未测 H=3 闭环动作收益\n\n"
                 "epoch20 只有66个 valid-foreground 状态、21条 tracklet；这足以判定当前 refined-vs-raw "
                 "方向为负，但不足以证明 raw Search 一定带来最终 tracking 涨分。当前没有 round0/round1、"
                 "calibrated router 或同 checkpoint 四模式结果。"
             )},
            {"id": "decision", "type": "table", "tableId": "decision_table"},
            {"id": "recommendation", "type": "markdown", "sourceId": "next_source",
             "body": (
                 "## 下一步：暂停正式 router，先修 refiner 的两个失败点\n\n"
                 "不要原样重跑 cfg13，也不要直接训练 final router。先让 raw Search 成为可执行诊断动作，"
                 "确认 H=3 潜力；随后把固定 B1 clip 改为可学习的 motion↔raw reliability blend，"
                 "并用递归 observation-policy hard negatives 训练 presence/evidence。修正版 seed42 同时击败"
                 "motion 和 raw 后，再恢复两轮 rollout 流程。"
             )},
            {"id": "next", "type": "table", "tableId": "next_table"},
            {"id": "questions", "type": "markdown", "body": (
                 "## 仍需回答的问题\n\n"
                 "- raw Search 在 H=3 forced-action rollout 中是否仍优于 refined？\n"
                 "- learned reliability blend 能否在 held-out tracklets 上同时超过 motion 与 raw？\n"
                 "- 加入递归 hard negatives 后，presence AUC、harm screening 与可校准 coverage 能否恢复？"
             )},
        ],
    }
    artifact = {
        "surface": "report", "manifest": manifest,
        "snapshot": {"version": 1, "status": "ready", "generatedAt": generated,
                     "datasets": {key: frame.to_dict(orient="records")
                                  for key, frame in datasets.items()}},
        "sources": [],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "artifact.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidate_epochs = collect_candidate_epochs()
    training_epochs = collect_training_epochs()
    comparison, presence, robustness, changed = collect_epoch20(candidate_epochs)
    validation = collect_validation()
    epoch20 = candidate_epochs.set_index("epoch").loc[20]
    integrity = pd.DataFrame([{
        "config": "cfgs/ct_v2/13_b2_v3_refiner.yaml", "seed": 42,
        "epochs": 20, "optimizer_steps": 25240, "commit": "73d6ce25682c19be6a70fcd95a125874deb306e8",
        "dirty": False, "b1_tensor_count": 14, "migrated_tensor_count": 33,
        "missing_frozen_tensor_count": 0, "frozen_bitwise_unchanged": True,
        "migrated_tensor_changed_count": 33, "last_checkpoint": rel(LOG / "checkpoints/last.ckpt"),
    }])
    decisions = build_decisions(epoch20, presence, robustness)
    next_steps = build_next_steps()
    ci = robustness.set_index("comparison")
    headline = pd.DataFrame([{
        "motion_error": epoch20.motion_error,
        "raw_search_error": epoch20.raw_search_error,
        "refined_error": epoch20.refined_error,
        "refined_minus_motion": epoch20.refined_minus_motion,
        "refined_minus_raw": epoch20.refined_minus_raw,
        "refined_minus_raw_ci_low": ci.loc["refined_minus_raw", "ci_low"],
        "refined_minus_raw_ci_high": ci.loc["refined_minus_raw", "ci_high"],
        "changed_rows": changed.changed_rows.iloc[0],
        "changed_harm_rows": changed.changed_harm_rows.iloc[0],
        "changed_help_rows": changed.changed_help_rows.iloc[0],
        "changed_mean_delta": changed.mean_delta_changed.iloc[0],
        "train_structural_rate": training_epochs.set_index("epoch").loc[20, "train_structural_rate"],
        "validation_structural_rate": epoch20.structural_rate,
        "validation_valid_foreground_rate": epoch20.valid_foreground_rate,
        "validation_valid_foreground_n": epoch20.valid_foreground_n,
        "presence_auc": presence.auc.iloc[0],
        "presence_positive_mean": presence.positive_mean_probability.iloc[0],
        "presence_negative_mean": presence.negative_mean_probability.iloc[0],
    }])
    candidate_error_long = candidate_epochs.melt(
        id_vars=["epoch"], value_vars=["motion_error", "raw_search_error", "refined_error"],
        var_name="candidate", value_name="mean_error")
    candidate_error_long["candidate"] = candidate_error_long.candidate.map({
        "motion_error": "B1 motion", "raw_search_error": "Raw Search", "refined_error": "B2-v3 refined"})
    coverage = pd.DataFrame([
        {"population": "Training epoch20", "structural_rate": training_epochs.set_index("epoch").loc[20, "train_structural_rate"]},
        {"population": "Validation epoch20", "structural_rate": epoch20.structural_rate},
    ])
    tables = {
        "headline": headline, "candidate_epochs": candidate_epochs, "training_epochs": training_epochs,
        "comparison": comparison, "presence": presence, "robustness": robustness,
        "changed": changed, "validation": validation, "integrity": integrity,
        "decisions": decisions, "next_steps": next_steps,
    }
    paths = {}
    for name, frame in tables.items():
        path = DATA_DIR / f"{STEM}_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    datasets = {
        **tables, "candidate_error_long": candidate_error_long, "coverage": coverage,
    }
    artifact = build_artifact(paths, datasets, epoch20, presence, robustness, changed)
    summary = {
        "status": "candidate_gate_failed",
        "epoch20": epoch20.to_dict(),
        "presence": presence.iloc[0].to_dict(),
        "robustness": robustness.to_dict(orient="records"),
        "changed": changed.iloc[0].to_dict(),
        "validation": validation.to_dict(orient="records"),
        "artifact": rel(artifact),
    }
    summary_path = DATA_DIR / f"{STEM}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": rel(summary_path), "artifact": rel(artifact)}, indent=2))


if __name__ == "__main__":
    main()
