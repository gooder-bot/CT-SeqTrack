"""Extend the v25 four-arm report with frozen SeqTrack and prior-B1 evidence.

The script is read-only for every ``output/`` directory.  It deliberately
excludes trajtrack and materializes only derived tables under
``artifacts/ct_checks/reports``.
"""

from __future__ import annotations

import copy
import csv
import json
import sqlite3
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
SEQ_ROOT = ROOT.parent / "seqtrack"
BASE_REPORT = ROOT / "artifacts/ct_checks/reports/20260824_v25_four_arm"
REPORT_DIR = ROOT / "artifacts/ct_checks/reports/20260825_seqtrack_comparison"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def scalar_series(run: Path, tag: str):
    event_root = run / "lightning_logs/version_0"
    accumulator = EventAccumulator(str(event_root))
    accumulator.Reload()
    return accumulator.Scalars(tag)


def final_metrics(run: Path, success_tag: str, precision_tag: str):
    success = scalar_series(run, success_tag)
    precision = scalar_series(run, precision_tag)
    if not success or len(success) != len(precision):
        raise RuntimeError(f"invalid validation series: {run}")
    late3_success = sum(row.value for row in success[-3:]) / min(3, len(success))
    late3_precision = sum(row.value for row in precision[-3:]) / min(3, len(precision))
    return {
        "final_success": success[-1].value,
        "final_precision": precision[-1].value,
        "late3_success": late3_success,
        "late3_precision": late3_precision,
        "validation_points": len(success),
    }


seq_runs = {
    "SeqTrack-60-high": SEQ_ROOT / "output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16",
    "SeqTrack-60-low": SEQ_ROOT / "output/20260801-2155-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16_seed42",
    "SeqTrack-180-gpu1": SEQ_ROOT / "output/20260629-1644-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_180ep_bs16_gpu1",
    "SeqTrack-180-gpu3": SEQ_ROOT / "output/20260702-0038-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_180ep_bs16_gpu3",
}
ct_runs = {
    "CT21-B0": ROOT / "output/20260813-0116-01_seqtrack3d_baseline-scratch_ct21_b0_car_60ep_bs16_s42",
    "CT21-B1": ROOT / "output/20260813-0119-02_ct_motion_v3-scratch_ct21_b1_only_car_60ep_bs16_s42",
    "v24-B0": ROOT / "output/20260822-2246-24_b0-ct24_b0_restore_seed42",
    "v24-B1": ROOT / "output/20260822-2252-24_b1-ct24_b1_dual_stream_fix_seed42",
}

seq_metrics = {
    name: final_metrics(path, "success/test", "precision/test")
    for name, path in seq_runs.items()
}

# The two 60-epoch SeqTrack logs record identical hparams except for the tag
# and worker count.  Verify this claim from the saved files instead of relying
# on directory names.
high_hparams = (
    seq_runs["SeqTrack-60-high"] / "lightning_logs/version_0/hparams.yaml"
).read_text(encoding="utf-8")
low_hparams = (
    seq_runs["SeqTrack-60-low"] / "lightning_logs/version_0/hparams.yaml"
).read_text(encoding="utf-8")


def normalize_seqtrack_hparams(text):
    normalized = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("tag:"):
            normalized.append(f"{' ' * (len(line) - len(line.lstrip()))}tag: <TAG>")
        elif stripped.startswith("workers:"):
            normalized.append(f"{' ' * (len(line) - len(line.lstrip()))}workers: <WORKERS>")
        else:
            normalized.append(line)
    return normalized


if normalize_seqtrack_hparams(high_hparams) != normalize_seqtrack_hparams(low_hparams):
    raise RuntimeError("SeqTrack high/low hparams differ beyond tag/workers")
ct_metrics = {
    name: final_metrics(
        path,
        "success/mini_val" if name.startswith("v24") else "success/test",
        "precision/mini_val" if name.startswith("v24") else "precision/test",
    )
    for name, path in ct_runs.items()
}
v25_summary = json.loads(
    (BASE_REPORT / "analysis_summary.json").read_text(encoding="utf-8")
)
v25_metrics = {row["arm"]: row for row in v25_summary["validation_summary"]}


def run_row(label, family, metrics, workers, cadence, evaluator, comparability, note):
    return {
        "run": label,
        "family": family,
        "final_success": metrics["final_success"],
        "final_precision": metrics["final_precision"],
        "late3_success": metrics["late3_success"],
        "late3_precision": metrics["late3_precision"],
        "validation_points": metrics["validation_points"],
        "workers": workers,
        "validation_cadence": cadence,
        "evaluator": evaluator,
        "comparability": comparability,
        "interpretation": note,
    }


comparison_rows = [
    run_row(
        "SeqTrack-60-high", "SeqTrack", seq_metrics["SeqTrack-60-high"], 12,
        "每5 epoch", "原版验证（读取当前帧 GT 尺寸）", "描述性",
        "高分随机轨迹；旧运行未保存代码 commit",
    ),
    run_row(
        "SeqTrack-60-low", "SeqTrack", seq_metrics["SeqTrack-60-low"], 4,
        "每5 epoch", "原版验证（读取当前帧 GT 尺寸）", "描述性",
        "低分随机轨迹；日志配置除 workers/tag 外与高分 run 相同",
    ),
    run_row(
        "SeqTrack-180-gpu1", "SeqTrack", seq_metrics["SeqTrack-180-gpu1"], 12,
        "每5 epoch", "原版验证（读取当前帧 GT 尺寸）", "描述性",
        "180 epoch 最终结果；不能与60 epoch final直接等同",
    ),
    run_row(
        "SeqTrack-180-gpu3", "SeqTrack", seq_metrics["SeqTrack-180-gpu3"], 12,
        "每5 epoch", "原版验证（读取当前帧 GT 尺寸）", "描述性",
        "相同日志配置的另一条180 epoch轨迹",
    ),
    run_row(
        "CT21-B0", "CT21", ct_metrics["CT21-B0"], 4,
        "每5 epoch", "旧 CT evaluator", "非隔离消融",
        "旧协议 B0；无跨臂 prefix hash",
    ),
    run_row(
        "CT21-B1", "CT21", ct_metrics["CT21-B1"], 4,
        "每5 epoch", "旧 CT evaluator", "非隔离消融",
        "correlated-candidate + legacy proposal fusion + fused loss",
    ),
    run_row(
        "v24-B0", "CT-v24", ct_metrics["v24-B0"], None,
        "每 epoch", "safe recursive evaluator", "同版本描述",
        "step1 prefix 已与其余 v24 臂分叉",
    ),
    run_row(
        "v24-B1", "CT-v24", ct_metrics["v24-B1"], None,
        "每 epoch", "safe recursive evaluator", "同版本描述",
        "proposal=observation；高于 v24 B0 不能归因于 B1",
    ),
]
for arm in ("B0", "B1-only", "Full"):
    comparison_rows.append(run_row(
        f"v25-{arm}", "CT-v25", v25_metrics[arm], 4,
        "每 epoch", "safe_first_frame_size_recursive_prediction",
        "v25正式协议内",
        (
            "独立安全 B0" if arm == "B0" else
            "proposal=observation；跟踪分数是该臂 B0 轨迹" if arm == "B1-only" else
            "无 calibration artifact，B3 fail-closed 到 observation"
        ),
    ))

bar_order = [
    "SeqTrack-60-high", "SeqTrack-60-low", "CT21-B0", "CT21-B1",
    "v24-B0", "v24-B1", "v25-B0", "v25-B1-only", "v25-Full",
]
bar_rows = [
    {**row, "order": bar_order.index(row["run"])}
    for row in comparison_rows if row["run"] in bar_order
]

seqtrack_curve_rows = []
for run_name in ("SeqTrack-60-high", "SeqTrack-60-low"):
    success = scalar_series(seq_runs[run_name], "success/test")
    precision = scalar_series(seq_runs[run_name], "precision/test")
    for index, (s_row, p_row) in enumerate(zip(success, precision), start=1):
        seqtrack_curve_rows.append({
            "run": run_name,
            "epoch": index * 5,
            "success": s_row.value,
            "precision": p_row.value,
            "workers": 12 if run_name.endswith("high") else 4,
        })

b1_comparison_rows = [
    {
        "version": "CT21-B1",
        "final_success": ct_metrics["CT21-B1"]["final_success"],
        "paired_b0_success": ct_metrics["CT21-B0"]["final_success"],
        "apparent_delta": ct_metrics["CT21-B1"]["final_success"] - ct_metrics["CT21-B0"]["final_success"],
        "b0_isolated": "否",
        "why_not_comparable": "correlated candidate、legacy post-Transformer fusion、fused/gate loss进入总损失",
    },
    {
        "version": "v24-B1",
        "final_success": ct_metrics["v24-B1"]["final_success"],
        "paired_b0_success": ct_metrics["v24-B0"]["final_success"],
        "apparent_delta": ct_metrics["v24-B1"]["final_success"] - ct_metrics["v24-B0"]["final_success"],
        "b0_isolated": "否",
        "why_not_comparable": "四臂从step1分叉；B1 learned MSE 还比CV差4.3%",
    },
    {
        "version": "v25-B1",
        "final_success": v25_metrics["B1-only"]["final_success"],
        "paired_b0_success": v25_metrics["B0"]["final_success"],
        "apparent_delta": v25_metrics["B1-only"]["final_success"] - v25_metrics["B0"]["final_success"],
        "b0_isolated": "否",
        "why_not_comparable": "输入/RNG一致但step1参数hash分叉；proposal仍为observation",
    },
]

random_evidence_rows = [
    {
        "evidence": "SeqTrack 60ep 同seed日志配置",
        "observation": "hparams仅workers 12→4及tag变化，Success 50.986→31.684（−19.302）",
        "supports": "worker/随机事务可以把模型送入高、低两个训练盆地",
        "confidence": "强支持；旧run无代码commit，非唯一变量证明",
    },
    {
        "evidence": "v25 四臂 B0 prefix",
        "observation": "initial和前100批输入一致，但step1参数hash全部不同",
        "supports": "分叉发生在第一次backward/Adam更新附近，而不是模块推理收益",
        "confidence": "已验证",
    },
    {
        "evidence": "v25 B1-only部署路径",
        "observation": "proposal=observation，final Success=33.443",
        "supports": "低分是该臂B0轨迹，不是B1 proposal直接伤害",
        "confidence": "已验证",
    },
    {
        "evidence": "v25 Full部署路径",
        "observation": "calibration缺失、B3 applied rate=0，final Success=52.553",
        "supports": "高分是Full臂自己的B0轨迹，不是B3有效",
        "confidence": "已验证",
    },
]

headline = [{
    "v25_b0_success": v25_metrics["B0"]["final_success"],
    "seqtrack_high_success": seq_metrics["SeqTrack-60-high"]["final_success"],
    "v25_b0_minus_seqtrack_high": (
        v25_metrics["B0"]["final_success"]
        - seq_metrics["SeqTrack-60-high"]["final_success"]
    ),
    "seqtrack_high_low_spread": (
        seq_metrics["SeqTrack-60-high"]["final_success"]
        - seq_metrics["SeqTrack-60-low"]["final_success"]
    ),
    "v25_b1_minus_v24_b1": (
        v25_metrics["B1-only"]["final_success"]
        - ct_metrics["v24-B1"]["final_success"]
    ),
    "full_b3_action_coverage": v25_summary["b3_validation"]["router_applied_rate"],
}]

snapshot = {
    "comparison_headline": headline,
    "comparison_runs": comparison_rows,
    "comparison_bars": bar_rows,
    "seqtrack_curves": seqtrack_curve_rows,
    "b1_history_comparison": b1_comparison_rows,
    "random_path_evidence": random_evidence_rows,
}

database_path = REPORT_DIR / "comparison_data.sqlite"
query = (
    "SELECT snapshot_json FROM comparison_snapshot "
    "WHERE snapshot_id = 'seqtrack_random_path_20260825'"
)
with sqlite3.connect(database_path) as connection:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS comparison_snapshot "
        "(snapshot_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO comparison_snapshot(snapshot_id, snapshot_json) "
        "VALUES (?, ?)",
        ("seqtrack_random_path_20260825", json.dumps(snapshot, ensure_ascii=False)),
    )
    selected = connection.execute(query).fetchone()
if selected is None or json.loads(selected[0]) != snapshot:
    raise RuntimeError("comparison SQLite snapshot verification failed")

source = {
    "id": "src_seqtrack_random_path",
    "label": "SeqTrack mini、CT21/v24/v25 TensorBoard 与协议审计的派生快照",
    "path": "artifacts/ct_checks/reports/20260825_seqtrack_comparison/comparison_data.sqlite",
    "query": {
        "engine": "sqlite",
        "sql": query,
        "tables_used": ["comparison_snapshot"],
        "description": "读取本报告经核对的有界比较快照",
    },
}

artifact = copy.deepcopy(json.loads(
    (BASE_REPORT / "artifact.json").read_text(encoding="utf-8")
))
manifest = artifact["manifest"]
manifest["title"] = "CT-SeqTrack 与 SeqTrack mini 随机轨迹诊断（2026-08-25）"
manifest["description"] = "v25四臂、历史B1与原版SeqTrack mini的可比性和随机训练轨迹诊断。"
manifest["generatedAt"] = "2026-08-25T00:00:00+08:00"
manifest["sources"].append(source)
artifact["sources"].append(source)
artifact["snapshot"]["generatedAt"] = "2026-08-25T00:00:00+08:00"
artifact["snapshot"]["datasets"].update(snapshot)

# Keep the original title block id and all prior evidence sections.
manifest["blocks"][0]["body"] = f"# {manifest['title']}"
executive = next(block for block in manifest["blocks"] if block["id"] == "executive_summary")
executive["sourceId"] = "src_seqtrack_random_path"
executive["body"] = (
    "## 技术摘要\n\n"
    "**v25 独立 B0 已在数值上回到原版 SeqTrack 的高分区间，但当前证据更支持‘随机训练轨迹恢复’，而不是已经完成严格事务复现。** "
    "v25 B0 为 50.690/59.280；原版 SeqTrack 高分 run 为 50.986/59.962，Success 只差 −0.295。"
    "但原版验证读取当前帧 GT 尺寸且旧 run 未记录代码 commit，所以这只能作为量级比较，不能作为安全等价证明。\n\n"
    "**v25 B1-only 的 33.443 不是 B1 在部署时把跟踪结果拉低。** 该臂仍输出 observation，且它与 B0 从 step1 起参数 hash 已分叉；"
    "同理，Full 的 52.553 发生在无 calibration artifact、B3 applied rate=0 的 fail-closed 状态，主要是 Full 臂自己的高分 B0 轨迹。\n\n"
    "原版 SeqTrack 两条60 epoch日志的已记录配置仅 workers=12/4和tag不同，Success 却为50.986/31.684。"
    "这与 v24/v25 的高低盆地模式高度一致，强烈支持 worker/RNG/CUDA/optimizer 事务导致随机路径分叉；"
    "但由于 SeqTrack 历史 run 缺少代码提交身份，仍需 step1 gradient/Adam hash 实验完成最终归因。"
)

manifest["cards"].extend([
    {
        "id": "card_v25_vs_seq",
        "dataset": "comparison_headline",
        "sourceId": source["id"],
        "description": "仅作数值量级比较；SeqTrack旧验证不满足safe evaluator。",
        "metrics": [
            {"label": "v25 B0 Success", "field": "v25_b0_success", "format": "number"},
            {"label": "vs Seq-high", "field": "v25_b0_minus_seqtrack_high", "format": "number", "signed": True},
        ],
    },
    {
        "id": "card_seq_spread",
        "dataset": "comparison_headline",
        "sourceId": source["id"],
        "description": "原版SeqTrack两条60ep seed42日志的高低轨迹差。",
        "metrics": [
            {"label": "SeqTrack轨迹跨度", "field": "seqtrack_high_low_spread", "format": "number"},
        ],
    },
    {
        "id": "card_b1_drop",
        "dataset": "comparison_headline",
        "sourceId": source["id"],
        "description": "v25与v24 B1-only的表面分差；两者均未通过B0 prefix隔离。",
        "metrics": [
            {"label": "v25 B1 − v24 B1", "field": "v25_b1_minus_v24_b1", "format": "number", "signed": True},
        ],
    },
    {
        "id": "card_full_action",
        "dataset": "comparison_headline",
        "sourceId": source["id"],
        "description": "Full最终分数发生时，B3实际部署动作覆盖率。",
        "metrics": [
            {"label": "Full B3 action coverage", "field": "full_b3_action_coverage", "format": "percent"},
        ],
    },
])

manifest["charts"].extend([
    {
        "id": "chart_cross_version_success",
        "title": "60 epoch mini final Success 对照",
        "subtitle": "不同协议/评估器混合，仅用于识别约31分与约50分的训练轨迹盆地",
        "intent": "comparison",
        "question": "历史B1与SeqTrack结果是否呈现相同的高低训练轨迹模式？",
        "rationale": "九条离散运行的一项同单位指标适合用柱图展示量级和双峰结构。",
        "type": "bar",
        "dataset": "comparison_bars",
        "sourceId": source["id"],
        "encodings": {
            "x": {"field": "run", "type": "nominal", "label": "Run"},
            "y": {"field": "final_success", "type": "quantitative", "label": "Final Success", "unit": "points"},
            "color": {"field": "family", "type": "nominal", "label": "Protocol family"},
            "tooltip": [
                {"field": "final_precision", "type": "quantitative", "label": "Final Precision"},
                {"field": "evaluator", "type": "nominal", "label": "Evaluator"},
                {"field": "comparability", "type": "nominal", "label": "Comparability"},
            ],
        },
        "layout": "full",
    },
    {
        "id": "chart_seqtrack_high_low",
        "title": "原版 SeqTrack 60 epoch mini Success 曲线",
        "subtitle": "两条日志均为seed42、batch16、每5 epoch验证；已记录配置仅workers/tag不同",
        "intent": "trend",
        "question": "SeqTrack的31分是否是末期偶发下降，还是从早期就进入低分轨迹？",
        "rationale": "每条12个有序验证点足以显示高低轨迹从epoch5开始分离并持续到epoch60。",
        "type": "line",
        "dataset": "seqtrack_curves",
        "sourceId": source["id"],
        "encodings": {
            "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
            "y": {"field": "success", "type": "quantitative", "label": "Success", "unit": "points"},
            "color": {"field": "run", "type": "nominal", "label": "Run"},
            "tooltip": [
                {"field": "precision", "type": "quantitative", "label": "Precision"},
                {"field": "workers", "type": "quantitative", "label": "Workers"},
            ],
        },
        "layout": "full",
    },
])

manifest["tables"].extend([
    {
        "id": "table_comparison_runs",
        "title": "SeqTrack、旧B1与v25运行对照",
        "subtitle": "final指标、运行协议及可比性；180ep运行单独标注",
        "dataset": "comparison_runs",
        "sourceId": source["id"],
        "defaultSort": {"field": "final_success", "direction": "desc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "run", "label": "Run", "type": "text"},
            {"field": "final_success", "label": "Final S", "format": "number"},
            {"field": "final_precision", "label": "Final P", "format": "number"},
            {"field": "workers", "label": "Workers", "format": "number"},
            {"field": "validation_cadence", "label": "Val cadence", "type": "text"},
            {"field": "comparability", "label": "Comparability", "type": "text"},
            {"field": "interpretation", "label": "Interpretation", "type": "text"},
        ],
    },
    {
        "id": "table_random_evidence",
        "title": "随机训练路径证据分级",
        "subtitle": "区分已验证事实、强支持证据和仍缺失的因果识别",
        "dataset": "random_path_evidence",
        "sourceId": source["id"],
        "defaultSort": {"field": "confidence", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "evidence", "label": "Evidence", "type": "text"},
            {"field": "observation", "label": "Observed", "type": "text"},
            {"field": "supports", "label": "Supports", "type": "text"},
            {"field": "confidence", "label": "Confidence", "type": "text"},
        ],
    },
    {
        "id": "table_b1_history",
        "title": "B1跨版本表面分数与隔离状态",
        "subtitle": "表面B1−B0差值不是模块净增益，只有matched prefix后才可归因",
        "dataset": "b1_history_comparison",
        "sourceId": source["id"],
        "defaultSort": {"field": "version", "direction": "asc"},
        "density": "spacious",
        "layout": "full",
        "columns": [
            {"field": "version", "label": "Version", "type": "text"},
            {"field": "final_success", "label": "B1 Final S", "format": "number"},
            {"field": "paired_b0_success", "label": "B0 Final S", "format": "number"},
            {"field": "apparent_delta", "label": "Apparent Δ", "format": "number", "movement": True},
            {"field": "b0_isolated", "label": "B0 isolated", "type": "text"},
            {"field": "why_not_comparable", "label": "Why not causal", "type": "text"},
        ],
    },
])

comparison_blocks = [
    {
        "id": "comparison_definitions",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 数值接近 SeqTrack 高分轨迹，但不是安全等价证明\n\n"
            "v25 B0 的 final Success/Precision 为 **50.690/59.280**；原版 SeqTrack 的高分60ep run为 **50.986/59.962**，"
            "差值仅 −0.295/−0.682。数值上可以说 B0 已回到 SeqTrack 的正常高分区间。"
            "但是 SeqTrack 旧验证读取当前帧 GT 尺寸，且没有 run-level commit 身份；因此不能将这两个数用于论文中的严格安全对齐声明。"
        ),
    },
    {
        "id": "comparison_metrics",
        "type": "metric-strip",
        "layout": "full",
        "cardIds": ["card_v25_vs_seq", "card_seq_spread", "card_b1_drop", "card_full_action"],
    },
    {
        "id": "comparison_chart_explanation",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 约31分和约50分更像两种B0训练轨迹，而不是模块开关的稳定效应\n\n"
            "跨版本结果反复落入两个区间：SeqTrack-low、v24-B0和v25-B1接近31–33分；SeqTrack-high、CT21/v24高轨迹及v25-B0/Full接近48–55分。"
            "由于v24/v25都从step1发生B0 hash分叉，柱图只能显示轨迹盆地，不能当作模块消融增益。"
        ),
    },
    {"id": "comparison_chart", "type": "chart", "layout": "full", "chartId": "chart_cross_version_success"},
    {"id": "comparison_table", "type": "table", "layout": "full", "tableId": "table_comparison_runs"},
    {
        "id": "seqtrack_random_finding",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 原版 SeqTrack 的低分从早期形成，随机事务解释得到强支持\n\n"
            "两条60ep SeqTrack日志均记录seed42、batch16、Adam 1e-4、preloading和每5 epoch验证；hparams逐行差异仅为workers 12→4和tag。"
            "高分轨迹在epoch5已达40.697，随后维持约46–52；低分轨迹epoch5为32.303，之后一直约27–32，最终相差19.302点。"
            "这不是第60轮checkpoint偶然抖动，而是训练早期进入了不同盆地。旧run没有代码commit，所以worker变化是主要嫌疑，不是已被单变量实验证实的唯一原因。"
        ),
    },
    {"id": "seqtrack_curve", "type": "chart", "layout": "full", "chartId": "chart_seqtrack_high_low"},
    {"id": "random_evidence_table", "type": "table", "layout": "full", "tableId": "table_random_evidence"},
    {
        "id": "b1_history_finding",
        "type": "markdown",
        "layout": "full",
        "sourceId": source["id"],
        "body": (
            "## 旧B1高分不等于当前B1模块退化\n\n"
            "CT21 B1曾达到55.397，但它不是当前B1-only的同口径版本：它使用correlated-candidate历史、legacy post-Transformer proposal fusion，"
            "并把fused/gate loss加入总损失，因而改变了B0训练目标和梯度。v24 B1为47.658，但其B0也从step1与v24 B0分叉，且learned motion MSE比CV差4.3%。"
            "当前v25 B1-only虽然跟踪只有33.443，部署输出仍是observation，而B1 learned RMSE在自身验证行上反而略优于CV 0.101m。"
            "因此可确认的是B1臂拿到了一条低分B0轨迹；不能据此判定B1网络突然失效，也不能保留旧B1高分作为模块增益。"
        ),
    },
    {"id": "b1_history_table", "type": "table", "layout": "full", "tableId": "table_b1_history"},
]

insert_at = next(
    index for index, block in enumerate(manifest["blocks"])
    if block["id"] == "headline_metrics"
)
manifest["blocks"][insert_at:insert_at] = comparison_blocks
manifest["blocks"].append({
    "id": "further_questions_20260825",
    "type": "markdown",
    "layout": "full",
    "sourceId": source["id"],
    "body": (
        "## 尚需一个最小实验回答的开放问题\n\n"
        "真正未决的不是‘是否存在随机轨迹’，而是第一次更新的差异来自backward还是Adam。"
        "在同一张空闲GPU顺序运行四臂，记录step1逐参数gradient hash、Adam state hash和更新后parameter hash即可二分："
        "gradient先分叉指向PointNet2/自定义CUDA backward；gradient一致而参数分叉指向optimizer foreach/参数组事务。"
        "在这个审计通过前，任何B1或Full相对B0的分差都不具备因果解释力。"
    ),
})

analysis_summary = {
    "generated_at": "2026-08-25T00:00:00+08:00",
    "comparison_runs": comparison_rows,
    "b1_history_comparison": b1_comparison_rows,
    "random_path_evidence": random_evidence_rows,
    "headline": headline[0],
    "seqtrack_60_hparams_check": {
        "equal_after_normalizing_tag_and_workers": True,
        "high_workers": 12,
        "low_workers": 4,
        "material_logged_differences": ["tag", "workers"],
    },
    "excluded": ["trajtrack", "B0 2x2", "historical leaked-score threshold"],
    "source_limitations": [
        "SeqTrack historical runs do not contain run-level git commit provenance.",
        "SeqTrack historical evaluator reads current-frame GT size and is descriptive only.",
        "CT21/v24/v25 are different training protocols; only within-protocol paired audits can support module attribution.",
    ],
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
    ("comparison_runs.csv", comparison_rows),
    ("seqtrack_curves.csv", seqtrack_curve_rows),
    ("b1_history_comparison.csv", b1_comparison_rows),
    ("random_path_evidence.csv", random_evidence_rows),
):
    with (REPORT_DIR / filename).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print(REPORT_DIR)
