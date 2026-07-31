#!/usr/bin/env python3
"""Analyze the completed CT-v2 B1motion-v2 seed42 screen.

The report deliberately separates:

* run completion and provenance quality;
* normal-validation outcome;
* baseline-task learning behavior before/after adapter activation;
* mechanism diagnostics that explain why exact step-0 identity did not
  preserve the baseline during training.

The epoch-60 ``last.ckpt`` is the primary checkpoint.  Best and late-window
metrics are diagnostics only and never replace the final result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "compare_results" / "data"
REPORT_DIR = ROOT / "compare_results" / "reports"
STEM = "b1motion_v2_seed42_20260730"

RUNS = {
    "B0": {
        "label": "B0 baseline (historical)",
        "path": (
            "output/"
            "20260725-2326-01_seqtrack3d_baseline-"
            "ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16"
        ),
    },
    "A0": {
        "label": "Legacy motion α=0",
        "path": (
            "output/"
            "20260728-1840-02_ct_motion_alpha000-"
            "ctv2_b1_motion_alpha000_car_seed42_60ep_bs16_"
            "gpu1_thread1_scratch"
        ),
    },
    "A025": {
        "label": "Legacy motion α=0.25",
        "path": (
            "output/"
            "20260728-1840-02_ct_motion_alpha025-"
            "ctv2_b1_motion_alpha025_car_seed42_60ep_bs16_"
            "gpu3_thread1_scratch"
        ),
    },
    "B1V2": {
        "label": "B1motion-v2 ordered + pre-crop",
        "path": (
            "output/"
            "20260730-0305-02_ct_motion-"
            "ctv2_b1_motion_v2_car_seed42_60ep_bs16_"
            "gpu2_thread1_scratch"
        ),
    },
}

TRAINING_LEAVES = {
    "loss_total": "loss_loss_total",
    "loss_center": "loss_loss_center",
    "loss_center_aux": "loss_loss_center_aux",
    "loss_center_motion": "loss_loss_center_motion",
    "loss_center_ref": "loss_loss_center_ref",
    "loss_angle": "loss_loss_angle",
    "loss_angle_aux": "loss_loss_angle_aux",
    "loss_angle_motion": "loss_loss_angle_motion",
    "loss_angle_ref": "loss_loss_angle_ref",
    "loss_seg": "loss_loss_seg",
    "loss_bc": "loss_loss_bc",
    "loss_velocity": "loss_loss_velocity",
    "loss_dynamics_displacement": "loss_loss_dynamics_displacement",
    "loss_trajectory_nll": "loss_loss_trajectory_nll",
    "loss_trajectory_adapter_norm": "loss_loss_trajectory_adapter_norm",
    "trajectory_sigma_mean": "loss_trajectory_sigma_mean",
    "trajectory_adapter_norm": "loss_trajectory_adapter_norm_mean",
    "trajectory_adapter_scale": "loss_trajectory_adapter_scale_mean",
    "trajectory_gap_activation": "loss_trajectory_gap_activation_mean",
    "trajectory_gap_ratio": "loss_trajectory_gap_ratio_mean",
    "trajectory_search_valid": "loss_trajectory_search_valid_mean",
    "trajectory_search_gap_ratio": "loss_trajectory_search_gap_ratio_mean",
    "search_expansion_points": "loss_ct_search_expansion_points_mean",
    "search_query_delta_t": "loss_ct_search_query_delta_t_mean",
    "search_predicted_displacement": (
        "loss_ct_search_predicted_displacement_mean"
    ),
    "foreground_score": "loss_obs_mean_fg_score",
    "foreground_points": "loss_obs_estimated_fg_points_mean",
    "valid_history_ratio": "loss_obs_valid_history_ratio",
}

CORE_LOSSES = (
    "loss_total",
    "loss_center",
    "loss_center_aux",
    "loss_center_motion",
    "loss_center_ref",
    "loss_angle",
    "loss_angle_aux",
    "loss_angle_motion",
    "loss_angle_ref",
    "loss_seg",
    "loss_bc",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_events(version_dir: Path, leaf: str) -> list[tuple[int, float]]:
    event_dir = version_dir / leaf
    if not event_dir.is_dir():
        return []
    accumulator = EventAccumulator(
        str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        return []
    return [
        (int(item.step), float(item.value))
        for item in accumulator.Scalars(tags[0])
    ]


def install_easydict_fallback() -> None:
    try:
        __import__("easydict")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("easydict")

    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

        def __setattr__(self, key, value):
            self[key] = value

    EasyDict.__module__ = "easydict"
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module


def checkpoint_metadata(path: Path, include_parameter_norms: bool) -> dict[str, Any]:
    install_easydict_fallback()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", {})
    result: dict[str, Any] = {
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "checkpoint_state_tensors": len(state_dict),
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256_file(path),
    }
    if not include_parameter_norms:
        return result

    prefixes = (
        "dynamics_encoder.",
        "trajectory_adapter.",
        "trajectory_search_encoder.",
    )
    for prefix in prefixes:
        tensors = [
            tensor.detach().float().reshape(-1)
            for key, tensor in state_dict.items()
            if key.startswith(prefix) and torch.is_tensor(tensor)
        ]
        if tensors:
            concatenated = torch.cat(tensors)
            result[f"{prefix[:-1]}_parameter_l2"] = float(
                torch.linalg.vector_norm(concatenated))
            result[f"{prefix[:-1]}_parameter_count"] = int(
                concatenated.numel())

    last_weight_key = "trajectory_adapter.net.2.weight"
    last_bias_key = "trajectory_adapter.net.2.bias"
    if last_weight_key in state_dict:
        result["trajectory_adapter_last_weight_l2"] = float(
            torch.linalg.vector_norm(
                state_dict[last_weight_key].detach().float()))
    if last_bias_key in state_dict:
        result["trajectory_adapter_last_bias_l2"] = float(
            torch.linalg.vector_norm(
                state_dict[last_bias_key].detach().float()))
    return result


def expected_training_shape(
        provenance: dict[str, Any]) -> tuple[int, int, int]:
    config = provenance["resolved_config"]
    frames = int(provenance["datasets"]["train"]["frames"])
    candidates = int(config.get("num_candidates", 1))
    batch_size = int(config["batch_size"])
    epochs = int(config["epoch"])
    steps_per_epoch = (frames * candidates) // batch_size
    expected_steps = steps_per_epoch * epochs
    expected_validations = (
        epochs // int(config["check_val_every_n_epoch"]))
    return steps_per_epoch, expected_steps, expected_validations


def epoch_means(
        events: list[tuple[int, float]], steps_per_epoch: int, epochs: int
) -> list[float | None]:
    if not events:
        return [None] * epochs
    by_step = {step: value for step, value in events}
    means: list[float | None] = []
    for epoch_index in range(epochs):
        start = epoch_index * steps_per_epoch
        values = [
            by_step[step]
            for step in range(start, start + steps_per_epoch)
            if step in by_step
        ]
        means.append(float(np.mean(values)) if values else None)
    return means


def collect_run(run_id: str, root: Path) -> dict[str, Any]:
    spec = RUNS[run_id]
    run_dir = root / spec["path"]
    version_dir = run_dir / "lightning_logs" / "version_0"
    provenance = read_json(run_dir / "run_provenance.json")
    steps_per_epoch, expected_steps, expected_validations = (
        expected_training_shape(provenance))
    config = provenance["resolved_config"]
    epochs = int(config["epoch"])

    success = scalar_events(version_dir, "metrics_test_success")
    precision = scalar_events(version_dir, "metrics_test_precision")
    if [row[0] for row in success] != [row[0] for row in precision]:
        raise RuntimeError(f"{run_id}: validation scalar steps do not match")
    validation = [
        {
            "run_id": run_id,
            "arm": spec["label"],
            "epoch": int(step // steps_per_epoch),
            "step": step,
            "success": success_value,
            "precision": precision_value,
        }
        for (step, success_value), (_, precision_value)
        in zip(success, precision)
    ]

    training_events = {
        metric: scalar_events(version_dir, leaf)
        for metric, leaf in TRAINING_LEAVES.items()
    }
    training_epochs = {
        metric: epoch_means(events, steps_per_epoch, epochs)
        for metric, events in training_events.items()
    }
    total_events = training_events["loss_total"]
    checkpoint = checkpoint_metadata(
        version_dir / "checkpoints" / "last.ckpt",
        include_parameter_norms=run_id == "B1V2",
    )
    training_complete = (
        len(total_events) == expected_steps
        and total_events[-1][0] == expected_steps - 1
        and len(validation) == expected_validations
        and checkpoint["checkpoint_epoch"] == epochs - 1
        and checkpoint["checkpoint_global_step"] == expected_steps
    )
    if not training_complete:
        raise RuntimeError(f"{run_id}: run is not complete")

    return {
        "run_id": run_id,
        "label": spec["label"],
        "run_dir": run_dir,
        "version_dir": version_dir,
        "provenance": provenance,
        "config": config,
        "steps_per_epoch": steps_per_epoch,
        "expected_steps": expected_steps,
        "expected_validations": expected_validations,
        "validation": validation,
        "training_events": training_events,
        "training_epochs": training_epochs,
        "training_complete": training_complete,
        **checkpoint,
    }


def mean_not_none(values: list[float | None]) -> float:
    selected = [value for value in values if value is not None]
    return float(np.mean(selected))


def summarize_run(run: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    validation = run["validation"]
    final = validation[-1]
    best_success = max(validation, key=lambda row: row["success"])
    best_precision = max(validation, key=lambda row: row["precision"])
    late3 = validation[-3:]
    baseline_final = baseline["validation"][-1]
    return {
        "run_id": run["run_id"],
        "arm": run["label"],
        "status": "COMPLETE",
        "final_success": round(final["success"], 6),
        "final_precision": round(final["precision"], 6),
        "delta_success_vs_b0": round(
            final["success"] - baseline_final["success"], 6),
        "delta_precision_vs_b0": round(
            final["precision"] - baseline_final["precision"], 6),
        "best_success": round(best_success["success"], 6),
        "best_success_epoch": best_success["epoch"],
        "best_precision": round(best_precision["precision"], 6),
        "best_precision_epoch": best_precision["epoch"],
        "late3_success": round(mean_not_none(
            [row["success"] for row in late3]), 6),
        "late3_precision": round(mean_not_none(
            [row["precision"] for row in late3]), 6),
    }


def build_training_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run_id in ("B0", "A0", "B1V2"):
        run = runs[run_id]
        epochs = int(run["config"]["epoch"])
        for epoch_index in range(epochs):
            row = {
                "run_id": run_id,
                "arm": run["label"],
                "epoch": epoch_index + 1,
            }
            for metric in TRAINING_LEAVES:
                value = run["training_epochs"][metric][epoch_index]
                row[metric] = (
                    None if value is None else round(float(value), 8))
            rows.append(row)
    return rows


def build_integrity_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run_id in ("B0", "A0", "A025", "B1V2"):
        run = runs[run_id]
        provenance = run["provenance"]
        rows.append({
            "run_id": run_id,
            "arm": run["label"],
            "training_status": "COMPLETE",
            "commit": provenance["git"]["commit"],
            "dirty_tracked": provenance["git"]["dirty_tracked"],
            "seed": provenance["seed"],
            "training_steps": len(run["training_events"]["loss_total"]),
            "expected_steps": run["expected_steps"],
            "validation_count": len(run["validation"]),
            "expected_validations": run["expected_validations"],
            "checkpoint_epoch": run["checkpoint_epoch"] + 1,
            "checkpoint_global_step": run["checkpoint_global_step"],
            "checkpoint_state_tensors": run["checkpoint_state_tensors"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "train_selection_sha256": provenance["datasets"]["train"][
                "virtual_rate_selection_sha256"],
            "val_selection_sha256": provenance["datasets"]["val"][
                "virtual_rate_selection_sha256"],
        })
    return rows


def value(run: dict[str, Any], metric: str, epoch: int) -> float:
    result = run["training_epochs"][metric][epoch - 1]
    if result is None:
        raise KeyError(f"{run['run_id']} lacks {metric} at epoch {epoch}")
    return float(result)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.inf


def build_driver_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    b0 = runs["B0"]
    v2 = runs["B1V2"]
    config = v2["config"]
    return [
        {
            "priority": 1,
            "finding": "主干在 adapter 开启前已经退化",
            "evidence": (
                f"epoch1 total loss {value(v2, 'loss_total', 1):.3f} vs "
                f"B0 {value(b0, 'loss_total', 1):.3f}; "
                f"center loss {value(v2, 'loss_center', 1):.3f} vs "
                f"{value(b0, 'loss_center', 1):.3f}. "
                "adapter warmup=2，epoch1–2 correction 精确为 0。"
            ),
            "interpretation": (
                "零初始化没有失败；35% 跳帧历史已先改变整个 B0 "
                "点云/框序列的训练分布。"
            ),
            "confidence": "verified",
        },
        {
            "priority": 2,
            "finding": "跳帧输入对 B0 主干不可辨识",
            "evidence": (
                f"trajectory_training_irregular_probability="
                f"{config['trajectory_training_irregular_probability']}; "
                "main_time_source=order，等间隔 token 不编码真实 frame gap。"
            ),
            "interpretation": (
                "相同时间 token 对应不同物理间隔，主干被要求拟合互相冲突的"
                "运动尺度。"
            ),
            "confidence": "verified-code",
        },
        {
            "priority": 3,
            "finding": "trajectory target 含输入不可识别的 anchor error",
            "evidence": (
                "encoder 只看以 candidate anchor 归一化的相对历史；"
                "trajectory label 却包含 current GT − candidate anchor。"
            ),
            "interpretation": (
                "对所有历史与 anchor 加共同平移，encoder 输入不变而 target "
                "改变；motion head 无法从自身输入恢复该误差。"
            ),
            "confidence": "verified-code",
        },
        {
            "priority": 4,
            "finding": "adapter 开启后不是受限的小残差",
            "evidence": (
                f"epoch3 correction L2={value(v2, 'trajectory_adapter_norm', 3):.3f}; "
                f"epoch60={value(v2, 'trajectory_adapter_norm', 60):.3f}; "
                f"raw norm² penalty={value(v2, 'loss_trajectory_adapter_norm', 60):.3f}, "
                f"effective penalty≈"
                f"{value(v2, 'loss_trajectory_adapter_norm', 60) * config['trajectory_adapter_l2_weight']:.4f}."
            ),
            "interpretation": (
                "normal_scale=0.1 只是输出乘数，不是范数上限；L2 权重过小，"
                "无法维持 B0 特征邻域。"
            ),
            "confidence": "verified",
        },
        {
            "priority": 5,
            "finding": "pre-crop 第二分支实际覆盖过低",
            "evidence": (
                f"训练 trajectory_search_valid mean="
                f"{100 * mean_not_none(v2['training_epochs']['trajectory_search_valid']):.2f}%; "
                f"配置 irregular probability="
                f"{100 * config['trajectory_training_irregular_probability']:.0f}%."
            ),
            "interpretation": (
                "大多数跳帧样本没有足够 extension points，search 分支不足以"
                "学习或证明 irregular robustness。"
            ),
            "confidence": "verified",
        },
        {
            "priority": 6,
            "finding": "invalid history 的 gap-ratio 诊断/输入未屏蔽",
            "evidence": (
                f"epoch60 mean gap ratio="
                f"{value(v2, 'trajectory_gap_ratio', 60):.1f}; "
                "transition_count=0 时 nominal_gap 被 clamp 到 0.001。"
            ),
            "interpretation": (
                "最终 correction 会被 valid=0 清零，因此不是主崩溃来源；"
                "但它污染诊断并留下不必要的数值风险。"
            ),
            "confidence": "verified-secondary",
        },
    ]


def build_next_experiment_rows() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "experiment": "E0 same-code B0, seed42, scratch",
            "change": (
                "用当前代码重新训练纯 B0；固定同一初始化、数据选择和 normal cadence。"
            ),
            "decision": "消除历史 commit/手工 patch 混杂，建立真正对照。",
        },
        {
            "order": 2,
            "experiment": "K1 data-shift-only kill test",
            "change": (
                "保留 ordered head，但 adapter_scale=0、search=false；"
                "只比较 irregular_probability=0 与 0.35。"
            ),
            "decision": "直接量化混合 cadence 对 B0 主干的伤害。",
        },
        {
            "order": 3,
            "experiment": "K2 adapter-only normal kill test",
            "change": (
                "irregular_probability=0、search=false；normal adapter 从严格 0 "
                "开始，只测试带相对范数硬上限的残差。"
            ),
            "decision": "验证 adapter 是否能在不改数据分布时保持/改善 normal。",
        },
        {
            "order": 4,
            "experiment": "K3 paired irregular branch",
            "change": (
                "主监督始终使用连续 B0 view；独立 irregular view 只训练 trajectory/"
                "search 与 consistency，不替换主干历史。"
            ),
            "decision": "保留 irregular 目标，同时避免污染 normal baseline 学习。",
        },
        {
            "order": 5,
            "experiment": "Search observability audit",
            "change": (
                "分开记录 trigger、extension point count、GT target recall（仅离线诊断）"
                "和 foreground evidence；不要直接降低 min_points。"
            ),
            "decision": "确认 search 是召回目标还是只扩大背景。",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value_: float) -> str:
    return f"{value_:.3f}"


def build_markdown(
        runs: dict[str, dict[str, Any]],
        summaries: list[dict[str, Any]],
        drivers: list[dict[str, Any]],
) -> str:
    b0 = runs["B0"]
    a0 = runs["A0"]
    v2 = runs["B1V2"]
    b0_final = b0["validation"][-1]
    v2_final = v2["validation"][-1]
    v2_summary = next(row for row in summaries if row["run_id"] == "B1V2")
    best_delta_success = (
        v2_summary["best_success"] - b0_final["success"])
    best_delta_precision = (
        v2_summary["best_precision"] - b0_final["precision"])
    late_total_ratio = ratio(
        value(v2, "loss_total", 60), value(b0, "loss_total", 60))
    valid_search = mean_not_none(
        v2["training_epochs"]["trajectory_search_valid"])
    adapter_effective_penalty = (
        value(v2, "loss_trajectory_adapter_norm", 60)
        * v2["config"]["trajectory_adapter_l2_weight"]
    )

    lines = [
        "# B1motion-v2 seed42 60-epoch 实验分析",
        "",
        "更新日期：2026-07-30",
        "",
        "## 结论",
        "",
        "**不涨点，且当前实现必须判定为 No-Go。** B1motion-v2 在 "
        "nuScenes-mini Car normal validation 的 epoch60 为 "
        f"**{v2_final['success']:.3f} Success / "
        f"{v2_final['precision']:.3f} Precision**，相对 B0 的 "
        f"{b0_final['success']:.3f} / {b0_final['precision']:.3f} 下降 "
        f"**{v2_final['success'] - b0_final['success']:.3f} / "
        f"{v2_final['precision'] - b0_final['precision']:.3f}**。"
        "这不是 final checkpoint 偶然抖动：它的最佳点也只有 "
        f"{v2_summary['best_success']:.3f} / "
        f"{v2_summary['best_precision']:.3f}，仍低于 B0 final "
        f"{best_delta_success:.3f} / {best_delta_precision:.3f}。",
        "",
        "本轮没有发现 B1motion-v2 的 random20 或 gap1124 结果，因此不能"
        "声称 irregular 协议涨点。即使之后某个 irregular 指标变好，normal "
        "下降 32.742 / 44.551 也已远超项目守门线，不能晋级。",
        "",
        "## 运行完整性与可比性",
        "",
        "- B1motion-v2 有 75,720/75,720 个训练 scalar、12/12 个验证点，"
        "last checkpoint 为 epoch60 / global step75,720；训练本身完整。",
        "- checkpoint 含 344 个 state tensors，与初始化检查的 "
        "320 个 B0 shared tensors + 24 个新增 tensors 一致。",
        "- provenance 的 `dirty_tracked=true` 来自服务器手工上传补丁；文件列表"
        "与 B1motion-v2 修改清单一致。这降低了可复现性等级，但不表示运行截断。",
        "- B0 是历史 commit `d86990c`，B1motion-v2 基于 `5f260e7` 加手工"
        "补丁。初始化检查证明 step-0 shared tensors 相同，但仍缺少当前最终代码"
        "上的 same-code B0 scratch，对精确因果效应需保留此 caveat。",
        "",
        "## Normal 验证结果",
        "",
        "| arm | final S | final P | ΔS vs B0 | ΔP vs B0 | best S (ep) | "
        "best P (ep) | late-3 S/P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['arm']} | {row['final_success']:.3f} | "
            f"{row['final_precision']:.3f} | "
            f"{row['delta_success_vs_b0']:+.3f} | "
            f"{row['delta_precision_vs_b0']:+.3f} | "
            f"{row['best_success']:.3f} ({row['best_success_epoch']}) | "
            f"{row['best_precision']:.3f} ({row['best_precision_epoch']}) | "
            f"{row['late3_success']:.3f} / "
            f"{row['late3_precision']:.3f} |"
        )

    lines.extend([
        "",
        "B1motion-v2 在 epoch5 达到本轮最好点 "
        f"{v2_summary['best_success']:.3f} / "
        f"{v2_summary['best_precision']:.3f}，之后总体持续下降；late-3 只有 "
        f"{v2_summary['late3_success']:.3f} / "
        f"{v2_summary['late3_precision']:.3f}。因此不是继续多训即可恢复的"
        "欠拟合，也不能用 early checkpoint 规避。",
        "",
        "## 退化从哪里开始",
        "",
        "### 1. adapter 关闭时已经先伤害主干",
        "",
        "`trajectory_adapter_warmup_epoch=2`，所以 epoch1–2 的 adapter "
        "correction 精确为 0。但 epoch1 B1motion-v2 total loss 已为 "
        f"{value(v2, 'loss_total', 1):.3f}，B0 仅 "
        f"{value(b0, 'loss_total', 1):.3f}；center loss 为 "
        f"{value(v2, 'loss_center', 1):.3f} vs "
        f"{value(b0, 'loss_center', 1):.3f}。这证明 **zero-init 本身生效，"
        "但它只保证相同输入下的 step-0 网络恒等，不保证训练数据仍是 B0**。",
        "",
        "当前 sampler 以 35% 概率把整个主路径的 `[t-1,t-2,t-3]` 改成"
        "真实跳帧历史。历史点云、历史框、motion label 和 Transformer 序列都"
        "随之改变；与此同时 `main_time_source=order` 仍把这些帧标成等间隔"
        "顺序 token。于是相同的主干时间输入对应不同物理 gap，B0 主干无法辨别"
        "运动尺度。这是最早、证据最强的退化来源。",
        "",
        "### 2. trajectory target 对 encoder 本身不可识别",
        "",
        "ordered encoder 输入是以最新 candidate crop anchor 归一化后的相对"
        "历史框；但 `anchor_relative_trajectory_targets()` 令 target 等于"
        "`current GT − candidate anchor`，把最新 anchor 的采样误差也塞进"
        "“velocity/displacement”。对全部历史框与 anchor 加同一个平移，encoder "
        "输入保持不变，而 target 会改变。因此 trajectory-only head 从数学上"
        "无法恢复这个平移误差；75% 非 candidate0 样本给它的是含不可观测项的"
        "监督。它把物理 motion 与 observation/refinement 应负责的定位修正混在"
        "一起，偏离了 M²-Track 的 motion proposal → observation refinement 分工。",
        "",
        "### 3. epoch3 后 adapter 残差过强",
        "",
        "adapter 在 epoch3 启用后，correction L2 立即达到 "
        f"{value(v2, 'trajectory_adapter_norm', 3):.3f}，epoch5 为 "
        f"{value(v2, 'trajectory_adapter_norm', 5):.3f}，epoch60 仍为 "
        f"{value(v2, 'trajectory_adapter_norm', 60):.3f}。"
        "`trajectory_adapter_normal_scale=0.1` 只是对 MLP 输出乘 0.1，"
        "并没有限制最终范数；epoch60 raw norm² penalty 为 "
        f"{value(v2, 'loss_trajectory_adapter_norm', 60):.3f}，乘 "
        f"`1e-4` 后只贡献约 {adapter_effective_penalty:.4f}，几乎不能约束"
        "feature drift。exact-zero 初始化因此在训练后失去保护作用。",
        "",
        "### 4. pre-crop search 覆盖不足",
        "",
        f"`trajectory_search_valid` 全程均值仅 {100 * valid_search:.2f}%，"
        "远低于 35% irregular sampling。说明多数触发候选没有达到 16 个"
        "extension points，第二 crop 分支很少真正提供额外观测。当前日志还把"
        "trigger 与“采到足够点”合并成一个 valid，无法判断是几何 trigger 少、"
        "还是扩区只有背景/空点。它既未救回 normal，也尚未形成可靠的 irregular "
        "学习信号。",
        "",
        "### 5. 训练没有回到 B0 解附近",
        "",
        f"epoch60 total loss 为 {value(v2, 'loss_total', 60):.3f}，"
        f"B0 为 {value(b0, 'loss_total', 60):.3f}，约 "
        f"{late_total_ratio:.2f} 倍。核心任务并非只有一个 auxiliary loss 变坏：",
        "",
        "| epoch60 component | B0 | α=0 legacy | B1motion-v2 | v2/B0 |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric in (
        "loss_center",
        "loss_center_aux",
        "loss_center_motion",
        "loss_center_ref",
        "loss_seg",
        "loss_bc",
        "loss_total",
    ):
        b0_value = value(b0, metric, 60)
        a0_value = value(a0, metric, 60)
        v2_value = value(v2, metric, 60)
        lines.append(
            f"| `{metric}` | {b0_value:.4f} | {a0_value:.4f} | "
            f"{v2_value:.4f} | {ratio(v2_value, b0_value):.2f}× |"
        )

    lines.extend([
        "",
        "foreground score/point count并未同步崩塌，说明问题不是简单的"
        "segmentation 看不到点；更符合“历史/目标合同冲突 + adapter feature "
        "drift 破坏 coarse motion 与后续 Transformer query”的模式。",
        "",
        "## 次要实现问题",
        "",
    ])
    for driver in drivers:
        if driver["priority"] < 5:
            continue
        lines.append(
            f"- **{driver['finding']}：** {driver['evidence']} "
            f"{driver['interpretation']}"
        )
    lines.extend([
        "",
        "`trajectory_nll` 变为负数本身不是 bug：Gaussian NLL 中的 "
        "`log_sigma` 项允许负值。真正的问题是 endpoint/velocity error 仍高且"
        "表示被送入 adapter；需要看 RMSE、NLL 与 calibration，而不是把 NLL "
        "符号当作成败。",
        "",
        "## 模块判定",
        "",
        "- **当前 B1motion-v2：不能涨点，停止 60-epoch、多 seed 和 full "
        "nuScenes 晋级。**",
        "- **不能据此否定所有 motion/trajectory 方向。** 有序编码、crop 前"
        "使用因果 motion、保留 B0 token、step-0 zero-init 都是合理原则；失败"
        "来自训练与可识别性合同没有一起修好。",
        "- **当前没有 random20/gap1124 证据。** 在 normal 修复前，跑这两个"
        "协议最多用于诊断，不能改变 No-Go 判定。",
        "",
        "## 下一步：先做归因 kill test，不再直接跑 60 epoch",
        "",
        "1. 用当前代码跑一个 same-code B0 scratch，消除 commit/手工补丁混杂。",
        "2. 做 `irregular_probability 0/0.35 × adapter off/on` 的短周期"
        "受控拆分；search 先关闭，避免三个因素同时变化。",
        "3. 主监督始终保留连续 B0 view；irregular view 改成独立 paired branch，"
        "只训练 trajectory/search/consistency，不替换主干历史。",
        "4. 将 normal adapter 永久设为 exact identity；irregular correction "
        "增加相对 feature norm 硬上限和 GT-free evidence gate，不能只乘 "
        "`normal_scale`。",
        "5. 拆分 target：trajectory head 预测 canonical physical motion；"
        "candidate-anchor correction 交给 observation/refinement，或让修正 head "
        "显式读取当前点云证据。",
        "6. 修复 invalid gap ratio：invalid 样本置 1，valid ratio 设合理上限；"
        "日志分别记录 trigger、extension availability 和 applied。",
        "",
        "建议每个 kill test 先跑 10–15 epoch、每 5 epoch 验证。只有 normal "
        "相对 same-code B0 达到 Success ≥ −0.3、Precision ≥ −0.5，且 core "
        "loss 不出现当前式分叉，才允许继续 60 epoch 和 irregular 协议。",
        "",
        "## 证据边界",
        "",
        "- 所有直接结果只有 seed42 / nuScenes-mini Car；效应非常大，足以否决"
        "当前实现，但不能估计修正版的统计方差。",
        "- B0 与 B1motion-v2 不在同一提交上；step-0 shared tensor identity "
        "降低了初始化混杂，却不能替代 same-code B0 retrain。",
        "- 没有 per-tracklet endpoint export，也没有本轮 random20/gap1124 "
        "输出；无法定位是否存在少量 irregular helpful subgroup。",
    ])
    return "\n".join(lines) + "\n"


def build_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": "validation_source",
            "label": "Reviewed TensorBoard normal-validation metrics",
            "path": f"compare_results/data/{STEM}_validation.csv",
            "query": {
                "language": "python",
                "engine": "DuckDB",
                "sql": (
                    "SELECT run_id, arm, epoch, step, success, precision "
                    f"FROM read_csv_auto('compare_results/data/{STEM}_validation.csv', "
                    "header = true) ORDER BY run_id, epoch"
                ),
                "description": (
                    "tools/analyze_b1motion_v2.py extracts the twelve reviewed "
                    "five-epoch validation checkpoints from each completed run."
                ),
                "executed_at": "2026-07-30",
                "filters": [
                    "nuScenes v1.0-mini Car validation",
                    "normal cadence",
                    "seed=42",
                    "60 training epochs",
                ],
                "metric_definitions": [
                    "Final metric: epoch-60 validation scalar from last.ckpt run.",
                    "Best metric: maximum among epochs 5,10,...,60; diagnostic only.",
                    "Late-3: arithmetic mean at epochs 50,55,60.",
                ],
            },
        },
        {
            "id": "training_source",
            "label": "Reviewed epoch-level training diagnostics",
            "path": f"compare_results/data/{STEM}_training_epochs.csv",
            "query": {
                "language": "python",
                "engine": "DuckDB",
                "sql": (
                    "SELECT * FROM read_csv_auto("
                    f"'compare_results/data/{STEM}_training_epochs.csv', "
                    "header = true) ORDER BY run_id, epoch"
                ),
                "description": (
                    "Per-epoch arithmetic means over exactly 1,262 logged "
                    "training steps, extracted from TensorBoard scalar leaves."
                ),
                "executed_at": "2026-07-30",
                "filters": [
                    "B0, legacy alpha=0, and B1motion-v2",
                    "seed=42",
                    "no smoothing",
                ],
                "metric_definitions": [
                    "loss_total is the logged optimization objective.",
                    "trajectory_adapter_norm is per-sample correction L2, averaged.",
                    "trajectory_search_valid is the share with a sampled extension.",
                ],
            },
        },
        {
            "id": "integrity_source",
            "label": "Run provenance and checkpoint integrity",
            "path": f"compare_results/data/{STEM}_integrity.csv",
            "query": {
                "language": "python",
                "description": (
                    "Completion checks combine run_provenance.json, all training "
                    "scalar steps, validation count, and last.ckpt metadata."
                ),
                "executed_at": "2026-07-30",
            },
        },
        {
            "id": "code_source",
            "label": "B1motion-v2 implementation audit",
            "path": "models/ct_v2/motion.py",
            "query": {
                "language": "python",
                "description": (
                    "Code review covers models/ct_v2/motion.py, "
                    "models/seqtrack3d.py, datasets/sampler.py, "
                    "utils/candidate_utils.py, utils/ct_history.py, and "
                    "datasets/misc_utils.py."
                ),
                "executed_at": "2026-07-30",
                "filters": [
                    "ordered trajectory path",
                    "trajectory adapter path",
                    "mixed-cadence sampling",
                    "candidate-anchor targets",
                ],
            },
        },
        {
            "id": "driver_source",
            "label": "Root-cause evidence register",
            "path": f"compare_results/data/{STEM}_drivers.csv",
            "query": {
                "language": "sql",
                "engine": "DuckDB",
                "sql": (
                    "SELECT priority, finding, evidence, interpretation, confidence "
                    f"FROM read_csv_auto('compare_results/data/{STEM}_drivers.csv', "
                    "header = true) ORDER BY priority"
                ),
                "description": (
                    "Reads the evidence register generated from reviewed "
                    "TensorBoard diagnostics and the named implementation paths."
                ),
                "executed_at": "2026-07-30",
            },
        },
        {
            "id": "next_experiments_source",
            "label": "Controlled follow-up experiment register",
            "path": f"compare_results/data/{STEM}_next_experiments.csv",
            "query": {
                "language": "sql",
                "engine": "DuckDB",
                "sql": (
                    "SELECT \"order\", experiment, change, decision "
                    "FROM read_csv_auto("
                    f"'compare_results/data/{STEM}_next_experiments.csv', "
                    "header = true) ORDER BY \"order\""
                ),
                "description": (
                    "Reads the ordered experiment plan produced by the "
                    "B1motion-v2 diagnostic analysis."
                ),
                "executed_at": "2026-07-30",
            },
        },
    ]


def build_artifact(
        summaries: list[dict[str, Any]],
        validation_rows: list[dict[str, Any]],
        training_rows: list[dict[str, Any]],
        drivers: list[dict[str, Any]],
        next_experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    sources = build_sources()
    v2 = next(row for row in summaries if row["run_id"] == "B1V2")
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "B1motion-v2 seed42 60-epoch 实验分析",
            "description": (
                "Normal-validation decision, learning-curve diagnostics, "
                "code-contract audit, and next experiments."
            ),
            "generatedAt": generated_at,
            "sources": sources,
            "charts": [
                {
                    "id": "success_curve",
                    "title": "Normal validation Success by epoch",
                    "subtitle": (
                        "B1motion-v2 peaks at epoch 5 and then diverges sharply "
                        "from B0."
                    ),
                    "type": "line",
                    "dataset": "validation_metrics",
                    "sourceId": "validation_source",
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        "y": {
                            "field": "success",
                            "type": "quantitative",
                            "label": "Success",
                        },
                        "color": {
                            "field": "arm",
                            "type": "nominal",
                            "label": "Run",
                        },
                        "tooltip": [
                            {"field": "arm", "type": "nominal", "label": "Run"},
                            {
                                "field": "epoch",
                                "type": "quantitative",
                                "label": "Epoch",
                            },
                            {
                                "field": "success",
                                "type": "quantitative",
                                "label": "Success",
                            },
                        ],
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Success",
                },
                {
                    "id": "precision_curve",
                    "title": "Normal validation Precision by epoch",
                    "subtitle": (
                        "The center-localization regression is already visible "
                        "at epoch 5 and grows through late training."
                    ),
                    "type": "line",
                    "dataset": "validation_metrics",
                    "sourceId": "validation_source",
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        "y": {
                            "field": "precision",
                            "type": "quantitative",
                            "label": "Precision",
                        },
                        "color": {
                            "field": "arm",
                            "type": "nominal",
                            "label": "Run",
                        },
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Precision",
                },
                {
                    "id": "training_loss_curve",
                    "title": "Mean training loss by epoch",
                    "subtitle": (
                        "B1motion-v2 remains far above both B0 and legacy alpha=0 "
                        "after adapter activation."
                    ),
                    "type": "line",
                    "dataset": "training_epochs",
                    "sourceId": "training_source",
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        "y": {
                            "field": "loss_total",
                            "type": "quantitative",
                            "label": "Mean total loss",
                        },
                        "color": {
                            "field": "arm",
                            "type": "nominal",
                            "label": "Run",
                        },
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Mean total loss",
                },
            ],
            "tables": [
                {
                    "id": "run_summary",
                    "title": "Run-level normal-validation summary",
                    "subtitle": (
                        "Epoch-60 last.ckpt is primary; best and late-3 are "
                        "stability diagnostics."
                    ),
                    "dataset": "run_summary",
                    "sourceId": "validation_source",
                    "density": "compact",
                    "defaultSort": {
                        "field": "final_success",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "arm", "label": "Run", "type": "text"},
                        {
                            "field": "final_success",
                            "label": "Final S",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "final_precision",
                            "label": "Final P",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "delta_success_vs_b0",
                            "label": "ΔS vs B0",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "delta_precision_vs_b0",
                            "label": "ΔP vs B0",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "best_success",
                            "label": "Best S",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "best_precision",
                            "label": "Best P",
                            "type": "number",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "driver_evidence",
                    "title": "Root-cause evidence",
                    "subtitle": (
                        "Verified observations are separated from causal "
                        "interpretation and secondary hygiene issues."
                    ),
                    "dataset": "driver_evidence",
                    "sourceId": "driver_source",
                    "density": "compact",
                    "defaultSort": {
                        "field": "priority",
                        "direction": "asc",
                    },
                    "columns": [
                        {
                            "field": "priority",
                            "label": "#",
                            "type": "number",
                        },
                        {
                            "field": "finding",
                            "label": "Finding",
                            "type": "text",
                        },
                        {
                            "field": "evidence",
                            "label": "Evidence",
                            "type": "text",
                        },
                        {
                            "field": "interpretation",
                            "label": "Interpretation",
                            "type": "text",
                        },
                        {
                            "field": "confidence",
                            "label": "Confidence",
                            "type": "text",
                        },
                    ],
                },
                {
                    "id": "next_experiments",
                    "title": "Controlled next experiments",
                    "subtitle": (
                        "Short attribution tests precede any new 60-epoch run."
                    ),
                    "dataset": "next_experiments",
                    "sourceId": "next_experiments_source",
                    "density": "compact",
                    "defaultSort": {
                        "field": "order",
                        "direction": "asc",
                    },
                    "columns": [
                        {"field": "order", "label": "#", "type": "number"},
                        {
                            "field": "experiment",
                            "label": "Experiment",
                            "type": "text",
                        },
                        {"field": "change", "label": "Change", "type": "text"},
                        {
                            "field": "decision",
                            "label": "Decision use",
                            "type": "text",
                        },
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# B1motion-v2 seed42 60-epoch 实验分析",
                },
                {
                    "id": "decision",
                    "type": "markdown",
                    "sourceId": "validation_source",
                    "body": (
                        "## 决策：当前实现 No-Go\n\n"
                        f"Epoch60 为 **{v2['final_success']:.3f} Success / "
                        f"{v2['final_precision']:.3f} Precision**，相对 B0 "
                        f"下降 **{v2['delta_success_vs_b0']:.3f} / "
                        f"{v2['delta_precision_vs_b0']:.3f}**。最佳点仍未"
                        "超过 B0，且没有 random20/gap1124 结果。当前模块不能"
                        "晋级多 seed、full dataset 或新的 60-epoch sweep。"
                    ),
                },
                {
                    "id": "summary_table_block",
                    "type": "table",
                    "tableId": "run_summary",
                },
                {
                    "id": "curve_interpretation",
                    "type": "markdown",
                    "sourceId": "validation_source",
                    "body": (
                        "## 验证曲线\n\n"
                        "B1motion-v2 在 epoch5 达到本轮最好点，随后总体下降；"
                        "late-3 也没有恢复。因此退化不是 final checkpoint 抖动或"
                        "简单欠训练。"
                    ),
                },
                {
                    "id": "success_chart_block",
                    "type": "chart",
                    "chartId": "success_curve",
                },
                {
                    "id": "precision_chart_block",
                    "type": "chart",
                    "chartId": "precision_curve",
                },
                {
                    "id": "learning_behavior",
                    "type": "markdown",
                    "sourceId": "training_source",
                    "body": (
                        "## 退化时点\n\n"
                        "Adapter 在 epoch1–2 被 warmup 强制为零，但 B1motion-v2 "
                        "主任务损失已经高于 B0；这证明 zero-init 生效，而 mixed-"
                        "cadence 数据分布先破坏了 B0 学习。Epoch3 adapter 启用后，"
                        "correction norm 立即跃升，形成第二次扰动。"
                    ),
                },
                {
                    "id": "training_chart_block",
                    "type": "chart",
                    "chartId": "training_loss_curve",
                },
                {
                    "id": "root_causes",
                    "type": "markdown",
                    "sourceId": "code_source",
                    "body": (
                        "## 根因\n\n"
                        "35% 跳帧替换了整个主干历史，但 `main_time_source=order` "
                        "仍隐藏真实 gap；trajectory head 的 candidate-anchor target "
                        "还包含相对历史输入不可识别的共同 anchor 平移误差。"
                        "`normal_scale=0.1` 又只是乘数而非范数上限，无法在训练后"
                        "维持 baseline identity。"
                    ),
                },
                {
                    "id": "driver_table_block",
                    "type": "table",
                    "tableId": "driver_evidence",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "sourceId": "code_source",
                    "body": (
                        "## 下一步\n\n"
                        "先重训 same-code B0，再用短周期 factorial kill test 拆分"
                        "mixed cadence 与 adapter。主监督保持连续 B0 view；irregular "
                        "history 改成 paired auxiliary branch。Trajectory head 只预测"
                        "可识别的 physical motion，anchor correction 必须读取当前点云"
                        "证据并受相对范数硬上限约束。"
                    ),
                },
                {
                    "id": "next_table_block",
                    "type": "table",
                    "tableId": "next_experiments",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 证据边界\n\n"
                        "只有 seed42 / mini normal 聚合指标；B0 与 B1motion-v2 "
                        "提交不同，虽通过 shared initialization identity 检查，仍需"
                        "same-code B0。没有 per-tracklet export，也没有本轮 random20 "
                        "或 gap1124 输出。效应幅度足以否决当前实现，但不能外推为"
                        "所有 motion prior 均无效。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "validation_metrics": validation_rows,
                "run_summary": summaries,
                "training_epochs": training_rows,
                "driver_evidence": drivers,
                "next_experiments": next_experiments,
            },
        },
        "sources": sources,
        "package_info": {
            "title": "B1motion-v2 seed42 60-epoch 实验分析",
            "generated_at": generated_at,
        },
    }
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()

    runs = {
        run_id: collect_run(run_id, root)
        for run_id in ("B0", "A0", "A025", "B1V2")
    }
    baseline = runs["B0"]
    summaries = [
        summarize_run(runs[run_id], baseline)
        for run_id in ("B0", "A0", "A025", "B1V2")
    ]
    validation_rows = [
        {
            **row,
            "success": round(float(row["success"]), 6),
            "precision": round(float(row["precision"]), 6),
        }
        for run_id in ("B0", "A0", "A025", "B1V2")
        for row in runs[run_id]["validation"]
    ]
    training_rows = build_training_rows(runs)
    integrity_rows = build_integrity_rows(runs)
    drivers = build_driver_rows(runs)
    next_experiments = build_next_experiment_rows()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DATA_DIR / f"{STEM}_validation.csv", validation_rows)
    write_csv(DATA_DIR / f"{STEM}_summary.csv", summaries)
    write_csv(DATA_DIR / f"{STEM}_training_epochs.csv", training_rows)
    write_csv(DATA_DIR / f"{STEM}_integrity.csv", integrity_rows)
    write_csv(DATA_DIR / f"{STEM}_drivers.csv", drivers)
    write_csv(
        DATA_DIR / f"{STEM}_next_experiments.csv", next_experiments)

    checkpoint_details = {
        key: value
        for key, value in runs["B1V2"].items()
        if key.startswith("checkpoint_")
        or key.endswith("_parameter_l2")
        or key.endswith("_parameter_count")
        or key.startswith("trajectory_adapter_last_")
    }
    (DATA_DIR / f"{STEM}_checkpoint_details.json").write_text(
        json.dumps(checkpoint_details, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = REPORT_DIR / f"{STEM}.md"
    report_path.write_text(
        build_markdown(runs, summaries, drivers), encoding="utf-8")
    artifact = build_artifact(
        summaries,
        validation_rows,
        training_rows,
        drivers,
        next_experiments,
    )
    artifact_path = REPORT_DIR / f"{STEM}_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"report: {report_path}")
    print(f"artifact: {artifact_path}")
    for row in summaries:
        print(
            f"{row['run_id']}: final="
            f"{fmt(row['final_success'])}/{fmt(row['final_precision'])} "
            f"delta_vs_b0="
            f"{row['delta_success_vs_b0']:+.3f}/"
            f"{row['delta_precision_vs_b0']:+.3f}"
        )
    print(json.dumps(checkpoint_details, indent=2))


if __name__ == "__main__":
    main()
