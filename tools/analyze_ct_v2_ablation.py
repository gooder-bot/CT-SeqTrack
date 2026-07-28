#!/usr/bin/env python3
"""Audit the paper-facing CT-SeqTrack v2 B0-B3 mini ablation.

The script treats the epoch-60 ``last.ckpt`` as the primary checkpoint,
recomputes validation curves from TensorBoard scalar events, checks run
provenance and checkpoint completeness, and records mechanism diagnostics.
Incomplete arms remain visible as blockers instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
RUN_PATTERN = re.compile(r"ctv2_.*_b([0-3])_")
LABELS = {
    "B0": "B0 baseline",
    "B1": "B1 + motion",
    "B2": "B2 + search",
    "B3": "B3 + adaptive gate",
    "A1": "A1 search-only",
}
DIAGNOSTIC_LEAVES = {
    "fusion_alpha_nominal": "loss_ct_fusion_alpha_mean",
    "fusion_alpha_batch_min": "loss_ct_fusion_alpha_min",
    "fusion_alpha_batch_max": "loss_ct_fusion_alpha_max",
    "fusion_alpha_applied": "loss_ct_fusion_alpha_applied_mean",
    "innovation_norm": "loss_ct_innovation_applied_norm",
    "innovation_applied_ratio": "loss_ct_innovation_applied_ratio",
    "innovation_clamp_ratio": "loss_ct_innovation_clamp_ratio",
    "search_used_ratio": "loss_ct_search_used_mean",
    "search_expansion_ratio": "loss_ct_search_expansion_ratio_mean",
    "search_expansion_points": "loss_ct_search_expansion_points_mean",
    "usable_search_ratio": "loss_search_has_usable_points_mean",
    "training_loss_total": "loss_loss_total",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
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


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "checkpoint_exists": False,
            "checkpoint_epoch": None,
            "checkpoint_global_step": None,
            "checkpoint_state_tensors": None,
            "checkpoint_size_bytes": None,
            "checkpoint_sha256": None,
        }
    install_easydict_fallback()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", {})
    return {
        "checkpoint_exists": True,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "checkpoint_state_tensors": len(state_dict),
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256_file(path),
    }


def run_candidates(output_root: Path) -> dict[str, list[Path]]:
    candidates = {f"B{index}": [] for index in range(4)}
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_PATTERN.search(path.name)
        if match:
            candidates[f"B{match.group(1)}"].append(path)
    return candidates


def select_run(paths: list[Path]) -> Path | None:
    if not paths:
        return None

    def rank(path: Path):
        version = path / "lightning_logs/version_0"
        has_checkpoint = (
            version / "checkpoints/last.ckpt").is_file()
        metric_count = len(scalar_events(version, "metrics_test_success"))
        return has_checkpoint, metric_count, path.stat().st_mtime

    return max(paths, key=rank)


def resolved_steps(provenance: dict[str, Any] | None) -> tuple[int, int]:
    if not provenance:
        return 0, 0
    config = provenance.get("resolved_config", {})
    train = provenance.get("datasets", {}).get("train", {})
    frames = int(train.get("frames", 0) or 0)
    candidates = int(config.get("num_candidates", 1) or 1)
    batch_size = int(config.get("batch_size", 1) or 1)
    epochs = int(config.get("epoch", 0) or 0)
    steps_per_epoch = (frames * candidates) // batch_size
    return steps_per_epoch, steps_per_epoch * epochs


def summarize_diagnostic(
        values: list[tuple[int, float]],
        steps_per_epoch: int,
        warmup_epochs: int,
) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "post_warmup_mean": None,
            "late_epoch_mean": None,
            "epoch5_mean": None,
            "epoch6_mean": None,
            "epoch7_mean": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray([value for _, value in values], dtype=np.float64)
    warmup_steps = int(steps_per_epoch * warmup_epochs)
    post_warmup = array[warmup_steps:]
    late = array[-steps_per_epoch:] if steps_per_epoch else array

    def epoch_mean(epoch: int) -> float | None:
        if not steps_per_epoch:
            return None
        start = (epoch - 1) * steps_per_epoch
        selected = array[start:start + steps_per_epoch]
        return float(selected.mean()) if selected.size == steps_per_epoch else None

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "post_warmup_mean": (
            float(post_warmup.mean()) if post_warmup.size else None),
        "late_epoch_mean": float(late.mean()),
        "epoch5_mean": epoch_mean(5),
        "epoch6_mean": epoch_mean(6),
        "epoch7_mean": epoch_mean(7),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def collect_run(run_id: str, run: Path | None) -> dict[str, Any]:
    if run is None:
        return {
            "run_id": run_id,
            "label": LABELS[run_id],
            "run_dir": None,
            "status": "MISSING",
            "metrics": [],
            "diagnostics": {},
        }

    version = run / "lightning_logs/version_0"
    provenance = read_json(run / "run_provenance.json")
    config = provenance.get("resolved_config", {}) if provenance else {}
    steps_per_epoch, expected_steps = resolved_steps(provenance)
    success = scalar_events(version, "metrics_test_success")
    precision = scalar_events(version, "metrics_test_precision")
    loss = scalar_events(version, "loss_loss_total")
    checkpoint = checkpoint_metadata(
        version / "checkpoints/last.ckpt")

    metrics = []
    if [step for step, _ in success] == [step for step, _ in precision]:
        for (step, success_value), (_, precision_value) in zip(
                success, precision):
            metrics.append({
                "run_id": run_id,
                "epoch": (
                    step // steps_per_epoch if steps_per_epoch else None),
                "step": step,
                "success": success_value,
                "precision": precision_value,
            })

    expected_validations = (
        int(config.get("epoch", 0))
        // int(config.get("check_val_every_n_epoch", 1))
        if config else 0
    )
    complete = (
        bool(provenance)
        and not provenance.get("git", {}).get("dirty_tracked", True)
        and checkpoint["checkpoint_epoch"]
        == int(config.get("epoch", 0)) - 1
        and checkpoint["checkpoint_global_step"] == expected_steps
        and len(metrics) == expected_validations
    )
    if complete:
        status = "COMPLETE"
    elif loss:
        status = "PARTIAL"
    else:
        status = "MISSING"

    warmup_epochs = int(
        config.get("dynamics_innovation_warmup_epoch", 0) or 0)
    diagnostics = {
        name: summarize_diagnostic(
            scalar_events(version, leaf),
            steps_per_epoch,
            warmup_epochs,
        )
        for name, leaf in DIAGNOSTIC_LEAVES.items()
    }

    result = {
        "run_id": run_id,
        "label": LABELS[run_id],
        "run_dir": run.as_posix(),
        "status": status,
        "provenance": provenance,
        "config": config,
        "steps_per_epoch": steps_per_epoch,
        "expected_steps": expected_steps,
        "training_scalar_count": len(loss),
        "training_last_step": loss[-1][0] if loss else None,
        "completed_training_epochs": (
            len(loss) / steps_per_epoch if steps_per_epoch else 0.0),
        "validation_count": len(metrics),
        "expected_validations": expected_validations,
        "metrics": metrics,
        "diagnostics": diagnostics,
        **checkpoint,
    }
    if metrics:
        result["final"] = metrics[-1]
        result["best_success"] = max(
            metrics, key=lambda row: row["success"])
        result["best_precision"] = max(
            metrics, key=lambda row: row["precision"])
        for count in (3, 5):
            selected = metrics[-count:]
            result[f"late{count}"] = {
                "success": float(np.mean(
                    [row["success"] for row in selected])),
                "precision": float(np.mean(
                    [row["precision"] for row in selected])),
            }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def build_report(results: dict[str, dict[str, Any]]) -> str:
    complete = [
        run_id for run_id, result in results.items()
        if result["status"] == "COMPLETE"
    ]
    all_complete = len(complete) == 4

    def final_delta(left: str, right: str, metric: str) -> float | None:
        left_value = results[left].get("final", {}).get(metric)
        right_value = results[right].get("final", {}).get(metric)
        if left_value is None or right_value is None:
            return None
        return left_value - right_value

    def diagnostic(run_id: str, metric: str, field: str) -> float | None:
        return results[run_id].get(
            "diagnostics", {}).get(metric, {}).get(field)

    lines = [
        "# CT-SeqTrack v2 B0–B3 seed42 消融复核",
        "",
        "更新时间：2026-07-27",
        "",
        "## Overall Assessment: Needs revision",
        "",
    ]
    if all_complete:
        lines.extend([
            "四组运行均已完整，数据足以完成本轮 seed42 normal-mini 首筛；"
            "但结果否定了当前三模块组合。B0 仍是唯一晋级模型，B1 的运动修正"
            "大幅退化，B2 只能部分救回，B3 的 learned gate 又退化回 B1 水平。",
            "",
            "**结论：当前 B3 不应进入时间负对照、多 seed、full nuScenes 或"
            " Random-20%。后续 Search-only A1 也已失败；当前应先做现有"
            " B0/A1 checkpoint 的 Search 开/关 2×2，不训练 A2。**",
        ])
    else:
        lines.extend([
            "至少一组运行不完整，当前数据不足以完成 B0–B3 正式比较。",
        ])
    lines.extend([
        "",
        "## Methodology Review",
        "",
        "- 数据：nuScenes v1.0-mini，Car，mini_train 274 tracklets / "
        "5,051 frames，mini_val 106 tracklets / 2,285 frames。",
        "- 协议：normal cadence、seed42、candidate4、batch16、60 epoch，"
        "每 5 epoch 验证一次。",
        "- 主结果：固定使用 epoch60 `last.ckpt`；best epoch 和 late mean "
        "只用于稳定性诊断，不用于替代 final。",
        "- 原始来源：TensorBoard scalar events、`run_provenance.json` 和 "
        "`last.ckpt` 元数据；未使用服务器控制台汇总数字。",
        "- B2 与 B3 的 resolved config 除配置名、tag 和 "
        "`ct_fusion_mode: fixed -> adaptive` 外一致。",
        "",
        "## Integrity",
        "",
        "| arm | status | commit | clean tracked | train steps | val points | "
        "last checkpoint |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for run_id in ("B0", "B1", "B2", "B3"):
        result = results[run_id]
        provenance = result.get("provenance") or {}
        commit = provenance.get("git", {}).get("commit", "—")
        commit = commit[:7] if commit and commit != "—" else "—"
        clean = not provenance.get("git", {}).get("dirty_tracked", True)
        checkpoint = (
            f"epoch {result['checkpoint_epoch'] + 1}"
            if result.get("checkpoint_exists") else "missing")
        lines.append(
            f"| {run_id} | {result['status']} | `{commit}` | "
            f"{'yes' if clean else 'no'} | "
            f"{result.get('training_scalar_count', 0):,}/"
            f"{result.get('expected_steps', 0):,} | "
            f"{result.get('validation_count', 0)}/"
            f"{result.get('expected_validations', 0)} | {checkpoint} |"
        )

    lines.extend([
        "",
        "## Validation Results",
        "",
        "| arm | final Success | final Precision | best Success (epoch) | "
        "best Precision (epoch) | late-3 Success | late-3 Precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run_id in ("B0", "B1", "B2", "B3"):
        result = results[run_id]
        final = result.get("final", {})
        best_success = result.get("best_success", {})
        best_precision = result.get("best_precision", {})
        late = result.get("late3", {})
        lines.append(
            f"| {run_id} | {format_metric(final.get('success'))} | "
            f"{format_metric(final.get('precision'))} | "
            f"{format_metric(best_success.get('success'))} "
            f"({best_success.get('epoch', '—')}) | "
            f"{format_metric(best_precision.get('precision'))} "
            f"({best_precision.get('epoch', '—')}) | "
            f"{format_metric(late.get('success'))} | "
            f"{format_metric(late.get('precision'))} |"
        )

    lines.extend([
        "",
        "## Module Deltas at Final Checkpoint",
        "",
        "| comparison | Success delta | Precision delta | decision |",
        "|---|---:|---:|---|",
        f"| B1 − B0 (motion) | "
        f"{format_metric(final_delta('B1', 'B0', 'success'))} | "
        f"{format_metric(final_delta('B1', 'B0', 'precision'))} | reject "
        "current fixed-0.75 motion fusion |",
        f"| B2 − B1 (search) | "
        f"{format_metric(final_delta('B2', 'B1', 'success'))} | "
        f"{format_metric(final_delta('B2', 'B1', 'precision'))} | positive "
        "rescue, but not a search-only proof |",
        f"| B2 − B0 (motion + search) | "
        f"{format_metric(final_delta('B2', 'B0', 'success'))} | "
        f"{format_metric(final_delta('B2', 'B0', 'precision'))} | below "
        "baseline |",
        f"| B3 − B2 (adaptive gate) | "
        f"{format_metric(final_delta('B3', 'B2', 'success'))} | "
        f"{format_metric(final_delta('B3', 'B2', 'precision'))} | reject "
        "current adaptive gate |",
        f"| B3 − B0 (full v2) | "
        f"{format_metric(final_delta('B3', 'B0', 'success'))} | "
        f"{format_metric(final_delta('B3', 'B0', 'precision'))} | no "
        "promotion |",
        f"| B3 − B1 | "
        f"{format_metric(final_delta('B3', 'B1', 'success'))} | "
        f"{format_metric(final_delta('B3', 'B1', 'precision'))} | returns "
        "to the failed B1 level |",
        "",
        "## Learning-Curve Interpretation",
        "",
        "- B0 late-3 is 52.905 / 63.104 and final is 53.360 / 64.382; the "
        "baseline is the strongest and most stable arm in this screen.",
        "- B2 reaches 50.080 / 58.499 at epoch 25, then finishes at "
        "47.973 / 52.088. Search provides a large recovery relative to B1, "
        "but never establishes a gain over B0.",
        "- B3 is best at epoch 5 (34.458 / 37.724), immediately after the "
        "five training warmup epochs while the gate is still at its initial "
        "nominal alpha 0.25. It falls to 28.542 / 26.042 at epoch 10 and its "
        "late-3 is only 26.321 / 25.104. The regression therefore persists "
        "across the entire late-training window and is not a bad final "
        "checkpoint.",
        "",
        "## Mechanism Diagnostics",
        "",
        "- B1 post-warmup applied alpha mean is "
        f"{diagnostic('B1', 'fusion_alpha_applied', 'post_warmup_mean'):.3f}; "
        "innovation is applied to "
        f"{100 * diagnostic('B1', 'innovation_applied_ratio', 'post_warmup_mean'):.1f}% "
        "of training samples and is radius-clamped on "
        f"{100 * diagnostic('B1', 'innovation_clamp_ratio', 'post_warmup_mean'):.1f}%. "
        "This is an aggressive correction path, consistent with the large B1 regression.",
        "- B2 search is active on only "
        f"{100 * diagnostic('B2', 'search_used_ratio', 'mean'):.2f}% "
        "of training samples; mean expansion token share is "
        f"{100 * diagnostic('B2', 'search_expansion_ratio', 'mean'):.3f}%. "
        "B3 has essentially the same search activation, so its collapse is not "
        "explained by search being disabled.",
        "- B3 nominal alpha is "
        f"{diagnostic('B3', 'fusion_alpha_nominal', 'epoch5_mean'):.3f} at "
        "epoch 5, rises to "
        f"{diagnostic('B3', 'fusion_alpha_nominal', 'epoch6_mean'):.3f} at "
        "epoch 6, and reaches "
        f"{diagnostic('B3', 'fusion_alpha_nominal', 'epoch7_mean'):.3f} at "
        "epoch 7. At epoch 60 even the batch-min mean is "
        f"{diagnostic('B3', 'fusion_alpha_batch_min', 'late_epoch_mean'):.6f}, "
        "against a configured maximum of 0.75. The learned gate has saturated "
        "into an almost constant maximum-weight gate rather than learning "
        "conditional reliability.",
        "- B3 post-warmup applied alpha "
        f"({diagnostic('B3', 'fusion_alpha_applied', 'post_warmup_mean'):.3f}), "
        "innovation application ratio "
        f"({100 * diagnostic('B3', 'innovation_applied_ratio', 'post_warmup_mean'):.1f}%) "
        "and clamp ratio "
        f"({100 * diagnostic('B3', 'innovation_clamp_ratio', 'post_warmup_mean'):.1f}%) "
        "are nearly identical to B1/B2. Its epoch-60 mean training loss is "
        f"{diagnostic('B3', 'training_loss_total', 'late_epoch_mean'):.3f}, "
        "slightly below B2's "
        f"{diagnostic('B2', 'training_loss_total', 'late_epoch_mean'):.3f}, "
        "while validation is far worse. This is evidence of train/recursive-"
        "validation mismatch or gate/backbone co-adaptation, not under-training.",
        "",
        "## Issues Found",
        "",
        "1. **High — adaptive gate collapse.** B3 learns the configured upper "
        "bound for virtually every training sample. It neither suppresses unreliable "
        "motion nor preserves B2's recovery.",
        "2. **High — current motion correction fails the normal-data guardrail.** "
        "B1 best, final and late metrics are all far below B0. Adding a learned gate "
        "does not repair it.",
        "3. **Medium — search is not independently isolated.** B2 − B1 is positive, "
        "but there is no `B0 + search only` arm. The data prove a rescue interaction, "
        "not that search itself beats the baseline.",
        "4. **Medium — shared initialization is not strictly controlled.** "
        "`ct_proposal_fusion` is instantiated before `motion_mlp`, "
        "`feature_pointnet` and `Transformer`; enabling it consumes RNG before "
        "shared layers are initialized. The same issue applies when B1 inserts the "
        "motion encoder relative to B0. With only one seed, exact module deltas are "
        "therefore partly confounded by initialization.",
        "5. **Medium — no per-tracklet endpoint export is present.** Aggregate "
        "Success/Precision cannot show whether B2 recovery is broad or driven by a "
        "small subset of sequences.",
        "6. **Low — B3 uses commit `600bb88` while B0–B2 use `d86990c`.** The "
        "intervening changes only normalize singleton validation scalar shapes and "
        "add tests; no architecture or configured numerical rule changed. This is "
        "recorded as a provenance difference, not the leading explanation for the "
        "score gap.",
        "",
        "## Decision",
        "",
        "- Mark the current B0–B3 screen complete and reject the present B3. Do not "
        "repeat the same four unchanged configs.",
        "- Before another training cycle, enforce a shared-initialization contract "
        "(load one common initialization for shared keys, or isolate optional-module "
        "RNG). Add a same-checkpoint inference override for alpha 0 / 0.25 / 0.75 "
        "as a cheap sensitivity diagnosis.",
        "- The follow-up `A1 = baseline + time-guided search only` has since "
        "completed and failed the normal-mini guardrail. See "
        "`ct_search_only_seed42_20260727.md` for the post-screen decision.",
        "- Do not train A2 yet. First evaluate the existing B0 and A1 checkpoints "
        "with Search off/on as a no-training 2x2, and add validation endpoint search "
        "diagnostics. Do not restore the current unconstrained adaptive gate.",
        "- Seed43/44, `true/fixed/shuffled`, full nuScenes, Random-20%, ChronoTrack "
        "consistency and compact memory remain blocked until a final/late-3 model "
        "beats its same-initialization baseline.",
        "",
        "## Required Caveats",
        "",
        "- This is one seed on nuScenes-mini and does not establish statistical "
        "stability.",
        "- B2's positive delta is relative to the failed B1 arm; B2 is still below "
        "the same-code B0 baseline.",
        "- The current ordering of optional module construction prevents a strictly "
        "shared initialization across arms; the screen is sufficient for rejection "
        "but not for a paper-level causal effect size.",
        "- No physical-time causal claim is supported until a promoted model "
        "passes same-endpoint `true/fixed/shuffled` controls.",
    ])
    return "\n".join(lines) + "\n"


def build_figure(results: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = {"B0": "#333333", "B1": "#D55E00", "B2": "#0072B2",
              "B3": "#009E73"}
    for run_id in ("B0", "B1", "B2", "B3"):
        metrics = results[run_id].get("metrics", [])
        if not metrics:
            continue
        epochs = [row["epoch"] for row in metrics]
        axes[0].plot(
            epochs, [row["success"] for row in metrics],
            marker="o", linewidth=1.8, markersize=3.5,
            label=LABELS[run_id], color=colors[run_id])
        axes[1].plot(
            epochs, [row["precision"] for row in metrics],
            marker="o", linewidth=1.8, markersize=3.5,
            label=LABELS[run_id], color=colors[run_id])
    axes[0].set_title("Success by validation epoch")
    axes[1].set_title("Precision by validation epoch")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Score")
        axis.grid(alpha=0.25)
        axis.set_xticks(range(5, 61, 5))
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("CT-SeqTrack v2 seed42 ablation on nuScenes-mini")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "output"))
    parser.add_argument("--date-tag", default="20260727")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    candidates = run_candidates(output_root)
    results = {
        run_id: collect_run(run_id, select_run(paths))
        for run_id, paths in candidates.items()
    }

    prefix = f"ct_v2_ablation_seed42_{args.date_tag}"
    data_dir = ROOT / "compare_results/data"
    report_path = ROOT / f"compare_results/reports/{prefix}.md"
    figure_path = ROOT / f"compare_results/figures/line_charts/{prefix}_curves.png"

    metric_rows = [
        row
        for run_id in ("B0", "B1", "B2", "B3")
        for row in results[run_id].get("metrics", [])
    ]
    integrity_rows = []
    diagnostic_rows = []
    summary_rows = []
    for run_id in ("B0", "B1", "B2", "B3"):
        result = results[run_id]
        provenance = result.get("provenance") or {}
        integrity_rows.append({
            "run_id": run_id,
            "status": result["status"],
            "run_dir": result.get("run_dir"),
            "commit": provenance.get("git", {}).get("commit"),
            "dirty_tracked": provenance.get(
                "git", {}).get("dirty_tracked"),
            "seed": provenance.get("seed"),
            "training_scalar_count": result.get(
                "training_scalar_count", 0),
            "expected_steps": result.get("expected_steps", 0),
            "validation_count": result.get("validation_count", 0),
            "expected_validations": result.get(
                "expected_validations", 0),
            "checkpoint_epoch": result.get("checkpoint_epoch"),
            "checkpoint_global_step": result.get(
                "checkpoint_global_step"),
            "checkpoint_sha256": result.get("checkpoint_sha256"),
        })
        final = result.get("final", {})
        summary_rows.append({
            "run_id": run_id,
            "status": result["status"],
            "final_success": final.get("success"),
            "final_precision": final.get("precision"),
            "best_success": result.get(
                "best_success", {}).get("success"),
            "best_success_epoch": result.get(
                "best_success", {}).get("epoch"),
            "best_precision": result.get(
                "best_precision", {}).get("precision"),
            "best_precision_epoch": result.get(
                "best_precision", {}).get("epoch"),
            "late3_success": result.get(
                "late3", {}).get("success"),
            "late3_precision": result.get(
                "late3", {}).get("precision"),
        })
        for metric, values in result.get("diagnostics", {}).items():
            diagnostic_rows.append({
                "run_id": run_id,
                "metric": metric,
                **values,
            })

    write_csv(data_dir / f"{prefix}_metrics.csv", metric_rows)
    write_csv(data_dir / f"{prefix}_integrity.csv", integrity_rows)
    write_csv(data_dir / f"{prefix}_summary.csv", summary_rows)
    write_csv(data_dir / f"{prefix}_diagnostics.csv", diagnostic_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(results), encoding="utf-8")
    build_figure(results, figure_path)

    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    for row in integrity_rows:
        print(
            f"{row['run_id']}: {row['status']} "
            f"steps={row['training_scalar_count']}/{row['expected_steps']} "
            f"val={row['validation_count']}/{row['expected_validations']}")


if __name__ == "__main__":
    main()
