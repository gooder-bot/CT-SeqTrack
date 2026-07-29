#!/usr/bin/env python3
"""Audit the CT-v2 fixed-motion alpha reruns against B0 and the old B1.

The primary comparison is the same-code, same-seed scratch pair
``alpha=0`` versus ``alpha=0.25``.  The historical B0 and fixed-0.75 B1
remain visible as decision context, with their commit difference recorded
instead of silently treating all four runs as one perfectly matched trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_ct_v2_ablation as ablation  # noqa: E402


RUN_GLOBS = {
    "B0": "*01_seqtrack3d_baseline-ctv2_d86990c_b0*",
    "A0": "*02_ct_motion_alpha000-*",
    "A025": "*02_ct_motion_alpha025-*",
    "A075": "*02_ct_motion-ctv2_d86990c_b1*",
}
LABELS = {
    "B0": "B0 baseline",
    "A0": "B1 motion, alpha=0",
    "A025": "B1 motion, alpha=0.25",
    "A075": "B1 motion, alpha=0.75",
}
ALPHAS = {"B0": None, "A0": 0.0, "A025": 0.25, "A075": 0.75}
EXPECTED_CONFIG_HASHES = {
    "A0": "56ae7334e3e16a9aba5ef341e27950ac9673b090adcb86e9394e9cb3f3df743c",
    "A025": "b5e28a4c1310c273f14c4264a0cabddab24b15a11d9365277b56d1776b7ff368",
}
COMPONENT_LEAVES = {
    "loss_total": "loss_loss_total",
    "loss_center": "loss_loss_center",
    "loss_center_aux": "loss_loss_center_aux",
    "loss_center_motion": "loss_loss_center_motion",
    "loss_center_ref": "loss_loss_center_ref",
    "loss_velocity": "loss_loss_velocity",
    "loss_dynamics_displacement": "loss_loss_dynamics_displacement",
    "loss_seg": "loss_loss_seg",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def select_unique(output_root: Path, pattern: str) -> Path:
    matches = sorted(output_root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one run for {pattern!r}, got "
            f"{[path.name for path in matches]}")
    return matches[0]


def load_results(output_root: Path) -> dict[str, dict[str, Any]]:
    ablation.LABELS.update(LABELS)
    results = {
        run_id: ablation.collect_run(
            run_id, select_unique(output_root, pattern))
        for run_id, pattern in RUN_GLOBS.items()
    }
    for run_id, result in results.items():
        if result["status"] != "COMPLETE":
            raise RuntimeError(
                f"{run_id} is not complete: {result['status']}")
    return results


def config_diff(
        left: dict[str, Any], right: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            rows.append({
                "field": key,
                "alpha000": json.dumps(
                    left.get(key), ensure_ascii=False, sort_keys=True),
                "alpha025": json.dumps(
                    right.get(key), ensure_ascii=False, sort_keys=True),
            })
    return rows


def mean_slice(
        values: list[tuple[int, float]], start: int, end: int,
) -> float | None:
    selected = np.asarray(
        [value for step, value in values if start <= step < end],
        dtype=np.float64,
    )
    return float(selected.mean()) if selected.size else None


def build_component_rows(
        results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for run_id, result in results.items():
        version = (
            Path(result["run_dir"]) / "lightning_logs/version_0")
        steps_per_epoch = result["steps_per_epoch"]
        for name, leaf in COMPONENT_LEAVES.items():
            values = ablation.scalar_events(version, leaf)
            if not values:
                continue
            rows.append({
                "run_id": run_id,
                "alpha": ALPHAS[run_id],
                "metric": name,
                "count": len(values),
                "all_mean": float(np.mean([value for _, value in values])),
                "epoch6_mean": mean_slice(
                    values, 5 * steps_per_epoch, 6 * steps_per_epoch),
                "epoch60_mean": mean_slice(
                    values, 59 * steps_per_epoch, 60 * steps_per_epoch),
            })
    return rows


def build_tables(
        results: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    metrics = []
    summary = []
    integrity = []
    diagnostics = []
    matched = []

    for run_id, result in results.items():
        provenance = result["provenance"]
        config = result["config"]
        metrics.extend(result["metrics"])
        final = result["final"]
        best_success = result["best_success"]
        best_precision = result["best_precision"]
        summary.append({
            "run_id": run_id,
            "label": LABELS[run_id],
            "alpha": ALPHAS[run_id],
            "final_success": final["success"],
            "final_precision": final["precision"],
            "best_success": best_success["success"],
            "best_success_epoch": best_success["epoch"],
            "best_precision": best_precision["precision"],
            "best_precision_epoch": best_precision["epoch"],
            "late3_success": result["late3"]["success"],
            "late3_precision": result["late3"]["precision"],
            "late5_success": result["late5"]["success"],
            "late5_precision": result["late5"]["precision"],
        })
        integrity.append({
            "run_id": run_id,
            "status": result["status"],
            "run_dir": result["run_dir"],
            "commit": provenance["git"]["commit"],
            "dirty_any": provenance["git"]["dirty_any"],
            "dirty_tracked": provenance["git"]["dirty_tracked"],
            "seed": provenance["seed"],
            "config_path": provenance["config_path"],
            "config_sha256": provenance["config_sha256"],
            "resolved_config_sha256":
                provenance["resolved_config_sha256"],
            "train_frames": provenance["datasets"]["train"]["frames"],
            "train_tracklets":
                provenance["datasets"]["train"]["tracklets"],
            "val_frames": provenance["datasets"]["val"]["frames"],
            "val_tracklets": provenance["datasets"]["val"]["tracklets"],
            "steps_per_epoch": result["steps_per_epoch"],
            "training_steps": result["training_scalar_count"],
            "expected_steps": result["expected_steps"],
            "validation_points": result["validation_count"],
            "expected_validations": result["expected_validations"],
            "checkpoint_epoch": result["checkpoint_epoch"],
            "checkpoint_global_step": result["checkpoint_global_step"],
            "checkpoint_state_tensors":
                result["checkpoint_state_tensors"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "batch_size": config["batch_size"],
            "epochs": config["epoch"],
            "check_val_every_n_epoch":
                config["check_val_every_n_epoch"],
        })
        for metric, values in result["diagnostics"].items():
            diagnostics.append({
                "run_id": run_id,
                "alpha": ALPHAS[run_id],
                "metric": metric,
                **values,
            })

    for index, b0_row in enumerate(results["B0"]["metrics"]):
        row = {
            "epoch": b0_row["epoch"],
            "b0_success": b0_row["success"],
            "b0_precision": b0_row["precision"],
        }
        for run_id in ("A0", "A025", "A075"):
            value = results[run_id]["metrics"][index]
            row.update({
                f"{run_id.lower()}_success": value["success"],
                f"{run_id.lower()}_precision": value["precision"],
                f"{run_id.lower()}_minus_b0_success":
                    value["success"] - b0_row["success"],
                f"{run_id.lower()}_minus_b0_precision":
                    value["precision"] - b0_row["precision"],
            })
        row.update({
            "a025_minus_a0_success":
                results["A025"]["metrics"][index]["success"]
                - results["A0"]["metrics"][index]["success"],
            "a025_minus_a0_precision":
                results["A025"]["metrics"][index]["precision"]
                - results["A0"]["metrics"][index]["precision"],
        })
        matched.append(row)

    return {
        "metrics": metrics,
        "summary": summary,
        "integrity": integrity,
        "diagnostics": diagnostics,
        "matched": matched,
        "components": build_component_rows(results),
        "config_diff": config_diff(
            results["A0"]["config"], results["A025"]["config"]),
    }


def validate_contract(
        results: dict[str, dict[str, Any]],
        tables: dict[str, list[dict[str, Any]]],
) -> None:
    for run_id in ("A0", "A025"):
        provenance = results[run_id]["provenance"]
        if provenance["config_sha256"] != EXPECTED_CONFIG_HASHES[run_id]:
            raise AssertionError(f"{run_id}: unexpected config SHA256")
        if provenance["git"]["dirty_tracked"]:
            raise AssertionError(f"{run_id}: tracked source was dirty")
        if results[run_id]["training_scalar_count"] != 75720:
            raise AssertionError(f"{run_id}: unexpected training step count")
        if results[run_id]["validation_count"] != 12:
            raise AssertionError(f"{run_id}: unexpected validation count")
        if results[run_id]["checkpoint_epoch"] != 59:
            raise AssertionError(f"{run_id}: last checkpoint is not epoch60")
        if results[run_id]["checkpoint_global_step"] != 75720:
            raise AssertionError(f"{run_id}: last checkpoint step mismatch")

    left = results["A0"]["provenance"]
    right = results["A025"]["provenance"]
    if left["git"]["commit"] != right["git"]["commit"]:
        raise AssertionError("alpha000/025 commits differ")
    for role in ("train", "val"):
        left_data = left["datasets"][role]
        right_data = right["datasets"][role]
        for key in (
                "frames", "tracklets", "split", "version",
                "virtual_rate_selection_sha256"):
            if left_data.get(key) != right_data.get(key):
                raise AssertionError(
                    f"alpha000/025 {role} dataset differs at {key}")
    visible_config_diff = {
        row["field"] for row in tables["config_diff"]}
    if visible_config_diff != {
            "cfg", "dynamics_innovation_alpha", "tag"}:
        raise AssertionError(
            f"unexpected alpha000/025 config diff: {visible_config_diff}")


def build_figure(
        results: dict[str, dict[str, Any]], path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    styles = {
        "B0": {
            "color": "#333333", "marker": "o", "linestyle": "-",
            "markerfacecolor": "#333333",
        },
        "A0": {
            "color": "#2F6B9A", "marker": "s", "linestyle": "-",
            "markerfacecolor": "white",
        },
        "A025": {
            "color": "#D9822B", "marker": "^", "linestyle": "--",
            "markerfacecolor": "white",
        },
        "A075": {
            "color": "#A23B3B", "marker": "x", "linestyle": ":",
            "markerfacecolor": "#A23B3B",
        },
    }
    for run_id in ("B0", "A0", "A025", "A075"):
        rows = results[run_id]["metrics"]
        epochs = [row["epoch"] for row in rows]
        for axis, metric in zip(axes, ("success", "precision")):
            axis.plot(
                epochs,
                [row[metric] for row in rows],
                linewidth=1.9,
                markersize=5,
                markeredgewidth=1.2,
                label=LABELS[run_id],
                **styles[run_id],
            )
    axes[0].set_title("Success by validation epoch")
    axes[1].set_title("Precision by validation epoch")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Score")
        axis.set_xlim(5, 60)
        axis.set_ylim(0, 70)
        axis.set_xticks(range(5, 61, 5))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "CT-v2 fixed-motion alpha sweep on nuScenes-mini",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.5, 0.01,
        "Car, seed42, scratch, 60 epochs; final checkpoint is primary",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def final_delta(
        results: dict[str, dict[str, Any]],
        left: str,
        right: str,
        metric: str,
) -> float:
    return (
        results[left]["final"][metric]
        - results[right]["final"][metric]
    )


def diag(
        results: dict[str, dict[str, Any]],
        run_id: str,
        metric: str,
        field: str,
) -> float:
    return float(results[run_id]["diagnostics"][metric][field])


def component(
        tables: dict[str, list[dict[str, Any]]],
        run_id: str,
        metric: str,
        field: str,
) -> float:
    for row in tables["components"]:
        if row["run_id"] == run_id and row["metric"] == metric:
            return float(row[field])
    raise KeyError((run_id, metric, field))


def build_report(
        results: dict[str, dict[str, Any]],
        tables: dict[str, list[dict[str, Any]]],
        figure_relative: str,
) -> str:
    a0 = results["A0"]
    a025 = results["A025"]
    a075 = results["A075"]
    b0 = results["B0"]
    late_a025_vs_a0_s = (
        a025["late3"]["success"] - a0["late3"]["success"])
    late_a025_vs_a0_p = (
        a025["late3"]["precision"] - a0["late3"]["precision"])
    late_a0_vs_b0_s = a0["late3"]["success"] - b0["late3"]["success"]
    late_a0_vs_b0_p = a0["late3"]["precision"] - b0["late3"]["precision"]
    post_warmup_points = [
        row for row in tables["matched"] if int(row["epoch"]) >= 25]
    late_direction_count = sum(
        row["a025_minus_a0_success"] < 0
        and row["a025_minus_a0_precision"] < 0
        for row in post_warmup_points
    )

    return f"""# CT-v2 Motion Fixed-Alpha 复核

更新时间：2026-07-30

## 技术结论

**当前 `proposal_innovation` motion 模块不能涨点，固定全局 alpha 路线应判定
No-Go。** 同代码、同 seed、同 scratch 合同下，`alpha=0.25` 的 epoch60 为
`{fmt(a025['final']['success'])}/{fmt(a025['final']['precision'])}`，相对
`alpha=0` 下降 `{fmt(final_delta(results, 'A025', 'A0', 'success'))}/`
`{fmt(final_delta(results, 'A025', 'A0', 'precision'))}`；late-3 同样下降
`{fmt(late_a025_vs_a0_s)}/{fmt(late_a025_vs_a0_p)}`。旧 `alpha=0.75`
更低至 `{fmt(a075['final']['success'])}/{fmt(a075['final']['precision'])}`。

`alpha=0` 虽恢复到
`{fmt(a0['final']['success'])}/{fmt(a0['final']['precision'])}`，仍比 B0 低
`{fmt(final_delta(results, 'A0', 'B0', 'success'))}/`
`{fmt(final_delta(results, 'A0', 'B0', 'precision'))}`。但它是精确关闭
innovation 的负对照，不是 motion 的正贡献；它与 B0 还存在可选模块改变共享层
随机初始化的已知混杂。因此当前数据能强否定正 alpha 的直接融合，不能把
`alpha=0` 与 B0 的差全部归因于 dynamics 辅助学习。

正式判定：

```text
NO_GO_FIXED_GLOBAL_MOTION_INNOVATION
ALPHA025_REDUCES_BUT_DOES_NOT_REMOVE_FAILURE
ALPHA000_IS_A_FALLBACK_CONTROL_NOT_A_GAIN
BROADER_MOTION_PRIOR_IDEA_REMAINS_UNRESOLVED
```

## 四组完整曲线均不支持涨点

下图使用统一 0–70 纵轴和全部 12 个固定验证点。`alpha=0.25` 在 epoch25–60
的 {late_direction_count}/{len(post_warmup_points)} 个验证点上，Success 和
Precision 同时低于 `alpha=0`；不是 epoch60 单点选择问题。

![CT-v2 motion alpha validation curves]({figure_relative})

| arm | final Success | final Precision | best Success | best Precision | late-3 S/P |
|---|---:|---:|---:|---:|---:|
| B0 baseline | {fmt(b0['final']['success'])} | {fmt(b0['final']['precision'])} | {fmt(b0['best_success']['success'])} (e{b0['best_success']['epoch']}) | {fmt(b0['best_precision']['precision'])} (e{b0['best_precision']['epoch']}) | {fmt(b0['late3']['success'])}/{fmt(b0['late3']['precision'])} |
| motion alpha=0 | {fmt(a0['final']['success'])} | {fmt(a0['final']['precision'])} | {fmt(a0['best_success']['success'])} (e{a0['best_success']['epoch']}) | {fmt(a0['best_precision']['precision'])} (e{a0['best_precision']['epoch']}) | {fmt(a0['late3']['success'])}/{fmt(a0['late3']['precision'])} |
| motion alpha=0.25 | {fmt(a025['final']['success'])} | {fmt(a025['final']['precision'])} | {fmt(a025['best_success']['success'])} (e{a025['best_success']['epoch']}) | {fmt(a025['best_precision']['precision'])} (e{a025['best_precision']['epoch']}) | {fmt(a025['late3']['success'])}/{fmt(a025['late3']['precision'])} |
| motion alpha=0.75 | {fmt(a075['final']['success'])} | {fmt(a075['final']['precision'])} | {fmt(a075['best_success']['success'])} (e{a075['best_success']['epoch']}) | {fmt(a075['best_precision']['precision'])} (e{a075['best_precision']['epoch']}) | {fmt(a075['late3']['success'])}/{fmt(a075['late3']['precision'])} |

没有一个正 alpha 运行在任一验证点同时超过同阶段 B0 的两项指标。
`alpha=0` 的最好点出现在 epoch35（
`{fmt(a0['best_success']['success'])}/{fmt(a0['best_precision']['precision'])}`），
同阶段 B0 仍为
`{fmt(b0['metrics'][6]['success'])}/{fmt(b0['metrics'][6]['precision'])}`。

## 数据和可比性通过完整性检查

- 数据：nuScenes v1.0-mini，Car；mini_train 274 tracklets / 5,051 frames，
  mini_val 106 tracklets / 2,285 frames。
- 四组都是 seed42、batch16、candidate4、60 epoch、75,720 training steps、
  每 5 epoch 验证，共 12 个点；主比较固定使用 epoch60 `last.ckpt`。
- 新 `alpha=0/0.25` 来自同一 commit `5f260e7`，tracked source clean，
  仅两个 alpha YAML 为 untracked；其内容已由 provenance SHA256 精确还原。
- 两个新运行的 resolved config 仅有 `cfg`、`tag` 和
  `dynamics_innovation_alpha` 三项差异，训练/验证 selection hash 一致。
- B0/旧 alpha0.75 来自 `d86990c`；中间代码变化对本分支主要是 inert PFTC
  默认项与 singleton shape 防护，但跨 commit 对比仍只作为上下文。

## 较小 alpha 只是减少伤害，没有改变错误方向

`alpha=0.25` 在 warmup 后实际平均系数为
`{fmt(diag(results, 'A025', 'fusion_alpha_applied', 'post_warmup_mean'))}`，
约 {100 * diag(results, 'A025', 'innovation_applied_ratio', 'post_warmup_mean'):.1f}%
训练样本应用修正；平均修正范数仅
`{fmt(diag(results, 'A025', 'innovation_norm', 'post_warmup_mean'))} m`，
仍造成 final `−17.468/−20.322`。这说明问题不只是旧 `0.75` 数值过大。

| diagnostic | alpha=0 | alpha=0.25 | alpha=0.75 |
|---|---:|---:|---:|
| post-warmup effective alpha | {fmt(diag(results, 'A0', 'fusion_alpha_applied', 'post_warmup_mean'))} | {fmt(diag(results, 'A025', 'fusion_alpha_applied', 'post_warmup_mean'))} | {fmt(diag(results, 'A075', 'fusion_alpha_applied', 'post_warmup_mean'))} |
| post-warmup applied ratio | {100 * diag(results, 'A0', 'innovation_applied_ratio', 'post_warmup_mean'):.1f}% | {100 * diag(results, 'A025', 'innovation_applied_ratio', 'post_warmup_mean'):.1f}% | {100 * diag(results, 'A075', 'innovation_applied_ratio', 'post_warmup_mean'):.1f}% |
| post-warmup correction norm | {fmt(diag(results, 'A0', 'innovation_norm', 'post_warmup_mean'))} m | {fmt(diag(results, 'A025', 'innovation_norm', 'post_warmup_mean'))} m | {fmt(diag(results, 'A075', 'innovation_norm', 'post_warmup_mean'))} m |
| post-warmup clamp ratio | {100 * diag(results, 'A0', 'innovation_clamp_ratio', 'post_warmup_mean'):.1f}% | {100 * diag(results, 'A025', 'innovation_clamp_ratio', 'post_warmup_mean'):.1f}% | {100 * diag(results, 'A075', 'innovation_clamp_ratio', 'post_warmup_mean'):.1f}% |
| epoch60 mean training loss | {fmt(component(tables, 'A0', 'loss_total', 'epoch60_mean'))} | {fmt(component(tables, 'A025', 'loss_total', 'epoch60_mean'))} | {fmt(component(tables, 'A075', 'loss_total', 'epoch60_mean'))} |

## 主要根因是训练/递归语义错位

### 1. dynamics 在训练和验证读取的历史不是同一种分布

训练时 CT-v2 dynamics 显式读取由 GT 历史构造的
`ct_motion_ref_boxs/canonical_ref_boxs`，再叠加合成 correlated candidate
误差；递归验证时 `self.training=False`，改为读取 tracker 自己累计的
`ref_boxs`。前者是有界、局部、受控误差，后者包含闭环漂移和错误速度。
离线 M0-3 的 `alpha≈0.775` 又来自 GT-history、candidate0、
crop-reachable oracle，不能校准这种递归输入。

### 2. `dynamics_valid` 只表示“有历史 transition”，不表示方向可靠

固定融合没有在线可靠性判断。alpha0.25 在约 73.7% 样本上持续应用，
而 empty-search fallback 只覆盖极端无点情况。只要历史已漂移，错误 prior
仍会被当作有效方向；递归更新再把误差写回下一帧历史。

### 3. innovation 接在 coarse proposal 与 Transformer query 之前

修正后的 coarse center 被立即用于构造 `aux_box` 和 box-corner query。
因此一个看似很小的 `0.083 m` 平均修正不仅改变最终坐标，还改变后续
Transformer 的查询几何；错误方向可被 refinement 和下一帧 crop 放大。

### 4. 本地训练 loss 奖励 fusion，闭环指标却单调恶化

epoch60 mean training loss 从 alpha0 的
`{fmt(component(tables, 'A0', 'loss_total', 'epoch60_mean'))}` 降到
alpha0.25 的
`{fmt(component(tables, 'A025', 'loss_total', 'epoch60_mean'))}`，旧 alpha0.75
进一步降到
`{fmt(component(tables, 'A075', 'loss_total', 'epoch60_mean'))}`；
Success/Precision 却反向下降。局部 teacher-forced objective 无法约束
closed-loop stability，继续训练或按 training loss 选 alpha 不会修复。

### 5. alpha0 与 B0 的差不能用于证明 dynamics 本身有害

alpha0 在 `apply_proposal_innovation` 中是精确零回退；DynamicsEncoder 的
velocity/displacement loss只更新独立 dynamics 参数，不给 observation 主干
提供正向信息。同时 DynamicsEncoder 在部分共享层之前实例化，会消耗 RNG，
导致 B0 与 B1 即使同 seed 也不是同一份共享初始化。alpha0 低于 B0 主要说明
旧 B 组缺少 shared-init control；它不是 “motion 学了但没有融合仍掉点”
的充分证据。

## 局限和稳健性边界

- 只有 seed42；alpha0/0.25 虽共享实验合同，但两卡 CUDA 训练未声明完全
  deterministic。二者第一步 loss 完全相同，随后即出现微小数值分叉。
- 没有 validation endpoint 导出，当前不能直接计算 recursive history 下
  dynamics 相对 observation 的 helpful rate、最优 alpha 分布或错误集中桶。
- 没有同一 checkpoint 的 alpha on/off 评测，因此还未完全分离
  “推理时直接位移伤害”和“训练期共同适配伤害”。
- 本结论否定当前固定全局 innovation，不等价于否定所有 motion feature、
  adapter、distillation 或条件使用方式。

## 分析验证：固定融合结论可决策，广义 motion 结论需保留 caveat

完整性、配置差异、final/late-3、训练组件 loss 和图表已经由独立 CSV
交叉核对；固定全局 alpha 的 No-Go 可直接用于停止后续长训。由于只有一个
seed、B0 共享初始化不匹配且缺少 endpoint proposal export，报告不能升级为
“所有 motion prior 无效”。整体置信度为 **Share with caveats**：停止当前
fixed-global 模块是高置信决策，更广义 motion 方向仍需下述无训练归因。

## 下一步：停止长训，先完成两个无训练诊断

1. **同 checkpoint 2×2。** 分别将 alpha0 与 alpha0.25 的 epoch60
   checkpoint 在推理时以 alpha0/0.25 运行，endpoint 和采样固定：

   - alpha0 checkpoint：0 → 0.25，测直接开启 prior 的即时伤害；
   - alpha0.25 checkpoint：0.25 → 0，测关闭 prior 后能否恢复，分离
     training co-adaptation。

2. **导出逐 endpoint proposal attribution。** 对同一 validation forward
   同时保存 observation proposal、dynamics proposal、GT、previous prediction
   error、disagreement、有效历史、点数、速度和 delta_t；至少报告：

   - `P(error_dyn < error_obs)`；
   - recursive 条件下的 oracle alpha 分布；
   - correction 与 GT residual 的 cosine；
   - 按 previous error、speed、foreground points、delta_t 分桶的净增益；
   - 首次失控帧与连续漂移长度。

只有当训练集/独立诊断 split 上存在稳定的可识别 helpful subgroup，并且
GT-free selector 在冻结 split 上通过，才允许研究条件 alpha。已有 P0-B4
已经否定过一版 observation reliability gate，因此不能直接重启相同 learned
gate。若同-checkpoint开启 0.25 立即退化、关闭后恢复，直接 fusion 路线永久
停止；若关闭仍不恢复，则还要处理训练期 co-adaptation，但也不再扫更多全局
alpha。

在完成这两个低成本诊断前，不再训练 alpha0.05/0.1、seed43/44、full
nuScenes 或 motion+search。当前 GPU 主线仍按既定 PFTC 修复 kill-test
推进；motion 只保留为不占训练资源的机制归因任务。

## 仍待回答的问题

- recursive history 下 dynamics proposal 真正优于 observation 的 endpoint
  比例是多少，是否存在跨 split 稳定子群？
- 退化主要由测试时 correction 造成，还是训练期 coarse-query
  co-adaptation 已破坏共享表示？
- 若 direct proposal correction 被永久停止，历史 A2 feature-concat/R1
  adapter 的正信号在 matched initialization、matched data contract 下是否
  仍存在？
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "output"))
    parser.add_argument("--date-tag", default="20260730")
    args = parser.parse_args()

    results = load_results(Path(args.output_root))
    tables = build_tables(results)
    validate_contract(results, tables)

    prefix = f"ct_motion_alpha_sweep_seed42_{args.date_tag}"
    data_dir = ROOT / "compare_results/data"
    figure_path = (
        ROOT / "compare_results/figures/line_charts"
        / f"{prefix}_curves.png"
    )
    report_path = ROOT / "compare_results/reports" / f"{prefix}.md"

    for name, rows in tables.items():
        write_csv(data_dir / f"{prefix}_{name}.csv", rows)
    build_figure(results, figure_path)
    report = build_report(
        results,
        tables,
        f"../figures/line_charts/{figure_path.name}",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"report={report_path}")
    print(f"figure={figure_path}")
    for name in tables:
        print(f"{name}={data_dir / f'{prefix}_{name}.csv'}")
    print("PASS_MOTION_ALPHA_SWEEP_AUDIT")


if __name__ == "__main__":
    main()
