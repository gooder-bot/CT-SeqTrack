#!/usr/bin/env python3
"""Audit the CT-SeqTrack seed42 search-only ablation."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ct_v2_ablation import collect_run, write_csv  # noqa: E402


RUN_MATCHES = {
    "B0": "01_seqtrack3d_baseline-ctv2_",
    "B1": "02_ct_motion-ctv2_",
    "B2": "03_ct_motion_search-ctv2_",
    "A1": "05_seqtrack3d_search_only-ctv2_",
}
DISPLAY_NAMES = {
    "B0": "B0 baseline",
    "B1": "B1 motion-only",
    "B2": "B2 motion + search",
    "A1": "A1 search-only",
}
DIAGNOSTICS = (
    "search_used_ratio",
    "search_expansion_ratio",
    "search_expansion_points",
    "usable_search_ratio",
    "training_loss_total",
)


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


def latest_matching_run(output_root: Path, fragment: str) -> Path:
    candidates = [
        path for path in output_root.iterdir()
        if path.is_dir() and fragment in path.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no output run contains required fragment: {fragment}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def checkpoint_topology(path: Path) -> dict[str, tuple[int, ...]]:
    install_easydict_fallback()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: tuple(value.shape)
        for key, value in payload.get("state_dict", {}).items()
    }


def config_differences(
        left: dict[str, Any],
        right: dict[str, Any],
) -> list[tuple[str, Any, Any]]:
    return [
        (key, left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]


def metric_delta(results, left: str, right: str, metric: str) -> float:
    return (
        results[left]["final"][metric]
        - results[right]["final"][metric]
    )


def late_delta(results, left: str, right: str, metric: str) -> float:
    return (
        results[left]["late3"][metric]
        - results[right]["late3"][metric]
    )


def format_score(value: float) -> str:
    return f"{value:.3f}"


def format_signed(value: float) -> str:
    return f"{value:+.3f}"


def build_report(
        results: dict[str, dict[str, Any]],
        config_diff: list[tuple[str, Any, Any]],
        topology_equal: bool,
        figure_relative_path: str,
) -> str:
    interaction_success = (
        results["B2"]["final"]["success"]
        - results["B1"]["final"]["success"]
        - results["A1"]["final"]["success"]
        + results["B0"]["final"]["success"]
    )
    interaction_precision = (
        results["B2"]["final"]["precision"]
        - results["B1"]["final"]["precision"]
        - results["A1"]["final"]["precision"]
        + results["B0"]["final"]["precision"]
    )
    a1_diag = results["A1"]["diagnostics"]
    b2_diag = results["B2"]["diagnostics"]
    lines = [
        "# CT-SeqTrack Search-only seed42 技术复核",
        "",
        "更新时间：2026-07-27",
        "",
        "## Technical Summary",
        "",
        "**当前 Search-only 实现不具备独立正贡献。** A1 final 为 "
        f"{format_score(results['A1']['final']['success'])} Success / "
        f"{format_score(results['A1']['final']['precision'])} Precision，"
        "相对 B0 分别下降 "
        f"{format_score(abs(metric_delta(results, 'A1', 'B0', 'success')))} / "
        f"{format_score(abs(metric_delta(results, 'A1', 'B0', 'precision')))}。"
        "其 best、final 和 late-3 全部远低于 B0，因此不能通过选 checkpoint "
        "挽救结论。",
        "",
        "A1 的训练 loss 与 B0 几乎相同，search 的训练激活和扩展比例又与 B2 "
        "一致，但递归验证从 epoch5 起就崩溃。这支持“训练与递归推理的 search "
        "分布/历史误差不匹配”或“search 与 motion 存在强交互”的诊断，"
        "不支持把失败简单归因于训练不足，也不能把它扩大为“任何搜索扩展都无效”。",
        "",
        "Overall Assessment: **Share with caveats**。结果足以否决当前 A1，"
        "但在缺少 validation/test 逐 endpoint search 使用率和服务器初始化"
        "等价 preflight 日志时，还不足以锁定唯一故障机制。",
        "",
        "## Search-only 在所有验证阶段都显著低于 baseline",
        "",
        "| arm | final Success | final Precision | best Success (epoch) | "
        "best Precision (epoch) | late-3 Success | late-3 Precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id in ("B0", "B1", "A1", "B2"):
        result = results[run_id]
        lines.append(
            f"| {DISPLAY_NAMES[run_id]} | "
            f"{format_score(result['final']['success'])} | "
            f"{format_score(result['final']['precision'])} | "
            f"{format_score(result['best_success']['success'])} "
            f"({result['best_success']['epoch']}) | "
            f"{format_score(result['best_precision']['precision'])} "
            f"({result['best_precision']['epoch']}) | "
            f"{format_score(result['late3']['success'])} | "
            f"{format_score(result['late3']['precision'])} |"
        )
    lines.extend([
        "",
        "| comparison | final Success delta | final Precision delta | interpretation |",
        "|---|---:|---:|---|",
        f"| A1 − B0: search-only | "
        f"{format_signed(metric_delta(results, 'A1', 'B0', 'success'))} | "
        f"{format_signed(metric_delta(results, 'A1', 'B0', 'precision'))} | "
        "standalone search fails |",
        f"| B1 − B0: motion-only | "
        f"{format_signed(metric_delta(results, 'B1', 'B0', 'success'))} | "
        f"{format_signed(metric_delta(results, 'B1', 'B0', 'precision'))} | "
        "standalone motion fails |",
        f"| B2 − B1: add search after motion | "
        f"{format_signed(metric_delta(results, 'B2', 'B1', 'success'))} | "
        f"{format_signed(metric_delta(results, 'B2', 'B1', 'precision'))} | "
        "positive rescue interaction |",
        f"| B2 − B0: combined model | "
        f"{format_signed(metric_delta(results, 'B2', 'B0', 'success'))} | "
        f"{format_signed(metric_delta(results, 'B2', 'B0', 'precision'))} | "
        "still below baseline |",
        "",
        "按四格 final 结果计算的描述性交互项为 "
        f"{format_signed(interaction_success)} Success / "
        f"{format_signed(interaction_precision)} Precision。它说明 search 的"
        "方向依赖于是否存在 motion 分支，而不是一个可直接相加的独立模块。"
        "由于两种网络结构的共享初始化没有统一 artifact 证明，该交互项只能"
        "作为诊断量，不能作为论文级因果效应。",
        "",
        "下图使用相同的 0–70 纵轴展示 12 个固定验证点。A1 曲线从未接近 B0；"
        "B2 仅作为 search 与 motion 交互的上下文，不替代 A1−B0 的主比较。",
        "",
        f"![B0/B1/A1/B2 validation curves]({figure_relative_path})",
        "",
        "## 训练侧几乎正常，问题集中在递归验证语义",
        "",
        "| diagnostic | B0 | A1 search-only | B2 motion+search |",
        "|---|---:|---:|---:|",
        f"| epoch60 mean training loss | "
        f"{results['B0']['diagnostics']['training_loss_total']['late_epoch_mean']:.4f} | "
        f"{a1_diag['training_loss_total']['late_epoch_mean']:.4f} | "
        f"{b2_diag['training_loss_total']['late_epoch_mean']:.4f} |",
        f"| training search-used sample ratio | "
        f"{100 * results['B0']['diagnostics']['search_used_ratio']['mean']:.3f}% | "
        f"{100 * a1_diag['search_used_ratio']['mean']:.3f}% | "
        f"{100 * b2_diag['search_used_ratio']['mean']:.3f}% |",
        f"| mean expansion token share | "
        f"{100 * results['B0']['diagnostics']['search_expansion_ratio']['mean']:.3f}% | "
        f"{100 * a1_diag['search_expansion_ratio']['mean']:.3f}% | "
        f"{100 * b2_diag['search_expansion_ratio']['mean']:.3f}% |",
        f"| mean expansion-only available points | "
        f"{results['B0']['diagnostics']['search_expansion_points']['mean']:.3f} | "
        f"{a1_diag['search_expansion_points']['mean']:.3f} | "
        f"{b2_diag['search_expansion_points']['mean']:.3f} |",
        "",
        "A1 与 B0 的 epoch60 training loss 只差约 0.0013；A1 与 B2 的 "
        "search-used ratio、expansion token share 和 expansion points 也近乎"
        "相同。因此 A1 不是因为 search 没有执行，也没有显示常规训练 loss "
        "发散。最合理的待验证假设是：训练中 mostly canonical/correlated "
        "history 只让少量样本启用 tube，而递归预测历史一旦产生偏差，tube "
        "可能抽入背景并形成误差反馈。当前 events 没有记录验证阶段的 search "
        "激活率，所以这仍是机制推断。",
        "",
        "## 范围、数据和指标口径",
        "",
        "- 数据：nuScenes v1.0-mini，Car；mini_train 274 tracklets / "
        "5,051 frames，mini_val 106 tracklets / 2,285 frames。",
        "- 协议：normal cadence、seed42、candidate4、batch16、60 epoch，"
        "每 5 epoch 验证一次；主结果固定使用 epoch60 `last.ckpt`。",
        "- Success/Precision 直接读取 TensorBoard validation scalars；"
        "best 和 late-3 只用于稳定性诊断。",
        "- A1 与 B0 的 checkpoint 都有 320 个同名、同 shape state tensors，"
        f"模型拓扑检查为 {'PASS' if topology_equal else 'FAIL'}。",
        "- A1/B0 resolved-config 的实质变化只包括 search 开关以及仅供 "
        "search 使用的 correlated history；配置名和 tag 属于 provenance。",
        "",
        "## 完整性和方法核验",
        "",
        "| arm | status | commit | train steps | validation points | last checkpoint |",
        "|---|---|---|---:|---:|---|",
    ])
    for run_id in ("B0", "B1", "A1", "B2"):
        result = results[run_id]
        commit = result["provenance"]["git"]["commit"][:7]
        lines.append(
            f"| {run_id} | {result['status']} | `{commit}` | "
            f"{result['training_scalar_count']:,}/{result['expected_steps']:,} | "
            f"{result['validation_count']}/{result['expected_validations']} | "
            f"epoch {result['checkpoint_epoch'] + 1} |"
        )
    visible_config_diff = ", ".join(key for key, _, _ in config_diff)
    lines.extend([
        "",
        f"A1/B0 resolved-config 差异字段为：`{visible_config_diff}`。B0 使用 "
        "`d86990c`，A1 使用 `052ae8d`；中间代码变化包括 batch1 shape 修复、"
        "search-only 解耦和测试，不改变 B0 的共享层定义。",
        "",
        "## 限制、稳健性与未解决证据",
        "",
        "1. **缺少服务器初始化等价日志。** 本地结果证明 checkpoint topology "
        "完全一致，源码也提供 seeded exact-init checker，但 "
        "`search_only_model_equivalence.log` 未随结果拉回，不能声称该"
        " preflight artifact 已审计通过。",
        "2. **缺少验证阶段 search diagnostics。** 当前只有训练阶段 search "
        "使用率；无法从现有 event 判断递归验证中 tube 的激活率、扩展点数及"
        "首次导致漂移的 endpoint。",
        "3. **单 seed、mini。** 该限制不妨碍否决幅度巨大的当前 A1，但不能"
        "估计更保守 search 设计的方差。",
        "4. **非因果时间结论。** A1 使用 true effective time，但尚未通过正常"
        "集 guardrail，因此不运行或解释 fixed/shuffled。",
        "",
        "## 推荐下一步：先做同 checkpoint 的 Search 开/关 2×2",
        "",
        "暂不训练 A2，也不调低 75/25 或加入新 gate。先用现有两个 checkpoint "
        "做四次无训练评测：",
        "",
        "| checkpoint | baseline crop | search-on crop | purpose |",
        "|---|---|---|---|",
        "| B0 final | 已有 B0 | 待测 | search 的纯推理影响 |",
        "| A1 final | 待测 | 已有 A1 | 训练暴露与推理 search 的分离 |",
        "",
        "若两个 checkpoint 都只在 search-on 时崩溃，可确认递归 search "
        "路径是主因；若 A1 在 search-off 下仍崩溃，则训练时稀疏的 expansion "
        "已改变模型。完成 2×2 后再决定删除当前 Search，还是增加 fail-closed "
        "条件、递归历史训练和更小扩展预算。",
        "",
        "## Further Questions",
        "",
        "- A1 checkpoint 关闭 search 后能否恢复到 B0 水平？",
        "- B0 checkpoint 仅在推理开启 search 是否立即跌到 A1 水平？",
        "- 递归验证的 search 激活率、扩展点数和首次漂移帧分别是多少？",
        "- B2 的恢复来自 motion proposal 抵消 search 背景，还是来自不同网络"
        "初始化/优化路径？",
    ])
    return "\n".join(lines) + "\n"


def build_figure(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = {
        "B0": {"color": "#333333", "marker": "o", "linestyle": "-"},
        "B1": {"color": "#888888", "marker": "^", "linestyle": "--"},
        "A1": {"color": "#D55E00", "marker": "s", "linestyle": "-"},
        "B2": {"color": "#0072B2", "marker": "D", "linestyle": "--"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for run_id in ("B0", "B1", "A1", "B2"):
        rows = results[run_id]["metrics"]
        epochs = [row["epoch"] for row in rows]
        for axis, metric in zip(axes, ("success", "precision")):
            axis.plot(
                epochs,
                [row[metric] for row in rows],
                linewidth=2.0,
                markersize=4.0,
                label=DISPLAY_NAMES[run_id],
                **styles[run_id],
            )
    axes[0].set_title("Success by validation epoch")
    axes[1].set_title("Precision by validation epoch")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Score")
        axis.set_xlim(4, 61)
        axis.set_ylim(0, 70)
        axis.set_xticks(range(5, 61, 5))
        axis.grid(color="#D9D9D9", alpha=0.55, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("CT-SeqTrack search-only ablation on nuScenes-mini (seed42)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    output_root = ROOT / "output"
    run_paths = {
        run_id: latest_matching_run(output_root, fragment)
        for run_id, fragment in RUN_MATCHES.items()
    }
    results = {
        run_id: collect_run(run_id, run_path)
        for run_id, run_path in run_paths.items()
    }
    incomplete = [
        run_id for run_id, result in results.items()
        if result["status"] != "COMPLETE"
    ]
    if incomplete:
        raise RuntimeError(
            "required runs are incomplete: " + ", ".join(incomplete))

    baseline_config = results["B0"]["config"]
    search_config = results["A1"]["config"]
    config_diff = config_differences(baseline_config, search_config)
    b0_checkpoint = (
        run_paths["B0"]
        / "lightning_logs/version_0/checkpoints/last.ckpt"
    )
    a1_checkpoint = (
        run_paths["A1"]
        / "lightning_logs/version_0/checkpoints/last.ckpt"
    )
    topology_equal = (
        checkpoint_topology(b0_checkpoint)
        == checkpoint_topology(a1_checkpoint)
    )

    prefix = "ct_search_only_seed42_20260727"
    data_dir = ROOT / "compare_results/data"
    report_path = ROOT / f"compare_results/reports/{prefix}.md"
    figure_path = (
        ROOT / f"compare_results/figures/line_charts/{prefix}_curves.png"
    )
    metric_rows = [
        row
        for run_id in ("B0", "B1", "A1", "B2")
        for row in results[run_id]["metrics"]
    ]
    summary_rows = []
    integrity_rows = []
    diagnostic_rows = []
    for run_id in ("B0", "B1", "A1", "B2"):
        result = results[run_id]
        summary_rows.append({
            "run_id": run_id,
            "label": DISPLAY_NAMES[run_id],
            "final_success": result["final"]["success"],
            "final_precision": result["final"]["precision"],
            "best_success": result["best_success"]["success"],
            "best_success_epoch": result["best_success"]["epoch"],
            "best_precision": result["best_precision"]["precision"],
            "best_precision_epoch": result["best_precision"]["epoch"],
            "late3_success": result["late3"]["success"],
            "late3_precision": result["late3"]["precision"],
            "late5_success": result["late5"]["success"],
            "late5_precision": result["late5"]["precision"],
        })
        integrity_rows.append({
            "run_id": run_id,
            "status": result["status"],
            "run_dir": result["run_dir"],
            "commit": result["provenance"]["git"]["commit"],
            "dirty_tracked": result["provenance"]["git"]["dirty_tracked"],
            "training_scalar_count": result["training_scalar_count"],
            "expected_steps": result["expected_steps"],
            "validation_count": result["validation_count"],
            "expected_validations": result["expected_validations"],
            "checkpoint_epoch": result["checkpoint_epoch"],
            "checkpoint_global_step": result["checkpoint_global_step"],
            "checkpoint_state_tensors": result["checkpoint_state_tensors"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        })
        for metric in DIAGNOSTICS:
            diagnostic_rows.append({
                "run_id": run_id,
                "metric": metric,
                **result["diagnostics"][metric],
            })

    interaction = {
        metric: (
            results["B2"]["final"][metric]
            - results["B1"]["final"][metric]
            - results["A1"]["final"][metric]
            + results["B0"]["final"][metric]
        )
        for metric in ("success", "precision")
    }
    delta_rows = [
        {
            "comparison": left + "_minus_" + right,
            "success_delta": metric_delta(results, left, right, "success"),
            "precision_delta": metric_delta(
                results, left, right, "precision"),
            "late3_success_delta": late_delta(
                results, left, right, "success"),
            "late3_precision_delta": late_delta(
                results, left, right, "precision"),
        }
        for left, right in (
            ("A1", "B0"),
            ("B1", "B0"),
            ("B2", "B1"),
            ("B2", "B0"),
            ("A1", "B1"),
        )
    ]
    delta_rows.append({
        "comparison": "descriptive_interaction",
        "success_delta": interaction["success"],
        "precision_delta": interaction["precision"],
        "late3_success_delta": None,
        "late3_precision_delta": None,
    })

    config_rows = [
        {
            "field": key,
            "B0": json.dumps(left, ensure_ascii=False),
            "A1": json.dumps(right, ensure_ascii=False),
        }
        for key, left, right in config_diff
    ]
    write_csv(data_dir / f"{prefix}_metrics.csv", metric_rows)
    write_csv(data_dir / f"{prefix}_summary.csv", summary_rows)
    write_csv(data_dir / f"{prefix}_integrity.csv", integrity_rows)
    write_csv(data_dir / f"{prefix}_diagnostics.csv", diagnostic_rows)
    write_csv(data_dir / f"{prefix}_deltas.csv", delta_rows)
    write_csv(data_dir / f"{prefix}_config_diff.csv", config_rows)
    build_figure(results, figure_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(
            results,
            config_diff,
            topology_equal,
            "../figures/line_charts/"
            + figure_path.name,
        ),
        encoding="utf-8",
    )

    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    for row in summary_rows:
        print(
            f"{row['run_id']}: "
            f"{row['final_success']:.3f}/"
            f"{row['final_precision']:.3f}")
    print(f"topology_equal={topology_equal}")


if __name__ == "__main__":
    main()
