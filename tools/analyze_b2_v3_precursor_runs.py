#!/usr/bin/env python3
"""Audit the three seed42 precursor reruns launched before B2-v3 training.

The script deliberately separates two questions:

1. Did the B1 / Search-v2.1 scratch reruns improve normal-mini tracking?
2. Do those runs constitute a trained B2-v3 model?

It writes inspectable CSV/JSON evidence, an executed notebook, and the
canonical artifact input used by the portable technical report.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "compare_results" / "data"
NOTEBOOK_DIR = ROOT / "compare_results" / "notebooks"
REPORT_DIR = (
    ROOT / "compare_results" / "reports"
    / "b2_v3_precursor_seed42_20260804"
)
STEM = "b2_v3_precursor_seed42_20260804"
STEPS_PER_EPOCH = 1262


RUNS = {
    "B0_HIST": {
        "label": "Historical B0",
        "path": "output/20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16",
        "role": "historical_guardrail",
    },
    "OLD_B1": {
        "label": "Previous B1-motion-v3",
        "path": "output/20260801-0117-02_ct_motion_v3-b1motion_v3_mini_car_60ep_bs16_seed42",
        "role": "previous_rerun",
    },
    "OLD_SEARCH": {
        "label": "Previous Search-v2.1 only",
        "path": "output/20260802-1530-08_seqtrack3d_search_v21-search_v21_mini_car_60ep_bs16_seed42",
        "role": "previous_rerun",
    },
    "OLD_FULL": {
        "label": "Previous Motion+Search-v2.1",
        "path": "output/20260802-1530-09_ct_motion_search_v21-motion_search_v21_mini_car_60ep_bs16_seed42",
        "role": "previous_rerun",
    },
    "NEW_B1": {
        "label": "New B1-motion-v3",
        "path": "output/20260803-2257-02_ct_motion_v3-b1_motion_v3_mini_car_60ep_bs16_seed42",
        "role": "requested_rerun",
    },
    "NEW_SEARCH": {
        "label": "New Search-v2.1 only",
        "path": "output/20260803-2258-08_seqtrack3d_search_v21-b2_search_only_v21_mini_car_60ep_bs16_seed42",
        "role": "requested_rerun",
    },
    "NEW_FULL": {
        "label": "New Motion+Search-v2.1",
        "path": "output/20260803-2258-09_ct_motion_search_v21-b2_search_motion_v21_mini_car_60ep_bs16_seed42",
        "role": "requested_rerun",
    },
}

NEW_RUN_IDS = ("NEW_B1", "NEW_SEARCH", "NEW_FULL")
CHART_RUN_IDS = ("B0_HIST", *NEW_RUN_IDS)


def version_dir(run_id: str) -> Path:
    return ROOT / RUNS[run_id]["path"] / "lightning_logs" / "version_0"


def scalar_events(run_id: str, leaf: str) -> list[tuple[int, float]]:
    path = version_dir(run_id) / leaf
    if not path.is_dir():
        return []
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        return []
    tag = "loss" if "loss" in tags else tags[0]
    return [
        (int(event.step), float(event.value))
        for event in accumulator.Scalars(tag)
    ]


def epoch_mean(events: list[tuple[int, float]], epoch: int) -> float | None:
    lower = (epoch - 1) * STEPS_PER_EPOCH
    upper = epoch * STEPS_PER_EPOCH
    values = [
        value for step, value in events
        if lower <= step < upper and math.isfinite(value)
    ]
    return float(statistics.mean(values)) if values else None


def collect_validation():
    rows = []
    by_run = {}
    for run_id, spec in RUNS.items():
        success = dict(scalar_events(run_id, "metrics_test_success"))
        precision = dict(scalar_events(run_id, "metrics_test_precision"))
        steps = sorted(set(success) & set(precision))
        if len(steps) != 12:
            raise RuntimeError(
                f"{run_id}: expected 12 validation points, got {len(steps)}")
        current = []
        for index, step in enumerate(steps, 1):
            row = {
                "run_id": run_id,
                "arm": spec["label"],
                "role": spec["role"],
                "epoch": index * 5,
                "step": step,
                "success": success[step],
                "precision": precision[step],
            }
            rows.append(row)
            current.append(row)
        by_run[run_id] = current
    return rows, by_run


def collect_summaries(by_run):
    rows = []
    for run_id, values in by_run.items():
        best_success = max(values, key=lambda row: row["success"])
        best_precision = max(values, key=lambda row: row["precision"])
        rows.append({
            "run_id": run_id,
            "arm": RUNS[run_id]["label"],
            "role": RUNS[run_id]["role"],
            "final_success": values[-1]["success"],
            "final_precision": values[-1]["precision"],
            "late3_success": statistics.mean(
                row["success"] for row in values[-3:]),
            "late3_precision": statistics.mean(
                row["precision"] for row in values[-3:]),
            "best_success": best_success["success"],
            "best_success_epoch": best_success["epoch"],
            "best_precision": best_precision["precision"],
            "best_precision_epoch": best_precision["epoch"],
        })
    return rows


def collect_comparisons(summaries):
    lookup = {row["run_id"]: row for row in summaries}
    pairs = (
        ("NEW_B1", "B0_HIST", "historical B0 guardrail"),
        ("NEW_SEARCH", "B0_HIST", "historical B0 guardrail"),
        ("NEW_FULL", "B0_HIST", "historical B0 guardrail"),
        ("NEW_FULL", "NEW_B1", "matched new B1"),
        ("NEW_FULL", "NEW_SEARCH", "matched new Search-only"),
        ("NEW_B1", "OLD_B1", "repeatability"),
        ("NEW_SEARCH", "OLD_SEARCH", "repeatability"),
        ("NEW_FULL", "OLD_FULL", "repeatability"),
    )
    rows = []
    for run_id, reference_id, basis in pairs:
        current = lookup[run_id]
        reference = lookup[reference_id]
        rows.append({
            "run_id": run_id,
            "reference_id": reference_id,
            "basis": basis,
            "delta_final_success": (
                current["final_success"] - reference["final_success"]),
            "delta_final_precision": (
                current["final_precision"] - reference["final_precision"]),
            "delta_late3_success": (
                current["late3_success"] - reference["late3_success"]),
            "delta_late3_precision": (
                current["late3_precision"] - reference["late3_precision"]),
        })
    return rows


TRAINING_METRICS = {
    "NEW_B1": {
        "kinematic_rmse": "loss_motion_v3_kinematic_rmse",
        "prior_rmse": "loss_motion_v3_prior_rmse",
        "gate_precision": "loss_motion_v3_gate_precision",
        "gate_applied_rate": "loss_motion_v3_gate_applied_rate",
        "observation_error": "loss_motion_v3_observation_error",
        "final_error": "loss_motion_v3_final_error",
    },
    "NEW_SEARCH": {
        "search_proposal_loss": "loss_loss_search_v21_proposal",
        "search_valid_rate": "loss_search_v21_candidate_valid_rate",
        "observation_error": "loss_advantage_observation_error",
        "search_error_valid": "loss_advantage_search_error_valid",
        "final_error": "loss_advantage_final_error",
        "search_applied_rate": "loss_advantage_search_applied_rate",
        "search_helpful_precision": (
            "loss_advantage_search_helpful_precision"),
    },
    "NEW_FULL": {
        "kinematic_rmse": "loss_motion_v3_kinematic_rmse",
        "prior_rmse": "loss_motion_v3_prior_rmse",
        "search_proposal_loss": "loss_loss_search_v21_proposal",
        "search_valid_rate": "loss_search_v21_candidate_valid_rate",
        "observation_error": "loss_advantage_observation_error",
        "motion_error_valid": "loss_advantage_motion_error_valid",
        "search_error_valid": "loss_advantage_search_error_valid",
        "final_error": "loss_advantage_final_error",
        "motion_helpful_rate": "loss_advantage_motion_helpful_rate",
        "search_helpful_precision": (
            "loss_advantage_search_helpful_precision"),
        "motion_weight": "loss_advantage_motion_weight",
        "search_weight": "loss_advantage_search_weight",
        "search_applied_rate": "loss_advantage_search_applied_rate",
    },
    "OLD_SEARCH": {
        "search_proposal_loss": "loss_loss_search_v21_proposal",
        "search_error_valid": "loss_advantage_search_error_valid",
    },
    "OLD_FULL": {
        "search_proposal_loss": "loss_loss_search_v21_proposal",
        "search_error_valid": "loss_advantage_search_error_valid",
    },
}


def collect_training_diagnostics():
    rows = []
    for run_id, metrics in TRAINING_METRICS.items():
        for metric, leaf in metrics.items():
            events = scalar_events(run_id, leaf)
            nonfinite = sum(
                not math.isfinite(value) for _, value in events)
            rows.append({
                "run_id": run_id,
                "arm": RUNS[run_id]["label"],
                "metric": metric,
                "event_count": len(events),
                "nonfinite_count": nonfinite,
                "epoch11_mean": epoch_mean(events, 11),
                "epoch20_mean": epoch_mean(events, 20),
                "epoch40_mean": epoch_mean(events, 40),
                "epoch60_mean": epoch_mean(events, 60),
            })
    return rows


def collect_integrity():
    rows = []
    for run_id in NEW_RUN_IDS:
        run_path = ROOT / RUNS[run_id]["path"]
        provenance = json.loads(
            (run_path / "run_provenance.json").read_text(encoding="utf-8"))
        config = provenance["resolved_config"]
        total = scalar_events(run_id, "loss_loss_total")
        validation = scalar_events(run_id, "metrics_test_success")
        last_checkpoint = version_dir(run_id) / "checkpoints" / "last.ckpt"
        rows.append({
            "run_id": run_id,
            "config_path": provenance["config_path"],
            "commit": provenance["git"]["commit"],
            "dirty": bool(provenance["git"]["dirty_any"]),
            "seed": provenance["seed"],
            "train_split": config["train_split"],
            "val_split": config["val_split"],
            "optimizer_steps": len(total),
            "validation_points": len(validation),
            "last_checkpoint_present": last_checkpoint.is_file(),
            "init_checkpoint": provenance.get("init_checkpoint_path"),
            "use_motion_conditioned_search_v3": bool(
                config.get("use_motion_conditioned_search_v3", False)),
            "use_action_consistent_router_v3": bool(
                config.get("use_action_consistent_router_v3", False)),
        })
    commits = {row["commit"] for row in rows}
    if len(commits) != 1:
        raise RuntimeError("new precursor runs do not share one commit")
    for row in rows:
        if (row["dirty"] or row["seed"] != 42
                or row["optimizer_steps"] != 75720
                or row["validation_points"] != 12
                or not row["last_checkpoint_present"]):
            raise RuntimeError(f"run-integrity failure: {row['run_id']}")
    return rows


def row_lookup(rows, key="run_id"):
    return {row[key]: row for row in rows}


def collect_decisions(summaries, diagnostics, integrity):
    summary = row_lookup(summaries)
    diag = {
        (row["run_id"], row["metric"]): row["epoch60_mean"]
        for row in diagnostics
    }
    all_not_v3 = all(
        not row["use_motion_conditioned_search_v3"]
        and not row["use_action_consistent_router_v3"]
        for row in integrity)
    b0 = summary["B0_HIST"]
    rows = []
    for run_id in NEW_RUN_IDS:
        current = summary[run_id]
        rows.append({
            "check": f"{run_id}_historical_b0_gain",
            "criterion": "+0.5 Success and +1.0 Precision at epoch60",
            "observed": (
                f"{current['final_success']-b0['final_success']:+.3f} S / "
                f"{current['final_precision']-b0['final_precision']:+.3f} P"),
            "status": "failed",
            "interpretation": (
                "Historical guardrail only; a same-commit B0 is still missing."),
        })
    rows.extend((
        {
            "check": "b2_v3_identity",
            "criterion": "Config13 refiner checkpoint with V3 path enabled",
            "observed": (
                "All three runs use configs 02/08/09; V3 flags are false."
                if all_not_v3 else "Unexpected V3 flag detected."),
            "status": "failed" if all_not_v3 else "review",
            "interpretation": "These are precursor/ablation runs, not B2-v3.",
        },
        {
            "check": "offline_to_closed_loop_transfer",
            "criterion": "Offline candidate improvement transfers to tracking",
            "observed": (
                f"NEW_FULL epoch60 offline final error "
                f"{diag[('NEW_FULL','final_error')]:.3f} vs observation "
                f"{diag[('NEW_FULL','observation_error')]:.3f}; tracking "
                f"{summary['NEW_FULL']['final_success']:.3f}/"
                f"{summary['NEW_FULL']['final_precision']:.3f}."),
            "status": "failed",
            "interpretation": (
                "One-step training diagnostics do not transfer under recursive "
                "rollout; this is the distribution-shift failure V3 targets."),
        },
        {
            "check": "b2_v3_promotion_gate",
            "criterion": (
                "Candidate RMSE + calibration + mini_val obs_vs_all gate"),
            "observed": (
                "No config13 epoch20 diagnostics, rollout router, packaged "
                "checkpoint, or four-mode metrics exist."),
            "status": "not_run",
            "interpretation": "B2-v3 success cannot yet be evaluated.",
        },
    ))
    return rows


def collect_next_steps():
    return [
        {
            "order": 1,
            "action": "Keep the existing strict seed42 V3 init",
            "why": (
                "The new v2.1 full source has worse epoch60 search proposal "
                "loss/error than the already selected old full source."),
            "gate": "Verify init SHA256 and run the two-optimizer-step preflight.",
        },
        {
            "order": 2,
            "action": "Train cfg13 B2-v3 refiner for exactly 20 epochs",
            "why": (
                "This is the first run that activates shared B1/B2 history and "
                "the supervised 384-d evidence path."),
            "gate": (
                "Use epoch20 last.ckpt; refined valid-foreground RMSE must beat "
                "both B1 motion and raw Search."),
        },
        {
            "order": 3,
            "action": "Run round0, provisional router, round1, merge and final calibration",
            "why": "The current full collapse confirms recursive state shift matters.",
            "gate": (
                "Calibration precision >=75%, harm <=10%, coverage 5-25%, "
                "at least 100 selected states."),
        },
        {
            "order": 4,
            "action": "Package and evaluate obs_only/obs_vs_motion/obs_vs_refined/obs_vs_all",
            "why": "Only same-checkpoint modes isolate the selective intervention.",
            "gate": (
                "obs_vs_all >54.132 S / 64.755 P and >=obs_only +0.5/+1.0."),
        },
        {
            "order": 5,
            "action": "Run seeds43/44 only after seed42 passes",
            "why": "The current evidence is one seed and no completed V3 model.",
            "gate": "Three-seed mean gain positive with no catastrophic seed.",
        },
    ]


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_notebook(paths, summaries, diagnostics):
    summary = row_lookup(summaries)
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "## tl;dr\n\n"
            "- 这三组是 B1/Search-v2.1 上游重跑，**不是 B2-v3 refiner**。\n"
            f"- 新 Motion+Search-v2.1 final 为 "
            f"**{summary['NEW_FULL']['final_success']:.3f} / "
            f"{summary['NEW_FULL']['final_precision']:.3f}**，闭环严重失败。\n"
            "- 不运行 seed43/44；下一步是从严格 init 训练 cfg13 20 epochs。"
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "分析单位为每个训练 run。Final=epoch60；Late-3=epoch50/55/60 "
            "算术均值。历史 B0 仅作 guardrail，因为当前 clean commit 没有 matched B0。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "ROOT = Path.cwd()\n"
            f"validation = pd.read_csv(ROOT / r'{paths['validation']}')\n"
            f"summary = pd.read_csv(ROOT / r'{paths['summary']}')\n"
            f"diagnostics = pd.read_csv(ROOT / r'{paths['diagnostics']}')\n"
            f"decisions = pd.read_csv(ROOT / r'{paths['decisions']}')\n"
            "summary"
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n完整性与身份判断："),
        nbformat.v4.new_code_cell(
            f"integrity = pd.read_csv(ROOT / r'{paths['integrity']}')\n"
            "integrity"
        ),
        nbformat.v4.new_markdown_cell(
            "## Results\n\n"
            "四条曲线展示历史 B0 guardrail 与三条新重跑。"
        ),
        nbformat.v4.new_code_cell(
            "visible = validation[validation.run_id.isin(" 
            "['B0_HIST','NEW_B1','NEW_SEARCH','NEW_FULL'])]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "for run_id, frame in visible.groupby('run_id'):\n"
            "    axes[0].plot(frame.epoch, frame.success, marker='o', label=run_id)\n"
            "    axes[1].plot(frame.epoch, frame.precision, marker='o', label=run_id)\n"
            "axes[0].set(title='Validation Success', xlabel='Epoch', ylabel='Success')\n"
            "axes[1].set(title='Validation Precision', xlabel='Epoch', ylabel='Precision')\n"
            "for axis in axes:\n"
            "    axis.grid(alpha=.2); axis.legend(fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbformat.v4.new_code_cell(
            "full = diagnostics[(diagnostics.run_id == 'NEW_FULL') & "
            "diagnostics.metric.isin(['observation_error','motion_error_valid',"
            "'search_error_valid','final_error'])][['metric','epoch60_mean']]\n"
            "full"
        ),
        nbformat.v4.new_code_cell("decisions"),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. 当前没有训练完成的 B2-v3，不能宣称 V3 成功或涨点。\n"
            "2. Search 候选会学习，但 v2.1 full 的 one-step 改善未转化为递归 tracking。\n"
            "3. 下一步应进入 cfg13→两轮 rollout→最终 router/package，而不是再重跑 cfg09。"
        ),
    ]
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook_path = NOTEBOOK_DIR / f"{STEM}.ipynb"
    nbformat.write(notebook, notebook_path)
    client = NotebookClient(
        notebook, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, notebook_path)
    return notebook_path


def make_artifact(paths, validation, summaries, diagnostics, integrity,
                  decisions, next_steps):
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = row_lookup(summaries)
    comparison_rows = [
        row for row in summaries
        if row["run_id"] in CHART_RUN_IDS
    ]
    validation_chart = [
        {**row, "series": RUNS[row["run_id"]]["label"]}
        for row in validation if row["run_id"] in CHART_RUN_IDS
    ]
    full_error_rows = [
        {
            "metric": row["metric"],
            "error": row["epoch60_mean"],
            "run": "New Motion+Search-v2.1",
        }
        for row in diagnostics
        if row["run_id"] == "NEW_FULL"
        and row["metric"] in (
            "observation_error", "motion_error_valid",
            "search_error_valid", "final_error")
    ]
    sources = [
        {
            "id": source_id,
            "label": label,
            "path": path,
            "query": {
                "language": "python",
                "engine": "Python/TensorBoard",
                "description": description,
                "executed_at": "2026-08-04",
                "sql": (
                    "SELECT * FROM read_csv_auto('" + path
                    + "', header=true)"),
            },
        }
        for source_id, label, path, description in (
            ("validation_source", "Raw validation scalar extraction",
             paths["validation"],
             "Twelve unsmoothed paired Success/Precision points per run."),
            ("summary_source", "Final and late-window summary",
             paths["summary"],
             "Final, late-3 and best diagnostics recomputed from validation."),
            ("diagnostic_source", "Epoch-level training diagnostics",
             paths["diagnostics"],
             "Epoch means recomputed from raw TensorBoard training scalars."),
            ("integrity_source", "Run provenance and completeness audit",
             paths["integrity"],
             "Config identity, clean commit, steps, validation and checkpoint checks."),
            ("decision_source", "Promotion decision register",
             paths["decisions"],
             "Applies the stated B2-v3 identity and promotion requirements."),
            ("next_source", "Ordered next-step register",
             paths["next_steps"],
             "Orders cfg13 refiner, rollout, calibration and promotion work."),
        )
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "B2-v3 上游三组重跑复核：尚未进入 V3 训练",
        "description": (
            "Seed42 normal-mini precursor results, run identity, closed-loop "
            "failure diagnosis, and the decision path into formal B2-v3."),
        "generatedAt": generated_at,
        "sources": sources,
        "charts": [
            {
                "id": "success_curve",
                "title": "Normal-mini validation Success by epoch",
                "subtitle": (
                    "Car, seed42; historical B0 is a guardrail, not a "
                    "same-commit control."),
                "type": "line",
                "dataset": "validation_chart",
                "sourceId": "validation_source",
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative",
                          "label": "Epoch"},
                    "y": {"field": "success", "type": "quantitative",
                          "label": "Success"},
                    "color": {"field": "series", "type": "nominal",
                              "label": "Run"},
                    "tooltip": [
                        {"field": "series", "type": "nominal",
                         "label": "Run"},
                        {"field": "epoch", "type": "quantitative",
                         "label": "Epoch"},
                        {"field": "success", "type": "quantitative",
                         "label": "Success"},
                    ],
                },
                "xAxisTitle": "Epoch",
                "yAxisTitle": "Success",
            },
            {
                "id": "precision_curve",
                "title": "Normal-mini validation Precision by epoch",
                "subtitle": (
                    "The v2.1 full arm remains near 27 after fusion is active."),
                "type": "line",
                "dataset": "validation_chart",
                "sourceId": "validation_source",
                "encodings": {
                    "x": {"field": "epoch", "type": "quantitative",
                          "label": "Epoch"},
                    "y": {"field": "precision", "type": "quantitative",
                          "label": "Precision"},
                    "color": {"field": "series", "type": "nominal",
                              "label": "Run"},
                    "tooltip": [
                        {"field": "series", "type": "nominal",
                         "label": "Run"},
                        {"field": "epoch", "type": "quantitative",
                         "label": "Epoch"},
                        {"field": "precision", "type": "quantitative",
                         "label": "Precision"},
                    ],
                },
                "xAxisTitle": "Epoch",
                "yAxisTitle": "Precision",
            },
            {
                "id": "offline_error_chart",
                "title": "Epoch-60 one-step training errors",
                "subtitle": (
                    "Motion+Search-v2.1; lower is better, but this does not "
                    "measure recursive tracking."),
                "type": "bar",
                "dataset": "full_error_rows",
                "sourceId": "diagnostic_source",
                "encodings": {
                    "x": {"field": "metric", "type": "nominal",
                          "label": "Candidate/output"},
                    "y": {"field": "error", "type": "quantitative",
                          "label": "Mean center error"},
                    "tooltip": [
                        {"field": "metric", "type": "nominal",
                         "label": "Candidate/output"},
                        {"field": "error", "type": "quantitative",
                         "label": "Mean center error"},
                    ],
                },
                "xAxisTitle": "Candidate/output",
                "yAxisTitle": "Mean center error",
            },
        ],
        "tables": [
            {
                "id": "summary_table",
                "title": "Normal-mini result summary",
                "subtitle": (
                    "Epoch60 is primary; late-3 averages epochs 50/55/60."),
                "dataset": "summary_visible",
                "sourceId": "summary_source",
                "density": "compact",
                "defaultSort": {"field": "final_success",
                                "direction": "desc"},
                "columns": [
                    {"field": "arm", "label": "Run", "type": "text"},
                    {"field": "final_success", "label": "Final S",
                     "type": "number", "format": "number"},
                    {"field": "final_precision", "label": "Final P",
                     "type": "number", "format": "number"},
                    {"field": "late3_success", "label": "Late-3 S",
                     "type": "number", "format": "number"},
                    {"field": "late3_precision", "label": "Late-3 P",
                     "type": "number", "format": "number"},
                ],
            },
            {
                "id": "identity_table",
                "title": "Run identity and completeness",
                "subtitle": (
                    "All requested runs are complete but none activates V3."),
                "dataset": "integrity",
                "sourceId": "integrity_source",
                "density": "compact",
                "defaultSort": {"field": "run_id", "direction": "asc"},
                "columns": [
                    {"field": "run_id", "label": "Run", "type": "text"},
                    {"field": "config_path", "label": "Config",
                     "type": "text"},
                    {"field": "optimizer_steps", "label": "Steps",
                     "type": "number", "format": "number"},
                    {"field": "validation_points", "label": "Val points",
                     "type": "number", "format": "number"},
                    {"field": "use_motion_conditioned_search_v3",
                     "label": "V3 enabled", "type": "boolean"},
                ],
            },
            {
                "id": "decision_table",
                "title": "Promotion checks",
                "subtitle": (
                    "Identity and missing V3 artifacts block a success claim."),
                "dataset": "decisions",
                "sourceId": "decision_source",
                "density": "spacious",
                "defaultSort": {"field": "check", "direction": "asc"},
                "columns": [
                    {"field": "check", "label": "Check", "type": "text"},
                    {"field": "criterion", "label": "Criterion",
                     "type": "text"},
                    {"field": "observed", "label": "Observed",
                     "type": "text"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {"field": "interpretation", "label": "Interpretation",
                     "type": "text"},
                ],
            },
            {
                "id": "next_table",
                "title": "Ordered B2-v3 execution path",
                "subtitle": "Do not advance to more seeds before seed42 passes.",
                "dataset": "next_steps",
                "sourceId": "next_source",
                "density": "spacious",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "#", "type": "number",
                     "format": "number"},
                    {"field": "action", "label": "Action", "type": "text"},
                    {"field": "why", "label": "Why", "type": "text"},
                    {"field": "gate", "label": "Gate", "type": "text"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown",
             "body": "# B2-v3 上游三组重跑复核：尚未进入 V3 训练"},
            {
                "id": "technical_summary", "type": "markdown",
                "sourceId": "summary_source",
                "body": (
                    "## 技术结论：不能判定 B2-v3 成功，三条上游均未形成涨点证据\n\n"
                    f"新 B1 final 为 **{summary['NEW_B1']['final_success']:.3f} / "
                    f"{summary['NEW_B1']['final_precision']:.3f}**，Search-only 为 "
                    f"**{summary['NEW_SEARCH']['final_success']:.3f} / "
                    f"{summary['NEW_SEARCH']['final_precision']:.3f}**，Motion+Search-v2.1 "
                    f"只有 **{summary['NEW_FULL']['final_success']:.3f} / "
                    f"{summary['NEW_FULL']['final_precision']:.3f}**。三者分别来自 configs "
                    "02/08/09，而不是 config13；当前没有 epoch20 V3 refiner、两轮 rollout、"
                    "final router 或四模式评测。因此 B2-v3 状态是 **NOT_RUN**，不是成功或失败。"
                ),
            },
            {"id": "summary_block", "type": "table",
             "tableId": "summary_table"},
            {
                "id": "curve_finding", "type": "markdown",
                "sourceId": "validation_source",
                "body": (
                    "## B1 接近历史 B0，但 Precision 仍低；v2.1 full 稳定崩溃\n\n"
                    "B1 相对历史 B0 final 为 **−0.042 Success / −1.809 Precision**；"
                    "Search-only 为 **−2.053 / −4.566**；full 为 **−26.606 / −37.505**。"
                    "当前 commit 没有 matched B0，因此这些差值只能作为 guardrail；但所有"
                    "方向都不足以支持涨点。"
                ),
            },
            {"id": "success_block", "type": "chart",
             "chartId": "success_curve"},
            {"id": "precision_block", "type": "chart",
             "chartId": "precision_curve"},
            {
                "id": "transfer_finding", "type": "markdown",
                "sourceId": "diagnostic_source",
                "body": (
                    "## Search 在 one-step 训练中有效，但没有跨过递归闭环\n\n"
                    "new full 的 epoch60 offline final error 为 **0.230**，低于 observation "
                    "的 **0.248**；Search valid 约 **30.7%**，被选 Search helpful precision "
                    "约 **58.4%**。但 recursive mini_val tracking 只有 26.75/26.88。"
                    "这不是 NaN 或 under-training，而是训练状态分布与执行状态分布严重失配，"
                    "正好说明 V3 的显式 abstain、action-consistent gain 与 on-policy aggregation "
                    "仍然有必要。"
                ),
            },
            {"id": "offline_error_block", "type": "chart",
             "chartId": "offline_error_chart"},
            {
                "id": "scope", "type": "markdown",
                "sourceId": "integrity_source",
                "body": (
                    "## 数据范围与身份：训练完整，实验命名不能替代配置事实\n\n"
                    "三条新 run 共享 clean commit `73d6ce25`、seed42、mini_train/mini_val，"
                    "各有 75,720 optimizer steps、12 个验证点和 epoch60 last.ckpt。"
                    "不过 config02 是 B1，config08/09 是 v2.1 advantage fusion；V3 flags 均为 false。"
                ),
            },
            {"id": "identity_block", "type": "table",
             "tableId": "identity_table"},
            {
                "id": "methodology", "type": "markdown",
                "body": (
                    "## 方法：Final 决策、Late-3 稳定性与 raw TensorBoard 复算\n\n"
                    "Final 固定为 epoch60；Late-3 是 epochs50/55/60 均值；best 只用于诊断。"
                    "训练指标按每 epoch 1,262 batches 从 unsmoothed scalar 重算，并检查非有限值。"
                ),
            },
            {"id": "decision_block", "type": "table",
             "tableId": "decision_table"},
            {
                "id": "limitations", "type": "markdown",
                "body": (
                    "## 限制与稳健性：没有 matched B0，也没有真正 V3 artifact\n\n"
                    "历史 B0 来自旧 commit，不能支持严格论文增益；同 seed 重跑也出现可见波动。"
                    "更重要的是 promotion 所需 candidate diagnostics、calibration、on-policy rollout "
                    "与 obs_vs_all 指标全部缺失，所以不能把 v2.1 的失败外推成 V3 已失败。"
                ),
            },
            {
                "id": "recommendation", "type": "markdown",
                "sourceId": "next_source",
                "body": (
                    "## 下一步：停止 cfg09 重跑，进入 cfg13 的完整闭环流程\n\n"
                    "保留现有严格 init，先做 2-step preflight，再训练 refiner 20 epochs。"
                    "只有候选 RMSE gate 通过后才导出 round0/round1、训练最终 router、package 并"
                    "执行四模式评测；seed42 不通过时不运行 seed43/44。"
                ),
            },
            {"id": "next_block", "type": "table", "tableId": "next_table"},
            {
                "id": "questions", "type": "markdown",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "- cfg13 epoch20 的 refined RMSE 是否同时低于 B1 motion 和 raw Search？\n"
                    "- dev/calibration 上 router 的 helpful precision、harm 与 coverage 是否过门？\n"
                    "- packaged obs_vs_all 是否超过 54.132/64.755，并相对 obs_only +0.5/+1.0？"
                ),
            },
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "validation_chart": validation_chart,
                "summary_visible": comparison_rows,
                "full_error_rows": full_error_rows,
                "integrity": integrity,
                "decisions": decisions,
                "next_steps": next_steps,
            },
        },
        "sources": [],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = REPORT_DIR / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact_path


def main():
    validation, by_run = collect_validation()
    summaries = collect_summaries(by_run)
    comparisons = collect_comparisons(summaries)
    diagnostics = collect_training_diagnostics()
    integrity = collect_integrity()
    decisions = collect_decisions(
        summaries, diagnostics, integrity)
    next_steps = collect_next_steps()

    paths = {
        "validation": f"compare_results/data/{STEM}_validation.csv",
        "summary": f"compare_results/data/{STEM}_summary.csv",
        "comparisons": f"compare_results/data/{STEM}_comparisons.csv",
        "diagnostics": f"compare_results/data/{STEM}_training_diagnostics.csv",
        "integrity": f"compare_results/data/{STEM}_integrity.csv",
        "decisions": f"compare_results/data/{STEM}_decisions.csv",
        "next_steps": f"compare_results/data/{STEM}_next_steps.csv",
    }
    tables = {
        "validation": validation,
        "summary": summaries,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "integrity": integrity,
        "decisions": decisions,
        "next_steps": next_steps,
    }
    for key, rows in tables.items():
        write_csv(ROOT / paths[key], rows)

    result = {
        "summary": summaries,
        "comparisons": comparisons,
        "decisions": decisions,
        "next_steps": next_steps,
        "source_paths": paths,
    }
    result_path = DATA_DIR / f"{STEM}.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    notebook_path = make_notebook(paths, summaries, diagnostics)
    artifact_path = make_artifact(
        paths, validation, summaries, diagnostics, integrity,
        decisions, next_steps)
    print(json.dumps({
        "result": str(result_path.relative_to(ROOT)),
        "notebook": str(notebook_path.relative_to(ROOT)),
        "artifact": str(artifact_path.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
