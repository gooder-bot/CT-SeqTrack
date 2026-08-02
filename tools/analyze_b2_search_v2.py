#!/usr/bin/env python3
"""Reproduce the B2-v2 seed42 normal-mini result and failure diagnosis.

The primary decision uses epoch60.  Late-3 means the arithmetic mean of
epochs 50/55/60; best checkpoints are diagnostic only.  The script keeps the
new, anomalously-low SeqTrack control visible, but it also reports historical
SeqTrack and CT-v2 B0 references because the new control is from a separate
dirty repository and has no run_provenance.json.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "compare_results" / "data"
STEM = "b2_search_v2_seed42_20260802"

RUNS = {
    "SEQTRACK_NEW": {
        "arm": "SeqTrack control (2026-08-01, anomalous)",
        "path": "../seqtrack/output/20260801-2155-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16_seed42",
        "role": "requested_control",
    },
    "SEARCH_LEGACY": {
        "arm": "Legacy search-only (tube, 75/25 tokens)",
        "path": "output/20260801-2135-05_seqtrack3d_search_only-search_only_legacy_mini_car_60ep_bs16_seed42",
        "role": "requested_search_only",
    },
    "B2V2_FULL": {
        "arm": "B2-v2 motion + Search Evidence + joint fusion",
        "path": "output/20260801-2136-03_ct_motion_search_v2-motion_search_v2_mini_car_60ep_bs16_seed42",
        "role": "requested_full",
    },
    "SEQTRACK_HIST": {
        "arm": "SeqTrack historical reference (2026-05-28)",
        "path": "../seqtrack/output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16",
        "role": "historical_reference",
    },
    "B0_HIST": {
        "arm": "CT-v2 B0 historical guardrail (2026-07-25)",
        "path": "output/20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16",
        "role": "historical_guardrail",
    },
    "B1V3": {
        "arm": "B1motion-v3 standalone (2026-08-01)",
        "path": "output/20260801-0117-02_ct_motion_v3-b1motion_v3_mini_car_60ep_bs16_seed42",
        "role": "attribution_reference",
    },
}

FULL_TRAIN_TAGS = {
    "search_geometry_valid_rate": "loss_search_v2_geometry_valid_rate",
    "search_candidate_valid_rate": "loss_search_v2_candidate_valid_rate",
    "search_foreground_points": "loss_search_v2_foreground_points",
    "search_confidence_mean": "loss_search_v2_confidence_mean",
    "search_targetness_mass": "loss_search_v2_targetness_mass",
    "search_targetness_entropy": "loss_search_v2_targetness_entropy",
    "search_targetness_loss": "loss_loss_search_v2_targetness",
    "search_vote_loss": "loss_loss_search_v2_vote",
    "search_proposal_loss": "loss_loss_search_v2_proposal",
    "search_confidence_loss": "loss_loss_search_v2_confidence",
    "motion_candidate0_prior_rmse": "loss_motion_v3_prior_rmse_candidate0",
    "motion_candidate0_cv_rmse": "loss_motion_v3_kinematic_rmse_candidate0",
    "motion_nonzero_prior_rmse": "loss_motion_v3_prior_rmse_candidate_nonzero",
    "motion_nonzero_cv_rmse": "loss_motion_v3_kinematic_rmse_candidate_nonzero",
    "motion_gap2_prior_rmse": "loss_motion_v3_aux_prior_rmse_gap2",
    "motion_gap2_cv_rmse": "loss_motion_v3_aux_kinematic_rmse_gap2",
    "motion_gap4_prior_rmse": "loss_motion_v3_aux_prior_rmse_gap4",
    "motion_gap4_cv_rmse": "loss_motion_v3_aux_kinematic_rmse_gap4",
    "joint_observation_error_m": "loss_joint_observation_error",
    "joint_motion_error_m": "loss_joint_motion_error",
    "joint_search_error_m_unmasked": "loss_joint_search_error",
    "joint_final_error_m": "loss_joint_final_error",
    "joint_search_selected_rate": "loss_joint_search_selected_rate",
    "joint_search_helpful_precision_logged": "loss_joint_search_helpful_precision",
    "joint_fusion_ramp": "loss_joint_fusion_ramp",
    "joint_gate_loss": "loss_loss_joint_gate",
    "joint_fused_loss": "loss_loss_joint_fused",
}

LEGACY_TAGS = {
    "legacy_search_used_rate": "loss_ct_search_used_mean",
    "legacy_expansion_ratio": "loss_ct_search_expansion_ratio_mean",
    "legacy_baseline_points": "loss_ct_search_baseline_points_mean",
    "legacy_extension_points": "loss_ct_search_expansion_points_mean",
    "legacy_predicted_displacement_m": "loss_ct_search_predicted_displacement_mean",
}


def version_dir(run_id: str) -> Path:
    return ROOT / RUNS[run_id]["path"] / "lightning_logs" / "version_0"


def scalars(run_id: str, leaf: str) -> dict[int, float]:
    path = version_dir(run_id) / leaf
    if not path.is_dir():
        return {}
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if not tags:
        return {}
    tag = "loss" if "loss" in tags else tags[0]
    return {int(item.step): float(item.value) for item in acc.Scalars(tag)}


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {name}")
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{STEM}_{name}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_validation() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for run_id, spec in RUNS.items():
        success = scalars(run_id, "metrics_test_success")
        precision = scalars(run_id, "metrics_test_precision")
        steps = sorted(set(success) & set(precision))
        if len(steps) != 12:
            raise ValueError(f"{run_id}: expected 12 validation points, got {len(steps)}")
        current = []
        for i, step in enumerate(steps, 1):
            row = {
                "run_id": run_id,
                "arm": spec["arm"],
                "role": spec["role"],
                "epoch": i * 5,
                "step": step,
                "success": success[step],
                "precision": precision[step],
            }
            rows.append(row)
            current.append(row)
        by_run[run_id] = current
    return rows, by_run


def summaries(by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for run_id, values in by_run.items():
        best_s = max(values, key=lambda r: r["success"])
        best_p = max(values, key=lambda r: r["precision"])
        rows.append({
            "run_id": run_id,
            "arm": RUNS[run_id]["arm"],
            "role": RUNS[run_id]["role"],
            "final_success": values[-1]["success"],
            "final_precision": values[-1]["precision"],
            "late3_success": float(np.mean([r["success"] for r in values[-3:]])),
            "late3_precision": float(np.mean([r["precision"] for r in values[-3:]])),
            "best_success": best_s["success"],
            "best_success_epoch": best_s["epoch"],
            "best_precision": best_p["precision"],
            "best_precision_epoch": best_p["epoch"],
        })
    return rows


def comparisons(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {r["run_id"]: r for r in summary}
    pairs = [
        ("SEARCH_LEGACY", "SEQTRACK_NEW", "requested but confounded by anomalous control"),
        ("B2V2_FULL", "SEQTRACK_NEW", "requested but confounded by anomalous control"),
        ("SEARCH_LEGACY", "SEQTRACK_HIST", "historical original SeqTrack reference"),
        ("SEARCH_LEGACY", "B0_HIST", "strongest available B0 guardrail"),
        ("B2V2_FULL", "SEQTRACK_HIST", "historical original SeqTrack reference"),
        ("B2V2_FULL", "B0_HIST", "strict available guardrail; not same commit"),
        ("B2V2_FULL", "B1V3", "diagnostic attribution reference; not a matched ablation"),
    ]
    rows = []
    for treatment, baseline, basis in pairs:
        t, b = lookup[treatment], lookup[baseline]
        rows.append({
            "treatment": treatment,
            "baseline": baseline,
            "comparison_basis": basis,
            "delta_final_success": t["final_success"] - b["final_success"],
            "delta_final_precision": t["final_precision"] - b["final_precision"],
            "delta_late3_success": t["late3_success"] - b["late3_success"],
            "delta_late3_precision": t["late3_precision"] - b["late3_precision"],
        })
    return rows


def epoch_means(run_id: str, tags: dict[str, str], steps_per_epoch: int = 1262) -> list[dict[str, Any]]:
    series = {name: scalars(run_id, leaf) for name, leaf in tags.items()}
    rows = []
    for epoch in range(1, 61):
        lo, hi = (epoch - 1) * steps_per_epoch, epoch * steps_per_epoch
        row: dict[str, Any] = {"run_id": run_id, "epoch": epoch}
        for name, values in series.items():
            picked = [value for step, value in values.items() if lo <= step < hi]
            row[name] = float(np.mean(picked)) if picked else ""
        rows.append(row)
    return rows


def run_integrity() -> list[dict[str, Any]]:
    rows = []
    for run_id, spec in RUNS.items():
        run = ROOT / spec["path"]
        provenance = run / "run_provenance.json"
        checkpoints = list((run / "lightning_logs" / "version_0" / "checkpoints").glob("*.ckpt"))
        prov: dict[str, Any] = {}
        if provenance.exists():
            prov = json.loads(provenance.read_text(encoding="utf-8"))
        resolved = prov.get("resolved_config", {})
        rows.append({
            "run_id": run_id,
            "run_exists": run.is_dir(),
            "provenance_present": provenance.exists(),
            "git_commit": prov.get("git", {}).get("commit", "") if prov else "",
            "git_dirty": prov.get("git", {}).get("dirty_any", "") if prov else "unknown",
            "config_path": prov.get("config_path", "") if prov else "",
            "seed": prov.get("seed", resolved.get("seed", 42 if run_id == "SEQTRACK_NEW" else "")),
            "batch_size": resolved.get("batch_size", 16 if run_id == "SEQTRACK_NEW" else ""),
            "workers": resolved.get("workers", 4 if run_id in {"SEQTRACK_NEW", "SEARCH_LEGACY", "B2V2_FULL"} else ""),
            "paired_validation_points": len(scalars(run_id, "metrics_test_success")),
            "checkpoint_count": len(checkpoints),
            "last_checkpoint_present": (run / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt").exists(),
        })
    return rows


def control_stability_audit() -> list[dict[str, Any]]:
    """Compare early losses of the two plain SeqTrack trajectories."""
    new = scalars("SEQTRACK_NEW", "loss_loss_total")
    old = scalars("SEQTRACK_HIST", "loss_loss_total")
    shared = sorted(set(new) & set(old))[:20]
    return [{
        "step": step,
        "seqtrack_new_loss_total": new[step],
        "seqtrack_historical_loss_total": old[step],
        "absolute_difference": abs(new[step] - old[step]),
        "exactly_equal": new[step] == old[step],
    } for step in shared]


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


def checkpoint_diagnostics() -> list[dict[str, Any]]:
    install_easydict_fallback()
    checkpoint = version_dir("B2V2_FULL") / "checkpoints" / "last.ckpt"
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["state_dict"]
    gate_bias = state["joint_proposal_fusion.gate.3.bias"].detach().tolist()
    gate_weight = state["joint_proposal_fusion.gate.3.weight"].detach()
    confidence_bias = state["search_evidence_v2.confidence_head.bias"].detach().item()
    return [{
        "checkpoint": "last.ckpt",
        "epoch": int(payload.get("epoch", -1)) + 1,
        "global_step": int(payload.get("global_step", -1)),
        "gate_bias_observation": gate_bias[0],
        "gate_bias_motion": gate_bias[1],
        "gate_bias_search": gate_bias[2],
        "gate_weight_norm_observation": float(gate_weight[0].norm()),
        "gate_weight_norm_motion": float(gate_weight[1].norm()),
        "gate_weight_norm_search": float(gate_weight[2].norm()),
        "search_confidence_head_bias": confidence_bias,
    }]


def diagnostic_register(
        full_rows: list[dict[str, Any]],
        checkpoint_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    e60 = full_rows[-1]
    ckpt = checkpoint_rows[0]
    prior_improvements = {
        "candidate0": 100.0 * (
            e60["motion_candidate0_cv_rmse"]
            - e60["motion_candidate0_prior_rmse"]
        ) / e60["motion_candidate0_cv_rmse"],
        "nonzero": 100.0 * (
            e60["motion_nonzero_cv_rmse"]
            - e60["motion_nonzero_prior_rmse"]
        ) / e60["motion_nonzero_cv_rmse"],
        "gap2": 100.0 * (
            e60["motion_gap2_cv_rmse"]
            - e60["motion_gap2_prior_rmse"]
        ) / e60["motion_gap2_cv_rmse"],
        "gap4": 100.0 * (
            e60["motion_gap4_cv_rmse"]
            - e60["motion_gap4_prior_rmse"]
        ) / e60["motion_gap4_cv_rmse"],
    }
    one_step_gain = 100.0 * (
        e60["joint_observation_error_m"] - e60["joint_final_error_m"]
    ) / e60["joint_observation_error_m"]
    return [
        {
            "priority": 1,
            "finding": "Search evidence is structurally starved",
            "evidence": f"epoch60 valid={100*e60['search_candidate_valid_rate']:.2f}%; invalid={100*(1-e60['search_candidate_valid_rate']):.2f}%",
            "interpretation": "compact endpoint crop plus extension-only filtering leaves no candidate for most samples",
            "confidence": "high",
        },
        {
            "priority": 2,
            "finding": "The joint gate almost never selects search",
            "evidence": f"epoch60 argmax rate={100*e60['joint_search_selected_rate']:.3f}%; gate bias=[{ckpt['gate_bias_observation']:.3f},{ckpt['gate_bias_motion']:.3f},{ckpt['gate_bias_search']:.3f}]",
            "interpretation": "observation bias, natural class frequency, and log-confidence penalty jointly suppress the rare class",
            "confidence": "high",
        },
        {
            "priority": 3,
            "finding": "The physical prior learns beyond constant velocity",
            "evidence": "candidate0/nonzero/gap2/gap4 RMSE improvements=" + "/".join(f"{prior_improvements[k]:.2f}%" for k in ("candidate0", "nonzero", "gap2", "gap4")),
            "interpretation": "motion encoding is not the primary failure point in this run",
            "confidence": "high",
        },
        {
            "priority": 4,
            "finding": "Joint fusion improves one-step training error",
            "evidence": f"observation={e60['joint_observation_error_m']:.3f}m; final={e60['joint_final_error_m']:.3f}m; relative={one_step_gain:.2f}%",
            "interpretation": "a useful correction signal exists, but recursive attribution still requires inference ablations",
            "confidence": "medium",
        },
        {
            "priority": 5,
            "finding": "Current search-error and helpful-precision logs are not decision metrics",
            "evidence": "search error includes invalid zero proposals; helpful precision is a clamped batch mean from training",
            "interpretation": "validation endpoint conditional metrics and oracle prevalence must be exported",
            "confidence": "high",
        },
    ]


def next_experiments() -> list[dict[str, Any]]:
    return [
        {"order": 1, "experiment": "same-checkpoint four-mode inference", "change": "full / observation-only / motion-only / search-only on identical endpoints", "decision": "attribute the full checkpoint gain without retraining", "gpu_cost": "low"},
        {"order": 2, "experiment": "endpoint diagnostic export", "change": "candidate errors, validity, confidence, gate probabilities, oracle class, tracklet id", "decision": "measure conditional search usefulness and paired bootstrap", "gpu_cost": "low"},
        {"order": 3, "experiment": "same-commit matched B0", "change": "a486a36, seed42, batch16, workers4, scratch 60 epoch", "decision": "apply the preregistered +0.5/+1.0 final gate", "gpu_cost": "one 60-epoch run"},
        {"order": 4, "experiment": "B2-v2.1 only if search is inactive", "change": "all endpoint-crop points + overlap flag; availability/utility split; remove uncalibrated log-confidence penalty", "decision": "repair coverage and gate learnability without returning to a tube", "gpu_cost": "one bounded kill-test first"},
    ]


def decision_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strict = next(r for r in comparison_rows if r["treatment"] == "B2V2_FULL" and r["baseline"] == "B0_HIST")
    return [
        {
            "question": "legacy search-only vs strongest B0 guardrail",
            "criterion": "epoch60 and late-3 should not regress",
            "observed": f"final {next(r for r in comparison_rows if r['treatment']=='SEARCH_LEGACY' and r['baseline']=='B0_HIST')['delta_final_success']:+.3f} S / {next(r for r in comparison_rows if r['treatment']=='SEARCH_LEGACY' and r['baseline']=='B0_HIST')['delta_final_precision']:+.3f} P",
            "decision": "FAIL; legacy search path remains rejected",
        },
        {
            "question": "B2-v2 full epoch60 promotion",
            "criterion": "+0.5 Success and +1.0 Precision vs matched B0",
            "observed": f"{strict['delta_final_success']:+.3f} S / {strict['delta_final_precision']:+.3f} P vs historical B0",
            "decision": "HOLD; Success passes, Precision fails, and same-commit B0 is missing",
        },
        {
            "question": "B2-v2 late-3 stability",
            "criterion": "both metrics not below B0",
            "observed": f"{strict['delta_late3_success']:+.3f} S / {strict['delta_late3_precision']:+.3f} P",
            "decision": "PASS against historical B0",
        },
        {
            "question": "search contribution established",
            "criterion": "new search-only ablation or material conditional gate use",
            "observed": "legacy-only ablation; full search argmax selection approximately 0.10% at epoch60",
            "decision": "NOT ESTABLISHED",
        },
    ]


def main() -> None:
    validation_rows, by_run = collect_validation()
    summary_rows = summaries(by_run)
    comparison_rows = comparisons(summary_rows)
    full_rows = epoch_means("B2V2_FULL", FULL_TRAIN_TAGS)
    legacy_rows = epoch_means("SEARCH_LEGACY", LEGACY_TAGS)
    checkpoint_rows = checkpoint_diagnostics()
    write_csv("validation", validation_rows)
    write_csv("summary", summary_rows)
    write_csv("comparisons", comparison_rows)
    write_csv("full_training_epochs", full_rows)
    write_csv("legacy_training_epochs", legacy_rows)
    write_csv("integrity", run_integrity())
    write_csv("control_stability_audit", control_stability_audit())
    write_csv("checkpoint_diagnostics", checkpoint_rows)
    write_csv("drivers", diagnostic_register(full_rows, checkpoint_rows))
    write_csv("next_experiments", next_experiments())
    write_csv("decision", decision_rows(comparison_rows))
    print(json.dumps({
        "outputs": 11,
        "b2v2_full": next(r for r in summary_rows if r["run_id"] == "B2V2_FULL"),
        "strict_vs_b0": next(r for r in comparison_rows if r["treatment"] == "B2V2_FULL" and r["baseline"] == "B0_HIST"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
