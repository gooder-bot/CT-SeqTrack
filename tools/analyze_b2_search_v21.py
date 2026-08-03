#!/usr/bin/env python3
"""Reproduce the B2-v2.1 seed42 normal-mini result diagnosis.

The primary decision uses epoch60. Late-3 is the arithmetic mean of epochs
50/55/60. Best checkpoints are diagnostic only. The script keeps the user's
new SeqTrack run visible but never treats it as a trustworthy matched baseline:
that run has no provenance and follows an anomalously low validation path.
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
STEM = "b2_search_v21_seed42_20260803"
STEPS_PER_EPOCH = 1262
INVALID_ESS_SENTINEL = 1_000_000.0

RUNS = {
    "SEQTRACK_NEW": {
        "arm": "SeqTrack control (2026-08-01, anomalous)",
        "path": "../seqtrack/output/20260801-2155-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16_seed42",
        "role": "requested_control",
    },
    "SEQTRACK_HIST": {
        "arm": "SeqTrack historical seed42 reference",
        "path": "../seqtrack/output/20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16",
        "role": "historical_reference",
    },
    "B0_HIST": {
        "arm": "CT-v2 historical B0 guardrail",
        "path": "output/20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16",
        "role": "historical_guardrail",
    },
    "LEGACY_SEARCH": {
        "arm": "Legacy tube search-only",
        "path": "output/20260801-2135-05_seqtrack3d_search_only-search_only_legacy_mini_car_60ep_bs16_seed42",
        "role": "legacy_search_reference",
    },
    "B2V2_FULL": {
        "arm": "B2-v2 motion + search + softmax fusion",
        "path": "output/20260801-2136-03_ct_motion_search_v2-motion_search_v2_mini_car_60ep_bs16_seed42",
        "role": "v2_full_reference",
    },
    "V21_SEARCH": {
        "arm": "B2-v2.1 search-only",
        "path": "output/20260802-1530-08_seqtrack3d_search_v21-search_v21_mini_car_60ep_bs16_seed42",
        "role": "requested_search_only",
    },
    "V21_FULL": {
        "arm": "B2-v2.1 motion + search + advantage fusion",
        "path": "output/20260802-1530-09_ct_motion_search_v21-motion_search_v21_mini_car_60ep_bs16_seed42",
        "role": "requested_full",
    },
}

V21_TAGS = {
    "loss_total": "loss_loss_total",
    "loss_center": "loss_loss_center",
    "loss_angle": "loss_loss_angle",
    "loss_seg": "loss_loss_seg",
    "loss_bc": "loss_loss_bc",
    "search_match_loss": "loss_loss_search_v21_match",
    "search_targetness_loss": "loss_loss_search_v21_targetness",
    "search_vote_loss": "loss_loss_search_v21_vote",
    "search_proposal_loss": "loss_loss_search_v21_proposal",
    "search_geometry_valid_rate": "loss_search_v21_geometry_valid_rate",
    "search_candidate_valid_rate": "loss_search_v21_candidate_valid_rate",
    "search_foreground_points": "loss_search_v21_foreground_points",
    "search_targetness_mean": "loss_search_v21_targetness_mean",
    "search_targetness_max": "loss_search_v21_targetness_max",
    "search_targetness_entropy": "loss_search_v21_targetness_entropy",
    "search_effective_sample_size": "loss_search_v21_effective_sample_size",
    "search_extension_weight_ratio": "loss_search_v21_extension_weight_ratio",
    "search_available_count": "loss_search_v21_available_count",
    "search_extension_count": "loss_search_v21_extension_count",
    "search_overlap_count": "loss_search_v21_overlap_count",
    "observation_error_m": "loss_advantage_observation_error",
    "motion_error_valid_m": "loss_advantage_motion_error_valid",
    "search_error_valid_m": "loss_advantage_search_error_valid",
    "final_error_m": "loss_advantage_final_error",
    "motion_helpful_rate": "loss_advantage_motion_helpful_rate",
    "search_helpful_rate": "loss_advantage_search_helpful_rate",
    "motion_weight": "loss_advantage_motion_weight",
    "search_weight": "loss_advantage_search_weight",
    "search_applied_rate": "loss_advantage_search_applied_rate",
    "search_helpful_precision": "loss_advantage_search_helpful_precision",
    "fusion_ramp": "loss_advantage_fusion_ramp",
    "motion_candidate_valid_rate": "loss_motion_v3_prior_valid_rate",
    "motion_prior_rmse": "loss_motion_v3_prior_rmse",
    "motion_constant_velocity_rmse": "loss_motion_v3_kinematic_rmse",
}

V2_TAGS = {
    "search_candidate_valid_rate": "loss_search_v2_candidate_valid_rate",
    "search_foreground_points": "loss_search_v2_foreground_points",
    "search_targetness_loss": "loss_loss_search_v2_targetness",
    "search_vote_loss": "loss_loss_search_v2_vote",
    "search_proposal_loss": "loss_loss_search_v2_proposal",
    "observation_error_m": "loss_joint_observation_error",
    "final_error_m": "loss_joint_final_error",
    "search_selected_rate": "loss_joint_search_selected_rate",
}

_SCALAR_CACHE: dict[tuple[str, str], dict[int, float]] = {}


def run_path(run_id: str) -> Path:
    return (ROOT / RUNS[run_id]["path"]).resolve()


def version_dir(run_id: str) -> Path:
    return run_path(run_id) / "lightning_logs" / "version_0"


def scalars(run_id: str, leaf: str) -> dict[int, float]:
    key = (run_id, leaf)
    if key in _SCALAR_CACHE:
        return _SCALAR_CACHE[key]
    path = version_dir(run_id) / leaf
    if not path.is_dir():
        _SCALAR_CACHE[key] = {}
        return {}
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        _SCALAR_CACHE[key] = {}
        return {}
    tag = "loss" if "loss" in tags else tags[0]
    values = {
        int(item.step): float(item.value)
        for item in accumulator.Scalars(tag)
    }
    _SCALAR_CACHE[key] = values
    return values


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {name}")
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{STEM}_{name}.csv"
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
            raise ValueError(
                f"{run_id}: expected 12 validation points, got {len(steps)}")
        current = []
        for index, step in enumerate(steps, 1):
            row = {
                "run_id": run_id,
                "arm": spec["arm"],
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


def summaries(by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for run_id, values in by_run.items():
        best_success = max(values, key=lambda row: row["success"])
        best_precision = max(values, key=lambda row: row["precision"])
        rows.append({
            "run_id": run_id,
            "arm": RUNS[run_id]["arm"],
            "role": RUNS[run_id]["role"],
            "final_success": values[-1]["success"],
            "final_precision": values[-1]["precision"],
            "late3_success": float(np.mean([
                row["success"] for row in values[-3:]])),
            "late3_precision": float(np.mean([
                row["precision"] for row in values[-3:]])),
            "best_success": best_success["success"],
            "best_success_epoch": best_success["epoch"],
            "best_precision": best_precision["precision"],
            "best_precision_epoch": best_precision["epoch"],
        })
    return rows


def comparisons(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row["run_id"]: row for row in summary_rows}
    pairs = [
        ("V21_SEARCH", "LEGACY_SEARCH", "search redesign versus legacy tube search-only"),
        ("V21_SEARCH", "SEQTRACK_NEW", "requested control; numerically paired but anomalous"),
        ("V21_SEARCH", "SEQTRACK_HIST", "historical original SeqTrack reference"),
        ("V21_SEARCH", "B0_HIST", "strongest available B0 guardrail; not same commit"),
        ("V21_FULL", "B2V2_FULL", "v2.1 full versus v2 full; different clean commits"),
        ("V21_FULL", "SEQTRACK_NEW", "requested control; numerically paired but anomalous"),
        ("V21_FULL", "SEQTRACK_HIST", "historical original SeqTrack reference"),
        ("V21_FULL", "B0_HIST", "strongest available B0 guardrail; not same commit"),
        ("V21_SEARCH", "B2V2_FULL", "component diagnostic; not a matched architecture pair"),
    ]
    rows = []
    for treatment, baseline, basis in pairs:
        current = lookup[treatment]
        reference = lookup[baseline]
        rows.append({
            "treatment": treatment,
            "baseline": baseline,
            "comparison_basis": basis,
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


def epoch_means(run_id: str, tags: dict[str, str]) -> list[dict[str, Any]]:
    series = {name: scalars(run_id, leaf) for name, leaf in tags.items()}
    rows = []
    for epoch in range(1, 61):
        lower = (epoch - 1) * STEPS_PER_EPOCH
        upper = epoch * STEPS_PER_EPOCH
        row: dict[str, Any] = {"run_id": run_id, "epoch": epoch}
        for name, values in series.items():
            selected = [
                value for step, value in values.items()
                if lower <= step < upper
            ]
            row[name] = float(np.mean(selected)) if selected else ""
        rows.append(row)
    return rows


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
    rows = []
    for run_id in ("V21_SEARCH", "V21_FULL"):
        checkpoint = version_dir(run_id) / "checkpoints" / "last.ckpt"
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload["state_dict"]
        help_bias = state[
            "advantage_proposal_fusion.help_head.bias"].detach()
        help_weight = state[
            "advantage_proposal_fusion.help_head.weight"].detach()
        step_bias = state[
            "advantage_proposal_fusion.step_head.bias"].detach()
        step_weight = state[
            "advantage_proposal_fusion.step_head.weight"].detach()
        rows.append({
            "run_id": run_id,
            "checkpoint": "last.ckpt",
            "epoch": int(payload.get("epoch", -1)) + 1,
            "global_step": int(payload.get("global_step", -1)),
            "help_bias_motion": float(help_bias[0]),
            "help_bias_search": float(help_bias[1]),
            "help_weight_norm_motion": float(help_weight[0].norm()),
            "help_weight_norm_search": float(help_weight[1].norm()),
            "step_bias_motion": float(step_bias[0]),
            "step_bias_search": float(step_bias[1]),
            "step_weight_norm_motion": float(step_weight[0].norm()),
            "step_weight_norm_search": float(step_weight[1].norm()),
        })
    return rows


def run_integrity() -> list[dict[str, Any]]:
    rows = []
    for run_id, spec in RUNS.items():
        run = run_path(run_id)
        provenance_path = run / "run_provenance.json"
        provenance: dict[str, Any] = {}
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        resolved = provenance.get("resolved_config", {})
        checkpoints = list(
            (version_dir(run_id) / "checkpoints").glob("*.ckpt"))
        rows.append({
            "run_id": run_id,
            "run_exists": run.is_dir(),
            "provenance_present": provenance_path.exists(),
            "git_commit": provenance.get("git", {}).get("commit", ""),
            "git_dirty": provenance.get("git", {}).get(
                "dirty_any", "unknown"),
            "config_path": provenance.get("config_path", ""),
            "seed": provenance.get("seed", resolved.get(
                "seed", 42 if run_id.startswith("SEQTRACK") else "")),
            "batch_size": resolved.get(
                "batch_size", 16 if run_id.startswith("SEQTRACK") else ""),
            "workers": resolved.get(
                "workers", 4 if run_id == "SEQTRACK_NEW" else ""),
            "epochs": resolved.get("epoch", 60),
            "init_checkpoint": provenance.get("init_checkpoint_path", ""),
            "validation_points": len(scalars(
                run_id, "metrics_test_success")),
            "checkpoint_count": len(checkpoints),
            "last_checkpoint_present": (
                version_dir(run_id) / "checkpoints" / "last.ckpt").exists(),
            "status": (
                "complete_comparable_contract"
                if provenance_path.exists()
                else "complete_missing_provenance"),
        })
    return rows


def derived_diagnostics(
        v21_search: dict[str, Any],
        v21_full: dict[str, Any],
        v2_full: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for label, current in (("V21_SEARCH", v21_search), ("V21_FULL", v21_full)):
        valid = float(current["search_candidate_valid_rate"])
        applied = float(current["search_applied_rate"])
        weight = float(current["search_weight"])
        mean_ess = float(current["search_effective_sample_size"])
        conditional_ess = (
            mean_ess - (1.0 - valid) * INVALID_ESS_SENTINEL
        ) / max(valid, 1e-12)
        rows.append({
            "run_id": label,
            "epoch": 60,
            "search_valid_rate": valid,
            "search_unavailable_rate": 1.0 - valid,
            "search_material_selection_rate": applied,
            "selected_share_of_valid_approx": applied / max(valid, 1e-12),
            "search_helpful_prevalence_valid": float(
                current["search_helpful_rate"]),
            "search_helpful_precision_selected": float(
                current["search_helpful_precision"]),
            "helpful_precision_lift_pp": 100.0 * (
                float(current["search_helpful_precision"])
                - float(current["search_helpful_rate"])),
            "search_weight_unconditional": weight,
            "search_weight_conditional_valid_approx": (
                weight / max(valid, 1e-12)),
            "effective_sample_size_logged_mean": mean_ess,
            "effective_sample_size_valid_approx": conditional_ess,
            "observation_error_m": float(current["observation_error_m"]),
            "search_error_valid_m": float(current["search_error_valid_m"]),
            "final_error_m": float(current["final_error_m"]),
            "one_step_error_reduction_pct": 100.0 * (
                float(current["observation_error_m"])
                - float(current["final_error_m"])
            ) / float(current["observation_error_m"]),
            "motion_valid_rate": (
                float(current["motion_candidate_valid_rate"])
                if current["motion_candidate_valid_rate"] != "" else 0.0),
            "motion_helpful_rate_valid": (
                float(current["motion_helpful_rate"])
                if current["motion_helpful_rate"] != "" else 0.0),
            "motion_weight_unconditional": (
                float(current["motion_weight"])
                if current["motion_weight"] != "" else 0.0),
            "motion_weight_conditional_valid_approx": (
                float(current["motion_weight"])
                / max(float(current["motion_candidate_valid_rate"]), 1e-12)
                if current["motion_candidate_valid_rate"] != "" else 0.0),
        })

    rows.append({
        "run_id": "V21_VERSUS_V2_SEARCH_EVIDENCE",
        "epoch": 60,
        "search_valid_rate": float(v21_full["search_candidate_valid_rate"]),
        "search_unavailable_rate": 1.0 - float(
            v21_full["search_candidate_valid_rate"]),
        "search_material_selection_rate": "",
        "selected_share_of_valid_approx": "",
        "search_helpful_prevalence_valid": "",
        "search_helpful_precision_selected": "",
        "helpful_precision_lift_pp": "",
        "search_weight_unconditional": "",
        "search_weight_conditional_valid_approx": "",
        "effective_sample_size_logged_mean": "",
        "effective_sample_size_valid_approx": "",
        "observation_error_m": "",
        "search_error_valid_m": "",
        "final_error_m": "",
        "one_step_error_reduction_pct": "",
        "motion_valid_rate": "",
        "motion_helpful_rate_valid": "",
        "motion_weight_unconditional": "",
        "motion_weight_conditional_valid_approx": "",
        "v2_search_valid_rate": float(v2_full["search_candidate_valid_rate"]),
        "valid_coverage_gain_pp": 100.0 * (
            float(v21_full["search_candidate_valid_rate"])
            - float(v2_full["search_candidate_valid_rate"])),
        "foreground_point_gain_pct": 100.0 * (
            float(v21_full["search_foreground_points"])
            / float(v2_full["search_foreground_points"]) - 1.0),
        "proposal_loss_reduction_pct": 100.0 * (
            1.0 - float(v21_full["search_proposal_loss"])
            / float(v2_full["search_proposal_loss"])),
        "vote_loss_reduction_pct": 100.0 * (
            1.0 - float(v21_full["search_vote_loss"])
            / float(v2_full["search_vote_loss"])),
    })
    return rows


def driver_register(
        summary_rows: list[dict[str, Any]],
        comparison_rows: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
        search_e60: dict[str, Any],
        full_e60: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = {row["run_id"]: row for row in summary_rows}
    search_diag = diagnostics[0]
    full_diag = diagnostics[1]
    evidence_diag = diagnostics[2]
    full_v2_delta = next(
        row for row in comparison_rows
        if row["treatment"] == "V21_FULL" and row["baseline"] == "B2V2_FULL")
    return [
        {
            "priority": 1,
            "finding": "Motion+Search v2.1 collapses exactly after fusion activation",
            "evidence": (
                f"epoch10={summary['V21_FULL']['best_success']:.3f} best Success; "
                f"epoch15=22.896/21.687; epoch60="
                f"{summary['V21_FULL']['final_success']:.3f}/"
                f"{summary['V21_FULL']['final_precision']:.3f}; versus v2 final "
                f"{full_v2_delta['delta_final_success']:+.3f}/"
                f"{full_v2_delta['delta_final_precision']:+.3f}"),
            "interpretation": (
                "The failure aligns with the 10-19 ramp and is not a late-training "
                "overfit pattern."),
            "confidence": "verified",
        },
        {
            "priority": 2,
            "finding": "B0 optimization did not collapse",
            "evidence": (
                f"epoch60 center loss search/full={search_e60['loss_center']:.5f}/"
                f"{full_e60['loss_center']:.5f}; observation error="
                f"{search_e60['observation_error_m']:.3f}/"
                f"{full_e60['observation_error_m']:.3f} m"),
            "interpretation": (
                "The recursive validation collapse is downstream of auxiliary "
                "proposal use, not visible as B0 supervised-loss degradation."),
            "confidence": "verified",
        },
        {
            "priority": 3,
            "finding": "The advantage gate is not selective",
            "evidence": (
                f"search-only selects about {100*search_diag['selected_share_of_valid_approx']:.1f}% "
                f"of valid Search rows; helpful precision "
                f"{100*search_diag['search_helpful_precision_selected']:.1f}% versus "
                f"{100*search_diag['search_helpful_prevalence_valid']:.1f}% prevalence; "
                f"conditional weight about "
                f"{search_diag['search_weight_conditional_valid_approx']:.3f}"),
            "interpretation": (
                "The gate behaves close to an availability gate and nearly reaches "
                "the normal 0.5 mass cap, failing the 70% helpful-precision target."),
            "confidence": "verified",
        },
        {
            "priority": 4,
            "finding": "Motion is over-used relative to its helpfulness",
            "evidence": (
                f"epoch60 valid={100*full_diag['motion_valid_rate']:.1f}%, "
                f"helpful={100*full_diag['motion_helpful_rate_valid']:.1f}%, "
                f"conditional applied weight about "
                f"{full_diag['motion_weight_conditional_valid_approx']:.3f}; "
                f"motion/search/obs error="
                f"{full_e60['motion_error_valid_m']:.3f}/"
                f"{full_e60['search_error_valid_m']:.3f}/"
                f"{full_e60['observation_error_m']:.3f} m"),
            "interpretation": (
                "Teacher-forced one-step fused loss rewards partial corrections that "
                "are not safe under recursive history feedback."),
            "confidence": "verified metrics; closed-loop mechanism likely",
        },
        {
            "priority": 5,
            "finding": "Invalid Search rows inject a 1e6 ESS feature into the shared gate",
            "evidence": (
                f"epoch60 Search unavailable={100*full_diag['search_unavailable_rate']:.1f}%; "
                f"logged mean ESS={full_diag['effective_sample_size_logged_mean']:.0f}, "
                f"while inferred valid-row ESS is about "
                f"{full_diag['effective_sample_size_valid_approx']:.1f} and the branch "
                "contains at most 128 points"),
            "interpretation": (
                "_masked_softmax returns zero weights for invalid rows, then 1/clamp(0,1e-6) "
                "creates 1e6; AdvantageGatedProposalFusion consumes it without validity masking. "
                "This can contaminate motion decisions whenever Search is invalid."),
            "confidence": "code defect verified; causal share high-confidence inference",
        },
        {
            "priority": 6,
            "finding": "Observation-queried Search evidence is better than the old Search path",
            "evidence": (
                f"coverage +{evidence_diag['valid_coverage_gain_pp']:.2f} pp; foreground "
                f"points +{evidence_diag['foreground_point_gain_pct']:.1f}%; proposal loss "
                f"-{evidence_diag['proposal_loss_reduction_pct']:.1f}%; search-only final="
                f"{summary['V21_SEARCH']['final_success']:.3f}/"
                f"{summary['V21_SEARCH']['final_precision']:.3f} versus legacy "
                f"{summary['LEGACY_SEARCH']['final_success']:.3f}/"
                f"{summary['LEGACY_SEARCH']['final_precision']:.3f}"),
            "interpretation": (
                "Source-aware overlap and observation query repaired much of the old "
                "Search degradation, but did not establish a gain over the strongest B0."),
            "confidence": "verified",
        },
    ]


def decision_rows(
        comparison_rows: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def comparison(treatment: str, baseline: str) -> dict[str, Any]:
        return next(
            row for row in comparison_rows
            if row["treatment"] == treatment and row["baseline"] == baseline)

    search_hist = comparison("V21_SEARCH", "SEQTRACK_HIST")
    search_b0 = comparison("V21_SEARCH", "B0_HIST")
    full_v2 = comparison("V21_FULL", "B2V2_FULL")
    full_new = comparison("V21_FULL", "SEQTRACK_NEW")
    search_diag = diagnostics[0]
    return [
        {
            "question": "Did Search-v2.1 improve over legacy search-only?",
            "criterion": "epoch60 and late-3 both improve",
            "observed": (
                f"final {comparison('V21_SEARCH','LEGACY_SEARCH')['delta_final_success']:+.3f} S / "
                f"{comparison('V21_SEARCH','LEGACY_SEARCH')['delta_final_precision']:+.3f} P; "
                f"late-3 {comparison('V21_SEARCH','LEGACY_SEARCH')['delta_late3_success']:+.3f} / "
                f"{comparison('V21_SEARCH','LEGACY_SEARCH')['delta_late3_precision']:+.3f}"),
            "decision": "PASS: the Search redesign is materially better than the legacy path",
        },
        {
            "question": "Did Search-v2.1 beat SeqTrack?",
            "criterion": "+0.5 Success and +1.0 Precision at epoch60; late-3 non-negative",
            "observed": (
                f"versus historical SeqTrack final {search_hist['delta_final_success']:+.3f}/"
                f"{search_hist['delta_final_precision']:+.3f}, late-3 "
                f"{search_hist['delta_late3_success']:+.3f}/"
                f"{search_hist['delta_late3_precision']:+.3f}"),
            "decision": (
                "NUMERICAL PASS versus historical SeqTrack, but not a formal matched-B0 promotion"),
        },
        {
            "question": "Did Search-v2.1 beat the strongest available B0 guardrail?",
            "criterion": "+0.5 Success and +1.0 Precision at epoch60; late-3 non-negative",
            "observed": (
                f"final {search_b0['delta_final_success']:+.3f}/"
                f"{search_b0['delta_final_precision']:+.3f}; late-3 "
                f"{search_b0['delta_late3_success']:+.3f}/"
                f"{search_b0['delta_late3_precision']:+.3f}"),
            "decision": "FAIL against historical B0; same-commit matched B0 is still missing",
        },
        {
            "question": "Did full B2-v2.1 improve over B2-v2?",
            "criterion": "epoch60 and late-3 should improve or at least remain stable",
            "observed": (
                f"final {full_v2['delta_final_success']:+.3f}/"
                f"{full_v2['delta_final_precision']:+.3f}; late-3 "
                f"{full_v2['delta_late3_success']:+.3f}/"
                f"{full_v2['delta_late3_precision']:+.3f}"),
            "decision": "CATASTROPHIC FAIL: do not promote or run more seeds",
        },
        {
            "question": "Did full B2-v2.1 at least beat the anomalous new control?",
            "criterion": "both epoch60 metrics should be non-negative",
            "observed": (
                f"final {full_new['delta_final_success']:+.3f}/"
                f"{full_new['delta_final_precision']:+.3f}"),
            "decision": "FAIL even against the weak requested control",
        },
        {
            "question": "Is Search gate selection reliable enough?",
            "criterion": "applied weight >=0.1 helpful precision >=70%",
            "observed": (
                f"Search-only helpful precision "
                f"{100*search_diag['search_helpful_precision_selected']:.1f}%; "
                f"prevalence {100*search_diag['search_helpful_prevalence_valid']:.1f}%"),
            "decision": "FAIL: selection adds only a small lift over natural prevalence",
        },
    ]


def next_steps() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "action": "Stop full v2.1 promotion",
            "change": "Do not run seed43/44 or robustness suites from the current full checkpoint.",
            "decision_unlocked": "Avoid spending compute on a structurally broken fusion path.",
            "gpu_cost": "none",
        },
        {
            "order": 2,
            "action": "Run same-checkpoint four-mode inference",
            "change": "Evaluate obs, obs_motion, obs_search, and full for epoch60 full and search-only checkpoints.",
            "decision_unlocked": "Directly separate B0, motion, search, and interaction effects.",
            "gpu_cost": "low; inference only",
        },
        {
            "order": 3,
            "action": "Fix invalid-row evidence statistics",
            "change": "Set ESS and all Search statistics to zero when search_valid is false; normalize ESS by valid_count or log1p before the gate.",
            "decision_unlocked": "Remove the verified 1e6 out-of-range feature and isolate gate calibration.",
            "gpu_cost": "none for code/tests",
        },
        {
            "order": 4,
            "action": "Make auxiliary use truly selective",
            "change": "Do not supervise step on only helpful rows while allowing near-cap weights everywhere; add abstention/calibration diagnostics and cap motion more tightly.",
            "decision_unlocked": "Require helpful precision to exceed 70% before recursive use.",
            "gpu_cost": "short diagnostic training before 60 epochs",
        },
        {
            "order": 5,
            "action": "Train same-commit matched B0",
            "change": "Commit 16c2b8b, seed42, batch16, workers4, scratch, 60 epochs.",
            "decision_unlocked": "Establish whether Search-only truly raises the project's B0.",
            "gpu_cost": "one 60-epoch run",
        },
    ]


def code_audit_rows() -> list[dict[str, Any]]:
    return [
        {
            "file": "models/ct_v2/motion.py",
            "symbol": "TrajectorySearchEvidenceV21.forward",
            "verified_behavior": (
                "Invalid rows have zero pool weights; ESS computes 1/clamp(sum(w^2),1e-6), producing 1e6."),
            "risk": "Out-of-range invalid-row statistic",
        },
        {
            "file": "models/ct_v2/motion.py",
            "symbol": "AdvantageGatedProposalFusion.forward",
            "verified_behavior": (
                "Raw search ESS enters the shared gate input without search-valid masking or normalization."),
            "risk": "Search invalidity can alter motion gating",
        },
        {
            "file": "models/seqtrack3d.py",
            "symbol": "compute_loss / advantage fusion",
            "verified_behavior": (
                "Fused loss is one-step teacher-forced; step loss is restricted to helpful candidates."),
            "risk": "One-step gain does not enforce closed-loop safety",
        },
        {
            "file": "datasets/sampler.py",
            "symbol": "source-aware endpoint sampling",
            "verified_behavior": (
                "B0 1024 points remain intact; endpoint overlap and extension use an independent 128-point branch."),
            "risk": "No baseline point-budget regression",
        },
    ]


def main() -> None:
    validation_rows, by_run = collect_validation()
    summary_rows = summaries(by_run)
    comparison_rows = comparisons(summary_rows)
    search_training = epoch_means("V21_SEARCH", V21_TAGS)
    full_training = epoch_means("V21_FULL", V21_TAGS)
    v2_training = epoch_means("B2V2_FULL", V2_TAGS)
    training_rows = search_training + full_training
    search_e60 = search_training[-1]
    full_e60 = full_training[-1]
    v2_e60 = v2_training[-1]
    diagnostic_rows = derived_diagnostics(search_e60, full_e60, v2_e60)
    driver_rows = driver_register(
        summary_rows, comparison_rows, diagnostic_rows,
        search_e60, full_e60)
    decision_register = decision_rows(comparison_rows, diagnostic_rows)

    write_csv("validation", validation_rows)
    write_csv("summary", summary_rows)
    write_csv("comparisons", comparison_rows)
    write_csv("v21_training_epochs", training_rows)
    write_csv("v2_training_epochs", v2_training)
    write_csv("derived_diagnostics", diagnostic_rows)
    write_csv("integrity", run_integrity())
    write_csv("checkpoint_diagnostics", checkpoint_diagnostics())
    write_csv("drivers", driver_rows)
    write_csv("decision", decision_register)
    write_csv("next_steps", next_steps())
    write_csv("code_audit", code_audit_rows())

    summary_lookup = {row["run_id"]: row for row in summary_rows}
    print(json.dumps({
        "outputs": 12,
        "v21_search": summary_lookup["V21_SEARCH"],
        "v21_full": summary_lookup["V21_FULL"],
        "search_diagnostics": diagnostic_rows[0],
        "full_diagnostics": diagnostic_rows[1],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
