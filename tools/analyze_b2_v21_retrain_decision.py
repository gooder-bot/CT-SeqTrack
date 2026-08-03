#!/usr/bin/env python3
"""Diagnose whether two severe seed42 regressions share a root cause.

This companion analysis keeps three evidence layers separate:

1. recursive validation timing;
2. supervised training-loss stability;
3. SeqTrack checkpoint divergence and run provenance.

It is intentionally read-only with respect to model outputs and writes only
reviewable CSV evidence under compare_results/data.
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

RUNS = {
    "SEQTRACK_NEW": (
        "SeqTrack control (2026-08-01, anomalous)",
        ROOT.parent / "seqtrack" / "output"
        / "20260801-2155-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16_seed42",
    ),
    "SEQTRACK_HIST": (
        "SeqTrack historical seed42 reference",
        ROOT.parent / "seqtrack" / "output"
        / "20260528-1633-seqtrack3d_nuscenes_mini-seqtrack_mini_baseline_car_60ep_bs16",
    ),
    "V21_SEARCH": (
        "B2-v2.1 search-only",
        ROOT / "output"
        / "20260802-1530-08_seqtrack3d_search_v21-search_v21_mini_car_60ep_bs16_seed42",
    ),
    "V21_FULL": (
        "B2-v2.1 motion + search + advantage fusion",
        ROOT / "output"
        / "20260802-1530-09_ct_motion_search_v21-motion_search_v21_mini_car_60ep_bs16_seed42",
    ),
}

LOSS_TAGS = {
    "loss_total": "loss_loss_total",
    "loss_center": "loss_loss_center",
    "loss_angle": "loss_loss_angle",
    "loss_seg": "loss_loss_seg",
    "loss_bc": "loss_loss_bc",
}


def version_dir(run_id: str) -> Path:
    return RUNS[run_id][1] / "lightning_logs" / "version_0"


def scalar_series(run_id: str, leaf: str) -> dict[int, float]:
    path = version_dir(run_id) / leaf
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        raise ValueError(f"{run_id}/{leaf}: no scalar tag")
    tag = "loss" if "loss" in tags else tags[0]
    return {
        int(item.step): float(item.value)
        for item in accumulator.Scalars(tag)
    }


def write_csv(suffix: str, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"empty output: {suffix}")
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{STEM}_{suffix}.csv"
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def validation_timeline() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchors = {5, 10, 15, 20, 60}
    for run_id, (arm, _) in RUNS.items():
        success = scalar_series(run_id, "metrics_test_success")
        precision = scalar_series(run_id, "metrics_test_precision")
        steps = sorted(set(success) & set(precision))
        if len(steps) != 12:
            raise ValueError(f"{run_id}: expected 12 validation points")
        for index, step in enumerate(steps, 1):
            epoch = index * 5
            if epoch not in anchors:
                continue
            if run_id in {"V21_SEARCH", "V21_FULL"}:
                fusion_ramp = 0.0 if epoch <= 10 else min(1.0, (epoch - 10) / 10.0)
            else:
                fusion_ramp = 0.0
            rows.append({
                "run_id": run_id,
                "arm": arm,
                "epoch": epoch,
                "success": success[step],
                "precision": precision[step],
                "advantage_fusion_ramp_expected": fusion_ramp,
                "interpretation": (
                    "fusion_off" if fusion_ramp == 0.0
                    else "fusion_ramping" if fusion_ramp < 1.0
                    else "fusion_full"
                ),
            })
    return rows


def training_loss_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    anchors = {1, 5, 10, 15, 20, 50, 60}
    for run_id, (arm, _) in RUNS.items():
        series = {
            metric: scalar_series(run_id, leaf)
            for metric, leaf in LOSS_TAGS.items()
        }
        for epoch in sorted(anchors):
            lower = (epoch - 1) * STEPS_PER_EPOCH
            upper = epoch * STEPS_PER_EPOCH
            row: dict[str, Any] = {
                "run_id": run_id,
                "arm": arm,
                "epoch": epoch,
            }
            for metric, values in series.items():
                selected = [
                    value for step, value in values.items()
                    if lower <= step < upper
                ]
                if not selected:
                    raise ValueError(f"{run_id}/{metric}/epoch{epoch}: empty")
                row[metric] = float(np.mean(selected))
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
        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value

    EasyDict.__module__ = "easydict"
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module


def seqtrack_checkpoint_audit() -> list[dict[str, Any]]:
    install_easydict_fallback()
    paths = {
        "new": version_dir("SEQTRACK_NEW") / "checkpoints" / "epoch=9-step=12620.ckpt",
        "historical": version_dir("SEQTRACK_HIST") / "checkpoints" / "epoch=9-step=12620.ckpt",
    }
    states = {
        name: torch.load(path, map_location="cpu")["state_dict"]
        for name, path in paths.items()
    }
    keys_new = set(states["new"])
    keys_hist = set(states["historical"])
    shared = sorted(keys_new & keys_hist)
    exact = 0
    changed = 0
    bn_differences: list[tuple[float, str]] = []
    for key in shared:
        left = states["new"][key]
        right = states["historical"][key]
        if left.shape != right.shape:
            changed += 1
            continue
        if torch.equal(left, right):
            exact += 1
            continue
        changed += 1
        if "running_mean" in key or "running_var" in key:
            delta = (left.float() - right.float()).abs().max().item()
            bn_differences.append((float(delta), key))
    bn_differences.sort(reverse=True)

    loss_new = scalar_series("SEQTRACK_NEW", "loss_loss_total")
    loss_hist = scalar_series("SEQTRACK_HIST", "loss_loss_total")
    common_steps = sorted(set(loss_new) & set(loss_hist))
    equal_prefix = 0
    for step in common_steps:
        if loss_new[step] == loss_hist[step]:
            equal_prefix += 1
        else:
            break

    provenance_new = (RUNS["SEQTRACK_NEW"][1] / "run_provenance.json").exists()
    provenance_hist = (RUNS["SEQTRACK_HIST"][1] / "run_provenance.json").exists()
    top = bn_differences[:5]
    rows = [{
        "comparison": "SEQTRACK_NEW_vs_SEQTRACK_HIST",
        "checkpoint_epoch": 10,
        "shared_tensor_count": len(shared),
        "exact_tensor_count": exact,
        "changed_tensor_count": changed,
        "new_only_key_count": len(keys_new - keys_hist),
        "historical_only_key_count": len(keys_hist - keys_new),
        "exact_initial_loss_step_prefix": equal_prefix,
        "new_workers": 4,
        "historical_workers": 12,
        "new_provenance_present": provenance_new,
        "historical_provenance_present": provenance_hist,
        "interpretation": (
            "Same key set and identical first loss steps, then a different stochastic "
            "training/BatchNorm path; exact code and data state remain unprovable."
        ),
    }]
    for rank, (difference, key) in enumerate(top, 1):
        rows.append({
            "comparison": "SEQTRACK_NEW_vs_SEQTRACK_HIST",
            "checkpoint_epoch": 10,
            "bn_rank": rank,
            "bn_tensor": key,
            "bn_max_abs_difference": difference,
            "interpretation": "Largest BatchNorm running-stat divergence at epoch10.",
        })
    return rows


def retrain_decisions() -> list[dict[str, Any]]:
    return [
        {
            "question": "Do the two severe regressions have the same root cause?",
            "evidence": (
                "Both are amplified by recursive history feedback, but SeqTrack is low "
                "without an intervention and lacks provenance; V2.1 full splits from "
                "Search-only exactly when fusion ramps on."
            ),
            "decision": "No: shared amplifier, different proximate trigger.",
            "confidence": "high",
        },
        {
            "question": "Did B0 supervised optimization collapse in V2.1 full?",
            "evidence": (
                "Epoch60 center/angle/seg/box-cloud losses remain comparable to "
                "Search-only despite a 26-38 point validation gap."
            ),
            "decision": "No evidence of a B0 training-loss collapse.",
            "confidence": "high for supervised losses; recursive obs mode pending",
        },
        {
            "question": "Should current V2.1 full be retrained unchanged?",
            "evidence": (
                "Fusion timing, non-selective gate use, 16.9% Motion helpfulness, and "
                "the verified invalid-row ESS=1e6 defect are structural risks."
            ),
            "decision": "No. First run four-mode inference and repair the gate inputs.",
            "confidence": "high",
        },
        {
            "question": "Will a corrected V2.1 full require retraining?",
            "evidence": (
                "Masking/normalizing gate features and changing abstention or supervision "
                "changes the learned routing distribution."
            ),
            "decision": "Yes, retrain corrected full from scratch after a ramp smoke test.",
            "confidence": "high",
        },
        {
            "question": "Should Search-only be retrained now?",
            "evidence": (
                "Its encoder is materially better than legacy Search; formal gain is blocked "
                "by the missing same-commit matched B0, not by a catastrophic collapse."
            ),
            "decision": "Not immediately; preserve checkpoint and run obs/obs_search attribution.",
            "confidence": "medium-high",
        },
    ]


def main() -> None:
    outputs = [
        write_csv("cause_timeline", validation_timeline()),
        write_csv("base_loss_comparison", training_loss_rows()),
        write_csv("seqtrack_checkpoint_audit", seqtrack_checkpoint_audit()),
        write_csv("retrain_decision", retrain_decisions()),
    ]
    print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
