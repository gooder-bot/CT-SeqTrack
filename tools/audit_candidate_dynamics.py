#!/usr/bin/env python3
"""M0-4: audit candidate-induced pseudo derivatives and dynamics errors.

The script is read-only with respect to the model and dataset. It evaluates the
existing independent per-history-box candidate perturbation on mini_train and
freezes a single M1 augmentation choice only when the preregistered checks pass.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m0_diagnostic_utils import (  # noqa: E402
    IndexedDiagnosticDataset,
    candidate_kinematics,
    distribution,
    git_provenance,
    load_checkpoint_state,
    masked_row_max,
    masked_row_mean,
    matching_state,
    require_clean_git,
    sha256_file,
    tensor_numpy,
    write_csv,
    write_json,
)


CSV_FIELDS = [
    "sample_index",
    "tracklet_id",
    "frame_id",
    "candidate_id",
    "resampled",
    "full_history",
    "crop_reachable",
    "dynamics_valid",
    "current_delta_t",
    "num_points_in_search",
    "current_target_points",
    "candidate_offset_xy_rms",
    "candidate_offset_yaw_rms",
    "velocity_jitter_mean",
    "velocity_jitter_max",
    "yaw_rate_jitter_mean",
    "yaw_rate_jitter_max",
    "acceleration_jitter_mean",
    "acceleration_jitter_max",
    "dynamics_proposal_error",
    "dynamics_x",
    "dynamics_y",
    "dynamics_z",
    "target_x",
    "target_y",
    "target_z",
]


def metric_summary(rows, key):
    return distribution([row[key] for row in rows])


def summarize_rows(rows, args):
    usable = [
        row for row in rows
        if not row["resampled"] and row["full_history"]
    ]
    by_candidate = {}
    metrics = (
        "velocity_jitter_mean",
        "velocity_jitter_max",
        "yaw_rate_jitter_mean",
        "acceleration_jitter_mean",
        "acceleration_jitter_max",
        "dynamics_proposal_error",
    )
    for candidate_id in range(4):
        selected = [row for row in usable if row["candidate_id"] == candidate_id]
        proposal_selected = [
            row for row in selected
            if row["crop_reachable"] and row["dynamics_valid"]
        ]
        candidate_summary = {
            "sample_count": len(selected),
            "reachable_dynamics_sample_count": len(proposal_selected),
        }
        for metric in metrics:
            source = proposal_selected if metric == "dynamics_proposal_error" else selected
            candidate_summary[metric] = metric_summary(source, metric)
        by_candidate[str(candidate_id)] = candidate_summary

    candidate0 = [row for row in usable if row["candidate_id"] == 0]
    nonzero = [row for row in usable if row["candidate_id"] in (1, 2, 3)]
    c0_velocity = metric_summary(candidate0, "velocity_jitter_mean")
    c0_acceleration = metric_summary(candidate0, "acceleration_jitter_mean")
    nz_velocity = metric_summary(nonzero, "velocity_jitter_mean")
    nz_acceleration = metric_summary(nonzero, "acceleration_jitter_mean")

    minimum_count_ok = all(
        by_candidate[str(candidate)]["sample_count"] >= args.min_samples_per_candidate
        for candidate in range(4)
    )
    c0_sanity_ok = (
        c0_velocity["p95"] is not None
        and c0_acceleration["p95"] is not None
        and c0_velocity["p95"] <= args.candidate0_tolerance
        and c0_acceleration["p95"] <= args.candidate0_tolerance
    )
    material_velocity = (
        nz_velocity["p50"] is not None
        and nz_velocity["p50"] >= args.velocity_jitter_threshold
    )
    material_acceleration = (
        nz_acceleration["p50"] is not None
        and nz_acceleration["p50"] >= args.acceleration_jitter_threshold
    )

    if not minimum_count_ok:
        decision = "INVALID_M0_CANDIDATE_AUDIT_INSUFFICIENT_SAMPLES"
        frozen_augmentation = None
    elif not c0_sanity_ok:
        decision = "INVALID_M0_CANDIDATE_AUDIT_CANONICAL_SANITY"
        frozen_augmentation = None
    elif material_velocity or material_acceleration:
        decision = "FREEZE_M1_SHARED_SE2"
        frozen_augmentation = "shared_se2"
    else:
        decision = "NO_GO_M1_AUGMENTATION_FREEZE"
        frozen_augmentation = None

    c0_proposal = metric_summary(
        [row for row in candidate0 if row["crop_reachable"] and row["dynamics_valid"]],
        "dynamics_proposal_error",
    )
    nz_proposal = metric_summary(
        [row for row in nonzero if row["crop_reachable"] and row["dynamics_valid"]],
        "dynamics_proposal_error",
    )
    proposal_mean_delta = None
    if c0_proposal["mean"] is not None and nz_proposal["mean"] is not None:
        proposal_mean_delta = nz_proposal["mean"] - c0_proposal["mean"]

    return {
        "schema": "ct_seqtrack_m0_candidate_audit_v1",
        "scope": "mini_train_full_history",
        "row_count": len(rows),
        "usable_full_history_count": len(usable),
        "resampled_count": sum(int(row["resampled"]) for row in rows),
        "by_candidate": by_candidate,
        "candidate0_sanity": {
            "tolerance": args.candidate0_tolerance,
            "velocity_jitter": c0_velocity,
            "acceleration_jitter": c0_acceleration,
            "passed": c0_sanity_ok,
        },
        "nonzero_candidate": {
            "velocity_jitter": nz_velocity,
            "acceleration_jitter": nz_acceleration,
            "dynamics_proposal_error": nz_proposal,
            "candidate0_dynamics_proposal_error": c0_proposal,
            "proposal_error_mean_delta_nonzero_minus_candidate0": proposal_mean_delta,
        },
        "preregistered_thresholds": {
            "min_samples_per_candidate": args.min_samples_per_candidate,
            "velocity_jitter_p50_m_per_s": args.velocity_jitter_threshold,
            "acceleration_jitter_p50_m_per_s2": args.acceleration_jitter_threshold,
        },
        "checks": {
            "minimum_count_ok": minimum_count_ok,
            "candidate0_sanity_ok": c0_sanity_ok,
            "material_velocity_jitter": material_velocity,
            "material_acceleration_jitter": material_acceleration,
        },
        "decision": decision,
        "frozen_m1_augmentation": frozen_augmentation,
        "smooth_drift_status": (
            "NOT_SELECTED: it changes physical derivatives and requires a separately "
            "frozen recursive-error process; this audit tests derivative preservation."
        ),
    }


def load_config(path, args):
    import yaml
    from easydict import EasyDict

    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    cfg.path = args.path or cfg.path
    if args.version is not None:
        cfg.version = args.version
    cfg.batch_size = args.batch_size
    cfg.workers = args.workers
    cfg.preloading = False
    cfg.tiny = getattr(cfg, "tiny", False)
    cfg.use_twc = False
    cfg.dynamics_time_mode = "true"
    if cfg.train_type.lower() != "train_motion_mf":
        raise ValueError("M0-4 requires train_type=train_motion_mf.")
    if int(cfg.num_candidates) != 4:
        raise ValueError("M0-4 requires num_candidates=4 (candidate0/1/2/3).")
    if bool(getattr(cfg, "use_augmentation", False)):
        raise ValueError("M0-4 audits candidate offsets only; keep use_augmentation=false.")
    return cfg


def load_dynamics_encoder(cfg, checkpoint):
    from models.dynamics import DynamicsEncoder

    encoder = DynamicsEncoder(
        hidden_dim=int(getattr(cfg, "dynamics_hidden_dim", 128)),
        eps=float(getattr(cfg, "dynamics_eps", 1e-3)),
        use_query_gap=bool(getattr(cfg, "dynamics_use_query_gap", True)),
    )
    state = load_checkpoint_state(checkpoint)
    expected = encoder.state_dict()
    prefixed_expected = {f"dynamics_encoder.{key}": value for key, value in expected.items()}
    selected, mismatches = matching_state(
        state, prefixed_expected, include=("dynamics_encoder.",)
    )
    if mismatches:
        raise RuntimeError(f"Dynamics checkpoint shape mismatch: {mismatches}")
    stripped = {
        key[len("dynamics_encoder."):]: value for key, value in selected.items()
    }
    missing = sorted(set(expected) - set(stripped))
    if missing:
        raise RuntimeError(
            "Checkpoint does not provide the complete trained DynamicsEncoder: "
            + ", ".join(missing)
        )
    encoder.load_state_dict(stripped, strict=True)
    return encoder, {
        "path": str(Path(checkpoint).resolve()),
        "sha256": sha256_file(checkpoint),
        "loaded_key_count": len(stripped),
        "loaded_keys": sorted(stripped),
    }


def build_loader(cfg, args):
    import torch
    from torch.utils.data import DataLoader
    from datasets import get_dataset

    split = args.split or cfg.train_split
    sampler = get_dataset(cfg, type=cfg.train_type, split=split, protocol_role="train")
    indexed = IndexedDiagnosticDataset(sampler)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        indexed,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        generator=generator,
    ), split


def batch_rows(batch, encoder, device, point_sample_size):
    import torch

    ref_boxes_t = batch["ref_boxs"].to(device)
    delta_t_t = batch["delta_t_real"].to(device)
    valid_mask_t = batch["valid_mask"].to(device)
    current_dt_t = batch["current_delta_t_real"].to(device)
    with torch.no_grad():
        _, _, dynamics_t, dynamics_valid_t = encoder(
            ref_boxes_t, delta_t_t, valid_mask_t, current_dt_t
        )

    ref_boxes = tensor_numpy(batch["ref_boxs"])
    canonical_boxes = tensor_numpy(batch["box_label_prev"])
    delta_t = tensor_numpy(batch["delta_t_real"])
    valid_mask = tensor_numpy(batch["valid_mask"]).astype(bool)
    kinematics = candidate_kinematics(
        ref_boxes, canonical_boxes, delta_t, valid_mask
    )
    velocity_mean = masked_row_mean(
        kinematics["velocity_jitter"], kinematics["transition_valid"]
    )
    velocity_max = masked_row_max(
        kinematics["velocity_jitter"], kinematics["transition_valid"]
    )
    yaw_mean = masked_row_mean(
        kinematics["yaw_rate_jitter"], kinematics["transition_valid"]
    )
    yaw_max = masked_row_max(
        kinematics["yaw_rate_jitter"], kinematics["transition_valid"]
    )
    acceleration_mean = masked_row_mean(
        kinematics["acceleration_jitter"], kinematics["acceleration_valid"]
    )
    acceleration_max = masked_row_max(
        kinematics["acceleration_jitter"], kinematics["acceleration_valid"]
    )

    dynamics = tensor_numpy(dynamics_t)
    dynamics_valid = tensor_numpy(dynamics_valid_t).reshape(-1) > 0.5
    target = tensor_numpy(batch["motion_label"])[:, 0, :3]
    proposal_error = np.linalg.norm(dynamics - target, axis=1)
    seg_label = tensor_numpy(batch["seg_label"])
    current_target_points = np.sum(seg_label[:, -point_sample_size:] > 0, axis=1)
    candidate_offsets = tensor_numpy(batch["candidate_offsets"])
    offset_xy_rms = np.sqrt(np.mean(np.sum(candidate_offsets[:, :, :2] ** 2, axis=2), axis=1))
    offset_yaw_rms = np.sqrt(np.mean(candidate_offsets[:, :, 2] ** 2, axis=1))

    arrays = {key: tensor_numpy(value) for key, value in batch.items() if key in {
        "_diagnostic_index", "_diagnostic_tracklet_id", "_diagnostic_resampled",
        "this_frame_id", "candidate_id", "current_delta_t_real",
        "num_points_in_search",
    }}
    rows = []
    for index in range(ref_boxes.shape[0]):
        row = {
            "sample_index": int(arrays["_diagnostic_index"][index]),
            "tracklet_id": int(arrays["_diagnostic_tracklet_id"][index]),
            "frame_id": int(arrays["this_frame_id"][index]),
            "candidate_id": int(arrays["candidate_id"][index]),
            "resampled": bool(arrays["_diagnostic_resampled"][index]),
            "full_history": bool(np.sum(valid_mask[index]) == valid_mask.shape[1]),
            "crop_reachable": bool(current_target_points[index] > 0),
            "dynamics_valid": bool(dynamics_valid[index]),
            "current_delta_t": float(arrays["current_delta_t_real"][index]),
            "num_points_in_search": float(arrays["num_points_in_search"][index]),
            "current_target_points": int(current_target_points[index]),
            "candidate_offset_xy_rms": float(offset_xy_rms[index]),
            "candidate_offset_yaw_rms": float(offset_yaw_rms[index]),
            "velocity_jitter_mean": float(velocity_mean[index]),
            "velocity_jitter_max": float(velocity_max[index]),
            "yaw_rate_jitter_mean": float(yaw_mean[index]),
            "yaw_rate_jitter_max": float(yaw_max[index]),
            "acceleration_jitter_mean": float(acceleration_mean[index]),
            "acceleration_jitter_max": float(acceleration_max[index]),
            "dynamics_proposal_error": float(proposal_error[index]),
            "dynamics_x": float(dynamics[index, 0]),
            "dynamics_y": float(dynamics[index, 1]),
            "dynamics_z": float(dynamics[index, 2]),
            "target_x": float(target[index, 0]),
            "target_y": float(target[index, 1]),
            "target_z": float(target[index, 2]),
        }
        rows.append(row)
    return rows


def self_test():
    canonical = np.asarray([[[2.0, 0, 0, 0], [1.0, 0, 0, 0], [0.0, 0, 0, 0]]])
    delta_t = np.asarray([[1.0, 1.0, 1.0]])
    valid = np.ones((1, 3), dtype=bool)
    exact = candidate_kinematics(canonical, canonical, delta_t, valid)
    assert np.max(exact["velocity_jitter"]) == 0.0
    assert np.max(exact["acceleration_jitter"]) == 0.0

    perturbed = canonical.copy()
    perturbed[0, 1, 0] += 0.2
    perturbed[0, 2, 0] -= 0.1
    noisy = candidate_kinematics(perturbed, canonical, delta_t, valid)
    assert float(np.max(noisy["velocity_jitter"])) > 0.1
    assert float(np.max(noisy["acceleration_jitter"])) > 0.1

    parser_args = argparse.Namespace(
        min_samples_per_candidate=1,
        candidate0_tolerance=1e-6,
        velocity_jitter_threshold=0.05,
        acceleration_jitter_threshold=0.1,
    )
    rows = []
    for candidate in range(4):
        rows.append({
            "candidate_id": candidate,
            "resampled": False,
            "full_history": True,
            "crop_reachable": True,
            "dynamics_valid": True,
            "velocity_jitter_mean": 0.0 if candidate == 0 else 0.2,
            "velocity_jitter_max": 0.0 if candidate == 0 else 0.3,
            "yaw_rate_jitter_mean": 0.0,
            "acceleration_jitter_mean": 0.0 if candidate == 0 else 0.4,
            "acceleration_jitter_max": 0.0 if candidate == 0 else 0.5,
            "dynamics_proposal_error": 0.1 + candidate * 0.01,
        })
    summary = summarize_rows(rows, parser_args)
    assert summary["decision"] == "FREEZE_M1_SHARED_SE2"
    print("audit_candidate_dynamics self-test: PASS")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cfg", default="cfgs/seqtrack3d_nuscenes_a2_order_dyn.yaml")
    parser.add_argument("--dynamics-checkpoint")
    parser.add_argument("--path")
    parser.add_argument("--version")
    parser.add_argument("--split")
    parser.add_argument("--tag", default="m0_candidate_standard_seed42")
    parser.add_argument("--output-root", default="output/diagnostics/m0_candidate_audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--min-samples-per-candidate", type=int, default=100)
    parser.add_argument("--candidate0-tolerance", type=float, default=1e-4)
    parser.add_argument("--velocity-jitter-threshold", type=float, default=0.05)
    parser.add_argument("--acceleration-jitter-threshold", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dynamics_checkpoint:
        raise SystemExit("--dynamics-checkpoint is required outside --self-test.")

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    provenance = require_clean_git(args.allow_dirty)
    cfg = load_config(args.cfg, args)
    encoder, checkpoint_report = load_dynamics_encoder(cfg, args.dynamics_checkpoint)
    device = torch.device(args.device)
    encoder.to(device).eval()
    loader, split = build_loader(cfg, args)

    rows = []
    for batch_index, batch in enumerate(loader):
        rows.extend(batch_rows(batch, encoder, device, int(cfg.point_sample_size)))
        if (batch_index + 1) % 100 == 0:
            print(f"processed batches={batch_index + 1}, rows={len(rows)}", flush=True)
        if args.max_batches > 0 and batch_index + 1 >= args.max_batches:
            break
    if not rows:
        raise RuntimeError("No candidate-audit rows were produced.")

    summary = summarize_rows(rows, args)
    summary["provenance"] = {
        "git": provenance,
        "config_path": str(Path(args.cfg).resolve()),
        "config_sha256": sha256_file(args.cfg),
        "dynamics_checkpoint": checkpoint_report,
        "split": split,
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_batches": args.max_batches,
    }
    output_dir = Path(args.output_root) / args.tag
    csv_path = output_dir / "candidate_dynamics_endpoints.csv"
    json_path = output_dir / "candidate_dynamics_summary.json"
    write_csv(csv_path, rows, CSV_FIELDS)
    write_json(json_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"candidate endpoint CSV: {csv_path}")
    print(f"candidate summary JSON: {json_path}")


if __name__ == "__main__":
    main()
