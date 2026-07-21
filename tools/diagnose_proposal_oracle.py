#!/usr/bin/env python3
"""M0-3: crop-reachable oracle on the segment d_obs -> d_dyn.

The diagnostic composes two frozen, already-trained pieces without training:
the complete A1 observation path and only the complete A2 DynamicsEncoder.
It never loads the incompatible 384-D A2 fusion head into the 256-D A1 head.
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
    distribution,
    load_checkpoint_state,
    matching_state,
    move_to_device,
    optimal_convex_blend,
    require_clean_git,
    sha256_file,
    tensor_numpy,
    tracklet_bootstrap_ci,
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
    "target_x",
    "target_y",
    "target_z",
    "observation_x",
    "observation_y",
    "observation_z",
    "dynamics_x",
    "dynamics_y",
    "dynamics_z",
    "oracle_x",
    "oracle_y",
    "oracle_z",
    "alpha_oracle",
    "alpha_interior",
    "innovation_norm",
    "observation_error",
    "dynamics_error",
    "oracle_error",
    "oracle_gain",
    "oracle_relative_gain",
]


def subset_summary(rows, useful_gain):
    if not rows:
        return {
            "sample_count": 0,
            "tracklet_count": 0,
            "observation_error": distribution([]),
            "dynamics_error": distribution([]),
            "oracle_error": distribution([]),
            "oracle_gain": distribution([]),
            "oracle_relative_gain": distribution([]),
            "alpha_oracle": distribution([]),
            "innovation_norm": distribution([]),
            "useful_gain_rate": None,
            "alpha_interior_rate": None,
        }
    return {
        "sample_count": len(rows),
        "tracklet_count": len({int(row["tracklet_id"]) for row in rows}),
        "observation_error": distribution([row["observation_error"] for row in rows]),
        "dynamics_error": distribution([row["dynamics_error"] for row in rows]),
        "oracle_error": distribution([row["oracle_error"] for row in rows]),
        "oracle_gain": distribution([row["oracle_gain"] for row in rows]),
        "oracle_relative_gain": distribution(
            [row["oracle_relative_gain"] for row in rows]
        ),
        "alpha_oracle": distribution([row["alpha_oracle"] for row in rows]),
        "innovation_norm": distribution([row["innovation_norm"] for row in rows]),
        "useful_gain_rate": float(np.mean([
            row["oracle_gain"] >= useful_gain for row in rows
        ])),
        "alpha_interior_rate": float(np.mean([
            row["alpha_interior"] for row in rows
        ])),
    }


def summarize_rows(rows, args):
    eligible = [
        row for row in rows
        if not row["resampled"]
        and row["crop_reachable"]
        and row["full_history"]
        and row["dynamics_valid"]
    ]
    primary = [row for row in eligible if row["candidate_id"] == 0]
    by_candidate = {
        str(candidate): subset_summary(
            [row for row in eligible if row["candidate_id"] == candidate],
            args.useful_gain,
        )
        for candidate in range(4)
    }
    long_gap = [row for row in primary if row["current_delta_t"] >= args.long_gap_threshold]
    sparse = [
        row for row in primary
        if row["current_target_points"] <= args.sparse_target_point_threshold
    ]
    long_gap_sparse = [
        row for row in long_gap
        if row["current_target_points"] <= args.sparse_target_point_threshold
    ]
    bins = {
        "long_gap": subset_summary(long_gap, args.useful_gain),
        "sparse": subset_summary(sparse, args.useful_gain),
        "long_gap_and_sparse": subset_summary(long_gap_sparse, args.useful_gain),
    }
    primary_summary = subset_summary(primary, args.useful_gain)
    bootstrap = tracklet_bootstrap_ci(
        primary,
        "oracle_gain",
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    supported_challenge_bins = [
        name for name, selected in (("long_gap", long_gap), ("sparse", sparse))
        if len(selected) >= args.min_challenge_samples
    ]
    positive_challenge_bins = [
        name for name in supported_challenge_bins
        if bins[name]["oracle_gain"]["mean"] is not None
        and bins[name]["oracle_gain"]["mean"] > 0.0
    ]

    checks = {
        "minimum_samples": len(primary) >= args.min_samples,
        "minimum_tracklets": primary_summary["tracklet_count"] >= args.min_tracklets,
        "mean_gain": (
            primary_summary["oracle_gain"]["mean"] is not None
            and primary_summary["oracle_gain"]["mean"] >= args.min_mean_gain
        ),
        "useful_gain_rate": (
            primary_summary["useful_gain_rate"] is not None
            and primary_summary["useful_gain_rate"] >= args.min_useful_gain_rate
        ),
        "tracklet_bootstrap_lower_positive": (
            bootstrap["ci95"][0] is not None and bootstrap["ci95"][0] > 0.0
        ),
        "challenge_bin_supported_and_positive": bool(positive_challenge_bins),
    }
    decision = (
        "GO_M2_PROPOSAL_INNOVATION"
        if all(checks.values())
        else "NO_GO_M2_PROPOSAL_INNOVATION"
    )
    return {
        "schema": "ct_seqtrack_m0_proposal_oracle_v1",
        "scope": (
            "candidate0 + crop-reachable + full-history + dynamics-valid mini_train; "
            "candidate1/2/3 are secondary diagnostics"
        ),
        "row_count": len(rows),
        "eligible_all_candidate_count": len(eligible),
        "resampled_count": sum(int(row["resampled"]) for row in rows),
        "primary": primary_summary,
        "by_candidate": by_candidate,
        "bins": bins,
        "tracklet_bootstrap_oracle_gain": bootstrap,
        "supported_challenge_bins": supported_challenge_bins,
        "positive_challenge_bins": positive_challenge_bins,
        "preregistered_thresholds": {
            "min_samples": args.min_samples,
            "min_tracklets": args.min_tracklets,
            "min_mean_gain_m": args.min_mean_gain,
            "useful_gain_m": args.useful_gain,
            "min_useful_gain_rate": args.min_useful_gain_rate,
            "long_gap_threshold_s": args.long_gap_threshold,
            "sparse_target_point_threshold": args.sparse_target_point_threshold,
            "min_challenge_samples": args.min_challenge_samples,
            "bootstrap_iterations": args.bootstrap_iterations,
        },
        "checks": checks,
        "decision": decision,
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
        raise ValueError("M0-3 requires train_type=train_motion_mf.")
    if int(cfg.num_candidates) != 4:
        raise ValueError("M0-3 requires candidate0/1/2/3 for primary/secondary reports.")
    if bool(getattr(cfg, "use_augmentation", False)):
        raise ValueError("M0-3 is a frozen proposal diagnostic; keep use_augmentation=false.")
    if not bool(getattr(cfg, "use_dynamics_encoder", False)):
        raise ValueError("M0-3 requires use_dynamics_encoder=true.")
    mode = str(getattr(cfg, "dynamics_motion_mode", "")).replace("-", "_")
    if mode not in ("residual", "residual_limited", "bounded_residual"):
        raise ValueError(
            "Use an A2 residual config so motion_obs_pred is the pure 256-D A1 head."
        )
    return cfg


def load_hybrid_model(cfg, observation_checkpoint, dynamics_checkpoint):
    from models import get_model

    model = get_model(cfg.net_model)(cfg)
    if int(model.motion_mlp[0].in_features) != 256:
        raise RuntimeError("Oracle observation head must remain the A1-compatible 256-D head.")
    target_state = model.state_dict()

    observation_source = load_checkpoint_state(observation_checkpoint)
    observation_exclude = (
        "dynamics_encoder.",
        "dynamics_residual_gate.",
        "observability_gate.",
    )
    observation_state, observation_mismatches = matching_state(
        observation_source, target_state, exclude=observation_exclude
    )
    observation_expected = {
        key for key in target_state
        if not any(key.startswith(prefix) for prefix in observation_exclude)
    }
    observation_missing = sorted(observation_expected - set(observation_state))
    if observation_mismatches or observation_missing:
        raise RuntimeError(
            "A1 checkpoint is not a complete shape-compatible observation path. "
            f"missing={observation_missing[:20]}, mismatches={observation_mismatches[:20]}"
        )
    model.load_state_dict(observation_state, strict=False)

    dynamics_source = load_checkpoint_state(dynamics_checkpoint)
    dynamics_state, dynamics_mismatches = matching_state(
        dynamics_source,
        target_state,
        include=("dynamics_encoder.",),
    )
    dynamics_expected = {
        key for key in target_state if key.startswith("dynamics_encoder.")
    }
    dynamics_missing = sorted(dynamics_expected - set(dynamics_state))
    if dynamics_mismatches or dynamics_missing:
        raise RuntimeError(
            "A2 checkpoint is not a complete shape-compatible DynamicsEncoder. "
            f"missing={dynamics_missing}, mismatches={dynamics_mismatches}"
        )
    model.load_state_dict(dynamics_state, strict=False)
    return model, {
        "composition": (
            "complete A1 observation/base path + complete A2 dynamics_encoder only; "
            "A2 384-D motion_mlp is intentionally excluded"
        ),
        "observation_checkpoint": {
            "path": str(Path(observation_checkpoint).resolve()),
            "sha256": sha256_file(observation_checkpoint),
            "loaded_key_count": len(observation_state),
        },
        "dynamics_checkpoint": {
            "path": str(Path(dynamics_checkpoint).resolve()),
            "sha256": sha256_file(dynamics_checkpoint),
            "loaded_key_count": len(dynamics_state),
            "loaded_keys": sorted(dynamics_state),
        },
        "motion_head_input_dim": int(model.motion_mlp[0].in_features),
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


def batch_rows(batch, model, device, point_sample_size):
    import torch

    device_batch = move_to_device(batch, device)
    with torch.no_grad():
        output = model(device_batch)
    if "motion_obs_pred" not in output or "dynamics_displacement_pred" not in output:
        raise RuntimeError("Residual config did not expose d_obs and d_dyn.")

    observation = tensor_numpy(output["motion_obs_pred"])[:, :3]
    dynamics = tensor_numpy(output["dynamics_displacement_pred"])
    target = tensor_numpy(batch["motion_label"])[:, 0, :3]
    alpha, oracle = optimal_convex_blend(observation, dynamics, target)
    observation_error = np.linalg.norm(observation - target, axis=1)
    dynamics_error = np.linalg.norm(dynamics - target, axis=1)
    oracle_error = np.linalg.norm(oracle - target, axis=1)
    oracle_gain = observation_error - oracle_error
    relative_gain = oracle_gain / np.maximum(observation_error, 1e-6)
    innovation_norm = np.linalg.norm(dynamics - observation, axis=1)
    dynamics_valid = tensor_numpy(output["dynamics_valid"]).reshape(-1) > 0.5
    valid_mask = tensor_numpy(batch["valid_mask"]).astype(bool)
    seg_label = tensor_numpy(batch["seg_label"])
    current_target_points = np.sum(seg_label[:, -point_sample_size:] > 0, axis=1)
    values = {
        key: tensor_numpy(value) for key, value in batch.items() if key in {
            "_diagnostic_index", "_diagnostic_tracklet_id", "_diagnostic_resampled",
            "this_frame_id", "candidate_id", "current_delta_t_real",
            "num_points_in_search",
        }
    }

    rows = []
    for index in range(observation.shape[0]):
        rows.append({
            "sample_index": int(values["_diagnostic_index"][index]),
            "tracklet_id": int(values["_diagnostic_tracklet_id"][index]),
            "frame_id": int(values["this_frame_id"][index]),
            "candidate_id": int(values["candidate_id"][index]),
            "resampled": bool(values["_diagnostic_resampled"][index]),
            "full_history": bool(np.sum(valid_mask[index]) == valid_mask.shape[1]),
            "crop_reachable": bool(current_target_points[index] > 0),
            "dynamics_valid": bool(dynamics_valid[index]),
            "current_delta_t": float(values["current_delta_t_real"][index]),
            "num_points_in_search": float(values["num_points_in_search"][index]),
            "current_target_points": int(current_target_points[index]),
            "target_x": float(target[index, 0]),
            "target_y": float(target[index, 1]),
            "target_z": float(target[index, 2]),
            "observation_x": float(observation[index, 0]),
            "observation_y": float(observation[index, 1]),
            "observation_z": float(observation[index, 2]),
            "dynamics_x": float(dynamics[index, 0]),
            "dynamics_y": float(dynamics[index, 1]),
            "dynamics_z": float(dynamics[index, 2]),
            "oracle_x": float(oracle[index, 0]),
            "oracle_y": float(oracle[index, 1]),
            "oracle_z": float(oracle[index, 2]),
            "alpha_oracle": float(alpha[index]),
            "alpha_interior": bool(1e-6 < alpha[index] < 1.0 - 1e-6),
            "innovation_norm": float(innovation_norm[index]),
            "observation_error": float(observation_error[index]),
            "dynamics_error": float(dynamics_error[index]),
            "oracle_error": float(oracle_error[index]),
            "oracle_gain": float(oracle_gain[index]),
            "oracle_relative_gain": float(relative_gain[index]),
        })
    return rows


def self_test():
    observation = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    dynamics = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    target = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    alpha, oracle = optimal_convex_blend(observation, dynamics, target)
    assert np.allclose(alpha, [0.5, 0.0])
    assert np.allclose(oracle, [[1.0, 0.0], [0.0, 0.0]])

    args = argparse.Namespace(
        useful_gain=0.05,
        long_gap_threshold=1.0,
        sparse_target_point_threshold=5,
        min_challenge_samples=1,
        bootstrap_iterations=200,
        seed=42,
        min_samples=1,
        min_tracklets=2,
        min_mean_gain=0.05,
        min_useful_gain_rate=0.15,
    )
    rows = []
    for tracklet in range(12):
        rows.append({
            "tracklet_id": tracklet,
            "candidate_id": 0,
            "resampled": False,
            "crop_reachable": True,
            "full_history": True,
            "dynamics_valid": True,
            "current_delta_t": 1.5,
            "current_target_points": 3,
            "observation_error": 0.5,
            "dynamics_error": 0.4,
            "oracle_error": 0.3,
            "oracle_gain": 0.2,
            "oracle_relative_gain": 0.4,
            "alpha_oracle": 0.5,
            "alpha_interior": True,
            "innovation_norm": 0.3,
        })
    summary = summarize_rows(rows, args)
    assert summary["decision"] == "GO_M2_PROPOSAL_INNOVATION"
    print("diagnose_proposal_oracle self-test: PASS")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cfg", default="cfgs/seqtrack3d_nuscenes_a2_residual_dyn.yaml")
    parser.add_argument("--observation-checkpoint")
    parser.add_argument("--dynamics-checkpoint")
    parser.add_argument("--path")
    parser.add_argument("--version")
    parser.add_argument("--split")
    parser.add_argument("--tag", default="m0_oracle_standard_seed42")
    parser.add_argument("--output-root", default="output/diagnostics/m0_proposal_oracle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-tracklets", type=int, default=20)
    parser.add_argument("--min-mean-gain", type=float, default=0.05)
    parser.add_argument("--useful-gain", type=float, default=0.05)
    parser.add_argument("--min-useful-gain-rate", type=float, default=0.15)
    parser.add_argument("--long-gap-threshold", type=float, default=1.0)
    parser.add_argument("--sparse-target-point-threshold", type=int, default=5)
    parser.add_argument("--min-challenge-samples", type=int, default=30)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.observation_checkpoint or not args.dynamics_checkpoint:
        raise SystemExit(
            "--observation-checkpoint and --dynamics-checkpoint are required."
        )

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    provenance = require_clean_git(args.allow_dirty)
    cfg = load_config(args.cfg, args)
    model, checkpoint_report = load_hybrid_model(
        cfg, args.observation_checkpoint, args.dynamics_checkpoint
    )
    device = torch.device(args.device)
    model.to(device).eval()
    loader, split = build_loader(cfg, args)

    rows = []
    for batch_index, batch in enumerate(loader):
        rows.extend(batch_rows(batch, model, device, int(cfg.point_sample_size)))
        if (batch_index + 1) % 100 == 0:
            print(f"processed batches={batch_index + 1}, rows={len(rows)}", flush=True)
        if args.max_batches > 0 and batch_index + 1 >= args.max_batches:
            break
    if not rows:
        raise RuntimeError("No proposal-oracle rows were produced.")

    summary = summarize_rows(rows, args)
    summary["provenance"] = {
        "git": provenance,
        "config_path": str(Path(args.cfg).resolve()),
        "config_sha256": sha256_file(args.cfg),
        "checkpoint_composition": checkpoint_report,
        "split": split,
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_batches": args.max_batches,
    }
    output_dir = Path(args.output_root) / args.tag
    csv_path = output_dir / "proposal_oracle_endpoints.csv"
    json_path = output_dir / "proposal_oracle_summary.json"
    write_csv(csv_path, rows, CSV_FIELDS)
    write_json(json_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"proposal endpoint CSV: {csv_path}")
    print(f"proposal summary JSON: {json_path}")


if __name__ == "__main__":
    main()
