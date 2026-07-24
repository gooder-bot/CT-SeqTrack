#!/usr/bin/env python3
"""Export matched endpoint predictions for M4 off/filter/tube ablations.

Unlike the passive M0 exporter, M4 changes recursive state and search support.
This exporter therefore runs the model's actual ``evaluate_one_sequence`` path
and persists its online predictions plus evaluation-only M4 diagnostics.  The
CSV intentionally satisfies ``summarize_m0_endpoints.py`` so all four arms can
be paired by endpoint and bootstrapped by tracklet.
"""

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import export_m0_endpoints as m0


M4_VARIANTS = ("off", "filter", "tube", "filter_tube")


def apply_m4_variant(cfg, variant, time_mode, fixed_delta_t):
    variant = str(variant).strip().lower()
    if variant not in M4_VARIANTS:
        raise ValueError(f"Unsupported M4 variant: {variant}")
    cfg.m4_variant = variant
    cfg.use_m4_state_filter = variant in ("filter", "filter_tube")
    cfg.use_m4_trajectory_tube = variant in ("tube", "filter_tube")
    cfg.m4_time_mode = str(time_mode).strip().lower()
    if cfg.m4_time_mode not in ("fixed", "real"):
        raise ValueError("m4_time_mode must be fixed or real")
    cfg.m4_fixed_delta_t = float(fixed_delta_t)
    if cfg.m4_fixed_delta_t <= 0:
        raise ValueError("m4_fixed_delta_t must be positive")
    # Deployment of an M3 checkpoint always uses the student.  The model hook
    # removes EMA teacher tensors before strict state loading.
    cfg.use_m3_path_distillation = False
    cfg.m3_variant = "off"
    cfg.use_twc = False
    return cfg


def finite_delta_t(sequence, frame_index, default_step):
    current = m0.crop_diag.frame_timestamp(sequence[frame_index], frame_index)
    previous = m0.crop_diag.frame_timestamp(
        sequence[frame_index - 1], frame_index - 1)
    delta_t = float(current) - float(previous)
    if not np.isfinite(delta_t) or delta_t <= 0:
        return float(default_step)
    return delta_t


def diagnostic_fields(diagnostic):
    diagnostic = diagnostic or {}
    return {
        "m4_prediction_valid": diagnostic.get("prediction_valid"),
        "m4_measurement_accepted": diagnostic.get("measurement_accepted"),
        "m4_reason": diagnostic.get("reason", ""),
        "m4_mahalanobis": diagnostic.get("mahalanobis"),
        "m4_num_points_search_baseline": diagnostic.get(
            "m4_num_points_search_baseline"),
        "m4_num_points_search_tube": diagnostic.get(
            "m4_num_points_search_tube"),
        "m4_num_points_search_union": diagnostic.get(
            "m4_num_points_search_union"),
        "m4_tube_width": diagnostic.get("m4_tube_width"),
        "m4_tube_length": diagnostic.get("m4_tube_length"),
        "m4_oracle_center_baseline": diagnostic.get(
            "m4_oracle_target_center_in_baseline"),
        "m4_oracle_center_tube": diagnostic.get(
            "m4_oracle_target_center_in_tube"),
        "m4_oracle_center_union": diagnostic.get(
            "m4_oracle_target_center_in_union"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export online M4 ablation predictions at endpoint level.")
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--protocol-cfg", default=None)
    parser.add_argument("--run-label", default="M4")
    parser.add_argument("--protocol-name", default="standard")
    parser.add_argument("--m4-variant", choices=M4_VARIANTS)
    parser.add_argument("--m4-time-mode", choices=("fixed", "real"), default="fixed")
    parser.add_argument("--m4-fixed-delta-t", type=float, default=0.5)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--virtual-rate-manifest", default=None)
    parser.add_argument("--allow-manifest-commit-mismatch", action="store_true")
    parser.add_argument("--output-dir", default="output/m4_endpoints")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        missing = [
            flag
            for flag, value in (
                ("--cfg", args.cfg),
                ("--weights", args.weights),
                ("--m4-variant", args.m4_variant),
            )
            if not value
        ]
        if missing:
            parser.error(
                "the following arguments are required unless --self-test is "
                f"used: {', '.join(missing)}"
            )
    return args


def self_test():
    class Config:
        pass

    for variant in M4_VARIANTS:
        cfg = apply_m4_variant(Config(), variant, "fixed", 0.5)
        assert cfg.use_m4_state_filter == (
            variant in ("filter", "filter_tube"))
        assert cfg.use_m4_trajectory_tube == (
            variant in ("tube", "filter_tube"))
        assert not cfg.use_m3_path_distillation
    fields = diagnostic_fields({
        "prediction_valid": True,
        "m4_oracle_target_center_in_union": 1.0,
    })
    assert fields["m4_prediction_valid"] is True
    assert fields["m4_oracle_center_union"] == 1.0
    print("M4 endpoint exporter self-test: PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    m0.load_runtime_dependencies()
    initial_git = m0.git_state()
    if args.require_clean_git and initial_git["dirty"]:
        raise RuntimeError(
            "M4 export requires a clean worktree: "
            f"{initial_git['status_porcelain']}")

    cfg_path = Path(args.cfg).resolve()
    weights_path = Path(args.weights).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    m0.torch.manual_seed(args.seed)
    if m0.torch.cuda.is_available():
        m0.torch.cuda.manual_seed_all(args.seed)
    m0.torch.set_float32_matmul_precision("high")

    cfg = m0.load_config(cfg_path)
    m0.merge_protocol_config(cfg, args.protocol_cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    cfg.preloading = bool(args.preloading)
    if args.virtual_rate_manifest is not None:
        cfg.virtual_rate_manifest_test = args.virtual_rate_manifest
        cfg.test_virtual_rate_manifest_allow_create = False
        cfg.test_virtual_rate_manifest_strict = True
        cfg.test_virtual_rate_manifest_require_commit_match = not bool(
            args.allow_manifest_commit_mismatch)
    apply_m4_variant(
        cfg, args.m4_variant, args.m4_time_mode, args.m4_fixed_delta_t)

    resolved_cfg = dict(cfg)
    resolved_cfg_sha = m0.canonical_sha256(resolved_cfg)
    cfg_sha = m0.sha256_file(cfg_path)
    weights_sha = m0.sha256_file(weights_path)
    split = args.split or cfg.test_split
    version = str(getattr(cfg, "version", "unknown"))
    device = m0.rec_diag.resolve_device(args.device)
    model = m0.load_model(cfg, weights_path, device)
    sampler = m0.get_dataset(
        cfg, type="test", split=split, protocol_role="test")
    dataset = getattr(sampler, "dataset", sampler)

    tracklet_limit = dataset.get_num_tracklets()
    if args.max_tracklets is not None:
        tracklet_limit = min(tracklet_limit, int(args.max_tracklets))

    rows = []
    all_ious = []
    all_distances = []
    start_time = time.time()
    with m0.torch.no_grad():
        for tracklet_id in range(tracklet_limit):
            tracklet_length = dataset.get_num_frames_tracklet(tracklet_id)
            if tracklet_length <= 0:
                continue
            sequence = [
                dataset.get_frames(tracklet_id, [frame_index])[0]
                for frame_index in range(tracklet_length)
            ]
            tracklet_key = m0.stable_tracklet_key(
                dataset, sequence, tracklet_id, version, split)
            ious, distances, predictions = model.evaluate_one_sequence(sequence)
            diagnostics = {
                int(item.get("frame_id", index)): item
                for index, item in enumerate(
                    getattr(model, "_m4_sequence_diagnostics", []))
            }
            if not (
                    len(ious) == len(distances)
                    == len(predictions) == tracklet_length):
                raise RuntimeError(
                    "M4 sequence output length mismatch: "
                    f"tracklet={tracklet_id} frames={tracklet_length} "
                    f"ious={len(ious)} distances={len(distances)} "
                    f"predictions={len(predictions)}")

            for frame_index, (iou, center_error, prediction) in enumerate(
                    zip(ious, distances, predictions)):
                current_gt = sequence[frame_index]["3d_bbox"]
                previous_gt = sequence[max(0, frame_index - 1)]["3d_bbox"]
                previous_prediction = predictions[max(0, frame_index - 1)]
                token = m0.crop_diag.frame_token(
                    sequence[frame_index], f"{tracklet_id}:{frame_index}")
                delta_t = (
                    None if frame_index == 0 else finite_delta_t(
                        sequence, frame_index, args.m4_fixed_delta_t))
                gt_displacement = float(np.linalg.norm(
                    m0.box_center(current_gt) - m0.box_center(previous_gt)))
                previous_error = float(np.linalg.norm(
                    m0.box_center(previous_prediction)
                    - m0.box_center(previous_gt)))
                diagnostic = diagnostics.get(frame_index, {})
                row = {
                    "run_label": args.run_label,
                    "protocol": args.protocol_name,
                    "time_mode": args.m4_time_mode,
                    "m4_variant": args.m4_variant,
                    "tracklet_id": int(tracklet_id),
                    "source_tracklet_id": m0.source_tracklet_id(
                        dataset, tracklet_id),
                    "tracklet_key": tracklet_key,
                    "frame_index": int(frame_index),
                    "source_frame_index": m0.source_frame_index(
                        dataset, tracklet_id, frame_index),
                    "frame_token": token,
                    "is_initial_frame": bool(frame_index == 0),
                    "full_history": bool(frame_index >= int(cfg.hist_num)),
                    "current_timestamp": m0.crop_diag.frame_timestamp(
                        sequence[frame_index], frame_index),
                    "current_delta_t_real": delta_t,
                    "gt_displacement_from_previous_gt": gt_displacement,
                    "previous_prediction_error": previous_error,
                    "iou": float(iou),
                    "center_error": float(center_error),
                    "empty_fallback": (
                        diagnostic.get("reason") == "empty_pointcloud"),
                    "checkpoint_sha256": weights_sha,
                    "config_sha256": cfg_sha,
                    "resolved_config_sha256": resolved_cfg_sha,
                    "version": version,
                    "split": split,
                    "path_variance_available": False,
                }
                row.update(m0.box_fields(
                    "prediction", prediction, bool(cfg.degrees)))
                row.update(m0.box_fields(
                    "ground_truth", current_gt, bool(cfg.degrees)))
                row.update(diagnostic_fields(diagnostic))
                rows.append(row)
                all_ious.append(float(iou))
                all_distances.append(float(center_error))

    if not rows:
        raise RuntimeError("No M4 endpoint rows were produced")

    tag = args.tag or (
        f"{args.run_label}_{args.protocol_name}_"
        f"{args.m4_variant}_{args.m4_time_mode}")
    output_dir = Path(args.output_dir) / m0.crop_diag.safe_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "m0_endpoints.csv"
    summary_path = output_dir / "m0_summary.json"
    resolved_path = output_dir / "resolved_config.json"
    m0.crop_diag.write_rows(csv_path, rows)
    summary = {
        "schema": "ct_seqtrack.m4_endpoint_export",
        "schema_version": 1,
        "run_label": args.run_label,
        "protocol": args.protocol_name,
        "m4_variant": args.m4_variant,
        "m4_time_mode": args.m4_time_mode,
        "cfg": str(cfg_path),
        "cfg_sha256": cfg_sha,
        "resolved_cfg_sha256": resolved_cfg_sha,
        "weights": str(weights_path),
        "weights_sha256": weights_sha,
        "split": split,
        "version": version,
        "seed": args.seed,
        "device": str(device),
        "tracklet_count": tracklet_limit,
        "endpoint_count": len(rows),
        "metrics": m0.tracking_scores(all_ious, all_distances),
        "dataset": m0.dataset_metadata(dataset),
        "git": initial_git,
        "runtime_seconds": float(time.time() - start_time),
        "note": (
            "Predictions come from the actual recursive M4 online path. "
            "Ground truth is used only for offline metrics/oracle diagnostics."),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(
            resolved_cfg, handle, ensure_ascii=False, indent=2,
            allow_nan=False, default=str)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"endpoint csv: {csv_path}")
    print(f"summary json: {summary_path}")


if __name__ == "__main__":
    main()
