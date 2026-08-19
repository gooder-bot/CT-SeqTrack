#!/usr/bin/env python3
"""Export checkpoint-free causal fixed-CV acquisition rows for preflight.

Past observations seed the geometry-only causal state.  Current GT is hidden
until candidate gaps, anchors, support and crops are fixed, then used only for
the exported supervision/count labels.  No checkpoint or learned score is
loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset
from datasets.sampler import motion_processing_mf
import datasets.points_utils as points_utils
from datasets.misc_utils import build_time_fields
from ctseqtrack.config import configure_ct_variant
from ctseqtrack.runtime.contracts import validate_scratch_training_contract
from utils.config import load_yaml_config
from ctseqtrack.data.recursive import (
    RecursiveTrackState,
    apply_training_reanchor,
    build_scene_partition_manifest,
    build_recursive_input_contract,
    scene_partition_tracklet_ids,
    stable_tracklet_partition,
)
from utils.ct_history import (
    normalize_causal_temporal_gaps,
    select_causal_temporal_candidates,
    select_uniform_temporal_candidates,
)
from ctseqtrack.data.search import estimate_ordered_trajectory


def fixed_cv_prediction(raw, state, config):
    contract = build_recursive_input_contract(
        state,
        raw["this_frame_id"],
        len(raw["prev_frame_ids"]),
        config,
        candidate_id=raw["candidate_id"],
        offsets=raw["history_offsets"],
    )
    history = state.history_boxes(
        contract["history_frame_ids"], contract["history_valid_mask"].tolist()
    )
    real_fields = build_time_fields(
        list(contract["history_timestamps"]),
        raw["this_frame"].get("timestamp"),
        frame_ids=contract["history_frame_ids"],
        current_frame_id=raw["this_frame_id"],
        use_real_time=bool(getattr(config, "use_real_time", True)),
        default_step=float(getattr(config, "default_time_step", 0.5)),
        pseudo_step=float(getattr(config, "pseudo_time_step", 0.1)),
    )
    estimate = estimate_ordered_trajectory(
        history,
        real_fields[1],
        contract["history_valid_mask"],
        max_speed=float(getattr(config, "ct_motion_max_speed", 20.0)),
        max_acceleration=float(getattr(config, "ct_motion_max_acceleration", 8.0)),
        max_displacement=float(getattr(config, "ct_motion_max_displacement", 12.0)),
        acceleration_weight=float(
            getattr(config, "ct_motion_acceleration_weight", 0.5)
        ),
        require_recent_transition=True,
    )
    if not estimate.get("valid", False):
        return None
    world_displacement = np.asarray(estimate["displacement_vector"], dtype=np.float64)
    local = np.asarray(history[0].rotation_matrix, dtype=np.float64).T @ (
        world_displacement
    )
    current_delta_t = max(float(estimate["query_delta_t"]), 1e-6)
    norm = float(np.linalg.norm(local[:2]))
    direction = (
        local[:2] / norm if norm > 1e-9 else np.asarray((1.0, 0.0), dtype=np.float64)
    )
    return {
        "valid": True,
        "mu_xy": local[:2].astype(np.float32),
        "velocity_xy": (local[:2] / current_delta_t).astype(np.float32),
        "direction_xy": direction.astype(np.float32),
        "log_sigma_parallel_perp": np.log(
            np.maximum(
                np.asarray(
                    (estimate["sigma_parallel"], estimate["sigma_perpendicular"]),
                    dtype=np.float32,
                ),
                1e-3,
            )
        ),
        "current_delta_t": current_delta_t,
        "gap_ratio": float(estimate["gap_ratio"]),
        "source_id": 2,
    }


def temporal_view(raw, gap, candidate_id):
    entry = raw["temporal_candidate_pool"][int(gap)]
    view = dict(raw)
    view.update(
        {
            "candidate_id": int(candidate_id),
            "candidate_gap_frames": int(gap),
            "prev_frames": entry["prev_frames"],
            "prev_frame_ids": list(entry["prev_frame_ids"]),
            "valid_mask": list(entry["valid_mask"]),
            "history_offsets": list(entry["history_offsets"]),
            "point_sampling_seeds": entry["point_sampling_seeds"],
            "current_sampling_seed": entry["current_sampling_seed"],
            "candidate_shared_transform": np.zeros(3, dtype=np.float32),
            "shadow_future": [],
        }
    )
    return view


def expand_raw(raw, state, config):
    gaps = normalize_causal_temporal_gaps(
        getattr(config, "ct_temporal_candidate_gaps", [2, 4, 8])
    )
    predictions = {}
    ratios = {}
    available = {}
    half_extent = np.maximum(
        0.5
        * np.asarray(state.target_size[:2], dtype=np.float64)
        * float(config.bb_scale)
        + float(config.bb_offset),
        1e-3,
    )
    for gap in [1] + gaps:
        view = temporal_view(raw, gap, 0)
        prediction = fixed_cv_prediction(view, state, config)
        predictions[gap] = prediction
        if gap != 1:
            endpoint = (
                np.zeros(2, dtype=np.float64)
                if prediction is None
                else np.asarray(prediction["mu_xy"], dtype=np.float64)
            )
            ratios[gap] = float(np.max(np.abs(endpoint) / half_extent))
            available[gap] = bool(
                all(int(value) for value in view["valid_mask"])
                and prediction is not None
            )
    policy = (
        str(getattr(config, "ct_candidate_policy", "causal_b1_boundary"))
        .strip()
        .lower()
    )
    if policy == "causal_temporal_uniform":
        selected = select_uniform_temporal_candidates(
            ratios,
            available,
            seed_parts=(
                int(getattr(config, "seed", 42) or 42),
                int(raw.get("online_epoch", 0)),
                str(raw["tracklet_key"]),
                int(raw["this_frame_id"]),
            ),
        )
    else:
        selected = select_causal_temporal_candidates(
            ratios,
            available,
            boundary_band=float(getattr(config, "ct_temporal_boundary_band", 0.2)),
        )
    canonical = temporal_view(raw, 1, 0)
    canonical.update(
        {
            "candidate_role": 0,
            "candidate_available": 1.0,
            "candidate_boundary_ratio": float(
                np.max(
                    np.abs(
                        predictions[1]["mu_xy"]
                        if predictions[1] is not None
                        else np.zeros(2)
                    )
                    / half_extent
                )
            ),
            "candidate_role_satisfied": 1.0,
            "candidate_gap_pool_ratios": {
                int(gap): float(ratio) for gap, ratio in ratios.items()
            },
        }
    )
    output = [(canonical, predictions[1])]
    for role_id in (1, 2):
        role = selected[role_id]
        gap = gaps[role_id - 1] if role["gap"] is None else int(role["gap"])
        view = temporal_view(raw, gap, role_id)
        view.update(
            {
                "candidate_role": role_id,
                "candidate_available": float(role["available"]),
                "candidate_boundary_ratio": float(role["boundary_ratio"]),
                "candidate_role_satisfied": float(role["role_satisfied"]),
            }
        )
        output.append((view, predictions[gap]))
    return output


def process_raw(raw, state, config, motion_prediction):
    contract = build_recursive_input_contract(
        state,
        raw["this_frame_id"],
        len(raw["prev_frame_ids"]),
        config,
        candidate_id=raw["candidate_id"],
        offsets=raw["history_offsets"],
    )
    candidate_id = int(raw["candidate_id"])
    contract["candidate_shared_transform"] = np.zeros(3, dtype=np.float32)
    contract["point_sampling_seeds"] = raw["point_sampling_seeds"]
    contract["current_sampling_seed"] = raw["current_sampling_seed"]
    payload = {
        key: value
        for key, value in raw.items()
        if key
        not in (
            "online_recursive_raw",
            "online_epoch",
            "online_batch_index",
            "online_slot",
            "shadow_future",
            "temporal_candidate_pool",
        )
    }
    payload.update(
        {
            "online_recursive_state": contract,
            "candidate_shared_transform": contract["candidate_shared_transform"],
            "point_sampling_seeds": contract["point_sampling_seeds"],
            "current_sampling_seed": contract["current_sampling_seed"],
            "motion_prediction": motion_prediction,
        }
    )
    started = time.perf_counter()
    processed = motion_processing_mf(payload, config)
    processing_time_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
    return {
        "candidate_id": candidate_id,
        "candidate_gap_frames": int(raw["candidate_gap_frames"]),
        "candidate_role": int(raw["candidate_role"]),
        "candidate_available": bool(raw["candidate_available"]),
        "boundary_ratio": float(raw["candidate_boundary_ratio"]),
        "role_satisfied": bool(raw["candidate_role_satisfied"]),
        "candidate_gap_pool_ratios": dict(raw.get("candidate_gap_pool_ratios", {})),
        "base_target_count": float(processed["ct_acquisition_base_target_count"]),
        "pool_target_count": float(
            processed["ct_acquisition_extension_pool_target_count"]
        ),
        "sampled_target_count": float(processed["ct_acquisition_sampled_target_count"]),
        "extension_pool_count": float(processed["ct_acquisition_extension_pool_count"]),
        "sampled_count": float(processed["ct_acquisition_sampled_count"]),
        "evidence_raw_point_count": float(processed["ct_evidence_raw_point_count"]),
        "evidence_base_unique_count": float(processed["ct_evidence_base_unique_count"]),
        "evidence_extension_unique_count": float(
            processed["ct_evidence_extension_unique_count"]
        ),
        "evidence_foreground_count": float(processed["ct_evidence_foreground_count"]),
        "available": bool(raw["candidate_available"]),
        "support_truncated": bool(processed.get("search_v3_support_truncated", False)),
        "processing_time_ms": float(processing_time_ms),
        "tracklet_key": str(raw["tracklet_key"]),
        "partition_group_key": str(raw.get("partition_group_key", raw["tracklet_key"])),
        "frame_id": int(raw["this_frame_id"]),
        "rollout_horizon": int(raw["rollout_horizon"]),
        "state_source": str(raw["state_source"]),
    }


def _fixed_cv_state_box(raw, state, config, prediction):
    """Convert the canonical fixed-CV displacement into the next state."""
    history_id = int(raw["prev_frame_ids"][0])
    anchor = state.history_boxes([history_id], [1])[0]
    if prediction is None or not bool(prediction.get("valid", False)):
        return copy.deepcopy(anchor)
    mu_xy = np.asarray(prediction["mu_xy"], dtype=np.float64).reshape(2)
    if not np.isfinite(mu_xy).all():
        return copy.deepcopy(anchor)
    return points_utils.getOffsetBB(
        anchor,
        [float(mu_xy[0]), float(mu_xy[1]), 0.0, 0.0],
        degrees=bool(config.degrees),
        use_z=False,
        limit_box=False,
    )


def _partition_tracklet_ids(base, config, partition, scene_manifest):
    if (
        str(getattr(config, "ct_partition_scheme", "tracklet_v1")).strip().lower()
        == "scene_v2"
    ):
        return scene_partition_tracklet_ids(scene_manifest, partition)
    selected = []
    seed = int(config.ct_partition_seed)
    for tracklet_id in range(base.get_num_tracklets()):
        key = (
            base.get_tracklet_key(tracklet_id)
            if hasattr(base, "get_tracklet_key")
            else str(tracklet_id)
        )
        if stable_tracklet_partition(key, seed) == partition:
            selected.append(tracklet_id)
    if not selected:
        raise ValueError(f"preflight partition {partition!r} is empty")
    return selected


def export_partition(dataset, config, partition, max_batches, scene_manifest=None):
    base = dataset.dataset
    tracklet_ids = _partition_tracklet_ids(base, config, partition, scene_manifest)
    horizons = [
        int(value)
        for value in getattr(config, "ct_recursive_rollout_horizons", [1, 2, 4, 8])
    ]
    rows = []
    query_count = 0
    stopped = False
    for tracklet_id in tracklet_ids:
        frame_count = int(base.get_num_frames_tracklet(tracklet_id))
        tracklet_key = str(
            base.get_tracklet_key(tracklet_id)
            if hasattr(base, "get_tracklet_key")
            else tracklet_id
        )
        group_key = str(
            base.get_partition_group_key(tracklet_id)
            if hasattr(base, "get_partition_group_key")
            else tracklet_key
        )
        first_frame = base.get_frames(tracklet_id, frame_ids=(0,))[0]
        for horizon_index, horizon in enumerate(horizons):
            state = RecursiveTrackState(
                int(tracklet_id),
                tracklet_key,
                first_frame["3d_bbox"],
                timestamps={0: first_frame.get("timestamp")},
            )
            for frame_id in range(1, frame_count):
                if max_batches is not None and query_count >= max_batches:
                    stopped = True
                    break
                raw = dataset._online_raw_view(
                    0,
                    query_count,
                    horizon_index,
                    tracklet_id,
                    frame_id,
                    candidate_id=0,
                    build_shadow=False,
                )
                raw["partition_group_key"] = group_key
                raw["rollout_horizon"] = horizon
                raw["state_source"] = "fixed_cv_recursive_prediction"
                apply_training_reanchor(raw, state, horizon, config)
                expanded = expand_raw(raw, state, config)
                for view, prediction in expanded:
                    view["partition_group_key"] = group_key
                    view["rollout_horizon"] = horizon
                    view["state_source"] = "fixed_cv_recursive_prediction"
                    row = process_raw(view, state, config, prediction)
                    row["partition"] = partition
                    rows.append(row)
                canonical_prediction = expanded[0][1]
                predicted_box = _fixed_cv_state_box(
                    expanded[0][0], state, config, canonical_prediction
                )
                if max(state.predictions) < frame_id:
                    state.append(
                        frame_id, predicted_box, raw["this_frame"].get("timestamp")
                    )
                query_count += 1
            if stopped:
                break
        if stopped:
            break
    tracklet_identity = [
        {
            "tracklet_key": str(
                base.get_tracklet_key(tracklet_id)
                if hasattr(base, "get_tracklet_key")
                else tracklet_id
            ),
            "frame_count": int(base.get_num_frames_tracklet(tracklet_id)),
        }
        for tracklet_id in sorted(tracklet_ids)
    ]
    identity_sha256 = hashlib.sha256(
        json.dumps(
            tracklet_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    expected_rows = (
        sum(
            max(0, int(base.get_num_frames_tracklet(tracklet_id)) - 1)
            for tracklet_id in tracklet_ids
        )
        * len(horizons)
        * 3
    )
    dropped_rows = max(0, expected_rows - len(rows))
    return rows, {
        "partition": partition,
        "tracklet_count": len(tracklet_ids),
        "prediction_frames": sum(
            max(0, int(base.get_num_frames_tracklet(tracklet_id)) - 1)
            for tracklet_id in tracklet_ids
        ),
        "horizons": horizons,
        "exported_rows": len(rows),
        "expected_rows": expected_rows,
        "dropped_rows": dropped_rows,
        "complete": bool(max_batches is None and dropped_rows == 0),
        "tracklet_identity_sha256": identity_sha256,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-manifest-output", required=True)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dynamics-time-manifest")
    reseed = parser.add_mutually_exclusive_group()
    reseed.add_argument(
        "--ct-reseed-enabled",
        dest="ct_recursive_reseed_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    reseed.add_argument(
        "--ct-no-reseed",
        dest="ct_recursive_reseed_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest_output = Path(args.data_manifest_output).resolve()
    if output.exists() or manifest_output.exists():
        raise FileExistsError("preflight outputs must not already exist")
    config = EasyDict(load_yaml_config(args.config))
    configure_ct_variant(config)
    candidate_policy = str(getattr(config, "ct_candidate_policy", "")).strip().lower()
    if candidate_policy not in ("causal_b1_boundary", "causal_temporal_uniform"):
        raise ValueError(
            "formal acquisition preflight requires a causal temporal policy"
        )
    if (
        int(getattr(config, "num_candidates", 0)) != 3
        or int(getattr(config, "ct_recursive_candidate_views", 0)) != 3
    ):
        raise ValueError("formal acquisition preflight requires exactly c0/c1/c2")
    if (
        str(getattr(config, "ct_recovery_candidate_policy", "off")).strip().lower()
        != "off"
    ):
        raise ValueError("formal acquisition preflight rejects GT-spatial recovery")
    if args.path is not None:
        config.path = args.path
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.dynamics_time_manifest is not None:
        config.dynamics_time_manifest = args.dynamics_time_manifest
    if hasattr(args, "ct_recursive_reseed_enabled"):
        config.ct_recursive_reseed_enabled = bool(args.ct_recursive_reseed_enabled)
    validate_scratch_training_contract(config)
    dataset = get_dataset(
        config, type=config.train_type, split=config.train_split, protocol_role="train"
    )
    rows = []
    partitions = []
    scene_manifest = None
    if (
        str(getattr(config, "ct_partition_scheme", "tracklet_v1")).strip().lower()
        == "scene_v2"
    ):
        scene_manifest = build_scene_partition_manifest(
            dataset.dataset, int(config.ct_partition_seed)
        )
    for partition in ("train", "dev"):
        part_rows, identity = export_partition(
            dataset, config, partition, args.max_batches, scene_manifest
        )
        rows.extend(part_rows)
        partitions.append(identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    protocol_version = int(getattr(config, "ct_protocol_version", 24))
    manifest = {
        "schema": (
            "ct_seqtrack.acquisition_data_manifest.v3"
            if protocol_version >= 25
            else "ct_seqtrack.acquisition_data_manifest.v2"
        ),
        "dataset": str(config.dataset),
        "split": str(config.train_split),
        "path": str(config.path),
        "seed": int(config.seed or 42),
        "history_source": (
            "fixed_cv_recursive_predictions_with_periodic_past_gt_reanchor"
            if protocol_version >= 25
            else "past_observation_fixed_cv_causal_geometry_audit"
        ),
        "state_write_source": (
            "fixed_cv_prediction" if protocol_version >= 25 else None
        ),
        "current_gt_role": {
            "candidate0": "target-count-label-only",
            "candidate1_2": "target-count-label-only",
        },
        "candidate_selection": (
            "fixed-cv endpoint boundary/outside; current GT hidden"
            if candidate_policy == "causal_b1_boundary"
            else "uniform valid temporal gaps; current GT hidden"
        ),
        "checkpoint_loaded": False,
        "complete": bool(all(item["complete"] for item in partitions)),
        "dropped_rows": sum(item["dropped_rows"] for item in partitions),
        "scene_partition_manifest_sha256": (
            scene_manifest.get("content_sha256")
            if isinstance(scene_manifest, dict)
            else None
        ),
        "scene_partition_manifest": scene_manifest,
        "max_batches": args.max_batches,
        "partitions": partitions,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
