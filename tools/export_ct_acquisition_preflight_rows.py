#!/usr/bin/env python3
"""Export checkpoint-free fixed-CV acquisition rows for preflight.

Past GT boxes are used only as the observed history of this geometry audit.
Current GT labels canonical target retention only.  No spatial candidate,
B0/B1 model, checkpoint, learned score or router is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset
from datasets.sampler import motion_processing_mf
from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.recursive_state import (
    OnlineRecursiveBatchSampler,
    RecursiveTrackState,
    build_recursive_input_contract,
)
def process_raw(raw, state, config):
    contract = build_recursive_input_contract(
        state, raw["this_frame_id"], len(raw["prev_frame_ids"]), config,
        candidate_id=raw["candidate_id"], offsets=raw["history_offsets"],
        epoch=raw.get("online_epoch", 0))
    candidate_id = int(raw["candidate_id"])
    if candidate_id != 0:
        raise RuntimeError("B2 preflight accepts canonical candidate0 only")
    payload = {
        key: value for key, value in raw.items()
        if key not in ("online_recursive_raw", "online_epoch",
                       "online_batch_index", "online_slot",
                       "shadow_future")}
    payload.update({
        "online_recursive_state": contract,
        "candidate_shared_transform": contract[
            "candidate_shared_transform"],
        "point_sampling_seeds": contract["point_sampling_seeds"],
        "current_sampling_seed": contract["current_sampling_seed"],
        # Absence forces the same constrained-CV path used when B1 is invalid.
        "motion_prediction": None,
    })
    processed = motion_processing_mf(payload, config)
    return {
        "candidate_id": candidate_id,
        "base_target_count": float(processed[
            "ct_acquisition_base_target_count"]),
        "pool_target_count": float(processed[
            "ct_acquisition_extension_pool_target_count"]),
        "sampled_target_count": float(processed[
            "ct_acquisition_sampled_target_count"]),
        "extension_pool_count": float(processed[
            "ct_acquisition_extension_pool_count"]),
        "sampled_count": float(processed[
            "ct_acquisition_sampled_count"]),
        "available": bool(float(processed[
            "ct_acquisition_extension_pool_count"]) > 0),
        "recovery_positive": bool(float(processed[
            "ct_recovery_positive"]) > 0),
        "recovery_fallback": bool(float(processed[
            "ct_recovery_fallback"]) > 0),
        "tracklet_key": str(raw["tracklet_key"]),
        "frame_id": int(raw["this_frame_id"]),
    }


def export_partition(dataset, config, partition, max_batches):
    sampler = OnlineRecursiveBatchSampler(
        dataset,
        slots=int(config.ct_recursive_tracklet_slots),
        candidate_views=int(config.ct_recursive_candidate_views),
        seed=int(config.seed or 42),
        partition_seed=int(config.ct_partition_seed),
        partition=partition,
        shadow_enabled=False,
    )
    states = {}
    rows = []
    for batch_index, indices in enumerate(sampler):
        if max_batches is not None and batch_index >= max_batches:
            break
        raw_items = [dataset[index] for index in indices]
        raw_items = [
            raw for raw in raw_items if int(raw["candidate_id"]) == 0]
        groups = {}
        for raw in raw_items:
            key = str(raw["tracklet_key"])
            state = states.get(key)
            if state is None:
                state = RecursiveTrackState(
                    int(raw["tracklet_id"]), key,
                    raw["first_frame"]["3d_bbox"],
                    timestamps={0: raw["first_frame"].get("timestamp")})
                states[key] = state
            row = process_raw(raw, state, config)
            row["partition"] = partition
            rows.append(row)
            groups.setdefault((key, int(raw["this_frame_id"])), raw)
        # Geometry preflight has no tracker weights.  Once this query has
        # been counted, append its observation (past GT for the next query)
        # so the next fixed-CV support has a causally ordered history.
        for (key, frame_id), raw in groups.items():
            state = states[key]
            if max(state.predictions) < frame_id:
                state.append(
                    frame_id, raw["this_frame"]["3d_bbox"],
                    raw["this_frame"].get("timestamp"))
    base = sampler.dataset.dataset
    tracklet_identity = [
        {
            "tracklet_key": str(
                base.get_tracklet_key(tracklet_id)
                if hasattr(base, "get_tracklet_key") else tracklet_id),
            "frame_count": int(base.get_num_frames_tracklet(tracklet_id)),
        }
        for tracklet_id in sorted(sampler.tracklet_ids)
    ]
    identity_sha256 = hashlib.sha256(json.dumps(
        tracklet_identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    rows_per_batch = sampler.slots
    expected_rows = len(sampler) * rows_per_batch
    return rows, {
        "partition": partition,
        "tracklet_count": len(sampler.tracklet_ids),
        "prediction_frames": sampler.prediction_frames,
        "exported_rows": len(rows),
        "expected_rows": expected_rows,
        "complete": bool(
            max_batches is None and len(rows) == expected_rows),
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
        "--ct-reseed-enabled", dest="ct_recursive_reseed_enabled",
        action="store_true", default=argparse.SUPPRESS)
    reseed.add_argument(
        "--ct-no-reseed", dest="ct_recursive_reseed_enabled",
        action="store_false", default=argparse.SUPPRESS)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest_output = Path(args.data_manifest_output).resolve()
    if output.exists() or manifest_output.exists():
        raise FileExistsError("preflight outputs must not already exist")
    config = EasyDict(load_yaml_config(args.config))
    configure_ct_variant(config)
    if args.path is not None:
        config.path = args.path
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.dynamics_time_manifest is not None:
        config.dynamics_time_manifest = args.dynamics_time_manifest
    if hasattr(args, "ct_recursive_reseed_enabled"):
        config.ct_recursive_reseed_enabled = bool(
            args.ct_recursive_reseed_enabled)
    dataset = get_dataset(
        config, type=config.train_type, split=config.train_split,
        protocol_role="train")
    rows = []
    partitions = []
    for partition in ("train", "dev"):
        part_rows, identity = export_partition(
            dataset, config, partition, args.max_batches)
        rows.extend(part_rows)
        partitions.append(identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8")
    manifest = {
        "schema": "ct_seqtrack.acquisition_data_manifest.v2",
        "dataset": str(config.dataset),
        "split": str(config.train_split),
        "path": str(config.path),
        "seed": int(config.seed or 42),
        "history_source": "past_observation_gt_fixed_cv_geometry_audit",
        "current_gt_role": {
            "candidate0": "canonical-target-count-label-only",
        },
        "checkpoint_loaded": False,
        "complete": bool(all(item["complete"] for item in partitions)),
        "max_batches": args.max_batches,
        "partitions": partitions,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
