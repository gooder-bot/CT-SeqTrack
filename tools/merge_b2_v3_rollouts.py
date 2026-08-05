#!/usr/bin/env python3
"""Merge observation-policy and one-round on-policy B2-v3 rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.selective_v3_common import (  # noqa: E402
    load_v3_rollout_artifact,
    write_v3_rollout_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-rollouts", required=True)
    parser.add_argument("--on-policy-rollouts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def rows_from_arrays(arrays):
    numeric = [
        "tracklet_id", "frame_id", "router_features", "candidate_valid",
        "candidate_residual_xy", "signed_gain", "candidate_cost",
        "observation_cost", "rollout_length",
    ]
    numeric.extend(key for key in (
        "candidate_success", "observation_success", "success_gain")
                   if key in arrays)
    for index in range(arrays["router_features"].shape[0]):
        row = {key: arrays[key][index] for key in numeric}
        row["tracklet_key"] = str(arrays["tracklet_key"][index])
        row["partition"] = str(arrays["partition"][index])
        yield row


def main():
    args = parse_args()
    observation, observation_manifest, observation_hashes = (
        load_v3_rollout_artifact(args.observation_rollouts))
    on_policy, on_policy_manifest, on_policy_hashes = (
        load_v3_rollout_artifact(args.on_policy_rollouts))
    if (observation_manifest.get("round") != 0
            or observation_manifest.get("state_policy") != "observation"):
        raise ValueError("first artifact must be round-0 observation policy")
    if (on_policy_manifest.get("round") != 1
            or on_policy_manifest.get("state_policy") != "router"):
        raise ValueError("second artifact must be round-1 router policy")
    invariant_fields = [
        "candidate_checkpoint_sha256", "config_sha256", "split", "seed",
        "horizon", "gamma", "tracklets_evaluated", "partition_tracklets"]
    formal_v4 = observation_manifest.get(
        "schema") == "ct_seqtrack.selective_rollout.v4"
    if formal_v4:
        invariant_fields.extend((
            "feature_schema_hash", "promotion_manifest_sha256"))
    for field in invariant_fields:
        if observation_manifest.get(field) != on_policy_manifest.get(field):
            raise ValueError(f"rollout rounds disagree on {field}")
    round1_calibration = on_policy_manifest.get(
        "state_policy_calibration", {})
    if (round1_calibration.get("status") != "passed"
            or round1_calibration.get("partition") != "dev"):
        raise ValueError(
            "round-1 rollout must use a passed dev-calibrated provisional router")

    partition_by_tracklet = {}
    for arrays in (observation, on_policy):
        for key, partition in zip(
                arrays["tracklet_key"].astype(str),
                arrays["partition"].astype(str)):
            previous = partition_by_tracklet.setdefault(key, partition)
            if previous != partition:
                raise RuntimeError(
                    f"tracklet partition changed across rounds: {key}")
    rows = list(rows_from_arrays(observation))
    rows.extend(rows_from_arrays(on_policy))
    manifest = {
        "schema": observation_manifest["schema"],
        "candidate_checkpoint": observation_manifest[
            "candidate_checkpoint"],
        "candidate_checkpoint_sha256": observation_manifest[
            "candidate_checkpoint_sha256"],
        "config_path": observation_manifest["config_path"],
        "config_sha256": observation_manifest["config_sha256"],
        "split": observation_manifest["split"],
        "seed": observation_manifest["seed"],
        "horizon": observation_manifest["horizon"],
        "gamma": observation_manifest["gamma"],
        "round": "merged_0_1",
        "state_policy": "observation_plus_router",
        "policy_after_intervention": "explicit_observation",
        "source_rounds": [
            {"manifest": observation_manifest, "hashes": observation_hashes},
            {"manifest": on_policy_manifest, "hashes": on_policy_hashes},
        ],
    }
    if formal_v4:
        manifest.update({
            "candidate_config_sha256": observation_manifest[
                "candidate_config_sha256"],
            "promotion_manifest_sha256": observation_manifest[
                "promotion_manifest_sha256"],
            "feature_schema": observation_manifest["feature_schema"],
            "feature_schema_hash": observation_manifest[
                "feature_schema_hash"],
        })
    npz_path, manifest_path = write_v3_rollout_artifact(
        args.output, rows, manifest)
    print(json.dumps({
        "rollout": str(npz_path),
        "manifest": str(manifest_path),
        "row_count": len(rows),
        "round0_rows": int(observation["router_features"].shape[0]),
        "round1_rows": int(on_policy["router_features"].shape[0]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
