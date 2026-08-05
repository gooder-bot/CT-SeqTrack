#!/usr/bin/env python3
"""Inject a final calibrated B2-v3 router without changing B0/B1/refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v2 import (  # noqa: E402
    B2_V3_PROTECTED_PREFIXES,
    SELECTIVE_V4_ROUTER_SCHEMA,
)
from tools.selective_innovation_common import (  # noqa: E402
    checkpoint_state_dict,
    sha256_file,
    torch_load,
)


ROUTER_PREFIX = "action_consistent_router_v3."
PROTECTED_PREFIXES = B2_V3_PROTECTED_PREFIXES


def tensor_hash(state, prefixes):
    digest = hashlib.sha256()
    keys = sorted(
        key for key in state
        if any(key.startswith(prefix) for prefix in prefixes))
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), keys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--promotion", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base_payload = torch_load(args.candidate_checkpoint)
    router_payload = torch_load(args.router)
    if router_payload.get("schema") != SELECTIVE_V4_ROUTER_SCHEMA:
        raise ValueError("unsupported B2-v3 router sidecar schema")
    calibration = router_payload.get("calibration", {})
    if calibration.get("status") != "passed":
        raise RuntimeError("only a calibration-passed router can be packaged")
    if calibration.get("partition") != "calibration":
        raise RuntimeError(
            "provisional dev-threshold routers cannot be packaged as final")
    if (calibration.get("method") != "recursive_tracklet_scan_v4"
            or calibration.get("final_recursive") is not True
            or int(calibration.get("intervention_count", 0)) <= 0
            or float(calibration.get("harm_rate", 1.0)) > 0.05):
        raise RuntimeError(
            "final packaging requires a non-empty safe recursive tracklet scan")

    candidate_sha256 = sha256_file(args.candidate_checkpoint)
    promotion_path = Path(args.promotion)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if (promotion.get("schema") != "ct_seqtrack.b2_v3_promotion.v2"
            or promotion.get("status") != "passed"):
        raise RuntimeError("router packaging requires a passed B2 promotion")
    if promotion.get("candidate_checkpoint_sha256") != candidate_sha256:
        raise RuntimeError(
            "B2 promotion belongs to a different candidate checkpoint")
    promotion_sha256 = sha256_file(promotion_path)
    if router_payload.get(
            "promotion_manifest_sha256") != promotion_sha256:
        raise RuntimeError(
            "router rollouts used a different B2 promotion manifest")
    if router_payload.get(
            "candidate_checkpoint_sha256") != candidate_sha256:
        raise RuntimeError(
            "router sidecar belongs to a different candidate checkpoint")
    candidate_config_sha256 = base_payload.get(
        "b2_v3_candidate_config_sha256")
    if (not candidate_config_sha256
            or promotion.get("candidate_config_sha256")
            != candidate_config_sha256
            or router_payload.get("candidate_config_sha256")
            != candidate_config_sha256):
        raise RuntimeError(
            "candidate checkpoint/config/promotion/router contract mismatch")
    rollout_manifest = router_payload.get("rollout_manifest", {})
    if (rollout_manifest.get("round") != "merged_0_1"
            or rollout_manifest.get("state_policy")
            != "observation_plus_router"):
        raise RuntimeError(
            "final router must be trained from merged round-0/round-1 rollouts")
    if rollout_manifest.get("candidate_checkpoint_sha256") != candidate_sha256:
        raise RuntimeError(
            "router rollouts were produced by a different candidate checkpoint")

    state = dict(checkpoint_state_dict(base_payload))
    protected_before, protected_keys = tensor_hash(
        state, PROTECTED_PREFIXES)
    router_state = router_payload.get("router_state_dict", {})
    if not router_state:
        raise ValueError("router sidecar contains no state_dict")
    nonfinite_router = sorted(
        key for key, value in router_state.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item()))
    if nonfinite_router:
        raise RuntimeError(
            "router sidecar contains non-finite tensors: "
            + ", ".join(nonfinite_router[:20]))
    normalization = router_payload.get("scalar_normalization", {})
    for metadata_key, state_key in {
            "mean": "scalar_feature_mean",
            "std": "scalar_feature_std",
            "p1": "scalar_clip_low",
            "p99": "scalar_clip_high",
    }.items():
        if state_key not in router_state or metadata_key not in normalization:
            raise RuntimeError(
                f"router sidecar lacks normalization {metadata_key}")
        expected = torch.as_tensor(
            normalization[metadata_key],
            dtype=router_state[state_key].dtype)
        if (expected.shape != router_state[state_key].shape
                or not torch.equal(expected, router_state[state_key])):
            raise RuntimeError(
                f"router normalization mismatch for {metadata_key}")
    existing_router_keys = sorted(
        key for key in state if key.startswith(ROUTER_PREFIX))
    sidecar_target_keys = sorted(ROUTER_PREFIX + key for key in router_state)
    if (existing_router_keys
            and existing_router_keys != sidecar_target_keys):
        missing = sorted(set(existing_router_keys) - set(sidecar_target_keys))
        extra = sorted(set(sidecar_target_keys) - set(existing_router_keys))
        raise RuntimeError(
            f"router key set mismatch; missing={missing}, extra={extra}")
    for key, value in router_state.items():
        target_key = ROUTER_PREFIX + key
        if (target_key in state
                and state[target_key].shape != value.shape):
            raise ValueError(f"router shape mismatch for {target_key}")
        state[target_key] = value
    protected_after, _ = tensor_hash(state, PROTECTED_PREFIXES)
    if protected_before != protected_after:
        raise RuntimeError("packaging changed B0/B1/refiner tensors")

    payload = dict(base_payload) if isinstance(base_payload, dict) else {}
    payload["state_dict"] = state
    payload["b2_v3_router_package"] = {
        "schema": "ct_seqtrack.selective_checkpoint.v4",
        "candidate_checkpoint": str(
            Path(args.candidate_checkpoint).resolve()),
        "candidate_checkpoint_sha256": candidate_sha256,
        "router_sidecar": str(Path(args.router).resolve()),
        "router_sidecar_sha256": sha256_file(args.router),
        "router_tensor_count": len(sidecar_target_keys),
        "protected_tensor_count": len(protected_keys),
        "protected_prefix_hash": protected_after,
        "calibration": calibration,
        "feature_schema": router_payload.get("feature_schema"),
        "feature_schema_hash": router_payload.get("feature_schema_hash"),
        "scalar_normalization": router_payload.get(
            "scalar_normalization"),
        "action_names": router_payload.get("action_names"),
        "step_ratios": router_payload.get("step_ratios"),
        "candidate_config_sha256": router_payload.get(
            "candidate_config_sha256"),
        "promotion_manifest": str(promotion_path.resolve()),
        "promotion_manifest_sha256": promotion_sha256,
        "rollout_hashes": router_payload.get("rollout_hashes"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "output": str(output),
        "router_tensor_count": len(sidecar_target_keys),
        "protected_prefix_hash": protected_after,
        "calibration": calibration,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
