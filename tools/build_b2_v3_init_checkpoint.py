#!/usr/bin/env python3
"""Strictly compose B0/B1 and migratable search tensors for B2-v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.selective_innovation_common import (  # noqa: E402
    checkpoint_state_dict,
    sha256_file,
    torch_load,
)


MIGRATED_SUBMODULES = (
    "point_mlp.", "source_embedding.", "query_projection.",
    "key_projection.", "key_norm.", "query_value_projection.",
    "query_norm.", "local_targetness_head.", "vote_head.",
)
B1_PREFIX = "physical_motion_encoder."
SOURCE_SEARCH_PREFIX = "search_evidence_v21."
TARGET_SEARCH_PREFIX = "state_aligned_search_refiner."


def strip_wrapper_prefix(state):
    for prefix in ("model.", "module."):
        if any(key.startswith(prefix) for key in state):
            return {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in state.items()
            }
    return dict(state)


def tensor_prefix_hash(state, prefix):
    digest = hashlib.sha256()
    keys = sorted(key for key in state if key.startswith(prefix))
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), keys


def validate_b1_state(state, expected_tensor_count=14):
    b1_hash, b1_keys = tensor_prefix_hash(state, B1_PREFIX)
    if len(b1_keys) != int(expected_tensor_count):
        raise RuntimeError(
            f"expected exactly {int(expected_tensor_count)} B1 tensors, "
            f"found {len(b1_keys)}")
    return b1_hash, b1_keys


def collect_migrated_search_state(search_state):
    migrated = {}
    counts = {}
    for submodule in MIGRATED_SUBMODULES:
        source_prefix = SOURCE_SEARCH_PREFIX + submodule
        selected = {
            TARGET_SEARCH_PREFIX + key[len(SOURCE_SEARCH_PREFIX):]: value
            for key, value in search_state.items()
            if key.startswith(source_prefix)
        }
        if not selected:
            raise RuntimeError(
                f"search checkpoint is missing {source_prefix}")
        migrated.update(selected)
        counts[submodule.rstrip(".")] = len(selected)
    return migrated, counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build strict two-source B2-v3 initialization")
    parser.add_argument("--base-checkpoint", required=True,
                        help="B2-v2 epoch60 checkpoint supplying B0/B1")
    parser.add_argument("--search-checkpoint", required=True,
                        help="B2-v2.1 full epoch60 checkpoint")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base_payload = torch_load(args.base_checkpoint)
    search_payload = torch_load(args.search_checkpoint)
    base_state = strip_wrapper_prefix(checkpoint_state_dict(base_payload))
    search_state = strip_wrapper_prefix(checkpoint_state_dict(search_payload))

    b1_hash, b1_keys = validate_b1_state(base_state)
    migrated, counts = collect_migrated_search_state(search_state)
    for key, value in migrated.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite migrated tensor: {key}")

    composed_state = dict(base_state)
    composed_state.update(migrated)
    payload = dict(base_payload) if isinstance(base_payload, dict) else {}
    payload["state_dict"] = composed_state
    payload["b2_v3_init"] = {
        "schema": "ct_seqtrack.b2_v3_init.v1",
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_sha256": sha256_file(args.base_checkpoint),
        "search_checkpoint": str(Path(args.search_checkpoint).resolve()),
        "search_sha256": sha256_file(args.search_checkpoint),
        "b1_prefix_hash": b1_hash,
        "b1_keys": b1_keys,
        "migrated_target_keys": sorted(migrated),
        "migrated_submodules": counts,
        "excluded_prefixes": [
            "advantage_proposal_fusion.", "b3_risk_router.",
            "motion_conditioned_search_refiner.", "source_fusion.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "output": str(output),
        "base_tensor_count": len(base_state),
        "b1_tensor_count": len(b1_keys),
        "b1_prefix_hash": b1_hash,
        "migrated_tensor_count": len(migrated),
        "migrated_submodules": counts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
