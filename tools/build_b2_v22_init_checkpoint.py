#!/usr/bin/env python3
"""Compose B0/B1 from B2-v2 with selected Search-v2.1 tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.selective_innovation_common import (  # noqa: E402
    checkpoint_state_dict,
    sha256_file,
    torch_load,
)


MIGRATED_SUBMODULES = (
    "point_mlp.",
    "source_embedding.",
    "query_projection.",
    "key_projection.",
    "key_norm.",
    "query_value_projection.",
    "query_norm.",
    "local_targetness_head.",
    "vote_head.",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the two-source B2-v2.2 initialization checkpoint")
    parser.add_argument("--base-checkpoint", required=True,
                        help="B2-v2 epoch60 checkpoint supplying B0/B1")
    parser.add_argument("--search-checkpoint", required=True,
                        help="B2-v2.1 full epoch60 checkpoint")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def strip_wrapper_prefix(state):
    if any(key.startswith("model.") for key in state):
        return {
            key[len("model."):] if key.startswith("model.") else key: value
            for key, value in state.items()
        }
    if any(key.startswith("module.") for key in state):
        return {
            key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state.items()
        }
    return dict(state)


def main():
    args = parse_args()
    base_payload = torch_load(args.base_checkpoint)
    search_payload = torch_load(args.search_checkpoint)
    base_state = strip_wrapper_prefix(checkpoint_state_dict(base_payload))
    search_state = strip_wrapper_prefix(checkpoint_state_dict(search_payload))
    source_prefix = "search_evidence_v21."
    target_prefix = "motion_conditioned_search_refiner."
    migrated = {}
    counts = {}
    for submodule in MIGRATED_SUBMODULES:
        source = source_prefix + submodule
        selected = {
            target_prefix + key[len(source_prefix):]: value
            for key, value in search_state.items()
            if key.startswith(source)
        }
        if not selected:
            raise RuntimeError(
                f"search checkpoint is missing migratable {source}")
        migrated.update(selected)
        counts[submodule.rstrip(".")] = len(selected)
    composed_state = dict(base_state)
    composed_state.update(migrated)
    payload = dict(base_payload) if isinstance(base_payload, dict) else {}
    payload["state_dict"] = composed_state
    payload["b2_v22_initialization"] = {
        "schema": "ct_seqtrack.b2_v22_init.v1",
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_sha256": sha256_file(args.base_checkpoint),
        "search_checkpoint": str(Path(args.search_checkpoint).resolve()),
        "search_sha256": sha256_file(args.search_checkpoint),
        "migrated_tensor_count": len(migrated),
        "migrated_submodules": counts,
        "excluded_prefixes": [
            "advantage_proposal_fusion.",
            "b3_risk_router.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(payload, output)
    print(json.dumps({
        "output": str(output),
        "base_tensor_count": len(base_state),
        "migrated_tensor_count": len(migrated),
        "migrated_submodules": counts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
