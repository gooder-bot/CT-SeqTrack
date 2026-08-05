#!/usr/bin/env python3
"""Expand a trained B2-v3 point encoder from 9 to 10 inputs losslessly."""

from __future__ import annotations

import argparse
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


SUFFIX = "state_aligned_search_refiner.point_mlp.0.weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch_load(args.checkpoint)
    state = checkpoint_state_dict(payload)
    keys = [key for key in state if key.endswith(SUFFIX)]
    if len(keys) != 1:
        raise RuntimeError(
            "checkpoint must contain exactly one B2 point input weight")
    key = keys[0]
    weight = state[key]
    if tuple(weight.shape) != (64, 9):
        raise RuntimeError(
            f"expected B2 point weight [64,9], got {tuple(weight.shape)}")
    expanded = weight.new_zeros((64, 10))
    expanded[:, :9].copy_(weight)
    state[key] = expanded
    if isinstance(payload, dict) and "state_dict" in payload:
        payload["state_dict"] = state
    else:
        payload = state
    if isinstance(payload, dict):
        payload["b2_geometry_expansion"] = {
            "schema": "ct_seqtrack.b2_geometry.v1",
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "source_sha256": sha256_file(args.checkpoint),
            "expanded_key": key,
            "initialization": "append_exact_zero_mahalanobis_column",
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps(payload["b2_geometry_expansion"], indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
