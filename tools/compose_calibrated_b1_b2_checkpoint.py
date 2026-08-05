#!/usr/bin/env python3
"""Compose calibrated B1 tensors with a trained dual-query B2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_b2_v3_init_checkpoint import (  # noqa: E402
    strip_wrapper_prefix,
    tensor_prefix_hash,
)
from tools.selective_innovation_common import (  # noqa: E402
    checkpoint_state_dict,
    sha256_file,
    torch_load,
)


PREFIX = "physical_motion_encoder."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2-checkpoint", required=True)
    parser.add_argument("--calibrated-b1-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    b2_payload = torch_load(args.b2_checkpoint)
    b1_payload = torch_load(args.calibrated_b1_checkpoint)
    b2_state = strip_wrapper_prefix(checkpoint_state_dict(b2_payload))
    b1_state = strip_wrapper_prefix(checkpoint_state_dict(b1_payload))
    b1_tensors = {
        key: value for key, value in b1_state.items()
        if key.startswith(PREFIX)}
    if len(b1_tensors) != 15:
        raise RuntimeError(
            "calibrated B1 checkpoint must contain exactly 15 tensors")
    if not any(key.endswith("log_sigma_calibration") for key in b1_tensors):
        raise RuntimeError("calibrated B1 scale buffer is missing")
    for key in [key for key in b2_state if key.startswith(PREFIX)]:
        b2_state.pop(key)
    b2_state.update(b1_tensors)
    b1_hash, b1_keys = tensor_prefix_hash(b2_state, PREFIX)
    payload = dict(b2_payload) if isinstance(b2_payload, dict) else {}
    payload["state_dict"] = b2_state
    metadata = dict(payload.get("b2_v3_init", {}))
    if not metadata:
        raise RuntimeError(
            "B2 checkpoint lacks strict b2_v3_init provenance")
    metadata.update({
        "b1_prefix_hash": b1_hash,
        "b1_keys": b1_keys,
        "calibrated_b1_checkpoint": str(
            Path(args.calibrated_b1_checkpoint).resolve()),
        "calibrated_b1_sha256": sha256_file(
            args.calibrated_b1_checkpoint),
    })
    payload["b2_v3_init"] = metadata
    calibration = b1_payload.get(
        "b1_uncertainty_calibration") if isinstance(
            b1_payload, dict) else None
    if (not isinstance(calibration, dict)
            or calibration.get("schema")
            != "ct_seqtrack.b1_uncertainty_calibration.v2"
            or len(calibration.get(
                "fixed_margin_parallel_perpendicular_95", [])) != 2):
        raise RuntimeError(
            "B1 checkpoint lacks a verified v2 calibration artifact and "
            "fixed residual margins")
    payload["b1_uncertainty_calibration"] = calibration
    payload["b1_b2_composition"] = {
        "schema": "ct_seqtrack.b1_b2_composition.v1",
        "b2_checkpoint": str(Path(args.b2_checkpoint).resolve()),
        "b2_sha256": sha256_file(args.b2_checkpoint),
        "calibrated_b1_checkpoint": str(
            Path(args.calibrated_b1_checkpoint).resolve()),
        "calibrated_b1_sha256": sha256_file(
            args.calibrated_b1_checkpoint),
        "b1_prefix_hash": b1_hash,
        "b1_tensor_count": len(b1_keys),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps(payload["b1_b2_composition"], indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
