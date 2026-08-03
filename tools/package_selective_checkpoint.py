#!/usr/bin/env python3
"""Package a passed signed-router sidecar into a frozen B2-v2.2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v2 import SELECTIVE_ROUTER_SCHEMA  # noqa: E402
from tools.selective_innovation_common import (  # noqa: E402
    checkpoint_state_dict,
    sha256_file,
    torch_load,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base_payload = torch_load(args.candidate_checkpoint)
    router_payload = torch_load(args.router)
    if router_payload.get("schema") != SELECTIVE_ROUTER_SCHEMA:
        raise ValueError("unsupported signed-router sidecar schema")
    calibration = router_payload.get("calibration", {})
    if calibration.get("status") != "passed":
        raise RuntimeError("only a calibration-passed router can be packaged")
    state = dict(checkpoint_state_dict(base_payload))
    router_state = router_payload.get("router_state_dict", {})
    if not router_state:
        raise ValueError("router sidecar contains no state_dict")
    replaced = 0
    for key, value in router_state.items():
        target_key = "signed_horizon_router." + key
        if target_key not in state:
            raise KeyError(
                f"candidate checkpoint is missing router tensor {target_key}")
        if state[target_key].shape != value.shape:
            raise ValueError(f"router tensor shape mismatch for {target_key}")
        state[target_key] = value
        replaced += 1
    payload = dict(base_payload) if isinstance(base_payload, dict) else {}
    payload["state_dict"] = state
    payload["selective_router_package"] = {
        "schema": "ct_seqtrack.selective_checkpoint.v2",
        "candidate_checkpoint": str(
            Path(args.candidate_checkpoint).resolve()),
        "candidate_checkpoint_sha256": sha256_file(
            args.candidate_checkpoint),
        "router_sidecar": str(Path(args.router).resolve()),
        "router_sidecar_sha256": sha256_file(args.router),
        "router_tensor_count": replaced,
        "calibration": calibration,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "output": str(output),
        "router_tensor_count": replaced,
        "calibration": calibration,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
