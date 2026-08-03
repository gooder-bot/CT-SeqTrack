#!/usr/bin/env python3
"""Package frozen B2-v2.1 sources and a passed CRPA router into one ckpt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import get_model  # noqa: E402
from tools.b3_crpa_common import (  # noqa: E402
    canonical_sha256,
    ConfigMap,
    load_matching_model_state,
    load_router_sidecar,
    sha256_file,
    torch_load,
)
from utils.config import load_yaml_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a strict, evaluation-ready B3 CRPA checkpoint")
    parser.add_argument("--base", required=True,
                        help="final B2-v2.1 Lightning checkpoint")
    parser.add_argument("--router", required=True,
                        help="passed router sidecar")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/10_b3_crpa_v1.yaml"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --force explicitly")
    config_dict = load_yaml_config(args.config)
    config = ConfigMap(config_dict)
    model = get_model(config.net_model)(config)
    load_report = load_matching_model_state(model, args.base)
    router_payload = load_router_sidecar(
        model.b3_risk_router, args.router, require_passed=True)

    expected_base_hashes = set()
    for round_name in ("round0", "round1"):
        item = router_payload.get("input_artifacts", {}).get(round_name)
        if item:
            value = item.get("manifest", {}).get("base_checkpoint_sha256")
            if value:
                expected_base_hashes.add(str(value))
    actual_base_hash = load_report["base_checkpoint_sha256"]
    if expected_base_hashes and expected_base_hashes != {actual_base_hash}:
        raise RuntimeError(
            "router rollouts were generated from a different B2 checkpoint")

    source_payload = torch_load(args.base, map_location="cpu")
    if not isinstance(source_payload, dict) or "state_dict" not in source_payload:
        source_payload = {"state_dict": model.state_dict()}
    else:
        source_payload = dict(source_payload)
        source_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    source_payload["optimizer_states"] = []
    source_payload["lr_schedulers"] = []
    source_payload["b3_crpa"] = {
        "schema": router_payload["schema"],
        "base_checkpoint_sha256": actual_base_hash,
        "router_sidecar_sha256": sha256_file(args.router),
        "config_sha256": canonical_sha256(config_dict),
        "calibration": router_payload["calibration"],
        "partitions": router_payload["partitions"],
        "deployment_only": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source_payload, output_path)

    verification_model = get_model(config.net_model)(config)
    packaged = torch_load(output_path, map_location="cpu")
    verification_model.load_state_dict(packaged["state_dict"], strict=True)
    for key, expected in model.state_dict().items():
        actual = verification_model.state_dict()[key]
        if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
            raise RuntimeError(f"packaged checkpoint changed tensor {key}")
    report = {
        "checkpoint": str(output_path.resolve()),
        "checkpoint_sha256": sha256_file(output_path),
        "base_checkpoint": str(Path(args.base).resolve()),
        "base_checkpoint_sha256": actual_base_hash,
        "router": str(Path(args.router).resolve()),
        "router_sha256": sha256_file(args.router),
        "state_tensors": len(packaged["state_dict"]),
        "strict_load_verified": True,
        "calibration_status": router_payload["calibration"]["status"],
    }
    report_path = output_path.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
