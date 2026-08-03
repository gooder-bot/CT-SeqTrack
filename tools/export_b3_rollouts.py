#!/usr/bin/env python3
"""Export GT-labelled CRPA contexts from the real recursive tracker path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.b3_crpa_common import (  # noqa: E402
    canonical_sha256,
    ConfigMap,
    load_matching_model_state,
    load_router_sidecar,
    sha256_file,
    write_rollout_artifact,
)
from utils.config import load_yaml_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export observation or on-policy CRPA recursive rollouts")
    parser.add_argument("--checkpoint", required=True,
                        help="final B2-v2.1 candidate checkpoint")
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/10_b3_crpa_v1.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--path", help="override dataset root")
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy", choices=("observation", "router"),
        default="observation")
    parser.add_argument(
        "--router", help="passed CRPA router sidecar for on-policy rollout")
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda, or an explicit torch device")
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--preloading", action="store_true")
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main():
    args = parse_args()
    from datasets import get_dataset
    from models import get_model

    if args.policy == "router" and not args.router:
        raise ValueError("--policy router requires --router")
    if args.policy == "observation" and args.router:
        raise ValueError("--router is only valid with --policy router")
    if args.max_tracklets is not None and args.max_tracklets <= 0:
        raise ValueError("--max-tracklets must be positive")

    raw_config = load_yaml_config(args.config)
    raw_config.update({
        "seed": args.seed,
        "test_split": args.split,
        "proposal_inference_mode": "full",
        "export_b3_rollouts": True,
        "b3_enabled_scale": 0.0 if args.policy == "observation" else 1.0,
        "preloading": bool(args.preloading),
    })
    if args.path:
        raw_config["path"] = args.path
    config = ConfigMap(raw_config)
    device = resolve_device(args.device)

    dataset = get_dataset(
        config, type="test", split=args.split, protocol_role="test")
    model = get_model(config.net_model)(config)
    load_report = load_matching_model_state(model, args.checkpoint)
    router_metadata = None
    if args.router:
        router_metadata = load_router_sidecar(
            model.b3_risk_router, args.router, require_passed=True)
    model.to(device)
    model.eval()

    tracklet_count = len(dataset)
    if args.max_tracklets is not None:
        tracklet_count = min(tracklet_count, args.max_tracklets)
    rows = []
    total_frames = 0
    with torch.inference_mode():
        for tracklet_id in range(tracklet_count):
            sequence = dataset[tracklet_id]
            model.evaluate_one_sequence(sequence)
            source_dataset = getattr(dataset, "dataset", dataset)
            if hasattr(source_dataset, "get_tracklet_key"):
                tracklet_key = source_dataset.get_tracklet_key(tracklet_id)
            else:
                tracklet_key = f"{args.split}/tracklet/{tracklet_id}"
            for row in model._b3_sequence_rollouts:
                item = dict(row)
                item["tracklet_id"] = tracklet_id
                item["tracklet_key"] = str(tracklet_key)
                rows.append(item)
            total_frames += len(sequence)
            print(
                f"[{tracklet_id + 1}/{tracklet_count}] "
                f"{tracklet_key}: {len(sequence)} frames")

    manifest = {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": canonical_sha256(raw_config),
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "base_checkpoint_sha256": load_report["base_checkpoint_sha256"],
        "base_matched_tensors": load_report["matched_tensors"],
        "router_sidecar": (
            str(Path(args.router).resolve()) if args.router else None),
        "router_sidecar_sha256": (
            sha256_file(args.router) if args.router else None),
        "router_training_schema": (
            router_metadata.get("schema") if router_metadata else None),
        "policy": args.policy,
        "split": args.split,
        "seed": args.seed,
        "tracklets_evaluated": tracklet_count,
        "frames_evaluated": total_frames,
        "gt_usage": "offline_labels_only",
    }
    npz_path, manifest_path = write_rollout_artifact(
        args.output, rows, manifest)
    summary = {
        "rollout": str(npz_path),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "tracklets": tracklet_count,
        "policy": args.policy,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
