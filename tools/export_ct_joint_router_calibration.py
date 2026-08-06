#!/usr/bin/env python3
"""Export held-out H=3 records for Joint Full router calibration.

This command intentionally reuses the online recursive training contract.  It
runs candidate 0 on-policy, creates observation/Search shadows from the same
pre-action state, and never enables gradients.  Only complete tracklets in the
stable calibration partition are visited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b3_crpa_common import (  # noqa: E402
    canonical_sha256,
    checkpoint_state_dict,
    ConfigMap,
    sha256_file,
    torch_load,
)
from utils.config import load_yaml_config  # noqa: E402


CALIBRATION_SCHEMA = "ct_seqtrack.joint_router_calibration.v1"


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_exact_joint_checkpoint(model, checkpoint_path):
    payload = torch_load(checkpoint_path, map_location="cpu")
    source = checkpoint_state_dict(payload)
    target = model.state_dict()
    candidates = [source]
    for prefix in ("model.", "module."):
        candidates.append({
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in source.items()
        })
    normalized = max(
        candidates,
        key=lambda candidate: sum(
            key in target and target[key].shape == value.shape
            for key, value in candidate.items()),
    )
    missing = sorted(set(target).difference(normalized))
    unexpected = sorted(set(normalized).difference(target))
    mismatched = sorted(
        key for key in set(target).intersection(normalized)
        if target[key].shape != normalized[key].shape)
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "calibration requires the exact selected Joint Full checkpoint; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={mismatched[:5]}")
    model.load_state_dict(normalized, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export no-grad H=3 Joint Full calibration records")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "cfgs/ct_v2/21_ct_joint_full.yaml")
    parser.add_argument("--path", help="override dataset root")
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preloading", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"calibration records already exist: {args.output}")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(
            f"calibration manifest already exists: {manifest_path}")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")

    from datasets import get_dataset
    from datasets.sampler import online_recursive_collate
    from models import get_model
    from utils.recursive_state import (
        OnlineRecursiveBatchSampler,
        stable_tracklet_partition,
    )

    raw_config = load_yaml_config(args.config)
    raw_config.update({
        "seed": int(args.seed),
        "ct_online_recursive_training": True,
        "ct_router_partition": "calibration",
        "preloading": bool(args.preloading),
    })
    if args.path:
        raw_config["path"] = args.path
    config = ConfigMap(raw_config)
    if not bool(config.get("use_ct_joint_full", False)):
        raise ValueError("calibration exporter requires use_ct_joint_full=true")
    if int(config.get("ct_router_horizon", 3)) != 3:
        raise ValueError("calibration exporter requires ct_router_horizon=3")

    dataset = get_dataset(
        config, type=config.train_type, split=args.split,
        protocol_role="train")
    # One causal slot makes every complete candidate-0 window eligible for a
    # shadow event.  The four recovery views remain in the batch but are never
    # written back or exported.
    batch_sampler = OnlineRecursiveBatchSampler(
        dataset, slots=1,
        candidate_views=int(config.ct_recursive_candidate_views),
        seed=args.seed, partition="calibration", shadow_interval=1,
        shadow_fraction=1.0)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 41001)
    loader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=args.workers,
        collate_fn=online_recursive_collate, generator=generator,
        pin_memory=False)

    device = resolve_device(args.device)
    model = get_model(config.net_model)(config)
    load_exact_joint_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()
    threshold = float(model.ct_joint_router.decision_threshold.detach().cpu())
    if threshold != 0.5:
        raise ValueError(
            "records must be generated from the selected joint checkpoint "
            f"at the training policy threshold 0.5, got {threshold}")
    model._ct_recursive_states = {}

    probabilities = []
    gains = []
    evidence_valid = []
    tracklet_keys = []
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = model._prepare_online_recursive_batch(raw_batch)
            output = model(batch)
            model._attach_h3_shadow_labels(batch, output)
            for row, context in enumerate(model._ct_online_batch_context):
                raw = context["raw"]
                if int(raw["candidate_id"]) != 0:
                    continue
                key = str(raw["tracklet_key"])
                if stable_tracklet_partition(key, args.seed) != "calibration":
                    raise RuntimeError(
                        f"non-calibration tracklet leaked into export: {key}")
                h3_valid = bool(float(batch["ct_h3_valid"][row]))
                deploy_valid = bool(float(
                    output["ct_router_evidence_valid"][row]))
                probabilities.append(float(output[
                    "ct_router_gate"][row].detach().cpu()))
                gains.append(float(batch["ct_h3_gain"][row].detach().cpu()))
                evidence_valid.append(bool(h3_valid and deploy_valid))
                tracklet_keys.append(key)
            model._commit_online_recursive_predictions(output)
            if (batch_index + 1) % 100 == 0:
                print(f"exported {len(probabilities)} canonical frames")

    if not probabilities:
        raise RuntimeError("calibration partition produced no canonical frames")
    arrays = {
        "router_probability": np.asarray(probabilities, dtype=np.float32),
        "h3_gain": np.asarray(gains, dtype=np.float32),
        "evidence_valid": np.asarray(evidence_valid, dtype=np.bool_),
        "tracklet_key": np.asarray(tracklet_keys, dtype=np.str_),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "schema": CALIBRATION_SCHEMA,
        "config_path": str(args.config.resolve()),
        "config_sha256": canonical_sha256(raw_config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split": str(args.split),
        "partition": "calibration",
        "seed": int(args.seed),
        "row_count": len(probabilities),
        "valid_row_count": int(np.sum(evidence_valid)),
        "tracklet_count": len(set(tracklet_keys)),
        "gradient_enabled": False,
        "canonical_policy_threshold": threshold,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
