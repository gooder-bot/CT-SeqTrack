#!/usr/bin/env python3
"""Train B2-v3's six-action router on signed closed-loop gains."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v2 import (  # noqa: E402
    SELECTIVE_V3_ROUTER_SCHEMA,
    action_consistent_router_loss,
    calibrate_action_threshold,
)
from tools.selective_v3_common import (  # noqa: E402
    ConfigMap,
    build_v3_router_from_config,
    load_v3_rollout_artifact,
)
from utils.config import load_yaml_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline B2-v3 action-router training")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/14_b2_v3_selective.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--threshold-partition", choices=("dev", "calibration"),
        default="calibration",
        help="use dev only for provisional round-1 policy; calibration once final")
    parser.add_argument("--min-selected-count", type=int, default=100)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def subset(arrays, partition):
    mask = arrays["partition"].astype(str) == str(partition)
    if not np.any(mask):
        raise RuntimeError(f"rollouts contain no {partition} rows")
    return {
        "features": arrays["router_features"][mask].astype(np.float32),
        "gain": arrays["signed_gain"][mask].astype(np.float32),
        "valid": arrays["candidate_valid"][mask].astype(np.float32),
    }


def make_loader(data, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(data["features"]),
        torch.from_numpy(data["gain"]),
        torch.from_numpy(data["valid"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle),
        generator=generator)


def validate_training_stage(manifest, threshold_partition, seed):
    if manifest.get("split") != "mini_train":
        raise ValueError("router training accepts mini_train rollouts only")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError("router seed must match the rollout partition seed")
    if threshold_partition == "dev":
        if (manifest.get("round") != 0
                or manifest.get("state_policy") != "observation"):
            raise ValueError(
                "provisional dev router requires round-0 observation rollouts")
        return
    if threshold_partition != "calibration":
        raise ValueError("unsupported router threshold partition")
    if (manifest.get("round") != "merged_0_1"
            or manifest.get("state_policy") != "observation_plus_router"):
        raise ValueError(
            "final calibration router requires merged round-0/round-1 rollouts")
    source_rounds = manifest.get("source_rounds", [])
    if len(source_rounds) != 2:
        raise ValueError("merged rollout provenance is incomplete")
    round1_manifest = source_rounds[1].get("manifest", {})
    round1_calibration = round1_manifest.get(
        "state_policy_calibration", {})
    if (round1_manifest.get("round") != 1
            or round1_manifest.get("state_policy") != "router"
            or round1_calibration.get("status") != "passed"
            or round1_calibration.get("partition") != "dev"):
        raise ValueError("merged rollout has an invalid round-1 policy")


def evaluate(router, loader, device):
    router.eval()
    totals = {"loss": 0.0, "loss_q10": 0.0, "loss_q50": 0.0}
    count = 0
    with torch.inference_mode():
        for features, gain, valid in loader:
            features = features.to(device)
            gain = gain.to(device)
            valid = valid.to(device)
            losses = action_consistent_router_loss(
                router.predict_export_features(features), gain, valid)
            batch = int(features.shape[0])
            for key in totals:
                totals[key] += float(losses[key].item()) * batch
            count += batch
    return {key: value / max(1, count) for key, value in totals.items()}


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size,
           args.min_selected_count) <= 0:
        raise ValueError("training/count settings must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer settings are invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    arrays, manifest, hashes = load_v3_rollout_artifact(args.rollouts)
    validate_training_stage(
        manifest, args.threshold_partition, args.seed)
    train = subset(arrays, "train")
    dev = subset(arrays, "dev")
    threshold_data = (
        dev if args.threshold_partition == "dev"
        else subset(arrays, "calibration"))
    config = ConfigMap(load_yaml_config(args.config))
    router = build_v3_router_from_config(config)
    if train["features"].shape[1] != router.export_feature_dim:
        raise ValueError(
            f"rollout feature width {train['features'].shape[1]} does not "
            f"match router {router.export_feature_dim}")

    scalar_start = (
        router.observation_dim + router.motion_dim + router.search_dim)
    scalar_train = train["features"][:, scalar_start:]
    router.set_scalar_normalization(
        scalar_train.mean(axis=0).astype(np.float32),
        np.maximum(
            scalar_train.std(axis=0).astype(np.float32), 1e-4))
    device = resolve_device(args.device)
    router.to(device)
    train_loader = make_loader(
        train, args.batch_size, shuffle=True, seed=args.seed)
    dev_loader = make_loader(
        dev, args.batch_size, shuffle=False, seed=args.seed)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_dev = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(args.epochs):
        router.train()
        train_total = 0.0
        train_count = 0
        for features, gain, valid in train_loader:
            features = features.to(device)
            gain = gain.to(device)
            valid = valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = action_consistent_router_loss(
                router.predict_export_features(features), gain, valid)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0)
            optimizer.step()
            batch = int(features.shape[0])
            train_total += float(losses["loss"].item()) * batch
            train_count += batch
        dev_metrics = evaluate(router, dev_loader, device)
        train_loss = train_total / max(1, train_count)
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
        })
        if dev_metrics["loss"] < best_dev - 1e-7:
            best_dev = dev_metrics["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(router.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch + 1:03d} train={train_loss:.6f} "
            f"dev={dev_metrics['loss']:.6f} stale={stale}")
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("router training did not produce a checkpoint")
    router.load_state_dict(best_state, strict=True)
    router.eval()

    with torch.inference_mode():
        prediction = router.predict_export_features(torch.from_numpy(
            threshold_data["features"]).to(device))
    calibration_error = None
    try:
        calibration_result = calibrate_action_threshold(
            prediction["q10"].cpu().numpy(),
            threshold_data["gain"], threshold_data["valid"],
            min_precision=0.75, max_harm_rate=0.10,
            min_coverage=0.05, max_coverage=0.25,
            helpful_margin=0.02,
            min_selected_count=args.min_selected_count)
        calibration_result.update({
            "status": "passed",
            "partition": args.threshold_partition,
        })
        router.set_gain_threshold(calibration_result["threshold"])
    except RuntimeError as error:
        calibration_error = str(error)
        calibration_result = {
            "status": "failed", "error": calibration_error,
            "threshold": None, "partition": args.threshold_partition,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SELECTIVE_V3_ROUTER_SCHEMA,
        "router_state_dict": {
            key: value.detach().cpu()
            for key, value in router.state_dict().items()},
        "scalar_feature_names": list(router.SCALAR_FEATURE_NAMES),
        "step_ratios": list(router.STEP_RATIOS),
        "training": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_dev_loss": best_dev,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "row_counts": {
                "train": int(train["features"].shape[0]),
                "dev": int(dev["features"].shape[0]),
                args.threshold_partition:
                    int(threshold_data["features"].shape[0]),
            },
            "history": history,
        },
        "calibration": calibration_result,
        "rollout_manifest": manifest,
        "rollout_hashes": hashes,
    }
    torch.save(payload, output)
    summary = {
        "schema": SELECTIVE_V3_ROUTER_SCHEMA,
        "output": str(output),
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev,
        "calibration": calibration_result,
        "row_counts": payload["training"]["row_counts"],
    }
    summary_path = output.with_suffix(output.suffix + ".json")
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if calibration_error is not None:
        raise RuntimeError(calibration_error)


if __name__ == "__main__":
    main()
