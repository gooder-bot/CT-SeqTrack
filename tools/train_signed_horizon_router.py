#!/usr/bin/env python3
"""Train and calibrate the frozen-candidate B2-v2.2 router offline."""

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
    SELECTIVE_ROUTER_SCHEMA,
    calibrate_gain_threshold,
    signed_horizon_router_loss,
)
from tools.selective_innovation_common import (  # noqa: E402
    ConfigMap,
    build_router_from_config,
    load_rollout_artifact,
)
from utils.config import load_yaml_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline signed H=3 selective-router training")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/12_b2_v22_selective.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
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


def evaluate(router, loader, device):
    router.eval()
    totals = {"loss": 0.0, "loss_q10": 0.0,
              "loss_q50": 0.0, "loss_step": 0.0}
    count = 0
    with torch.inference_mode():
        for features, gain, valid in loader:
            features = features.to(device)
            gain = gain.to(device)
            valid = valid.to(device)
            losses = signed_horizon_router_loss(
                router.predict_export_features(features), gain, valid)
            batch = int(features.shape[0])
            for key in totals:
                totals[key] += float(losses[key].item()) * batch
            count += batch
    return {key: value / max(1, count) for key, value in totals.items()}


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) <= 0:
        raise ValueError("epochs, patience, and batch-size must be positive")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer settings are invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    arrays, manifest, hashes = load_rollout_artifact(args.rollouts)
    if manifest.get("split") != "mini_train":
        raise ValueError("router training accepts mini_train rollouts only")
    train = subset(arrays, "train")
    dev = subset(arrays, "dev")
    calibration = subset(arrays, "calibration")
    config = ConfigMap(load_yaml_config(args.config))
    router = build_router_from_config(config)
    expected_dim = router.export_feature_dim
    if train["features"].shape[1] != expected_dim:
        raise ValueError(
            f"rollout feature width {train['features'].shape[1]} "
            f"does not match router {expected_dim}")

    scalar_start = (
        router.observation_dim + router.motion_dim + router.search_dim)
    scalar_train = train["features"][:, scalar_start:]
    scalar_mean = scalar_train.mean(axis=0).astype(np.float32)
    scalar_std = scalar_train.std(axis=0).astype(np.float32)
    scalar_std = np.maximum(scalar_std, 1e-4)
    router.set_scalar_normalization(scalar_mean, scalar_std)
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
            losses = signed_horizon_router_loss(
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
            "epoch": epoch + 1,
            "train_loss": train_loss,
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

    calibration_features = torch.from_numpy(
        calibration["features"]).to(device)
    with torch.inference_mode():
        prediction = router.predict_export_features(calibration_features)
    q10 = prediction["q10"].cpu().numpy()
    step_class = prediction["step_logits"].argmax(dim=2).cpu().numpy()
    calibration_error = None
    try:
        calibration_result = calibrate_gain_threshold(
            q10,
            calibration["gain"],
            calibration["valid"],
            step_class=step_class,
            min_precision=0.75,
            max_harm_rate=0.10,
            min_coverage=0.05,
            max_coverage=0.25,
            helpful_margin=0.02,
        )
        calibration_result["status"] = "passed"
        router.set_gain_threshold(calibration_result["threshold"])
    except RuntimeError as error:
        calibration_error = str(error)
        calibration_result = {
            "status": "failed",
            "error": calibration_error,
            "threshold": None,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        key: value.detach().cpu() for key, value in router.state_dict().items()
    }
    payload = {
        "schema": SELECTIVE_ROUTER_SCHEMA,
        "router_state_dict": cpu_state,
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
                "calibration": int(calibration["features"].shape[0]),
            },
            "history": history,
        },
        "calibration": calibration_result,
        "rollout_manifest": manifest,
        "rollout_hashes": hashes,
    }
    torch.save(payload, output)
    summary_path = output.with_suffix(output.suffix + ".json")
    summary_payload = {
        "schema": SELECTIVE_ROUTER_SCHEMA,
        "output": str(output),
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev,
        "calibration": calibration_result,
        "row_counts": payload["training"]["row_counts"],
    }
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary_payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    if calibration_error is not None:
        raise RuntimeError(calibration_error)


if __name__ == "__main__":
    main()
