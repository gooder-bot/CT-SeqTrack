#!/usr/bin/env python3
"""Train and calibrate the CRPA router on frozen recursive rollout data."""

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

from models.ct_v2.crpa import (  # noqa: E402
    CRPA_ROUTER_SCHEMA,
    crpa_router_loss,
    stable_tracklet_partition,
)
from tools.b3_crpa_common import (  # noqa: E402
    build_router_from_config,
    canonical_sha256,
    ConfigMap,
    load_rollout_artifact,
)
from utils.config import load_yaml_config  # noqa: E402


REQUIRED_FIELDS = (
    "router_features", "candidate_valid", "oracle_gain",
    "oracle_step_ratio", "candidate_residual_xy", "observation_xy",
    "target_xy", "step_cap", "tracklet_key",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline closed-loop CRPA router training")
    parser.add_argument("--round0", required=True)
    parser.add_argument("--round1")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/10_b3_crpa_v1.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--max-coverage", type=float, default=0.30)
    parser.add_argument("--max-harm-rate", type=float, default=0.10)
    parser.add_argument("--min-precision", type=float, default=0.70)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def validate_arrays(arrays, expected_feature_dim=None):
    missing = [key for key in REQUIRED_FIELDS if key not in arrays]
    if missing:
        raise KeyError("rollout artifact is missing: " + ", ".join(missing))
    feature_dim = int(arrays["router_features"].shape[1])
    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise ValueError(
            f"router feature width {feature_dim} != {expected_feature_dim}")


def concatenate_sources(round0, round1=None):
    """Aggregate Round-0 and double-weighted Round-1 training rows."""
    sources = [(round0, 0)]
    if round1 is not None:
        sources.extend(((round1, 1), (round1, 1)))
    keys = set.intersection(*(set(source) for source, _ in sources))
    merged = {
        key: np.concatenate([source[key] for source, _ in sources], axis=0)
        for key in keys
    }
    merged["rollout_round"] = np.concatenate([
        np.full(source["router_features"].shape[0], round_id, dtype=np.int64)
        for source, round_id in sources
    ])
    return merged


def partition_masks(tracklet_keys, seed):
    assignments = stable_tracklet_partition(
        np.unique(tracklet_keys).tolist(), seed=seed)
    labels = np.asarray([assignments[str(key)] for key in tracklet_keys])
    masks = {name: labels == name for name in ("train", "dev", "calibration")}
    if any(not mask.any() for mask in masks.values()):
        counts = {key: int(value.sum()) for key, value in masks.items()}
        raise RuntimeError(
            f"tracklet hash split produced an empty partition: {counts}")
    return assignments, masks


def make_loader(arrays, mask, batch_size, shuffle):
    indices = np.flatnonzero(mask)
    dataset = TensorDataset(
        torch.from_numpy(arrays["router_features"][indices]).float(),
        torch.from_numpy(arrays["oracle_gain"][indices]).float(),
        torch.from_numpy(arrays["oracle_step_ratio"][indices]).float(),
        torch.from_numpy(arrays["candidate_valid"][indices]).float(),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, drop_last=False)


def compute_scalar_normalization(router, features):
    scalar = features[:, -router.scalar_dim:].astype(np.float64)
    mean = scalar.mean(axis=0)
    std = scalar.std(axis=0)
    std = np.maximum(std, 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def evaluate_loss(router, loader, device):
    router.eval()
    sums = {}
    count = 0
    with torch.inference_mode():
        for features, gain, step, valid in loader:
            features = features.to(device)
            prediction = router.predict_exported_features(features)
            losses = crpa_router_loss(
                prediction, gain.to(device), step.to(device), valid.to(device))
            batch_count = features.shape[0]
            count += batch_count
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value) * batch_count
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_router(router, arrays, masks, args, device):
    train_loader = make_loader(
        arrays, masks["train"], args.batch_size, shuffle=True)
    dev_loader = make_loader(
        arrays, masks["dev"], args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    router.to(device)
    best_state = None
    best_dev = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(args.epochs):
        router.train()
        running = 0.0
        sample_count = 0
        for features, gain, step, valid in train_loader:
            features = features.to(device)
            gain = gain.to(device)
            step = step.to(device)
            valid = valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = router.predict_exported_features(features)
            losses = crpa_router_loss(prediction, gain, step, valid)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(losses["loss"].detach()) * features.shape[0]
            sample_count += features.shape[0]
        dev = evaluate_loss(router, dev_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(sample_count, 1),
            **{f"dev_{key}": value for key, value in dev.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if dev["loss"] < best_dev - 1e-7:
            best_dev = dev["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(router.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("CRPA router training did not produce a checkpoint")
    router.load_state_dict(best_state, strict=True)
    return {
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev,
        "epochs_run": len(history),
        "history": history,
    }


def predict_numpy(router, features, device, batch_size=4096):
    outputs = {"q10": [], "q50": [], "step_ratio": []}
    router.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            tensor = torch.from_numpy(
                features[start:start + batch_size]).float().to(device)
            prediction = router.predict_exported_features(tensor)
            for key in outputs:
                outputs[key].append(prediction[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def calibrate_router(
        router, arrays, mask, device,
        min_coverage=0.05, max_coverage=0.30,
        max_harm_rate=0.10, min_precision=0.70):
    indices = np.flatnonzero(mask)
    prediction = predict_numpy(
        router, arrays["router_features"][indices], device)
    valid = arrays["candidate_valid"][indices].astype(bool)
    q10 = np.where(valid, prediction["q10"], -np.inf)
    selected_candidate = np.argmax(q10, axis=1)
    selected_score = np.max(q10, axis=1)
    row_index = np.arange(len(indices))
    selected_step = prediction["step_ratio"][row_index, selected_candidate]
    step_cap = arrays["step_cap"][indices].reshape(-1)
    residual = arrays["candidate_residual_xy"][indices][
        row_index, selected_candidate]
    observation = arrays["observation_xy"][indices]
    target = arrays["target_xy"][indices]
    corrected = observation + (
        selected_step * step_cap)[:, None] * residual
    observation_error = np.linalg.norm(observation - target, axis=1)
    corrected_error = np.linalg.norm(corrected - target, axis=1)
    improvement = observation_error - corrected_error

    finite_scores = selected_score[np.isfinite(selected_score)]
    if finite_scores.size:
        unique_scores = np.unique(finite_scores)
        if unique_scores.size > 2000:
            unique_scores = np.quantile(
                unique_scores, np.linspace(0.0, 1.0, 2000))
        thresholds = np.unique(np.maximum(
            np.nextafter(unique_scores, -np.inf), 0.0))
        thresholds = np.unique(np.concatenate((np.asarray([0.0]), thresholds)))
    else:
        thresholds = np.asarray([0.0])

    candidates = []
    for threshold in thresholds:
        intervene = np.isfinite(selected_score) & (selected_score > threshold)
        coverage = float(intervene.mean())
        if intervene.any():
            selected_improvement = improvement[intervene]
            harm_rate = float((selected_improvement < -0.02).mean())
            precision = float((selected_improvement > 0.02).mean())
            source_counts = {
                "motion": int((selected_candidate[intervene] == 0).sum()),
                "search": int((selected_candidate[intervene] == 1).sum()),
            }
        else:
            harm_rate = 0.0
            precision = 0.0
            source_counts = {"motion": 0, "search": 0}
        applied_gain = float(np.where(intervene, improvement, 0.0).mean())
        feasible = (
            float(min_coverage) <= coverage <= float(max_coverage)
            and harm_rate <= float(max_harm_rate)
            and precision >= float(min_precision))
        candidates.append({
            "threshold": float(threshold),
            "coverage": coverage,
            "harm_rate": harm_rate,
            "precision": precision,
            "mean_endpoint_gain": applied_gain,
            "source_counts": source_counts,
            "feasible": feasible,
        })
    feasible = [item for item in candidates if item["feasible"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda item: (
                item["mean_endpoint_gain"], item["precision"],
                item["threshold"]))
        status = "passed"
    else:
        finite_max = float(np.max(finite_scores)) if finite_scores.size else 0.0
        chosen = {
            "threshold": max(0.0, finite_max + 1.0),
            "coverage": 0.0,
            "harm_rate": 0.0,
            "precision": 0.0,
            "mean_endpoint_gain": 0.0,
            "source_counts": {"motion": 0, "search": 0},
            "feasible": False,
        }
        status = "failed"
    return {
        "status": status,
        "chosen": chosen,
        "constraints": {
            "min_coverage": float(min_coverage),
            "max_coverage": float(max_coverage),
            "max_harm_rate": float(max_harm_rate),
            "min_precision": float(min_precision),
            "help_margin_m": 0.02,
            "harm_margin_m": 0.02,
        },
        "calibration_rows": len(indices),
        "thresholds_evaluated": len(candidates),
    }


def cpu_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def main():
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")
    if not 0.0 <= args.min_coverage <= args.max_coverage <= 1.0:
        raise ValueError("coverage constraints are invalid")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    config = ConfigMap(load_yaml_config(args.config))
    router = build_router_from_config(config)
    round0, manifest0, hashes0 = load_rollout_artifact(args.round0)
    validate_arrays(round0, router.export_feature_dim)
    round1 = manifest1 = hashes1 = None
    if args.round1:
        round1, manifest1, hashes1 = load_rollout_artifact(args.round1)
        validate_arrays(round1, router.export_feature_dim)
        if set(round0["tracklet_key"].tolist()) != set(
                round1["tracklet_key"].tolist()):
            raise ValueError("Round-0 and Round-1 tracklet sets differ")

    arrays = concatenate_sources(round0, round1)
    assignments, masks = partition_masks(arrays["tracklet_key"], args.seed)
    scalar_mean, scalar_std = compute_scalar_normalization(
        router, arrays["router_features"][masks["train"]])
    router.set_scalar_normalization(scalar_mean, scalar_std)
    training = train_router(router, arrays, masks, args, device)

    calibration_source = round1 if round1 is not None else round0
    calibration_labels = np.asarray([
        assignments[str(key)] for key in calibration_source["tracklet_key"]])
    calibration = calibrate_router(
        router,
        calibration_source,
        calibration_labels == "calibration",
        device,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        max_harm_rate=args.max_harm_rate,
        min_precision=args.min_precision,
    )
    router.set_gain_threshold(calibration["chosen"]["threshold"])
    partition_tracklet_counts = {
        name: sum(value == name for value in assignments.values())
        for name in ("train", "dev", "calibration")
    }
    partition_row_counts = {
        name: int(mask.sum()) for name, mask in masks.items()
    }
    input_artifacts = {
        "round0": {"manifest": manifest0, "hashes": hashes0},
        "round1": (
            {"manifest": manifest1, "hashes": hashes1}
            if round1 is not None else None),
    }
    payload = {
        "schema": CRPA_ROUTER_SCHEMA,
        "router_state_dict": cpu_state_dict(router),
        "router_config": {
            "observation_dim": router.observation_dim,
            "motion_dim": router.motion_dim,
            "search_dim": router.search_dim,
            "scalar_dim": router.scalar_dim,
            "export_feature_dim": router.export_feature_dim,
            "scalar_feature_names": list(router.SCALAR_FEATURE_NAMES),
            "normal_step_cap": router.normal_step_cap,
            "gap_step_cap": router.gap_step_cap,
            "radius_base": router.radius_base,
            "radius_per_second": router.radius_per_second,
            "radius_max": router.radius_max,
        },
        "training": training,
        "calibration": calibration,
        "partitions": {
            "seed": args.seed,
            "tracklet_counts": partition_tracklet_counts,
            "row_counts": partition_row_counts,
            "assignment_sha256": canonical_sha256(assignments),
        },
        "input_artifacts": input_artifacts,
        "aggregation": {
            "round0_weight": 1,
            "round1_weight": 2 if round1 is not None else 0,
            "calibration_source": "round1" if round1 is not None else "round0",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    json_payload = copy.deepcopy(payload)
    json_payload.pop("router_state_dict")
    json_payload["router_sidecar"] = str(output_path.resolve())
    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps({
        "router": str(output_path),
        "report": str(json_path),
        "status": calibration["status"],
        "chosen": calibration["chosen"],
        "best_epoch": training["best_epoch"],
    }, indent=2, sort_keys=True))
    if calibration["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
