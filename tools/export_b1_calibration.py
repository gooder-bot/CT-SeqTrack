#!/usr/bin/env python3
"""Export held-out train-tracklet B1 residuals for scale calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset, points_utils  # noqa: E402
from utils.checkpoint_loading import load_initial_weights  # noqa: E402
from models import get_model  # noqa: E402
from models.ct_v2.crpa import stable_tracklet_partition  # noqa: E402
from tools.selective_innovation_common import canonical_sha256  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402
from utils.replay_cache import (  # noqa: E402
    b1_calibration_config_sha256,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--partition", choices=("train", "dev", "calibration"),
                        default="calibration")
    parser.add_argument("--path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return device


def main():
    args = parse_args()
    raw_config = load_yaml_config(args.config)
    raw_config.update({
        "seed": args.seed,
        "test_split": args.split,
        "preloading": bool(args.preloading),
        "use_b1motion_v3": True,
        "use_calibrated_motion_uncertainty": True,
        "use_motion_v3_legacy_fusion": False,
        "proposal_inference_mode": "observation",
    })
    if args.path:
        raw_config["path"] = args.path
    config = EasyDict(raw_config)
    dataset = get_dataset(
        config, type="test", split=args.split, protocol_role="test")
    source_dataset = getattr(dataset, "dataset", dataset)
    model = get_model(config.net_model)(config)
    load_initial_weights(model, args.checkpoint)
    model.to(resolve_device(args.device)).eval()
    row_limit = len(dataset)
    if args.max_tracklets is not None:
        row_limit = min(row_limit, int(args.max_tracklets))
    rows = []
    selected_tracklets = 0
    with torch.inference_mode():
        for tracklet_id in range(row_limit):
            tracklet_key = (
                source_dataset.get_tracklet_key(tracklet_id)
                if hasattr(source_dataset, "get_tracklet_key")
                else f"{args.split}/tracklet/{tracklet_id}")
            if stable_tracklet_partition(
                    tracklet_key, args.seed) != args.partition:
                continue
            selected_tracklets += 1
            sequence = dataset[tracklet_id]
            history = [sequence[0]["3d_bbox"]]
            for frame_id in range(1, len(sequence)):
                prediction = model.predict_motion_prepass(
                    sequence, frame_id, history)
                target_local = points_utils.transform_box(
                    sequence[frame_id]["3d_bbox"], history[-1])
                target_xy = np.asarray(
                    target_local.center[:2], dtype=np.float32)
                rows.append({
                    "error_xy": target_xy - np.asarray(
                        prediction["mu_xy"], dtype=np.float32),
                    "velocity_xy": np.asarray(
                        prediction["velocity_xy"], dtype=np.float32),
                    "basis_velocity_xy": np.asarray(
                        prediction["basis_velocity_xy"], dtype=np.float32),
                    "direction_xy": np.asarray(
                        prediction["direction_xy"], dtype=np.float32),
                    "log_sigma_pp": np.asarray(
                        prediction["log_sigma_parallel_perp"],
                        dtype=np.float32),
                    "valid": np.float32(prediction["valid"]),
                    "gap_ratio": np.float32(prediction["gap_ratio"]),
                    "tracklet_key": str(tracklet_key),
                    "frame_id": np.int64(frame_id),
                })
                data, reference_box = model.build_input_dict(
                    sequence, frame_id, history)
                if torch.sum(data["points"][:, :, :3]) == 0:
                    history.append(reference_box)
                else:
                    candidate_box, _, _ = model.evaluate_one_sample(
                        data, ref_box=reference_box)
                    history.append(candidate_box)
    if not rows:
        raise RuntimeError("B1 calibration export produced no rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        error_xy=np.stack([row["error_xy"] for row in rows]),
        velocity_xy=np.stack([row["velocity_xy"] for row in rows]),
        basis_velocity_xy=np.stack([
            row["basis_velocity_xy"] for row in rows]),
        direction_xy=np.stack([row["direction_xy"] for row in rows]),
        log_sigma_pp=np.stack([row["log_sigma_pp"] for row in rows]),
        valid=np.asarray([row["valid"] for row in rows]),
        gap_ratio=np.asarray([row["gap_ratio"] for row in rows]),
        tracklet_key=np.asarray([row["tracklet_key"] for row in rows]),
        frame_id=np.asarray([row["frame_id"] for row in rows]),
    )
    manifest = {
        "schema": "ct_seqtrack.b1_calibration.v2",
        "dataset": str(getattr(config, 'dataset', 'unknown')),
        "split": args.split,
        "partition": args.partition,
        "seed": args.seed,
        "config_sha256": canonical_sha256(raw_config),
        "b1_config_sha256": b1_calibration_config_sha256(config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "tracklets": selected_tracklets,
        "rows": len(rows),
        "artifact_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
