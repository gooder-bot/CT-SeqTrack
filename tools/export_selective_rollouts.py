#!/usr/bin/env python3
"""Export true three-frame counterfactual rollouts for B2-v2.2.

Each intervention branch is re-cropped and re-forwarded.  GT is read only
after predictions have been produced, solely to calculate offline costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v2.selective_innovation import (  # noqa: E402
    discounted_tracking_cost,
    stable_tracklet_partition,
)
from tools.selective_innovation_common import (  # noqa: E402
    ConfigMap,
    canonical_sha256,
    load_matching_model_state,
    write_rollout_artifact,
)
from utils.config import load_yaml_config  # noqa: E402


STEP_RATIOS = (0.25, 0.5, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export signed H=3 B2-v2.2 intervention gains")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/12_b2_v22_selective.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--path", help="override the nuScenes dataset root")
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def forward_step(model, sequence, frame_id, history, candidate=-1,
                 step_ratio=0.25):
    data, reference_box = model.build_input_dict(
        sequence, frame_id, history)
    if torch.sum(data["points"][:, :, :3]) == 0:
        return reference_box, None
    data["selective_forced_candidate"] = torch.tensor(
        [int(candidate)], device=model.device, dtype=torch.long)
    data["selective_forced_step_ratio"] = torch.tensor(
        [float(step_ratio)], device=model.device, dtype=torch.float32)
    candidate_box, _, output = model.evaluate_one_sample(
        data, ref_box=reference_box)
    return candidate_box, output


def rollout_branch(model, sequence, start_frame, history, horizon,
                   first_candidate=-1, first_step=0.25):
    branch_history = list(history)
    predicted_boxes = []
    first_output = None
    for offset in range(horizon):
        frame_id = start_frame + offset
        candidate = first_candidate if offset == 0 else -1
        step = first_step if offset == 0 else 0.25
        predicted_box, output = forward_step(
            model, sequence, frame_id, branch_history,
            candidate=candidate, step_ratio=step)
        branch_history.append(predicted_box)
        predicted_boxes.append(predicted_box)
        if offset == 0:
            first_output = output
        if offset == 0 and output is None:
            return predicted_boxes, None
    return predicted_boxes, first_output


def rollout_cost(model, sequence, start_frame, boxes, gamma):
    from utils.metrics import estimateAccuracy, estimateOverlap

    ious = []
    distances = []
    for offset, predicted_box in enumerate(boxes):
        target_box = sequence[start_frame + offset]["3d_bbox"]
        ious.append(estimateOverlap(
            target_box, predicted_box,
            dim=model.config.IoU_space, up_axis=model.config.up_axis))
        distances.append(estimateAccuracy(
            target_box, predicted_box,
            dim=model.config.IoU_space, up_axis=model.config.up_axis))
    return discounted_tracking_cost(ious, distances, gamma=gamma)


def main():
    args = parse_args()
    if args.split != "mini_train":
        raise ValueError(
            "router rollouts are restricted to mini_train; mini_val is held out")
    if args.horizon != 3:
        raise ValueError("the frozen v2.2 protocol requires horizon=3")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0,1]")
    if args.max_tracklets is not None and args.max_tracklets <= 0:
        raise ValueError("max-tracklets must be positive")

    from datasets import get_dataset
    from models import get_model

    raw_config = load_yaml_config(args.config)
    raw_config.update({
        "seed": args.seed,
        "test_split": args.split,
        "proposal_inference_mode": "full_selective",
        "use_signed_horizon_router": True,
        "signed_router_enabled_scale": 1.0,
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
    model.to(device)
    model.eval()

    source_dataset = getattr(dataset, "dataset", dataset)
    tracklet_count = len(dataset)
    if args.max_tracklets is not None:
        tracklet_count = min(tracklet_count, args.max_tracklets)
    rows = []
    partition_tracklets = {"train": 0, "dev": 0, "calibration": 0}
    with torch.inference_mode():
        for tracklet_id in range(tracklet_count):
            sequence = dataset[tracklet_id]
            tracklet_key = (
                source_dataset.get_tracklet_key(tracklet_id)
                if hasattr(source_dataset, "get_tracklet_key")
                else f"{args.split}/tracklet/{tracklet_id}")
            partition = stable_tracklet_partition(tracklet_key, args.seed)
            partition_tracklets[partition] += 1
            history = [sequence[0]["3d_bbox"]]
            emitted = 0
            for frame_id in range(1, len(sequence)):
                if frame_id + args.horizon > len(sequence):
                    # Still advance the recursive observation state so every
                    # earlier labelled state had its genuine history.
                    observation_box, _ = forward_step(
                        model, sequence, frame_id, history, candidate=-1)
                    history.append(observation_box)
                    continue

                observation_boxes, context_output = rollout_branch(
                    model, sequence, frame_id, history, args.horizon,
                    first_candidate=-1)
                if context_output is None:
                    history.append(observation_boxes[0])
                    continue
                observation_cost = rollout_cost(
                    model, sequence, frame_id, observation_boxes, args.gamma)
                candidate_valid = context_output[
                    "signed_candidate_valid"].detach().cpu().numpy(
                    ).reshape(2).astype(np.float32)
                signed_gain = np.zeros((2, 3), dtype=np.float32)
                candidate_cost = np.full(
                    (2, 3), observation_cost, dtype=np.float32)
                for candidate in range(2):
                    if candidate_valid[candidate] <= 0:
                        continue
                    for step_id, step_ratio in enumerate(STEP_RATIOS):
                        branch_boxes, _ = rollout_branch(
                            model, sequence, frame_id, history, args.horizon,
                            first_candidate=candidate,
                            first_step=step_ratio)
                        cost = rollout_cost(
                            model, sequence, frame_id, branch_boxes, args.gamma)
                        candidate_cost[candidate, step_id] = cost
                        # Negative values are intentionally preserved.
                        signed_gain[candidate, step_id] = observation_cost - cost

                row = {
                    "tracklet_id": np.int64(tracklet_id),
                    "tracklet_key": str(tracklet_key),
                    "partition": partition,
                    "frame_id": np.int64(frame_id),
                    "router_features": context_output[
                        "signed_router_features"].detach().cpu().numpy(
                        ).reshape(-1).astype(np.float32),
                    "candidate_valid": candidate_valid,
                    "candidate_residual_xy": context_output[
                        "signed_candidate_residual_xy"].detach().cpu().numpy(
                        ).reshape(2, 2).astype(np.float32),
                    "signed_gain": signed_gain,
                    "candidate_cost": candidate_cost,
                    "observation_cost": np.float32(observation_cost),
                    "rollout_length": np.int64(args.horizon),
                }
                rows.append(row)
                emitted += 1
                history.append(observation_boxes[0])
            print(
                f"[{tracklet_id + 1}/{tracklet_count}] {tracklet_key}: "
                f"{emitted} states ({partition})")

    manifest = {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": canonical_sha256(raw_config),
        "candidate_checkpoint": str(Path(args.checkpoint).resolve()),
        "candidate_checkpoint_sha256": load_report["checkpoint_sha256"],
        "matched_tensors": load_report["matched_tensors"],
        "split": args.split,
        "seed": args.seed,
        "horizon": args.horizon,
        "gamma": args.gamma,
        "tracklets_evaluated": tracklet_count,
        "partition_tracklets": partition_tracklets,
        "policy_after_intervention": "observation",
        "gt_usage": "cost_calculation_after_closed_loop_prediction_only",
    }
    npz_path, manifest_path = write_rollout_artifact(
        args.output, rows, manifest)
    print(json.dumps({
        "rollout": str(npz_path),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "tracklets": tracklet_count,
        "partitions": partition_tracklets,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
