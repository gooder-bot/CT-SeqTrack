#!/usr/bin/env python3
"""Export six-action H=3 counterfactual labels for B2-v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v2 import ActionConsistentInnovationRouter  # noqa: E402
from models.ct_v2.selective_innovation import (  # noqa: E402
    discounted_tracking_cost,
    stable_tracklet_partition,
)
from tools.selective_innovation_common import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.selective_v3_common import (  # noqa: E402
    ConfigMap,
    load_matching_v3_model_state,
    load_v3_router_sidecar,
    write_v3_rollout_artifact,
)
from utils.config import load_yaml_config  # noqa: E402


STEP_RATIOS = (0.25, 0.5, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export B2-v3 action-consistent H=3 gains")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/14_b2_v3_selective.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--path")
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--state-policy", choices=("observation", "router"),
        default="observation")
    parser.add_argument(
        "--router-sidecar",
        help="required for the round-1 on-policy state distribution")
    parser.add_argument("--round", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def forward_step(model, sequence, frame_id, history, policy, step_ratio=None):
    data, reference_box = model.build_input_dict(
        sequence, frame_id, history)
    if torch.sum(data["points"][:, :, :3]) == 0:
        return reference_box, None
    data["selective_v3_policy_override"] = torch.tensor(
        [int(policy)], device=model.device, dtype=torch.long)
    if int(policy) >= 0:
        if step_ratio is None:
            raise ValueError("forced candidate requires an explicit step")
        data["selective_v3_forced_step_ratio"] = torch.tensor(
            [float(step_ratio)], device=model.device, dtype=torch.float32)
    candidate_box, _, output = model.evaluate_one_sample(
        data, ref_box=reference_box)
    return candidate_box, output


def rollout_branch(model, sequence, start_frame, history, horizon,
                   first_policy, first_step=None):
    """Execute one action, then force observation for every future frame."""
    branch_history = list(history)
    predicted_boxes = []
    first_output = None
    for offset in range(horizon):
        frame_id = start_frame + offset
        if offset == 0:
            policy = first_policy
            step = first_step
        else:
            policy = ActionConsistentInnovationRouter.POLICY_OBSERVATION
            step = None
        predicted_box, output = forward_step(
            model, sequence, frame_id, branch_history, policy, step)
        branch_history.append(predicted_box)
        predicted_boxes.append(predicted_box)
        if offset == 0:
            first_output = output
            if output is None:
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
        raise ValueError("B2-v3 router rollouts are restricted to mini_train")
    if args.horizon != 3:
        raise ValueError("B2-v3 uses a fixed H=3 rollout")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0,1]")
    if args.max_tracklets is not None and args.max_tracklets <= 0:
        raise ValueError("max-tracklets must be positive")
    if args.state_policy == "router" and not args.router_sidecar:
        raise ValueError("router state policy requires --router-sidecar")
    if args.state_policy == "observation" and args.router_sidecar:
        raise ValueError("round-0 observation policy must not load a router")
    if args.round != (1 if args.state_policy == "router" else 0):
        raise ValueError("round must be 0=observation or 1=router")

    from datasets import get_dataset
    from models import get_model

    raw_config = load_yaml_config(args.config)
    raw_config.update({
        "seed": args.seed,
        "test_split": args.split,
        "proposal_inference_mode": "obs_vs_all",
        "use_motion_conditioned_search_v3": True,
        "use_action_consistent_router_v3": True,
        "router_v3_enabled_scale": 1.0,
        "preloading": bool(args.preloading),
    })
    if args.path:
        raw_config["path"] = args.path
    config = ConfigMap(raw_config)
    device = resolve_device(args.device)
    dataset = get_dataset(
        config, type="test", split=args.split, protocol_role="test")
    model = get_model(config.net_model)(config)
    load_report = load_matching_v3_model_state(model, args.checkpoint)
    router_payload = None
    if args.router_sidecar:
        router_payload = load_v3_router_sidecar(
            model.action_consistent_router_v3,
            args.router_sidecar,
            require_passed=True)
        calibration = router_payload.get("calibration", {})
        if calibration.get("partition") != "dev":
            raise RuntimeError(
                "round-1 state policy requires a dev-calibrated provisional router")
        source_manifest = router_payload.get("rollout_manifest", {})
        expected_config_hash = canonical_sha256(raw_config)
        expected_source = {
            "round": 0,
            "state_policy": "observation",
            "split": args.split,
            "seed": args.seed,
            "horizon": args.horizon,
            "gamma": args.gamma,
            "candidate_checkpoint_sha256": load_report[
                "checkpoint_sha256"],
            "config_sha256": expected_config_hash,
        }
        mismatched = [
            key for key, expected in expected_source.items()
            if source_manifest.get(key) != expected]
        if mismatched:
            raise RuntimeError(
                "provisional router does not match this round-1 export: "
                + ", ".join(mismatched))
        if int(router_payload.get("training", {}).get("seed", -1)) != args.seed:
            raise RuntimeError("provisional router seed does not match round-1")
    model.to(device)
    model.eval()

    source_dataset = getattr(dataset, "dataset", dataset)
    tracklet_count = len(dataset)
    if args.max_tracklets is not None:
        tracklet_count = min(tracklet_count, args.max_tracklets)
    rows = []
    partition_tracklets = {"train": 0, "dev": 0, "calibration": 0}
    state_policy_id = (
        ActionConsistentInnovationRouter.POLICY_OBSERVATION
        if args.state_policy == "observation"
        else ActionConsistentInnovationRouter.POLICY_AUTO)
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
                    state_box, _ = forward_step(
                        model, sequence, frame_id, history, state_policy_id)
                    history.append(state_box)
                    continue

                observation_boxes, context_output = rollout_branch(
                    model, sequence, frame_id, history, args.horizon,
                    ActionConsistentInnovationRouter.POLICY_OBSERVATION)
                if context_output is None:
                    history.append(observation_boxes[0])
                    continue
                observation_cost = rollout_cost(
                    model, sequence, frame_id, observation_boxes, args.gamma)
                candidate_valid = context_output[
                    "router_v3_candidate_valid"].detach().cpu().numpy(
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
                            first_policy=candidate,
                            first_step=step_ratio)
                        cost = rollout_cost(
                            model, sequence, frame_id, branch_boxes, args.gamma)
                        candidate_cost[candidate, step_id] = cost
                        signed_gain[candidate, step_id] = observation_cost - cost
                rows.append({
                    "tracklet_id": np.int64(tracklet_id),
                    "tracklet_key": str(tracklet_key),
                    "partition": partition,
                    "frame_id": np.int64(frame_id),
                    "router_features": context_output[
                        "router_v3_features"].detach().cpu().numpy(
                        ).reshape(-1).astype(np.float32),
                    "candidate_valid": candidate_valid,
                    "candidate_residual_xy": context_output[
                        "router_v3_candidate_residual_xy"].detach(
                        ).cpu().numpy().reshape(2, 2).astype(np.float32),
                    "signed_gain": signed_gain,
                    "candidate_cost": candidate_cost,
                    "observation_cost": np.float32(observation_cost),
                    "rollout_length": np.int64(args.horizon),
                })
                emitted += 1
                if args.state_policy == "observation":
                    history.append(observation_boxes[0])
                else:
                    state_box, _ = forward_step(
                        model, sequence, frame_id, history,
                        ActionConsistentInnovationRouter.POLICY_AUTO)
                    history.append(state_box)
            print(
                f"[{tracklet_id + 1}/{tracklet_count}] {tracklet_key}: "
                f"{emitted} states ({partition}, {args.state_policy})")

    router_hash = (
        sha256_file(args.router_sidecar) if args.router_sidecar else None)
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
        "round": args.round,
        "state_policy": args.state_policy,
        "state_policy_router": (
            str(Path(args.router_sidecar).resolve())
            if args.router_sidecar else None),
        "state_policy_router_sha256": router_hash,
        "state_policy_calibration": (
            router_payload.get("calibration") if router_payload else None),
        "tracklets_evaluated": tracklet_count,
        "partition_tracklets": partition_tracklets,
        "policy_after_intervention": "explicit_observation",
        "gt_usage": "cost_calculation_after_closed_loop_prediction_only",
    }
    npz_path, manifest_path = write_v3_rollout_artifact(
        args.output, rows, manifest)
    print(json.dumps({
        "rollout": str(npz_path),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "round": args.round,
        "state_policy": args.state_policy,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
