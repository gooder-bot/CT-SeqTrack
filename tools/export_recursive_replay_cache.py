#!/usr/bin/env python3
"""Export frozen B0/B1 recursive histories without current-GT input fields."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402
from datasets.misc_utils import (  # noqa: E402
    get_history_frame_ids_and_masks,
    get_last_n_bounding_boxes,
)
from utils.checkpoint_loading import load_initial_weights  # noqa: E402
from models import get_model  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402
from utils.replay_cache import (  # noqa: E402
    B0_STATE_PREFIXES,
    B1_STATE_PREFIXES,
    replay_config_sha256,
    sha256_file,
    sha256_json,
    tensor_prefixes_sha256,
    write_recursive_replay_cache,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="frozen B1 checkpoint used for the rollout")
    parser.add_argument("--b0-checkpoint", required=True,
                        help="matched B0 checkpoint provenance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def device_from_arg(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return device


def box_row(box):
    yaw = float(box.orientation.radians * box.orientation.axis[-1])
    return np.concatenate((
        np.asarray(box.center, dtype=np.float64),
        np.asarray(box.wlh, dtype=np.float64),
        np.asarray([yaw], dtype=np.float64),
    )).tolist()


def b1_json(prediction):
    result = {}
    for key in (
            "mu_xy", "log_sigma_parallel_perp", "covariance_xy",
            "basis_velocity_xy", "direction_xy", "velocity_xy", "feature"):
        result[key] = np.asarray(prediction[key]).tolist()
    result.update({
        "valid": bool(prediction["valid"]),
        "gap_ratio": float(prediction["gap_ratio"]),
        "source_id": int(prediction["source_id"]),
    })
    return result


def git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def checkpoint_state(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint has no state_dict")
    for prefix in ("model.", "module."):
        if any(key.startswith(prefix) for key in state):
            state = {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in state.items()
            }
            break
    return state


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw_config = load_yaml_config(args.config)
    raw_config.update({
        "seed": args.seed,
        "test_split": args.split,
        "preloading": bool(args.preloading),
        "use_b1motion_v3": True,
        "use_motion_v3_legacy_fusion": False,
        "use_motion_conditioned_search_v3": False,
        "use_asymmetric_dual_query": False,
        "use_raw_search_candidate": False,
        "use_b1_prepass_support": False,
        "use_uncertainty_geometry": False,
        "use_action_consistent_router_v3": False,
        "b2_v3_freeze_candidate_producers": False,
        "b2_v3_require_packaged_router": False,
        "proposal_inference_mode": "observation",
    })
    if args.path:
        raw_config["path"] = args.path
    config = EasyDict(raw_config)
    dataset = get_dataset(
        config, type="test", split=args.split, protocol_role="test")
    model = get_model(config.net_model)(config)
    load_initial_weights(model, args.checkpoint)
    model_b0_hash = tensor_prefixes_sha256(
        model.state_dict(), B0_STATE_PREFIXES)
    supplied_b0_hash = tensor_prefixes_sha256(
        checkpoint_state(args.b0_checkpoint), B0_STATE_PREFIXES)
    if model_b0_hash != supplied_b0_hash:
        raise RuntimeError(
            "--b0-checkpoint does not match the B0 state used by replay")
    device = device_from_arg(args.device)
    model.to(device).eval()
    source_dataset = getattr(dataset, "dataset", dataset)
    count = len(dataset)
    if args.max_tracklets is not None:
        count = min(count, int(args.max_tracklets))
    records = []
    with torch.inference_mode():
        for tracklet_id in range(count):
            sequence = dataset[tracklet_id]
            tracklet_key = (
                source_dataset.get_tracklet_key(tracklet_id)
                if hasattr(source_dataset, "get_tracklet_key")
                else f"{args.split}/tracklet/{tracklet_id}")
            history = [sequence[0]["3d_bbox"]]
            for frame_id in range(1, len(sequence)):
                prediction = model.predict_motion_prepass(
                    sequence, frame_id, history)
                data, reference_box = model.build_input_dict(
                    sequence, frame_id, history)
                history_ids, valid_mask = get_history_frame_ids_and_masks(
                    frame_id, model.hist_num)
                del history_ids
                history_boxes = get_last_n_bounding_boxes(
                    history, valid_mask)
                records.append({
                    "tracklet_key": str(tracklet_key),
                    "frame_id": int(frame_id),
                    "history_boxes_world": [
                        box_row(box) for box in history_boxes],
                    "history_valid_mask": [int(value) for value in
                                             valid_mask],
                    "delta_t": data[
                        "motion_main_delta_t"][0].detach().cpu().tolist(),
                    "current_delta_t": float(data[
                        "motion_main_current_delta_t"][0].detach().cpu()),
                    "anchor_world": [
                        *np.asarray(history_boxes[0].center,
                                    dtype=np.float64).tolist(),
                        float(history_boxes[0].orientation.radians
                              * history_boxes[0].orientation.axis[-1]),
                    ],
                    "b1": b1_json(prediction),
                    "source": "recursive_frozen_b0_b1",
                })
                if torch.sum(data["points"][:, :, :3]) == 0:
                    history.append(reference_box)
                else:
                    candidate_box, _, _ = model.evaluate_one_sample(
                        data, ref_box=reference_box)
                    history.append(candidate_box)
            print(f"[{tracklet_id + 1}/{count}] {tracklet_key}")
    manifest = {
        "dataset": str(getattr(config, 'dataset', 'unknown')),
        "split": args.split,
        "replay_config_sha256": replay_config_sha256(config),
        "commit": git_commit(),
        "b0_state_sha256": model_b0_hash,
        "b1_state_sha256": tensor_prefixes_sha256(
            model.state_dict(), B1_STATE_PREFIXES),
        "b1_calibration_sha256": sha256_json(getattr(
            model, "_b1_uncertainty_calibration", None)),
        "b0_checkpoint_sha256": sha256_file(args.b0_checkpoint),
        "b1_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "config_path": str(Path(args.config).resolve()),
        "split": args.split,
        "seed": args.seed,
        "tracklets": count,
    }
    written = write_recursive_replay_cache(args.output, manifest, records)
    print(f"wrote {written['record_count']} replay records to {args.output}")


if __name__ == "__main__":
    main()
