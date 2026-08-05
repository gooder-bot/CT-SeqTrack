#!/usr/bin/env python3
"""Freeze the B3 q10 threshold using real recursive calibration tracklets."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v2 import (  # noqa: E402
    ActionConsistentInnovationRouter,
    SELECTIVE_V4_ROUTER_SCHEMA,
    stable_tracklet_partition,
)
from tools.selective_innovation_common import (  # noqa: E402
    sha256_file,
    torch_load,
)
from tools.selective_v3_common import (  # noqa: E402
    ConfigMap,
    load_matching_v3_model_state,
    load_v3_router_sidecar,
)
from utils.config import load_yaml_config  # noqa: E402
from utils.replay_cache import b2_candidate_config_sha256  # noqa: E402


CALIBRATION_METHOD = "recursive_tracklet_scan_v4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate B3 on actual recursive mini_train tracklets")
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--promotion", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs/ct_v2/18_b1_b2_b3_selective.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--path")
    parser.add_argument("--split", default="mini_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-harm-rate", type=float, default=0.05)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def select_recursive_calibration_result(
        results, baseline_success, max_harm_rate=0.05):
    """Choose the best non-empty safe policy, treating abstention as baseline."""
    accepted = []
    for result in results:
        count = int(result.get("intervention_count", 0))
        success = float(result.get("recursive_success", float("nan")))
        harm_rate = float(result.get("harm_rate", float("nan")))
        if (count <= 0 or not np.isfinite(success)
                or not np.isfinite(harm_rate)
                or harm_rate > float(max_harm_rate)
                or success <= float(baseline_success) + 1e-12):
            continue
        accepted.append(result)
    if not accepted:
        raise RuntimeError(
            "no non-empty recursive q10 policy beats observation while "
            "satisfying harmful intervention <=5%")
    return max(accepted, key=lambda item: (
        float(item["recursive_success"]),
        -float(item["harm_rate"]),
        int(item["intervention_count"]),
        float(item["threshold"]),
    ))


def _calibration_sequences(dataset, seed):
    source_dataset = getattr(dataset, "dataset", dataset)
    selected = []
    for tracklet_id in range(len(dataset)):
        key = (
            source_dataset.get_tracklet_key(tracklet_id)
            if hasattr(source_dataset, "get_tracklet_key")
            else f"mini_train/tracklet/{tracklet_id}")
        if stable_tracklet_partition(key, seed) == "calibration":
            selected.append((tracklet_id, str(key)))
    if not selected:
        raise RuntimeError("mini_train contains no calibration tracklets")
    return selected


def _evaluate_same_input(model, data, reference_box, policy):
    """Execute a policy without rebuilding or resampling the current input."""
    if torch.sum(data["points"][:, :, :3]) == 0:
        return reference_box, None
    data["selective_v3_policy_override"] = torch.tensor(
        [int(policy)], device=model.device, dtype=torch.long)
    candidate_box, _, output = model.evaluate_one_sample(
        data, ref_box=reference_box)
    return candidate_box, output


def evaluate_recursive_policy(
        model, dataset, selected_tracklets, *, threshold=None):
    """Measure frame-weighted Success and local harmful interventions."""
    from utils.metrics import estimateOverlap

    router = model.action_consistent_router_v3
    observation_only = threshold is None
    if not observation_only:
        router.set_gain_threshold(float(threshold))
    all_ious = []
    per_tracklet_success = []
    intervention_count = 0
    harmful_count = 0
    with torch.inference_mode():
        for tracklet_id, _ in selected_tracklets:
            sequence = dataset[tracklet_id]
            history = [sequence[0]["3d_bbox"]]
            tracklet_ious = [1.0]
            for frame_id in range(1, len(sequence)):
                data, reference_box = model.build_input_dict(
                    sequence, frame_id, history)
                policy = (
                    ActionConsistentInnovationRouter.POLICY_OBSERVATION
                    if observation_only
                    else ActionConsistentInnovationRouter.POLICY_AUTO)
                predicted_box, output = _evaluate_same_input(
                    model, data, reference_box, policy)
                target = sequence[frame_id]["3d_bbox"]
                predicted_iou = float(estimateOverlap(
                    target, predicted_box, dim=model.config.IoU_space,
                    up_axis=model.config.up_axis))
                selected = 0
                if output is not None and not observation_only:
                    selected = int(output[
                        "router_v3_selected_candidate"
                    ].detach().cpu().reshape(-1)[0])
                if selected > 0:
                    intervention_count += 1
                    observation_box, _ = _evaluate_same_input(
                        model, data, reference_box,
                        ActionConsistentInnovationRouter.POLICY_OBSERVATION)
                    observation_iou = float(estimateOverlap(
                        target, observation_box, dim=model.config.IoU_space,
                        up_axis=model.config.up_axis))
                    harmful_count += int(predicted_iou < observation_iou)
                history.append(predicted_box)
                tracklet_ious.append(predicted_iou)
            all_ious.extend(tracklet_ious)
            per_tracklet_success.append(float(np.mean(tracklet_ious)))
    return {
        "threshold": None if threshold is None else float(threshold),
        "recursive_success": float(np.mean(all_ious)),
        "tracklet_mean_success": float(np.mean(per_tracklet_success)),
        "frame_count": len(all_ious),
        "tracklet_count": len(selected_tracklets),
        "intervention_count": intervention_count,
        "harmful_intervention_count": harmful_count,
        "harm_rate": (
            float(harmful_count / intervention_count)
            if intervention_count else 0.0),
        "coverage": float(
            intervention_count / max(1, len(all_ious) - len(selected_tracklets))),
    }


def main():
    args = parse_args()
    if args.split != "mini_train":
        raise ValueError("formal B3 calibration is restricted to mini_train")
    if not 0.0 <= args.max_harm_rate <= 1.0:
        raise ValueError("max-harm-rate must be in [0,1]")
    router_payload = torch_load(args.router)
    if router_payload.get("schema") != SELECTIVE_V4_ROUTER_SCHEMA:
        raise RuntimeError("recursive calibration requires a v4 router sidecar")
    screening = router_payload.get("calibration", {})
    if (screening.get("status") != "passed"
            or screening.get("partition") != "calibration"
            or screening.get("method") != "counterfactual_h3_screening"
            or screening.get("final_recursive") is not False):
        raise RuntimeError(
            "input router must be the calibration-partition H=3 screening sidecar")
    thresholds = screening.get("threshold_candidates", [])
    if not thresholds or not all(np.isfinite(thresholds)):
        raise RuntimeError("router sidecar has no finite threshold scan grid")

    promotion_path = Path(args.promotion)
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    candidate_sha = sha256_file(args.candidate_checkpoint)
    promotion_sha = sha256_file(promotion_path)
    if (promotion.get("schema") != "ct_seqtrack.b2_v3_promotion.v2"
            or promotion.get("status") != "passed"
            or promotion.get("candidate_checkpoint_sha256") != candidate_sha):
        raise RuntimeError("recursive calibration requires the matched B2 promotion")
    if (router_payload.get("candidate_checkpoint_sha256") != candidate_sha
            or router_payload.get(
                "promotion_manifest_sha256") != promotion_sha):
        raise RuntimeError("router/candidate/promotion provenance mismatch")

    from datasets import get_dataset
    from models import get_model

    raw_config = load_yaml_config(args.config)
    raw_config.update({
        "seed": args.seed,
        "test_split": args.split,
        "proposal_inference_mode": "selective",
        "use_motion_conditioned_search_v3": True,
        "use_action_consistent_router_v3": True,
        "router_v3_enabled_scale": 1.0,
        "preloading": bool(args.preloading),
    })
    if args.path:
        raw_config["path"] = args.path
    config_hash = b2_candidate_config_sha256(raw_config)
    if promotion.get("candidate_config_sha256") != config_hash:
        raise RuntimeError(
            "recursive calibration candidate config differs from promotion")
    if router_payload.get("candidate_config_sha256") != config_hash:
        raise RuntimeError("recursive calibration config differs from rollouts")
    config = ConfigMap(raw_config)
    dataset = get_dataset(
        config, type="test", split=args.split, protocol_role="test")
    model = get_model(config.net_model)(config)
    load_report = load_matching_v3_model_state(
        model, args.candidate_checkpoint)
    if load_report["checkpoint_sha256"] != candidate_sha:
        raise RuntimeError("loaded candidate checkpoint SHA changed")
    load_v3_router_sidecar(
        model.action_consistent_router_v3, args.router,
        require_passed=True)
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    selected_tracklets = _calibration_sequences(dataset, args.seed)
    baseline = evaluate_recursive_policy(
        model, dataset, selected_tracklets, threshold=None)
    results = []
    scan_thresholds = sorted(set(map(float, thresholds)))
    for index, threshold in enumerate(scan_thresholds):
        result = evaluate_recursive_policy(
            model, dataset, selected_tracklets, threshold=threshold)
        results.append(result)
        print(
            f"[{index + 1}/{len(scan_thresholds)}] threshold={threshold:.7g} "
            f"Success={result['recursive_success']:.6f} "
            f"harm={result['harm_rate']:.4f} "
            f"n={result['intervention_count']}")
    best = select_recursive_calibration_result(
        results, baseline["recursive_success"], args.max_harm_rate)
    calibration = {
        **best,
        "status": "passed",
        "partition": "calibration",
        "method": CALIBRATION_METHOD,
        "final_recursive": True,
        "baseline_recursive_success": baseline["recursive_success"],
        "recursive_success_gain": (
            best["recursive_success"] - baseline["recursive_success"]),
        "max_harm_rate": float(args.max_harm_rate),
        "candidate_count": len(results),
        "dataset": str(getattr(config, "dataset", "")),
        "split": args.split,
        "seed": args.seed,
    }

    output_payload = copy.deepcopy(router_payload)
    output_payload["calibration"] = calibration
    output_payload["recursive_calibration"] = {
        "baseline": baseline,
        "threshold_results": results,
        "tracklet_keys": [key for _, key in selected_tracklets],
        "candidate_checkpoint_sha256": candidate_sha,
        "candidate_config_sha256": config_hash,
        "promotion_manifest_sha256": promotion_sha,
    }
    router_state = output_payload["router_state_dict"]
    threshold_key = "calibrated_gain_threshold"
    if threshold_key not in router_state:
        raise RuntimeError("router state lacks calibrated_gain_threshold")
    router_state[threshold_key] = torch.as_tensor(
        calibration["threshold"],
        dtype=router_state[threshold_key].dtype).reshape_as(
            router_state[threshold_key])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output)
    summary = {
        "schema": SELECTIVE_V4_ROUTER_SCHEMA,
        "output": str(output),
        "calibration": calibration,
        "baseline": baseline,
    }
    summary_path = output.with_suffix(output.suffix + ".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
