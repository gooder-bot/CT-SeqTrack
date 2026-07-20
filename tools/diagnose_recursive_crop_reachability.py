import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EasyDict = None
crop_diag = None
get_dataset = None
get_model = None
torch = None


MODE_PREVIOUS_GT = "previous_gt"
MODE_PREVIOUS_PRED = "previous_a1_pred"
MODE_GT_CV = "gt_history_cv"
MODE_PRED_CV = "a1_pred_history_cv"
MODES = (MODE_PREVIOUS_GT, MODE_PREVIOUS_PRED, MODE_GT_CV, MODE_PRED_CV)


def load_runtime_dependencies():
    global EasyDict, crop_diag, get_dataset, get_model, torch
    if EasyDict is not None:
        return

    import torch as torch_module
    from easydict import EasyDict as EasyDictClass

    from datasets import get_dataset as get_dataset_function
    from models import get_model as get_model_function
    from tools import diagnose_crop_reachability as crop_diag_module

    crop_diag_module.load_runtime_dependencies()
    EasyDict = EasyDictClass
    crop_diag = crop_diag_module
    get_dataset = get_dataset_function
    get_model = get_model_function
    torch = torch_module


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        cfg = EasyDict(yaml.load(config_file, Loader=yaml.FullLoader))
    if "preloading" not in cfg:
        cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_center(box):
    return np.asarray(box.center, dtype=np.float64)


def center_error(anchor_box, target_box):
    return float(np.linalg.norm(box_center(anchor_box) - box_center(target_box)))


def constant_velocity_box(
    previous_previous_box,
    previous_box,
    previous_previous_time,
    previous_time,
    current_time,
):
    anchor = copy.deepcopy(previous_box)
    history_gap = float(previous_time) - float(previous_previous_time)
    query_gap = float(current_time) - float(previous_time)
    if history_gap <= 0.0 or query_gap <= 0.0:
        return anchor, False
    if not np.isfinite(history_gap) or not np.isfinite(query_gap):
        return anchor, False

    velocity = (box_center(previous_box) - box_center(previous_previous_box)) / history_gap
    anchor.center = box_center(previous_box) + velocity * query_gap
    return anchor, True


def endpoint_key(tracklet_id, frame_index, token):
    return int(tracklet_id), int(frame_index), str(token)


def load_reference_endpoints(path):
    ordered = []
    seen = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as input_file:
        for row in csv.DictReader(input_file):
            if row.get("crop_mode") not in (None, "", "base"):
                continue
            key = endpoint_key(row["tracklet_id"], row["frame_index"], row["frame_token"])
            if key in seen:
                raise RuntimeError(f"Duplicate reference endpoint: {key}")
            seen.add(key)
            ordered.append(key)
    if not ordered:
        raise RuntimeError(f"No base endpoints found in reference CSV: {path}")
    return ordered


def resolve_device(device_text):
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device_text}")
    return device


def load_a1_model(cfg, weights_path, device):
    model_class = get_model(cfg.net_model)
    model = model_class.load_from_checkpoint(
        str(weights_path),
        config=cfg,
        map_location="cpu",
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_baseline_step(model, sequence, frame_index, results_bbs):
    data_dict, reference_box = model.build_input_dict(sequence, frame_index, results_bbs)
    empty_fallback = bool(torch.sum(data_dict["points"][:, :, :3]).item() == 0.0)
    if empty_fallback:
        candidate_box = copy.deepcopy(reference_box)
    else:
        candidate_box, *_ = model.evaluate_one_sample(data_dict, ref_box=reference_box)
    return candidate_box, empty_fallback, int(data_dict["num_points_in_search"].item())


def summarize_mode_extras(rows):
    result = {}
    for mode in MODES:
        subset = [row for row in rows if row["crop_mode"] == mode]
        result[mode] = {
            "anchor_error": crop_diag.finite_summary([row["anchor_error"] for row in subset]),
            "center_outside_streak": crop_diag.finite_summary(
                [row["center_outside_streak"] for row in subset]
            ),
            "target_loss_streak": crop_diag.finite_summary(
                [row["target_loss_streak"] for row in subset]
            ),
            "available_rate": float(np.mean([row["anchor_available"] for row in subset]))
            if subset
            else None,
        }
    return result


def summarize_baseline_predictions(endpoint_rows):
    if not endpoint_rows:
        return {}
    return {
        "previous_prediction_error": crop_diag.finite_summary(
            [row["previous_prediction_error"] for row in endpoint_rows]
        ),
        "current_prediction_error": crop_diag.finite_summary(
            [row["current_prediction_error"] for row in endpoint_rows]
        ),
        "baseline_search_point_count": crop_diag.finite_summary(
            [row["baseline_search_point_count"] for row in endpoint_rows]
        ),
        "empty_fallback_rate": float(np.mean([row["empty_fallback"] for row in endpoint_rows])),
        "empty_fallback_count": int(np.sum([row["empty_fallback"] for row in endpoint_rows])),
    }


def validate_reference(
    observed_keys,
    reference_keys,
    tracklet_limit,
    allow_partial_reference,
):
    if reference_keys is None:
        return None

    expected = {key for key in reference_keys if key[0] < tracklet_limit}
    observed = set(observed_keys)
    unexpected = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unexpected:
        raise RuntimeError(
            f"Observed {len(unexpected)} endpoints absent from the reference; first={unexpected[:3]}"
        )
    if missing and not allow_partial_reference:
        raise RuntimeError(
            f"Missing {len(missing)} reference endpoints; first={missing[:3]}. "
            "Use --allow-partial-reference only for an intentional smoke run."
        )
    return {
        "reference_endpoint_count": len(expected),
        "observed_endpoint_count": len(observed),
        "missing_endpoint_count": len(missing),
        "unexpected_endpoint_count": len(unexpected),
        "allow_partial_reference": bool(allow_partial_reference),
        "exact_match": not missing and not unexpected,
    }


def self_test():
    class DummyBox:
        def __init__(self, center):
            self.center = np.asarray(center, dtype=np.float64)

    older = DummyBox([0.0, 0.0, 0.0])
    newer = DummyBox([2.0, 0.0, 0.0])
    predicted, available = constant_velocity_box(older, newer, 0.0, 2.0, 5.0)
    if not available or not np.allclose(predicted.center, [5.0, 0.0, 0.0]):
        raise RuntimeError(f"constant-velocity self-test failed: {predicted.center}")

    fallback, available = constant_velocity_box(older, newer, 2.0, 2.0, 3.0)
    if available or not np.allclose(fallback.center, newer.center):
        raise RuntimeError("invalid-time fallback self-test failed")

    keys = [(0, 3, "a"), (0, 4, "b")]
    report = validate_reference(keys, keys, tracklet_limit=1, allow_partial_reference=False)
    if not report["exact_match"]:
        raise RuntimeError(f"reference self-test failed: {report}")
    print("recursive crop reachability self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one baseline A1 recursive trajectory and compare previous-GT, previous-A1, "
            "GT-history CV, and A1-prediction-history CV crop reachability on the same endpoints."
        )
    )
    parser.add_argument("--cfg")
    parser.add_argument("--weights")
    parser.add_argument("--reference-endpoints-csv", default=None)
    parser.add_argument("--allow-partial-reference", action="store_true")
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--max-endpoints", type=int, default=None)
    parser.add_argument("--require-full-history", action="store_true")
    parser.add_argument("--target-wlh-factor", type=float, default=1.0)
    parser.add_argument("--delta-t-bins", default="0.5,1.0,2.0")
    parser.add_argument("--displacement-bins", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--target-point-bins", default="0,5,20")
    parser.add_argument("--prediction-error-bins", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--output-dir", default="output/diagnostics/recursive_crop_reachability")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--model-load-smoke",
        action="store_true",
        help="Load the configured A1 checkpoint on the requested device and exit before dataset construction.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.cfg is None or args.weights is None:
        parser.error("--cfg and --weights are required unless --self-test is used.")

    load_runtime_dependencies()
    cfg_path = Path(args.cfg).resolve()
    weights_path = Path(args.weights).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    cfg = load_config(cfg_path)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    cfg.preloading = bool(args.preloading)
    split = args.split if args.split is not None else cfg.train_split

    device = resolve_device(args.device)
    model = load_a1_model(cfg, weights_path, device)
    if args.model_load_smoke:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            "recursive crop model-load smoke: PASS "
            f"device={device} parameters={parameter_count} weights_sha256={sha256_file(weights_path)}"
        )
        return
    sampler = get_dataset(cfg, type="test", split=split)
    dataset = getattr(sampler, "dataset", sampler)

    reference_path = (
        Path(args.reference_endpoints_csv).resolve()
        if args.reference_endpoints_csv is not None
        else None
    )
    reference_keys = load_reference_endpoints(reference_path) if reference_path else None

    base_scale = float(cfg.bb_scale)
    base_offset = float(cfg.bb_offset)
    hist_num = int(cfg.hist_num)
    tracklet_limit = dataset.get_num_tracklets()
    if args.max_tracklets is not None:
        tracklet_limit = min(tracklet_limit, args.max_tracklets)

    rows = []
    endpoint_rows = []
    observed_keys = []
    endpoint_count = 0
    start_time = time.time()
    stop = False

    with torch.no_grad():
        for tracklet_id in range(tracklet_limit):
            tracklet_length = dataset.get_num_frames_tracklet(tracklet_id)
            if tracklet_length < 2:
                continue
            sequence = [
                dataset.get_frames(tracklet_id, [frame_index])[0]
                for frame_index in range(tracklet_length)
            ]
            results_bbs = [copy.deepcopy(sequence[0]["3d_bbox"])]
            center_outside_streak = {mode: 0 for mode in MODES}
            target_loss_streak = {mode: 0 for mode in MODES}

            for frame_index in range(1, tracklet_length):
                current_frame = sequence[frame_index]
                previous_frame = sequence[frame_index - 1]
                previous_gt = previous_frame["3d_bbox"]
                previous_pred = results_bbs[-1]
                current_gt = current_frame["3d_bbox"]

                previous_time = crop_diag.frame_timestamp(previous_frame, frame_index - 1)
                current_time = crop_diag.frame_timestamp(current_frame, frame_index)
                current_delta_t = float(current_time - previous_time)
                displacement_norm = center_error(previous_gt, current_gt)
                previous_prediction_error = center_error(previous_pred, previous_gt)

                gt_cv_anchor = copy.deepcopy(previous_gt)
                pred_cv_anchor = copy.deepcopy(previous_pred)
                gt_cv_available = False
                pred_cv_available = False
                if frame_index >= 2:
                    older_frame = sequence[frame_index - 2]
                    older_time = crop_diag.frame_timestamp(older_frame, frame_index - 2)
                    gt_cv_anchor, gt_cv_available = constant_velocity_box(
                        older_frame["3d_bbox"],
                        previous_gt,
                        older_time,
                        previous_time,
                        current_time,
                    )
                    pred_cv_anchor, pred_cv_available = constant_velocity_box(
                        results_bbs[-2],
                        previous_pred,
                        older_time,
                        previous_time,
                        current_time,
                    )

                anchors = {
                    MODE_PREVIOUS_GT: (previous_gt, True),
                    MODE_PREVIOUS_PRED: (previous_pred, True),
                    MODE_GT_CV: (gt_cv_anchor, gt_cv_available),
                    MODE_PRED_CV: (pred_cv_anchor, pred_cv_available),
                }
                mode_metrics = {}
                for mode, (anchor, available) in anchors.items():
                    metrics = crop_diag.evaluate_crop(
                        current_frame["pc"],
                        current_gt,
                        anchor,
                        base_scale,
                        base_offset,
                        args.target_wlh_factor,
                    )
                    mode_metrics[mode] = (anchor, available, metrics)

                candidate_box, empty_fallback, baseline_search_point_count = run_baseline_step(
                    model, sequence, frame_index, results_bbs
                )
                results_bbs.append(candidate_box)
                current_prediction_error = center_error(candidate_box, current_gt)

                full_history = frame_index >= hist_num
                if args.require_full_history and not full_history:
                    continue

                token = crop_diag.frame_token(current_frame, f"{tracklet_id}:{frame_index}")
                key = endpoint_key(tracklet_id, frame_index, token)
                observed_keys.append(key)
                endpoint_metadata = {
                    "tracklet_id": tracklet_id,
                    "frame_index": frame_index,
                    "frame_token": token,
                    "current_delta_t": current_delta_t,
                    "displacement_norm": displacement_norm,
                    "previous_prediction_error": previous_prediction_error,
                    "current_prediction_error": current_prediction_error,
                    "baseline_search_point_count": baseline_search_point_count,
                    "empty_fallback": empty_fallback,
                    "full_history": full_history,
                }
                endpoint_rows.append(dict(endpoint_metadata))

                for mode in MODES:
                    anchor, available, metrics = mode_metrics[mode]
                    center_outside_streak[mode] = (
                        0 if metrics["center_inside"] else center_outside_streak[mode] + 1
                    )
                    target_lost = (
                        metrics["target_point_count"] > 0 and not metrics["has_target_point"]
                    )
                    target_loss_streak[mode] = (
                        target_loss_streak[mode] + 1 if target_lost else 0
                    )
                    row = dict(endpoint_metadata)
                    row.update(
                        {
                            "crop_mode": mode,
                            "anchor_available": bool(available),
                            "anchor_error": center_error(anchor, current_gt),
                            "center_outside_streak": center_outside_streak[mode],
                            "target_loss_streak": target_loss_streak[mode],
                        }
                    )
                    row.update(metrics)
                    rows.append(row)

                endpoint_count += 1
                if args.max_endpoints is not None and endpoint_count >= args.max_endpoints:
                    stop = True
                    break
            if stop:
                break

    if not rows:
        raise RuntimeError("No recursive diagnostic endpoints were produced.")

    reference_report = validate_reference(
        observed_keys,
        reference_keys,
        tracklet_limit,
        args.allow_partial_reference,
    )
    delta_t_bins = crop_diag.parse_float_list(args.delta_t_bins)
    displacement_bins = crop_diag.parse_float_list(args.displacement_bins)
    target_point_bins = crop_diag.parse_float_list(args.target_point_bins)
    prediction_error_bins = crop_diag.parse_float_list(args.prediction_error_bins)
    summary = crop_diag.summarize_rows(
        rows,
        delta_t_bins,
        displacement_bins,
        target_point_bins,
    )
    summary.update(
        {
            "mode_extras": summarize_mode_extras(rows),
            "baseline_predictions": summarize_baseline_predictions(endpoint_rows),
            "previous_prediction_error_buckets": crop_diag.summarize_buckets(
                rows, "previous_prediction_error", prediction_error_bins
            ),
            "cfg": str(cfg_path),
            "cfg_sha256": sha256_file(cfg_path),
            "weights": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
            "reference_endpoints_csv": str(reference_path) if reference_path else None,
            "reference_endpoints_sha256": sha256_file(reference_path) if reference_path else None,
            "reference_match": reference_report,
            "split": split,
            "virtual_rate_mode": str(getattr(cfg, "virtual_rate_mode", "none")),
            "base_scale": base_scale,
            "base_offset": base_offset,
            "target_wlh_factor": args.target_wlh_factor,
            "require_full_history": args.require_full_history,
            "seed": args.seed,
            "device": str(device),
            "runtime_seconds": float(time.time() - start_time),
            "note": (
                "One baseline A1 trajectory is run recursively with its normal previous-prediction "
                "anchor. The other anchors are passive counterfactual crop diagnostics on the same "
                "endpoint and do not alter the prediction history. previous_gt and gt_history_cv "
                "are oracle references; previous_a1_pred and a1_pred_history_cv are GT-free."
            ),
        }
    )

    tag = args.tag or f"{Path(args.cfg).stem}_{split}"
    output_dir = Path(args.output_dir) / crop_diag.safe_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "recursive_crop_reachability_endpoints.csv"
    summary_path = output_dir / "recursive_crop_reachability_summary.json"
    crop_diag.write_rows(csv_path, rows)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"endpoint csv: {csv_path}")
    print(f"summary json: {summary_path}")


if __name__ == "__main__":
    main()
