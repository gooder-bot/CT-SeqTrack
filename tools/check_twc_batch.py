import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402


def load_config(path):
    with open(path, "r") as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    if "preloading" not in cfg:
        cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def parse_offsets(value):
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def first_row(value):
    array = to_numpy(value)
    return array[0] if array.ndim > 0 else array


def has_full_history(view, hist_num):
    valid_mask = to_numpy(view["valid_mask"])
    if valid_mask.ndim == 1:
        return bool(valid_mask.sum() >= int(hist_num))
    return bool(valid_mask[0].sum() >= int(hist_num))


def summarize_view(name, view):
    print(f"{name} prev_frame_ids: {first_row(view.get('prev_frame_ids', []))}")
    print(f"{name} history_offsets: {first_row(view.get('history_offsets', []))}")
    for key in ("candidate_id", "timestamps", "delta_T", "delta_t", "current_delta_t",
                "current_timestamp", "num_points_in_search", "valid_mask",
                "coordinate_anchor", "candidate_offsets", "point_sampling_seeds",
                "current_sampling_seed"):
        if key not in view:
            print(f"{name} {key}: <missing>")
            continue
        value = to_numpy(view[key])
        print(f"{name} {key} shape={value.shape}: {value[0] if value.ndim > 0 else value}")


def check_shapes(view_a, view_b):
    keys = sorted(set(view_a.keys()) & set(view_b.keys()))
    mismatches = []
    for key in keys:
        value_a, value_b = view_a[key], view_b[key]
        if torch.is_tensor(value_a) and torch.is_tensor(value_b):
            if tuple(value_a.shape) != tuple(value_b.shape):
                mismatches.append((key, tuple(value_a.shape), tuple(value_b.shape)))
        else:
            array_a, array_b = to_numpy(value_a), to_numpy(value_b)
            if array_a.shape != array_b.shape:
                mismatches.append((key, array_a.shape, array_b.shape))
    return mismatches


def check_shared_frame_values(view_a, view_b, value_key, eps):
    required = ("prev_frame_ids", value_key)
    for key in required:
        if key not in view_a or key not in view_b:
            raise RuntimeError(f"TWC shared-coordinate check requires {key} in both views.")

    frame_ids_a = to_numpy(view_a["prev_frame_ids"])
    frame_ids_b = to_numpy(view_b["prev_frame_ids"])
    offsets_a = to_numpy(view_a[value_key])
    offsets_b = to_numpy(view_b[value_key])
    if frame_ids_a.ndim == 1:
        frame_ids_a, frame_ids_b = frame_ids_a[None], frame_ids_b[None]
        offsets_a, offsets_b = offsets_a[None], offsets_b[None]

    shared_ok = []
    max_gaps = []
    for ids_a, ids_b, values_a, values_b in zip(
            frame_ids_a, frame_ids_b, offsets_a, offsets_b):
        map_a = {int(frame_id): values_a[idx] for idx, frame_id in enumerate(ids_a)}
        map_b = {int(frame_id): values_b[idx] for idx, frame_id in enumerate(ids_b)}
        common_ids = sorted(set(map_a) & set(map_b))
        if not common_ids:
            shared_ok.append(False)
            max_gaps.append(float("inf"))
            continue
        gap = max(float(np.max(np.abs(map_a[frame_id] - map_b[frame_id])))
                  for frame_id in common_ids)
        shared_ok.append(gap <= eps)
        max_gaps.append(gap)
    return np.asarray(shared_ok, dtype=bool), np.asarray(max_gaps, dtype=np.float32)


def check_shared_history_xyz(view_a, view_b, hist_num, point_sample_size, eps):
    frame_ids_a = to_numpy(view_a["prev_frame_ids"])
    frame_ids_b = to_numpy(view_b["prev_frame_ids"])
    points_a = to_numpy(view_a["points"])
    points_b = to_numpy(view_b["points"])
    hist_points_a = points_a[:, :hist_num * point_sample_size, :3].reshape(
        -1, hist_num, point_sample_size, 3)
    hist_points_b = points_b[:, :hist_num * point_sample_size, :3].reshape(
        -1, hist_num, point_sample_size, 3)

    shared_ok = []
    max_gaps = []
    for ids_a, ids_b, values_a, values_b in zip(
            frame_ids_a, frame_ids_b, hist_points_a, hist_points_b):
        map_a = {int(frame_id): values_a[idx] for idx, frame_id in enumerate(ids_a)}
        map_b = {int(frame_id): values_b[idx] for idx, frame_id in enumerate(ids_b)}
        common_ids = sorted(set(map_a) & set(map_b))
        if not common_ids:
            shared_ok.append(False)
            max_gaps.append(float("inf"))
            continue
        gap = max(float(np.max(np.abs(map_a[frame_id] - map_b[frame_id])))
                  for frame_id in common_ids)
        shared_ok.append(gap <= eps)
        max_gaps.append(gap)
    return np.asarray(shared_ok, dtype=bool), np.asarray(max_gaps, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--require-full-history", action="store_true")
    parser.add_argument("--pseudo-time", action="store_true")
    parser.add_argument("--view-a-offsets", default=None,
                        help="Comma-separated offsets, e.g. 1,2,3")
    parser.add_argument("--view-b-offsets", default=None,
                        help="Comma-separated offsets, e.g. 1,3,5")
    parser.add_argument("--candidate-zero-only", action="store_true",
                        help="Force paired TWC views to use candidate_id=0.")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.pseudo_time:
        cfg.use_real_time = False
    cfg.batch_size = args.batch_size
    cfg.workers = args.workers
    cfg.use_twc = True
    if args.candidate_zero_only:
        cfg.twc_candidate_zero_only = True
    if args.view_a_offsets is not None:
        cfg.twc_view_a_offsets = parse_offsets(args.view_a_offsets)
    if args.view_b_offsets is not None:
        cfg.twc_view_b_offsets = parse_offsets(args.view_b_offsets)

    split = args.split if args.split is not None else cfg.train_split
    dataset = get_dataset(cfg, type=cfg.train_type, split=split)
    configured_candidates = int(getattr(cfg, "num_candidates", 1))
    expected_length = int(dataset.dataset.get_num_frames_total()) * configured_candidates
    if int(dataset.num_candidates) != configured_candidates:
        raise RuntimeError(
            "TWC changed sampler.num_candidates: "
            f"sampler={dataset.num_candidates}, config={configured_candidates}.")
    if len(dataset) != expected_length:
        raise RuntimeError(
            "TWC changed the number of samples per epoch: "
            f"len(dataset)={len(dataset)}, expected={expected_length}.")
    print(
        f"dataset_length={len(dataset)}, num_candidates={dataset.num_candidates}, "
        f"base_frames={dataset.dataset.get_num_frames_total()}")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    batch = None
    for batch_idx, candidate in enumerate(loader):
        if batch_idx < args.skip_batches:
            continue
        if "view_a" not in candidate or "view_b" not in candidate:
            raise RuntimeError("Dataset did not return paired TWC views. Check cfg.use_twc.")
        if args.require_full_history:
            if not (has_full_history(candidate["view_a"], cfg.hist_num)
                    and has_full_history(candidate["view_b"], cfg.hist_num)):
                continue
        batch = candidate
        print(f"using batch_idx={batch_idx}")
        break

    if batch is None:
        raise RuntimeError("No paired batch matched the requested filters.")

    view_a, view_b = batch["view_a"], batch["view_b"]
    summarize_view("view_a", view_a)
    summarize_view("view_b", view_b)

    shape_mismatches = check_shapes(view_a, view_b)
    if shape_mismatches:
        print("shape_mismatches:")
        for key, shape_a, shape_b in shape_mismatches:
            print(f"  {key}: {shape_a} vs {shape_b}")
        raise RuntimeError("Paired TWC views have incompatible tensor shapes.")
    else:
        print("shape_mismatches: none")

    current_a = to_numpy(view_a["current_timestamp"])
    current_b = to_numpy(view_b["current_timestamp"])
    candidate_id_a = to_numpy(view_a["candidate_id"]).reshape(-1)
    candidate_id_b = to_numpy(view_b["candidate_id"]).reshape(-1)
    delta_a = to_numpy(view_a["delta_T"])
    delta_b = to_numpy(view_b["delta_T"])
    if "coordinate_anchor" not in view_a or "coordinate_anchor" not in view_b:
        raise RuntimeError(
            "Missing coordinate_anchor. The old normalized ref_box check cannot verify "
            "that paired views share a coordinate system.")
    anchor_a = to_numpy(view_a["coordinate_anchor"])
    anchor_b = to_numpy(view_b["coordinate_anchor"])

    timestamp_eps = float(getattr(cfg, "twc_timestamp_eps", 1e-6))
    anchor_eps = float(getattr(cfg, "twc_anchor_eps", 1e-4))
    candidate_offset_eps = float(getattr(cfg, "twc_candidate_offset_eps", 1e-8))
    delta_eps = float(getattr(cfg, "twc_delta_eps", 1e-5))

    same_current_timestamp = np.abs(current_a - current_b) <= timestamp_eps
    same_candidate_id = candidate_id_a == candidate_id_b
    same_coordinate_anchor = np.max(np.abs(anchor_a - anchor_b), axis=1) <= anchor_eps
    shared_candidate_offsets, shared_offset_gap = check_shared_frame_values(
        view_a, view_b, "candidate_offsets", candidate_offset_eps)
    shared_point_sampling_seeds, shared_sampling_seed_gap = check_shared_frame_values(
        view_a, view_b, "point_sampling_seeds", 0.0)
    same_current_sampling_seed = (
        to_numpy(view_a["current_sampling_seed"]).reshape(-1)
        == to_numpy(view_b["current_sampling_seed"]).reshape(-1)
    )
    point_eps = float(getattr(cfg, "twc_point_eps", 1e-6))
    shared_history_xyz, shared_history_xyz_gap = check_shared_history_xyz(
        view_a, view_b, int(cfg.hist_num), int(cfg.point_sample_size), point_eps)
    current_points_a = to_numpy(view_a["points"])[:, -int(cfg.point_sample_size):, :3]
    current_points_b = to_numpy(view_b["points"])[:, -int(cfg.point_sample_size):, :3]
    current_points_gap = np.max(
        np.abs(current_points_a - current_points_b), axis=(1, 2))
    same_current_points = current_points_gap <= point_eps
    if "history_offsets" in view_a and "history_offsets" in view_b:
        history_gap = np.max(np.abs(to_numpy(view_a["history_offsets"])
                                    - to_numpy(view_b["history_offsets"])), axis=1)
        history_source = "history_offsets"
    elif "delta_T_real" in view_a and "delta_T_real" in view_b:
        history_gap = np.max(np.abs(to_numpy(view_a["delta_T_real"])
                                    - to_numpy(view_b["delta_T_real"])), axis=1)
        history_source = "delta_T_real"
    elif "timestamps_real" in view_a and "timestamps_real" in view_b:
        history_gap = np.max(np.abs(to_numpy(view_a["timestamps_real"])[:, :-1]
                                    - to_numpy(view_b["timestamps_real"])[:, :-1]), axis=1)
        history_source = "timestamps_real"
    else:
        history_gap = np.max(np.abs(delta_a - delta_b), axis=1)
        history_source = "delta_T"
    different_history_path = history_gap > delta_eps
    full_history_a = np.sum(to_numpy(view_a["valid_mask"]), axis=1) >= int(cfg.hist_num)
    full_history_b = np.sum(to_numpy(view_b["valid_mask"]), axis=1) >= int(cfg.hist_num)
    same_search_crop_count = (
        to_numpy(view_a["num_points_in_search"]).reshape(-1)
        == to_numpy(view_b["num_points_in_search"]).reshape(-1)
    )
    twc_valid = (
        same_current_timestamp.reshape(-1)
        & same_candidate_id
        & same_coordinate_anchor
        & shared_candidate_offsets
        & shared_point_sampling_seeds
        & same_current_sampling_seed
        & shared_history_xyz
        & same_current_points
        & same_search_crop_count
        & different_history_path
        & full_history_a
        & full_history_b
    )

    print(f"candidate_id_a: {candidate_id_a.tolist()}")
    print(f"candidate_id_b: {candidate_id_b.tolist()}")
    print(f"same_candidate_id: {same_candidate_id.tolist()}")
    print(f"same_current_timestamp: {same_current_timestamp.tolist()}")
    print(f"same_coordinate_anchor: {same_coordinate_anchor.tolist()}")
    print(f"shared_candidate_offsets: {shared_candidate_offsets.tolist()}")
    print(f"shared_candidate_offset_gap: {shared_offset_gap.tolist()}")
    print(f"shared_point_sampling_seeds: {shared_point_sampling_seeds.tolist()}")
    print(f"shared_point_sampling_seed_gap: {shared_sampling_seed_gap.tolist()}")
    print(f"same_current_sampling_seed: {same_current_sampling_seed.tolist()}")
    print(f"shared_history_xyz: {shared_history_xyz.tolist()}")
    print(f"shared_history_xyz_gap: {shared_history_xyz_gap.tolist()}")
    print(f"same_current_points: {same_current_points.tolist()}")
    print(f"current_points_gap: {current_points_gap.tolist()}")
    print(f"history_difference_source: {history_source}")
    print(f"history_gap: {history_gap.tolist()}")
    print(f"different_history_path: {different_history_path.tolist()}")
    print(f"full_history_a: {full_history_a.tolist()}")
    print(f"full_history_b: {full_history_b.tolist()}")
    print(f"same_search_crop_count: {same_search_crop_count.tolist()}")
    print(f"twc_valid: {twc_valid.tolist()}")

    if not np.all(twc_valid):
        raise RuntimeError("At least one paired sample failed the TWC invariants.")

    candidate_zero_only = bool(getattr(cfg, "twc_candidate_zero_only", True))
    if not candidate_zero_only and configured_candidates > 1:
        nonzero = candidate_id_a > 0
        if not np.any(nonzero):
            raise RuntimeError(
                "This batch did not exercise candidate>0. Use batch-size >= num_candidates "
                "and do not pass --candidate-zero-only.")
        anchor_offset_norm = np.linalg.norm(
            to_numpy(view_a["candidate_offsets"])[nonzero, 0], axis=1)
        if not np.all(anchor_offset_norm > 1e-8):
            raise RuntimeError("A nonzero candidate unexpectedly used a zero anchor offset.")
        if not np.all(shared_offset_gap[nonzero] <= candidate_offset_eps):
            raise RuntimeError("Nonzero candidates do not share offsets across TWC views.")
        print(f"nonzero_anchor_offset_norm: {anchor_offset_norm.tolist()}")


if __name__ == "__main__":
    main()
