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
    for key in ("timestamps", "delta_T", "delta_t", "current_delta_t",
                "current_timestamp", "num_points_in_search", "valid_mask"):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
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
    else:
        print("shape_mismatches: none")

    current_a = to_numpy(view_a["current_timestamp"])
    current_b = to_numpy(view_b["current_timestamp"])
    delta_a = to_numpy(view_a["delta_T"])
    delta_b = to_numpy(view_b["delta_T"])
    anchor_a = to_numpy(view_a["ref_boxs"])[:, 0]
    anchor_b = to_numpy(view_b["ref_boxs"])[:, 0]

    timestamp_eps = float(getattr(cfg, "twc_timestamp_eps", 1e-6))
    anchor_eps = float(getattr(cfg, "twc_anchor_eps", 1e-4))
    delta_eps = float(getattr(cfg, "twc_delta_eps", 1e-5))

    same_current_timestamp = np.abs(current_a - current_b) <= timestamp_eps
    same_anchor_ref_box = np.max(np.abs(anchor_a - anchor_b), axis=1) <= anchor_eps
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
    twc_valid = (
        same_current_timestamp.reshape(-1)
        & same_anchor_ref_box
        & different_history_path
        & full_history_a
        & full_history_b
    )

    print(f"same_current_timestamp: {same_current_timestamp.tolist()}")
    print(f"same_anchor_ref_box: {same_anchor_ref_box.tolist()}")
    print(f"history_difference_source: {history_source}")
    print(f"history_gap: {history_gap.tolist()}")
    print(f"different_history_path: {different_history_path.tolist()}")
    print(f"full_history_a: {full_history_a.tolist()}")
    print(f"full_history_b: {full_history_b.tolist()}")
    print(f"twc_valid: {twc_valid.tolist()}")

    if not np.any(twc_valid):
        raise RuntimeError("No valid TWC sample in this batch.")


if __name__ == "__main__":
    main()
