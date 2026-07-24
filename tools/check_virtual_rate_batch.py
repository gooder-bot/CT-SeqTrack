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


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_raw_dataset(wrapped):
    return getattr(wrapped, "dataset", wrapped)


def anno_timestamp(dataset, anno):
    timestamp_reader = getattr(dataset, "_anno_timestamp", None)
    if not callable(timestamp_reader):
        raise TypeError(
            f"{dataset.__class__.__name__} does not expose _anno_timestamp")
    return float(timestamp_reader(anno))


def print_summary(dataset):
    summary = getattr(dataset, "virtual_rate_summary", {})
    print("virtual_rate_summary:")
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print("num_tracklets:", dataset.get_num_tracklets())
    print("num_frames_total:", dataset.get_num_frames_total())
    if hasattr(dataset, "kitti_hv_intervals"):
        print("kitti_hv_intervals:", dataset.kitti_hv_intervals)
    print("virtual_rate_selection_sha256:", getattr(
        dataset, "virtual_rate_selection_sha256", ""))
    print("virtual_rate_manifest_content_sha256:", getattr(
        dataset, "virtual_rate_manifest_content_sha256", ""))
    print("dynamics_time_summary:", getattr(dataset, "dynamics_time_summary", {}))


def print_tracklets(dataset, limit):
    meta_list = getattr(dataset, "virtual_rate_meta", [])
    if not meta_list:
        meta_list = [
            {
                "source_tracklet": idx,
                "original_len": dataset.get_num_frames_tracklet(idx),
                "kept_len": dataset.get_num_frames_tracklet(idx),
                "keep_indices": list(range(dataset.get_num_frames_tracklet(idx))),
            }
            for idx in range(dataset.get_num_tracklets())
        ]

    print("tracklet_examples:")
    for out_idx, meta in enumerate(meta_list[:limit]):
        annos = dataset.tracklet_anno_list[out_idx]
        timestamps = np.array(
            [anno_timestamp(dataset, anno) for anno in annos],
            dtype=np.float64)
        gaps = np.diff(timestamps)
        rounded_gaps = np.round(gaps, 6).tolist()
        cv = float(gaps.std() / gaps.mean()) if gaps.size > 0 and gaps.mean() > 0 else 0.0
        print(
            f"  tracklet[{out_idx}] source={meta.get('source_tracklet')} "
            f"len={meta.get('kept_len')}/{meta.get('original_len')} "
            f"keep={meta.get('keep_indices')}"
        )
        print(f"    timestamp_gaps={rounded_gaps}")
        print(f"    gap_mean={gaps.mean() if gaps.size else 0.0:.6f} gap_cv={cv:.6f}")


def has_full_history(batch, hist_num):
    if isinstance(batch, dict) and "view_a" in batch and "view_b" in batch:
        batch = batch["view_a"]
    if "valid_mask" not in batch:
        return False
    valid_mask = to_numpy(batch["valid_mask"])
    if valid_mask.ndim == 1:
        return valid_mask.sum() >= int(hist_num)
    return valid_mask[0].sum() >= int(hist_num)


def print_loaded_batch(cfg, split, role, args):
    dataset = get_dataset(
        cfg, type=cfg.train_type, split=split, protocol_role=role)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    batch = None
    for batch_idx, candidate in enumerate(loader):
        if batch_idx < args.skip_batches:
            continue
        if args.require_full_history and not has_full_history(candidate, cfg.hist_num):
            continue
        batch = candidate
        print("loaded_batch_idx:", batch_idx)
        break
    if batch is None:
        raise RuntimeError("No batch available.")
    if isinstance(batch, dict) and "view_a" in batch and "view_b" in batch:
        batch = batch["view_a"]

    print("loaded_batch:")
    for key in ("prev_frame_ids", "history_offsets", "valid_mask", "timestamps_real",
                "delta_T_real", "delta_t_real", "current_delta_t_real",
                "timestamps_effective", "delta_T_effective", "delta_t_effective",
                "current_delta_t_effective", "dynamics_time_mode_id",
                "current_timestamp"):
        if key not in batch:
            print(f"  {key}: <missing>")
            continue
        value = to_numpy(batch[key])
        shown = value[0] if value.ndim > 0 else value
        print(f"  {key} shape={value.shape}: {shown}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--role", choices=("train", "val", "test"), default=None)
    parser.add_argument("--type", default="test",
                        help="Use test for metadata-only inspection; train_motion_mf for train sampler.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--load-batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--require-full-history", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    cfg.preloading = False
    cfg.batch_size = args.batch_size
    cfg.workers = args.workers

    split = args.split
    if split is None:
        split = cfg.test_split if args.type == "test" else cfg.train_split

    role = args.role or ('train' if args.type.startswith('train') else 'test')
    wrapped = get_dataset(
        cfg, type=args.type, split=split, protocol_role=role)
    dataset = get_raw_dataset(wrapped)
    print(f"cfg: {args.cfg}")
    print(f"split: {split}")
    print_summary(dataset)
    print_tracklets(dataset, args.limit)
    if args.load_batch:
        print_loaded_batch(cfg, split, role, args)


if __name__ == "__main__":
    main()
