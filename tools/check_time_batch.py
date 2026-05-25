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
        return EasyDict(yaml.load(f, Loader=yaml.FullLoader))


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def summarize_batch(batch, cfg):
    points = to_numpy(batch["points"])
    if points.ndim == 2:
        points = points[None, ...]

    hist_num = int(cfg.hist_num)
    point_sample_size = int(cfg.point_sample_size)
    expected_frames = hist_num + 1
    expected_points = expected_frames * point_sample_size

    print("points shape:", points.shape)
    print("expected points per sample:", expected_points)

    sample_points = points[0]
    for frame_idx in range(expected_frames):
        start = frame_idx * point_sample_size
        end = start + point_sample_size
        frame_times = sample_points[start:end, 3]
        unique_times = np.unique(np.round(frame_times, 6))
        print(
            f"frame {frame_idx}: time min={frame_times.min():.6f}, "
            f"max={frame_times.max():.6f}, unique={unique_times[:8].tolist()}"
        )

    for key in ("timestamps", "delta_T", "delta_t", "current_delta_t", "current_timestamp", "valid_mask"):
        if key not in batch:
            print(f"{key}: <missing>")
            continue
        value = to_numpy(batch[key])
        print(f"{key} shape={value.shape}: {value[0] if value.ndim > 0 else value}")


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

        if args.require_full_history:
            valid_mask = to_numpy(candidate["valid_mask"])
            if valid_mask.ndim > 1 and valid_mask[0].sum() < int(cfg.hist_num):
                continue

        batch = candidate
        break

    if batch is None:
        raise RuntimeError("No batch matched the requested filters.")

    summarize_batch(batch, cfg)


if __name__ == "__main__":
    main()
