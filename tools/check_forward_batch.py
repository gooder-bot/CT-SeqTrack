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
from models import get_model  # noqa: E402


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


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def is_paired_batch(batch):
    return isinstance(batch, dict) and "view_a" in batch and "view_b" in batch


def has_full_history(batch, hist_num):
    if is_paired_batch(batch):
        return has_full_history(batch["view_a"], hist_num) and has_full_history(batch["view_b"], hist_num)
    valid_mask = to_numpy(batch["valid_mask"])
    if valid_mask.ndim > 1:
        return bool(valid_mask[0].sum() >= int(hist_num))
    return bool(valid_mask.sum() >= int(hist_num))


def summarize_time_fields(batch):
    if is_paired_batch(batch):
        print("view_a:")
        summarize_time_fields(batch["view_a"])
        print("view_b:")
        summarize_time_fields(batch["view_b"])
        return
    for key in ("timestamps", "delta_T", "timestamps_real", "delta_T_real",
                "delta_t", "current_delta_t",
                "num_points_in_search", "valid_mask"):
        if key not in batch:
            print(f"{key}: <missing>")
            continue
        value = to_numpy(batch[key])
        print(f"{key} shape={value.shape}: {value[0] if value.ndim > 0 else value}")


def summarize_output(output):
    if is_paired_batch(output):
        print("output view_a:")
        summarize_output(output["view_a"])
        print("output view_b:")
        summarize_output(output["view_b"])
        return
    for key, value in output.items():
        if torch.is_tensor(value):
            finite = bool(torch.isfinite(value).all().item())
            print(f"{key}: shape={tuple(value.shape)}, finite={finite}")


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
    parser.add_argument("--twc", action="store_true",
                        help="Temporarily enable P4 paired-view TWC batch mode.")
    parser.add_argument("--obs-gate", action="store_true",
                        help="Temporarily enable P5 observability gate with dynamics branch.")
    parser.add_argument("--no-loss", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.pseudo_time:
        cfg.use_real_time = False
    if args.twc:
        cfg.use_twc = True
        cfg.twc_candidate_zero_only = True
    if args.obs_gate:
        cfg.use_dynamics_encoder = True
        cfg.use_observability_gate = True
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
            if not has_full_history(candidate, cfg.hist_num):
                continue
        batch = candidate
        print(f"using batch_idx={batch_idx}")
        break

    if batch is None:
        raise RuntimeError("No batch matched the requested filters.")

    summarize_time_fields(batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = get_model(cfg.net_model)(cfg).to(device)
    model.eval()
    batch = move_to_device(batch, device)

    with torch.no_grad():
        output = model(batch)
        summarize_output(output)
        if not args.no_loss:
            if device.type != "cuda":
                print("loss: skipped because compute_loss currently creates CUDA tensors.")
            else:
                loss_dict = model.compute_loss(batch, output)
                for key, value in loss_dict.items():
                    finite = bool(torch.isfinite(value).all().item())
                    print(f"{key}: {value.item():.6f}, finite={finite}")


if __name__ == "__main__":
    main()
