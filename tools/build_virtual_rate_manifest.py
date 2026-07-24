import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from easydict import EasyDict


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--role", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--kitti-hv-interval",
        default=None,
        help="Official KITTI-HV interval (1/2/3/5/10 or 'all').")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--gap-pattern", nargs="*", type=int, default=None)
    parser.add_argument(
        "--stride", type=int, default=None,
        help=(
            "Single-path stride for generic CT cadence tests. Official "
            "KITTI-HV should use --kitti-hv-interval instead."))
    parser.add_argument("--drop-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-gap", type=int, default=None)
    parser.add_argument("--min-tracklet-len", type=int, default=None)
    keep_last_group = parser.add_mutually_exclusive_group()
    keep_last_group.add_argument(
        "--keep-last", dest="keep_last", action="store_true",
        help="Force the final frame into every tracklet.")
    keep_last_group.add_argument(
        "--no-keep-last", dest="keep_last", action="store_false",
        help="Preserve exact stride cadence without an irregular final gap.")
    parser.set_defaults(keep_last=None)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow tracked source changes. Formal frozen manifests should omit this.")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.kitti_hv_interval is not None:
        setattr(
            cfg,
            f"{args.role}_kitti_hv_interval",
            args.kitti_hv_interval,
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(ROOT), text=True).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "Tracked files are dirty. Commit/push the protocol code before building "
            "a formal manifest, or use --allow-dirty for a non-formal smoke test.")

    role_prefix = args.role
    overrides = {
        "virtual_rate_mode": args.mode,
        "virtual_rate_gap_pattern": args.gap_pattern,
        "virtual_rate_stride": args.stride,
        "virtual_rate_drop_prob": args.drop_prob,
        "virtual_rate_seed": args.seed,
        "virtual_rate_max_gap": args.max_gap,
        "virtual_rate_min_tracklet_len": args.min_tracklet_len,
        "virtual_rate_keep_last": args.keep_last,
    }
    for key, value in overrides.items():
        if value is not None and value != []:
            setattr(cfg, f"{role_prefix}_{key}", value)
    setattr(cfg, f"virtual_rate_manifest_{role_prefix}", args.output)
    setattr(cfg, f"{role_prefix}_virtual_rate_manifest_allow_create", True)
    cfg.preloading = False

    split = args.split if args.split is not None else cfg.test_split
    wrapped = get_dataset(
        cfg, type="train_motion_mf" if args.role == "train" else "test",
        split=split, protocol_role=args.role)
    dataset = getattr(wrapped, "dataset", wrapped)
    print(f"saved manifest: {args.output}")
    print("summary:", getattr(dataset, "virtual_rate_summary", {}))
    print("content_sha256:", dataset.virtual_rate_manifest_content_sha256)
    print("file_sha256:", dataset.virtual_rate_manifest_file_sha256)


if __name__ == "__main__":
    main()
