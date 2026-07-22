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
    with open(path, "r") as handle:
        cfg = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def merge_protocol_config(cfg, path):
    if path is None:
        return
    with open(path, "r", encoding="utf-8") as handle:
        protocol_cfg = yaml.load(handle, Loader=yaml.FullLoader)
    for key, value in protocol_cfg.items():
        if "virtual_rate" in key:
            setattr(cfg, key, value)


def main():
    parser = argparse.ArgumentParser(
        description="Build an offline split-wide shuffled-dt permutation manifest.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--protocol-cfg", default=None,
        help="Merge only virtual-rate fields from this frozen protocol config.")
    parser.add_argument(
        "--virtual-rate-manifest", default=None,
        help="Frozen cadence manifest whose endpoint selection the mapping must use.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(ROOT), text=True).strip()
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "Tracked files are dirty. Commit/push first, or use --allow-dirty only "
            "for a non-formal smoke manifest.")

    cfg = load_config(args.cfg)
    merge_protocol_config(cfg, args.protocol_cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    # The physical endpoint set is loaded first. The generated file is then
    # consumed by a separate shuffled-mode run.
    setattr(cfg, f"{args.role}_dynamics_time_mode", "true")
    setattr(cfg, f"dynamics_time_manifest_{args.role}", "")
    if args.virtual_rate_manifest is not None:
        setattr(cfg, f"{args.role}_virtual_rate_manifest", args.virtual_rate_manifest)
        setattr(cfg, f"virtual_rate_manifest_{args.role}", args.virtual_rate_manifest)
        setattr(cfg, f"{args.role}_virtual_rate_manifest_strict", True)
        setattr(cfg, f"{args.role}_virtual_rate_manifest_require_commit_match", True)
    split = args.split or getattr(cfg, f"{args.role}_split", cfg.test_split)
    wrapped = get_dataset(
        cfg,
        type="train_motion_mf" if args.role == "train" else "test",
        split=split,
        protocol_role=args.role,
    )
    dataset = getattr(wrapped, "dataset", wrapped)
    result = dataset.build_dynamics_time_manifest(args.output, seed=args.seed)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
