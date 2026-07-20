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


def main():
    parser = argparse.ArgumentParser(
        description="Build an offline split-wide shuffled-dt permutation manifest.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
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
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    # The physical endpoint set is loaded first. The generated file is then
    # consumed by a separate shuffled-mode run.
    setattr(cfg, f"{args.role}_dynamics_time_mode", "true")
    setattr(cfg, f"dynamics_time_manifest_{args.role}", "")
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
