import argparse
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
    parser.add_argument("--mode", default=None)
    parser.add_argument("--gap-pattern", nargs="*", type=int, default=None)
    parser.add_argument("--drop-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-gap", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.mode is not None:
        cfg.virtual_rate_mode = args.mode
    if args.gap_pattern:
        cfg.virtual_rate_gap_pattern = args.gap_pattern
    if args.drop_prob is not None:
        cfg.virtual_rate_drop_prob = args.drop_prob
    if args.seed is not None:
        cfg.virtual_rate_seed = args.seed
    if args.max_gap is not None:
        cfg.virtual_rate_max_gap = args.max_gap
    cfg.virtual_rate_manifest = args.output
    cfg.preloading = False

    split = args.split if args.split is not None else cfg.test_split
    wrapped = get_dataset(cfg, type="test", split=split)
    dataset = getattr(wrapped, "dataset", wrapped)
    print(f"saved manifest: {args.output}")
    print("summary:", getattr(dataset, "virtual_rate_summary", {}))


if __name__ == "__main__":
    main()
