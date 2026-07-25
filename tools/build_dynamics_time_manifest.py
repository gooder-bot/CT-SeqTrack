import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402


def load_config(path):
    cfg = EasyDict(load_yaml_config(path))
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


def apply_frozen_virtual_rate_protocol(cfg, path, role):
    """Replay every cadence field recorded by a schema-v2 manifest."""
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "ct_seqtrack.virtual_rate_manifest":
        raise ValueError(
            "Unsupported virtual-rate manifest; expected schema-v2 CT manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Virtual-rate manifest does not contain protocol fields")
    if "kitti_hv_intervals" in manifest:
        intervals = list(manifest["kitti_hv_intervals"])
        setattr(
            cfg,
            f"{role}_kitti_hv_interval",
            intervals[0] if len(intervals) == 1 else intervals,
        )
    key_map = {
        "gap_pattern": "virtual_rate_gap_pattern",
        "stride": "virtual_rate_stride",
        "drop_every": "virtual_rate_drop_every",
        "drop_prob": "virtual_rate_drop_prob",
        "seed": "virtual_rate_seed",
        "max_gap": "virtual_rate_max_gap",
        "keep_first": "virtual_rate_keep_first",
        "keep_last": "virtual_rate_keep_last",
        "min_tracklet_len": "virtual_rate_min_tracklet_len",
        "burst_keep_lengths": "virtual_rate_burst_keep_lengths",
        "burst_skip_lengths": "virtual_rate_burst_skip_lengths",
    }
    setattr(cfg, f"{role}_virtual_rate_mode", "manifest")
    for manifest_key, config_key in key_map.items():
        if manifest_key in protocol:
            setattr(cfg, f"{role}_{config_key}", protocol[manifest_key])


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
        "--kitti-hv-interval",
        default=None,
        help="Official KITTI-HV interval when no cadence manifest supplies it.")
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
    if args.kitti_hv_interval is not None:
        setattr(
            cfg,
            f"{args.role}_kitti_hv_interval",
            args.kitti_hv_interval,
        )
    # The physical endpoint set is loaded first. The generated file is then
    # consumed by a separate shuffled-mode run.
    setattr(cfg, f"{args.role}_dynamics_time_mode", "true")
    setattr(cfg, f"dynamics_time_manifest_{args.role}", "")
    if args.virtual_rate_manifest is not None:
        apply_frozen_virtual_rate_protocol(
            cfg, args.virtual_rate_manifest, args.role)
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
