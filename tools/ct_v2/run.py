#!/usr/bin/env python3
"""Single entry point for the active CT-SeqTrack v2 experiment matrix."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "baseline": ROOT / "cfgs/ct_v2/01_seqtrack3d_baseline.yaml",
    "search_only": ROOT / "cfgs/ct_v2/05_seqtrack3d_search_only.yaml",
    "baseline_full": ROOT / "cfgs/ct_v2/01_seqtrack3d_baseline_full.yaml",
    "motion": ROOT / "cfgs/ct_v2/02_ct_motion.yaml",
    "motion_search": ROOT / "cfgs/ct_v2/03_ct_motion_search.yaml",
    "full": ROOT / "cfgs/ct_v2/04_ct_seqtrack_v2.yaml",
    "full_dataset": ROOT / "cfgs/ct_v2/04_ct_seqtrack_v2_full.yaml",
    "pftc_unweighted": (
        ROOT / "cfgs/ct_v2/06_seqtrack3d_pftc_unweighted.yaml"),
    "pftc": ROOT / "cfgs/ct_v2/07_seqtrack3d_dt_pftc.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--variant", choices=CONFIGS, default="full")
    parser.add_argument("--checkpoint")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--path", help="override the nuScenes dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--protocol", choices=("normal", "random20", "gap1124"),
        default="normal")
    parser.add_argument(
        "--time-mode", choices=("true", "fixed", "shuffled"), default="true")
    parser.add_argument(
        "--time-manifest",
        help="required for shuffled time; ignored by true/fixed")
    parser.add_argument(
        "--pftc-weight", type=float,
        help="override the frozen PFTC lambda")
    parser.add_argument(
        "--preflight", action="store_true",
        help="run 200 training batches with PFTC total weight forced to zero")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args):
    if args.mode == "test" and not args.checkpoint:
        raise ValueError("--checkpoint is required in test mode")
    if args.mode == "train" and args.protocol != "normal":
        raise ValueError("v2 training is fixed to the normal dataset protocol")
    if args.preflight and (
            args.mode != "train"
            or args.variant not in ("pftc_unweighted", "pftc")):
        raise ValueError(
            "--preflight requires train mode and a PFTC variant")
    if (args.mode == "train"
            and args.variant in ("pftc_unweighted", "pftc")
            and not args.preflight
            and args.pftc_weight is None):
        raise ValueError(
            "formal PFTC training requires the preflight-frozen "
            "--pftc-weight")
    if args.mode == "train" and args.checkpoint:
        raise ValueError(
            "use --init-checkpoint for model initialization; --checkpoint is "
            "reserved for test/resume semantics")
    if args.time_mode == "shuffled" and not args.time_manifest:
        raise ValueError("--time-manifest is required for shuffled time")

    default_tag = f"{args.variant}-{args.protocol}-{args.time_mode}"
    if args.preflight:
        default_tag += "-pftc-preflight-200"
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--cfg",
        str(CONFIGS[args.variant]),
        "--seed",
        str(args.seed),
        "--tag",
        args.tag or default_tag,
        "--dynamics_time_mode",
        args.time_mode,
    ]
    if args.path:
        command.extend(("--path", args.path))
    if args.pftc_weight is not None and not args.preflight:
        command.extend(("--pftc_weight", str(args.pftc_weight)))
    if args.preflight:
        command.extend((
            "--pftc_weight", "0.0",
            "--epoch", "1",
            "--limit_train_batches", "200",
            "--check_val_every_n_epoch", "2",
        ))
    if args.mode == "test":
        command.extend(("--test", "--checkpoint", args.checkpoint))
    elif args.init_checkpoint:
        command.extend(("--init_checkpoint", args.init_checkpoint))
    if args.time_manifest:
        command.extend(("--dynamics_time_manifest", args.time_manifest))
    if args.protocol == "random20":
        command.extend((
            "--test_virtual_rate_mode", "random_drop",
            "--test_virtual_rate_drop_prob", "0.2",
            "--test_virtual_rate_seed", str(args.seed),
            "--test_virtual_rate_max_gap", "5",
        ))
    elif args.protocol == "gap1124":
        command.extend((
            "--test_virtual_rate_mode", "gap_pattern",
        ))
    return command


def main():
    args = parse_args()
    command = build_command(args)
    print(subprocess.list2cmdline(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
