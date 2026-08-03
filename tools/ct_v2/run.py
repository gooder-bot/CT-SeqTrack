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
    "motion_v3": ROOT / "cfgs/ct_v2/02_ct_motion_v3.yaml",
    "search_v21": ROOT / "cfgs/ct_v2/08_seqtrack3d_search_v21.yaml",
    "motion_search_v21": (
        ROOT / "cfgs/ct_v2/09_ct_motion_search_v21.yaml"),
    "b3_crpa_v1": ROOT / "cfgs/ct_v2/10_b3_crpa_v1.yaml",
    "b2_v22_refiner": ROOT / "cfgs/ct_v2/11_b2_v22_refiner.yaml",
    "b2_v22_selective": ROOT / "cfgs/ct_v2/12_b2_v22_selective.yaml",
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
    parser.add_argument(
        "--resume-checkpoint",
        help="training checkpoint that restores model, optimizer, and epoch state")
    parser.add_argument("--path", help="override the nuScenes dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--check-val-every-n-epoch", type=int)
    parser.add_argument("--save-top-k", type=int)
    parser.add_argument("--limit-train-batches", type=float)
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
        "--fusion-off", action="store_true",
        help="evaluate a motion_v3 checkpoint with exact observation-only output")
    parser.add_argument(
        "--proposal-mode",
        choices=(
            "obs", "obs_motion", "obs_search", "full",
            "obs_motion_search", "full_selective"),
        help="evaluate a B2 checkpoint under a same-weight proposal mode")
    parser.add_argument(
        "--preflight", action="store_true",
        help="run 200 training batches with PFTC total weight forced to zero")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args):
    if args.mode == "test" and not args.checkpoint:
        raise ValueError("--checkpoint is required in test mode")
    if args.resume_checkpoint and args.mode != "train":
        raise ValueError("--resume-checkpoint is training-only")
    if args.resume_checkpoint and (
            args.checkpoint or args.init_checkpoint):
        raise ValueError(
            "--resume-checkpoint cannot be combined with checkpoint initialization")
    if (args.mode == "train" and args.variant == "b2_v22_refiner"
            and not (args.init_checkpoint or args.resume_checkpoint)):
        raise ValueError(
            "B2-v2.2 refiner training requires the composed "
            "--init-checkpoint (or an explicit --resume-checkpoint)")
    if args.mode == "train" and args.variant == "b2_v22_selective":
        raise ValueError(
            "the signed router is trained offline; b2_v22_selective is "
            "evaluation-only")
    if args.mode == "train" and args.protocol != "normal":
        raise ValueError("v2 training is fixed to the normal dataset protocol")
    if args.preflight and (
            args.mode != "train"
            or args.variant not in ("pftc_unweighted", "pftc")):
        raise ValueError(
            "--preflight requires train mode and a PFTC variant")
    if args.fusion_off and (
            args.mode != "test" or args.variant != "motion_v3"):
        raise ValueError(
            "--fusion-off requires test mode and --variant motion_v3")
    if args.proposal_mode and (
            args.mode != "test"
            or args.variant not in (
                "search_v21", "motion_search_v21", "b3_crpa_v1",
                "b2_v22_refiner", "b2_v22_selective")):
        raise ValueError(
            "--proposal-mode requires test mode and a supported B2 variant")
    if (args.proposal_mode == "obs_search"
            and args.variant in ("b2_v22_refiner", "b2_v22_selective")):
        raise ValueError("B2-v2.2 has no independent obs_search mode")
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
    if args.fusion_off:
        default_tag += "-fusion-off"
    if args.proposal_mode:
        default_tag += f"-{args.proposal_mode}"
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
    if args.epochs is not None:
        if args.mode != "train" or args.epochs <= 0:
            raise ValueError("--epochs must be positive and is training-only")
        command.extend(("--epoch", str(args.epochs)))
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("--workers must be non-negative")
        command.extend(("--workers", str(args.workers)))
    if args.batch_size is not None:
        if args.mode != "train" or args.batch_size <= 0:
            raise ValueError(
                "--batch-size must be positive and is training-only")
        command.extend(("--batch_size", str(args.batch_size)))
    if args.preloading:
        command.append("--preloading")
    if args.check_val_every_n_epoch is not None:
        if args.mode != "train" or args.check_val_every_n_epoch <= 0:
            raise ValueError(
                "--check-val-every-n-epoch must be positive and training-only")
        command.extend((
            "--check_val_every_n_epoch",
            str(args.check_val_every_n_epoch),
        ))
    if args.save_top_k is not None:
        if args.mode != "train" or args.save_top_k < 0:
            raise ValueError(
                "--save-top-k must be non-negative and training-only")
        command.extend(("--save_top_k", str(args.save_top_k)))
    if args.limit_train_batches is not None:
        if args.mode != "train" or args.limit_train_batches <= 0:
            raise ValueError(
                "--limit-train-batches must be positive and training-only")
        command.extend((
            "--limit_train_batches", str(args.limit_train_batches)))
    if args.pftc_weight is not None and not args.preflight:
        command.extend(("--pftc_weight", str(args.pftc_weight)))
    if args.fusion_off:
        command.extend(("--motion_v3_fusion_scale", "0.0"))
    if args.proposal_mode:
        command.extend(("--proposal-mode", args.proposal_mode))
    if args.preflight:
        command.extend((
            "--pftc_weight", "0.0",
            "--epoch", "1",
            "--limit_train_batches", "200",
            "--check_val_every_n_epoch", "2",
        ))
    if args.mode == "test":
        command.extend(("--test", "--checkpoint", args.checkpoint))
    elif args.resume_checkpoint:
        command.extend(("--checkpoint", args.resume_checkpoint))
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
