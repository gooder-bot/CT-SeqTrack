"""Build a fail-closed CT-SeqTrack B3 calibration artifact from JSONL."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_variant import configure_ct_variant
from utils.action_calibration import (
    action_calibration_config_identity,
    calibrate_actions,
    sha256_file,
    sha256_json,
)
from utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-sha256")
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--config")
    config.add_argument("--config-sha256")
    manifest = parser.add_mutually_exclusive_group(required=True)
    manifest.add_argument("--tracklet-manifest")
    manifest.add_argument("--tracklet-manifest-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    with Path(args.rows).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    checkpoint_sha256 = (
        sha256_file(args.checkpoint) if args.checkpoint
        else args.checkpoint_sha256)
    if args.config:
        resolved_config = load_yaml_config(args.config)
        configure_ct_variant(resolved_config)
        config_sha256 = sha256_json(
            action_calibration_config_identity(resolved_config))
    else:
        config_sha256 = args.config_sha256
    tracklet_manifest_sha256 = (
        sha256_file(args.tracklet_manifest) if args.tracklet_manifest
        else args.tracklet_manifest_sha256)
    artifact = calibrate_actions(
        rows, checkpoint_sha256, config_sha256,
        tracklet_manifest_sha256, seed=args.seed,
        resamples=args.bootstrap_resamples)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not artifact["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
