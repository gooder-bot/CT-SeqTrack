"""Build a fail-closed CT-SeqTrack B3 calibration artifact from JSONL."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_variant import configure_ct_variant
from utils.action_calibration import (
    action_calibration_config_identity,
    audit_action_thresholds,
    calibrate_actions,
    select_action_thresholds,
    sha256_file,
    sha256_json,
)
from utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", help="legacy v24 one-stage rows")
    parser.add_argument("--selection-rows")
    parser.add_argument("--audit-rows")
    parser.add_argument("--selection-artifact")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-sha256")
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--config")
    config.add_argument("--config-sha256")
    manifest = parser.add_mutually_exclusive_group()
    manifest.add_argument("--tracklet-manifest")
    manifest.add_argument("--tracklet-manifest-sha256")
    parser.add_argument("--selection-scene-manifest")
    parser.add_argument("--selection-scene-manifest-sha256")
    parser.add_argument("--audit-scene-manifest")
    parser.add_argument("--audit-scene-manifest-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--min-scenes", type=int, default=10)
    parser.add_argument("--min-actions", type=int, default=100)
    parser.add_argument("--min-coverage", type=float, default=0.01)
    parser.add_argument("--max-harmful-upper", type=float, default=0.05)
    args = parser.parse_args()

    modes = sum(bool(value) for value in (
        args.rows, args.selection_rows, args.audit_rows))
    if modes != 1:
        parser.error(
            "choose exactly one of --rows, --selection-rows, --audit-rows")

    def load_rows(path):
        row_path = Path(path)
        with row_path.open("r", encoding="utf-8", newline="") as handle:
            if row_path.suffix.lower() == ".csv":
                return list(csv.DictReader(handle))
            return [json.loads(line) for line in handle if line.strip()]

    def digest(path, supplied):
        if bool(path) == bool(supplied):
            parser.error("provide exactly one manifest path or SHA256")
        return sha256_file(path) if path else supplied

    def scene_digest(path, supplied, partition):
        if bool(path) == bool(supplied):
            parser.error("provide exactly one scene manifest path or SHA256")
        if not path:
            return supplied
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "ct_seqtrack.scene_partition_manifest.v1":
            parser.error("scene manifest has an unsupported schema")
        observed = payload.get("content_sha256")
        content = dict(payload)
        content.pop("content_sha256", None)
        if observed != sha256_json(content):
            parser.error("scene manifest content SHA256 mismatch")
        partition_row = payload.get("partitions", {}).get(partition)
        if (not isinstance(partition_row, dict)
                or not partition_row.get("content_sha256")):
            parser.error(f"scene manifest lacks partition {partition}")
        return str(partition_row["content_sha256"])

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
    common = {
        "seed": args.seed,
        "resamples": args.bootstrap_resamples,
    }
    if args.rows:
        tracklet_manifest_sha256 = digest(
            args.tracklet_manifest, args.tracklet_manifest_sha256)
        artifact = calibrate_actions(
            load_rows(args.rows), checkpoint_sha256, config_sha256,
            tracklet_manifest_sha256, **common)
    elif args.selection_rows:
        selection_manifest_sha256 = scene_digest(
            args.selection_scene_manifest,
            args.selection_scene_manifest_sha256,
            "calibration_select")
        artifact = select_action_thresholds(
            load_rows(args.selection_rows), checkpoint_sha256,
            config_sha256, selection_manifest_sha256,
            min_scenes=args.min_scenes, min_actions=args.min_actions,
            min_coverage=args.min_coverage,
            max_harmful_upper=args.max_harmful_upper, **common)
    else:
        if not args.selection_artifact:
            parser.error("--audit-rows requires --selection-artifact")
        selection_manifest_sha256 = scene_digest(
            args.selection_scene_manifest,
            args.selection_scene_manifest_sha256,
            "calibration_select")
        audit_manifest_sha256 = scene_digest(
            args.audit_scene_manifest, args.audit_scene_manifest_sha256,
            "calibration_audit")
        with Path(args.selection_artifact).open(
                "r", encoding="utf-8") as handle:
            selection_artifact = json.load(handle)
        artifact = audit_action_thresholds(
            load_rows(args.audit_rows), selection_artifact,
            checkpoint_sha256, config_sha256,
            selection_manifest_sha256, audit_manifest_sha256,
            min_scenes=args.min_scenes, min_actions=args.min_actions,
            min_coverage=args.min_coverage,
            max_harmful_upper=args.max_harmful_upper, **common)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not artifact["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
