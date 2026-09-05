"""Build a fail-closed CT-SeqTrack B3 calibration artifact from JSONL."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_variant import configure_ct_variant
from utils.action_calibration import (
    CONSENSUS_FEATURE_SCHEMA,
    action_calibration_config_identity,
    calibrate_actions,
    sha256_file,
    sha256_json,
)
from utils.config import load_yaml_config


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    required_consensus = tuple(CONSENSUS_FEATURE_SCHEMA.split(","))
    missing = sorted({
        key for row in rows for key in required_consensus if key not in row})
    if missing:
        raise ValueError(
            "v26 calibration rows lack consensus features: "
            + ", ".join(missing))
    return rows


def validate_rows_manifest(
        manifest_path, rows_path, rows, partition,
        checkpoint_sha256, config_sha256):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "ct_seqtrack.action_rows.v2":
        raise ValueError("action-row manifest schema mismatch")
    expected = {
        "partition": str(partition),
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_sha256": str(config_sha256),
        "rows_sha256": sha256_file(rows_path),
        "tracklet_keys_sha256": sha256_json(sorted({
            str(row["tracklet_id"]) for row in rows})),
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"action-row manifest {key} mismatch")
    if int(manifest.get("rows", -1)) != len(rows):
        raise ValueError("action-row manifest row count mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--dev-rows", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--checkpoint-sha256")
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--config")
    config.add_argument("--config-sha256")
    manifest = parser.add_mutually_exclusive_group(required=True)
    manifest.add_argument("--tracklet-manifest")
    manifest.add_argument("--tracklet-manifest-sha256")
    dev_manifest = parser.add_mutually_exclusive_group(required=True)
    dev_manifest.add_argument("--dev-tracklet-manifest")
    dev_manifest.add_argument("--dev-tracklet-manifest-sha256")
    parser.add_argument("--code-version")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    rows = load_rows(args.rows)
    dev_rows = load_rows(args.dev_rows)
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
    if args.tracklet_manifest:
        validate_rows_manifest(
            args.tracklet_manifest, args.rows, rows, "calibration",
            checkpoint_sha256, config_sha256)
    if args.dev_tracklet_manifest:
        validate_rows_manifest(
            args.dev_tracklet_manifest, args.dev_rows, dev_rows, "dev",
            checkpoint_sha256, config_sha256)
    tracklet_manifest_sha256 = (
        sha256_file(args.tracklet_manifest) if args.tracklet_manifest
        else args.tracklet_manifest_sha256)
    dev_tracklet_manifest_sha256 = (
        sha256_file(args.dev_tracklet_manifest)
        if args.dev_tracklet_manifest
        else args.dev_tracklet_manifest_sha256)
    code_version = args.code_version
    if not code_version:
        code_version = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True).strip()
    artifact = calibrate_actions(
        rows, checkpoint_sha256, config_sha256,
        tracklet_manifest_sha256,
        dev_rows=dev_rows,
        dev_tracklet_manifest_sha256=dev_tracklet_manifest_sha256,
        code_version=code_version, seed=args.seed,
        resamples=args.bootstrap_resamples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not artifact["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    if "--v27" in sys.argv[1:]:
        from tools.ct_action_v27_runtime import calibrate_main
        calibrate_main()
    else:
        main()
