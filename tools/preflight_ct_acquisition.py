#!/usr/bin/env python3
"""Build an independent, checkpoint-free acquisition preflight artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctseqtrack.runtime.acquisition import (
    acquisition_config_identity,
    build_preflight_artifact,
    sha256_json,
    validate_preflight_artifact,
)
from utils.config import load_yaml_config
from ctseqtrack.config import configure_ct_variant
from ctseqtrack.runtime.contracts import validate_scratch_training_contract


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list) or not payload:
        raise ValueError("preflight input must contain a non-empty row list")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", required=True, help="JSON/JSONL rows from the fixed-CV sampler pass"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--path")
    parser.add_argument("--dynamics-time-manifest")
    reseed = parser.add_mutually_exclusive_group()
    reseed.add_argument(
        "--ct-reseed-enabled",
        dest="ct_recursive_reseed_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    reseed.add_argument(
        "--ct-no-reseed",
        dest="ct_recursive_reseed_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    rows_path = Path(args.rows).resolve()
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.data_manifest).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    config = load_yaml_config(config_path)
    configure_ct_variant(config)
    if args.path is not None:
        config["path"] = args.path
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.dynamics_time_manifest is not None:
        config["dynamics_time_manifest"] = args.dynamics_time_manifest
    if hasattr(args, "ct_recursive_reseed_enabled"):
        config["ct_recursive_reseed_enabled"] = bool(args.ct_recursive_reseed_enabled)
    validate_scratch_training_contract(config)
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_schema = (
        "ct_seqtrack.acquisition_data_manifest.v3"
        if int(config.get("ct_protocol_version", 24)) >= 25
        else "ct_seqtrack.acquisition_data_manifest.v2"
    )
    if (
        not isinstance(data_manifest, dict)
        or data_manifest.get("schema") != expected_manifest_schema
        or data_manifest.get("checkpoint_loaded") is not False
        or data_manifest.get("complete") is not True
        or int(data_manifest.get("dropped_rows", 0)) != 0
    ):
        raise ValueError(
            "--data-manifest must be a complete checkpoint-free acquisition "
            "manifest; --max-batches output is diagnostic-only"
        )
    seed = int(args.seed if args.seed is not None else config.get("seed", 42) or 42)
    artifact = build_preflight_artifact(
        load_rows(rows_path),
        config_identity={
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "resolved_sha256": sha256_json(config),
            "acquisition": acquisition_config_identity(config),
        },
        data_manifest_identity={
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "manifest_sha256": sha256_json(data_manifest),
            "manifest": data_manifest,
        },
        seed=seed,
    )
    artifact["source_rows"] = {
        "path": str(rows_path),
        "sha256": sha256_file(rows_path),
    }
    validate_preflight_artifact(artifact, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, sort_keys=True))
    if not artifact["passed"]:
        failed = sorted(key for key, value in artifact["criteria"].items() if not value)
        raise SystemExit("acquisition preflight failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
