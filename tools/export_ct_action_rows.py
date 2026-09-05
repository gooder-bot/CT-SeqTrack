#!/usr/bin/env python3
"""Export held-out v26 B3 action rows from one scratch checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_variant import configure_ct_variant  # noqa: E402
from utils.action_calibration import (  # noqa: E402
    CONSENSUS_FEATURE_SCHEMA,
    action_calibration_config_identity,
    sha256_file,
    sha256_json,
)
from utils.checkpoint_loading import load_initial_weights  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402
from utils.recursive_state import stable_tracklet_partition  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train_track")
    parser.add_argument("--partition", choices=("calibration", "dev"),
                        required=True)
    parser.add_argument("--path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preloading", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return device


def main():
    from easydict import EasyDict
    from datasets import get_dataset
    from models import get_model
    args = parse_args()
    raw_config = load_yaml_config(args.config)
    seed = int(args.seed if args.seed is not None
               else raw_config.get("seed", 42) or 42)
    raw_config.update({
        "seed": seed,
        "test_split": args.split,
        "preloading": bool(args.preloading),
        "proposal_inference_mode": "observation",
        "export_proposal_diagnostics": True,
        "export_v3_candidate_diagnostics": True,
        "ct_action_calibration_path": None,
    })
    if args.path:
        raw_config["path"] = args.path
    configure_ct_variant(raw_config)
    if raw_config.get("ct_variant") != "full":
        raise ValueError("B3 action export requires the v26 Full config")
    if not raw_config.get("ct_enable_v26_recovery", False):
        raise ValueError("B3 action export requires v26 recovery")
    config = EasyDict(raw_config)
    dataset = get_dataset(
        config, type="test", split=args.split, protocol_role="test")
    source_dataset = getattr(dataset, "dataset", dataset)
    model = get_model(config.net_model)(config)
    load_initial_weights(model, args.checkpoint, require_complete=True)
    model.to(resolve_device(args.device)).eval()

    rows = []
    selected_keys = []
    limit = len(dataset)
    if args.max_tracklets is not None:
        limit = min(limit, int(args.max_tracklets))
    with torch.inference_mode():
        for tracklet_index in range(limit):
            tracklet_key = (
                source_dataset.get_tracklet_key(tracklet_index)
                if hasattr(source_dataset, "get_tracklet_key")
                else f"{args.split}/tracklet/{tracklet_index}")
            if stable_tracklet_partition(
                    tracklet_key, seed) != args.partition:
                continue
            selected_keys.append(str(tracklet_key))
            sequence = dataset[tracklet_index]
            model.evaluate_one_sequence(sequence)
            for row in getattr(
                    model, "_proposal_sequence_diagnostics", ()):
                item = dict(row)
                if int(float(item.get("acquisition_schema_version", -1))) != 3:
                    raise RuntimeError(
                        "action export received a non-v26 diagnostic row")
                item.update({
                    "tracklet_id": str(tracklet_key),
                    "tracklet_index": int(tracklet_index),
                    "partition": args.partition,
                })
                rows.append(item)
    if not rows:
        raise RuntimeError("v26 action export produced no rows")
    consensus_fields = tuple(CONSENSUS_FEATURE_SCHEMA.split(","))
    missing = sorted({field for row in rows for field in consensus_fields
                      if field not in row})
    if missing:
        raise RuntimeError(
            "v26 action rows lack consensus fields: " + ", ".join(missing))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "ct_seqtrack.action_rows.v2",
        "dataset": str(getattr(config, "dataset", "unknown")),
        "split": args.split,
        "partition": args.partition,
        "seed": seed,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_json(
            action_calibration_config_identity(config)),
        "consensus_feature_schema": CONSENSUS_FEATURE_SCHEMA,
        "tracklets": len(selected_keys),
        "tracklet_keys_sha256": sha256_json(sorted(selected_keys)),
        "rows": len(rows),
        "rows_sha256": sha256_file(output),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    if "--v27" in sys.argv[1:]:
        from tools.ct_action_v27_runtime import export_main
        export_main()
    else:
        main()
