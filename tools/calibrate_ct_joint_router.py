"""Calibrate and persist the Joint Full router threshold.

The input is an ``.npz`` exported from the selected joint checkpoint with
arrays named ``router_probability``, ``h3_gain``, ``evidence_valid`` and
``tracklet_key``.  Every row must belong to the held-out calibration
tracklet partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ct_v2 import calibrate_joint_router_threshold
from tools.b3_crpa_common import sha256_file, torch_load
from utils.recursive_state import stable_tracklet_partition


def load_calibration_records(path, seed):
    with np.load(path, allow_pickle=False) as records:
        required = {
            "router_probability", "h3_gain", "evidence_valid",
            "tracklet_key",
        }
        missing = sorted(required.difference(records.files))
        if missing:
            raise KeyError(
                "calibration records are missing: " + ", ".join(missing))
        probabilities = np.asarray(
            records["router_probability"], dtype=np.float64).reshape(-1)
        gains = np.asarray(records["h3_gain"], dtype=np.float64).reshape(-1)
        valid = np.asarray(records["evidence_valid"], dtype=bool).reshape(-1)
        tracklet_keys = np.asarray(records["tracklet_key"]).astype(str).reshape(-1)
    if not (probabilities.shape == gains.shape == valid.shape
            == tracklet_keys.shape):
        raise ValueError("all calibration record arrays must have equal length")
    leaked = sorted({
        key for key in tracklet_keys
        if stable_tracklet_partition(key, seed) != "calibration"
    })
    if leaked:
        preview = ", ".join(leaked[:5])
        raise ValueError(
            "router calibration contains non-calibration tracklets: "
            + preview)
    return probabilities, gains, valid, tracklet_keys


def write_calibrated_checkpoint(source, destination, threshold):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("calibration output must not overwrite its source")
    if destination.exists():
        raise FileExistsError(f"calibration output already exists: {destination}")
    checkpoint = torch_load(source, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    matching = [
        key for key in state_dict
        if key.endswith("ct_joint_router.decision_threshold")]
    if len(matching) != 1:
        raise KeyError(
            "expected exactly one joint router threshold in checkpoint, "
            f"found {matching}")
    key = matching[0]
    state_dict[key] = state_dict[key].new_tensor(float(threshold))
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = (
        args.report.resolve() if args.report is not None
        else args.output.with_suffix(args.output.suffix + ".calibration.json"))
    if report_path.exists():
        raise FileExistsError(f"calibration report already exists: {report_path}")
    probabilities, gains, valid, tracklet_keys = load_calibration_records(
        args.records, args.seed)
    result = calibrate_joint_router_threshold(
        probabilities, gains, valid,
        minimum_threshold=0.5,
        min_precision=0.75,
        max_harm_rate=0.05,
        min_coverage=0.05,
        max_coverage=0.25,
        helpful_margin=0.15,
    )
    state_key = write_calibrated_checkpoint(
        args.checkpoint, args.output, result["threshold"])
    report = {
        **result,
        "seed": int(args.seed),
        "row_count": int(len(probabilities)),
        "valid_row_count": int(valid.sum()),
        "tracklet_count": int(len(set(tracklet_keys.tolist()))),
        "state_dict_key": state_key,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "records": str(args.records.resolve()),
        "records_sha256": sha256_file(args.records),
        "output_checkpoint": str(args.output.resolve()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
