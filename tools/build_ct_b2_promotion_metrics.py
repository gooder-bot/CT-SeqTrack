#!/usr/bin/env python3
"""Build B2 promotion metrics from dev-candidate0 diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if "epoch" not in payload:
        raise ValueError("selected checkpoint lacks its completed epoch")
    return int(payload["epoch"]) + 1


def success_auc(overlaps):
    # Match utils.metrics.TorchSuccess exactly: repository evaluation uses
    # float32 overlap tensors and float32 threshold endpoints.
    overlaps = torch.as_tensor(overlaps, dtype=torch.float32)
    thresholds = torch.linspace(0.0, 1.0, steps=21, dtype=torch.float32)
    curve = torch.stack([
        (overlaps >= threshold).to(torch.float32).mean()
        for threshold in thresholds])
    return float(torch.trapz(curve, x=thresholds) * 100.0)


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError("candidate diagnostics contain no rows")
    return rows


def as_float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"candidate diagnostic lacks numeric {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"candidate diagnostic has non-finite {key}")
    return value


def build_metrics(
        rows, raw_success, observation_success, margin=0.05,
        action_epsilon=1e-8):
    if any("partition" not in row or "candidate_id" not in row
           or "tracklet_id" not in row or "frame_id" not in row
           for row in rows):
        raise ValueError(
            "promotion diagnostics require explicit partition, candidate_id, "
            "tracklet_id and frame_id fields")
    if any(str(row["partition"]) != "dev"
           or int(float(row["candidate_id"])) != 0 for row in rows):
        raise ValueError(
            "promotion diagnostic files may contain only dev candidate0")
    row_keys = [
        (int(float(row["tracklet_id"])), int(float(row["frame_id"])), 0)
        for row in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("promotion diagnostics contain duplicate rows")
    for row in rows:
        pool_count = as_float(row, "pool_target_count")
        sampled_count = as_float(row, "sampled_target_count")
        if pool_count < 0 or sampled_count < 0 or sampled_count > pool_count:
            raise ValueError("promotion diagnostics have invalid target counts")
        if as_float(row, "observation_error") < 0 or as_float(
                row, "raw_search_error") < 0:
            raise ValueError("promotion diagnostics have negative errors")
        if as_float(row, "search_valid") not in (0.0, 1.0):
            raise ValueError("search_valid must be binary")
        if as_float(row, "router_applied_gate") not in (0.0, 1.0):
            raise ValueError("router_applied_gate must be binary")
        for key in ("observation_iou", "raw_search_iou", "selective_iou"):
            if not 0.0 <= as_float(row, key) <= 1.0:
                raise ValueError(f"{key} must be in [0, 1]")
    primary = [row for row in rows
               if str(row["partition"]) == "dev"
               and int(float(row["candidate_id"])) == 0]
    if not primary:
        raise ValueError("promotion population has no dev candidate0 rows")
    eligible = [row for row in primary
                if as_float(row, "pool_target_count") > 0]
    if not eligible:
        raise RuntimeError("promotion has zero eligible acquisition rows")
    retained = [row for row in eligible
                if as_float(row, "sampled_target_count") > 0]
    pool_sum = sum(as_float(row, "pool_target_count") for row in eligible)
    sampled_sum = sum(
        as_float(row, "sampled_target_count") for row in eligible)
    raw_actions = [row for row in primary
                   if bool(int(as_float(row, "search_valid")))
                   and abs(as_float(row, "raw_search_error")
                           - as_float(row, "observation_error"))
                   > action_epsilon]
    center_gains = [
        as_float(row, "observation_error")
        - as_float(row, "raw_search_error") for row in raw_actions]
    iou_gains = [
        as_float(row, "raw_search_iou")
        - as_float(row, "observation_iou") for row in raw_actions]
    helpful = [gain for gain, iou_gain in zip(center_gains, iou_gains)
               if gain > margin and iou_gain >= 0.0]
    harmful = [gain for gain, iou_gain in zip(center_gains, iou_gains)
               if gain < -margin or iou_gain < 0.0]
    action_count = len(raw_actions)
    selective_actions = [
        row for row in primary
        if bool(int(as_float(row, "router_applied_gate")))]
    selective_helpful = [
        row for row in selective_actions
        if as_float(row, "observation_error")
        - as_float(row, "selective_error") > margin]
    selective_harmful = [
        row for row in selective_actions
        if as_float(row, "selective_error")
        - as_float(row, "observation_error") > margin]
    # Diagnostics begin at frame 1.  Tracking initializes frame 0 from GT,
    # so add one IoU=1 endpoint for every dev tracklet before reproducing the
    # repository's exact 21-threshold Success AUC.
    tracklet_count = len({
        int(float(row["tracklet_id"])) for row in primary})
    frame0 = [1.0] * tracklet_count
    computed_observation_success = success_auc(
        frame0 + [as_float(row, "observation_iou") for row in primary])
    computed_raw_success = success_auc(
        frame0 + [as_float(row, "raw_search_iou") for row in primary])
    computed_selective_success = success_auc(
        frame0 + [as_float(row, "selective_iou") for row in primary])
    if abs(float(raw_success) - computed_raw_success) > 1e-4:
        raise ValueError(
            "--raw-search-success does not match candidate diagnostics")
    if abs(float(observation_success) - computed_observation_success) > 1e-4:
        raise ValueError(
            "--observation-success does not match candidate diagnostics")
    return {
        "population": "dev_candidate0",
        "diagnostic_rows": len(primary),
        "diagnostic_tracklets": tracklet_count,
        "acquisition_eligible_rows": len(eligible),
        "acquisition_retained_rows": len(retained),
        "acquisition_row_recall": len(retained) / len(eligible),
        "acquisition_point_recall": (
            sampled_sum / pool_sum if pool_sum > 0 else None),
        "raw_action_count": action_count,
        "raw_action_rate": action_count / len(primary),
        "raw_helpful_precision": (
            len(helpful) / action_count if action_count else 0.0),
        "raw_harmful_rate": (
            len(harmful) / action_count if action_count else 0.0),
        "raw_center_gain": (
            sum(center_gains) / action_count if action_count else 0.0),
        "raw_iou_gain": (
            sum(iou_gains) / action_count if action_count else 0.0),
        "raw_oracle_center_headroom": (
            sum(max(value, 0.0) for value in center_gains) / action_count
            if action_count else 0.0),
        "raw_oracle_iou_headroom": (
            sum(max(value, 0.0) for value in iou_gains) / action_count
            if action_count else 0.0),
        "selective_action_count": len(selective_actions),
        "selective_action_rate": len(selective_actions) / len(primary),
        "selective_helpful_precision": (
            len(selective_helpful) / len(selective_actions)
            if selective_actions else 0.0),
        "selective_harmful_rate": (
            len(selective_harmful) / len(selective_actions)
            if selective_actions else 0.0),
        "selective_success": computed_selective_success,
        "selective_success_delta": (
            computed_selective_success - computed_observation_success),
        "raw_search_success": float(raw_success),
        "observation_success": float(observation_success),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-diagnostics", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="selected real-memory B2 checkpoint")
    parser.add_argument("--raw-search-success", type=float, required=True)
    parser.add_argument("--observation-success", type=float, required=True)
    parser.add_argument("--help-margin", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = load_rows(args.candidate_diagnostics)
    selected_epoch = checkpoint_epoch(args.checkpoint)
    if any("epoch" not in row for row in rows):
        raise ValueError(
            "promotion diagnostics require an explicit validation epoch")
    diagnostic_epochs = {int(float(row["epoch"])) for row in rows}
    if diagnostic_epochs != {selected_epoch}:
        raise ValueError(
            "candidate diagnostics do not match the selected checkpoint "
            f"epoch {selected_epoch}: observed {sorted(diagnostic_epochs)}")
    metrics = build_metrics(
        rows,
        args.raw_search_success, args.observation_success,
        margin=args.help_margin)
    metrics.update({
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_checkpoint_epoch": selected_epoch,
        "candidate_diagnostics_sha256": [
            sha256_file(path) for path in args.candidate_diagnostics],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
