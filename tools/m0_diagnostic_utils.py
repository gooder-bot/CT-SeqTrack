"""Shared, read-only helpers for the M0 proposal diagnostics.

This module deliberately avoids importing the project at module import time so
that the synthetic self-tests can run on a lightweight analysis machine.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(root=ROOT):
    def run(*args):
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def require_clean_git(allow_dirty=False, root=ROOT):
    provenance = git_provenance(root)
    if provenance["dirty"] and not allow_dirty:
        raise RuntimeError(
            "M0 frozen diagnostics require a clean worktree. Commit/stash the "
            "listed changes or pass --allow-dirty only for a smoke test:\n"
            + "\n".join(provenance["status_porcelain"][:20])
        )
    return provenance


def finite_array(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def distribution(values):
    array = finite_array(values)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(quantiles[5]),
    }


def wrap_angle(values):
    values = np.asarray(values, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def masked_row_mean(values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for index in range(values.shape[0]):
        selected = values[index][mask[index] & np.isfinite(values[index])]
        if selected.size:
            result[index] = float(np.mean(selected))
    return result


def masked_row_max(values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for index in range(values.shape[0]):
        selected = values[index][mask[index] & np.isfinite(values[index])]
        if selected.size:
            result[index] = float(np.max(selected))
    return result


def candidate_kinematics(ref_boxes, canonical_boxes, delta_t, valid_mask, eps=1e-3):
    """Measure derivative corruption caused by candidate box perturbations.

    Histories are ordered recent-to-old. ``delta_t[:, 1:H]`` is the physical
    gap for each adjacent history transition, matching DynamicsEncoder.
    """
    ref_boxes = np.asarray(ref_boxes, dtype=np.float64)
    canonical_boxes = np.asarray(canonical_boxes, dtype=np.float64)
    delta_t = np.asarray(delta_t, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if ref_boxes.shape != canonical_boxes.shape or ref_boxes.ndim != 3:
        raise ValueError("ref_boxes/canonical_boxes must share shape [B,H,4].")
    if ref_boxes.shape[2] != 4:
        raise ValueError("box tensors must have four components [x,y,z,yaw].")
    batch_size, hist_num, _ = ref_boxes.shape
    if delta_t.shape != (batch_size, hist_num):
        raise ValueError("delta_t must have shape [B,H].")
    if valid_mask.shape != (batch_size, hist_num):
        raise ValueError("valid_mask must have shape [B,H].")

    if hist_num < 2:
        empty = np.empty((batch_size, 0), dtype=np.float64)
        return {
            "velocity_jitter": empty,
            "yaw_rate_jitter": empty,
            "acceleration_jitter": empty,
            "transition_valid": np.empty((batch_size, 0), dtype=bool),
            "acceleration_valid": np.empty((batch_size, 0), dtype=bool),
        }

    gaps = np.maximum(delta_t[:, 1:hist_num], eps)
    transition_valid = valid_mask[:, :-1] & valid_mask[:, 1:] & (
        delta_t[:, 1:hist_num] > eps
    )
    ref_velocity = (ref_boxes[:, :-1, :3] - ref_boxes[:, 1:, :3]) / gaps[..., None]
    canonical_velocity = (
        canonical_boxes[:, :-1, :3] - canonical_boxes[:, 1:, :3]
    ) / gaps[..., None]
    velocity_jitter = np.linalg.norm(ref_velocity - canonical_velocity, axis=2)

    ref_yaw_delta = wrap_angle(ref_boxes[:, :-1, 3] - ref_boxes[:, 1:, 3])
    canonical_yaw_delta = wrap_angle(
        canonical_boxes[:, :-1, 3] - canonical_boxes[:, 1:, 3]
    )
    yaw_rate_jitter = np.abs(wrap_angle(ref_yaw_delta - canonical_yaw_delta)) / gaps

    if hist_num < 3:
        acceleration_jitter = np.empty((batch_size, 0), dtype=np.float64)
        acceleration_valid = np.empty((batch_size, 0), dtype=bool)
    else:
        accel_gaps = np.maximum(0.5 * (gaps[:, :-1] + gaps[:, 1:]), eps)
        ref_accel = (ref_velocity[:, :-1] - ref_velocity[:, 1:]) / accel_gaps[..., None]
        canonical_accel = (
            canonical_velocity[:, :-1] - canonical_velocity[:, 1:]
        ) / accel_gaps[..., None]
        acceleration_jitter = np.linalg.norm(ref_accel - canonical_accel, axis=2)
        acceleration_valid = transition_valid[:, :-1] & transition_valid[:, 1:]

    return {
        "velocity_jitter": velocity_jitter,
        "yaw_rate_jitter": yaw_rate_jitter,
        "acceleration_jitter": acceleration_jitter,
        "transition_valid": transition_valid,
        "acceleration_valid": acceleration_valid,
    }


def optimal_convex_blend(observation, dynamics, target, eps=1e-12):
    """Return the per-sample optimum on the segment observation -> dynamics."""
    observation = np.asarray(observation, dtype=np.float64)
    dynamics = np.asarray(dynamics, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if observation.shape != dynamics.shape or observation.shape != target.shape:
        raise ValueError("observation, dynamics and target must share shape [B,D].")
    innovation = dynamics - observation
    denominator = np.sum(innovation * innovation, axis=1)
    numerator = np.sum((target - observation) * innovation, axis=1)
    alpha = np.zeros_like(denominator)
    nonzero = denominator > eps
    alpha[nonzero] = numerator[nonzero] / denominator[nonzero]
    alpha = np.clip(alpha, 0.0, 1.0)
    oracle = observation + alpha[:, None] * innovation
    return alpha, oracle


def tracklet_bootstrap_ci(rows, metric, iterations=10000, seed=42):
    grouped = {}
    for row in rows:
        value = float(row[metric])
        if not math.isfinite(value):
            continue
        grouped.setdefault(int(row["tracklet_id"]), []).append(value)
    tracklet_means = np.asarray(
        [np.mean(values) for values in grouped.values()], dtype=np.float64
    )
    if tracklet_means.size == 0:
        return {"tracklet_count": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        tracklet_means,
        size=(int(iterations), tracklet_means.size),
        replace=True,
    )
    boot_means = sampled.mean(axis=1)
    ci = np.quantile(boot_means, [0.025, 0.975])
    return {
        "tracklet_count": int(tracklet_means.size),
        "mean": float(tracklet_means.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    value = ""
                cleaned[key] = value
            writer.writerow(cleaned)


class IndexedDiagnosticDataset:
    """Attach stable requested-index metadata without changing the sampler."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        anno_id = self.sampler.get_anno_index(index)
        expected_candidate = self.sampler.get_candidate_index(index)
        tracklet_id, expected_frame = self.sampler._locate_tracklet(anno_id)
        sample = self.sampler[index]
        if not isinstance(sample, dict) or "view_a" in sample:
            raise RuntimeError("M0 diagnostics require a single-view MotionTrackingSamplerMF.")
        actual_candidate = int(np.asarray(sample["candidate_id"]).reshape(-1)[0])
        actual_frame = int(np.asarray(sample["this_frame_id"]).reshape(-1)[0])
        resampled = actual_candidate != expected_candidate or actual_frame != expected_frame
        sample["_diagnostic_index"] = np.int64(index)
        sample["_diagnostic_tracklet_id"] = np.int64(-1 if resampled else tracklet_id)
        sample["_diagnostic_expected_frame_id"] = np.int64(expected_frame)
        sample["_diagnostic_resampled"] = np.int64(resampled)
        return sample


def move_to_device(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def tensor_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_checkpoint_state(path):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)}")
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError("Checkpoint must contain a string-keyed state_dict.")
    normalized = {}
    for key, value in state.items():
        for prefix in ("model.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value
    return normalized


def matching_state(source, target_state, include=None, exclude=None):
    selected = {}
    shape_mismatches = []
    for key, value in source.items():
        if include is not None and not any(key.startswith(prefix) for prefix in include):
            continue
        if exclude is not None and any(key.startswith(prefix) for prefix in exclude):
            continue
        if key not in target_state:
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches.append(
                {"key": key, "source": list(value.shape), "target": list(target_state[key].shape)}
            )
            continue
        selected[key] = value
    return selected, shape_mismatches
