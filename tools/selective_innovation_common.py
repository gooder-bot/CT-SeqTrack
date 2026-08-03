"""Shared artifact and checkpoint helpers for B2-v2.2 tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.ct_v2 import (
    SELECTIVE_ROLLOUT_SCHEMA,
    SELECTIVE_ROUTER_SCHEMA,
    SignedHorizonInnovationRouter,
)


class ConfigMap(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def checkpoint_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state_dict")
    return state


def normalize_state_dict(source, target):
    candidates = [("none", source)]
    for prefix in ("model.", "module."):
        candidates.append((prefix, {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in source.items()
        }))
    return max(
        candidates,
        key=lambda item: sum(
            key in target and hasattr(value, "shape")
            and target[key].shape == value.shape
            for key, value in item[1].items()),
    )


def load_matching_model_state(model, checkpoint_path):
    payload = torch_load(checkpoint_path)
    source = checkpoint_state_dict(payload)
    target = model.state_dict()
    prefix, normalized = normalize_state_dict(source, target)
    matched = {
        key: value for key, value in normalized.items()
        if key in target and hasattr(value, "shape")
        and target[key].shape == value.shape
    }
    critical = (
        "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
        "feature_pointnet.", "Transformer.", "physical_motion_encoder.",
        "motion_conditioned_search_refiner.",
    )
    missing = [
        item for item in critical
        if not any(key.startswith(item) for key in matched)]
    if missing:
        raise RuntimeError(
            "B2-v2.2 checkpoint is missing: " + ", ".join(missing))
    target.update(matched)
    model.load_state_dict(target, strict=True)
    return {
        "source_prefix": prefix,
        "matched_tensors": len(matched),
        "target_tensors": len(target),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def build_router_from_config(config):
    return SignedHorizonInnovationRouter(
        observation_dim=256,
        motion_dim=int(getattr(config, "motion_v3_hidden_dim", 128)),
        search_dim=int(getattr(config, "search_v22_feature_dim", 128)),
        observation_stats_dim=5,
        context_dim=int(getattr(config, "signed_router_context_dim", 32)),
        hidden_dim=int(getattr(config, "signed_router_hidden_dim", 96)),
        gain_threshold=float(getattr(config, "signed_gain_threshold", 0.0)),
        radius_base=float(getattr(config, "signed_radius_base", 0.5)),
        radius_per_second=float(getattr(
            config, "signed_radius_per_second", 0.5)),
        radius_max=float(getattr(config, "signed_radius_max", 2.0)),
        normal_step_cap=float(getattr(
            config, "signed_normal_step_cap", 0.20)),
        gap_step_cap=float(getattr(config, "signed_gap_step_cap", 0.35)),
    )


def rollout_paths(path):
    path = Path(path)
    if path.is_dir():
        return path / "selective_rollouts.npz", path / "manifest.json"
    if path.suffix.lower() != ".npz":
        raise ValueError("rollout artifact must be an NPZ file or directory")
    return path, path.with_name("manifest.json")


def write_rollout_artifact(output_dir, rows, manifest):
    if not rows:
        raise ValueError("cannot write an empty selective rollout artifact")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_keys = (
        "tracklet_id", "frame_id", "router_features", "candidate_valid",
        "candidate_residual_xy", "signed_gain", "candidate_cost",
        "observation_cost", "rollout_length",
    )
    arrays = {
        key: np.stack([row[key] for row in rows], axis=0)
        for key in numeric_keys
    }
    arrays["tracklet_key"] = np.asarray(
        [str(row["tracklet_key"]) for row in rows], dtype=np.str_)
    arrays["partition"] = np.asarray(
        [str(row["partition"]) for row in rows], dtype=np.str_)
    npz_path = output_dir / "selective_rollouts.npz"
    np.savez_compressed(npz_path, **arrays)
    payload = dict(manifest)
    payload.update({
        "schema": SELECTIVE_ROLLOUT_SCHEMA,
        "row_count": len(rows),
        "tracklet_count": len(set(arrays["tracklet_key"].tolist())),
        "router_feature_dim": int(arrays["router_features"].shape[1]),
        "step_ratios": [0.25, 0.5, 1.0],
        "content_sha256": canonical_sha256({
            "tracklet_key": arrays["tracklet_key"].tolist(),
            "frame_id": arrays["frame_id"].tolist(),
            "signed_gain_shape": arrays["signed_gain"].shape,
        }),
    })
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return npz_path, manifest_path


def load_rollout_artifact(path):
    npz_path, manifest_path = rollout_paths(path)
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("selective rollout artifact is incomplete")
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("schema") != SELECTIVE_ROLLOUT_SCHEMA:
        raise ValueError("unsupported selective rollout schema")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    rows = int(arrays["router_features"].shape[0])
    if rows != int(manifest.get("row_count", -1)):
        raise ValueError("selective rollout row count mismatch")
    for key, value in arrays.items():
        if value.shape[0] != rows:
            raise ValueError(f"rollout field {key} has a bad row count")
    if arrays["signed_gain"].shape[1:] != (2, 3):
        raise ValueError("selective rollout signed_gain shape is invalid")
    return arrays, manifest, {
        "npz_sha256": sha256_file(npz_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def load_router_sidecar(router, path, require_passed=True):
    payload = torch_load(path)
    if payload.get("schema") != SELECTIVE_ROUTER_SCHEMA:
        raise ValueError("unsupported signed router sidecar schema")
    status = str(payload.get("calibration", {}).get("status", "unknown"))
    if require_passed and status != "passed":
        raise RuntimeError(
            f"signed router calibration is {status!r}, expected 'passed'")
    router.load_state_dict(payload["router_state_dict"], strict=True)
    return payload
