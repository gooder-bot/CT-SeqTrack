"""Shared, side-effect-free helpers for the B3 CRPA command-line tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.ct_v2 import ClosedLoopRiskAwareProposalRouter
from models.ct_v2.crpa import CRPA_ROLLOUT_SCHEMA, CRPA_ROUTER_SCHEMA


class ConfigMap(dict):
    """Minimal EasyDict-compatible mapping without an extra tool dependency."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(
            Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def checkpoint_state_dict(payload) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    state_dict = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint does not contain a state_dict mapping")
    return state_dict


def load_matching_model_state(model, checkpoint_path: str | Path):
    """Load B2 tensors into B3 while leaving the new router initialized."""
    payload = torch_load(checkpoint_path, map_location="cpu")
    source = checkpoint_state_dict(payload)
    target = model.state_dict()
    candidates = [("none", source)]
    for prefix in ("model.", "module."):
        candidates.append((prefix, {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in source.items()
        }))
    prefix, normalized = max(
        candidates,
        key=lambda item: sum(
            key in target and target[key].shape == value.shape
            for key, value in item[1].items()),
    )
    matched = {
        key: value for key, value in normalized.items()
        if key in target and target[key].shape == value.shape
    }
    critical = (
        "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
        "feature_pointnet.", "Transformer.", "physical_motion_encoder.",
        "search_evidence_v21.",
    )
    missing = [
        item for item in critical
        if not any(key.startswith(item) for key in matched)]
    if missing:
        raise RuntimeError(
            "B2 checkpoint is missing CRPA source modules: "
            + ", ".join(missing))
    target.update(matched)
    model.load_state_dict(target, strict=True)
    return {
        "source_prefix": prefix,
        "matched_tensors": len(matched),
        "target_tensors": len(target),
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
    }


def build_router_from_config(config) -> ClosedLoopRiskAwareProposalRouter:
    return ClosedLoopRiskAwareProposalRouter(
        observation_dim=256,
        motion_dim=int(getattr(config, "motion_v3_hidden_dim", 128)),
        search_dim=int(getattr(config, "search_v21_feature_dim", 128)),
        observation_stats_dim=5,
        context_dim=int(getattr(config, "b3_router_context_dim", 32)),
        hidden_dim=int(getattr(config, "b3_router_hidden_dim", 96)),
        gain_threshold=float(getattr(config, "b3_gain_threshold", 0.0)),
        radius_base=float(getattr(config, "b3_radius_base", 0.5)),
        radius_per_second=float(getattr(
            config, "b3_radius_per_second", 0.5)),
        radius_max=float(getattr(config, "b3_radius_max", 2.0)),
        normal_step_cap=float(getattr(
            config, "b3_normal_step_cap", 0.35)),
        gap_step_cap=float(getattr(config, "b3_gap_step_cap", 0.60)),
    )


def load_router_sidecar(router, path: str | Path, require_passed=True):
    payload = torch_load(path, map_location="cpu")
    if payload.get("schema") != CRPA_ROUTER_SCHEMA:
        raise ValueError("unsupported CRPA router sidecar schema")
    status = str(payload.get("calibration", {}).get("status", "unknown"))
    if require_passed and status != "passed":
        raise RuntimeError(
            f"CRPA router calibration status is {status!r}, expected 'passed'")
    router.load_state_dict(payload["router_state_dict"], strict=True)
    return payload


def rollout_paths(path: str | Path):
    path = Path(path)
    if path.is_dir():
        return path / "b3_rollouts.npz", path / "manifest.json"
    if path.suffix.lower() != ".npz":
        raise ValueError("rollout artifact must be an NPZ file or directory")
    return path, path.with_name("manifest.json")


def load_rollout_artifact(path: str | Path):
    npz_path, manifest_path = rollout_paths(path)
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"incomplete rollout artifact: {npz_path}, {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("schema") != CRPA_ROLLOUT_SCHEMA:
        raise ValueError("unsupported CRPA rollout schema")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    row_count = int(arrays["router_features"].shape[0])
    if row_count != int(manifest.get("row_count", -1)):
        raise ValueError("rollout manifest row count mismatch")
    for key, value in arrays.items():
        if value.shape[0] != row_count:
            raise ValueError(f"rollout field {key} has a mismatched row count")
    return arrays, manifest, {
        "npz_sha256": sha256_file(npz_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def write_rollout_artifact(
        output_dir: str | Path,
        rows: list[dict],
        manifest: dict):
    if not rows:
        raise ValueError("cannot write an empty CRPA rollout artifact")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_keys = [
        "frame_id", "router_features", "observation_xy", "target_xy",
        "candidate_residual_xy", "candidate_valid", "step_cap",
        "fusion_radius", "oracle_gain", "oracle_alpha",
        "oracle_step_ratio", "observation_error",
        "oracle_candidate_error", "tracklet_id",
    ]
    arrays = {
        key: np.stack([row[key] for row in rows], axis=0)
        for key in numeric_keys
    }
    arrays["tracklet_key"] = np.asarray(
        [str(row["tracklet_key"]) for row in rows], dtype=np.str_)
    npz_path = output_dir / "b3_rollouts.npz"
    np.savez_compressed(npz_path, **arrays)
    payload = dict(manifest)
    payload.update({
        "schema": CRPA_ROLLOUT_SCHEMA,
        "row_count": len(rows),
        "tracklet_count": len(set(arrays["tracklet_key"].tolist())),
        "router_feature_dim": int(arrays["router_features"].shape[1]),
    })
    payload["content_sha256"] = canonical_sha256({
        "tracklet_key": arrays["tracklet_key"].tolist(),
        "frame_id": arrays["frame_id"].tolist(),
        "router_feature_shape": arrays["router_features"].shape,
    })
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return npz_path, manifest_path
