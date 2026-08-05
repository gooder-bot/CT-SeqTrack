"""Artifact, checkpoint, and router helpers for B2-v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.ct_v2 import (
    ActionConsistentInnovationRouter,
    SELECTIVE_V3_ROLLOUT_SCHEMA,
    SELECTIVE_V3_ROUTER_SCHEMA,
    SELECTIVE_V4_ROLLOUT_SCHEMA,
    SELECTIVE_V4_ROUTER_SCHEMA,
)
from tools.selective_innovation_common import (
    ConfigMap,
    canonical_sha256,
    checkpoint_state_dict,
    normalize_state_dict,
    sha256_file,
    torch_load,
)
from utils.checkpoint_loading import apply_b1_calibration_contract
from utils.replay_cache import b2_candidate_config_sha256


def build_v3_router_from_config(config):
    return ActionConsistentInnovationRouter(
        observation_dim=256,
        motion_dim=int(getattr(config, "motion_v3_hidden_dim", 128)),
        search_dim=3 * int(getattr(config, "search_v3_feature_dim", 128)),
        observation_stats_dim=5,
        context_dim=int(getattr(config, "router_v3_context_dim", 32)),
        hidden_dim=int(getattr(config, "router_v3_hidden_dim", 96)),
        gain_threshold=float(getattr(config, "router_v3_gain_threshold", 0.0)),
        radius_base=float(getattr(config, "router_v3_radius_base", 0.5)),
        radius_per_second=float(getattr(
            config, "router_v3_radius_per_second", 0.5)),
        radius_max=float(getattr(config, "router_v3_radius_max", 2.0)),
        normal_step_cap=float(getattr(
            config, "router_v3_normal_step_cap", 0.20)),
        gap_step_cap=float(getattr(config, "router_v3_gap_step_cap", 0.35)),
        scalar_only=bool(getattr(config, "router_v3_scalar_only", False)),
        use_utility_feature=bool(getattr(
            config, "router_v3_use_utility_feature", False)),
    )


def tensor_prefix_hash(state, prefix):
    digest = hashlib.sha256()
    keys = sorted(key for key in state if key.startswith(prefix))
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), keys


def load_matching_v3_model_state(model, checkpoint_path):
    payload = torch_load(checkpoint_path)
    source = checkpoint_state_dict(payload)
    target = model.state_dict()
    prefix, normalized = normalize_state_dict(source, target)
    shape_mismatch = sorted(
        key for key, value in normalized.items()
        if key in target and target[key].shape != value.shape)
    if shape_mismatch:
        raise RuntimeError(
            "B2-v3 checkpoint shape mismatch: "
            + ", ".join(shape_mismatch[:20]))
    matched = {
        key: value for key, value in normalized.items()
        if key in target and target[key].shape == value.shape
    }
    missing_target = sorted(
        key for key in set(target) - set(matched)
        if not key.startswith("action_consistent_router_v3."))
    unexpected_source = sorted(set(normalized) - set(target))
    if missing_target or unexpected_source:
        raise RuntimeError(
            "B2-v3 candidate checkpoint key set mismatch; "
            f"missing={missing_target[:20]}, extra={unexpected_source[:20]}")
    nonfinite = sorted(
        key for key, value in matched.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item()))
    if nonfinite:
        raise RuntimeError(
            "B2-v3 candidate checkpoint contains non-finite tensors: "
            + ", ".join(nonfinite[:20]))
    required_frozen_prefixes = (
        "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
        "motion_state_mlp.", "feature_pointnet.", "Transformer.",
        "physical_motion_encoder.",
    )
    missing = [
        required for required in (
            *required_frozen_prefixes, "state_aligned_search_refiner.")
        if not any(key.startswith(required) for key in matched)]
    if missing:
        raise RuntimeError(
            "B2-v3 checkpoint is missing: " + ", ".join(missing))
    frozen_hashes = payload.get("b2_v3_frozen_reference_hashes")
    if not isinstance(frozen_hashes, dict):
        raise RuntimeError(
            "B2-v3 candidate checkpoint lacks frozen-weight provenance")
    if sorted(frozen_hashes) != sorted(required_frozen_prefixes):
        raise RuntimeError(
            "B2-v3 frozen hash manifest does not cover the complete B0/B1 set")
    for required in required_frozen_prefixes:
        actual_hash, keys = tensor_prefix_hash(matched, required)
        if not keys or actual_hash != frozen_hashes.get(required):
            raise RuntimeError(
                f"B2-v3 frozen checkpoint hash mismatch: {required}")
    apply_b1_calibration_contract(model, payload, matched)
    model_config = getattr(model, "config", {})
    if bool(getattr(
            model_config, "require_b2_candidate_config_contract", False)):
        stored_candidate_config = payload.get(
            "b2_v3_candidate_config_sha256")
        expected_candidate_config = b2_candidate_config_sha256(model_config)
        if (not stored_candidate_config
                or stored_candidate_config != expected_candidate_config):
            raise RuntimeError(
                "B2 candidate checkpoint/config contract mismatch")
    target.update(matched)
    model.load_state_dict(target, strict=True)
    return {
        "source_prefix": prefix,
        "matched_tensors": len(matched),
        "target_tensors": len(target),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "frozen_prefix_hashes": dict(frozen_hashes),
    }


def load_v3_router_sidecar(router, path, require_passed=True):
    payload = torch_load(path)
    expected_schema = (
        SELECTIVE_V4_ROUTER_SCHEMA
        if router.use_utility_feature else SELECTIVE_V3_ROUTER_SCHEMA)
    if payload.get("schema") != expected_schema:
        raise ValueError("unsupported B2-v3 router sidecar schema")
    if expected_schema == SELECTIVE_V4_ROUTER_SCHEMA:
        if payload.get("feature_schema_hash") != router.feature_schema_hash:
            raise ValueError("router sidecar feature schema hash mismatch")
        if payload.get("feature_schema") != router.feature_schema:
            raise ValueError("router sidecar feature names/order mismatch")
    status = str(payload.get("calibration", {}).get("status", "unknown"))
    if require_passed and status != "passed":
        raise RuntimeError(
            f"B2-v3 router calibration is {status!r}, expected 'passed'")
    router.load_state_dict(payload["router_state_dict"], strict=True)
    return payload


def rollout_paths(path):
    path = Path(path)
    if path.is_dir():
        return path / "selective_v3_rollouts.npz", path / "manifest.json"
    if path.suffix.lower() != ".npz":
        raise ValueError("rollout artifact must be an NPZ file or directory")
    return path, path.with_name("manifest.json")


def write_v3_rollout_artifact(output_dir, rows, manifest):
    if not rows:
        raise ValueError("cannot write an empty B2-v3 rollout artifact")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_keys = [
        "tracklet_id", "frame_id", "router_features", "candidate_valid",
        "candidate_residual_xy", "signed_gain", "candidate_cost",
        "observation_cost", "rollout_length",
    ]
    for optional in (
            "candidate_success", "observation_success", "success_gain"):
        if optional in rows[0]:
            numeric_keys.append(optional)
    arrays = {
        key: np.stack([row[key] for row in rows], axis=0)
        for key in numeric_keys
    }
    arrays["tracklet_key"] = np.asarray(
        [str(row["tracklet_key"]) for row in rows], dtype=np.str_)
    arrays["partition"] = np.asarray(
        [str(row["partition"]) for row in rows], dtype=np.str_)
    npz_path = output_dir / "selective_v3_rollouts.npz"
    np.savez_compressed(npz_path, **arrays)
    payload = dict(manifest)
    payload.update({
        "schema": payload.get("schema", SELECTIVE_V3_ROLLOUT_SCHEMA),
        "row_count": len(rows),
        "tracklet_count": len(set(arrays["tracklet_key"].tolist())),
        "router_feature_dim": int(arrays["router_features"].shape[1]),
        "npz_sha256": sha256_file(npz_path),
        "step_ratios": [0.25, 0.5, 1.0],
        "content_sha256": canonical_sha256({
            "tracklet_key": arrays["tracklet_key"].tolist(),
            "frame_id": arrays["frame_id"].tolist(),
            "signed_gain": hashlib.sha256(
                arrays["signed_gain"].tobytes()).hexdigest(),
        }),
    })
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return npz_path, manifest_path


def load_v3_rollout_artifact(path):
    npz_path, manifest_path = rollout_paths(path)
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("B2-v3 rollout artifact is incomplete")
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("schema") not in (
            SELECTIVE_V3_ROLLOUT_SCHEMA, SELECTIVE_V4_ROLLOUT_SCHEMA):
        raise ValueError("unsupported B2-v3 rollout schema")
    if (manifest.get("npz_sha256") is not None
            and sha256_file(npz_path) != manifest.get("npz_sha256")):
        raise ValueError("B2-v3 rollout NPZ SHA256 mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if manifest.get("schema") == SELECTIVE_V4_ROLLOUT_SCHEMA:
        for required in (
                "feature_schema", "feature_schema_hash",
                "candidate_checkpoint_sha256", "candidate_config_sha256",
                "promotion_manifest_sha256", "npz_sha256"):
            if not manifest.get(required):
                raise ValueError(
                    f"formal rollout manifest lacks {required}")
        for required in (
                "candidate_success", "observation_success", "success_gain"):
            if required not in arrays:
                raise ValueError(
                    f"formal rollout arrays lack {required}")
    rows = int(arrays["router_features"].shape[0])
    if rows != int(manifest.get("row_count", -1)):
        raise ValueError("B2-v3 rollout row count mismatch")
    if arrays["signed_gain"].shape[1:] != (2, 3):
        raise ValueError("B2-v3 signed_gain shape is invalid")
    if arrays["candidate_valid"].shape[1:] != (2,):
        raise ValueError("B2-v3 candidate_valid shape is invalid")
    if manifest.get("schema") == SELECTIVE_V4_ROLLOUT_SCHEMA:
        if (arrays["candidate_success"].shape[1:] != (2, 3)
                or arrays["success_gain"].shape[1:] != (2, 3)
                or arrays["observation_success"].shape[1:] != ()):
            raise ValueError("formal recursive Success arrays are invalid")
    for key, value in arrays.items():
        if value.shape[0] != rows:
            raise ValueError(f"rollout field {key} has a bad row count")
    return arrays, manifest, {
        "npz_sha256": sha256_file(npz_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


__all__ = [
    "ConfigMap", "build_v3_router_from_config",
    "load_matching_v3_model_state", "load_v3_router_sidecar",
    "load_v3_rollout_artifact", "write_v3_rollout_artifact",
    "tensor_prefix_hash",
]
