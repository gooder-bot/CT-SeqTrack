"""Deterministic sampling helpers for candidate and paired views."""

import hashlib

import numpy as np


def stable_uint32_seed(base_seed, *parts):
    payload = "::".join((str(int(base_seed)), *(str(part) for part in parts)))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def deterministic_candidate_offset(candidate_id, config, *seed_parts):
    if int(candidate_id) == 0:
        return np.zeros(3, dtype=np.float32)
    base_seed = int(getattr(config, "seed", 42) or 42)
    rng = np.random.default_rng(stable_uint32_seed(
        base_seed, "candidate", int(candidate_id), *seed_parts))
    offset = rng.uniform(low=-0.3, high=0.3, size=3).astype(np.float32)
    offset[2] *= 5 if config.degrees else np.deg2rad(5)
    return offset


def deterministic_recovery_candidate_offset(
        candidate_id, config, anchor_box, target_box, *seed_parts):
    """Create train-only weak-boundary and strict-miss acquisition views."""
    candidate_id = int(candidate_id)
    if candidate_id == 0:
        return np.zeros(3, dtype=np.float32)
    if candidate_id not in (1, 2):
        return deterministic_candidate_offset(
            candidate_id, config, *seed_parts)
    base_seed = int(getattr(config, "seed", 42) or 42)
    named_seed = stable_uint32_seed(
        base_seed, "recovery-role", candidate_id, *seed_parts)
    rng = np.random.default_rng(named_seed)
    axis = int(named_seed % 2)
    sign = -1.0 if int((named_seed // 2) % 2) else 1.0
    anchor_center = np.asarray(anchor_box.center, dtype=np.float64)
    target_center = np.asarray(target_box.center, dtype=np.float64)
    target_local = np.asarray(anchor_box.rotation_matrix, dtype=np.float64).T @ (
        target_center - anchor_center)
    anchor_size = np.asarray(anchor_box.wlh, dtype=np.float64)
    target_size = np.asarray(target_box.wlh, dtype=np.float64)
    crop_half_extent = (
        0.5 * anchor_size[axis] * float(getattr(config, "bb_scale", 1.0))
        + float(getattr(config, "bb_offset", 0.0)))
    target_half_extent = 0.5 * target_size[axis]
    if candidate_id == 1:
        desired_target_offset = max(
            0.0, crop_half_extent - 0.35 * target_half_extent)
    else:
        desired_target_offset = (
            crop_half_extent + target_half_extent + 0.05)
    natural = deterministic_candidate_offset(
        candidate_id, config, *seed_parts)
    transform = natural.astype(np.float64)
    transform[:2] = target_local[:2]
    transform[axis] = target_local[axis] - sign * desired_target_offset
    transform[1 - axis] += float(rng.uniform(-0.05, 0.05))
    return transform.astype(np.float32)


def deterministic_point_seed(config, *seed_parts):
    base_seed = int(getattr(config, "seed", 42) or 42)
    return stable_uint32_seed(base_seed, "points", *seed_parts)


def sample_candidate_offset(candidate_id, config):
    if int(candidate_id) == 0:
        return np.zeros(3, dtype=np.float32)
    offset = np.random.uniform(low=-0.3, high=0.3, size=3).astype(np.float32)
    offset[2] *= 5 if config.degrees else np.deg2rad(5)
    return offset


def build_shared_candidate_offset_map(candidate_id, frame_ids, config):
    return {
        int(frame_id): sample_candidate_offset(candidate_id, config)
        for frame_id in sorted(set(int(frame_id) for frame_id in frame_ids))
    }


def candidate_offsets_for_frame_ids(frame_ids, offset_map):
    return np.stack([
        np.asarray(offset_map[int(frame_id)], dtype=np.float32)
        for frame_id in frame_ids], axis=0)


def sample_point_sampling_seed():
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def build_shared_point_sampling_seed_map(frame_ids):
    return {
        int(frame_id): sample_point_sampling_seed()
        for frame_id in sorted(set(int(frame_id) for frame_id in frame_ids))
    }


def point_sampling_seeds_for_frame_ids(frame_ids, seed_map):
    return np.asarray([
        int(seed_map[int(frame_id)]) for frame_id in frame_ids],
        dtype=np.int64)

