"""Lightweight deterministic helpers shared by samplers and smoke tests."""

import hashlib
import numpy as np


def stable_uint32_seed(base_seed, *parts):
    """Return a process-independent seed for one named stochastic operation.

    Python's ``hash`` is intentionally salted per process and the global NumPy
    RNG is shared by otherwise unrelated optional branches.  A short SHA256
    digest gives each operation its own reproducible namespace without
    advancing either global RNG.
    """
    payload = "::".join((str(int(base_seed)), *(str(part) for part in parts)))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def deterministic_candidate_offset(candidate_id, config, *seed_parts):
    """SeqTrack-style perturbation generated from an explicit named seed."""
    if int(candidate_id) == 0:
        return np.zeros(3, dtype=np.float32)
    base_seed = int(getattr(config, "seed", 42) or 42)
    rng = np.random.default_rng(stable_uint32_seed(
        base_seed, "candidate", int(candidate_id), *seed_parts))
    offset = rng.uniform(low=-0.3, high=0.3, size=3).astype(np.float32)
    offset[2] *= 5 if config.degrees else np.deg2rad(5)
    return offset


def deterministic_point_seed(config, *seed_parts):
    """Return a named point-sampling seed without touching global RNG."""
    base_seed = int(getattr(config, "seed", 42) or 42)
    return stable_uint32_seed(base_seed, "points", *seed_parts)


def sample_candidate_offset(candidate_id, config):
    """Sample one SeqTrack3D-style historical-box perturbation."""
    if int(candidate_id) == 0:
        return np.zeros(3, dtype=np.float32)

    offset = np.random.uniform(low=-0.3, high=0.3, size=3).astype(np.float32)
    offset[2] *= 5 if config.degrees else np.deg2rad(5)
    return offset


def build_shared_candidate_offset_map(candidate_id, frame_ids, config):
    """Build candidate offsets keyed by absolute frame id for paired views."""
    return {
        int(frame_id): sample_candidate_offset(candidate_id, config)
        for frame_id in sorted(set(int(frame_id) for frame_id in frame_ids))
    }


def candidate_offsets_for_frame_ids(frame_ids, offset_map):
    return np.stack(
        [np.asarray(offset_map[int(frame_id)], dtype=np.float32) for frame_id in frame_ids],
        axis=0,
    )


def sample_point_sampling_seed():
    """Sample a seed that can be passed to numpy.default_rng()."""
    return int(np.random.randint(0, np.iinfo(np.int32).max))


def build_shared_point_sampling_seed_map(frame_ids):
    """Build point-regularization seeds keyed by absolute history frame id."""
    return {
        int(frame_id): sample_point_sampling_seed()
        for frame_id in sorted(set(int(frame_id) for frame_id in frame_ids))
    }


def point_sampling_seeds_for_frame_ids(frame_ids, seed_map):
    return np.asarray(
        [int(seed_map[int(frame_id)]) for frame_id in frame_ids],
        dtype=np.int64,
    )
