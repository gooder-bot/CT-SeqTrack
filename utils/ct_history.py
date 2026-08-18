"""Dependency-light contracts for CT training-history augmentation."""

import hashlib
import numpy as np


_CT_HISTORY_TRAINING_MODES = {
    "canonical": "canonical",
    "clean": "canonical",
    "correlated": "correlated_candidate",
    "correlated_candidate": "correlated_candidate",
    "recursive": "recursive_candidate",
    "recursive_candidate": "recursive_candidate",
    "recursive_replay": "recursive_replay",
}


def normalize_ct_history_training_mode(mode):
    """Normalize the history mode while keeping legacy defaults clean."""
    key = str(mode if mode is not None else "canonical").lower().replace("-", "_")
    if key not in _CT_HISTORY_TRAINING_MODES:
        raise ValueError(
            "ct_history_training_mode must be 'canonical', "
            "'correlated_candidate', 'recursive_candidate', or "
            "'recursive_replay'")
    return _CT_HISTORY_TRAINING_MODES[key]


CT_HISTORY_MODE_IDS = {
    "canonical": 0,
    "correlated_candidate": 1,
    "recursive_candidate": 2,
    "recursive_replay": 3,
}


def select_b2_v3_history_mode(
        tracklet_key, frame_id, candidate_id, seed=42):
    """Choose the one causal history consumed by both B1 and B2-v3.

    Candidate zero is always canonical.  Every other assignment is a stable
    50/50 hash split, so changing point-cloud sampling or worker order cannot
    change the state seen by either branch.
    """
    candidate_id = int(candidate_id)
    if candidate_id == 0:
        return "canonical"
    payload = (
        f"b2-v3::{int(seed)}::{str(tracklet_key)}::"
        f"{int(frame_id)}::{candidate_id}"
    ).encode("utf-8")
    bucket = hashlib.sha256(payload).digest()[0] & 1
    return (
        "correlated_candidate" if bucket == 0
        else "recursive_candidate"
    )


def b2_v3_history_mode_id(mode):
    """Return a serialized id for the shared-history audit trail."""
    normalized = normalize_ct_history_training_mode(mode)
    return CT_HISTORY_MODE_IDS[normalized]


def correlate_candidate_offsets(offsets, correlation, anchor_offset):
    """Turn independent offsets into a smooth, anchor-controlled trajectory."""
    offsets = np.asarray(offsets, dtype=np.float64)
    anchor_offset = np.asarray(anchor_offset, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("candidate offsets must have shape [history, 3]")
    if anchor_offset.shape != (3,):
        raise ValueError("history anchor offset must have shape [3]")
    if not np.isfinite(offsets).all() or not np.isfinite(anchor_offset).all():
        raise ValueError("candidate history offsets must be finite")
    correlation = float(correlation)
    if not 0.0 <= correlation < 1.0:
        raise ValueError("ct_history_correlation must be in [0, 1)")
    if len(offsets) == 0:
        return offsets.astype(np.float32)

    correlated = np.empty_like(offsets)
    correlated[0] = anchor_offset
    for index in range(1, len(offsets)):
        correlated[index] = (
            correlation * correlated[index - 1]
            + (1.0 - correlation) * offsets[index]
        )
    return correlated.astype(np.float32)


def build_ct_history_offsets(
        candidate_offsets,
        candidate_id,
        candidate_trajectory_mode,
        training_mode="canonical",
        correlation=0.75,
        recursive_error_scale=1.0):
    """Return motion/search offsets for CT training.

    ``None`` for search offsets means that an already coherent shared-SE(2)
    candidate trajectory should be reused directly.
    """
    candidate_offsets = np.asarray(candidate_offsets, dtype=np.float32)
    if candidate_offsets.ndim != 2 or candidate_offsets.shape[1] != 3:
        raise ValueError("candidate offsets must have shape [history, 3]")
    training_mode = normalize_ct_history_training_mode(training_mode)
    clean_offsets = np.zeros_like(candidate_offsets)
    if training_mode == "canonical" or int(candidate_id) == 0:
        return clean_offsets, clean_offsets.copy()
    if candidate_trajectory_mode == "shared_se2":
        return clean_offsets, None
    if candidate_trajectory_mode != "independent":
        raise ValueError(
            "candidate_trajectory_mode must be independent or shared_se2")

    motion_offsets = correlate_candidate_offsets(
        candidate_offsets,
        correlation,
        anchor_offset=np.zeros(3, dtype=np.float32),
    )
    if training_mode == "recursive_candidate":
        recursive_error_scale = float(recursive_error_scale)
        if recursive_error_scale < 1.0:
            raise ValueError("ct_history_recursive_error_scale must be >= 1")
        age_scale = np.sqrt(
            np.arange(len(motion_offsets), dtype=np.float32) + 1.0)
        age_scale[0] = 0.0
        motion_offsets = (
            motion_offsets
            * age_scale[:, None]
            * recursive_error_scale
        ).astype(np.float32)
        # The newest box remains the actual crop anchor while older history
        # accumulates correlated drift relative to it.
        search_offsets = (
            candidate_offsets[0:1] + motion_offsets
        ).astype(np.float32)
    else:
        search_offsets = correlate_candidate_offsets(
            candidate_offsets,
            correlation,
            anchor_offset=candidate_offsets[0],
        )
    return motion_offsets, search_offsets


def build_irregular_history_offsets(
        hist_num, query_gap, transition_gaps):
    """Build causal frame offsets for mixed-cadence training.

    ``query_gap`` separates the current frame from the newest history frame;
    subsequent positive increments walk farther into the past.  The returned
    offsets are strictly increasing and therefore cannot leak a future frame.
    """
    hist_num = int(hist_num)
    query_gap = int(query_gap)
    transition_gaps = [int(value) for value in transition_gaps]
    if hist_num <= 0:
        raise ValueError("hist_num must be positive")
    if query_gap <= 0:
        raise ValueError("query_gap must be positive")
    if any(value <= 0 for value in transition_gaps):
        raise ValueError("trajectory training gaps must be positive")
    if not transition_gaps:
        transition_gaps = [1]
    offsets = [query_gap]
    for index in range(1, hist_num):
        increment = transition_gaps[
            (index - 1) % len(transition_gaps)]
        offsets.append(offsets[-1] + increment)
    return offsets


def build_alternating_aux_history_offsets(
        hist_num, sample_index, query_gaps=(2, 4),
        transition_gaps=(1, 2)):
    """Select a deterministic box-only cadence without consuming any RNG."""
    query_gaps = [int(value) for value in query_gaps]
    if not query_gaps or any(value <= 0 for value in query_gaps):
        raise ValueError("auxiliary query gaps must be positive")
    gap = query_gaps[int(sample_index) % len(query_gaps)]
    return build_irregular_history_offsets(
        hist_num, gap, transition_gaps)


def normalize_causal_temporal_gaps(gaps=(2, 4, 8)):
    """Validate the registered temporal-candidate query gaps."""
    values = [int(value) for value in gaps]
    if len(values) < 2:
        raise ValueError("causal temporal candidates require at least two gaps")
    if any(value <= 1 for value in values):
        raise ValueError("auxiliary temporal gaps must be greater than one")
    if values != sorted(set(values)):
        raise ValueError("causal temporal gaps must be unique and increasing")
    return values


def build_causal_temporal_history_offsets(hist_num, gap):
    """Return the complete local history ``[g,g+1,...]`` for one gap."""
    return build_irregular_history_offsets(hist_num, int(gap), [1])


def select_causal_temporal_candidates(
        boundary_ratios, available, *, boundary_band=0.2):
    """Select deterministic boundary/outside roles without target labels.

    Candidate 2 first takes the closest endpoint strictly outside the B0 crop.
    Candidate 1 then takes the closest remaining endpoint to the boundary.  If
    no endpoint is outside, the two largest ratios become boundary and fallback
    outside roles.  Returned role ids are the serialized candidate ids 1/2.
    """
    ratios = {
        int(gap): float(value) for gap, value in boundary_ratios.items()}
    availability = {
        int(gap): bool(value) for gap, value in available.items()}
    if set(ratios) != set(availability):
        raise ValueError("boundary ratios and availability must share gaps")
    if not ratios:
        raise ValueError("candidate selection requires at least one gap")
    if any(not np.isfinite(value) or value < 0.0 for value in ratios.values()):
        raise ValueError("boundary ratios must be finite and non-negative")
    boundary_band = float(boundary_band)
    if boundary_band < 0.0:
        raise ValueError("boundary band must be non-negative")

    valid_gaps = sorted(gap for gap in ratios if availability[gap])
    selected = {}
    outside = sorted(
        (gap for gap in valid_gaps if ratios[gap] > 1.0),
        key=lambda gap: (ratios[gap] - 1.0, gap),
    )
    if outside:
        outside_gap = outside[0]
        selected[2] = outside_gap
        remaining = [gap for gap in valid_gaps if gap != outside_gap]
        if remaining:
            selected[1] = min(
                remaining, key=lambda gap: (abs(ratios[gap] - 1.0), gap))
    else:
        descending = sorted(valid_gaps, key=lambda gap: (-ratios[gap], gap))
        if descending:
            selected[1] = descending[0]
        if len(descending) > 1:
            selected[2] = descending[1]

    result = {}
    for role_id in (1, 2):
        gap = selected.get(role_id)
        ratio = 0.0 if gap is None else ratios[gap]
        role_satisfied = (
            abs(ratio - 1.0) <= boundary_band
            if role_id == 1 else ratio > 1.0)
        result[role_id] = {
            "gap": gap,
            "available": gap is not None,
            "boundary_ratio": ratio,
            "role_satisfied": bool(gap is not None and role_satisfied),
        }
    return result


def select_uniform_temporal_candidates(
        boundary_ratios, available, *, seed_parts=()):
    """Choose up to two valid gaps uniformly without target/B1 ranking.

    Hash ordering is deterministic for an online slot but changes with the
    supplied epoch/tracklet/frame identity.  Boundary ratios are returned only
    as diagnostics and never affect selection.
    """
    ratios = {
        int(gap): float(value) for gap, value in boundary_ratios.items()}
    availability = {
        int(gap): bool(value) for gap, value in available.items()}
    if set(ratios) != set(availability) or not ratios:
        raise ValueError(
            "uniform candidate ratios/availability must share non-empty gaps")
    if any(not np.isfinite(value) or value < 0.0 for value in ratios.values()):
        raise ValueError("boundary ratios must be finite and non-negative")
    prefix = "::".join(str(value) for value in seed_parts)
    valid_gaps = [gap for gap in sorted(ratios) if availability[gap]]
    ordered = sorted(valid_gaps, key=lambda gap: (
        hashlib.sha256(
            f"causal-uniform::{prefix}::{gap}".encode("utf-8")
        ).digest(),
        gap,
    ))
    result = {}
    for role_id in (1, 2):
        gap = ordered[role_id - 1] if len(ordered) >= role_id else None
        result[role_id] = {
            "gap": gap,
            "available": gap is not None,
            "boundary_ratio": 0.0 if gap is None else ratios[gap],
            # For the uniform control the registered role is simply
            # "selected uniform auxiliary", not boundary/outside.
            "role_satisfied": gap is not None,
        }
    return result
