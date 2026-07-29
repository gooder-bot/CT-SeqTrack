"""Dependency-light contracts for CT-v2 training-history augmentation."""

import numpy as np


_CT_HISTORY_TRAINING_MODES = {
    "canonical": "canonical",
    "clean": "canonical",
    "correlated": "correlated_candidate",
    "correlated_candidate": "correlated_candidate",
    "recursive": "recursive_candidate",
    "recursive_candidate": "recursive_candidate",
}


def normalize_ct_history_training_mode(mode):
    """Normalize the history mode while keeping legacy defaults clean."""
    key = str(mode if mode is not None else "canonical").lower().replace("-", "_")
    if key not in _CT_HISTORY_TRAINING_MODES:
        raise ValueError(
            "ct_history_training_mode must be 'canonical', "
            "'correlated_candidate', or 'recursive_candidate'")
    return _CT_HISTORY_TRAINING_MODES[key]


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
