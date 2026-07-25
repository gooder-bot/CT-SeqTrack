"""Deterministic time-guided search support for CT-SeqTrack v2."""

import copy
import math

import numpy as np


def _finite_positive(value, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(value) or value <= 0:
        return float(fallback)
    return value


def _bounded_velocity(history_boxes, delta_t, valid_mask, max_speed):
    velocities = []
    weights = []
    pair_count = min(len(history_boxes) - 1, len(delta_t) - 1)
    for index in range(max(pair_count, 0)):
        if valid_mask is not None:
            if not (bool(valid_mask[index]) and bool(valid_mask[index + 1])):
                continue
        gap = _finite_positive(delta_t[index + 1], fallback=0.0)
        if gap <= 0:
            continue
        newer = np.asarray(history_boxes[index].center, dtype=np.float64)
        older = np.asarray(history_boxes[index + 1].center, dtype=np.float64)
        velocity = (newer - older) / gap
        speed = float(np.linalg.norm(velocity[:2]))
        if speed > float(max_speed):
            velocity = velocity * (float(max_speed) / max(speed, 1e-9))
        velocities.append(velocity)
        weights.append(1.0 / float(index + 1))
    if not velocities:
        return None
    return np.average(
        np.stack(velocities, axis=0),
        axis=0,
        weights=np.asarray(weights, dtype=np.float64),
    )


def build_time_guided_search_box(
        history_boxes,
        delta_t,
        valid_mask=None,
        base_length=4.0,
        base_width=2.0,
        max_length=16.0,
        max_width=6.0,
        max_speed=20.0,
        max_displacement=12.0,
        width_per_second=0.25,
        min_displacement=0.2):
    """Build a bounded tube from the latest box toward a real-time prediction.

    The function never replaces the baseline crop.  It only describes an
    additional region that callers may sample with a fixed expansion quota.
    No ground-truth current box or oracle reachability signal is consumed.
    """
    if len(history_boxes) < 2:
        return None, {"valid": False, "reason": "insufficient_history"}
    query_gap = _finite_positive(
        delta_t[0] if len(delta_t) else None, fallback=0.0)
    if query_gap <= 0:
        return None, {"valid": False, "reason": "invalid_query_gap"}
    velocity = _bounded_velocity(
        history_boxes, delta_t, valid_mask, max_speed=max_speed)
    if velocity is None:
        return None, {"valid": False, "reason": "no_valid_transition"}

    start = np.asarray(history_boxes[0].center, dtype=np.float64)
    displacement = velocity * query_gap
    planar_norm = float(np.linalg.norm(displacement[:2]))
    if planar_norm > float(max_displacement):
        displacement = displacement * (
            float(max_displacement) / max(planar_norm, 1e-9))
        planar_norm = float(max_displacement)
    if planar_norm < float(min_displacement):
        return None, {
            "valid": False,
            "reason": "stationary",
            "query_delta_t": query_gap,
            "speed": float(np.linalg.norm(velocity[:2])),
        }

    end = start + displacement
    yaw = math.atan2(displacement[1], displacement[0])
    length = min(
        float(max_length),
        max(float(base_length), float(base_length) + planar_norm),
    )
    width = min(
        float(max_width),
        max(
            float(base_width),
            float(base_width) + float(width_per_second) * query_gap,
        ),
    )

    tube = copy.deepcopy(history_boxes[0])
    tube.center = 0.5 * (start + end)
    orientation_type = history_boxes[0].orientation.__class__
    tube.orientation = orientation_type(axis=[0, 0, 1], radians=yaw)
    tube.wlh = np.asarray(tube.wlh, dtype=np.float64).copy()
    tube.wlh[0] = width
    tube.wlh[1] = length
    return tube, {
        "valid": True,
        "reason": "ok",
        "query_delta_t": query_gap,
        "speed": float(np.linalg.norm(velocity[:2])),
        "displacement": planar_norm,
        "length": length,
        "width": width,
    }


def _sample_rows(points, sample_size, rng):
    points = np.asarray(points)
    if sample_size <= 0:
        return points[:0]
    if points.ndim != 2:
        raise ValueError("search points must be a [N, C] array")
    if len(points) <= 2:
        return np.zeros((sample_size, points.shape[1]), dtype=np.float32)
    indices = rng.choice(
        len(points), size=int(sample_size), replace=int(sample_size) > len(points))
    return points[indices]


def _extension_only(primary, expanded, tolerance=1e-6):
    if len(expanded) == 0:
        return expanded
    if len(primary) == 0:
        return expanded
    tolerance = _finite_positive(tolerance, fallback=1e-6)
    coordinate_dims = min(primary.shape[1], expanded.shape[1], 3)
    primary_keys = {
        tuple(row) for row in np.rint(
            primary[:, :coordinate_dims] / tolerance).astype(np.int64)
    }
    expanded_keys = np.rint(
        expanded[:, :coordinate_dims] / tolerance).astype(np.int64)
    keep = np.fromiter(
        (tuple(row) not in primary_keys for row in expanded_keys),
        dtype=bool,
        count=len(expanded_keys),
    )
    return expanded[keep]


def stratified_search_sample(
        baseline_points,
        expanded_points,
        sample_size,
        baseline_ratio=0.75,
        min_expansion_points=32,
        seed=None,
        tolerance=1e-6):
    """Sample a fixed budget while guaranteeing a baseline-crop majority."""
    baseline_points = np.asarray(baseline_points)
    expanded_points = np.asarray(expanded_points)
    sample_size = int(sample_size)
    baseline_ratio = float(baseline_ratio)
    if sample_size <= 0:
        raise ValueError("point_sample_size must be positive")
    if not 0.5 <= baseline_ratio <= 1.0:
        raise ValueError("ct_search_baseline_ratio must be in [0.5, 1.0]")
    min_expansion_points = int(min_expansion_points)
    if min_expansion_points < 3:
        raise ValueError("ct_search_min_expansion_points must be at least 3")

    extension = _extension_only(
        baseline_points, expanded_points, tolerance=tolerance)
    rng = np.random.default_rng(seed)
    if len(extension) < min_expansion_points:
        sampled = _sample_rows(baseline_points, sample_size, rng)
        return sampled, {
            "baseline_sample_count": sample_size,
            "expansion_sample_count": 0,
            "expansion_available_count": int(len(extension)),
        }

    baseline_count = int(round(sample_size * baseline_ratio))
    baseline_count = min(max(baseline_count, 1), sample_size)
    expansion_count = sample_size - baseline_count
    sampled = np.concatenate((
        _sample_rows(baseline_points, baseline_count, rng),
        _sample_rows(extension, expansion_count, rng),
    ), axis=0)
    return sampled, {
        "baseline_sample_count": baseline_count,
        "expansion_sample_count": expansion_count,
        "expansion_available_count": int(len(extension)),
    }
