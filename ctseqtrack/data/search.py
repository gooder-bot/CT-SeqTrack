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


def estimate_ordered_trajectory(
    history_boxes,
    delta_t,
    valid_mask=None,
    max_speed=20.0,
    max_acceleration=8.0,
    max_displacement=12.0,
    acceleration_weight=0.5,
    require_recent_transition=False,
):
    """Causally extrapolate an ordered box history without learned weights.

    Histories are expected in the project's native recent-to-old order.  The
    estimate deliberately uses the two most recent valid transitions instead
    of a permutation-invariant mean/max pool: the newest velocity is the base
    state and the preceding velocity only contributes a bounded acceleration.
    This dependency-light twin is usable inside dataloader workers and during
    recursive evaluation before the network crop exists.
    """
    if len(history_boxes) < 2:
        return {"valid": False, "reason": "insufficient_history"}
    query_gap = _finite_positive(delta_t[0] if len(delta_t) else None, fallback=0.0)
    if query_gap <= 0:
        return {"valid": False, "reason": "invalid_query_gap"}

    pair_count = min(len(history_boxes) - 1, len(delta_t) - 1)
    transitions = []
    constraint_clipped = False
    for index in range(max(pair_count, 0)):
        if valid_mask is not None and not (
            bool(valid_mask[index]) and bool(valid_mask[index + 1])
        ):
            continue
        gap = _finite_positive(delta_t[index + 1], fallback=0.0)
        if gap <= 0:
            continue
        newer = np.asarray(history_boxes[index].center, dtype=np.float64)
        older = np.asarray(history_boxes[index + 1].center, dtype=np.float64)
        velocity = (newer - older) / gap
        planar_speed = float(np.linalg.norm(velocity[:2]))
        if planar_speed > float(max_speed):
            constraint_clipped = True
            velocity = velocity * (float(max_speed) / max(planar_speed, 1e-9))
        transitions.append((index, gap, velocity))
    if not transitions:
        return {"valid": False, "reason": "no_valid_transition"}
    if bool(require_recent_transition) and transitions[0][0] != 0:
        return {"valid": False, "reason": "invalid_recent_transition"}

    # ``transitions`` follows recent-to-old order, so entry zero is the state
    # closest to the query.  Older motion is used only to estimate acceleration.
    _, recent_gap, recent_velocity = transitions[0]
    acceleration = np.zeros(3, dtype=np.float64)
    if len(transitions) >= 2:
        _, older_gap, older_velocity = transitions[1]
        acceleration_gap = max(0.5 * (recent_gap + older_gap), 1e-6)
        acceleration = (recent_velocity - older_velocity) / acceleration_gap
        acceleration_norm = float(np.linalg.norm(acceleration[:2]))
        if acceleration_norm > float(max_acceleration):
            constraint_clipped = True
            acceleration *= float(max_acceleration) / max(acceleration_norm, 1e-9)

    displacement = (
        recent_velocity * query_gap
        + float(acceleration_weight) * 0.5 * acceleration * query_gap**2
    )
    planar_norm = float(np.linalg.norm(displacement[:2]))
    if planar_norm > float(max_displacement):
        constraint_clipped = True
        displacement *= float(max_displacement) / max(planar_norm, 1e-9)
        planar_norm = float(max_displacement)

    transition_gaps = np.asarray([item[1] for item in transitions], dtype=np.float64)
    nominal_gap = float(np.median(transition_gaps))
    gap_ratio = query_gap / max(nominal_gap, 1e-6)
    if len(transitions) >= 2:
        velocity_residual = recent_velocity - transitions[1][2]
        velocity_spread = float(np.linalg.norm(velocity_residual[:2]))
    else:
        velocity_spread = 0.0
    # A bounded, interpretable scale proxy.  It is not presented as a learned
    # posterior; it only controls how much pre-crop area is made available.
    sigma_parallel = min(
        4.0,
        0.25 + 0.25 * query_gap + 0.5 * velocity_spread * query_gap,
    )
    sigma_perpendicular = min(
        3.0,
        0.20 + 0.15 * query_gap + 0.25 * velocity_spread * query_gap,
    )
    return {
        "valid": True,
        "reason": "ok",
        "query_delta_t": query_gap,
        "nominal_delta_t": nominal_gap,
        "gap_ratio": gap_ratio,
        "velocity": recent_velocity,
        "acceleration": acceleration,
        "displacement_vector": displacement,
        "displacement": planar_norm,
        "constraint_clipped": constraint_clipped,
        "sigma_parallel": sigma_parallel,
        "sigma_perpendicular": sigma_perpendicular,
    }


def build_ordered_trajectory_search_box(
    history_boxes,
    delta_t,
    valid_mask=None,
    base_length=4.0,
    base_width=2.0,
    max_length=20.0,
    max_width=8.0,
    max_speed=20.0,
    max_acceleration=8.0,
    max_displacement=12.0,
    acceleration_weight=0.5,
    sigma_parallel_scale=2.0,
    sigma_perpendicular_scale=2.0,
    min_displacement=0.2,
    min_delta_t=0.75,
    min_gap_ratio=1.5,
    allow_normal_cadence=False,
    require_recent_transition=False,
):
    """Build an uncertainty-aware second crop from ordered causal history.

    The normal baseline crop is never replaced.  Unless
    ``allow_normal_cadence`` is requested, the second crop is activated only
    for an absolute long gap or a gap that is large relative to the observed
    history.  No current-frame ground truth or point statistic is consumed.
    """
    estimate = estimate_ordered_trajectory(
        history_boxes,
        delta_t,
        valid_mask=valid_mask,
        max_speed=max_speed,
        max_acceleration=max_acceleration,
        max_displacement=max_displacement,
        acceleration_weight=acceleration_weight,
        require_recent_transition=require_recent_transition,
    )
    if not estimate.get("valid", False):
        return None, estimate
    irregular = estimate["query_delta_t"] >= float(min_delta_t) or estimate[
        "gap_ratio"
    ] >= float(min_gap_ratio)
    if not allow_normal_cadence and not irregular:
        estimate.update({"valid": False, "reason": "normal_cadence"})
        return None, estimate
    if estimate["displacement"] < float(min_displacement):
        estimate.update({"valid": False, "reason": "stationary"})
        return None, estimate

    start = np.asarray(history_boxes[0].center, dtype=np.float64)
    displacement = estimate["displacement_vector"]
    end = start + displacement
    yaw = math.atan2(displacement[1], displacement[0])
    length = min(
        float(max_length),
        max(
            float(base_length),
            float(base_length)
            + estimate["displacement"]
            + float(sigma_parallel_scale) * estimate["sigma_parallel"],
        ),
    )
    width = min(
        float(max_width),
        max(
            float(base_width),
            float(base_width)
            + float(sigma_perpendicular_scale) * estimate["sigma_perpendicular"],
        ),
    )
    tube = copy.deepcopy(history_boxes[0])
    tube.center = 0.5 * (start + end)
    orientation_type = history_boxes[0].orientation.__class__
    tube.orientation = orientation_type(axis=[0, 0, 1], radians=yaw)
    tube.wlh = np.asarray(tube.wlh, dtype=np.float64).copy()
    tube.wlh[0] = width
    tube.wlh[1] = length
    estimate.update(
        {
            "valid": True,
            "reason": "ok",
            "length": length,
            "width": width,
        }
    )
    return tube, estimate


def _signed_box_yaw(box):
    """Return a nuScenes-compatible box yaw in radians."""
    return float(box.orientation.radians * box.orientation.axis[-1])


def _wrap_radians(angle):
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def build_trajectory_endpoint_search_box(
    history_boxes,
    delta_t,
    valid_mask=None,
    max_speed=20.0,
    max_acceleration=8.0,
    max_displacement=12.0,
    acceleration_weight=0.5,
    max_yaw_rate=math.pi / 2.0,
    min_displacement=0.2,
    require_recent_transition=False,
):
    """Build the compact B2-v2 crop at the predicted trajectory endpoint.

    The latest historical box is copied without changing its dimensions.  Its
    center and yaw are extrapolated causally from historical boxes only, so the
    same function can be used by training and recursive inference without
    access to the current-frame ground truth.
    """
    estimate = estimate_ordered_trajectory(
        history_boxes,
        delta_t,
        valid_mask=valid_mask,
        max_speed=max_speed,
        max_acceleration=max_acceleration,
        max_displacement=max_displacement,
        acceleration_weight=acceleration_weight,
        require_recent_transition=require_recent_transition,
    )
    if not estimate.get("valid", False):
        return None, estimate
    if estimate["displacement"] < float(min_displacement):
        estimate.update({"valid": False, "reason": "stationary"})
        return None, estimate

    pair_count = min(len(history_boxes) - 1, len(delta_t) - 1)
    yaw_rate = 0.0
    yaw_rate_valid = False
    for index in range(max(pair_count, 0)):
        if valid_mask is not None and not (
            bool(valid_mask[index]) and bool(valid_mask[index + 1])
        ):
            continue
        transition_gap = _finite_positive(delta_t[index + 1], fallback=0.0)
        if transition_gap <= 0:
            continue
        newer_yaw = _signed_box_yaw(history_boxes[index])
        older_yaw = _signed_box_yaw(history_boxes[index + 1])
        yaw_rate = _wrap_radians(newer_yaw - older_yaw) / transition_gap
        yaw_rate = float(np.clip(yaw_rate, -float(max_yaw_rate), float(max_yaw_rate)))
        yaw_rate_valid = True
        break

    latest = history_boxes[0]
    endpoint = copy.deepcopy(latest)
    endpoint_center = (
        np.asarray(latest.center, dtype=np.float64) + estimate["displacement_vector"]
    )
    endpoint.center = endpoint_center
    endpoint_yaw = _signed_box_yaw(latest)
    if yaw_rate_valid:
        endpoint_yaw += yaw_rate * estimate["query_delta_t"]
    orientation_type = latest.orientation.__class__
    endpoint.orientation = orientation_type(
        axis=[0, 0, 1], radians=_wrap_radians(endpoint_yaw)
    )
    # Explicitly preserve the latest box support.  In particular, uncertainty
    # and background point count never enlarge the crop geometry.
    endpoint.wlh = np.asarray(latest.wlh, dtype=np.float64).copy()
    estimate.update(
        {
            "valid": True,
            "reason": "ok",
            "yaw_rate": yaw_rate,
            "endpoint_center": endpoint_center.copy(),
            "endpoint_yaw": _wrap_radians(endpoint_yaw),
        }
    )
    return endpoint, estimate


def build_uncertainty_prior_tube(
    latest_box,
    mu_xy,
    sigma_parallel_perpendicular,
    velocity_xy,
    valid=True,
    coverage_scale=2.448,
    min_direction_speed=0.2,
    max_length=24.0,
    max_width=10.0,
    source_id=1,
    direction_xy=None,
):
    """Build a base-preserving prior tube from a learned local B1 state.

    ``mu_xy`` and ``velocity_xy`` are expressed in the latest recursive box
    coordinate system.  This function consumes no current-frame ground truth.
    The caller unions the returned support with the untouched B0 crop.
    """
    if latest_box is None or not bool(valid):
        return None, {
            "valid": False,
            "reason": "invalid_motion_prior",
            "source_id": int(source_id),
        }
    mu_xy = np.asarray(mu_xy, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma_parallel_perpendicular, dtype=np.float64).reshape(-1)
    velocity_xy = np.asarray(velocity_xy, dtype=np.float64).reshape(-1)
    if (
        mu_xy.size != 2
        or sigma.size != 2
        or velocity_xy.size != 2
        or not np.isfinite(mu_xy).all()
        or not np.isfinite(sigma).all()
        or not np.isfinite(velocity_xy).all()
    ):
        return None, {
            "valid": False,
            "reason": "non_finite_motion_prior",
            "source_id": int(source_id),
        }
    coverage_scale = max(float(coverage_scale), 0.0)
    sigma = np.maximum(sigma, 1e-3)
    latest_yaw = _signed_box_yaw(latest_box)
    cosine = math.cos(latest_yaw)
    sine = math.sin(latest_yaw)

    def local_to_world(vector):
        return np.asarray(
            (
                cosine * vector[0] - sine * vector[1],
                sine * vector[0] + cosine * vector[1],
            ),
            dtype=np.float64,
        )

    world_displacement = local_to_world(mu_xy)
    world_velocity = local_to_world(velocity_xy)
    displacement_norm = float(np.linalg.norm(world_displacement))
    speed = float(np.linalg.norm(world_velocity))
    if direction_xy is not None:
        local_direction = np.asarray(direction_xy, dtype=np.float64).reshape(-1)
        if local_direction.size != 2 or not np.isfinite(local_direction).all():
            return None, {
                "valid": False,
                "reason": "invalid_motion_direction",
                "source_id": int(source_id),
            }
        direction_norm = float(np.linalg.norm(local_direction))
        if direction_norm <= 1e-9:
            local_direction = np.asarray((1.0, 0.0), dtype=np.float64)
        else:
            local_direction = local_direction / direction_norm
        world_direction = local_to_world(local_direction)
        direction_yaw = math.atan2(world_direction[1], world_direction[0])
    elif speed >= float(min_direction_speed):
        direction_yaw = math.atan2(world_velocity[1], world_velocity[0])
    elif displacement_norm > 1e-6:
        direction_yaw = math.atan2(world_displacement[1], world_displacement[0])
    else:
        direction_yaw = latest_yaw

    size = np.asarray(latest_box.wlh, dtype=np.float64)
    object_width = max(float(size[0]), 1e-3)
    object_length = max(float(size[1]), 1e-3)
    parallel_half_extent = (
        0.5 * displacement_norm + 0.5 * object_length + coverage_scale * float(sigma[0])
    )
    perpendicular_half_extent = 0.5 * object_width + coverage_scale * float(sigma[1])
    requested_length = 2.0 * parallel_half_extent
    requested_width = 2.0 * perpendicular_half_extent
    length = min(float(max_length), requested_length)
    width = min(float(max_width), requested_width)
    truncated = length + 1e-9 < requested_length or width + 1e-9 < requested_width
    if not np.isfinite(length + width) or length <= 0 or width <= 0:
        return None, {
            "valid": False,
            "reason": "invalid_support_size",
            "source_id": int(source_id),
        }

    tube = copy.deepcopy(latest_box)
    center = np.asarray(latest_box.center, dtype=np.float64).copy()
    center[:2] += 0.5 * world_displacement
    tube.center = center
    orientation_type = latest_box.orientation.__class__
    tube.orientation = orientation_type(
        axis=[0, 0, 1], radians=_wrap_radians(direction_yaw)
    )
    tube.wlh = size.copy()
    tube.wlh[0] = width
    tube.wlh[1] = length
    return tube, {
        "valid": True,
        "reason": "ok",
        "source_id": int(source_id),
        "displacement": displacement_norm,
        "speed": speed,
        "sigma_parallel": float(sigma[0]),
        "sigma_perpendicular": float(sigma[1]),
        "coverage_scale": coverage_scale,
        "length": length,
        "width": width,
        "requested_length": requested_length,
        "requested_width": requested_width,
        "truncated": bool(truncated),
        "endpoint_center": (
            np.asarray(latest_box.center, dtype=np.float64)
            + np.r_[world_displacement, 0.0]
        ),
    }


def build_b1_uncertainty_support(
    latest_box,
    prediction,
    *,
    use_dynamic_sigma,
    fixed_margins=(2.0, 1.0),
    coverage_scale=2.448,
    standardized_residual_quantile=(1.0, 1.0),
    min_direction_speed=0.2,
    max_length=24.0,
    max_width=10.0
):
    """Pure B1-to-support contract shared by training and inference.

    ``prediction`` is a box-only B1 result in the latest recursive
    anchor coordinate system.  Fixed margins are represented as 95% residual
    half-extents and converted back to sigma units only for the common tube
    implementation.  The current-frame target is deliberately absent from
    this interface.
    """
    if not isinstance(prediction, dict):
        return None, {
            "valid": False,
            "reason": "missing_motion_prior",
            "source_id": 0,
        }
    scale = max(float(coverage_scale), 1e-6)
    residual_quantile = np.asarray(
        standardized_residual_quantile, dtype=np.float64
    ).reshape(-1)
    if (
        residual_quantile.size != 2
        or not np.isfinite(residual_quantile).all()
        or np.any(residual_quantile <= 0)
    ):
        return None, {
            "valid": False,
            "reason": "invalid_standardized_residual_quantile",
            "source_id": int(prediction.get("source_id", 1)),
        }
    if bool(use_dynamic_sigma):
        log_sigma = np.asarray(
            prediction.get("log_sigma_parallel_perp"), dtype=np.float64
        )
        if log_sigma.size != 2 or not np.isfinite(log_sigma).all():
            return None, {
                "valid": False,
                "reason": "invalid_dynamic_sigma",
                "source_id": int(prediction.get("source_id", 1)),
            }
        sigma = np.exp(log_sigma.reshape(2)) * residual_quantile
    else:
        margins = np.asarray(fixed_margins, dtype=np.float64).reshape(-1)
        if margins.size != 2 or not np.isfinite(margins).all():
            return None, {
                "valid": False,
                "reason": "invalid_fixed_margins",
                "source_id": int(prediction.get("source_id", 1)),
            }
        sigma = np.maximum(margins, 1e-3) / scale
    support, diagnostics = build_uncertainty_prior_tube(
        latest_box,
        prediction.get("mu_xy", (0.0, 0.0)),
        sigma,
        prediction.get("velocity_xy", (0.0, 0.0)),
        valid=bool(prediction.get("valid", False)),
        coverage_scale=scale,
        min_direction_speed=float(min_direction_speed),
        max_length=float(max_length),
        max_width=float(max_width),
        source_id=int(prediction.get("source_id", 1)),
        direction_xy=prediction.get("direction_xy"),
    )
    diagnostics["standardized_residual_quantile"] = residual_quantile.astype(
        np.float64
    ).tolist()
    return support, diagnostics


def resolve_b1_search_support(
    history_boxes,
    delta_t,
    valid_mask,
    *,
    prediction=None,
    use_b1_prepass=False,
    use_dynamic_sigma=False,
    fixed_margins=(2.0, 1.0),
    coverage_scale=2.448,
    standardized_residual_quantile=(1.0, 1.0),
    min_direction_speed=0.2,
    max_length=24.0,
    max_width=10.0,
    fallback_max_speed=20.0,
    fallback_max_acceleration=8.0,
    fallback_max_displacement=12.0,
    fallback_acceleration_weight=0.5,
    fallback_max_yaw_rate=math.pi / 2.0,
    fallback_min_displacement=0.2,
    fallback_require_recent_transition=False
):
    """Resolve the identical B1/fallback/base-only support in all paths."""
    query_delta_t = float(delta_t[0]) if len(delta_t) else 0.0
    if (
        bool(use_b1_prepass)
        and isinstance(prediction, dict)
        and bool(prediction.get("valid", False))
    ):
        support, diagnostics = build_b1_uncertainty_support(
            history_boxes[0],
            prediction,
            use_dynamic_sigma=use_dynamic_sigma,
            fixed_margins=fixed_margins,
            coverage_scale=coverage_scale,
            standardized_residual_quantile=(standardized_residual_quantile),
            min_direction_speed=min_direction_speed,
            max_length=max_length,
            max_width=max_width,
        )
        if support is not None:
            diagnostics.update(
                {
                    "query_delta_t": float(
                        prediction.get("current_delta_t", query_delta_t)
                    ),
                    "gap_ratio": float(prediction.get("gap_ratio", 1.0)),
                    "prior_source": "b1",
                }
            )
            return support, diagnostics
    support, diagnostics = build_trajectory_endpoint_search_box(
        history_boxes,
        delta_t,
        valid_mask=valid_mask,
        max_speed=fallback_max_speed,
        max_acceleration=fallback_max_acceleration,
        max_displacement=fallback_max_displacement,
        acceleration_weight=fallback_acceleration_weight,
        max_yaw_rate=fallback_max_yaw_rate,
        min_displacement=fallback_min_displacement,
        require_recent_transition=fallback_require_recent_transition,
    )
    diagnostics["prior_source"] = "fallback_cv" if support is not None else "base_only"
    diagnostics["source_id"] = 2 if support is not None else 0
    diagnostics.setdefault("query_delta_t", query_delta_t)
    diagnostics.setdefault("gap_ratio", 1.0)
    return support, diagnostics


def resolve_joint_search_geometry(history_boxes, delta_t, valid_mask, **kwargs):
    """Return endpoint and swept-tube supports from one causal prior.

    B1-valid rows use its learned mean for both geometries.  Invalid B1 rows
    reuse the same constrained kinematic estimate for endpoint and tube.  No
    current-frame annotation is accepted by this interface.
    """
    endpoint_or_tube, diagnostics = resolve_b1_search_support(
        history_boxes, delta_t, valid_mask, **kwargs
    )
    if endpoint_or_tube is None:
        return None, None, diagnostics
    latest = history_boxes[0]
    prior_source = diagnostics.get("prior_source")
    fixed_margins = np.asarray(
        kwargs.get("fixed_margins", (2.0, 1.0)), dtype=np.float64
    ).reshape(-1)
    if fixed_margins.size != 2 or not np.isfinite(fixed_margins).all():
        raise ValueError("joint fixed margins must contain two finite values")

    if prior_source == "b1":
        tube = endpoint_or_tube
        endpoint = copy.deepcopy(latest)
        endpoint.center = np.asarray(
            diagnostics["endpoint_center"], dtype=np.float64
        ).copy()
        endpoint.orientation = copy.deepcopy(tube.orientation)
        endpoint.wlh = np.asarray(latest.wlh, dtype=np.float64).copy()
        support_scale = max(
            float(
                diagnostics.get("coverage_scale", kwargs.get("coverage_scale", 2.448))
            ),
            0.0,
        )
        parallel_margin = support_scale * float(
            diagnostics.get(
                "sigma_parallel", fixed_margins[0] / max(support_scale, 1e-6)
            )
        )
        perpendicular_margin = support_scale * float(
            diagnostics.get(
                "sigma_perpendicular", fixed_margins[1] / max(support_scale, 1e-6)
            )
        )
        endpoint.wlh[0] = min(
            float(kwargs.get("max_width", 10.0)),
            float(endpoint.wlh[0]) + 2.0 * perpendicular_margin,
        )
        endpoint.wlh[1] = min(
            float(kwargs.get("max_length", 24.0)),
            float(endpoint.wlh[1]) + 2.0 * parallel_margin,
        )
    else:
        endpoint = endpoint_or_tube
        displacement_world = np.asarray(endpoint.center, dtype=np.float64) - np.asarray(
            latest.center, dtype=np.float64
        )
        rotation_inv = np.asarray(latest.rotation_matrix, dtype=np.float64).T
        local_mu = (rotation_inv @ displacement_world)[:2]
        query_dt = max(float(diagnostics.get("query_delta_t", 0.0)), 1e-3)
        local_velocity = local_mu / query_dt
        scale = max(float(kwargs.get("coverage_scale", 2.448)), 1e-6)
        tube, tube_diagnostics = build_uncertainty_prior_tube(
            latest,
            local_mu,
            np.maximum(fixed_margins, 1e-3) / scale,
            local_velocity,
            valid=True,
            coverage_scale=scale,
            min_direction_speed=float(kwargs.get("min_direction_speed", 0.2)),
            max_length=float(kwargs.get("max_length", 24.0)),
            max_width=float(kwargs.get("max_width", 10.0)),
            source_id=int(diagnostics.get("source_id", 2)),
            direction_xy=local_velocity,
        )
        if tube is None:
            return (
                endpoint,
                None,
                {
                    **diagnostics,
                    "valid": False,
                    "reason": tube_diagnostics.get("reason", "invalid_fallback_tube"),
                },
            )
        diagnostics = {**tube_diagnostics, **diagnostics}
    diagnostics.update(
        {
            "valid": True,
            "endpoint_support_center": np.asarray(
                endpoint.center, dtype=np.float64
            ).copy(),
            "tube_support_center": np.asarray(tube.center, dtype=np.float64).copy(),
        }
    )
    return endpoint, tube, diagnostics


def _sample_rows(points, sample_size, rng):
    points = np.asarray(points)
    if sample_size <= 0:
        return points[:0]
    if points.ndim != 2:
        raise ValueError("search points must be a [N, C] array")
    if len(points) <= 2:
        return np.zeros((sample_size, points.shape[1]), dtype=np.float32)
    indices = rng.choice(
        len(points), size=int(sample_size), replace=int(sample_size) > len(points)
    )
    return points[indices]


def _extension_only(primary, expanded, tolerance=1e-6):
    if len(expanded) == 0:
        return expanded
    if len(primary) == 0:
        return expanded
    tolerance = _finite_positive(tolerance, fallback=1e-6)
    coordinate_dims = min(primary.shape[1], expanded.shape[1], 3)
    primary_keys = {
        tuple(row)
        for row in np.rint(primary[:, :coordinate_dims] / tolerance).astype(np.int64)
    }
    expanded_keys = np.rint(expanded[:, :coordinate_dims] / tolerance).astype(np.int64)
    keep = np.fromiter(
        (tuple(row) not in primary_keys for row in expanded_keys),
        dtype=bool,
        count=len(expanded_keys),
    )
    return expanded[keep]


def sample_source_aware_endpoint_points(
    baseline_points,
    endpoint_points,
    sample_size=128,
    extension_quota=64,
    min_points=3,
    seed=None,
    tolerance=1e-6,
):
    """Sample a compact endpoint crop without starving overlap evidence.

    Endpoint returns are deduplicated by XYZ and labelled according to whether
    the identical return also belongs to the baseline crop.  Extension-only
    points receive a reserved quota, while stable overlap points may fill the
    rest of the independent branch.  Sampling is local, without replacement,
    and therefore cannot consume or perturb the B0 sampling RNG.

    Returns:
        points: ``[sample_size, C]`` float32 array with zero padding.
        valid_mask: ``[sample_size]`` float32 mask.
        source: ``[sample_size]`` int64 array (0=overlap, 1=extension).
        diagnostics: availability and selected-source counts.
    """
    baseline_points = np.asarray(baseline_points)
    endpoint_points = np.asarray(endpoint_points)
    sample_size = int(sample_size)
    extension_quota = int(extension_quota)
    min_points = int(min_points)
    if sample_size <= 0:
        raise ValueError("endpoint point_count must be positive")
    if not 0 <= extension_quota <= sample_size:
        raise ValueError("endpoint extension_quota must be in [0, point_count]")
    if min_points < 3:
        raise ValueError("endpoint min_points must be at least 3")
    if baseline_points.ndim != 2 or endpoint_points.ndim != 2:
        raise ValueError("search point arrays must have shape [N, C]")
    if baseline_points.shape[1] != endpoint_points.shape[1]:
        raise ValueError("baseline and endpoint points must share channels")

    tolerance = _finite_positive(tolerance, fallback=1e-6)
    coordinate_dims = min(baseline_points.shape[1], endpoint_points.shape[1], 3)
    baseline_keys = {
        tuple(row)
        for row in np.rint(baseline_points[:, :coordinate_dims] / tolerance).astype(
            np.int64
        )
    }

    if len(endpoint_points):
        endpoint_keys = np.rint(
            endpoint_points[:, :coordinate_dims] / tolerance
        ).astype(np.int64)
        _, unique_indices = np.unique(endpoint_keys, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        unique_points = endpoint_points[unique_indices]
        unique_keys = endpoint_keys[unique_indices]
        is_extension = np.fromiter(
            (tuple(row) not in baseline_keys for row in unique_keys),
            dtype=bool,
            count=len(unique_keys),
        )
    else:
        unique_points = endpoint_points
        is_extension = np.zeros((0,), dtype=bool)

    extension_indices = np.flatnonzero(is_extension)
    overlap_indices = np.flatnonzero(~is_extension)
    available_count = int(len(unique_points))
    extension_count = int(len(extension_indices))
    overlap_count = int(len(overlap_indices))

    output = np.zeros((sample_size, baseline_points.shape[1]), dtype=np.float32)
    valid_mask = np.zeros((sample_size,), dtype=np.float32)
    source = np.zeros((sample_size,), dtype=np.int64)
    inactive = {
        "active": False,
        "sample_count": 0,
        "available_count": available_count,
        "extension_count": extension_count,
        "overlap_count": overlap_count,
        "selected_extension_count": 0,
        "selected_overlap_count": 0,
    }
    if available_count < min_points:
        return output, valid_mask, source, inactive

    rng = np.random.default_rng(seed)
    extension_order = rng.permutation(extension_indices)
    overlap_order = rng.permutation(overlap_indices)
    initial_extension = min(extension_quota, len(extension_order), sample_size)
    selected_parts = [extension_order[:initial_extension]]
    selected_sources = [np.ones((initial_extension,), dtype=np.int64)]

    remaining = sample_size - initial_extension
    selected_overlap = min(remaining, len(overlap_order))
    if selected_overlap:
        selected_parts.append(overlap_order[:selected_overlap])
        selected_sources.append(np.zeros((selected_overlap,), dtype=np.int64))
    remaining -= selected_overlap

    extra_extension = min(remaining, len(extension_order) - initial_extension)
    if extra_extension:
        selected_parts.append(
            extension_order[initial_extension : initial_extension + extra_extension]
        )
        selected_sources.append(np.ones((extra_extension,), dtype=np.int64))

    selected_indices = np.concatenate(selected_parts)
    selected_source = np.concatenate(selected_sources)
    sample_count = int(len(selected_indices))
    output[:sample_count] = unique_points[selected_indices].astype(
        np.float32, copy=False
    )
    valid_mask[:sample_count] = 1.0
    source[:sample_count] = selected_source
    selected_extension_count = int(selected_source.sum())
    return (
        output,
        valid_mask,
        source,
        {
            "active": True,
            "sample_count": sample_count,
            "available_count": available_count,
            "extension_count": extension_count,
            "overlap_count": overlap_count,
            "selected_extension_count": selected_extension_count,
            "selected_overlap_count": sample_count - selected_extension_count,
        },
    )


def sample_joint_novel_extensions(
    baseline_points,
    endpoint_points,
    tube_points,
    endpoint_quota=128,
    tube_quota=128,
    seed=None,
    tolerance=1e-6,
):
    """Build an extension-only endpoint/tube pool without duplicate returns.

    The B0 crop is used only as the exclusion set.  A LiDAR return present in
    both acquisition regions is retained once and receives source id ``3``;
    endpoint-only and tube-only returns receive ids ``1`` and ``2``.  The
    output is padded to ``endpoint_quota + tube_quota`` and is sampled by a
    branch-local Generator, so it cannot perturb the B0 sampling stream.
    """
    baseline_points = np.asarray(baseline_points)
    endpoint_points = np.asarray(endpoint_points)
    tube_points = np.asarray(tube_points)
    endpoint_quota = int(endpoint_quota)
    tube_quota = int(tube_quota)
    if endpoint_quota <= 0 or tube_quota <= 0:
        raise ValueError("endpoint and tube quotas must be positive")
    if any(
        points.ndim != 2 for points in (baseline_points, endpoint_points, tube_points)
    ):
        raise ValueError("search point arrays must have shape [N, C]")
    if not (
        baseline_points.shape[1] == endpoint_points.shape[1] == tube_points.shape[1]
    ):
        raise ValueError("baseline, endpoint and tube points must share channels")

    tolerance = _finite_positive(tolerance, fallback=1e-6)
    coordinate_dims = min(baseline_points.shape[1], 3)

    def keys(points):
        return np.rint(points[:, :coordinate_dims] / tolerance).astype(np.int64)

    baseline_keys = {tuple(row) for row in keys(baseline_points)}
    merged = {}
    branch_available = {1: 0, 2: 0}
    for branch_id, points in ((1, endpoint_points), (2, tube_points)):
        seen_in_branch = set()
        for point, key_row in zip(points, keys(points)):
            key = tuple(key_row)
            if key in baseline_keys or key in seen_in_branch:
                continue
            seen_in_branch.add(key)
            branch_available[branch_id] += 1
            if key in merged:
                merged[key] = (merged[key][0], 3)
            else:
                merged[key] = (point, branch_id)

    endpoint_candidates = [value for value in merged.values() if value[1] in (1, 3)]
    tube_candidates = [value for value in merged.values() if value[1] == 2]
    rng = np.random.default_rng(seed)

    def select(values, quota):
        if len(values) <= quota:
            return list(values)
        order = rng.choice(len(values), size=quota, replace=False)
        return [values[int(index)] for index in order]

    selected_endpoint = select(endpoint_candidates, endpoint_quota)
    selected_tube = select(tube_candidates, tube_quota)
    selected = selected_endpoint + selected_tube
    output_size = endpoint_quota + tube_quota
    output = np.zeros((output_size, baseline_points.shape[1]), dtype=np.float32)
    valid_mask = np.zeros((output_size,), dtype=np.float32)
    source = np.zeros((output_size,), dtype=np.int64)
    for index, (point, source_id) in enumerate(selected):
        output[index] = point.astype(np.float32, copy=False)
        valid_mask[index] = 1.0
        source[index] = int(source_id)

    source_counts = {
        source_id: int(sum(value[1] == source_id for value in merged.values()))
        for source_id in (1, 2, 3)
    }
    return (
        output,
        valid_mask,
        source,
        {
            "active": bool(selected),
            "sample_count": int(len(selected)),
            "available_count": int(len(merged)),
            "endpoint_available_count": int(branch_available[1]),
            "tube_available_count": int(branch_available[2]),
            "endpoint_only_count": source_counts[1],
            "tube_only_count": source_counts[2],
            "both_count": source_counts[3],
            "selected_endpoint_count": int(len(selected_endpoint)),
            "selected_tube_count": int(len(selected_tube)),
            # Internal sampler diagnostic.  Callers must not expose raw points as
            # model input; it is used only to count pre-sampling target support.
            "_pool_points": np.asarray(
                [value[0] for value in merged.values()], dtype=np.float32
            ).reshape(-1, baseline_points.shape[1]),
        },
    )


def combined_search_support_statistics(
    point_arrays, valid_masks, support_sources, voxel_size=0.2
):
    """Summarize real endpoint/tube support after sampling.

    ``support_sources`` uses the stable contract 0=already present in the B0
    crop and 1=expansion-only.  Extension returns are deduplicated jointly
    across endpoint and tube, so the same LiDAR return cannot make Search look
    useful twice merely because it was covered by both crops.
    """
    if not (len(point_arrays) == len(valid_masks) == len(support_sources)):
        raise ValueError("search support arrays/masks/sources must align")
    valid_points = []
    extension_points = []
    for points, mask, source in zip(point_arrays, valid_masks, support_sources):
        points = np.asarray(points)
        mask = np.asarray(mask).reshape(-1) > 0
        source = np.asarray(source).reshape(-1)
        if points.ndim != 2 or len(points) != len(mask):
            raise ValueError("search support points must have shape [N,C]")
        if len(source) != len(mask):
            raise ValueError("search support source must match points")
        finite = np.isfinite(points[:, : min(3, points.shape[1])]).all(axis=1)
        selected = mask & finite
        if selected.any():
            valid_points.append(points[selected, :3])
        extension = selected & (source == 1)
        if extension.any():
            extension_points.append(points[extension, :3])

    valid = (
        np.concatenate(valid_points, axis=0)
        if valid_points
        else np.empty((0, 3), dtype=np.float32)
    )
    extension = (
        np.concatenate(extension_points, axis=0)
        if extension_points
        else np.empty((0, 3), dtype=np.float32)
    )
    if len(valid):
        valid_keys = np.rint(valid / 1e-6).astype(np.int64)
        total_count = int(np.unique(valid_keys, axis=0).shape[0])
    else:
        total_count = 0
    if len(extension):
        exact_keys = np.rint(extension / 1e-6).astype(np.int64)
        extension_count = int(np.unique(exact_keys, axis=0).shape[0])
        safe_voxel = max(float(voxel_size), 1e-6)
        voxel_keys = np.floor(extension / safe_voxel).astype(np.int64)
        extension_voxels = int(np.unique(voxel_keys, axis=0).shape[0])
    else:
        extension_count = 0
        extension_voxels = 0
    return {
        "total_count": total_count,
        "extension_count": extension_count,
        "extension_voxels": extension_voxels,
    }


def useful_search_coverage_need(
    query_delta_t,
    gap_ratio,
    endpoint_xy,
    reference_wlh,
    baseline_point_count,
    min_delta_t=0.75,
    min_gap_ratio=1.5,
    min_endpoint_ratio=0.6,
    sparse_base_points=64,
    bb_scale=1.0,
    bb_offset=0.0,
):
    """Test whether a second crop can cover evidence missing from B0."""
    endpoint = np.asarray(endpoint_xy, dtype=np.float64).reshape(2)
    size = np.asarray(reference_wlh, dtype=np.float64).reshape(-1)
    if size.size < 2 or not np.isfinite(endpoint).all():
        return False, 0.0
    # nuScenes Box.wlh is width/length/height.  The local x/y axes use the
    # corresponding half extents after the ordinary B0 crop expansion.
    half_extent = np.maximum(0.5 * size[:2] * float(bb_scale) + float(bb_offset), 1e-3)
    endpoint_ratio = float(np.max(np.abs(endpoint) / half_extent))
    irregular = float(query_delta_t) >= float(min_delta_t) or float(gap_ratio) >= float(
        min_gap_ratio
    )
    needed = (
        irregular
        or endpoint_ratio >= float(min_endpoint_ratio)
        or int(baseline_point_count) < int(sparse_base_points)
    )
    return bool(needed), endpoint_ratio
