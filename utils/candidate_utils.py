"""Physically consistent candidate-trajectory helpers for M1.

The legacy SeqTrack3D sampler perturbs every historical box independently in
that box's local coordinate system.  That remains the default for backward
compatibility.  The ``shared_se2`` mode implemented here applies one rigid
world-frame SE(2) transform to the complete historical candidate trajectory.
"""

import copy

import numpy as np
from pyquaternion import Quaternion

from utils.ct_history import (
    build_ct_history_offsets,
    normalize_ct_history_training_mode,
)


_CANDIDATE_TRAJECTORY_MODES = {
    "independent": "independent",
    "legacy": "independent",
    "per_frame": "independent",
    "shared": "shared_se2",
    "shared_se2": "shared_se2",
    "shared_world_se2": "shared_se2",
}


def normalize_candidate_trajectory_mode(mode):
    """Normalize the candidate augmentation mode without changing defaults."""
    key = str(mode if mode is not None else "independent").lower().replace("-", "_")
    if key not in _CANDIDATE_TRAJECTORY_MODES:
        raise ValueError(
            "candidate_trajectory_mode must be 'independent' or 'shared_se2'")
    return _CANDIDATE_TRAJECTORY_MODES[key]


def box_yaw(box, degrees=False):
    """Return signed z-axis yaw for a nuScenes-compatible Box."""
    angle = box.orientation.degrees if degrees else box.orientation.radians
    return float(angle * box.orientation.axis[-1])


def wrap_yaw(angle, degrees=False):
    period = 360.0 if degrees else 2.0 * np.pi
    half = period / 2.0
    return float((float(angle) + half) % period - half)


def anchor_relative_trajectory_targets(
        current_box, anchor_box, current_delta_t, degrees=False, eps=1e-3):
    """Express the next-box target in the actual crop-anchor coordinates.

    The online tracker predicts from its latest estimated box, not from the
    unavailable latest ground-truth box.  Training candidates therefore need
    a target that includes both physical motion and correction of the sampled
    anchor error.  Returning both displacement and rate keeps every ordered
    trajectory loss in this same local frame.
    """
    current_delta_t = max(float(current_delta_t), float(eps))
    world_displacement = (
        np.asarray(current_box.center, dtype=np.float64)
        - np.asarray(anchor_box.center, dtype=np.float64)
    )
    local_displacement = (
        np.asarray(anchor_box.rotation_matrix, dtype=np.float64).T
        @ world_displacement
    )
    yaw_displacement = wrap_yaw(
        box_yaw(current_box, degrees) - box_yaw(anchor_box, degrees),
        degrees,
    )
    trajectory_displacement = np.concatenate((
        local_displacement,
        np.asarray([yaw_displacement], dtype=np.float64),
    )).astype(np.float32)
    velocity = (local_displacement / current_delta_t).astype(np.float32)
    return trajectory_displacement, velocity


def physical_motion_targets(
        current_box, latest_history_box, anchor_box, current_delta_t,
        degrees=False, eps=1e-3):
    """Return candidate-translation-independent physical xy motion.

    ``anchor_box`` defines only the coordinate axes.  The displacement origin
    remains the latest ground-truth history box, so candidate translation error
    cannot leak into a target that is predicted from relative history alone.
    This keeps the trajectory prior responsible for physical motion while the
    observation branch remains responsible for correcting the online anchor.
    """
    current_delta_t = max(float(current_delta_t), float(eps))
    world_displacement = (
        np.asarray(current_box.center, dtype=np.float64)
        - np.asarray(latest_history_box.center, dtype=np.float64)
    )
    local_displacement = (
        np.asarray(anchor_box.rotation_matrix, dtype=np.float64).T
        @ world_displacement
    )
    displacement_xy = local_displacement[:2].astype(np.float32)
    velocity_xy = (displacement_xy / current_delta_t).astype(np.float32)
    return displacement_xy, velocity_xy


def reexpress_motion_prediction(
        prediction, source_anchor, target_anchor):
    """Re-express an unbatched B1 prediction in a B2 crop frame.

    Endpoint-like quantities receive the full local-frame SE(2) transform;
    vectors and covariances are rotated only.  Returning the original object
    for an exact identity transform keeps the canonical online path bitwise
    unchanged.
    """
    if not isinstance(prediction, dict):
        raise TypeError("motion prediction must be a dictionary")
    source_center = np.asarray(source_anchor.center[:2], dtype=np.float64)
    target_center = np.asarray(target_anchor.center[:2], dtype=np.float64)
    source_rotation = np.asarray(
        source_anchor.rotation_matrix[:2, :2], dtype=np.float64)
    target_rotation = np.asarray(
        target_anchor.rotation_matrix[:2, :2], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (
            source_center, target_center,
            source_rotation, target_rotation)):
        raise ValueError("motion prediction anchors must be finite")
    if (np.array_equal(source_center, target_center)
            and np.array_equal(source_rotation, target_rotation)):
        return prediction
    output = dict(prediction)
    relative_rotation = target_rotation.T @ source_rotation

    def transform_endpoint(value):
        array = np.asarray(value)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError("motion prediction endpoint must be finite [2]")
        world = source_center + source_rotation @ array.astype(np.float64)
        return (target_rotation.T @ (world - target_center)).astype(
            array.dtype, copy=False)

    def rotate_vector(value):
        array = np.asarray(value)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError("motion prediction vector must be finite [2]")
        return (relative_rotation @ array.astype(np.float64)).astype(
            array.dtype, copy=False)

    for key in ("mu_xy", "prior_xy", "kinematic_prior_xy"):
        if key in output:
            output[key] = transform_endpoint(output[key])
    for key in ("direction_xy", "velocity_xy", "basis_velocity_xy"):
        if key in output:
            output[key] = rotate_vector(output[key])
    if "covariance_xy" in output:
        covariance = np.asarray(output["covariance_xy"])
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise ValueError(
                "motion prediction covariance must be finite [2,2]")
        output["covariance_xy"] = (
            relative_rotation @ covariance.astype(np.float64)
            @ relative_rotation.T).astype(covariance.dtype, copy=False)
    return output


def yaw_rotation_matrix(angle, degrees=False):
    angle_rad = np.deg2rad(angle) if degrees else float(angle)
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.asarray(
        [[cosine, -sine, 0.0],
         [sine, cosine, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def validate_shared_se2_transform(transform):
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"shared SE(2) transform must have shape (3,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("shared SE(2) transform contains non-finite values")
    return value


def shared_se2_world_translation(anchor_box, transform, degrees=False):
    """Convert anchor-local ``dx,dy`` into the shared world translation."""
    transform = validate_shared_se2_transform(transform)
    anchor_rotation = yaw_rotation_matrix(box_yaw(anchor_box, degrees), degrees)
    return anchor_rotation @ np.asarray([transform[0], transform[1], 0.0])


def apply_shared_se2_to_box(box, anchor_box, transform, degrees=False):
    """Apply one world SE(2) transform around ``anchor_box`` to ``box``.

    ``dx,dy`` are expressed in the latest-history anchor frame. ``dtheta`` is
    expressed in degrees or radians according to ``degrees``.  Z, box size and
    timestamps are unchanged.
    """
    transform = validate_shared_se2_transform(transform)
    if np.array_equal(transform, np.zeros(3, dtype=transform.dtype)):
        return copy.deepcopy(box)
    dtheta = float(transform[2])
    rotation = yaw_rotation_matrix(dtheta, degrees)
    world_translation = shared_se2_world_translation(anchor_box, transform, degrees)
    anchor_center = np.asarray(anchor_box.center, dtype=np.float64)
    center = np.asarray(box.center, dtype=np.float64)

    transformed = copy.deepcopy(box)
    transformed.center = (
        anchor_center + rotation @ (center - anchor_center) + world_translation
    )
    delta_rotation = (
        Quaternion(axis=[0, 0, 1], degrees=dtheta)
        if degrees else Quaternion(axis=[0, 0, 1], radians=dtheta)
    )
    transformed.orientation = delta_rotation * transformed.orientation
    if hasattr(transformed, "velocity"):
        transformed.velocity = rotation @ np.asarray(transformed.velocity, dtype=np.float64)
    return transformed


def apply_shared_se2_to_boxes(boxes, transform, degrees=False, anchor_index=0):
    """Apply a single rigid transform to an entire historical trajectory."""
    if not boxes:
        raise ValueError("shared SE(2) requires at least one historical box")
    if not 0 <= int(anchor_index) < len(boxes):
        raise ValueError("anchor_index is outside the historical box sequence")
    anchor_box = boxes[int(anchor_index)]
    return [
        apply_shared_se2_to_box(box, anchor_box, transform, degrees=degrees)
        for box in boxes
    ]


def boxes_to_anchor_parameters(boxes, anchor_box, degrees=False):
    """Express boxes as ``[x,y,z,yaw]`` in a common anchor coordinate frame."""
    anchor_center = np.asarray(anchor_box.center, dtype=np.float64)
    anchor_rotation_inv = yaw_rotation_matrix(
        -box_yaw(anchor_box, degrees), degrees)
    anchor_yaw = box_yaw(anchor_box, degrees)
    parameters = []
    for box in boxes:
        local_center = anchor_rotation_inv @ (
            np.asarray(box.center, dtype=np.float64) - anchor_center)
        local_yaw = wrap_yaw(box_yaw(box, degrees) - anchor_yaw, degrees)
        parameters.append(np.concatenate((local_center, [local_yaw])))
    return np.asarray(parameters, dtype=np.float32)


def build_b1_physical_contract(
        current_box, ground_truth_history, recursive_history,
        current_delta_t, degrees=False, eps=1e-3):
    """Build the candidate-invariant B1 input/label pair.

    Recursive boxes define the input and coordinate axes.  The newest GT
    history box defines only the physical displacement origin.  Candidate
    recovery boxes are intentionally absent from this interface.
    """
    ground_truth_history = list(ground_truth_history)
    recursive_history = list(recursive_history)
    if not ground_truth_history or not recursive_history:
        raise ValueError("B1 physical contract requires non-empty histories")
    if len(ground_truth_history) != len(recursive_history):
        raise ValueError("GT and recursive B1 histories must align")
    anchor = recursive_history[0]
    ref_boxs = boxes_to_anchor_parameters(
        recursive_history, anchor, degrees=degrees)
    target_xy, velocity_xy = physical_motion_targets(
        current_box, ground_truth_history[0], anchor, current_delta_t,
        degrees=degrees, eps=eps)
    return {
        "ref_boxs": ref_boxs,
        "target_xy": target_xy,
        "velocity_xy": velocity_xy,
        "anchor": anchor,
    }


def equivalent_local_offsets(boxes, transformed_boxes, degrees=False):
    """Return per-box local offsets equivalent to an already applied transform.

    This is audit metadata only.  The shared path never replays these offsets
    independently through ``getOffsetBB``.
    """
    if len(boxes) != len(transformed_boxes):
        raise ValueError("boxes and transformed_boxes must have the same length")
    offsets = []
    for box, transformed in zip(boxes, transformed_boxes):
        rotation_inv = yaw_rotation_matrix(-box_yaw(box, degrees), degrees)
        local_translation = rotation_inv @ (
            np.asarray(transformed.center) - np.asarray(box.center))
        delta_yaw = wrap_yaw(
            box_yaw(transformed, degrees) - box_yaw(box, degrees), degrees)
        offsets.append([local_translation[0], local_translation[1], delta_yaw])
    return np.asarray(offsets, dtype=np.float32)


def apply_local_candidate_offset(box, offset, degrees=False):
    """Apply an in-range SeqTrack3D-style local ``dx,dy,dyaw`` offset."""
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("local candidate offset must be a finite [3] vector")
    if np.array_equal(offset, np.zeros(3, dtype=offset.dtype)):
        return copy.deepcopy(box)

    transformed = copy.deepcopy(box)
    center = np.asarray(box.center, dtype=np.float64)
    rotation = Quaternion(matrix=box.rotation_matrix)
    transformed.translate(-center)
    transformed.rotate(rotation.inverse)
    delta_rotation = (
        Quaternion(axis=[0, 0, 1], degrees=float(offset[2]))
        if degrees else Quaternion(axis=[0, 0, 1], radians=float(offset[2]))
    )
    transformed.rotate(delta_rotation)
    transformed.translate(np.asarray([offset[0], offset[1], 0.0]))
    transformed.rotate(rotation)
    transformed.translate(center)
    return transformed


def build_ct_training_histories(
        canonical_boxes,
        candidate_boxes,
        candidate_offsets,
        candidate_id,
        candidate_trajectory_mode,
        training_mode="canonical",
        correlation=0.75,
        recursive_error_scale=1.0,
        degrees=False):
    """Build motion/search histories without changing canonical supervision.

    Motion history stays anchored to the latest canonical box, so its physical
    displacement target remains in the correct coordinate frame.  Search
    history stays anchored to the actual candidate crop, matching the recursive
    predicted-box path used at evaluation.
    """
    if not canonical_boxes:
        raise ValueError("CT history construction requires at least one box")
    if len(canonical_boxes) != len(candidate_boxes):
        raise ValueError("canonical and candidate histories must have equal length")

    candidate_trajectory_mode = normalize_candidate_trajectory_mode(
        candidate_trajectory_mode)
    candidate_offsets = np.asarray(candidate_offsets, dtype=np.float32)
    if candidate_offsets.shape != (len(canonical_boxes), 3):
        raise ValueError(
            "candidate offsets must match the historical box sequence")
    motion_offsets, search_offsets = build_ct_history_offsets(
        candidate_offsets,
        candidate_id,
        candidate_trajectory_mode,
        training_mode=normalize_ct_history_training_mode(training_mode),
        correlation=correlation,
        recursive_error_scale=recursive_error_scale,
    )
    motion_boxes = [
        apply_local_candidate_offset(box, offset, degrees=degrees)
        for box, offset in zip(canonical_boxes, motion_offsets)
    ]
    if search_offsets is None:
        search_boxes = [copy.deepcopy(box) for box in candidate_boxes]
    else:
        search_boxes = [
            apply_local_candidate_offset(box, offset, degrees=degrees)
            for box, offset in zip(canonical_boxes, search_offsets)
        ]
    return motion_boxes, search_boxes


def canonical_dynamics_targets(previous_boxes, current_box, current_delta_t,
                               degrees=False, eps=1e-3):
    """Build current displacement/velocity labels from the canonical GT path.

    Targets are computed before candidate perturbation and expressed in the
    most recent historical GT box frame.  Candidate augmentation therefore
    cannot manufacture velocity or acceleration supervision.
    """
    if not previous_boxes:
        raise ValueError("canonical dynamics targets require history")
    current_delta_t = max(float(current_delta_t), float(eps))
    recent = previous_boxes[0]
    rotation_inv = yaw_rotation_matrix(-box_yaw(recent, degrees), degrees)
    displacement = rotation_inv @ (
        np.asarray(current_box.center, dtype=np.float64)
        - np.asarray(recent.center, dtype=np.float64)
    )
    displacement = displacement.astype(np.float32)
    velocity = (displacement / current_delta_t).astype(np.float32)
    return displacement, velocity
