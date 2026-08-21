"""Formal four-view B0 geometry checks with an optional real-loader pass."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.candidate_utils import (  # noqa: E402
    apply_shared_se2_to_box,
    apply_shared_se2_to_boxes,
    boxes_to_anchor_parameters,
    canonical_dynamics_targets,
    equivalent_local_offsets,
    normalize_candidate_trajectory_mode,
)
from utils.config import load_yaml_config  # noqa: E402


class Box:
    """Minimal nuScenes-compatible box used to keep this check dataset-free."""

    def __init__(self, center, size, orientation):
        self.center = np.asarray(center, dtype=np.float64)
        self.wlh = np.asarray(size, dtype=np.float64)
        self.orientation = orientation
        self.velocity = np.zeros(3, dtype=np.float64)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


def apply_local_offset(box, offset, degrees):
    transformed = Box(box.center.copy(), box.wlh.copy(), box.orientation)
    translation = box.rotation_matrix @ np.asarray([offset[0], offset[1], 0.0])
    transformed.center = transformed.center + translation
    delta = (
        Quaternion(axis=[0, 0, 1], degrees=float(offset[2]))
        if degrees else Quaternion(axis=[0, 0, 1], radians=float(offset[2]))
    )
    transformed.orientation = delta * transformed.orientation
    return transformed


def make_box(center, yaw, degrees):
    orientation = (
        Quaternion(axis=[0, 0, 1], degrees=yaw)
        if degrees else Quaternion(axis=[0, 0, 1], radians=yaw)
    )
    return Box(center=center, size=[1.8, 4.2, 1.6], orientation=orientation)


def assert_box_close(left, right, atol=1e-6):
    if not np.allclose(left.center, right.center, atol=atol, rtol=0.0):
        raise AssertionError(f"box centers differ: {left.center} != {right.center}")
    rotation_gap = left.rotation_matrix @ right.rotation_matrix.T
    if not np.allclose(rotation_gap, np.eye(3), atol=atol, rtol=0.0):
        raise AssertionError("box rotations differ")
    if not np.array_equal(left.wlh, right.wlh):
        raise AssertionError("shared SE(2) changed box size")


def run_case(degrees):
    yaw_values = [20.0, 13.0, -4.0] if degrees else np.deg2rad([20.0, 13.0, -4.0])
    transform = np.asarray(
        [0.22, -0.11, 4.0 if degrees else np.deg2rad(4.0)],
        dtype=np.float32,
    )
    previous = [
        make_box([8.0, -1.0, 0.5], yaw_values[0], degrees),
        make_box([7.1, -1.3, 0.5], yaw_values[1], degrees),
        make_box([5.8, -1.1, 0.5], yaw_values[2], degrees),
    ]
    current = make_box([9.2, -0.4, 0.5], yaw_values[0], degrees)

    identity = apply_shared_se2_to_boxes(
        previous, np.zeros(3, dtype=np.float32), degrees=degrees)
    for original, transformed in zip(previous, identity):
        assert_box_close(original, transformed, atol=1e-7)

    transformed = apply_shared_se2_to_boxes(previous, transform, degrees=degrees)
    canonical = boxes_to_anchor_parameters(previous, previous[0], degrees=degrees)
    augmented = boxes_to_anchor_parameters(transformed, transformed[0], degrees=degrees)
    if not np.allclose(canonical, augmented, atol=2e-6, rtol=0.0):
        gap = float(np.max(np.abs(canonical - augmented)))
        raise AssertionError(f"anchor-normalized trajectory changed; max gap={gap}")

    # Audit offsets must reproduce each already-transformed box when replayed
    # independently, even though the shared path never uses that replay.
    local_offsets = equivalent_local_offsets(previous, transformed, degrees=degrees)
    replayed = [
        apply_local_offset(box, offset, degrees)
        for box, offset in zip(previous, local_offsets)
    ]
    for expected, actual in zip(transformed, replayed):
        assert_box_close(expected, actual, atol=2e-6)

    displacement, velocity = canonical_dynamics_targets(
        previous, current, current_delta_t=0.5, degrees=degrees)
    transformed_current = apply_shared_se2_to_box(
        current, previous[0], transform, degrees=degrees)
    displacement_aug, velocity_aug = canonical_dynamics_targets(
        transformed, transformed_current, current_delta_t=0.5, degrees=degrees)
    if not np.allclose(displacement, displacement_aug, atol=2e-6, rtol=0.0):
        raise AssertionError("shared SE(2) changed the physical displacement label")
    if not np.allclose(velocity, velocity_aug, atol=2e-6, rtol=0.0):
        raise AssertionError("shared SE(2) changed the physical velocity label")

    # Two history paths that contain the same absolute frame receive exactly
    # the same candidate box when they share the sample-level transform.
    view_a = apply_shared_se2_to_boxes(
        [previous[0], previous[1]], transform, degrees=degrees)
    view_b = apply_shared_se2_to_boxes(
        [previous[0], previous[2]], transform, degrees=degrees)
    assert_box_close(view_a[0], view_b[0], atol=1e-7)


def choose_full_history_index(sampler, hist_num, candidate_id):
    for tracklet_id in range(sampler.dataset.get_num_tracklets()):
        if sampler.dataset.get_num_frames_tracklet(tracklet_id) > hist_num:
            anno_id = sampler.tracklet_start_ids[tracklet_id] + hist_num
            return anno_id * sampler.num_candidates + int(candidate_id)
    raise RuntimeError("No full-history tracklet is available")


def deterministic_sample(sampler, index, seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    return sampler[index]


def run_loader_check(args):
    from easydict import EasyDict
    from datasets import get_dataset

    cfg = EasyDict(load_yaml_config(args.cfg))
    cfg.preloading = False
    cfg.tiny = False
    cfg.candidate_trajectory_mode = "shared_se2"
    if args.path:
        cfg.path = args.path
    if args.version:
        cfg.version = args.version
    split = args.split or cfg.train_split
    sampler = get_dataset(cfg, type=cfg.train_type, split=split)
    samples = {
        candidate_id: deterministic_sample(
            sampler,
            choose_full_history_index(sampler, cfg.hist_num, candidate_id),
            args.seed + candidate_id,
        )
        for candidate_id in range(4)
    }
    sample0 = samples[0]

    if int(sample0["candidate_trajectory_mode_id"]) != 1:
        raise AssertionError("real loader did not activate shared_se2")
    if not np.array_equal(sample0["candidate_shared_transform"], np.zeros(3)):
        raise AssertionError("candidate0 shared transform is not exactly zero")
    for candidate_id in (1, 2, 3):
        sample = samples[candidate_id]
        if np.array_equal(sample["candidate_shared_transform"], np.zeros(3)):
            raise AssertionError(
                f"candidate{candidate_id} shared transform is unexpectedly zero")
        for key in (
                "canonical_ref_boxs", "dynamics_displacement_label", "velocity_label"):
            if not np.allclose(sample0[key], sample[key], atol=2e-6, rtol=0.0):
                raise AssertionError(
                    f"candidate{candidate_id} changed canonical field {key}")
    for sample in samples.values():
        if not np.allclose(
                sample["ref_boxs"], sample["canonical_ref_boxs"],
                atol=2e-5, rtol=0.0):
            raise AssertionError(
                "shared SE(2) did not preserve the anchor-normalized reference trajectory")

    print("formal B0 shared world-SE(2) real-loader checks: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg")
    parser.add_argument("--path")
    parser.add_argument("--version")
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    assert normalize_candidate_trajectory_mode("legacy") == "independent"
    assert normalize_candidate_trajectory_mode("shared-world-se2") == "shared_se2"
    run_case(degrees=False)
    run_case(degrees=True)
    print("formal B0 shared world-SE(2) dataset-free checks: PASS")
    if args.cfg:
        run_loader_check(args)


if __name__ == "__main__":
    main()
