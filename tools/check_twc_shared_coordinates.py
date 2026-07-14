#!/usr/bin/env python3
"""Dataset-free smoke test for shared TWC perturbations and sampling seeds."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.twc_utils import (  # noqa: E402
    build_shared_candidate_offset_map,
    build_shared_point_sampling_seed_map,
    candidate_offsets_for_frame_ids,
    point_sampling_seeds_for_frame_ids,
)


def main():
    config = SimpleNamespace(degrees=False)
    frame_ids_a = [9, 8, 7]
    frame_ids_b = [9, 7, 5]

    np.random.seed(42)
    offset_map = build_shared_candidate_offset_map(
        candidate_id=2,
        frame_ids=frame_ids_a + frame_ids_b,
        config=config,
    )
    offsets_a = candidate_offsets_for_frame_ids(frame_ids_a, offset_map)
    offsets_b = candidate_offsets_for_frame_ids(frame_ids_b, offset_map)

    assert np.allclose(offsets_a[0], offsets_b[0]), "t-1 anchor offset is not shared"
    assert np.allclose(offsets_a[2], offsets_b[1]), "shared t-3 offset is not shared"
    assert offsets_a.shape == (3, 3) and offsets_b.shape == (3, 3)

    seed_map = build_shared_point_sampling_seed_map(frame_ids_a + frame_ids_b)
    seeds_a = point_sampling_seeds_for_frame_ids(frame_ids_a, seed_map)
    seeds_b = point_sampling_seeds_for_frame_ids(frame_ids_b, seed_map)
    assert seeds_a[0] == seeds_b[0], "t-1 point-sampling seed is not shared"
    assert seeds_a[2] == seeds_b[1], "shared t-3 point-sampling seed is not shared"

    zero_map = build_shared_candidate_offset_map(
        candidate_id=0,
        frame_ids=frame_ids_a + frame_ids_b,
        config=config,
    )
    assert all(np.allclose(value, 0.0) for value in zero_map.values())

    print("TWC shared-coordinate smoke test: PASS")
    print("view_a offsets:", offsets_a)
    print("view_b offsets:", offsets_b)
    print("view_a point-sampling seeds:", seeds_a)
    print("view_b point-sampling seeds:", seeds_b)


if __name__ == "__main__":
    main()
