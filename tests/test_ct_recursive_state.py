import copy
import unittest

import numpy as np
from pyquaternion import Quaternion

from utils.ct_search import (
    combined_search_support_statistics,
    estimate_ordered_trajectory,
    useful_search_coverage_need,
)
from utils.recursive_state import (
    build_recursive_input_contract,
    commit_canonical_prediction,
    OnlineRecursiveBatchSampler,
    RecursiveTrackState,
    stable_tracklet_partition,
)


class DummyBox:
    def __init__(self, x=0.0):
        self.center = np.asarray([x, 0.0, 0.0], dtype=np.float64)
        self.wlh = np.asarray([2.0, 4.0, 1.5], dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=0.0)


class RecursiveTrackStateTest(unittest.TestCase):
    def test_history_is_prediction_backed_and_clone_is_independent(self):
        state = RecursiveTrackState(3, "track/3", DummyBox())
        state.append(1, DummyBox(1.25), timestamp=0.5)
        contract = state.history_contract([1, 0, 0], [1, 1, 0])
        self.assertEqual(contract["history_boxes_world"].shape, (3, 7))
        self.assertAlmostEqual(contract["history_boxes_world"][0, 0], 1.25)
        clone = state.clone()
        clone.predictions[1].center[0] = 9.0
        self.assertAlmostEqual(state.predictions[1].center[0], 1.25)

    def test_shared_input_contract_is_deterministic_and_prediction_backed(self):
        class Config:
            seed = 17
            degrees = True

        state = RecursiveTrackState(3, "track/3", DummyBox())
        state.append(1, DummyBox(1.25), timestamp=0.5)
        first = build_recursive_input_contract(
            state, 2, 3, Config(), candidate_id=2)
        second = build_recursive_input_contract(
            state, 2, 3, Config(), candidate_id=2)
        self.assertEqual(first["history_frame_ids"], [1, 0, 0])
        self.assertAlmostEqual(first["history_boxes_world"][0, 0], 1.25)
        np.testing.assert_array_equal(
            first["candidate_shared_transform"],
            second["candidate_shared_transform"])
        np.testing.assert_array_equal(
            first["point_sampling_seeds"], second["point_sampling_seeds"])

    def test_only_candidate_zero_can_write_canonical_state(self):
        state = RecursiveTrackState(3, "track/3", DummyBox())
        self.assertFalse(commit_canonical_prediction(
            state, 2, 1, DummyBox(9.0)))
        self.assertEqual(sorted(state.predictions), [0])
        self.assertTrue(commit_canonical_prediction(
            state, 0, 1, DummyBox(1.0)))
        self.assertEqual(sorted(state.predictions), [0, 1])


class SearchSupportStatisticsTest(unittest.TestCase):
    def test_extension_is_deduplicated_across_endpoint_and_tube(self):
        endpoint = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        tube = np.asarray([
            [1.0, 0.0, 0.0],
            [1.25, 0.0, 0.0],
        ], dtype=np.float32)
        result = combined_search_support_statistics(
            (endpoint, tube),
            (np.ones(2), np.ones(2)),
            (np.asarray([0, 1]), np.asarray([1, 1])),
            voxel_size=0.2,
        )
        self.assertEqual(result["total_count"], 3)
        self.assertEqual(result["extension_count"], 2)
        self.assertEqual(result["extension_voxels"], 2)

    def test_baseline_only_reuse_never_creates_extension_support(self):
        baseline = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        result = combined_search_support_statistics(
            (baseline, baseline.copy()),
            (np.ones(2), np.ones(2)),
            (np.zeros(2), np.zeros(2)),
            voxel_size=0.2,
        )
        self.assertEqual(result["total_count"], 2)
        self.assertEqual(result["extension_count"], 0)
        self.assertEqual(result["extension_voxels"], 0)

    def test_trajectory_reports_any_motion_constraint_clipping(self):
        history = [DummyBox(10.0), DummyBox(0.0), DummyBox(-10.0)]
        result = estimate_ordered_trajectory(
            history, [0.5, 0.5, 0.5], valid_mask=[1, 1, 1],
            max_speed=2.0, require_recent_transition=True)
        self.assertTrue(result["valid"])
        self.assertTrue(result["constraint_clipped"])

    def test_normal_cadence_requires_sparse_or_far_coverage_evidence(self):
        common = dict(
            query_delta_t=0.5,
            gap_ratio=1.0,
            reference_wlh=np.asarray([2.0, 4.0, 1.5]),
            min_delta_t=0.75,
            min_gap_ratio=1.5,
            min_endpoint_ratio=0.6,
            sparse_base_points=64,
        )
        ordinary, _ = useful_search_coverage_need(
            endpoint_xy=np.asarray([0.1, 0.1]),
            baseline_point_count=128, **common)
        sparse, _ = useful_search_coverage_need(
            endpoint_xy=np.asarray([0.1, 0.1]),
            baseline_point_count=32, **common)
        far, _ = useful_search_coverage_need(
            endpoint_xy=np.asarray([0.7, 0.1]),
            baseline_point_count=128, **common)
        self.assertFalse(ordinary)
        self.assertTrue(sparse)
        self.assertTrue(far)


class _FakeSequenceDataset:
    hist_num = 3

    def __init__(self, keys, lengths):
        self.keys = keys
        self.lengths = lengths

    def get_num_tracklets(self):
        return len(self.keys)

    def get_num_frames_tracklet(self, index):
        return self.lengths[index]

    def get_tracklet_key(self, index):
        return self.keys[index]


class _FakeMotionSampler:
    def __init__(self, base, candidates=4):
        self.dataset = base
        self.num_candidates = candidates


class OnlineRecursiveBatchSamplerTest(unittest.TestCase):
    def test_batches_group_candidates_and_advance_tracklets_causally(self):
        seed = 42
        keys = [f"track/{index}" for index in range(40)]
        train_keys = [
            key for key in keys
            if stable_tracklet_partition(key, seed) == "train"]
        base = _FakeSequenceDataset(train_keys[:4], [7, 7, 7, 7])
        sampler = OnlineRecursiveBatchSampler(
            _FakeMotionSampler(base), slots=2, candidate_views=4,
            seed=seed, partition="train", shadow_interval=2,
            shadow_fraction=0.5)
        batches = iter(sampler)
        first = next(batches)
        second = next(batches)
        self.assertEqual(len(first), 8)
        for slot in (0, 1):
            rows = first[slot * 4:(slot + 1) * 4]
            self.assertEqual([row[5] for row in rows], [0, 1, 2, 3])
            self.assertEqual(len({row[3] for row in rows}), 1)
            self.assertEqual(len({row[4] for row in rows}), 1)
        first_by_track = {row[3]: row[4] for row in first[::4]}
        second_by_track = {row[3]: row[4] for row in second[::4]}
        for tracklet in set(first_by_track) & set(second_by_track):
            self.assertEqual(
                second_by_track[tracklet], first_by_track[tracklet] + 1)
        self.assertEqual(sum(bool(row[6]) for row in first), 1)
        self.assertEqual(sum(bool(row[6]) for row in second), 0)
        third = next(batches)
        shadow_rows = [row for row in third if bool(row[6])]
        self.assertEqual(len(shadow_rows), 1)
        self.assertEqual(shadow_rows[0][2], 1)
        self.assertEqual(shadow_rows[0][5], 0)
        remaining = list(batches)
        self.assertTrue(all(len(batch) == 8 for batch in remaining))
        self.assertEqual(len(sampler), 12)


if __name__ == "__main__":
    unittest.main()
