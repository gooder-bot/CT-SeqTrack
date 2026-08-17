import copy
import unittest
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from utils.ct_search import (
    combined_search_support_statistics,
    estimate_ordered_trajectory,
    resolve_joint_search_geometry,
    useful_search_coverage_need,
)
from utils.candidate_utils import (
    build_b1_physical_contract,
    physical_motion_targets,
)
from utils.recursive_state import (
    build_recursive_input_contract,
    commit_canonical_prediction,
    OnlineRecursiveBatchSampler,
    RecursiveTrackState,
    rotating_rollout_horizon,
    stable_tracklet_partition,
)


ROOT = Path(__file__).resolve().parents[1]


class DummyBox:
    def __init__(self, x=0.0):
        self.center = np.asarray([x, 0.0, 0.0], dtype=np.float64)
        self.wlh = np.asarray([2.0, 4.0, 1.5], dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=0.0)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


class RecursiveTrackStateTest(unittest.TestCase):
    def test_rollout_horizons_rotate_across_all_four_slots(self):
        first = [
            rotating_rollout_horizon([1, 2, 4, 8], slot, 0, 4)
            for slot in range(4)]
        second = [
            rotating_rollout_horizon([1, 2, 4, 8], slot, 1, 4)
            for slot in range(4)]
        self.assertEqual(first, [1, 2, 4, 8])
        self.assertEqual(second, [2, 4, 8, 1])

    def test_four_horizons_repeat_and_rotate_across_sixteen_slots(self):
        first = [
            rotating_rollout_horizon([1, 2, 4, 8], slot, 0, 16)
            for slot in range(16)]
        second = [
            rotating_rollout_horizon([1, 2, 4, 8], slot, 1, 16)
            for slot in range(16)]
        self.assertEqual(first, [1, 2, 4, 8] * 4)
        self.assertEqual(second, [2, 4, 8, 1] * 4)

    def test_legacy_observation_safe_size_uses_first_frame(self):
        source = (ROOT / "datasets/sampler.py").read_text(encoding="utf-8")
        legacy_processing = source.split(
            "def motion_processing(data, config", 1)[1].split(
                "def motion_processing_mf", 1)[0]
        self.assertIn(
            "data['first_frame']['3d_bbox'].wlh", legacy_processing)
        self.assertNotIn("coordinate_anchor_box", legacy_processing)

    def test_v2_names_gt_recursive_and_candidate_histories_separately(self):
        source = (ROOT / "datasets/sampler.py").read_text(encoding="utf-8")
        processing = source.split(
            "def motion_processing_mf", 1)[1].split(
                "class PartitionedTestTrackingSampler", 1)[0]
        for name in (
                "ground_truth_history",
                "recursive_history",
                "candidate_history"):
            self.assertIn(name, processing)
        self.assertIn(
            "search_v2_history_boxes = (\n"
            "                recursive_history if joint_contract_v2",
            processing)

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

    def test_training_reseed_rewrites_only_observed_history(self):
        state = RecursiveTrackState(3, "track/3", DummyBox())
        state.append(1, DummyBox(8.0), timestamp=0.5)
        state.append(2, DummyBox(9.0), timestamp=1.0)
        state.reseed_history(
            [2, 1, 0], [DummyBox(2.0), DummyBox(1.0), DummyBox(0.0)],
            [1.0, 0.5, 0.0], before_frame_id=3, rollout_horizon=4)
        self.assertEqual(state.rollout_horizon, 4)
        self.assertEqual(state.reseed_count, 1)
        self.assertEqual(state.rollout_age(3), 0)
        self.assertAlmostEqual(state.predictions[2].center[0], 2.0)
        with self.assertRaises(ValueError):
            state.reseed_history(
                [3], [DummyBox(3.0)], [1.5],
                before_frame_id=3, rollout_horizon=4)

    def test_b1_physical_target_uses_gt_origin_and_recursive_axes(self):
        current_gt = DummyBox(4.0)
        previous_gt = DummyBox(1.0)
        recursive_anchor = DummyBox(100.0)
        first, _ = physical_motion_targets(
            current_gt, previous_gt, recursive_anchor, 0.5)
        translated_anchor = DummyBox(-100.0)
        second, _ = physical_motion_targets(
            current_gt, previous_gt, translated_anchor, 0.5)
        np.testing.assert_array_equal(first, np.asarray([3.0, 0.0]))
        np.testing.assert_array_equal(first, second)

    def test_four_recovery_views_share_byte_identical_b1_contract(self):
        ground_truth = [DummyBox(2.0), DummyBox(1.0), DummyBox(0.0)]
        recursive = [DummyBox(20.0), DummyBox(18.0), DummyBox(16.0)]
        contracts = [
            build_b1_physical_contract(
                DummyBox(4.0), ground_truth, recursive, 0.5)
            for _candidate_id in range(4)]
        for contract in contracts[1:]:
            np.testing.assert_array_equal(
                contract["ref_boxs"], contracts[0]["ref_boxs"])
            np.testing.assert_array_equal(
                contract["target_xy"], contracts[0]["target_xy"])

    def test_training_and_inference_prepass_share_the_pure_contract(self):
        source = (ROOT / "models/seqtrack3d.py").read_text(encoding="utf-8")
        training = source.split(
            "    def _online_motion_prepass_batch", 1)[1].split(
                "    def _process_online_raw", 1)[0]
        inference = source.split(
            "    def _predict_motion_prepass_contract", 1)[1].split(
                "    @torch.no_grad()\n"
                "    def predict_motion_prepass", 1)[0]
        public_inference = source.split(
            "    def predict_motion_prepass", 1)[1].split(
                "    def forward", 1)[0]
        self.assertIn(
            "self._build_motion_prepass_inputs_contract(", training)
        self.assertIn(
            "self._build_motion_prepass_inputs_contract(", inference)
        self.assertIn(
            "self._predict_motion_prepass_contract(", public_inference)
        self.assertIn("np.stack([item['ref_boxs']", training)
        self.assertNotIn("['3d_bbox']", training)


class SearchSupportStatisticsTest(unittest.TestCase):
    def test_joint_geometry_uses_b1_for_endpoint_and_tube(self):
        history = [DummyBox(2.0), DummyBox(1.0), DummyBox(0.0)]
        prediction = {
            "valid": True,
            "mu_xy": np.asarray([3.0, 0.0], dtype=np.float32),
            "velocity_xy": np.asarray([3.0, 0.0], dtype=np.float32),
            "direction_xy": np.asarray([1.0, 0.0], dtype=np.float32),
            "log_sigma_parallel_perp": np.log(
                np.asarray([0.5, 0.5], dtype=np.float32)),
            "current_delta_t": 1.0,
            "gap_ratio": 1.0,
            "source_id": 1,
        }
        endpoint, tube, diagnostics = resolve_joint_search_geometry(
            history, [1.0, 1.0, 1.0], [1, 1, 1],
            prediction=prediction, use_b1_prepass=True,
            use_dynamic_sigma=False, fixed_margins=(2.0, 1.0))
        self.assertEqual(diagnostics["prior_source"], "b1")
        self.assertIsNotNone(endpoint)
        self.assertIsNotNone(tube)
        self.assertAlmostEqual(endpoint.center[0], 5.0)
        np.testing.assert_array_equal(
            diagnostics["endpoint_support_center"], endpoint.center)
        np.testing.assert_array_equal(
            diagnostics["tube_support_center"], tube.center)

    def test_joint_geometry_falls_back_only_when_b1_is_invalid(self):
        history = [DummyBox(2.0), DummyBox(1.0), DummyBox(0.0)]
        endpoint, tube, diagnostics = resolve_joint_search_geometry(
            history, [1.0, 1.0, 1.0], [1, 1, 1],
            prediction={"valid": False}, use_b1_prepass=True,
            fallback_min_displacement=0.0)
        self.assertEqual(diagnostics["prior_source"], "fallback_cv")
        self.assertIsNotNone(endpoint)
        self.assertIsNotNone(tube)

    def test_zero_displacement_is_geometry_not_an_availability_veto(self):
        history = [DummyBox(0.0), DummyBox(0.0), DummyBox(0.0)]
        endpoint, tube, diagnostics = resolve_joint_search_geometry(
            history, [1.0, 1.0, 1.0], [1, 1, 1],
            prediction={"valid": False}, use_b1_prepass=True,
            fallback_min_displacement=0.0)
        self.assertTrue(diagnostics["valid"])
        self.assertEqual(diagnostics["displacement"], 0.0)
        self.assertIsNotNone(endpoint)
        self.assertIsNotNone(tube)

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
    def test_training_seed_does_not_change_fixed_tracklet_partition(self):
        keys = [f"track/{index}" for index in range(80)]
        base = _FakeSequenceDataset(keys, [5] * len(keys))
        dataset = _FakeMotionSampler(base)
        first = OnlineRecursiveBatchSampler(
            dataset, slots=2, candidate_views=4, seed=42,
            partition_seed=42, partition="train",
            shadow_fraction=0.5)
        second = OnlineRecursiveBatchSampler(
            dataset, slots=2, candidate_views=4, seed=44,
            partition_seed=42, partition="train",
            shadow_fraction=0.5)
        self.assertEqual(first.tracklet_ids, second.tracklet_ids)
        self.assertNotEqual(first.seed, second.seed)

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

    def test_b3_disabled_sampler_never_builds_shadow_windows(self):
        seed = 42
        keys = [f"track/{index}" for index in range(40)]
        train_keys = [
            key for key in keys
            if stable_tracklet_partition(key, seed) == "train"]
        base = _FakeSequenceDataset(train_keys[:2], [7, 7])
        sampler = OnlineRecursiveBatchSampler(
            _FakeMotionSampler(base), slots=2, candidate_views=4,
            seed=seed, partition="train", shadow_interval=1,
            shadow_fraction=0.5, shadow_enabled=False)
        self.assertTrue(all(
            not bool(row[6]) for batch in sampler for row in batch))

    def test_set_epoch_replays_the_exact_epoch_order(self):
        seed = 42
        keys = [f"track/{index}" for index in range(80)]
        train_keys = [
            key for key in keys
            if stable_tracklet_partition(key, seed) == "train"]
        base = _FakeSequenceDataset(train_keys[:16], [4] * 16)
        sampler = OnlineRecursiveBatchSampler(
            _FakeMotionSampler(base), slots=16, candidate_views=4,
            seed=seed, partition="train", shadow_enabled=False)
        sampler.set_epoch(3)
        first = next(iter(sampler))
        sampler.set_epoch(3)
        replayed = next(iter(sampler))
        self.assertEqual(first, replayed)
        self.assertEqual(len(first), 64)
        self.assertEqual(sum(row[5] == 0 for row in first), 16)


if __name__ == "__main__":
    unittest.main()
