import unittest
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

from models.ct_v2.contracts import (
    build_search_usable_mask,
    resolve_observation_delta_t,
)
from models.ct_v2.fusion import (
    ProposalFusionGate,
    ReliabilityGatedProposalFusion,
)
from models.ct_v2.motion import (
    ContinuousTimeMotionEncoder,
    JointProposalFusion,
    OrderedPhysicalMotionEncoder,
    OrderedTrajectoryEncoder,
    TrajectorySearchEvidence,
    ZeroInitTrajectoryAdapter,
)
from models.dynamics import apply_proposal_innovation
from utils.candidate_utils import (
    anchor_relative_trajectory_targets,
    physical_motion_targets,
)
from utils.ct_history import (
    build_alternating_aux_history_offsets,
    build_irregular_history_offsets,
    build_ct_history_offsets,
    correlate_candidate_offsets,
)
from utils.config import load_yaml_config
from utils.ct_search import (
    build_ordered_trajectory_search_box,
    build_trajectory_endpoint_search_box,
    build_time_guided_search_box,
    sample_padded_search_extension,
    sample_search_extension,
    stratified_search_sample,
)


ROOT = Path(__file__).resolve().parents[1]


class DummyBox:
    def __init__(self, center):
        self.center = np.asarray(center, dtype=np.float64)
        self.wlh = np.asarray([2.0, 4.0, 1.5], dtype=np.float64)
        self.orientation = DummyOrientation()


class DummyOrientation:
    def __init__(self, axis=None, radians=0.0):
        self.axis = [0, 0, 1] if axis is None else axis
        self.radians = radians


class OrientedDummyBox:
    def __init__(self, center, yaw):
        self.center = np.asarray(center, dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=float(yaw))

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


class TrajectoryTargetContractTest(unittest.TestCase):
    def test_target_uses_actual_crop_anchor_frame(self):
        anchor = OrientedDummyBox([1.0, 2.0, 0.0], np.pi / 2.0)
        current = OrientedDummyBox([1.0, 4.0, 0.0], 2.0 * np.pi / 3.0)
        displacement, velocity = anchor_relative_trajectory_targets(
            current, anchor, current_delta_t=2.0)

        np.testing.assert_allclose(
            displacement,
            [2.0, 0.0, 0.0, np.pi / 6.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(velocity, [1.0, 0.0, 0.0], atol=1e-6)

    def test_anchor_error_is_part_of_online_target(self):
        candidate_anchor = OrientedDummyBox([0.3, -0.2, 0.0], 0.0)
        current = OrientedDummyBox([1.0, 0.0, 0.0], 0.0)
        displacement, _ = anchor_relative_trajectory_targets(
            current, candidate_anchor, current_delta_t=0.5)

        np.testing.assert_allclose(
            displacement, [0.7, 0.2, 0.0, 0.0], atol=1e-6)


class PhysicalMotionTargetContractTest(unittest.TestCase):
    def test_candidate_translation_does_not_enter_physical_target(self):
        latest = OrientedDummyBox([1.0, 1.0, 0.0], 0.0)
        current = OrientedDummyBox([2.0, 3.0, 0.0], 0.0)
        anchor_a = OrientedDummyBox([0.0, 0.0, 0.0], 0.0)
        anchor_b = OrientedDummyBox([25.0, -40.0, 0.0], 0.0)

        displacement_a, velocity_a = physical_motion_targets(
            current, latest, anchor_a, current_delta_t=2.0)
        displacement_b, velocity_b = physical_motion_targets(
            current, latest, anchor_b, current_delta_t=2.0)

        np.testing.assert_array_equal(displacement_a, displacement_b)
        np.testing.assert_array_equal(velocity_a, velocity_b)
        np.testing.assert_allclose(displacement_a, [1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(velocity_a, [0.5, 1.0], atol=1e-6)

    def test_anchor_yaw_only_rotates_the_common_coordinate_axes(self):
        latest = OrientedDummyBox([1.0, 1.0, 0.0], 0.0)
        current = OrientedDummyBox([2.0, 3.0, 0.0], 0.0)
        rotated_anchor = OrientedDummyBox(
            [100.0, -100.0, 0.0], np.pi / 2.0)

        displacement, _ = physical_motion_targets(
            current, latest, rotated_anchor, current_delta_t=1.0)

        np.testing.assert_allclose(displacement, [2.0, -1.0], atol=1e-6)


class SearchExpansionTest(unittest.TestCase):
    def test_real_time_controls_predicted_distance(self):
        boxes = [DummyBox([1.0, 0.0, 0.0]), DummyBox([0.0, 0.0, 0.0])]
        tube, diagnostics = build_time_guided_search_box(
            boxes, delta_t=[0.5, 1.0], valid_mask=[1, 1])
        self.assertIsNotNone(tube)
        self.assertTrue(diagnostics["valid"])
        self.assertAlmostEqual(diagnostics["displacement"], 0.5, places=6)
        np.testing.assert_allclose(tube.center, [1.25, 0.0, 0.0])

    def test_fixed_budget_preserves_baseline_majority(self):
        baseline = np.stack((
            np.arange(100), np.zeros(100), np.zeros(100), np.ones(100)),
            axis=1,
        ).astype(np.float32)
        expansion = baseline.copy()
        expansion[:, 1] = 10.0
        sampled, diagnostics = stratified_search_sample(
            baseline, expansion, sample_size=20, baseline_ratio=0.75, seed=7)
        self.assertEqual(sampled.shape, (20, 4))
        self.assertEqual(diagnostics["baseline_sample_count"], 15)
        self.assertEqual(diagnostics["expansion_sample_count"], 5)

    def test_no_extension_is_exact_baseline_budget(self):
        baseline = np.arange(80, dtype=np.float32).reshape(20, 4)
        sampled, diagnostics = stratified_search_sample(
            baseline, baseline.copy(), sample_size=16, seed=3)
        self.assertEqual(sampled.shape, (16, 4))
        self.assertEqual(diagnostics["expansion_sample_count"], 0)

    def test_tiny_extension_does_not_get_oversampled(self):
        baseline = np.arange(160, dtype=np.float32).reshape(40, 4)
        tiny_extension = baseline[:8].copy()
        tiny_extension[:, 1] += 1000.0
        _, diagnostics = stratified_search_sample(
            baseline,
            tiny_extension,
            sample_size=32,
            min_expansion_points=16,
            seed=9,
        )
        self.assertEqual(diagnostics["expansion_available_count"], 8)
        self.assertEqual(diagnostics["expansion_sample_count"], 0)

    def test_ordered_crop_is_off_for_normal_and_on_before_long_gap(self):
        boxes = [
            DummyBox([2.0, 0.0, 0.0]),
            DummyBox([1.0, 0.0, 0.0]),
            DummyBox([0.0, 0.0, 0.0]),
        ]
        normal, normal_diag = build_ordered_trajectory_search_box(
            boxes,
            delta_t=[0.5, 0.5, 0.5],
            valid_mask=[1, 1, 1],
        )
        self.assertIsNone(normal)
        self.assertEqual(normal_diag["reason"], "normal_cadence")

        long_gap, long_diag = build_ordered_trajectory_search_box(
            boxes,
            delta_t=[1.0, 0.5, 0.5],
            valid_mask=[1, 1, 1],
        )
        self.assertIsNotNone(long_gap)
        self.assertTrue(long_diag["valid"])
        self.assertAlmostEqual(long_diag["gap_ratio"], 2.0)
        self.assertGreater(long_diag["displacement"], 0.0)

    def test_separate_extension_does_not_change_baseline_tokens(self):
        baseline = np.stack((
            np.arange(20), np.zeros(20), np.zeros(20)),
            axis=1,
        ).astype(np.float32)
        expanded = baseline.copy()
        expanded[:, 1] = 5.0
        baseline_copy = baseline.copy()
        sampled, diagnostics = sample_search_extension(
            baseline, expanded, sample_size=8, min_expansion_points=4, seed=4)
        np.testing.assert_array_equal(baseline, baseline_copy)
        self.assertEqual(sampled.shape, (8, 3))
        self.assertTrue(diagnostics["active"])


class EndpointSearchV2Test(unittest.TestCase):
    def test_endpoint_crop_is_compact_and_available_at_normal_cadence(self):
        boxes = [
            DummyBox([2.0, 0.0, 0.0]),
            DummyBox([1.0, 0.0, 0.0]),
            DummyBox([0.0, 0.0, 0.0]),
        ]
        endpoint, diagnostics = build_trajectory_endpoint_search_box(
            boxes,
            delta_t=[0.5, 0.5, 0.5],
            valid_mask=[1, 1, 1],
        )
        self.assertIsNotNone(endpoint)
        self.assertTrue(diagnostics["valid"])
        np.testing.assert_allclose(endpoint.center, [3.0, 0.0, 0.0])
        np.testing.assert_array_equal(endpoint.wlh, boxes[0].wlh)

    def test_endpoint_yaw_rate_is_bounded(self):
        boxes = [
            DummyBox([2.0, 0.0, 0.0]),
            DummyBox([1.0, 0.0, 0.0]),
        ]
        boxes[0].orientation = DummyOrientation(radians=np.pi)
        boxes[1].orientation = DummyOrientation(radians=0.0)
        _, diagnostics = build_trajectory_endpoint_search_box(
            boxes, delta_t=[1.0, 0.1], valid_mask=[1, 1])
        self.assertLessEqual(
            abs(diagnostics["yaw_rate"]), np.pi / 2.0 + 1e-9)

    def test_sparse_extension_is_padded_without_repetition(self):
        baseline = np.zeros((1, 3), dtype=np.float32)
        for count in (3, 127, 128, 140):
            expanded = np.stack((
                np.arange(1, count + 1),
                np.ones(count),
                np.zeros(count)), axis=1).astype(np.float32)
            points, mask, diagnostics = sample_padded_search_extension(
                baseline, expanded, sample_size=128, seed=17)
            expected = min(count, 128)
            self.assertTrue(diagnostics["active"])
            self.assertEqual(int(mask.sum()), expected)
            self.assertEqual(
                len(np.unique(points[mask > 0, :3], axis=0)), expected)
            np.testing.assert_array_equal(points[mask == 0], 0.0)

    def test_empty_and_subthreshold_extension_are_finite_inactive(self):
        baseline = np.zeros((1, 3), dtype=np.float32)
        for expanded in (
                np.empty((0, 3), dtype=np.float32),
                np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.float32)):
            points, mask, diagnostics = sample_padded_search_extension(
                baseline, expanded, sample_size=128, seed=5)
            self.assertFalse(diagnostics["active"])
            self.assertEqual(mask.sum(), 0)
            self.assertTrue(np.isfinite(points).all())

    def test_extension_rng_does_not_touch_numpy_global_state(self):
        baseline = np.zeros((1, 3), dtype=np.float32)
        expanded = np.stack((
            np.arange(200), np.ones(200), np.zeros(200)), axis=1)
        state = np.random.get_state()
        sample_padded_search_extension(
            baseline, expanded, sample_size=128, seed=13)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])


class SearchEvidenceEncoderTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        torch.manual_seed(11)
        return {
            "point_inputs": torch.randn(2, 8, 9),
            "point_xy": torch.randn(2, 8, 2),
            "point_valid_mask": torch.tensor([
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
            "geometry_valid": torch.tensor([1.0, 0.0]),
            "observation_feature": torch.randn(2, 256),
            "motion_feature": torch.randn(2, 128),
            "observation_stats": torch.randn(2, 5),
            "query_delta_t": torch.tensor([0.5, 0.5]),
            "gap_ratio": torch.ones(2),
            "sigma_parallel": torch.ones(2),
            "sigma_perpendicular": torch.ones(2),
            "available_count": torch.tensor([3.0, 0.0]),
        }

    def test_padding_is_ignored_and_empty_rows_have_no_nan(self):
        encoder = TrajectorySearchEvidence()
        inputs = self._inputs()
        output = encoder(**inputs)
        changed = dict(inputs)
        changed["point_inputs"] = inputs["point_inputs"].clone()
        changed["point_xy"] = inputs["point_xy"].clone()
        changed["point_inputs"][0, 3:] = 1e6
        changed["point_xy"][0, 3:] = -1e6
        changed_output = encoder(**changed)
        for key in (
                "search_evidence_token", "search_proposal_xy",
                "search_confidence", "search_targetness_mass"):
            torch.testing.assert_close(output[key][0], changed_output[key][0])
        self.assertTrue(all(
            torch.isfinite(value).all()
            for value in output.values() if torch.is_tensor(value)))
        torch.testing.assert_close(
            output["search_proposal_xy"][1], torch.zeros(2))
        self.assertEqual(output["search_confidence"][1].item(), 0.0)


class JointProposalFusionTest(unittest.TestCase):
    @staticmethod
    def _fusion():
        return JointProposalFusion(
            observation_dim=8, motion_dim=6, search_dim=7,
            observation_stats_dim=5, context_dim=4, hidden_dim=12)

    @staticmethod
    def _inputs():
        return {
            "observation_box": torch.tensor([[1.0, 2.0, 3.0, 0.4]]),
            "observation_feature": torch.zeros(1, 8),
            "observation_stats": torch.zeros(1, 5),
            "motion_feature": torch.zeros(1, 6),
            "motion_proposal_xy": torch.tensor([[20.0, 2.0]]),
            "motion_log_sigma_xy": torch.zeros(1, 2),
            "motion_valid": torch.ones(1),
            "history_valid_ratio": torch.ones(1),
            "search_evidence_token": torch.zeros(1, 7),
            "search_proposal_xy": torch.tensor([[1.0, 30.0]]),
            "search_confidence": torch.ones(1),
            "search_targetness_mass": torch.ones(1),
            "search_entropy": torch.zeros(1),
            "search_valid": torch.ones(1),
            "query_delta_t": torch.tensor([1.0]),
            "gap_ratio": torch.ones(1),
        }

    def test_initial_gate_is_observation_biased(self):
        _, diagnostics = self._fusion()(**self._inputs())
        self.assertGreater(
            diagnostics["joint_gate_probability"][0, 0].item(), 0.97)

    def test_warmup_and_invalid_candidates_are_bitwise_observation(self):
        fusion = self._fusion()
        inputs = self._inputs()
        warmup, _ = fusion(**inputs, enabled_scale=0.0)
        self.assertTrue(torch.equal(warmup, inputs["observation_box"]))
        inputs["motion_valid"] = torch.zeros(1)
        inputs["search_valid"] = torch.zeros(1)
        invalid, _ = fusion(**inputs, enabled_scale=1.0)
        self.assertTrue(torch.equal(invalid, inputs["observation_box"]))

    def test_correction_is_radius_bounded_and_keeps_z_yaw(self):
        fusion = self._fusion()
        with torch.no_grad():
            fusion.gate[-1].bias.copy_(torch.tensor([0.0, 20.0, 20.0]))
        inputs = self._inputs()
        final, diagnostics = fusion(**inputs, enabled_scale=1.0)
        correction = torch.linalg.norm(
            final[:, :2] - inputs["observation_box"][:, :2], dim=1)
        self.assertTrue(torch.all(
            correction <= diagnostics["joint_fusion_radius"] + 1e-6))
        torch.testing.assert_close(
            final[:, 2:], inputs["observation_box"][:, 2:])

    def test_fused_loss_cannot_update_candidate_sources(self):
        fusion = self._fusion()
        inputs = self._inputs()
        for key in (
                "observation_box", "observation_feature", "motion_feature",
                "motion_proposal_xy", "search_evidence_token",
                "search_proposal_xy"):
            inputs[key] = inputs[key].requires_grad_()
        final, diagnostics = fusion(**inputs, enabled_scale=1.0)
        (final[:, :2].sum()
         + diagnostics["joint_gate_probability"].sum()).backward()
        for key in (
                "observation_box", "observation_feature", "motion_feature",
                "motion_proposal_xy", "search_evidence_token",
                "search_proposal_xy"):
            self.assertIsNone(inputs[key].grad, key)
        self.assertTrue(any(
            parameter.grad is not None for parameter in fusion.parameters()))


class ProposalFusionTest(unittest.TestCase):
    def test_gate_accepts_singleton_column_scalars_during_validation(self):
        gate = ProposalFusionGate(
            observation_dim=8,
            dynamics_dim=4,
            observation_stats_dim=5,
            max_alpha=0.35,
            init_alpha=0.05,
        )
        alpha, diagnostics = gate(
            torch.zeros(1, 8),
            torch.zeros(1, 4),
            torch.zeros(1, 3),
            torch.ones(1, 3),
            torch.zeros(1, 5),
            torch.tensor([[1.0]]),
            torch.tensor([[0.5]]),
            torch.tensor([[0.25]]),
        )
        self.assertEqual(tuple(alpha.shape), (1, 1))
        self.assertAlmostEqual(alpha.item(), 0.05, places=5)
        self.assertAlmostEqual(
            diagnostics["ct_fusion_alpha"].item(), 0.05, places=5)
        self.assertEqual(diagnostics["ct_fusion_valid"].item(), 1.0)

    def test_innovation_accepts_singleton_columns_from_adaptive_gate(self):
        observation = torch.zeros(1, 3)
        dynamics = torch.tensor([[1.0, 0.0, 0.0]])
        gate = ProposalFusionGate(
            observation_dim=8,
            dynamics_dim=4,
            observation_stats_dim=5,
            max_alpha=0.75,
            init_alpha=0.25,
        )
        alpha, _ = gate(
            torch.zeros(1, 8),
            torch.zeros(1, 4),
            observation,
            dynamics,
            torch.zeros(1, 5),
            torch.tensor([[1.0]]),
            torch.tensor([[0.5]]),
            torch.tensor([[0.25]]),
        )
        final, diagnostics = apply_proposal_innovation(
            observation,
            dynamics,
            dynamics_valid=torch.tensor([[1.0]]),
            current_delta_t=torch.tensor([[0.5]]),
            alpha=alpha,
            enabled_scale=1.0,
        )
        self.assertEqual(tuple(final.shape), (1, 3))
        torch.testing.assert_close(
            diagnostics["dynamics_innovation_alpha"],
            torch.tensor([0.25]),
        )
        torch.testing.assert_close(
            diagnostics["dynamics_innovation_radius"],
            torch.tensor([0.75]),
        )
        torch.testing.assert_close(
            diagnostics["dynamics_innovation_applied_norm"],
            torch.tensor([0.1875]),
        )

    def test_gate_keeps_nominal_alpha_separate_from_valid_mask(self):
        gate = ProposalFusionGate(
            observation_dim=8,
            dynamics_dim=4,
            observation_stats_dim=5,
            max_alpha=0.35,
            init_alpha=0.05,
        )
        alpha, diagnostics = gate(
            torch.zeros(2, 8),
            torch.zeros(2, 4),
            torch.zeros(2, 3),
            torch.ones(2, 3),
            torch.zeros(2, 5),
            torch.tensor([[1.0], [0.0]]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.25, 0.25]),
        )
        self.assertAlmostEqual(alpha[0].item(), 0.05, places=5)
        self.assertEqual(alpha[1].item(), 0.0)
        torch.testing.assert_close(
            diagnostics["ct_fusion_alpha"],
            torch.tensor([0.05, 0.05]),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            diagnostics["ct_fusion_valid"], torch.tensor([1.0, 0.0]))
        _, innovation = apply_proposal_innovation(
            torch.zeros(2, 3),
            torch.ones(2, 3),
            diagnostics["ct_fusion_valid"],
            current_delta_t=torch.tensor([0.5, 0.5]),
            alpha=alpha,
            enabled_scale=0.5,
        )
        torch.testing.assert_close(
            innovation["dynamics_innovation_alpha"],
            diagnostics["ct_fusion_alpha"]
            * diagnostics["ct_fusion_valid"]
            * 0.5,
        )

    def test_fixed_alpha_applied_includes_valid_mask_and_warmup(self):
        observation = torch.zeros(2, 3)
        dynamics = torch.ones(2, 3)
        valid = torch.tensor([1.0, 0.0])

        _, active = apply_proposal_innovation(
            observation,
            dynamics,
            valid,
            current_delta_t=torch.tensor([0.5, 0.5]),
            alpha=0.75,
            enabled_scale=0.5,
        )
        torch.testing.assert_close(
            active["dynamics_innovation_alpha"],
            torch.tensor([0.375, 0.0]),
        )

        _, warmup = apply_proposal_innovation(
            observation,
            dynamics,
            valid,
            current_delta_t=torch.tensor([0.5, 0.5]),
            alpha=0.75,
            enabled_scale=0.0,
        )
        self.assertTrue(torch.equal(
            warmup["dynamics_innovation_alpha"], torch.zeros(2)))


class ContinuousTimeMotionTest(unittest.TestCase):
    def test_query_displacement_scales_with_current_delta_t(self):
        torch.manual_seed(4)
        encoder = ContinuousTimeMotionEncoder(hidden_dim=16)
        boxes = torch.tensor([[
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]])
        delta_t = torch.tensor([[0.5, 1.0, 1.0]])
        valid = torch.ones(1, 3)
        _, velocity, displacement_half, has_transition = encoder(
            boxes, delta_t, valid, current_delta_t=torch.tensor([0.5]))
        _, velocity_two, displacement_two, _ = encoder(
            boxes, delta_t, valid, current_delta_t=torch.tensor([2.0]))
        self.assertEqual(has_transition.item(), 1.0)
        torch.testing.assert_close(displacement_half, velocity * 0.5)
        torch.testing.assert_close(displacement_two, velocity_two * 2.0)

    def test_invalid_padded_history_has_no_motion(self):
        encoder = ContinuousTimeMotionEncoder(hidden_dim=16)
        boxes = torch.zeros(1, 3, 4)
        delta_t = torch.tensor([[0.5, 0.5, 0.5]])
        valid = torch.tensor([[1.0, 0.0, 0.0]])
        _, velocity, displacement, has_transition = encoder(
            boxes, delta_t, valid)
        self.assertEqual(has_transition.item(), 0.0)
        self.assertTrue(torch.equal(velocity, torch.zeros_like(velocity)))
        self.assertTrue(torch.equal(
            displacement, torch.zeros_like(displacement)))


class OrderedTrajectoryMotionTest(unittest.TestCase):
    def test_encoder_distinguishes_acceleration_order(self):
        torch.manual_seed(17)
        encoder = OrderedTrajectoryEncoder(hidden_dim=16, step_dim=8)
        # Both histories contain transition velocities {1, 3}; only their
        # chronological order differs.
        accelerating = torch.tensor([[[
            4.0, 0.0, 0.0, 0.0
        ], [
            1.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])
        decelerating = torch.tensor([[[
            4.0, 0.0, 0.0, 0.0
        ], [
            3.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])
        delta_t = torch.ones(1, 3)
        valid = torch.ones(1, 3)
        result_a = encoder.forward_trajectory(
            accelerating, delta_t, valid, current_delta_t=torch.ones(1))
        result_b = encoder.forward_trajectory(
            decelerating, delta_t, valid, current_delta_t=torch.ones(1))
        self.assertFalse(torch.equal(
            result_a["feature"], result_b["feature"]))
        self.assertNotEqual(
            result_a["displacement"][0, 0].item(),
            result_b["displacement"][0, 0].item(),
        )

    def test_zero_init_adapter_is_exact_observation_identity(self):
        adapter = ZeroInitTrajectoryAdapter(
            feature_dim=8,
            trajectory_dim=4,
            search_dim=3,
            hidden_dim=6,
        )
        observation = torch.randn(2, 8)
        adapted, diagnostics = adapter(
            observation,
            torch.randn(2, 4),
            torch.randn(2, 3),
            torch.zeros(2, 4),
            torch.tensor([1.0, 2.0]),
            torch.ones(2, 1),
            torch.tensor([0.0, 1.0]),
        )
        self.assertTrue(torch.equal(adapted, observation))
        self.assertTrue(torch.equal(
            diagnostics["trajectory_adapter_correction"],
            torch.zeros_like(observation),
        ))


class OrderedPhysicalMotionTest(unittest.TestCase):
    @staticmethod
    def _constant_velocity_history():
        return torch.tensor([[[
            2.0, 0.0, 0.0, 0.0
        ], [
            1.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])

    def test_measured_time_controls_velocity_and_query_integration(self):
        encoder = OrderedPhysicalMotionEncoder(hidden_dim=16, step_dim=8)
        boxes = self._constant_velocity_history()
        valid = torch.ones(1, 3)

        short = encoder(
            boxes,
            torch.tensor([[0.5, 1.0, 1.0]]),
            valid,
            current_delta_t=torch.tensor([0.5]),
        )
        long_query = encoder(
            boxes,
            torch.tensor([[2.0, 1.0, 1.0]]),
            valid,
            current_delta_t=torch.tensor([2.0]),
        )
        faster_history = encoder(
            boxes,
            torch.tensor([[0.5, 0.5, 0.5]]),
            valid,
            current_delta_t=torch.tensor([0.5]),
        )

        torch.testing.assert_close(short["velocity_xy"], torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(short["prior_xy"], torch.tensor([[0.5, 0.0]]))
        torch.testing.assert_close(long_query["prior_xy"], torch.tensor([[2.0, 0.0]]))
        torch.testing.assert_close(
            faster_history["velocity_xy"], torch.tensor([[2.0, 0.0]]))
        torch.testing.assert_close(
            faster_history["prior_xy"], torch.tensor([[1.0, 0.0]]))

    def test_chronological_order_is_visible_to_the_gru(self):
        torch.manual_seed(19)
        encoder = OrderedPhysicalMotionEncoder(hidden_dim=16, step_dim=8)
        accelerating = torch.tensor([[[
            4.0, 0.0, 0.0, 0.0
        ], [
            1.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])
        decelerating = torch.tensor([[[
            4.0, 0.0, 0.0, 0.0
        ], [
            3.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])
        delta_t = torch.ones(1, 3)
        valid = torch.ones(1, 3)

        result_a = encoder(accelerating, delta_t, valid, torch.ones(1))
        result_b = encoder(decelerating, delta_t, valid, torch.ones(1))

        self.assertFalse(torch.equal(result_a["feature"], result_b["feature"]))
        self.assertFalse(torch.equal(result_a["prior_xy"], result_b["prior_xy"]))

    def test_invalid_or_nonfinite_history_has_no_prior(self):
        encoder = OrderedPhysicalMotionEncoder(hidden_dim=16, step_dim=8)
        boxes = self._constant_velocity_history().repeat(2, 1, 1)
        boxes[1, 1, 0] = float("nan")
        valid = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = encoder(
            boxes,
            torch.full((2, 3), 0.5),
            valid,
            current_delta_t=torch.tensor([0.5, 0.5]),
        )

        torch.testing.assert_close(result["valid"], torch.zeros(2))
        torch.testing.assert_close(result["prior_xy"], torch.zeros(2, 2))

    def test_invalid_old_padding_is_a_true_gru_noop(self):
        torch.manual_seed(23)
        encoder = OrderedPhysicalMotionEncoder(hidden_dim=16, step_dim=8)
        full = self._constant_velocity_history()
        padded = encoder(
            full,
            torch.tensor([[0.5, 1.0, 99.0]]),
            torch.tensor([[1.0, 1.0, 0.0]]),
            current_delta_t=torch.tensor([0.5]),
        )
        unpadded = encoder(
            full[:, :2],
            torch.tensor([[0.5, 1.0]]),
            torch.ones(1, 2),
            current_delta_t=torch.tensor([0.5]),
        )

        torch.testing.assert_close(padded["feature"], unpadded["feature"])
        torch.testing.assert_close(padded["prior_xy"], unpadded["prior_xy"])


class ReliableMotionFusionTest(unittest.TestCase):
    @staticmethod
    def _fusion():
        return ReliabilityGatedProposalFusion(
            observation_dim=8,
            motion_dim=6,
            observation_stats_dim=5,
            hidden_dim=10,
            context_dim=4,
            max_alpha=0.5,
            init_probability=0.01,
            radius_base=0.25,
            radius_per_second=0.5,
            radius_max=1.25,
            time_scale=0.5,
        )

    @staticmethod
    def _inputs():
        return {
            "observation_box": torch.tensor([[1.0, 2.0, 3.0, 0.4]]),
            "observation_feature": torch.zeros(1, 8),
            "observation_stats": torch.zeros(1, 5),
            "motion_feature": torch.zeros(1, 6),
            "prior_xy": torch.tensor([[10.0, 2.0]]),
            "prior_velocity_xy": torch.tensor([[2.0, 0.0]]),
            "prior_valid": torch.ones(1),
            "gap_ratio": torch.ones(1),
            "history_valid_ratio": torch.ones(1),
            "current_delta_t": torch.tensor([0.5]),
        }

    def test_warmup_invalid_and_nonfinite_inputs_are_exact_identity(self):
        fusion = self._fusion()
        inputs = self._inputs()
        observation = inputs["observation_box"]

        warmup, _ = fusion(**inputs, enabled_scale=0.0)
        self.assertTrue(torch.equal(warmup, observation))

        invalid_inputs = dict(inputs)
        invalid_inputs["prior_valid"] = torch.zeros(1)
        invalid, _ = fusion(**invalid_inputs, enabled_scale=1.0)
        self.assertTrue(torch.equal(invalid, observation))

        nonfinite_inputs = dict(inputs)
        nonfinite_inputs["prior_xy"] = torch.tensor([[float("nan"), 2.0]])
        nonfinite, _ = fusion(**nonfinite_inputs, enabled_scale=1.0)
        self.assertTrue(torch.equal(nonfinite, observation))

    def test_xy_is_bounded_and_z_yaw_are_never_changed(self):
        fusion = self._fusion()
        with torch.no_grad():
            fusion.gate[-1].bias.fill_(20.0)
        inputs = self._inputs()
        final, diagnostics = fusion(**inputs, enabled_scale=1.0)

        torch.testing.assert_close(
            final[:, 2:], inputs["observation_box"][:, 2:])
        correction_norm = torch.linalg.norm(
            final[:, :2] - inputs["observation_box"][:, :2], dim=1)
        radius = torch.tensor([0.5])
        self.assertTrue(torch.all(correction_norm <= 0.5 * radius + 1e-6))
        torch.testing.assert_close(diagnostics["motion_fusion_radius"], radius)

    def test_fused_loss_updates_only_gate_parameters(self):
        fusion = self._fusion()
        inputs = self._inputs()
        inputs["observation_box"] = inputs["observation_box"].requires_grad_()
        inputs["observation_feature"] = inputs[
            "observation_feature"].requires_grad_()
        inputs["motion_feature"] = inputs["motion_feature"].requires_grad_()
        inputs["prior_xy"] = inputs["prior_xy"].requires_grad_()
        inputs["prior_velocity_xy"] = inputs[
            "prior_velocity_xy"].requires_grad_()

        final, diagnostics = fusion(**inputs, enabled_scale=1.0)
        loss = final[:, :2].sum() + diagnostics[
            "motion_gate_probability"].sum()
        loss.backward()

        self.assertIsNone(inputs["observation_box"].grad)
        self.assertIsNone(inputs["observation_feature"].grad)
        self.assertIsNone(inputs["motion_feature"].grad)
        self.assertIsNone(inputs["prior_xy"].grad)
        self.assertIsNone(inputs["prior_velocity_xy"].grad)
        self.assertTrue(any(
            parameter.grad is not None for parameter in fusion.parameters()))


class ContinuousTimeContractTest(unittest.TestCase):
    @staticmethod
    def _stats_input(effective=True):
        data = {
            "valid_mask": torch.ones(1, 3),
            "num_points_in_search": torch.tensor([20.0]),
            "current_delta_t": torch.tensor([0.5]),
            "current_delta_t_real": torch.tensor([0.5]),
        }
        if effective:
            data["current_delta_t_effective"] = torch.tensor([2.0])
        return data

    def test_ct_v2_observation_stats_use_effective_time(self):
        selected, real, effective = resolve_observation_delta_t(
            self._stats_input(effective=True),
            torch.zeros(1, 2, 8),
            use_ct_v2=True,
            default_time_step=0.5,
        )
        self.assertAlmostEqual(selected.item(), 2.0)
        self.assertAlmostEqual(real.item(), 0.5)
        self.assertAlmostEqual(effective.item(), 2.0)

    def test_ct_v2_time_falls_back_and_legacy_stays_real(self):
        reference = torch.zeros(1, 2, 8)
        fallback, _, _ = resolve_observation_delta_t(
            self._stats_input(effective=False),
            reference,
            use_ct_v2=True,
            default_time_step=0.5,
        )
        legacy, _, _ = resolve_observation_delta_t(
            self._stats_input(effective=True),
            reference,
            use_ct_v2=False,
            default_time_step=0.5,
        )
        self.assertAlmostEqual(fallback.item(), 0.5)
        self.assertAlmostEqual(legacy.item(), 0.5)

    def test_usable_search_threshold_matches_regularizer(self):
        reference = torch.ones(4, 1)
        aux = {"obs_num_points_search": torch.tensor([0.0, 1.0, 2.0, 3.0])}
        mask = build_search_usable_mask({}, aux, reference)
        torch.testing.assert_close(
            mask, torch.tensor([[0.0], [0.0], [0.0], [1.0]]))
        explicit = build_search_usable_mask(
            {"search_has_usable_points": torch.tensor([1.0, 0.0, 1.0, 0.0])},
            aux,
            reference,
        )
        torch.testing.assert_close(
            explicit, torch.tensor([[1.0], [0.0], [1.0], [0.0]]))


class CorrelatedHistoryTest(unittest.TestCase):
    @staticmethod
    def _offsets():
        return np.asarray([
            [0.30, -0.20, 0.08],
            [-0.25, 0.25, -0.07],
            [0.20, -0.30, 0.06],
        ], dtype=np.float32)

    def test_correlated_offsets_reduce_high_frequency_error(self):
        raw = self._offsets()
        correlated = correlate_candidate_offsets(
            raw, correlation=0.75, anchor_offset=np.zeros(3))
        self.assertTrue(np.array_equal(correlated[0], np.zeros(3)))
        self.assertLess(
            np.linalg.norm(np.diff(correlated, axis=0)),
            np.linalg.norm(np.diff(raw, axis=0)),
        )

    def test_candidate_zero_uses_exact_clean_offsets(self):
        motion, search = build_ct_history_offsets(
            self._offsets(),
            candidate_id=0,
            candidate_trajectory_mode="independent",
            training_mode="correlated_candidate",
        )
        np.testing.assert_array_equal(motion, np.zeros_like(motion))
        np.testing.assert_array_equal(search, np.zeros_like(search))

    def test_motion_and_search_keep_their_required_anchors(self):
        offsets = self._offsets()
        motion, search = build_ct_history_offsets(
            offsets,
            candidate_id=1,
            candidate_trajectory_mode="independent",
            training_mode="correlated_candidate",
            correlation=0.75,
        )
        np.testing.assert_array_equal(motion[0], np.zeros(3))
        np.testing.assert_array_equal(search[0], offsets[0])

    def test_recursive_history_grows_error_with_age(self):
        offsets = self._offsets()
        motion, search = build_ct_history_offsets(
            offsets,
            candidate_id=1,
            candidate_trajectory_mode="independent",
            training_mode="recursive_candidate",
            correlation=0.75,
            recursive_error_scale=4.0,
        )
        np.testing.assert_array_equal(motion[0], np.zeros(3))
        np.testing.assert_array_equal(search[0], offsets[0])
        self.assertGreater(
            np.linalg.norm(motion[-1]),
            np.linalg.norm(correlate_candidate_offsets(
                offsets, 0.75, np.zeros(3))[-1]),
        )

    def test_shared_se2_preserves_motion_and_reuses_search_candidate(self):
        motion, search = build_ct_history_offsets(
            self._offsets(),
            candidate_id=2,
            candidate_trajectory_mode="shared_se2",
            training_mode="correlated_candidate",
        )
        np.testing.assert_array_equal(motion, np.zeros_like(motion))
        self.assertIsNone(search)

    def test_irregular_training_offsets_are_causal_and_increasing(self):
        offsets = build_irregular_history_offsets(
            hist_num=3, query_gap=2, transition_gaps=[1, 2, 4])
        self.assertEqual(offsets, [2, 3, 5])
        self.assertTrue(all(
            newer < older
            for newer, older in zip(offsets[:-1], offsets[1:])
        ))

    def test_v3_auxiliary_cadence_alternates_without_consuming_rng(self):
        torch.manual_seed(1234)
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()

        even = build_alternating_aux_history_offsets(3, 0)
        odd = build_alternating_aux_history_offsets(3, 1)

        self.assertEqual(even, [2, 3, 5])
        self.assertEqual(odd, [4, 5, 7])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        numpy_state_after = np.random.get_state()
        self.assertEqual(numpy_state[0], numpy_state_after[0])
        np.testing.assert_array_equal(numpy_state[1], numpy_state_after[1])
        self.assertEqual(numpy_state[2:], numpy_state_after[2:])


class ConfigCompositionTest(unittest.TestCase):
    def test_b2_v2_resolves_scratch_three_candidate_contract(self):
        config = load_yaml_config(
            ROOT / "cfgs/ct_v2/03_ct_motion_search_v2.yaml")
        self.assertTrue(config["use_b1motion_v3"])
        self.assertFalse(config["use_motion_v3_legacy_fusion"])
        self.assertTrue(config["use_search_evidence_v2"])
        self.assertTrue(config["use_joint_proposal_fusion"])
        self.assertFalse(config["use_time_guided_search"])
        self.assertFalse(config["use_trajectory_search"])
        self.assertEqual(config["point_sample_size"], 1024)
        self.assertEqual(config["search_v2_point_count"], 128)
        self.assertEqual(config["search_v2_min_points"], 3)
        self.assertEqual(config["batch_size"], 16)
        self.assertEqual(config["epoch"], 60)
        self.assertEqual(config["check_val_every_n_epoch"], 5)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["joint_fusion_warmup_epochs"], 10)
        self.assertEqual(config["joint_fusion_ramp_epochs"], 10)

    def test_motion_v3_isolated_reliable_post_transformer_config(self):
        config = load_yaml_config(
            ROOT / "cfgs/ct_v2/02_ct_motion_v3.yaml")
        self.assertTrue(config["use_b1motion_v3"])
        for key in (
                "use_ct_v2",
                "use_dynamics_encoder",
                "use_ordered_trajectory_encoder",
                "use_trajectory_adapter",
                "use_trajectory_search",
                "use_time_guided_search",
                "use_point_feature_tc"):
            self.assertFalse(config[key], key)
        self.assertEqual(config["trajectory_training_irregular_probability"], 0.0)
        self.assertEqual(config["ct_history_training_mode"], "correlated_candidate")
        self.assertEqual(config["ct_history_recursive_error_scale"], 1.0)
        self.assertEqual(config["motion_v3_aux_query_gaps"], [2, 4])
        self.assertEqual(config["motion_v3_aux_transition_gaps"], [1, 2])
        self.assertEqual(config["motion_v3_warmup_epoch"], 10)
        self.assertEqual(config["motion_v3_fusion_scale"], 1.0)
        self.assertEqual(config["motion_v3_alpha_max"], 0.5)

    def test_new_b1_is_ordered_precrop_without_fixed_innovation(self):
        config = load_yaml_config(
            ROOT / "cfgs/ct_v2/02_ct_motion.yaml")
        self.assertTrue(config["use_ordered_trajectory_encoder"])
        self.assertTrue(config["use_trajectory_search"])
        self.assertTrue(config["use_trajectory_adapter"])
        self.assertFalse(config["use_time_guided_search"])
        self.assertEqual(
            config["dynamics_motion_mode"], "trajectory_adapter")
        self.assertEqual(
            config["ct_history_training_mode"], "recursive_candidate")
        self.assertEqual(config["dynamics_innovation_alpha"], 0.0)
        self.assertEqual(config["dynamics_innovation_scale"], 0.0)
        self.assertFalse(config["trajectory_search_allow_normal_cadence"])

    def test_rejected_alpha_controls_still_resolve_legacy_b1(self):
        config = load_yaml_config(
            ROOT / "cfgs/ct_v2/02_ct_motion_alpha025.yaml")
        self.assertFalse(config["use_ordered_trajectory_encoder"])
        self.assertFalse(config["use_trajectory_search"])
        self.assertEqual(
            config["dynamics_motion_mode"], "proposal_innovation")
        self.assertEqual(config["dynamics_innovation_alpha"], 0.25)

    def test_search_only_keeps_baseline_model_and_correlated_search(self):
        config = load_yaml_config(
            ROOT / "cfgs/ct_v2/05_seqtrack3d_search_only.yaml")
        self.assertFalse(config["use_ct_v2"])
        self.assertFalse(config["use_dynamics_encoder"])
        self.assertTrue(config["use_time_guided_search"])
        self.assertEqual(config["dynamics_motion_mode"], "feature")
        self.assertEqual(config["velocity_weight"], 0.0)
        self.assertEqual(config["dynamics_displacement_weight"], 0.0)
        self.assertEqual(config["dynamics_innovation_alpha"], 0.0)
        self.assertEqual(config["dynamics_innovation_scale"], 0.0)
        self.assertEqual(
            config["ct_history_training_mode"], "correlated_candidate")
        self.assertEqual(
            config["ct_search_training_history"], "correlated_candidate")

    def test_v2_full_config_resolves_ablation_chain(self):
        config = load_yaml_config(
            ROOT / "cfgs/ct_v2/04_ct_seqtrack_v2.yaml")
        self.assertTrue(config["use_ct_v2"])
        self.assertTrue(config["use_dynamics_encoder"])
        self.assertTrue(config["use_time_guided_search"])
        self.assertEqual(config["ct_fusion_mode"], "adaptive")
        self.assertEqual(
            config["ct_history_training_mode"], "correlated_candidate")
        self.assertEqual(
            config["ct_search_training_history"], "correlated_candidate")
        self.assertFalse(config["use_twc"])
        self.assertFalse(config["use_m3_path_distillation"])

    def test_legacy_flat_yaml_still_loads(self):
        config = load_yaml_config(
            ROOT / "cfgs/seqtrack3d_nuscenes_a1_order.yaml")
        self.assertEqual(config["net_model"], "seqtrack3d")
        self.assertFalse(config["use_dynamics_encoder"])


if __name__ == "__main__":
    unittest.main()
