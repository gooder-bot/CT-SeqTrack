import unittest
from pathlib import Path

import numpy as np
import torch

from models.ct_v2.contracts import (
    build_search_usable_mask,
    resolve_observation_delta_t,
)
from models.ct_v2.fusion import ProposalFusionGate
from models.ct_v2.motion import ContinuousTimeMotionEncoder
from models.dynamics import apply_proposal_innovation
from utils.ct_history import (
    build_ct_history_offsets,
    correlate_candidate_offsets,
)
from utils.config import load_yaml_config
from utils.ct_search import (
    build_time_guided_search_box,
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
        self.axis = axis
        self.radians = radians


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

    def test_shared_se2_preserves_motion_and_reuses_search_candidate(self):
        motion, search = build_ct_history_offsets(
            self._offsets(),
            candidate_id=2,
            candidate_trajectory_mode="shared_se2",
            training_mode="correlated_candidate",
        )
        np.testing.assert_array_equal(motion, np.zeros_like(motion))
        self.assertIsNone(search)


class ConfigCompositionTest(unittest.TestCase):
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
