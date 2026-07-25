import unittest
from pathlib import Path

import numpy as np
import torch

from models.ct_v2.fusion import ProposalFusionGate
from models.ct_v2.motion import ContinuousTimeMotionEncoder
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
    def test_gate_is_observation_biased_and_invalid_is_zero(self):
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
        self.assertTrue(torch.equal(alpha.squeeze(1), diagnostics["ct_fusion_alpha"]))


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


class ConfigCompositionTest(unittest.TestCase):
    def test_v2_full_config_resolves_ablation_chain(self):
        config = load_yaml_config(
            ROOT / "cfgs/ct_v2/04_ct_seqtrack_v2.yaml")
        self.assertTrue(config["use_ct_v2"])
        self.assertTrue(config["use_dynamics_encoder"])
        self.assertTrue(config["use_time_guided_search"])
        self.assertEqual(config["ct_fusion_mode"], "adaptive")
        self.assertFalse(config["use_twc"])
        self.assertFalse(config["use_m3_path_distillation"])

    def test_legacy_flat_yaml_still_loads(self):
        config = load_yaml_config(
            ROOT / "cfgs/seqtrack3d_nuscenes_a1_order.yaml")
        self.assertEqual(config["net_model"], "seqtrack3d")
        self.assertFalse(config["use_dynamics_encoder"])


if __name__ == "__main__":
    unittest.main()
