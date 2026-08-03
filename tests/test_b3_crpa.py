import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.ct_v2.crpa import (
    CRPA_ROUTER_SCHEMA,
    counterfactual_gain_targets,
    crpa_router_loss,
    stable_tracklet_partition,
)
from models.ct_v2.motion import ClosedLoopRiskAwareProposalRouter
from tools.b3_crpa_common import load_router_sidecar
from tools.train_b3_router import calibrate_router
from utils.config import load_yaml_config


class ClosedLoopRiskAwareProposalRouterTest(unittest.TestCase):
    def build_router(self):
        return ClosedLoopRiskAwareProposalRouter(
            observation_dim=4,
            motion_dim=3,
            search_dim=2,
            context_dim=4,
            hidden_dim=8,
        )

    def inputs(self):
        return {
            "observation_box": torch.tensor([[0.0, 0.0, 2.0, 0.3]]),
            "observation_feature": torch.ones(1, 4),
            "observation_stats": torch.zeros(1, 5),
            "observation_entropy": torch.tensor([0.4]),
            "observation_refinement_xy": torch.tensor([[0.1, -0.1]]),
            "motion_feature": torch.ones(1, 3),
            "motion_proposal_xy": torch.tensor([[1.0, 0.0]]),
            "motion_log_sigma_xy": torch.zeros(1, 2),
            "motion_valid": torch.ones(1),
            "history_valid_ratio": torch.ones(1),
            "search_evidence_token": torch.ones(1, 2),
            "search_proposal_xy": torch.tensor([[0.0, 1.0]]),
            "search_valid": torch.ones(1),
            "search_targetness_mean": torch.tensor([0.4]),
            "search_targetness_max": torch.tensor([0.8]),
            "search_entropy": torch.tensor([0.2]),
            "search_effective_sample_size": torch.tensor([4.0]),
            "search_extension_weight_ratio": torch.tensor([0.5]),
            "search_available_count": torch.tensor([16.0]),
            "search_extension_count": torch.tensor([8.0]),
            "search_overlap_count": torch.tensor([3.0]),
            "query_delta_t": torch.tensor([0.5]),
            "gap_ratio": torch.tensor([1.0]),
        }

    def test_untrained_router_is_exact_observation_fallback(self):
        router = self.build_router().eval()
        inputs = self.inputs()
        result, diagnostics = router(**inputs)
        self.assertTrue(torch.equal(result, inputs["observation_box"]))
        self.assertEqual(diagnostics["b3_selected_index"].item(), 0)
        self.assertEqual(diagnostics["b3_abstained"].item(), 1.0)
        self.assertTrue(torch.all(
            diagnostics["b3_gain_quantiles"][:, :, 0]
            <= diagnostics["b3_gain_quantiles"][:, :, 1]))

    def test_top1_bounded_writeback_preserves_z_and_yaw(self):
        router = self.build_router().eval()
        with torch.no_grad():
            router.median_gain_head.bias.copy_(torch.tensor([0.30, 0.20]))
            router.gain_spread_head.bias.fill_(math.log(math.expm1(0.01)))
            router.step_head.bias.zero_()
        inputs = self.inputs()
        result, diagnostics = router(**inputs)
        self.assertEqual(diagnostics["b3_selected_index"].item(), 1)
        self.assertEqual(
            int((diagnostics["b3_applied_weight"] > 0).sum().item()), 1)
        self.assertGreater(result[0, 0].item(), 0.0)
        self.assertLessEqual(result[0, 0].item(), 0.35 + 1e-6)
        torch.testing.assert_close(result[:, 2:], inputs["observation_box"][:, 2:])

    def test_disabled_or_nonfinite_candidates_are_exact_identity(self):
        router = self.build_router().eval()
        with torch.no_grad():
            router.median_gain_head.bias.fill_(1.0)
            router.gain_spread_head.bias.fill_(math.log(math.expm1(0.01)))
        inputs = self.inputs()
        disabled, _ = router(**inputs, enabled_scale=0.0)
        self.assertTrue(torch.equal(disabled, inputs["observation_box"]))
        inputs["motion_valid"] = torch.zeros(1)
        inputs["search_proposal_xy"] = torch.full((1, 2), float("nan"))
        invalid, diagnostics = router(**inputs)
        self.assertTrue(torch.equal(invalid, inputs["observation_box"]))
        self.assertEqual(diagnostics["b3_selected_index"].item(), 0)

    def test_export_feature_prediction_shape_and_normalization(self):
        router = self.build_router()
        features = torch.randn(3, router.export_feature_dim)
        mean = torch.linspace(-1.0, 1.0, router.scalar_dim)
        std = torch.linspace(0.5, 1.5, router.scalar_dim)
        router.set_scalar_normalization(mean, std)
        prediction = router.predict_exported_features(features)
        self.assertEqual(prediction["q10"].shape, (3, 2))
        self.assertTrue(torch.all(prediction["q10"] <= prediction["q50"]))


class CrpaOfflineContractsTest(unittest.TestCase):
    def test_counterfactual_targets_obey_bounded_step(self):
        result = counterfactual_gain_targets(
            observation_xy=torch.tensor([[0.0, 0.0]]),
            target_xy=torch.tensor([[1.0, 0.0]]),
            candidate_residual_xy=torch.tensor([[[2.0, 0.0], [0.0, 1.0]]]),
            candidate_valid=torch.ones(1, 2),
            step_cap=torch.tensor([0.35]),
        )
        torch.testing.assert_close(
            result["oracle_alpha"], torch.tensor([[0.35, 0.0]]))
        torch.testing.assert_close(
            result["oracle_gain"], torch.tensor([[0.70, 0.0]]), atol=1e-5, rtol=0)
        torch.testing.assert_close(
            result["oracle_step_ratio"], torch.tensor([[1.0, 0.0]]))

    def test_router_loss_is_finite_and_backpropagates(self):
        router = ClosedLoopRiskAwareProposalRouter(
            observation_dim=4, motion_dim=3, search_dim=2,
            context_dim=4, hidden_dim=8)
        prediction = router.predict_exported_features(
            torch.randn(5, router.export_feature_dim))
        losses = crpa_router_loss(
            prediction,
            oracle_gain=torch.rand(5, 2) * 0.2,
            oracle_step_ratio=torch.rand(5, 2),
            candidate_valid=torch.ones(5, 2),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(router.median_gain_head.weight.grad)

    def test_tracklet_partition_is_deterministic_and_atomic(self):
        keys = [f"tracklet-{index}" for index in range(100)]
        first = stable_tracklet_partition(keys + keys[:10], seed=42)
        second = stable_tracklet_partition(reversed(keys), seed=42)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "dev", "calibration"})

    def test_calibration_selects_safe_sparse_interventions(self):
        class StubRouter:
            def eval(self):
                return self

            def predict_exported_features(self, features):
                q10 = features[:, :2]
                return {
                    "q10": q10,
                    "q50": q10,
                    "step_ratio": torch.ones_like(q10),
                }

        rows = 20
        features = np.full((rows, 2), -0.1, dtype=np.float32)
        features[:4, 0] = 0.2
        valid = np.zeros((rows, 2), dtype=np.float32)
        valid[:, 0] = 1.0
        residual = np.zeros((rows, 2, 2), dtype=np.float32)
        residual[:, 0, 0] = 1.0
        target = np.zeros((rows, 2), dtype=np.float32)
        target[:4, 0] = 1.0
        arrays = {
            "router_features": features,
            "candidate_valid": valid,
            "candidate_residual_xy": residual,
            "observation_xy": np.zeros((rows, 2), dtype=np.float32),
            "target_xy": target,
            "step_cap": np.ones(rows, dtype=np.float32),
        }
        result = calibrate_router(
            StubRouter(), arrays, np.ones(rows, dtype=bool),
            torch.device("cpu"))
        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(result["chosen"]["coverage"], 0.2)
        self.assertEqual(result["chosen"]["harm_rate"], 0.0)
        self.assertEqual(result["chosen"]["precision"], 1.0)

    def test_router_sidecar_roundtrip_is_strict(self):
        router = ClosedLoopRiskAwareProposalRouter(
            observation_dim=4, motion_dim=3, search_dim=2,
            context_dim=4, hidden_dim=8)
        router.set_gain_threshold(0.12)
        payload = {
            "schema": CRPA_ROUTER_SCHEMA,
            "router_state_dict": router.state_dict(),
            "calibration": {"status": "passed"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.pt"
            torch.save(payload, path)
            restored = ClosedLoopRiskAwareProposalRouter(
                observation_dim=4, motion_dim=3, search_dim=2,
                context_dim=4, hidden_dim=8)
            load_router_sidecar(restored, path)
            self.assertAlmostEqual(
                restored.calibrated_gain_threshold.item(), 0.12, places=6)
            for key, value in router.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_b3_config_selects_crpa_exclusively(self):
        config = load_yaml_config(
            Path(__file__).resolve().parents[1]
            / "cfgs/ct_v2/10_b3_crpa_v1.yaml")
        self.assertTrue(config["use_b3_risk_router"])
        self.assertFalse(config["use_advantage_proposal_fusion"])
        self.assertEqual(config["b3_normal_step_cap"], 0.35)
        self.assertEqual(config["b3_gap_step_cap"], 0.60)


if __name__ == "__main__":
    unittest.main()

