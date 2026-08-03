import unittest

import numpy as np
import torch

from models.ct_v2.selective_innovation import (
    MotionConditionedSearchRefiner,
    SignedHorizonInnovationRouter,
    calibrate_gain_threshold,
    discounted_tracking_cost,
    signed_horizon_router_loss,
    stable_tracklet_partition,
)


class MotionConditionedSearchRefinerTest(unittest.TestCase):
    def _forward(self, second_row_valid=True):
        torch.manual_seed(3)
        batch, points = 2, 8
        module = MotionConditionedSearchRefiner()
        mask = torch.ones(batch, points)
        geometry = torch.ones(batch)
        motion_valid = torch.ones(batch)
        if not second_row_valid:
            mask[1].zero_()
            geometry[1] = 0
            motion_valid[1] = 0
        motion_xy = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
        output = module(
            torch.randn(batch, points, 9),
            torch.randn(batch, points, 2),
            torch.randn(batch, points, 2),
            mask,
            torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * batch),
            geometry,
            torch.randn(batch, 2),
            torch.randn(batch, 256),
            torch.randn(batch, 128),
            motion_xy,
            motion_valid,
            torch.randn(batch, 5),
            torch.tensor([0.5, 0.5]),
            torch.ones(batch),
            torch.ones(batch),
            torch.ones(batch),
            torch.tensor([8.0, 0.0 if not second_row_valid else 8.0]),
            torch.tensor([4.0, 0.0 if not second_row_valid else 4.0]),
            torch.tensor([4.0, 0.0 if not second_row_valid else 4.0]),
        )
        return module, motion_xy, output

    def test_refined_candidate_is_centered_on_motion_and_bounded(self):
        module, motion_xy, output = self._forward()
        residual = output["motion_search_refined_xy"] - motion_xy
        radius = output["search_v22_refinement_radius"]
        self.assertTrue(torch.all(
            torch.linalg.norm(residual, dim=1) <= radius + 1e-6))
        self.assertTrue(torch.allclose(
            residual, output["motion_search_refinement_residual_xy"]))
        self.assertEqual(module.point_mlp[0].in_features, 9)

    def test_invalid_row_has_zero_finite_statistics(self):
        _, _, output = self._forward(second_row_valid=False)
        self.assertEqual(float(output["search_raw_ess"][1]), 0.0)
        self.assertEqual(float(output["search_normalized_ess"][1]), 0.0)
        self.assertEqual(
            float(output["motion_search_candidate_available"][1]), 0.0)
        for value in output.values():
            if torch.is_tensor(value):
                self.assertTrue(torch.isfinite(value).all())

    def test_presence_is_not_availability(self):
        _, _, output = self._forward()
        self.assertTrue(torch.all(
            output["motion_search_candidate_available"] == 1))
        # Cold-start presence is 0.1 and therefore safely abstains.
        self.assertTrue(torch.allclose(
            output["search_presence_probability"],
            torch.full((2,), 0.1), atol=1e-5))
        self.assertTrue(torch.all(
            output["motion_search_candidate_valid"] == 0))


class SignedHorizonRouterTest(unittest.TestCase):
    def _inputs(self, requires_grad=False, gap=1.0):
        batch = 2
        def tensor(*shape):
            return torch.randn(*shape, requires_grad=requires_grad)
        return dict(
            observation_box=tensor(batch, 4),
            observation_feature=tensor(batch, 256),
            observation_stats=tensor(batch, 5),
            observation_entropy=torch.rand(batch),
            observation_refinement_xy=tensor(batch, 2),
            motion_feature=tensor(batch, 128),
            motion_proposal_xy=tensor(batch, 2),
            motion_log_sigma_xy=tensor(batch, 2),
            motion_valid=torch.ones(batch),
            history_valid_ratio=torch.ones(batch),
            search_feature=tensor(batch, 128),
            motion_search_xy=tensor(batch, 2),
            motion_search_valid=torch.ones(batch),
            search_presence=torch.ones(batch),
            search_targetness_mean=torch.rand(batch),
            search_targetness_max=torch.rand(batch),
            search_targetness_entropy=torch.rand(batch),
            search_normalized_ess=torch.rand(batch),
            search_extension_weight_ratio=torch.rand(batch),
            search_available_count=torch.full((batch,), 8.0),
            search_extension_count=torch.full((batch,), 4.0),
            search_overlap_count=torch.full((batch,), 4.0),
            search_support_anchor_xy=tensor(batch, 2),
            search_raw_vote_xy=tensor(batch, 2),
            query_delta_t=torch.full((batch,), 0.5),
            gap_ratio=torch.full((batch,), gap),
        )

    def test_cold_router_is_bitwise_observation_fallback(self):
        router = SignedHorizonInnovationRouter()
        inputs = self._inputs()
        final, diagnostics = router(**inputs)
        self.assertTrue(torch.equal(final, inputs["observation_box"]))
        self.assertTrue(torch.all(diagnostics["signed_abstained"] == 1))
        self.assertEqual(diagnostics["signed_gain_quantiles"].shape, (2, 2, 2))
        self.assertEqual(diagnostics["signed_step_logits"].shape, (2, 2, 3))

    def test_forced_top1_uses_discrete_step_and_protocol_caps(self):
        router = SignedHorizonInnovationRouter()
        normal_inputs = self._inputs(gap=1.0)
        _, normal = router(
            **normal_inputs,
            forced_candidate=torch.tensor([0, 1]),
            forced_step_ratio=torch.ones(2))
        self.assertTrue(torch.allclose(
            normal["signed_applied_alpha"], torch.full((2,), 0.20)))
        self.assertTrue(torch.equal(
            normal["signed_selected_candidate"], torch.tensor([1, 2])))
        gap_inputs = self._inputs(gap=2.0)
        _, gap = router(
            **gap_inputs,
            forced_candidate=torch.tensor([0, 1]),
            forced_step_ratio=torch.ones(2))
        self.assertTrue(torch.allclose(
            gap["signed_applied_alpha"], torch.full((2,), 0.35)))

    def test_router_features_detach_candidate_producers(self):
        router = SignedHorizonInnovationRouter()
        inputs = self._inputs(requires_grad=True)
        _, diagnostics = router(**inputs)
        diagnostics["signed_gain_quantiles"].sum().backward()
        self.assertIsNone(inputs["observation_feature"].grad)
        self.assertIsNone(inputs["motion_feature"].grad)
        self.assertIsNone(inputs["search_feature"].grad)
        self.assertIsNone(inputs["motion_proposal_xy"].grad)
        self.assertIsNotNone(router.median_gain_head.weight.grad)


class SignedHorizonSupervisionTest(unittest.TestCase):
    def test_negative_gain_is_preserved_by_loss(self):
        router = SignedHorizonInnovationRouter(
            observation_dim=4, motion_dim=3, search_dim=3)
        features = torch.randn(2, router.export_feature_dim)
        prediction = router.predict_export_features(features)
        gain = torch.tensor([
            [[-0.3, -0.2, -0.1], [-0.4, -0.5, -0.6]],
            [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
        ])
        losses = signed_horizon_router_loss(
            prediction, gain, torch.ones(2, 2))
        self.assertLess(float(losses["best_signed_gain"][0, 0]), 0.0)
        self.assertLess(float(losses["best_signed_gain"][0, 1]), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_discounted_cost_and_harmful_gain(self):
        observation_cost = discounted_tracking_cost(
            [0.9, 0.9, 0.9], [0.1, 0.1, 0.1])
        harmful_cost = discounted_tracking_cost(
            [0.2, 0.2, 0.2], [2.0, 2.0, 2.0])
        self.assertLess(observation_cost - harmful_cost, 0.0)

    def test_calibration_uses_applied_step(self):
        rows = 20
        q10 = np.stack((
            np.linspace(0.01, 0.20, rows),
            np.linspace(0.0, 0.10, rows)), axis=1)
        gain = np.full((rows, 2, 3), -0.1, dtype=np.float64)
        gain[:, 0, 0] = 0.05
        gain[:, 1, 0] = 0.04
        valid = np.ones((rows, 2), dtype=bool)
        step_class = np.zeros((rows, 2), dtype=np.int64)
        result = calibrate_gain_threshold(
            q10, gain, valid, step_class=step_class,
            min_coverage=0.05, max_coverage=0.25)
        self.assertGreaterEqual(result["helpful_precision"], 0.75)
        self.assertLessEqual(result["harm_rate"], 0.10)

    def test_tracklet_partition_is_stable(self):
        first = stable_tracklet_partition("scene/track/1", seed=42)
        second = stable_tracklet_partition("scene/track/1", seed=42)
        self.assertEqual(first, second)
        self.assertIn(first, ("train", "dev", "calibration"))


if __name__ == "__main__":
    unittest.main()
