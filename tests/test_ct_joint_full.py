import math
import unittest
from pathlib import Path

import torch

from models.ct_v2 import (
    calibrate_joint_router_threshold,
    JointFullSearchRefiner,
    JointScalarResidualRouter,
    OrderedPhysicalMotionEncoder,
)
from utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]


class SharedKinematicAnchorTest(unittest.TestCase):
    def build_encoder(self):
        return OrderedPhysicalMotionEncoder(
            hidden_dim=16,
            step_dim=8,
            shared_kinematic_anchor=True,
            max_acceleration=8.0,
            max_displacement=12.0,
            acceleration_weight=0.5,
        )

    def test_anchor_and_envelope_match_deterministic_formula(self):
        encoder = self.build_encoder()
        boxes = torch.tensor([[[
            3.0, 0.0, 0.0, 0.0
        ], [
            1.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])
        output = encoder(
            boxes, torch.ones(1, 3), torch.ones(1, 3),
            current_delta_t=torch.tensor([2.0]))

        # recent v=2, older v=1, a=1; 2*2 + .25*1*2^2 = 5.
        torch.testing.assert_close(
            output["kinematic_prior_xy"], torch.tensor([[5.0, 0.0]]))
        torch.testing.assert_close(
            output["envelope_parallel_perp"],
            torch.tensor([[1.75, 1.0]]))
        torch.testing.assert_close(
            output["prior_xy"], output["kinematic_prior_xy"])

    def test_learned_center_is_bounded_by_shared_envelope(self):
        encoder = self.build_encoder()
        with torch.no_grad():
            encoder.velocity_residual_head.bias.copy_(
                torch.tensor([20.0, -20.0]))
        boxes = torch.tensor([[[
            2.0, 0.0, 0.0, 0.0
        ], [
            1.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])
        output = encoder(
            boxes, torch.ones(1, 3), torch.ones(1, 3),
            current_delta_t=torch.tensor([1.0]))

        unit = output["residual_unit_parallel_perp"]
        self.assertTrue(bool((unit.abs() <= 1.0).all()))
        sigma = torch.exp(output["log_sigma_parallel_perp"])
        self.assertTrue(bool((sigma >= 0.1).all()))
        self.assertTrue(bool((sigma <= output[
            "envelope_parallel_perp"] + 1e-6).all()))

    def test_invalid_history_has_exact_zero_learned_residual(self):
        encoder = self.build_encoder()
        output = encoder(
            torch.zeros(2, 3, 4),
            torch.ones(2, 3),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            current_delta_t=torch.ones(2))
        torch.testing.assert_close(output["valid"], torch.zeros(2))
        torch.testing.assert_close(
            output["residual_xy"], torch.zeros(2, 2))
        torch.testing.assert_close(output["prior_xy"], torch.zeros(2, 2))

    def test_b1_off_fallback_matches_kinematics_without_parameter_use(self):
        encoder = self.build_encoder().eval()
        boxes = torch.tensor([[[
            3.0, 0.0, 0.0, 0.0
        ], [
            1.0, 0.0, 0.0, 0.0
        ], [
            0.0, 0.0, 0.0, 0.0
        ]]])
        delta_t = torch.ones(1, 3)
        valid = torch.ones(1, 3)
        learned = encoder(
            boxes, delta_t, valid, current_delta_t=torch.tensor([2.0]))
        fallback = encoder.kinematic_fallback(
            boxes, delta_t, valid, current_delta_t=torch.tensor([2.0]))
        torch.testing.assert_close(
            fallback["prior_xy"], learned["kinematic_prior_xy"])
        self.assertTrue(torch.equal(
            fallback["residual_xy"], torch.zeros_like(
                fallback["residual_xy"])))
        self.assertFalse(fallback["prior_xy"].requires_grad)


class JointSearchCouplingTest(unittest.TestCase):
    def inputs(self, motion_scale=1.0, motion_valid=1.0):
        torch.manual_seed(7)
        batch, points = 2, 12
        return {
            "encoded_points": torch.randn(batch, points, 5),
            "point_xy": torch.randn(batch, points, 2),
            "point_valid_mask": torch.ones(batch, points),
            "point_source": torch.cat((
                torch.zeros(batch, points // 2),
                torch.ones(batch, points // 2)), dim=1).long(),
            "observation_query": torch.randn(batch, 64),
            "observation_stats": torch.randn(batch, 5),
            "motion_feature": motion_scale * torch.randn(batch, 16),
            "kinematic_xy": torch.tensor([[1.0, 0.0], [0.5, 0.2]]),
            "learned_xy": torch.tensor([[1.2, 0.1], [0.3, 0.3]]),
            "residual_unit": torch.tensor([[0.2, 0.1], [-0.2, 0.1]]),
            "sigma_parallel_perp": torch.full((batch, 2), 0.5),
            "envelope_parallel_perp": torch.ones(batch, 2),
            "direction_xy": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "motion_valid": torch.full((batch,), motion_valid),
            "query_delta_t": torch.tensor([0.5, 2.0]),
            "gap_ratio": torch.tensor([1.0, 4.0]),
            "history_valid_ratio": torch.ones(batch),
        }

    def test_invalid_b1_is_exact_observation_query_fallback(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).eval()
        output = module(**self.inputs(motion_valid=0.0))
        torch.testing.assert_close(
            output["ct_query_gate"], torch.zeros(2))
        self.assertTrue(torch.equal(
            output["ct_query_search"],
            output["ct_query_observation"]))

    def test_layer_norm_prevents_motion_scale_from_scaling_query_norm(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).eval()
        small = module(**self.inputs(motion_scale=1.0))
        large = module(**self.inputs(motion_scale=100.0))
        small_norm = torch.linalg.norm(
            small["ct_query_residual"], dim=1)
        large_norm = torch.linalg.norm(
            large["ct_query_residual"], dim=1)
        torch.testing.assert_close(small_norm, large_norm, atol=2e-4, rtol=2e-4)

    def test_query_gate_zero_removes_only_motion_match_residual(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).eval()
        output = module(**self.inputs(motion_valid=0.0))
        self.assertTrue(bool(torch.isfinite(
            output["ct_search_targetness_logits"]).all()))
        self.assertTrue(bool((output[
            "ct_search_candidate_valid"] == 1).all()))

    def test_explicit_alpha_zero_is_bitwise_query_noop(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0,
            query_gate_scale=0.0).eval()
        output = module(**self.inputs())
        self.assertTrue(torch.equal(
            output["ct_query_search"],
            output["ct_query_observation"]))

    def test_support_invalid_is_exact_query_fallback_and_no_candidate(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).eval()
        inputs = self.inputs()
        inputs["search_support_valid"] = torch.zeros(2)
        output = module(**inputs)
        self.assertTrue(torch.equal(
            output["ct_query_search"], output["ct_query_observation"]))
        self.assertTrue(bool((output[
            "ct_search_candidate_valid"] == 0).all()))

    def test_refiner_executes_only_the_support_valid_subbatch(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).eval()
        inputs = self.inputs()
        inputs["search_support_valid"] = torch.tensor([1.0, 0.0])
        seen_batch_sizes = []

        def record_batch_size(_module, arguments, _output):
            seen_batch_sizes.append(int(arguments[0].shape[0]))

        handle = module.point_encoder[0].register_forward_hook(
            record_batch_size)
        try:
            output = module(**inputs)
        finally:
            handle.remove()
        self.assertEqual(seen_batch_sizes, [1])
        self.assertTrue(torch.equal(
            output["ct_query_search"][1],
            output["ct_query_observation"][1]))
        self.assertEqual(
            float(output["ct_search_candidate_valid"][1]), 0.0)

    def test_overlap_only_points_are_not_router_candidates(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).eval()
        inputs = self.inputs()
        inputs["point_source"] = torch.zeros_like(inputs["point_source"])
        output = module(**inputs)
        self.assertTrue(bool((output[
            "ct_search_structural_valid"] == 1).all()))
        self.assertTrue(bool((output[
            "ct_search_new_support_valid"] == 0).all()))
        self.assertTrue(bool((output[
            "ct_search_candidate_valid"] == 0).all()))

    def test_presence_rejection_is_exact_deployed_query_noop(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0,
            presence_init_probability=0.1,
            presence_threshold=0.5).eval()
        output = module(**self.inputs())
        self.assertTrue(bool((output["ct_search_effective"] == 0).all()))
        self.assertTrue(torch.equal(
            output["ct_query_search"],
            output["ct_query_observation"]))
        self.assertTrue(bool((output["ct_query_gate"] == 0).all()))
        self.assertTrue(bool((output[
            "ct_query_gate_internal"] > 0).all()))

    def test_joint_backward_detaches_motion_but_reaches_point_heads(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).train()
        inputs = self.inputs()
        inputs["motion_feature"].requires_grad_()
        output = module(**inputs)
        loss = (
            output["ct_search_targetness_logits"].mean()
            + output["ct_search_raw_xy"].mean()
            + output["ct_query_gate"].mean())
        loss.backward()
        self.assertIsNone(inputs["motion_feature"].grad)
        self.assertIsNotNone(module.vote_head[-1].weight.grad)

    def test_three_way_branch_source_contract_accepts_tube_id(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0).eval()
        inputs = self.inputs()
        inputs["point_branch_source"] = torch.cat((
            torch.ones(2, 6, dtype=torch.long),
            torch.full((2, 6), 2, dtype=torch.long)), dim=1)
        output = module(**inputs)
        self.assertEqual(output["ct_search_raw_xy"].shape, (2, 2))


class JointRouterTest(unittest.TestCase):
    def test_calibration_never_lowers_training_threshold(self):
        result = calibrate_joint_router_threshold(
            probabilities=[0.95, 0.9, 0.4, 0.2] + [0.1] * 16,
            h3_gain=[0.3, 0.25, -0.2, -0.1] + [0.0] * 16,
            valid=[1, 1, 1, 1] + [0] * 16,
            min_coverage=0.05,
            max_coverage=0.25,
        )
        self.assertGreaterEqual(result["threshold"], 0.5)
        self.assertEqual(result["harm_rate"], 0.0)

    def test_cold_router_is_exact_abstain_and_minus_b3_is_full_step(self):
        router = JointScalarResidualRouter(init_probability=0.05).eval()
        observation = torch.zeros(2, 4)
        raw = torch.tensor([[1.0, 0.0], [4.0, 0.0]])
        common = dict(
            observation_box=observation,
            raw_search_xy=raw,
            candidate_valid=torch.ones(2),
            observation_stats=torch.zeros(2, 5),
            targetness_mean=torch.ones(2),
            targetness_max=torch.ones(2),
            targetness_entropy=torch.zeros(2),
            normalized_ess=torch.ones(2),
            query_gate=torch.zeros(2),
            query_delta_t=torch.tensor([1.0, 4.0]),
            gap_ratio=torch.ones(2),
        )
        full_box, full = router(**common, enabled=True)
        minus_b3_box, minus_b3 = router(**common, enabled=False)
        torch.testing.assert_close(
            full["ct_router_gate"], torch.full((2,), 0.05))
        self.assertTrue(torch.equal(full_box, observation))
        torch.testing.assert_close(
            full["ct_router_applied_gate"], torch.zeros(2))
        torch.testing.assert_close(
            minus_b3["ct_router_applied_gate"], torch.ones(2))
        # Radius is 1m for dt=1 and capped at 2m for dt=4.
        torch.testing.assert_close(
            minus_b3_box[:, 0], torch.tensor([1.0, 2.0]))

    def test_no_new_support_is_bitwise_observation_fallback(self):
        router = JointScalarResidualRouter().eval()
        observation = torch.randn(2, 4)
        final, diagnostics = router(
            observation_box=observation,
            raw_search_xy=torch.randn(2, 2),
            candidate_valid=torch.zeros(2),
            observation_stats=torch.zeros(2, 5),
            targetness_mean=torch.ones(2),
            targetness_max=torch.ones(2),
            targetness_entropy=torch.zeros(2),
            normalized_ess=torch.ones(2),
            query_gate=torch.zeros(2),
            query_delta_t=torch.ones(2),
            gap_ratio=torch.ones(2),
            enabled=False,
        )
        self.assertTrue(torch.equal(final, observation))
        self.assertTrue(bool((diagnostics[
            "ct_router_evidence_valid"] == 0).all()))

    def test_router_losses_do_not_backpropagate_to_candidate_producers(self):
        router = JointScalarResidualRouter().train()
        observation = torch.zeros(2, 4, requires_grad=True)
        raw = torch.tensor(
            [[1.0, 0.0], [0.5, 0.2]], requires_grad=True)
        final, diagnostics = router(
            observation_box=observation,
            raw_search_xy=raw,
            candidate_valid=torch.ones(2),
            observation_stats=torch.zeros(2, 5),
            targetness_mean=torch.ones(2),
            targetness_max=torch.ones(2),
            targetness_entropy=torch.zeros(2),
            normalized_ess=torch.ones(2),
            query_gate=torch.zeros(2),
            query_delta_t=torch.ones(2),
            gap_ratio=torch.ones(2),
        )
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                diagnostics["ct_router_logit"], torch.ones(2))
            + diagnostics["ct_router_soft_box"][:, :2].mean())
        loss.backward()
        self.assertIsNone(observation.grad)
        self.assertIsNone(raw.grad)
        self.assertIsNotNone(router.gate[-1].weight.grad)
        self.assertGreater(
            float(router.gate[-1].weight.grad.abs().sum()), 0.0)


class JointFullConfigTest(unittest.TestCase):
    def test_full_and_ablation_configs_are_decision_complete(self):
        full = load_yaml_config(
            ROOT / "cfgs/ct_v2/21_ct_joint_full.yaml")
        self.assertTrue(full["use_ct_joint_full"])
        self.assertTrue(full["use_b1motion_v3"])
        self.assertFalse(full["use_b1_prepass_support"])
        self.assertFalse(full["use_recursive_replay_cache"])
        self.assertEqual(full["motion_v3_warmup_epoch"], 0)
        self.assertTrue(full["ct_enable_shared_motion_anchor"])
        self.assertTrue(full["ct_enable_dynamic_residual_bound"])
        self.assertTrue(full["ct_enable_query_reliability_gate"])
        self.assertEqual(full["point_sample_size"], 1024)
        self.assertEqual(full["ct_expansion_point_count"], 256)
        self.assertTrue(full["ct_online_recursive_training"])
        self.assertEqual(full["candidate_trajectory_mode"], "shared_se2")
        self.assertEqual(full["ct_search_min_extension_points"], 8)
        self.assertEqual(full["ct_router_init_probability"], 0.01)
        self.assertNotIn("ct_correction_warmup_epochs", full)
        self.assertEqual(
            full["ct_endpoint_quota"] + full["ct_tube_quota"], 256)
        for name, key in (
                ("21_ct_joint_minus_b1.yaml", "ct_enable_b1"),
                ("21_ct_joint_minus_b2.yaml", "ct_enable_b2"),
                ("21_ct_joint_minus_b3.yaml", "ct_enable_b3")):
            config = load_yaml_config(ROOT / "cfgs/ct_v2" / name)
            self.assertFalse(config[key])
        b2_only = load_yaml_config(
            ROOT / "cfgs/ct_v2/21_ct_joint_b2_only.yaml")
        self.assertFalse(b2_only["ct_enable_b1"])
        self.assertTrue(b2_only["ct_enable_b2"])
        self.assertFalse(b2_only["ct_enable_b3"])
        self.assertTrue(b2_only["ct_online_recursive_training"])


if __name__ == "__main__":
    unittest.main()
