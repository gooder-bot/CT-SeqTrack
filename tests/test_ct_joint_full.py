import math
import unittest
from pathlib import Path

import torch

from models.ct_v2 import (
    calibrate_joint_router_threshold,
    counterfactual_query_targets,
    JointFullSearchRefiner,
    JointScalarResidualRouter,
    OrderedPhysicalMotionEncoder,
)
from utils.config import load_yaml_config
from tools.bootstrap_ct_joint_results import (
    paired_bootstrap,
    paired_bootstrap_multiseed,
)


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

    def test_v2_low_presence_is_continuous_evidence_not_a_hard_veto(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0,
            contract_version=2, presence_hard_gate=False,
            presence_init_probability=0.01,
            presence_threshold=0.5).eval()
        output = module(**self.inputs())
        self.assertTrue(bool((output[
            "ct_search_presence_probability"] < 0.5).all()))
        self.assertTrue(bool((output["ct_search_available"] == 1).all()))
        self.assertTrue(bool((output["ct_search_effective"] == 1).all()))

    def test_v2_counterfactual_arms_do_not_read_predicted_alpha(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0,
            contract_version=2, presence_hard_gate=False).eval()
        with torch.no_grad():
            module.motion_query_projection.weight.normal_(0.0, 0.1)
            module.query_gate[-1].bias.fill_(-4.0)
        low_alpha = module(**self.inputs())
        with torch.no_grad():
            module.query_gate[-1].bias.fill_(4.0)
        high_alpha = module(**self.inputs())
        self.assertTrue(torch.equal(
            low_alpha["ct_search_raw_obs_xy"],
            high_alpha["ct_search_raw_obs_xy"]))
        self.assertTrue(torch.equal(
            low_alpha["ct_search_raw_motion_xy"],
            high_alpha["ct_search_raw_motion_xy"]))
        self.assertFalse(torch.equal(
            low_alpha["ct_search_raw_alpha_xy"],
            high_alpha["ct_search_raw_alpha_xy"]))
        torch.testing.assert_close(
            low_alpha["ct_search_presence_probability"],
            high_alpha["ct_search_presence_probability"])

    def test_alpha_probability_is_pre_dropout_calibration_score(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.5,
            contract_version=2, presence_hard_gate=False).train()
        torch.manual_seed(3)
        output = module(**self.inputs())
        torch.testing.assert_close(
            output["ct_query_gate_probability"],
            torch.sigmoid(output["ct_query_gate_logit"]))
        self.assertTrue(bool((output[
            "ct_query_gate_probability"] > 0).all()))

    def test_v2_motion_counterfactual_arm_trains_query_projection(self):
        module = JointFullSearchRefiner(
            motion_dim=16, motion_dropout=0.0,
            contract_version=2, presence_hard_gate=False).train()
        output = module(**self.inputs())
        loss = (
            output["ct_search_targetness_logits_obs"].mean()
            + output["ct_search_targetness_logits_motion"].mean()
            + output["ct_search_raw_obs_xy"].mean()
            + output["ct_search_raw_motion_xy"].mean())
        loss.backward()
        gradient = module.motion_query_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_counterfactual_labels_depend_only_on_two_detached_arms(self):
        raw_obs = torch.tensor(
            [[1.0, 0.0], [0.0, 0.0], [0.10, 0.0]],
            requires_grad=True)
        raw_motion = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.12, 0.0]],
            requires_grad=True)
        target = torch.zeros(3, 2)
        labels = counterfactual_query_targets(
            raw_obs, raw_motion, target, margin=0.05)
        torch.testing.assert_close(
            labels["target"], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(
            labels["valid"], torch.tensor([1.0, 1.0, 0.0]))
        self.assertFalse(labels["obs_error"].requires_grad)
        self.assertFalse(labels["motion_error"].requires_grad)

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

    def test_v2_quality_and_presence_diagnostics_cannot_veto_minus_b3(self):
        router = JointScalarResidualRouter(
            contract_version=2, presence_hard_gate=False).eval()
        observation = torch.zeros(2, 4)
        final, diagnostics = router(
            observation_box=observation,
            raw_search_xy=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            candidate_valid=torch.ones(2),
            observation_stats=torch.zeros(2, 5),
            targetness_mean=torch.zeros(2),
            targetness_max=torch.zeros(2),
            targetness_entropy=torch.zeros(2),
            normalized_ess=torch.zeros(2),
            query_gate=torch.zeros(2),
            query_delta_t=torch.ones(2),
            gap_ratio=torch.ones(2),
            extension_mass_ratio=torch.zeros(2),
            extension_vote_rms=torch.full((2,), float('inf')),
            presence_probability=torch.full((2,), 0.01),
            total_point_count=torch.tensor([3.0, 3.0]),
            extension_point_count=torch.ones(2),
            extension_voxels=torch.ones(2),
            coverage_need=torch.zeros(2),
            quality_valid=torch.zeros(2),
            enabled=False,
        )
        torch.testing.assert_close(
            diagnostics["ct_router_evidence_valid"], torch.ones(2))
        torch.testing.assert_close(
            diagnostics["ct_router_applied_gate"], torch.ones(2))
        self.assertFalse(torch.equal(final, observation))

    def test_nonfinite_router_context_abstains_with_finite_outputs(self):
        router = JointScalarResidualRouter().eval()
        observation = torch.randn(2, 4)
        final, diagnostics = router(
            observation_box=observation,
            raw_search_xy=torch.randn(2, 2),
            candidate_valid=torch.tensor([float("nan"), 1.0]),
            observation_stats=torch.full((2, 5), float("nan")),
            targetness_mean=torch.full((2,), float("nan")),
            targetness_max=torch.ones(2),
            targetness_entropy=torch.full((2,), float("inf")),
            normalized_ess=torch.ones(2),
            query_gate=torch.zeros(2),
            query_delta_t=torch.tensor([float("nan"), float("inf")]),
            gap_ratio=torch.tensor([float("nan"), float("inf")]),
            extension_mass_ratio=torch.full((2,), float("nan")),
            extension_vote_rms=torch.zeros(2),
            presence_probability=torch.ones(2),
            enabled=False,
        )
        self.assertTrue(bool(torch.isfinite(final).all()))
        self.assertTrue(bool(torch.isfinite(
            diagnostics["ct_router_logit"]).all()))
        self.assertTrue(torch.equal(final, observation))

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
        self.assertEqual(full["ct_query_dim"], 64)
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

    def test_repaired_config_is_versioned_without_mutating_v1(self):
        legacy = load_yaml_config(
            ROOT / "cfgs/ct_v2/21_ct_joint_full.yaml")
        repaired = load_yaml_config(
            ROOT / "cfgs/ct_v2/22_ct_joint_repaired.yaml")
        self.assertNotIn("ct_joint_contract_version", legacy)
        self.assertEqual(repaired["ct_joint_contract_version"], 2)
        self.assertEqual(
            repaired["ct_recursive_rollout_horizons"], [1, 2, 4, 8])
        self.assertTrue(repaired["use_b1_prepass_support"])
        self.assertFalse(repaired["ct_presence_hard_gate"])
        self.assertTrue(repaired["ct_recursive_reseed_enabled"])
        self.assertEqual(repaired["ct_partition_seed"], 42)
        for name in (
                "22_ct_joint_repaired_b0.yaml",
                "22_ct_joint_repaired_minus_b1.yaml",
                "22_ct_joint_repaired_minus_b2.yaml",
                "22_ct_joint_repaired_minus_b3.yaml",
                "22_ct_joint_repaired_fault_old_recursive.yaml",
                "22_ct_joint_repaired_fault_presence_hard.yaml",
                "22_ct_joint_repaired_fault_alpha_self.yaml",
                "22_ct_joint_repaired_fault_kinematic_search.yaml",
                "22_ct_joint_repaired_full.yaml",
                "22_ct_joint_repaired_b0_full.yaml",
                "22_ct_joint_repaired_minus_b1_full.yaml",
                "22_ct_joint_repaired_minus_b2_full.yaml",
                "22_ct_joint_repaired_minus_b3_full.yaml"):
            config = load_yaml_config(ROOT / "cfgs/ct_v2" / name)
            self.assertEqual(config["ct_joint_contract_version"], 2)
        for name in (
                "22_ct_joint_repaired_b0.yaml",
                "22_ct_joint_repaired_minus_b1.yaml",
                "22_ct_joint_repaired_b0_full.yaml",
                "22_ct_joint_repaired_minus_b1_full.yaml"):
            config = load_yaml_config(ROOT / "cfgs/ct_v2" / name)
            self.assertFalse(config["ct_enable_b1"])
            self.assertFalse(config["use_b1_prepass_support"])

    def test_h3_shadow_bypasses_structural_b1_and_joint_full_paths(self):
        source = (ROOT / "models/seqtrack3d.py").read_text(encoding="utf-8")
        shadow_source = source.split(
            "    def _shadow_forward(self, batch, seed):", 1)[1].split(
                "    def _attach_h3_shadow_labels", 1)[0]
        for assignment in (
                "self.use_ct_joint_full = False",
                "self.use_b1motion_v3 = False",
                "self.use_ct_joint_full = previous_joint_full",
                "self.use_b1motion_v3 = previous_motion_v3"):
            self.assertIn(assignment, shadow_source)

    def test_joint_diagnostic_export_has_a_schema_specific_aggregator(self):
        source = (ROOT / "models/base_model.py").read_text(encoding="utf-8")
        row_source = source.split(
            "    def _build_ct_joint_diagnostic_row", 1)[1].split(
                "    @staticmethod\n    def _write_csv_rows", 1)[0]
        for field in (
                '"search_geometry_source_id"',
                '"raw_obs_error"',
                '"raw_motion_error"',
                '"alpha_counterfactual_uplift"',
                '"presence_probability"',
                '"presence_target"'):
            self.assertIn(field, row_source)
        writer_source = source.split(
            "    def _write_proposal_test_diagnostics(self):", 1)[1].split(
                "    def _write_b3_test_rollouts", 1)[0]
        self.assertIn(
            'if "router_applied_gate" in group[0]:', writer_source)
        self.assertIn(
            'if bool(row["router_applied_gate"])', writer_source)

    def test_main_supports_bounded_validation_preflight(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("'--limit_val_batches'", source)
        self.assertIn(
            "limit_val_batches=getattr(\n"
            "                             cfg, 'limit_val_batches', 1.0)",
            source)

    def test_router_calibration_uses_deployment_recursive_state(self):
        source = (ROOT / "tools/export_ct_joint_router_calibration.py").read_text(
            encoding="utf-8")
        self.assertIn('"ct_recursive_reseed_enabled": False', source)
        self.assertIn(
            '"recursive_state_policy": "deployment_no_reseed"', source)

    def test_masked_mean_is_available_when_b1_is_disabled(self):
        source = (ROOT / "models/seqtrack3d.py").read_text(encoding="utf-8")
        loss_source = source.split(
            "    def compute_loss(self, data, output):", 1)[1].split(
                "    def on_train_epoch_start", 1)[0]
        helper_offset = loss_source.index(
            "        def masked_mean(per_sample, valid):")
        b1_branch_offset = loss_source.index(
            "        if (self.use_b1motion_v3")
        joint_branch_offset = loss_source.index(
            "        if self.use_ct_joint_full and self.ct_enable_b2:")
        self.assertLess(helper_offset, b1_branch_offset)
        self.assertLess(helper_offset, joint_branch_offset)

    def test_minus_b1_does_not_train_the_motion_counterfactual_arm(self):
        source = (ROOT / "models/seqtrack3d.py").read_text(encoding="utf-8")
        loss_source = source.split(
            "    def compute_loss(self, data, output):", 1)[1].split(
            "    def on_train_epoch_start", 1)[0]
        self.assertIn(
            "counterfactual_arms_enabled = bool(\n"
            "                self.ct_joint_contract_version >= 2\n"
            "                and self.ct_query_counterfactual_supervision\n"
            "                and self.ct_enable_b1)",
            loss_source)

    def test_inference_prepass_obeys_b1_ablation_gate(self):
        source = (ROOT / "models/base_model.py").read_text(encoding="utf-8")
        sequence_source = source.split(
            "    def evaluate_one_sequence(self, sequence):", 1)[1].split(
            "    def _m4_timestamp", 1)[0]
        self.assertIn('"use_b1_prepass_support", False', sequence_source)
        self.assertIn('"ct_enable_b1", True', sequence_source)

    def test_tracklet_paired_bootstrap_uses_exact_endpoint_keys(self):
        baseline = {
            ("a", 1): (0.2, 1.0),
            ("a", 2): (0.4, 0.8),
            ("b", 1): (0.3, 1.2),
        }
        repaired = {
            key: (iou + 0.2, max(0.0, distance - 0.3))
            for key, (iou, distance) in baseline.items()}
        result = paired_bootstrap(
            baseline, repaired, draws=100, seed=7)
        self.assertGreater(
            result["delta"]["success"]["point_estimate"], 0.0)
        self.assertGreater(
            result["delta"]["precision"]["point_estimate"], 0.0)
        with self.assertRaises(ValueError):
            paired_bootstrap(
                baseline, {("a", 1): repaired[("a", 1)]}, draws=2)

    def test_bootstrap_does_not_duplicate_an_exported_initial_frame(self):
        baseline = {
            ("a", 0): (1.0, 0.0),
            ("a", 1): (0.2, 1.0),
        }
        repaired = {
            ("a", 0): (1.0, 0.0),
            ("a", 1): (0.4, 0.7),
        }
        result = paired_bootstrap(
            baseline, repaired, draws=10, seed=9)
        self.assertEqual(result["endpoint_count_including_initial"], 2)

    def test_multiseed_bootstrap_reports_mean_across_identical_endpoints(self):
        baseline = {
            ("a", 0): (1.0, 0.0),
            ("a", 1): (0.2, 1.0),
            ("b", 0): (1.0, 0.0),
            ("b", 1): (0.3, 1.2),
        }
        repaired_a = {
            key: (iou + (0.0 if key[1] == 0 else 0.2), distance)
            for key, (iou, distance) in baseline.items()}
        repaired_b = {
            key: (iou + (0.0 if key[1] == 0 else 0.1), distance)
            for key, (iou, distance) in baseline.items()}
        result = paired_bootstrap_multiseed(
            [baseline, baseline], [repaired_a, repaired_b],
            draws=20, seed=11)
        self.assertEqual(result["seed_count"], 2)
        self.assertGreater(result["delta"]["success"]["point_estimate"], 0)

    def test_metric_endpoint_export_covers_empty_and_initial_frames(self):
        source = (ROOT / "models/base_model.py").read_text(encoding="utf-8")
        test_source = source.split(
            "    def test_step(self, batch, batch_idx):", 1)[1].split(
            "    def on_test_epoch_start", 1)[0]
        self.assertIn("zip(ious, distances)", test_source)
        self.assertIn('"tracking_endpoints.csv"', source)


if __name__ == "__main__":
    unittest.main()
