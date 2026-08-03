import unittest
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from models.ct_v2.selective_innovation import (
    ActionConsistentInnovationRouter,
    StateAlignedSearchRefiner,
    action_consistent_router_loss,
    calibrate_action_threshold,
    validate_b2_v3_router_package,
)
from utils.ct_history import (
    b2_v3_history_mode_id,
    select_b2_v3_history_mode,
)
from tools.build_b2_v3_init_checkpoint import (
    collect_migrated_search_state,
    validate_b1_state,
)
from tools.package_b2_v3_checkpoint import (
    PROTECTED_PREFIXES,
    tensor_hash as packaged_tensor_hash,
)
from tools.merge_b2_v3_rollouts import main as merge_rollouts_main
from tools.selective_v3_common import (
    load_matching_v3_model_state,
    load_v3_rollout_artifact,
    tensor_prefix_hash,
    write_v3_rollout_artifact,
)
from tools.train_action_router_v3 import validate_training_stage


class SharedHistoryContractTest(unittest.TestCase):
    def test_candidate_zero_is_canonical_and_assignment_is_stable(self):
        self.assertEqual(
            select_b2_v3_history_mode("track", 7, 0, 42), "canonical")
        first = select_b2_v3_history_mode("track", 7, 2, 42)
        second = select_b2_v3_history_mode("track", 7, 2, 42)
        self.assertEqual(first, second)
        self.assertIn(first, ("correlated_candidate", "recursive_candidate"))
        self.assertIn(b2_v3_history_mode_id(first), (1, 2))

    def test_assignment_does_not_consume_numpy_rng(self):
        np.random.seed(11)
        expected = np.random.rand(4)
        np.random.seed(11)
        select_b2_v3_history_mode("track", 8, 3, 42)
        actual = np.random.rand(4)
        self.assertTrue(np.array_equal(actual, expected))


class StateAlignedRefinerTest(unittest.TestCase):
    def _inputs(self):
        batch, points = 2, 8
        return dict(
            point_inputs=torch.randn(batch, points, 9),
            point_xy=torch.randn(batch, points, 2),
            delta_to_motion=torch.randn(batch, points, 2),
            point_valid_mask=torch.ones(batch, points),
            point_source=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * batch),
            geometry_valid=torch.ones(batch),
            support_anchor_xy=torch.randn(batch, 2),
            observation_feature=torch.randn(batch, 256),
            motion_feature=torch.randn(batch, 128),
            motion_proposal_xy=torch.randn(batch, 2),
            motion_valid=torch.ones(batch),
            observation_stats=torch.randn(batch, 5),
            query_delta_t=torch.full((batch,), 0.5),
            gap_ratio=torch.ones(batch),
            sigma_parallel=torch.ones(batch),
            sigma_perpendicular=torch.ones(batch),
            available_count=torch.full((batch,), 8.0),
            extension_count=torch.full((batch,), 4.0),
            overlap_count=torch.full((batch,), 4.0),
        )

    def test_low_presence_remains_structurally_valid(self):
        module = StateAlignedSearchRefiner()
        output = module(**self._inputs())
        self.assertFalse(hasattr(module, "source_fusion"))
        self.assertEqual(
            output["search_v3_evidence_components"].shape, (2, 384))
        self.assertTrue(torch.equal(
            output["search_v3_evidence_components"],
            torch.cat((
                output["search_v3_overlap_token"],
                output["search_v3_extension_token"],
                output["search_v3_motion_observation_context"],
            ), dim=1)))
        self.assertTrue(torch.allclose(
            output["search_v3_presence_probability"],
            torch.full((2,), 0.1), atol=1e-5))
        self.assertTrue(torch.all(
            output["motion_search_v3_candidate_structural_valid"] == 1))

    def test_new_paths_receive_gradient_across_two_steps(self):
        torch.manual_seed(5)
        module = StateAlignedSearchRefiner()
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        for _ in range(2):
            output = module(**self._inputs())
            loss = (
                output["search_v3_presence_logit"].mean()
                + output["search_v3_evidence_components"].pow(2).mean()
                + output["motion_search_v3_refined_xy"].pow(2).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertIsNotNone(module.context_projection[0].weight.grad)
        self.assertIsNotNone(module.presence_head[0].weight.grad)
        self.assertIsNotNone(module.point_mlp[0].weight.grad)
        for parameter in (
                module.context_projection[0].weight,
                module.film_scale.weight,
                module.film_shift.weight,
                module.motion_geometry_mlp[0].weight,
                module.presence_head[0].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_non_finite_motion_prior_is_hard_invalid_and_outputs_finite(self):
        module = StateAlignedSearchRefiner()
        inputs = self._inputs()
        inputs["motion_proposal_xy"][0, 0] = float("nan")
        output = module(**inputs)
        self.assertEqual(float(output[
            "motion_search_v3_candidate_structural_valid"][0]), 0.0)
        for value in output.values():
            if torch.is_tensor(value):
                self.assertTrue(torch.isfinite(value).all())


class ActionRouterTest(unittest.TestCase):
    def _inputs(self, batch=2):
        def tensor(*shape):
            return torch.randn(*shape)
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
            search_feature=tensor(batch, 384),
            motion_search_xy=tensor(batch, 2),
            motion_search_valid=torch.ones(batch),
            search_presence=torch.full((batch,), 0.1),
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
            gap_ratio=torch.ones(batch),
        )

    def test_cold_auto_and_forced_observation_both_abstain(self):
        router = ActionConsistentInnovationRouter()
        inputs = self._inputs()
        auto, diagnostics = router(**inputs)
        self.assertTrue(torch.equal(auto, inputs["observation_box"]))
        self.assertEqual(diagnostics["router_v3_gain_q10"].shape, (2, 2, 3))
        forced, forced_diag = router(
            **inputs,
            policy_override=torch.full((2,), router.POLICY_OBSERVATION))
        self.assertTrue(torch.equal(forced, inputs["observation_box"]))
        self.assertTrue(torch.all(forced_diag["router_v3_abstained"] == 1))
        with torch.no_grad():
            for parameter in router.parameters():
                parameter.normal_(mean=0.0, std=2.0)
        forced_after, _ = router(
            **inputs,
            policy_override=torch.full((2,), router.POLICY_OBSERVATION))
        self.assertTrue(torch.equal(forced_after, inputs["observation_box"]))

    def test_forced_action_executes_exact_candidate_and_step(self):
        router = ActionConsistentInnovationRouter()
        _, diagnostics = router(
            **self._inputs(),
            policy_override=torch.tensor([
                router.POLICY_MOTION, router.POLICY_REFINED]),
            forced_step_ratio=torch.tensor([0.25, 1.0]))
        self.assertTrue(torch.equal(
            diagnostics["router_v3_selected_candidate"],
            torch.tensor([1, 2])))
        self.assertTrue(torch.equal(
            diagnostics["router_v3_selected_step_index"],
            torch.tensor([0, 2])))

    def test_auto_selects_and_executes_the_same_highest_q10_action(self):
        router = ActionConsistentInnovationRouter()
        with torch.no_grad():
            router.median_gain_head.bias.zero_()
            router.median_gain_head.bias[5] = 1.0
        _, diagnostics = router(**self._inputs())
        self.assertTrue(torch.equal(
            diagnostics["router_v3_selected_candidate"],
            torch.tensor([2, 2])))
        self.assertTrue(torch.equal(
            diagnostics["router_v3_selected_step_index"],
            torch.tensor([2, 2])))
        flat = diagnostics["router_v3_gain_q10"].reshape(2, 6)
        self.assertTrue(torch.equal(flat.argmax(dim=1), torch.tensor([5, 5])))

    def test_action_mask_does_not_change_intrinsic_router_features(self):
        torch.manual_seed(13)
        router = ActionConsistentInnovationRouter()
        with torch.no_grad():
            router.median_gain_head.bias.zero_()
            router.median_gain_head.bias[5] = 1.0
        inputs = self._inputs()
        unrestricted_box, unrestricted = router(**inputs)
        all_box, all_allowed = router(
            **inputs, action_allowed_mask=torch.ones(2, 2))
        self.assertTrue(torch.equal(unrestricted_box, all_box))
        for key in (
                "router_v3_features", "router_v3_gain_q10",
                "router_v3_gain_q50", "router_v3_selected_candidate",
                "router_v3_selected_step_index", "router_v3_correction_xy"):
            self.assertTrue(torch.equal(unrestricted[key], all_allowed[key]))

        motion_only_box, motion_only = router(
            **inputs,
            action_allowed_mask=torch.tensor([[1.0, 0.0]] * 2))
        self.assertTrue(torch.equal(
            motion_only["router_v3_features"],
            unrestricted["router_v3_features"]))
        self.assertTrue(torch.equal(
            motion_only["router_v3_candidate_valid"],
            torch.ones(2, 2)))
        self.assertTrue(torch.equal(
            motion_only["router_v3_action_allowed"],
            torch.tensor([[1.0, 0.0]] * 2)))
        self.assertFalse(torch.equal(motion_only_box, unrestricted_box))
        self.assertTrue(torch.all(
            motion_only["router_v3_selected_candidate"] != 2))

    def test_each_action_is_supervised_and_projection_gets_gradient(self):
        router = ActionConsistentInnovationRouter(
            observation_dim=4, motion_dim=3, search_dim=6)
        features = torch.randn(4, router.export_feature_dim)
        gain = torch.linspace(-0.6, 0.6, 24).reshape(4, 2, 3)
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
        for _ in range(2):
            prediction = router.predict_export_features(features)
            losses = action_consistent_router_loss(
                prediction, gain, torch.ones(4, 2))
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
        self.assertEqual(losses["action_valid"].shape, (4, 2, 3))
        self.assertIsNotNone(router.search_projection[1].weight.grad)
        self.assertGreater(
            float(router.search_projection[1].weight.grad.abs().sum()), 0.0)

    def test_calibration_uses_exact_action(self):
        rows = 100
        q10 = np.zeros((rows, 2, 3), dtype=np.float64)
        q10[:, 0, 1] = np.linspace(0.01, 1.0, rows)
        gains = np.full_like(q10, -0.2)
        gains[:, 0, 1] = 0.1
        result = calibrate_action_threshold(
            q10, gains, np.ones((rows, 2), dtype=bool),
            min_coverage=0.05, max_coverage=0.25,
            min_selected_count=5)
        self.assertGreaterEqual(result["helpful_precision"], 0.75)
        self.assertLessEqual(result["harm_rate"], 0.10)


class StrictInitializationTest(unittest.TestCase):
    def test_missing_b1_tensor_fails(self):
        complete = {
            f"physical_motion_encoder.tensor_{index}": torch.tensor(index)
            for index in range(14)
        }
        _, keys = validate_b1_state(complete)
        self.assertEqual(len(keys), 14)
        complete.pop(keys[0])
        with self.assertRaises(RuntimeError):
            validate_b1_state(complete)

    def test_migration_requires_every_declared_submodule(self):
        refiner = StateAlignedSearchRefiner().state_dict()
        source = {}
        migratable_roots = (
            "point_mlp.", "source_embedding.", "query_projection.",
            "key_projection.", "key_norm.", "query_value_projection.",
            "query_norm.", "local_targetness_head.", "vote_head.",
        )
        for key, value in refiner.items():
            if key.startswith(migratable_roots):
                source["search_evidence_v21." + key] = value
        migrated, _ = collect_migrated_search_state(source)
        self.assertEqual(len(migrated), 33)
        broken = {
            key: value for key, value in source.items()
            if not key.startswith("search_evidence_v21.vote_head.")
        }
        with self.assertRaises(RuntimeError):
            collect_migrated_search_state(broken)

    def test_rollout_loader_requires_exact_keys_and_frozen_hashes(self):
        class DummyV3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for name in (
                        "seg_pointnet", "mini_pointnet", "motion_mlp",
                        "motion_state_mlp", "feature_pointnet", "Transformer",
                        "physical_motion_encoder",
                        "state_aligned_search_refiner",
                        "action_consistent_router_v3"):
                    setattr(self, name, torch.nn.Linear(1, 1, bias=False))

        prefixes = (
            "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
            "motion_state_mlp.", "feature_pointnet.", "Transformer.",
            "physical_motion_encoder.",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.ckpt"
            model = DummyV3()
            state = model.state_dict()
            hashes = {
                prefix: tensor_prefix_hash(state, prefix)[0]
                for prefix in prefixes
            }
            torch.save({
                "state_dict": state,
                "b2_v3_frozen_reference_hashes": hashes,
            }, path)
            report = load_matching_v3_model_state(DummyV3(), path)
            self.assertEqual(report["matched_tensors"], len(state))

            broken = dict(state)
            broken.pop("motion_state_mlp.weight")
            torch.save({
                "state_dict": broken,
                "b2_v3_frozen_reference_hashes": hashes,
            }, path)
            with self.assertRaises(RuntimeError):
                load_matching_v3_model_state(DummyV3(), path)

    def test_final_router_package_is_required_and_integrity_checked(self):
        state = {
            prefix + "weight": torch.ones(1)
            for prefix in PROTECTED_PREFIXES
        }
        state.update({
            "action_consistent_router_v3.weight": torch.ones(1),
            "action_consistent_router_v3.calibrated_gain_threshold":
                torch.tensor(0.125),
        })
        protected_hash, protected_keys = packaged_tensor_hash(
            state, PROTECTED_PREFIXES)
        payload = {
            "state_dict": state,
            "b2_v3_router_package": {
                "schema": "ct_seqtrack.selective_checkpoint.v3",
                "router_tensor_count": 2,
                "protected_tensor_count": len(protected_keys),
                "protected_prefix_hash": protected_hash,
                "calibration": {
                    "status": "passed",
                    "partition": "calibration",
                    "threshold": 0.125,
                },
            },
        }
        validate_b2_v3_router_package(payload)
        with self.assertRaises(RuntimeError):
            validate_b2_v3_router_package({"state_dict": state})
        tampered = dict(payload)
        tampered["state_dict"] = dict(state)
        tampered["state_dict"][
            "action_consistent_router_v3.calibrated_gain_threshold"
        ] = torch.tensor(0.25)
        with self.assertRaises(RuntimeError):
            validate_b2_v3_router_package(tampered)


class TwoRoundRolloutTest(unittest.TestCase):
    @staticmethod
    def _row(frame_id):
        return {
            "tracklet_id": np.int64(0),
            "tracklet_key": "track/0",
            "partition": "train",
            "frame_id": np.int64(frame_id),
            "router_features": np.zeros((803,), dtype=np.float32),
            "candidate_valid": np.ones((2,), dtype=np.float32),
            "candidate_residual_xy": np.zeros((2, 2), dtype=np.float32),
            "signed_gain": np.zeros((2, 3), dtype=np.float32),
            "candidate_cost": np.ones((2, 3), dtype=np.float32),
            "observation_cost": np.float32(1.0),
            "rollout_length": np.int64(3),
        }

    @staticmethod
    def _manifest(round_id, state_policy):
        manifest = {
            "candidate_checkpoint": "candidate.ckpt",
            "candidate_checkpoint_sha256": "same-checkpoint-hash",
            "config_path": "config.yaml",
            "config_sha256": "same-config-hash",
            "split": "mini_train",
            "seed": 42,
            "horizon": 3,
            "gamma": 0.8,
            "round": round_id,
            "state_policy": state_policy,
            "policy_after_intervention": "explicit_observation",
            "tracklets_evaluated": 1,
            "partition_tracklets": {
                "train": 1, "dev": 0, "calibration": 0},
        }
        if round_id == 1:
            manifest["state_policy_calibration"] = {
                "status": "passed", "partition": "dev"}
        return manifest

    def test_rounds_merge_without_crossing_tracklet_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            round0 = root / "round0"
            round1 = root / "round1"
            merged = root / "merged"
            write_v3_rollout_artifact(
                round0, [self._row(1)], self._manifest(0, "observation"))
            write_v3_rollout_artifact(
                round1, [self._row(2)], self._manifest(1, "router"))
            argv = [
                "merge_b2_v3_rollouts.py",
                "--observation-rollouts", str(round0),
                "--on-policy-rollouts", str(round1),
                "--output", str(merged),
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                    io.StringIO()):
                merge_rollouts_main()
            arrays, manifest, _ = load_v3_rollout_artifact(merged)
            self.assertEqual(arrays["router_features"].shape[0], 2)
            self.assertEqual(manifest["round"], "merged_0_1")
            self.assertEqual(
                manifest["policy_after_intervention"],
                "explicit_observation")

    def test_router_training_stage_is_strict(self):
        round0 = self._manifest(0, "observation")
        validate_training_stage(round0, "dev", 42)
        with self.assertRaises(ValueError):
            validate_training_stage(round0, "calibration", 42)

        round1 = self._manifest(1, "router")
        merged = {
            **round0,
            "round": "merged_0_1",
            "state_policy": "observation_plus_router",
            "source_rounds": [
                {"manifest": round0}, {"manifest": round1}],
        }
        validate_training_stage(merged, "calibration", 42)
        with self.assertRaises(ValueError):
            validate_training_stage(merged, "dev", 42)


if __name__ == "__main__":
    unittest.main()
