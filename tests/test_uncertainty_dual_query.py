import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

from models.attn.Models import Seq2SeqFormer
from models.ct_v2.motion import (
    OrderedPhysicalMotionEncoder,
    physical_motion_uncertainty_loss,
)
from models.ct_v2.selective_innovation import (
    AsymmetricDualQueryAdapter,
    StateAlignedSearchRefiner,
    require_nonzero_finite_gradient,
    validate_trainable_parameter_prefixes,
)
from utils.ct_search import (
    build_b1_uncertainty_support,
    build_uncertainty_prior_tube,
    resolve_b1_search_support,
)
from tools.calibrate_b1_uncertainty import (
    fit_calibration,
    load_and_validate_manifest,
)
from utils.config import load_yaml_config
from utils.replay_cache import (
    b1_calibration_config_sha256,
    sha256_file,
)


class _Box:
    def __init__(self, center=(10.0, 5.0, 0.0), yaw=np.pi / 2.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.wlh = np.asarray((2.0, 4.0, 1.5), dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=float(yaw))


class TransformerDecoderStateTest(unittest.TestCase):
    def test_opt_in_state_preserves_default_box_output(self):
        torch.manual_seed(17)
        model = Seq2SeqFormer(
            d_word_vec=64, d_model=64, d_inner=128,
            n_layers=1, n_head=2, d_k=32, d_v=32, dropout=0.0)
        model.eval()
        target = torch.randn(2, 32, 4)
        source = torch.randn(2, 512, 128)
        valid = torch.ones(2, 3)
        with torch.no_grad():
            default_boxes = model(target, source, valid)
            boxes, state = model(
                target, source, valid, return_decoder_state=True)
        self.assertTrue(torch.equal(default_boxes, boxes))
        self.assertEqual(tuple(state.shape), (2, 4, 64))
        self.assertTrue(torch.equal(boxes, model.l2(state)))

    def test_l4_is_bit_compatible_with_legacy_reshape(self):
        torch.manual_seed(29)
        model = Seq2SeqFormer(
            d_word_vec=64, d_model=64, d_inner=128,
            n_layers=1, n_head=2, d_k=32, d_v=32, dropout=0.0).eval()
        target = torch.randn(2, 32, 4)
        source = torch.randn(2, 512, 128)
        with torch.no_grad():
            current = model(target, source, torch.ones(2, 4))
            projected_source = model.proj(source)
            projected_target = model.proj2(target)
            local = model.encoder(
                projected_source.reshape(-1, 128, 64))[0]
            global_state = model.encoder_global(
                projected_source, global_feature=True)[0]
            encoded = torch.cat((
                local.reshape(2, 4 * 128, 64), global_state), dim=1)
            decoded = model.decoder(
                projected_target, None, encoded, None)[0]
            legacy = model.l2(model.l1(decoded.view(2, 4, 64 * 8)))
        self.assertTrue(torch.equal(current, legacy))

    def test_variable_frame_and_token_counts(self):
        model = Seq2SeqFormer(
            d_word_vec=64, d_model=64, d_inner=128,
            n_layers=1, n_head=2, d_k=32, d_v=32, dropout=0.0).eval()
        with torch.no_grad():
            boxes, state = model(
                torch.randn(2, 24, 4), torch.randn(2, 288, 128),
                torch.ones(2, 3), return_decoder_state=True)
        self.assertEqual(tuple(boxes.shape), (2, 3, 4))
        self.assertEqual(tuple(state.shape), (2, 3, 64))


class PhysicalMotionUncertaintyTest(unittest.TestCase):
    def test_formal_configs_share_the_b1_calibration_contract(self):
        root = Path(__file__).resolve().parents[1] / "cfgs" / "ct_seqtrack"
        formal_names = (
            "24_b1.yaml",
            "24_full_minus_b3.yaml",
            "24_full.yaml",
        )
        configs = {
            name: load_yaml_config(root / name) for name in formal_names}
        hashes = {
            b1_calibration_config_sha256(configs[name])
            for name in formal_names}
        self.assertEqual(len(hashes), 1)
        self.assertTrue(all(
            config.get("observation_safe_bbox_size") is True
            for config in configs.values()))

    def _history(self, stationary=False):
        boxes = torch.zeros(2, 3, 4)
        if not stationary:
            boxes[:, 1, 0] = -1.0
            boxes[:, 2, 0] = -2.0
        delta_t = torch.tensor([[0.5, 0.5, 0.5]] * 2)
        valid = torch.ones(2, 3)
        return boxes, delta_t, valid

    def test_covariance_is_positive_and_low_speed_is_isotropic(self):
        module = OrderedPhysicalMotionEncoder(
            hidden_dim=16,
            step_dim=8,
            motion_aligned_uncertainty=True,
            min_direction_speed=0.2,
        )
        boxes, delta_t, valid = self._history(stationary=True)
        output = module(boxes, delta_t, valid, torch.full((2,), 0.5))
        covariance = output["covariance_xy"]
        eigenvalues = torch.linalg.eigvalsh(covariance)
        self.assertTrue(torch.all(eigenvalues > 0))
        self.assertTrue(torch.allclose(
            output["log_sigma_parallel_perp"][:, 0],
            output["log_sigma_parallel_perp"][:, 1]))
        self.assertEqual(tuple(output["mu_xy"].shape), (2, 2))

    def test_nll_is_finite_and_calibration_is_applied(self):
        module = OrderedPhysicalMotionEncoder(
            hidden_dim=16,
            step_dim=8,
            motion_aligned_uncertainty=True,
        )
        module.set_uncertainty_calibration([0.2, -0.1])
        boxes, delta_t, valid = self._history()
        output = module(boxes, delta_t, valid, torch.full((2,), 0.5))
        terms = physical_motion_uncertainty_loss(
            output["mu_xy"],
            torch.tensor([[1.1, 0.2], [0.9, -0.1]]),
            output["log_sigma_parallel_perp"],
            output["motion_direction_xy"],
            output["valid"],
        )
        self.assertTrue(torch.isfinite(terms["nll_per_sample"]).all())
        self.assertTrue(torch.all(terms["valid"] == 1))

    def test_learned_velocity_does_not_redefine_uncertainty_axis(self):
        module = OrderedPhysicalMotionEncoder(
            hidden_dim=16, step_dim=8,
            motion_aligned_uncertainty=True)
        with torch.no_grad():
            module.velocity_residual_head.bias.copy_(torch.tensor([0.0, 4.0]))
        boxes, delta_t, valid = self._history()
        output = module(boxes, delta_t, valid, torch.full((2,), 0.5))
        self.assertTrue(torch.allclose(
            output["basis_velocity_xy"][:, 1], torch.zeros(2)))
        self.assertTrue(torch.all(output["velocity_xy"][:, 1] > 0))
        self.assertTrue(torch.allclose(
            output["direction_xy"], torch.tensor([[1.0, 0.0]] * 2)))

    def test_calibrator_recovers_parallel_perpendicular_scales(self):
        rng = np.random.default_rng(41)
        rows = 6000
        predicted_sigma = rng.uniform(0.2, 1.0, size=(rows, 2))
        expected_scale = np.asarray((2.0, 0.5))
        error = rng.normal(size=(rows, 2)) * predicted_sigma * expected_scale
        result = fit_calibration({
            "error_xy": error,
            "kinematic_error_xy": error * 2.0,
            "velocity_xy": np.tile([1.0, 0.0], (rows, 1)),
            "log_sigma_pp": np.log(predicted_sigma),
            "valid": np.ones(rows),
            "gap_ratio": np.ones(rows),
        })
        np.testing.assert_allclose(
            result["scale_parallel_perpendicular"],
            expected_scale, rtol=0.04, atol=0.02)
        self.assertTrue(result["promotion"]["criteria"][
            "learned_mean_beats_kinematic"])

    def test_calibration_manifest_binds_partition_artifact_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "residuals.npz"
            checkpoint = root / "source.ckpt"
            np.savez(artifact, value=np.ones(1))
            checkpoint.write_bytes(b"checkpoint")
            manifest = {
                "schema": "ct_seqtrack.b1_calibration.v2",
                "dataset": "nuscenes",
                "split": "mini_train",
                "partition": "calibration",
                "config_sha256": "a" * 64,
                "b1_config_sha256": "b" * 64,
                "checkpoint_sha256": sha256_file(checkpoint),
                "artifact_sha256": sha256_file(artifact),
            }
            manifest_path = artifact.with_suffix(
                artifact.suffix + ".manifest.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_and_validate_manifest(artifact, checkpoint)
            self.assertEqual(loaded, manifest)
            manifest["partition"] = "dev"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_and_validate_manifest(artifact, checkpoint)


class AsymmetricDualQueryTest(unittest.TestCase):
    def test_zero_init_and_invalid_motion_preserve_observation_query(self):
        torch.manual_seed(23)
        module = AsymmetricDualQueryAdapter(
            observation_dim=64, motion_dim=128)
        observation = torch.randn(3, 64, requires_grad=True)
        motion = torch.randn(3, 128, requires_grad=True)
        query, diagnostics = module(
            observation,
            motion,
            torch.randn(3, 2),
            torch.full((3,), 0.5),
            torch.ones(3),
            torch.tensor([1.0, 0.0, 1.0]),
        )
        self.assertTrue(torch.equal(query, observation.detach()))
        self.assertEqual(float(diagnostics["dual_query_gate"][1]), 0.0)
        query.sum().backward()
        self.assertIsNone(observation.grad)
        self.assertIsNone(motion.grad)

    def test_refiner_loss_updates_adapter_but_not_b0_or_b1_inputs(self):
        torch.manual_seed(29)
        adapter = AsymmetricDualQueryAdapter(
            observation_dim=64, motion_dim=128, hidden_dim=32)
        refiner = StateAlignedSearchRefiner(
            point_dim=9, query_observation_dim=64,
            require_motion_valid=False)
        batch, points = 2, 12
        observation_query = torch.randn(
            batch, 64, requires_grad=True)
        motion_feature = torch.randn(batch, 128, requires_grad=True)
        coarse_feature = torch.randn(batch, 256, requires_grad=True)
        search_query, _ = adapter(
            observation_query, motion_feature, torch.zeros(batch, 2),
            torch.ones(batch), torch.ones(batch), torch.ones(batch))
        output = refiner(
            point_inputs=torch.randn(batch, points, 9),
            point_xy=torch.randn(batch, points, 2),
            delta_to_motion=torch.randn(batch, points, 2),
            point_valid_mask=torch.ones(batch, points),
            point_source=torch.tensor([[0] * 6 + [1] * 6] * batch),
            geometry_valid=torch.ones(batch),
            support_anchor_xy=torch.zeros(batch, 2),
            observation_feature=coarse_feature,
            motion_feature=motion_feature,
            motion_proposal_xy=torch.zeros(batch, 2),
            motion_valid=torch.ones(batch),
            observation_stats=torch.randn(batch, 5),
            query_delta_t=torch.ones(batch),
            gap_ratio=torch.ones(batch),
            sigma_parallel=torch.ones(batch),
            sigma_perpendicular=torch.ones(batch),
            available_count=torch.full((batch,), float(points)),
            extension_count=torch.full((batch,), 6.0),
            overlap_count=torch.full((batch,), 6.0),
            query_feature=search_query,
        )
        loss = (
            output["search_v3_raw_vote_xy"].pow(2).mean()
            + output["search_v3_match_logits"].pow(2).mean())
        loss.backward()
        self.assertTrue(require_nonzero_finite_gradient(
            adapter.named_parameters(), ""))
        self.assertIsNone(observation_query.grad)
        self.assertIsNone(motion_feature.grad)
        self.assertIsNone(coarse_feature.grad)

    def test_search_candidate_can_remain_valid_without_motion(self):
        module = StateAlignedSearchRefiner(
            query_observation_dim=64,
            require_motion_valid=False,
            predict_utility=True,
        )
        batch, points = 2, 8
        output = module(
            point_inputs=torch.randn(batch, points, 9),
            point_xy=torch.randn(batch, points, 2),
            delta_to_motion=torch.randn(batch, points, 2),
            point_valid_mask=torch.ones(batch, points),
            point_source=torch.tensor([[0] * 4 + [1] * 4] * batch),
            geometry_valid=torch.ones(batch),
            support_anchor_xy=torch.randn(batch, 2),
            observation_feature=torch.randn(batch, 256),
            motion_feature=torch.randn(batch, 128),
            motion_proposal_xy=torch.zeros(batch, 2),
            motion_valid=torch.zeros(batch),
            observation_stats=torch.randn(batch, 5),
            query_delta_t=torch.full((batch,), 0.5),
            gap_ratio=torch.ones(batch),
            sigma_parallel=torch.ones(batch),
            sigma_perpendicular=torch.ones(batch),
            available_count=torch.full((batch,), 8.0),
            extension_count=torch.full((batch,), 4.0),
            overlap_count=torch.full((batch,), 4.0),
            query_feature=torch.randn(batch, 64),
        )
        self.assertTrue(torch.all(output[
            "motion_search_v3_candidate_structural_valid"] == 1))
        self.assertEqual(tuple(output["search_v3_utility_logit"].shape), (2,))

    def test_formal_trainable_prefix_and_first_gradient_contract(self):
        module = torch.nn.Module()
        module.state_aligned_search_refiner = torch.nn.Linear(2, 2)
        module.asymmetric_dual_query = AsymmetricDualQueryAdapter(
            observation_dim=2, motion_dim=3, hidden_dim=4)
        validate_trainable_parameter_prefixes(
            module.named_parameters(),
            ("state_aligned_search_refiner.", "asymmetric_dual_query."),
            {"state_aligned_search_refiner.", "asymmetric_dual_query."})
        observation = torch.randn(4, 2)
        search, _ = module.asymmetric_dual_query(
            observation, torch.randn(4, 3), torch.zeros(4, 2),
            torch.ones(4), torch.ones(4), torch.ones(4))
        (search - torch.ones_like(search)).pow(2).mean().backward()
        self.assertTrue(require_nonzero_finite_gradient(
            module.named_parameters(), "asymmetric_dual_query."))


class SupportContractTest(unittest.TestCase):
    def test_shared_resolver_is_identical_for_replay_and_online_prediction(self):
        history = [
            _Box(center=(1.0, 0.0, 0.0), yaw=0.0),
            _Box(center=(0.0, 0.0, 0.0), yaw=0.0),
        ]
        prediction = {
            "mu_xy": [2.0, 0.5],
            "velocity_xy": [3.0, 1.0],
            "direction_xy": [1.0, 0.0],
            "log_sigma_parallel_perp": [0.0, -0.2],
            "valid": True,
            "source_id": 1,
            "current_delta_t": 0.5,
            "gap_ratio": 1.25,
        }
        kwargs = dict(
            prediction=prediction,
            use_b1_prepass=True,
            use_dynamic_sigma=True,
            max_length=24.0,
            max_width=10.0,
        )
        online_box, online_diag = resolve_b1_search_support(
            history, [0.5, 0.5], [True, True], **kwargs)
        replay_box, replay_diag = resolve_b1_search_support(
            history, [0.5, 0.5], [True, True], **kwargs)
        np.testing.assert_allclose(online_box.center, replay_box.center)
        np.testing.assert_allclose(online_box.wlh, replay_box.wlh)
        self.assertAlmostEqual(
            float(online_box.orientation.radians),
            float(replay_box.orientation.radians), places=7)
        self.assertEqual(online_diag["prior_source"], "b1")
        for key in (
                "valid", "prior_source", "source_id", "truncated",
                "requested_length", "requested_width", "length", "width",
                "query_delta_t", "gap_ratio"):
            self.assertEqual(online_diag[key], replay_diag[key])
        np.testing.assert_allclose(
            online_diag["endpoint_center"], replay_diag["endpoint_center"])

    def test_shared_resolver_marks_cv_fallback_and_base_only(self):
        moving_history = [
            _Box(center=(1.0, 0.0, 0.0), yaw=0.0),
            _Box(center=(0.0, 0.0, 0.0), yaw=0.0),
        ]
        box, diagnostics = resolve_b1_search_support(
            moving_history, [0.5, 0.5], [True, True],
            prediction={"valid": False}, use_b1_prepass=True)
        self.assertIsNotNone(box)
        self.assertEqual(diagnostics["prior_source"], "fallback_cv")
        self.assertEqual(diagnostics["source_id"], 2)

        box, diagnostics = resolve_b1_search_support(
            moving_history[:1], [0.5], [True],
            prediction={"valid": False}, use_b1_prepass=True)
        self.assertIsNone(box)
        self.assertEqual(diagnostics["prior_source"], "base_only")
        self.assertEqual(diagnostics["source_id"], 0)

    def test_direction_and_truncation_are_explicit(self):
        box, diagnostics = build_b1_uncertainty_support(
            _Box(center=(0.0, 0.0, 0.0), yaw=0.0),
            {
                "mu_xy": [8.0, 0.0],
                "velocity_xy": [0.0, 5.0],
                "direction_xy": [1.0, 0.0],
                "log_sigma_parallel_perp": [0.0, 0.0],
                "valid": True,
                "source_id": 1,
            },
            use_dynamic_sigma=False,
            fixed_margins=(3.0, 2.0),
            max_length=6.0,
            max_width=3.0,
        )
        self.assertIsNotNone(box)
        self.assertAlmostEqual(float(box.orientation.radians), 0.0, places=6)
        self.assertTrue(diagnostics["truncated"])
        self.assertGreater(
            diagnostics["requested_length"], diagnostics["length"])

    def test_raw_vote_is_not_legacy_clipped_and_geometry_has_mahalanobis(self):
        module = StateAlignedSearchRefiner(
            point_dim=10, query_observation_dim=64,
            require_motion_valid=False)
        batch, points = 1, 8
        with torch.no_grad():
            module.vote_head[-1].weight.zero_()
            module.vote_head[-1].bias.copy_(torch.tensor([4.0, 0.0]))
        output = module(
            point_inputs=torch.zeros(batch, points, 10),
            point_xy=torch.zeros(batch, points, 2),
            delta_to_motion=torch.zeros(batch, points, 2),
            point_valid_mask=torch.ones(batch, points),
            point_source=torch.tensor([[0] * 4 + [1] * 4]),
            geometry_valid=torch.ones(batch),
            support_anchor_xy=torch.zeros(batch, 2),
            observation_feature=torch.zeros(batch, 256),
            motion_feature=torch.zeros(batch, 128),
            motion_proposal_xy=torch.zeros(batch, 2),
            motion_valid=torch.zeros(batch),
            observation_stats=torch.zeros(batch, 5),
            query_delta_t=torch.full((batch,), 0.5),
            gap_ratio=torch.ones(batch),
            sigma_parallel=torch.ones(batch),
            sigma_perpendicular=torch.ones(batch),
            available_count=torch.full((batch,), 8.0),
            extension_count=torch.full((batch,), 4.0),
            overlap_count=torch.full((batch,), 4.0),
            query_feature=torch.zeros(batch, 64),
        )
        raw_norm = torch.linalg.norm(output["search_v3_raw_vote_xy"], dim=1)
        clipped_norm = torch.linalg.norm(
            output["motion_search_v3_refined_xy"], dim=1)
        self.assertGreater(float(raw_norm[0]), float(clipped_norm[0]))
        self.assertGreater(float(raw_norm[0]), 3.0)


class PriorTubeTest(unittest.TestCase):
    def test_tube_uses_local_motion_and_expected_extent(self):
        box = _Box()
        tube, diagnostics = build_uncertainty_prior_tube(
            box,
            mu_xy=[4.0, 0.0],
            sigma_parallel_perpendicular=[1.0, 0.5],
            velocity_xy=[2.0, 0.0],
            coverage_scale=2.0,
        )
        self.assertTrue(diagnostics["valid"])
        # Local +x is world +y under the latest pi/2 yaw.
        np.testing.assert_allclose(tube.center, [10.0, 7.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(tube.wlh[1]), 12.0, places=6)
        self.assertAlmostEqual(float(tube.wlh[0]), 4.0, places=6)
        np.testing.assert_allclose(
            diagnostics["endpoint_center"], [10.0, 9.0, 0.0], atol=1e-6)

    def test_invalid_prior_requests_fallback(self):
        tube, diagnostics = build_uncertainty_prior_tube(
            _Box(), [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], valid=False)
        self.assertIsNone(tube)
        self.assertFalse(diagnostics["valid"])


if __name__ == "__main__":
    unittest.main()
