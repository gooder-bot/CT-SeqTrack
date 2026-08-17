import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.ct_v2 import (
    build_box_memory_tokens,
    apply_memory_control,
    extension_target_bearing_mask,
    B2EvidenceAcquirer,
    B3SelectiveUpdater,
)
from utils.ct_search import sample_joint_novel_extensions
from utils.config import load_yaml_config
from utils.recursive_state import OnlineRecursiveBatchSampler
from utils.training_isolation import (
    CheckpointableRNG,
    assert_training_transaction_equal,
    capture_training_transaction_state,
    isolated_constructor_rng,
)
from tools.promote_ct_b2_evidence import evaluate as evaluate_b2_promotion


ROOT = Path(__file__).resolve().parents[1]


class NovelExtensionAcquisitionTest(unittest.TestCase):
    def test_baseline_is_excluded_and_cross_branch_duplicates_are_marked(self):
        baseline = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        endpoint = np.asarray([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float32)
        tube = np.asarray([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ], dtype=np.float32)
        points, valid, source, diagnostics = sample_joint_novel_extensions(
            baseline, endpoint, tube, endpoint_quota=4, tube_quota=4,
            seed=7)
        selected = points[valid > 0]
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(np.unique(selected[:, :3], axis=0)), 3)
        self.assertFalse(any(np.array_equal(row, baseline[0])
                             for row in selected))
        self.assertFalse(any(np.array_equal(row, baseline[1])
                             for row in selected))
        self.assertEqual(set(source[valid > 0].tolist()), {1, 2, 3})
        self.assertEqual(diagnostics['both_count'], 1)


class MemoryAttentionContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.batch_size = 2
        self.history_features = torch.randn(2, 3, 1024, 64)
        self.history_points = torch.randn(2, 3, 1024, 5)
        self.history_boxes = torch.zeros(2, 3, 4)
        self.box_size = torch.full((2, 3), 4.0)

    def build_inputs(self):
        memory, memory_mask = build_box_memory_tokens(
            self.history_features,
            self.history_points,
            self.history_boxes,
            self.box_size,
            torch.ones(2, 3),
        )
        base_features = torch.randn(
            2, 1024, 64, requires_grad=True)
        return {
            'extension_points': torch.randn(2, 256, 5),
            'extension_valid_mask': torch.ones(2, 256),
            'extension_source': torch.ones(2, 256, dtype=torch.long),
            'current_base_features': base_features,
            'current_base_valid_mask': torch.ones(2, 1024),
            'memory_tokens': memory,
            'memory_valid_mask': memory_mask,
            'observation_box': torch.zeros(2, 4),
            'observation_stats': torch.zeros(2, 5),
            'b1_center_xy': torch.zeros(2, 2),
            'b1_sigma_parallel_perp': torch.ones(2, 2),
            'b1_direction_xy': torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            'b1_valid': torch.ones(2),
            'query_delta_t': torch.ones(2),
            'gap_ratio': torch.ones(2),
        }

    def test_memory_is_fixed_36_tokens_and_does_not_use_labels(self):
        tokens, valid = build_box_memory_tokens(
            self.history_features,
            self.history_points,
            self.history_boxes,
            self.box_size,
            torch.ones(2, 3),
        )
        self.assertEqual(tokens.shape, (2, 36, 64))
        self.assertEqual(valid.shape, (2, 36))
        self.assertFalse(tokens.requires_grad)

    def test_memory_metadata_contains_age_pose_role_and_identity(self):
        tokens, valid, metadata = build_box_memory_tokens(
            self.history_features,
            self.history_points,
            self.history_boxes,
            self.box_size,
            torch.ones(2, 3),
            history_timestamps=torch.tensor([
                [-3.0, -2.0, -1.0], [-3.0, -2.0, -1.0]]),
            current_timestamp=torch.zeros(2),
            current_box=torch.zeros(2, 4),
            return_metadata=True,
        )
        self.assertEqual(metadata.shape, (2, 36, 8))
        self.assertTrue(bool(torch.isfinite(metadata).all()))
        real = apply_memory_control(tokens, valid, metadata, 'real')
        misaligned = apply_memory_control(
            tokens, valid, metadata, 'time_misaligned')
        self.assertTrue(torch.equal(real[0], misaligned[0]))
        self.assertTrue(torch.equal(real[1], misaligned[1]))
        self.assertEqual(real[0].shape, misaligned[0].shape)
        self.assertFalse(torch.equal(real[2], misaligned[2]))

    def test_attention_is_extension_query_over_base_plus_memory(self):
        module = B2EvidenceAcquirer().train()
        inputs = self.build_inputs()
        output = module(**inputs)
        self.assertEqual(
            output['ct_cross_attention_weights'].shape,
            (2, 4, 256, 1024 + 36))
        self.assertEqual(
            output['ct_extension_query_features'].shape, (2, 256, 64))
        self.assertFalse(
            output['b0_current_base_features_detached'].requires_grad)
        output['ct_search_targetness_logits'].mean().backward()
        self.assertIsNone(inputs['current_base_features'].grad)

    def test_empty_extension_is_exact_observation_noop(self):
        module = B2EvidenceAcquirer().eval()
        inputs = self.build_inputs()
        inputs['extension_valid_mask'].zero_()
        inputs['observation_box'] = torch.randn(2, 4)
        output = module(**inputs)
        self.assertTrue(torch.equal(
            output['ct_b2_raw_box'], inputs['observation_box']))
        self.assertTrue(bool((output['ct_b2_available'] == 0).all()))
        self.assertTrue(torch.equal(
            output['ct_b2_no_extension_box'], inputs['observation_box']))

    def test_b3_is_independent_of_removed_b2_utility_aliases(self):
        router = B3SelectiveUpdater().eval()
        observation = torch.zeros(2, 4)
        raw = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0, 0.0]])
        kwargs = dict(
            observation_box=observation,
            raw_box=raw,
            availability=torch.ones(2),
            base_evidence=torch.zeros(2, 64),
            extension_evidence=torch.zeros(2, 64),
            base_presence_probability=torch.ones(2),
            extension_presence_probability=torch.ones(2),
            observation_stats=torch.zeros(2, 5),
            b1_sigma_parallel_perp=torch.ones(2, 2),
            query_delta_t=torch.ones(2),
            gap_ratio=torch.ones(2),
        )
        _, first = router(
            **kwargs, h1_utility_logit=torch.tensor([-20.0, 20.0]),
            h1_expected_gain=torch.tensor([-10.0, 10.0]))
        _, second = router(
            **kwargs, h1_utility_logit=torch.tensor([20.0, -20.0]),
            h1_expected_gain=torch.tensor([10.0, -10.0]))
        self.assertTrue(torch.equal(
            first['ct_b3_action_score'], second['ct_b3_action_score']))

    def test_b3_requires_calibration_and_bounds_action(self):
        router = B3SelectiveUpdater(
            require_calibration=True, radius_base=0.2,
            radius_per_second=0.3, radius_max=0.5).eval()
        observation = torch.zeros(1, 4)
        kwargs = dict(
            observation_box=observation,
            raw_box=torch.tensor([[4.0, 0.0, 0.0, 0.0]]),
            availability=torch.ones(1),
            base_evidence=torch.zeros(1, 64),
            extension_evidence=torch.zeros(1, 64),
            base_presence_probability=torch.ones(1),
            extension_presence_probability=torch.ones(1),
            observation_stats=torch.zeros(1, 5),
            b1_sigma_parallel_perp=torch.ones(1, 2),
            query_delta_t=torch.ones(1),
            gap_ratio=torch.ones(1),
        )
        uncalibrated, output = router(**kwargs)
        self.assertTrue(torch.equal(uncalibrated, observation))
        self.assertLessEqual(
            float(torch.linalg.norm(
                output['ct_router_bounded_residual_xy'], dim=1)), 0.5)
        router.install_calibration(0.0, 0.0)
        calibrated, output = router(**kwargs)
        self.assertFalse(torch.equal(calibrated, observation))
        self.assertLessEqual(float(torch.linalg.norm(
            calibrated[:, :2] - observation[:, :2], dim=1)), 0.5)

    def test_b3_nonfinite_input_is_exact_observation(self):
        router = B3SelectiveUpdater().eval()
        observation = torch.randn(1, 4)
        final, output = router(
            observation_box=observation,
            raw_box=torch.tensor([[float('nan'), 0.0, 0.0, 0.0]]),
            availability=torch.ones(1),
            base_evidence=torch.zeros(1, 64),
            extension_evidence=torch.zeros(1, 64),
            base_presence_probability=torch.ones(1),
            extension_presence_probability=torch.ones(1),
            observation_stats=torch.zeros(1, 5),
            b1_sigma_parallel_perp=torch.ones(1, 2),
            query_delta_t=torch.ones(1),
            gap_ratio=torch.ones(1),
        )
        self.assertTrue(torch.equal(final, observation))
        self.assertEqual(float(output['ct_router_applied_gate']), 0.0)
        self.assertTrue(bool(torch.isfinite(
            output['ct_router_bounded_residual_xy']).all()))

    def test_h3_invalid_path_is_bitwise_observation(self):
        router = B3SelectiveUpdater().eval()
        observation = torch.randn(2, 4)
        final, output = router(
            observation_box=observation,
            raw_box=torch.randn(2, 4),
            availability=torch.zeros(2),
            base_evidence=torch.zeros(2, 64),
            extension_evidence=torch.zeros(2, 64),
            base_presence_probability=torch.ones(2),
            extension_presence_probability=torch.ones(2),
            h1_utility_logit=torch.full((2,), 20.0),
            h1_expected_gain=torch.ones(2),
            observation_stats=torch.zeros(2, 5),
            b1_sigma_parallel_perp=torch.ones(2, 2),
            query_delta_t=torch.ones(2),
            gap_ratio=torch.ones(2),
        )
        self.assertTrue(torch.equal(final, observation))
        self.assertTrue(bool((output['ct_b3_final_gate'] == 0).all()))


class AcquisitionLossIsolationTest(unittest.TestCase):
    def test_absence_row_has_no_raw_regression_gradient(self):
        raw_box = torch.zeros(1, 4, requires_grad=True)
        extension_presence = torch.zeros(1, requires_grad=True)
        labels = torch.zeros(1, 2)
        valid = torch.ones(1, 2)
        mask = extension_target_bearing_mask(
            torch.ones(1), labels, valid)
        raw_error = F.smooth_l1_loss(
            raw_box[:, :2], torch.ones(1, 2), reduction='none').mean(1)
        raw_loss = (raw_error * mask).sum() / mask.sum().clamp_min(1.0)
        presence_loss = F.binary_cross_entropy_with_logits(
            extension_presence, torch.zeros_like(extension_presence))
        (raw_loss + presence_loss).backward()
        self.assertTrue(torch.equal(raw_box.grad, torch.zeros_like(raw_box)))
        self.assertGreater(float(extension_presence.grad.abs()), 0.0)


class _SequenceDataset:
    def get_num_tracklets(self):
        return 80

    def get_tracklet_key(self, index):
        return f'track/{index}'

    def get_num_frames_tracklet(self, index):
        return 3


class _SamplerDataset:
    num_candidates = 1
    dataset = _SequenceDataset()


class RecoveryAndIsolationContractTest(unittest.TestCase):
    def test_shadow_disabled_accepts_sixteen_candidate0_slots(self):
        sampler = OnlineRecursiveBatchSampler(
            _SamplerDataset(), slots=16, candidate_views=1,
            partition='train', partition_seed=42,
            shadow_fraction=0.25, shadow_enabled=False)
        first = next(iter(sampler))
        self.assertEqual(len(first), 16)
        self.assertTrue(all(row[5] == 0 and not row[6] for row in first))

    def test_private_rng_restores_global_state_and_checkpoints(self):
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        rng = CheckpointableRNG(99)
        with rng.fork(torch.device('cpu')):
            first = torch.rand(4)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        state = rng.state_dict()
        restored = CheckpointableRNG(1)
        restored.load_state_dict(state)
        with rng.fork(torch.device('cpu')):
            expected = torch.rand(4)
        with restored.fork(torch.device('cpu')):
            actual = torch.rand(4)
        self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(torch.equal(first, expected))

    def test_named_constructor_domains_are_stable_and_non_interfering(self):
        torch.manual_seed(31)
        before = torch.get_rng_state().clone()
        with isolated_constructor_rng(42, 'b1.motion'):
            b1_first = torch.nn.Linear(4, 4)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with isolated_constructor_rng(42, 'b2.extension_memory'):
            b2_first = torch.nn.Linear(4, 4)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with isolated_constructor_rng(42, 'b1.motion'):
            b1_second = torch.nn.Linear(4, 4)
        with isolated_constructor_rng(42, 'b2.extension_memory'):
            b2_second = torch.nn.Linear(4, 4)
        self.assertTrue(torch.equal(
            b1_first.weight, b1_second.weight))
        self.assertTrue(torch.equal(
            b2_first.weight, b2_second.weight))
        self.assertFalse(torch.equal(b1_first.weight, b2_first.weight))

    def test_plugin_large_gradient_does_not_change_b0_transaction(self):
        def run(include_plugin):
            torch.manual_seed(5)
            b0 = torch.nn.Parameter(torch.tensor([1.0]))
            plugin = torch.nn.Parameter(torch.tensor([1.0]))
            optimizer_b0 = torch.optim.Adam([b0], lr=0.01)
            optimizer_plugin = torch.optim.Adam([plugin], lr=0.01)
            loss_b0 = (b0 - 0.25).pow(2).sum()
            loss_plugin = plugin.pow(2).sum() * 1e20
            b0_grad = torch.autograd.grad(
                loss_b0, [b0], retain_graph=include_plugin)[0]
            b0.grad = b0_grad
            b0_norm = torch.nn.utils.clip_grad_norm_([b0], 0.1)
            optimizer_b0.step()
            if include_plugin:
                plugin.grad = torch.autograd.grad(loss_plugin, [plugin])[0]
                torch.nn.utils.clip_grad_norm_([plugin], 1.0)
                optimizer_plugin.step()
            return capture_training_transaction_state(
                [('b0', b0)], optimizer_b0,
                inputs=torch.tensor([1.0]), loss=loss_b0,
                clip_norm=b0_norm,
                clip_coefficient=torch.clamp(0.1 / b0_norm, max=1.0))

        assert_training_transaction_equal(run(False), run(True))

    def test_recovery_matrix_and_v3_config_contracts(self):
        names = {
            '23_b0_candidate0_reseed.yaml': (True, False),
            '23_b0_candidate0_no_reseed.yaml': (False, False),
            '23_b0_candidate0_reseed_rng_shift.yaml': (True, True),
            '23_b0_candidate0_no_reseed_rng_shift.yaml': (False, True),
        }
        for name, expected in names.items():
            config = load_yaml_config(ROOT / 'cfgs' / 'ct_v2' / name)
            self.assertEqual(config['num_candidates'], 1)
            self.assertEqual(config['ct_recursive_candidate_views'], 1)
            self.assertEqual(config['ct_recursive_tracklet_slots'], 16)
            self.assertEqual(
                (config['ct_recursive_reseed_enabled'],
                 config['ct_b0_rng_shift_control']), expected)
            self.assertFalse(config['use_ct_joint_full'])
        v3 = load_yaml_config(
            ROOT / 'cfgs' / 'ct_v2' / '23_ct_evidence_memory.yaml')
        self.assertEqual(v3['ct_joint_contract_version'], 3)
        self.assertTrue(v3['ct_separate_optimizers'])
        self.assertEqual(v3['batch_size'], 16)
        self.assertEqual(v3['ct_recursive_tracklet_slots'], 16)
        self.assertEqual(v3['ct_recursive_candidate_views'], 4)
        self.assertEqual(v3['ct_auxiliary_microbatch_size'], 16)
        self.assertEqual(v3['ct_b0_lr'], 1e-4)
        self.assertEqual(v3['ct_initialization_policy'], 'scratch_only')
        self.assertFalse(v3['require_b1_calibration_artifact'])
        self.assertEqual(v3['ct_search_feature_dim'], 64)
        self.assertEqual(v3['ct_memory_attention_dropout'], 0.0)
        for seed in (43, 44):
            repeated = load_yaml_config(
                ROOT / 'cfgs' / 'ct_v2'
                / f'23_b0_candidate0_no_reseed_seed{seed}.yaml')
            self.assertEqual(repeated['seed'], seed)
            self.assertFalse(repeated['ct_recursive_reseed_enabled'])

    def test_b3_promotion_is_fail_closed(self):
        metrics = {
            'acquisition_row_recall': 0.55,
            'acquisition_eligible_rows': 100,
            'raw_helpful_precision': 0.80,
            'raw_harmful_rate': 0.04,
            'raw_action_count': 100,
            'raw_action_rate': 0.1,
            'raw_center_gain': 0.1,
            'raw_iou_gain': 0.01,
            'raw_oracle_center_headroom': 0.2,
            'raw_oracle_iou_headroom': 0.02,
        }
        criteria, passed = evaluate_b2_promotion(metrics)
        self.assertTrue(passed)
        self.assertTrue(all(criteria.values()))
        metrics['raw_harmful_rate'] = 0.051
        _, passed = evaluate_b2_promotion(metrics)
        self.assertFalse(passed)
        metrics['raw_harmful_rate'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            evaluate_b2_promotion(metrics)
        metrics['raw_harmful_rate'] = -0.01
        with self.assertRaisesRegex(ValueError, r'\[0, 1\]'):
            evaluate_b2_promotion(metrics)
        metrics['raw_harmful_rate'] = 0.04


if __name__ == '__main__':
    unittest.main()
