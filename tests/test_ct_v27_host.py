"""Execute real host methods without importing the unavailable CUDA backbone."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.ct_v2.action_v27 import B3UtilityUpdater
from models.ct_v2.evidence_memory import (B2EvidenceAcquirer, build_box_memory_tokens,
    apply_memory_control, extension_target_bearing_mask)
from models.ct_v2.motion import OrderedPhysicalMotionEncoder
from models.ct_v2.pipeline import B0Observation
from models.ct_v2.pipeline_contracts import (MotionPriorOutput, EvidenceOutput, DecisionOutput,
    reexpress_motion_prior, validate_motion_prior_support_alignment)
from utils.b1_acquisition import b1_input_digest
from utils.training_isolation import (CheckpointableRNG, candidate_stratified_mean,
    update_cumulative_binary_class_balance)
from utils.v27_input import build_v27_eval_input
from tests.test_ct_v27_input_flow import sampler_runtime, _case


def _method(name, **extra):
    path = Path(__file__).resolve().parents[1] / 'models/seqtrack3d.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    definition = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                      for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == name)
    definition.decorator_list = []
    namespace = dict(globals(), **extra)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def _host_case(runtime, *, training=False, b3=True, empty=False, wide_support=False,
               nonfinite_learned=False):
    _, config, sequence, state, _, data_host, prediction = _case(runtime)
    if wide_support:
        prediction = dict(prediction, acquisition_margin_parallel_perp=np.asarray((6., 3.)))
    if nonfinite_learned:
        prediction = dict(prediction, valid=False, mu_xy=np.full(2, np.nan),
            direction_xy=np.full(2, np.nan), log_sigma_parallel_perp=np.full(2, np.nan))
    batch, _ = build_v27_eval_input(data_host, sequence, 8, state.results_bbs,
                                    recursive_state=state, motion_prediction=prediction)
    if empty:
        for key in ('b0_valid_mask', 'b0_unique_mask', 'ct_base_evidence_valid_mask',
                    'ct_base_unique_mask', 'ct_extension_valid_mask', 'ct_search_support_valid'):
            batch[key].zero_()
        for key in ('b0_point_ids', 'ct_base_point_ids', 'ct_extension_point_ids'):
            batch[key].fill_(-1)
        batch['ct_extension_source'].zero_()
    module = OrderedPhysicalMotionEncoder(enable_v27=True, adaptive_acquisition_margin=True,
                                          shared_kinematic_anchor=True)
    motion = module(batch['motion_main_ref_boxs'], batch['motion_main_delta_t'],
                    batch['motion_main_valid_mask'], batch['motion_main_current_delta_t'],
                    acquisition_features=batch['motion_acquisition_features'])
    # A later learned recomputation is intentionally distinct from acquired support.
    motion['prior_xy'] = motion['prior_xy'] + 3.
    if nonfinite_learned:
        for key in ('prior_xy', 'motion_direction_xy', 'log_sigma_parallel_perp'):
            motion[key] = torch.full_like(motion[key], float('nan'))
        motion['valid'] = torch.zeros_like(motion['valid'])
    host = SimpleNamespace(config=config, ct_enable_v27=True, ct_enable_v26_recovery=True,
        ct_enable_b3=b3, ct_expansion_point_count=768, training=training,
        b0_observation_contract=B0Observation(), ct_plugin_rng=CheckpointableRNG(42),
        ct_joint_search_refiner=B2EvidenceAcquirer(v27_enabled=True,
            relation_aware_sampling=True, robust_consensus_voting=True),
        ct_joint_router=B3UtilityUpdater(require_calibration=True),
        encode_point_time=lambda value: value,
        proposal_inference_mode='full' if b3 else 'bounded_always')
    host.ct_joint_search_refiner.train(training)
    host.ct_joint_router.train(training)
    observation = torch.zeros(1, 4, requires_grad=True)
    features = torch.randn(1, 4, 1024, 64, requires_grad=True)
    output = {'b0_point_aligned_features': features}
    rng = torch.get_rng_state().clone()
    final = _method('_forward_ct_contract_v3')(host, batch, output, observation,
        torch.zeros(1, 5), motion, torch.ones(1), 1, 4, 1024, coarse_box=observation)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.isfinite(final).all() and final.shape == (1, 4)
    torch.testing.assert_close(output['ct_b1_candidate_center_xy'], batch['search_v3_support_anchor_xy'])
    return host, batch, output, final, observation, features, module


@pytest.mark.parametrize('b3', [True, False])
@pytest.mark.parametrize('empty', [True, False])
def test_real_host_acquisition_record_empty_and_deployment(sampler_runtime, b3, empty):
    host, batch, output, final, observation, _, _ = _host_case(
        sampler_runtime, b3=b3, empty=empty)
    if empty or b3:  # B3 without a calibration artifact is a no-op.
        torch.testing.assert_close(final, observation)
    else:
        radius = .5 + .5 * batch['search_v3_query_delta_t']
        assert torch.linalg.norm(final[:, :2] - observation[:, :2], dim=1) <= radius + 1e-6


def test_real_host_b2_and_b3_gradient_ownership(sampler_runtime):
    host, _, output, final, observation, features, b1 = _host_case(sampler_runtime, training=True)
    torch.testing.assert_close(final, observation)
    output['ct_b3_help_logit'].sum().backward(retain_graph=True)
    assert host.ct_joint_router.helpful_head.weight.grad is not None
    assert all(p.grad is None for p in host.ct_joint_search_refiner.parameters())
    assert all(p.grad is None for p in b1.parameters())
    assert observation.grad is None and features.grad is None
    output['ct_relation_logits_prepool'].square().mean().backward()
    assert any(p.grad is not None for p in host.ct_joint_search_refiner.parameters())
    assert all(p.grad is None for p in b1.parameters())
    assert observation.grad is None and features.grad is None


def test_real_host_recovers_from_nonfinite_learned_prior_using_actual_cv_record(sampler_runtime):
    _, batch, output, final, _, _, _ = _host_case(sampler_runtime, nonfinite_learned=True)
    assert batch['search_v3_prior_source_id'].item() == 2
    assert torch.isfinite(output['ct_b2_raw_box']).all()
    assert torch.isfinite(final).all()


@pytest.mark.parametrize('empty', [False, True])
def test_real_host_loss_selected_presence_counters_and_gradient_ownership(sampler_runtime, empty):
    from utils.v27_training import compute_b3_utility_loss
    host, batch, output, _, observation, features, b1 = _host_case(
        sampler_runtime, training=True, empty=empty, wide_support=True)
    output['observation_aux_estimation_boxes'] = observation
    # Current GT dimensions are a loss-only field, separate from fixed input size.
    batch['target_bbox_size'] = batch['bbox_size'] * torch.tensor([[1.2, 1.1, 1.3]])
    for population in ('targetness', 'relation'):
        for label in ('positive', 'negative'):
            setattr(host, f'ct_{population}_running_{label}_points', torch.tensor(0., dtype=torch.float64))
            setattr(host, f'ct_{population}_running_{label}_weight', torch.tensor(1., dtype=torch.float64))
    host._ct_online_targetness_class_weights = lambda *args, **kwargs: _method(
        '_ct_online_targetness_class_weights')(host, *args, **kwargs)
    for name, default in (('ct_targetness_weight', 1.), ('ct_relation_weight', .2),
                          ('ct_vote_weight', 1.), ('ct_raw_search_weight', 1.),
                          ('ct_presence_weight', .1), ('ct_router_weight', 1.)):
        setattr(host, name, float(getattr(host.config, name, default)))
    batch['ct_extension_labels'].zero_()
    if not empty:
        selected = output['ct_extension_selected_indices'][0][
            output['ct_extension_selected_valid_mask'][0] > 0]
        remaining = batch['ct_extension_valid_mask'][0].bool().clone()
        remaining[selected] = False
        assert remaining.any(), 'wide support must expose unselected valid prepool points'
        batch['ct_extension_labels'][0, torch.nonzero(remaining)[0]] = 1.
    losses = _method('_compute_ct_contract_v3_loss')(host, batch, output, batch['box_label'][:, :2])
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert losses['ct_b2_extension_presence_target_rate'].item() == 0
    assert host.ct_targetness_running_positive_points.item() == 0
    assert host.ct_relation_running_positive_points.item() == int(not empty)
    if not empty:
        assert host.ct_relation_running_negative_points > host.ct_targetness_running_negative_points
    expected = compute_b3_utility_loss(batch, output, host.config)
    torch.testing.assert_close(losses['loss_ct_b3_total'], expected['loss'] * host.ct_router_weight)
    losses['loss_ct_b3_total'].backward(retain_graph=True)
    assert any(p.grad is not None for p in host.ct_joint_router.parameters())
    assert all(p.grad is None for p in host.ct_joint_search_refiner.parameters())
    losses['loss_ct_b2_total'].backward()
    assert any(p.grad is not None for p in host.ct_joint_search_refiner.parameters())
    for module in (host.ct_joint_search_refiner, host.ct_joint_router):
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    assert all(p.grad is None for p in b1.parameters())
    assert observation.grad is None and features.grad is None
@pytest.mark.parametrize('mode', ['true', 'fixed', 'shuffled'])
def test_real_prepass_builder_matches_sampler_with_nondefault_time_scale(sampler_runtime, mode):
    sampler, config, sequence, state, payload, _, prediction = _case(sampler_runtime)
    from datasets.misc_utils import build_time_fields, build_effective_time_fields, normalize_dynamics_time_mode
    config.time_scale = .25
    config.dynamics_time_mode = mode
    config.dynamics_fixed_delta_t = .3
    if mode in ('shuffled', 'fixed'):
        for index, frame in enumerate(sequence):
            frame['_ct_dynamics_time_mode'] = mode
            frame['_ct_effective_timestamp'] = (100. + .3 * index if mode == 'fixed'
                                               else 100. + .1 * index + .025 * index ** 2)
    ids = payload['prev_frame_ids']
    contract = payload['online_recursive_state']
    host = SimpleNamespace(config=config, hist_num=3, ct_enable_v27=True)
    arrays = _method('_build_motion_prepass_inputs_contract', build_time_fields=build_time_fields,
        build_effective_time_fields=build_effective_time_fields,
        normalize_dynamics_time_mode=normalize_dynamics_time_mode)(
            host, state.history_boxes(ids, payload['valid_mask']), ids, payload['valid_mask'],
            contract['history_timestamps'], sequence[8]['timestamp'],
            [sequence[i].get('_ct_effective_timestamp') for i in ids],
            sequence[8].get('_ct_effective_timestamp'), mode, 8,
            history_quality=contract['history_quality'], recursive_age=contract['recursive_age'],
            first_frame_wlh=state.target_size)
    payload['motion_prediction'] = dict(prediction, current_delta_t=arrays['current_delta_t'])
    payload['_ct_inference'] = True
    result = sampler.motion_processing_mf(payload, config)
    assert b1_input_digest(arrays) == result['motion_input_digest']
    np.testing.assert_array_equal(arrays['acquisition_features'], result['motion_acquisition_features'])


@pytest.mark.parametrize('reference', [False, True])
def test_b0_and_reference_eval_require_no_motion_prepass(sampler_runtime, reference):
    _, _, sequence, state, _, host, _ = _case(sampler_runtime, 'b0')
    del host.predict_motion_prepass
    if reference:
        del host.ct_enable_b1
        host.config.model_name = 'SEQTRACK3D'
    batch, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs, recursive_state=state)
    assert batch['points'].shape == (1, 4096, 5)
    assert 'motion_acquisition_features' not in batch
