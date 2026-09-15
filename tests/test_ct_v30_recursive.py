"""v30 真正 host 的动作提交、隔离、合法 H3 与 epoch 边界恢复。"""
import copy

import numpy as np
import pytest
import torch

from tests.test_ct_v30_host import (construct30, batch30, full_model_runtime,  # noqa: F401
                                  sampler_runtime)  # noqa: F401
from tests.test_ct_v29_perf_diagnostics import _future_raw
from utils.online_contract import validate_online_resume_contract
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state
from utils.v28_numerical_audit import snapshot, compare_values
from utils.v30_policy import mechanism_behavior_policy
from utils.v30_training import compute_b3_mode_utility_loss
from utils.v27_input import build_v27_eval_input


def _context(model, state, sequence, kind, frame=8):
    epoch = next(i for i in range(1000) if mechanism_behavior_policy(
        model.config.seed, i, state.tracklet_key, frame)['kind'] == kind)
    raw = dict(online_slot=0, candidate_id=0, this_frame_id=frame,
               prev_frame_ids=[frame - 1, frame - 2, frame - 3], this_frame=sequence[frame],
               online_epoch=epoch, tracklet_key=state.tracklet_key)
    model._ct_online_batch_context = [dict(raw=raw, state=state)]
    return mechanism_behavior_policy(model.config.seed, epoch, state.tracklet_key, frame)


@pytest.mark.parametrize('kind', ['never', 'always', 'threshold', 'explore'])
def test_real_host_behavior_and_deployment_choose_and_commit_same_action(full_model_runtime, kind):
    model = construct30(full_model_runtime).eval()
    batch, sequence, state = batch30(full_model_runtime, model, batch_size=1)
    policy = _context(model, state, sequence, kind)
    with torch.no_grad():
        model.ct_joint_router.expected_success_gain_head.bias.fill_(.4)
        model.ct_joint_router.expected_precision_gain_head.bias.fill_(.4)
        training_output = model(batch)
    assert training_output['ct_b3_action_valid'].any()
    model._apply_v30_mechanism_policy(training_output)
    model.ct_joint_router.install_policy(policy)
    with torch.no_grad():
        deployed = model(batch)
    for key in ('ct_b3_chosen_action_index', 'ct_router_applied_gate', 'aux_estimation_boxes',
                'ct_b3_diagnostic_action_index', 'ct_router_bounded_residual_xy',
                'ct_b3_help_probability', 'ct_b3_harm_probability', 'ct_b3_expected_center_gain',
                'ct_b3_expected_iou_gain', 'ct_b3_h3_utility', 'ct_router_residual_xy',
                'ct_router_clip_rate', 'ct_b3_bounded_action_features', 'ct_b3_mode_summary'):
        assert torch.equal(training_output[key], deployed[key]), key
    if kind != 'never':
        assert training_output['ct_router_applied_gate'].item() == 1
    if kind == 'always':
        assert training_output['ct_b3_chosen_action_index'].item() == 1
    anchor = state.results_bbs[-1]
    accepted = training_output['aux_estimation_boxes'][0]
    expected = model._local_prediction_to_world(accepted, anchor)
    model._commit_online_recursive_predictions(training_output)
    np.testing.assert_allclose(state.results_bbs[8].center, expected.center, atol=1e-6)
    future = copy.deepcopy(sequence[8])
    future.update(frame_id=9, timestamp=sequence[8]['timestamp'] + 500000)
    sequence.append(future)
    next_input, _ = build_v27_eval_input(model, sequence, 9, state.results_bbs, recursive_state=state)
    np.testing.assert_allclose(next_input['coordinate_anchor'][0, :3], expected.center, atol=1e-6)


def test_actual_b3_loss_has_no_b2_b1_b0_gradient(full_model_runtime):
    model = construct30(full_model_runtime).train()
    batch, _, _ = batch30(full_model_runtime, model, batch_size=2)
    output = model._forward_safe_mechanism(batch)
    assert output['ct_b3_action_valid'].any()
    loss = compute_b3_mode_utility_loss(batch, output, model.config)['loss']
    parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    gradients = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
    trained_router = False
    for (name, _), gradient in zip(parameters, gradients):
        if name.startswith('ct_joint_router.'):
            trained_router |= gradient is not None and bool(gradient.abs().sum() > 0)
        else:
            assert gradient is None, name
    assert trained_router


def test_evidence_first_full_ablation_has_trainable_b2_and_shared_full_action(full_model_runtime):
    model = construct30(full_model_runtime, 'ablate_evidence_first_full').train()
    assert not model.ct_enable_b3
    assert not any(name.startswith('ct_joint_router.') for name, _ in model.named_parameters())
    batch, sequence, state = batch30(full_model_runtime, model, batch_size=1)
    output = model._forward_safe_mechanism(batch)
    assert output['ct_b3_action_valid'][0, 1]
    _context(model, state, sequence, 'never')
    model._apply_v30_mechanism_policy(output)
    assert output['ct_b3_chosen_action_index'].item() == 1
    losses = model.compute_loss(batch, output)
    assert losses['loss_ct_raw_search'].item() == 0
    assert torch.isfinite(losses['loss_ct_mode_quality'])
    losses['loss_plugin_transaction'].backward()
    assert model.ct_joint_search_refiner.mode_builder.quality_head[-1].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        deployed = model(batch)
    assert deployed['ct_b3_chosen_action_index'].item() == 1
    assert torch.equal(deployed['aux_estimation_boxes'], deployed['ct_b3_action_boxes'][:, 1])
    assert torch.equal(deployed['ct_router_soft_box'], deployed['aux_estimation_boxes'])


def test_v30_legal_h3_executes_four_shadow_branches_without_mutating_main_state(full_model_runtime):
    from utils.v27_training import attach_h3_shadow_labels_v27
    model = construct30(full_model_runtime).train()
    raw, state = _future_raw(full_model_runtime[0], model)
    model.config.ct_h3_diagnostic_keep_ratio = 1.
    # 只固定当前干预动作；未来两帧都通过真实 v30 host/sampler observation 通路。
    output = dict(observation_aux_estimation_boxes=torch.zeros(1, 4),
                  ct_router_bounded_residual_xy=torch.tensor([[.375, 0.]]), ct_b2_available=torch.ones(1))
    before = snapshot(dict(state=model.state_dict(), rng=capture_global_rng_state(),
                           flags={name: module.training for name, module in model.named_modules()}))
    batch = {}
    attach_h3_shadow_labels_v27(model, batch, output)
    assert batch['ct_h3_failure_reason'] == ['ok'], batch['ct_h3_failure_reason']
    assert batch['ct_shadow_forward_count'].item() == 4
    assert batch['ct_h3_valid'].item() == 1
    assert max(state.predictions) == 7
    after = snapshot(dict(state=model.state_dict(), rng=capture_global_rng_state(),
                          flags={name: module.training for name, module in model.named_modules()}))
    assert compare_values(before, after) is None


def test_epoch_boundary_save_restore_preserves_mode_action_and_adam_identity(full_model_runtime):
    model = construct30(full_model_runtime).train()
    optimizer = model.configure_optimizers()['optimizer']
    batch, sequence, state = batch30(full_model_runtime, model, batch_size=2)
    output = model._forward_safe_mechanism(batch)
    model.compute_loss(batch, output)['loss_plugin_transaction'].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model._ct_epoch_boundary_complete = True
    checkpoint = dict(epoch=0, state_dict=copy.deepcopy(model.state_dict()),
                      optimizer_states=[copy.deepcopy(optimizer.state_dict())],
                      hyper_parameters={'config': copy.deepcopy(dict(model.config))})
    model.on_save_checkpoint(checkpoint)
    validate_online_resume_contract(checkpoint, model.config)
    restored = construct30(full_model_runtime).train()
    other_optimizer = restored.configure_optimizers()['optimizer']
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    other_optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    assert compare_values(snapshot(optimizer.state_dict()), snapshot(other_optimizer.state_dict())) is None
    rng = checkpoint['ct_global_rng_state']
    outputs = []
    for host in (model, restored):
        restore_global_rng_state(rng)
        with torch.no_grad():
            current = host._forward_safe_mechanism(batch)
        outputs.append(current)
    for key in ('ct_mode_centers_xy', 'ct_mode_quality', 'ct_mode_valid', 'ct_mode_seed_point_ids',
                'ct_b3_action_boxes', 'ct_b3_action_scores', 'ct_b3_best_action_index'):
        assert torch.equal(outputs[0][key], outputs[1][key]), key
    assert checkpoint['ct_online_resume_contract']['fields']['ct_b3_action_contract'] == 'three_modes_half_full_xy_v1'
    assert mechanism_behavior_policy(42, 1, state.tracklet_key, 8) == mechanism_behavior_policy(
        restored.config.seed, checkpoint['epoch'] + 1, state.tracklet_key, 8)


def test_six_action_labels_match_actual_world_execution_with_different_gt_size_and_yaw(full_model_runtime):
    from pyquaternion import Quaternion
    from tests.test_ct_v27_input_flow import _case
    from utils.tracking_metrics_v27 import action_metric_gains
    model = construct30(full_model_runtime).eval()
    _, _, sequence, state, _, _, _ = _case(full_model_runtime[0], 'full')
    target = sequence[8]['3d_bbox']
    target.wlh *= np.asarray([1.3, .7, 1.5])
    target.orientation = Quaternion(axis=[0, 0, 1], radians=.61)
    batch, anchor = build_v27_eval_input(model, sequence, 8, state.results_bbs, recursive_state=state)
    with torch.no_grad():
        output = model(batch)
    labels = compute_b3_mode_utility_loss(batch, output, model.config)
    assert labels['action_valid'].any()
    observation = model._local_prediction_to_world(output['observation_aux_estimation_boxes'][0], anchor)
    np.testing.assert_allclose(observation.wlh, sequence[0]['3d_bbox'].wlh)
    assert not np.array_equal(observation.wlh, target.wlh)
    radius = output['ct_router_radius'][0].item()
    for index in range(6):
        if not labels['action_valid'][0, index]:
            continue
        action = model._local_prediction_to_world(output['ct_b3_action_boxes'][0, index], anchor)
        oracle = action_metric_gains(observation, action, target, up_axis=(0, 0, 1), dim=3)
        for name, key in (('success', 'success_gain'), ('precision', 'precision_gain'),
                          ('distance', 'center_gain'), ('iou', 'iou_gain')):
            expected = oracle[key] / radius if name == 'distance' else oracle[key]
            assert labels['action_h1_' + name + '_gain'][0, index].item() == pytest.approx(expected, abs=2e-6)


def test_v30_auxiliary_motion_uses_own_anchor_and_real_query_interval(full_model_runtime):
    model = construct30(full_model_runtime).eval()
    batch, sequence, state = batch30(full_model_runtime, model, batch_size=1)
    assert batch['motion_main_current_delta_t'].item() == pytest.approx(.5)
    assert batch['motion_aux_current_delta_t'].item() == pytest.approx(1.)
    np.testing.assert_allclose(batch['motion_main_delta_t'][0], [.5, .5, .5])
    np.testing.assert_allclose(batch['motion_aux_delta_t'][0], [1., 1., 1.])
    for branch, frame in (('main', 7), ('aux', 6)):
        displacement_world = sequence[8]['3d_bbox'].center - sequence[frame]['3d_bbox'].center
        anchor = state.results_bbs[frame]
        expected = anchor.rotation_matrix.T @ displacement_world
        np.testing.assert_allclose(batch['motion_' + branch + '_target_xy'][0], expected[:2], atol=1e-6)
    assert batch['motion_acquisition_features'].shape[-1] == 21
    assert batch['motion_aux_acquisition_features'].shape[-1] == 21


def test_wholeempty_measurements_keep_legal_history_query_prediction_at_deployment(full_model_runtime):
    model = construct30(full_model_runtime, 'b0').eval()
    batch, _, _ = batch30(full_model_runtime, model, batch_size=2)
    for key in ('b0_point_valid_mask', 'b0_valid_mask', 'b0_unique_mask', 'b0_raw_point_count'):
        batch[key].zero_()
    batch['b0_point_ids'].fill_(-1)
    assert batch['valid_mask'].any()  # 帧/递归框合法，只是所有历史与当前 LiDAR 都为空。
    decoded = []
    handle = model.Transformer.register_forward_hook(lambda module, args, value:
        decoded.append((value[0] if isinstance(value, tuple) else value).detach().clone()))
    try:
        with torch.no_grad():
            output = model(batch)
        assert torch.isfinite(decoded[0]).all()
        assert decoded[0][:, -1].abs().sum() > 0
        assert torch.equal(output['observation_aux_estimation_boxes'], decoded[0][:, -1])
    finally:
        handle.remove()
