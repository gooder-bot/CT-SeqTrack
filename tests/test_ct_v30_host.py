"""v30 正式配置下的真实网络、共享观测事务与递归接口。"""
import copy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from tests.test_ct_v27_input_flow import sampler_runtime, _case  # noqa: F401
from tests.test_ct_v27_full_model import full_model_runtime, _training_batch  # noqa: F401
from tests.test_ct_v29_b0_host import observation_routing, b0_buffers
from tests.test_ct_v27_b0_alignment_real import _digest_tensors
from utils.config import load_yaml_config
from utils.v27_input import build_v27_eval_input

ROOT = Path(__file__).resolve().parents[1]


def construct30(runtime, arm='full_cfc'):
    config = runtime[0][3](load_yaml_config(ROOT / 'cfgs/ct_seqtrack' / f'30_{arm}_mini.yaml'))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    return runtime[2](config)


def batch30(runtime, model, batch_size=2):
    if not model.use_b1motion_v3:
        return _training_batch(runtime, model, batch_size)
    variant = 'full' if model.ct_enable_b3 else 'full_minus_b3'
    sampler, _, sequence, state, payload, _, _ = _case(runtime[0], variant)
    payload.update(is_initial_query=False, history_reference_reliable=False)
    payload['motion_prediction'] = model.predict_motion_prepass(
        sequence, 8, state.results_bbs, recursive_state=state)
    config = copy.deepcopy(model.config)
    config.candidate_trajectory_mode = 'shared_se2'
    row = sampler.motion_processing_mf(payload, config)
    row['ct_recursive_state_age'] = np.float32(state.rollout_age(8))
    row['ct_recursive_state_age_valid'] = np.float32(1.)
    return default_collate([row for _ in range(batch_size)]), sequence, state


@pytest.mark.parametrize('arm', ['b0', 'full_cfc', 'full_gru'])
def test_v30_complete_real_forward_backward_adam(full_model_runtime, arm):
    model = construct30(full_model_runtime, arm).train()
    batch, _, _ = batch30(full_model_runtime, model, batch_size=2)
    output = model(batch)
    losses = model.compute_loss(batch, output)
    assert torch.isfinite(losses['loss_total'])
    losses['loss_total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = model.configure_optimizers()['optimizer']
    optimizer.step()
    if model.ct_enable_b3:
        assert output['ct_b3_action_boxes'].shape == (2, 6, 4)
        assert output['ct_mode_features'].shape == (2, 3, 64)
        assert batch['motion_acquisition_features'].shape == (2, 21)
        assert losses['loss_ct_raw_search'].item() == 0.
        assert 'loss_ct_mode_quality' in losses
        for prefix in ('physical_motion_encoder.', 'ct_joint_search_refiner.', 'ct_joint_router.'):
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                for name, p in model.named_parameters() if name.startswith(prefix)), prefix


def test_three_arms_share_b0_input_gradient_bn_adam_rng(full_model_runtime):
    baseline = construct30(full_model_runtime, 'b0')
    batch, _, _ = _training_batch(full_model_runtime, baseline, batch_size=2)
    batch['b0_point_valid_mask'][0, -1] = False
    snapshots = []
    for arm in ('b0', 'full_cfc', 'full_gru'):
        model = construct30(full_model_runtime, arm).train()
        optimizer = model.configure_optimizers()['optimizer']
        with observation_routing(model):
            output = model(batch)
            loss = model.compute_loss(batch, output)['loss_b0_transaction']
        loss.backward()
        params = [(name, p) for name, p in model.named_parameters() if not model._ct_any_plugin_parameter(name)]
        gradient = _digest_tensors((name, p.grad) for name, p in params if p.grad is not None)
        optimizer.step()
        snapshots.append((float(loss.detach()), gradient, _digest_tensors(params),
            _digest_tensors(sorted(b0_buffers(model).items())), torch.get_rng_state().numpy().tobytes()))
    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_current_gt_cannot_change_crop_context_modes_or_actions(full_model_runtime):
    model = construct30(full_model_runtime).eval()
    _, _, sequence, state, _, _, _ = _case(full_model_runtime[0], 'full')
    baseline, _ = build_v27_eval_input(model, sequence, 8, state.results_bbs, recursive_state=state)
    altered = copy.deepcopy(sequence)
    altered[8]['3d_bbox'].center += np.array([30., -25., 2.])
    altered[8]['3d_bbox'].wlh *= 2
    changed, _ = build_v27_eval_input(model, altered, 8, state.results_bbs, recursive_state=state)
    for key in ('points', 'b0_point_ids', 'motion_acquisition_features', 'ct_acquisition_support_half_size',
                'ct_extension_points', 'ct_extension_point_ids'):
        assert torch.equal(baseline[key], changed[key]), key
    assert not torch.equal(baseline['box_label'], changed['box_label'])
    with torch.no_grad():
        first, second = model(baseline), model(changed)
    for key in ('ct_mode_centers_xy', 'ct_mode_quality', 'ct_b3_action_boxes', 'ct_b3_action_scores'):
        assert torch.equal(first[key], second[key]), key
    assert torch.equal(first['aux_estimation_boxes'], first['observation_aux_estimation_boxes'])


def test_plugin_transaction_has_no_b0_gradient_or_bn_change(full_model_runtime):
    model = construct30(full_model_runtime).train()
    batch, _, _ = batch30(full_model_runtime, model, batch_size=2)
    before = b0_buffers(model)
    output = model._forward_safe_mechanism(batch)
    loss = model.compute_loss(batch, output)['loss_plugin_transaction']
    gradients = torch.autograd.grad(loss, [p for name, p in model.named_parameters()
        if not model._ct_any_plugin_parameter(name)], allow_unused=True)
    assert all(value is None for value in gradients)
    assert all(torch.equal(value, b0_buffers(model)[name]) for name, value in before.items())
