"""真实 v29 B0 网络与三臂隔离；仅外部数据/CUDA导入依赖使用既有测试壳。"""
import contextlib
import copy
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from tests.test_ct_v27_full_model import full_model_runtime, _training_batch
from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v27_b0_alignment_real import _digest_tensors
from utils.config import load_yaml_config
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state
from utils.v27_input import build_v27_eval_input


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('b0', 'full_cfc', 'full_gru')


def construct(runtime, arm='b0'):
    config = runtime[0][3](load_yaml_config(ROOT / 'cfgs/ct_seqtrack' / f'29_{arm}_nuscenes_full.yaml'))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    return runtime[2](config)


@contextlib.contextmanager
def observation_routing(model):
    fields = ('use_ct_joint_full', 'use_b1motion_v3', 'ct_enable_b1', 'ct_enable_b2', 'ct_enable_b3')
    saved = {name: getattr(model, name) for name in fields}
    try:
        for name in fields:
            setattr(model, name, False)
        yield
    finally:
        for name, value in saved.items():
            setattr(model, name, value)


def mechanism_batch(runtime, model):
    sampler, _, sequence, state, payload, _, _ = _case(runtime[0], 'full')
    payload['is_initial_query'] = False
    payload['motion_prediction'] = model.predict_motion_prepass(
        sequence, 8, state.results_bbs, recursive_state=state)
    config = copy.deepcopy(model.config)
    config.candidate_trajectory_mode = 'shared_se2'
    row = sampler.motion_processing_mf(payload, config)
    row['ct_recursive_state_age'] = np.float32(state.rollout_age(8))
    row['ct_recursive_state_age_valid'] = np.float32(1.)
    return default_collate([row])


def b0_buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()
            if name.startswith(('seg_pointnet.', 'mini_pointnet.', 'motion_mlp.',
                                'motion_state_mlp.', 'feature_pointnet.', 'Transformer.',
                                'time_encoder.'))}


def test_v29_actual_three_arm_complete_batch_b0_forward_loss_grad_adam_identity(full_model_runtime):
    base = construct(full_model_runtime)
    shared, _, _ = _training_batch(full_model_runtime, base, batch_size=16)
    shared['candidate_id'] = torch.tensor([0, 0, 1, 3] * 4)
    assert shared['b0_coarse_target'].shape == (16, 4)
    expected = None
    for arm in ARMS:
        model = construct(full_model_runtime, arm).train()
        optimizer = model.configure_optimizers()['optimizer']
        optimizer.zero_grad(set_to_none=True)
        with observation_routing(model):
            output = model(shared)
            losses = model.compute_loss(shared, output)
            total = model._ct_candidate_weighted_observation_loss(shared, output, losses)
        output_digest = _digest_tensors((name, output[name]) for name in (
            'estimation_boxes', 'observation_aux_estimation_boxes', 'seg_logits',
            'pred_bc', 'motion_pred', 'updated_ref_boxs'))
        loss_digest = _digest_tensors(sorted((name, value) for name, value in losses.items()
                                             if torch.is_tensor(value)))
        if model.use_b1motion_v3:
            rng = capture_global_rng_state()
            flags = {name: module.training for name, module in model.named_modules()}
            buffers = b0_buffers(model)
            try:
                plugin_batch = mechanism_batch(full_model_runtime, model)
                plugin_output = model._forward_safe_mechanism(plugin_batch)
                plugin_loss = model.compute_loss(plugin_batch, plugin_output)['loss_plugin_transaction']
                assert torch.isfinite(plugin_loss)
                plugin_to_b0 = torch.autograd.grad(plugin_loss,
                    [value for name, value in model.named_parameters()
                     if not model._ct_any_plugin_parameter(name)],
                    retain_graph=True, allow_unused=True)
                assert all(value is None for value in plugin_to_b0)
                total = total + plugin_loss
            finally:
                restore_global_rng_state(rng)
            assert flags == {name: module.training for name, module in model.named_modules()}
            assert all(torch.equal(value, b0_buffers(model)[name]) for name, value in buffers.items())
        assert torch.isfinite(total)
        total.backward()
        b0 = [(name, value) for name, value in model.named_parameters()
              if not model._ct_any_plugin_parameter(name)]
        assert all(torch.isfinite(value.grad).all() for _, value in b0 if value.grad is not None)
        gradients = _digest_tensors((name, value.grad) for name, value in b0 if value.grad is not None)
        optimizer.step()
        observed = (output_digest, loss_digest, gradients, _digest_tensors(b0),
                    _digest_tensors(sorted(b0_buffers(model).items())),
                    _digest_tensors((f'{name}.{key}', value) for name, parameter in b0
                        for key, value in sorted(optimizer.state[parameter].items()) if torch.is_tensor(value)),
                    hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
        if expected is None:
            expected = observed
        else:
            assert observed == expected, arm


@pytest.mark.parametrize('count', [0, 1, 2])
def test_v29_real_eval_sparse_identity_current_gt_invariance_and_whole_empty_hold(full_model_runtime, count):
    model = construct(full_model_runtime).eval()
    _, _, sequence, state, _, _, _ = _case(full_model_runtime[0], 'b0')
    sequence = copy.deepcopy(sequence)
    points = np.repeat(state.results_bbs[-1].center[:, None], count, axis=1)
    sequence[8]['pc'] = full_model_runtime[0][1].PointCloud(points)
    batch, _ = build_v27_eval_input(model, sequence, 8, state.results_bbs, recursive_state=state)
    assert batch['b0_unique_mask'][0, -1].sum() == count
    assert bool(batch['b0_point_valid_mask'][0, -1].any()) == (count > 0)
    assert batch['b0_point_valid_mask'][0, :-1].any()
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(output['aux_estimation_boxes']).all()
    assert output['aux_estimation_boxes'].abs().sum() > 0
    changed_sequence = copy.deepcopy(sequence)
    changed_sequence[8]['3d_bbox'].center += np.asarray([17., -13., 0.])
    changed_sequence[8]['3d_bbox'].wlh *= 1.4
    changed, _ = build_v27_eval_input(model, changed_sequence, 8, state.results_bbs, recursive_state=state)
    with torch.no_grad():
        altered = model(changed)
    assert torch.equal(output['aux_estimation_boxes'], altered['aux_estimation_boxes'])
    assert not torch.equal(batch['box_label'], changed['box_label'])
    batch['b0_point_valid_mask'].fill_(False)
    with torch.no_grad():
        empty = model(batch)
    assert torch.equal(empty['aux_estimation_boxes'], torch.zeros(1, 4))


def test_v29_host_passes_physical_layout_and_existence_measurement_masks(full_model_runtime):
    model = construct(full_model_runtime).eval()
    batch, _, _ = _training_batch(full_model_runtime, model, batch_size=2)
    batch['valid_mask'][:, 1] = 0
    batch['b0_point_valid_mask'][:, 2] = False
    seen = {}
    feature_hook = model.feature_pointnet.register_forward_pre_hook(
        lambda _module, args: seen.update(feature_input=args[0].detach().clone()))
    transformer_hook = model.Transformer.register_forward_pre_hook(
        lambda _module, args, kwargs: seen.update(history=args[2].clone(), kwargs=kwargs),
        with_kwargs=True)
    try:
        with torch.no_grad():
            output = model(batch)
    finally:
        feature_hook.remove()
        transformer_hook.remove()
    source = model.encode_point_time(batch['points']).transpose(1, 2)
    source = torch.cat((source, batch['candidate_bc'].transpose(1, 2)), dim=1)
    expected = torch.stack([source[b, :, frame * 1024:(frame + 1) * 1024]
                            for b in range(2) for frame in range(4)])
    assert torch.equal(seen['feature_input'], expected)
    assert torch.equal(seen['history'], batch['valid_mask'])
    assert seen['kwargs']['enable_v29'] is True
    assert torch.equal(seen['kwargs']['frame_measurement_valid'],
                       batch['b0_point_valid_mask'].any(dim=-1))
    assert torch.count_nonzero(output['updated_ref_boxs'][:, 1]) == 0
    assert torch.isfinite(output['aux_estimation_boxes']).all()


def test_v29_sampler_keeps_gt_history_supervision_when_predicted_history_changes(full_model_runtime):
    from datasets import points_utils
    from nuscenes.utils.geometry_utils import points_in_box
    from pyquaternion import Quaternion

    model = construct(full_model_runtime).eval()
    _, _, sequence, state, _, _, _ = _case(full_model_runtime[0], 'b0')
    baseline, anchor = build_v27_eval_input(model, sequence, 8, state.results_bbs, recursive_state=state)
    changed_state = copy.deepcopy(state)
    changed_state.predictions[6].center += np.asarray([1.5, -.8, 0.])
    changed_state.predictions[6].rotate(Quaternion(axis=[0, 0, 1], radians=.2))
    changed, _ = build_v27_eval_input(model, sequence, 8, changed_state.results_bbs,
                                     recursive_state=changed_state)
    assert not torch.equal(baseline['ref_boxs'], changed['ref_boxs'])
    for name in ('box_label_prev', 'motion_label', 'motion_state_label', 'b0_coarse_target', 'box_label'):
        assert torch.equal(baseline[name], changed[name]), name

    sampled = changed['points'][0].reshape(4, 1024, -1).numpy()
    for index, frame_id in enumerate((7, 6, 5)):
        gt_local = points_utils.transform_box(sequence[frame_id]['3d_bbox'], anchor)
        expected_seg = points_in_box(gt_local, sampled[index, :, :3].T, model.config.bb_scale)
        np.testing.assert_array_equal(changed['seg_label'][0, index * 1024:(index + 1) * 1024], expected_seg)
        expected_bc = points_utils.get_point_to_box_distance(sampled[index, :, :3], gt_local)
        np.testing.assert_allclose(changed['prev_bc'][0, index], expected_bc, rtol=1e-6, atol=1e-6)
    expected_physical = anchor.rotation_matrix.T @ (
        sequence[8]['3d_bbox'].center - sequence[7]['3d_bbox'].center)
    np.testing.assert_allclose(changed['b0_coarse_target'][0, :3], expected_physical, rtol=1e-6, atol=1e-6)
