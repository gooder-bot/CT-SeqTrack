"""v30 长历史访问、真实时间速度标签与空测量监督。"""
import copy

import numpy as np
import pytest
import torch

from tests.test_ct_v27_input_flow import sampler_runtime, _case  # noqa: F401
from tests.test_ct_v27_full_model import full_model_runtime  # noqa: F401
from tests.test_ct_v30_host import construct30, batch30
from utils.v29_rollin import prepare_observation_batch, observation_item
from models.ct_v2.observation_reference import seqtrack_reference_loss


def test_empty_measurement_and_missing_history_keep_box_motion_supervision(full_model_runtime):
    model = construct30(full_model_runtime, 'b0').eval()
    data, _, _ = batch30(full_model_runtime, model)
    data['b0_point_valid_mask'].zero_()
    data['valid_mask'].zero_()
    output = model(data)
    for name in ('seg_logits', 'pred_bc', 'updated_ref_boxs'):
        output[name].retain_grad()
    loss = seqtrack_reference_loss(data, output, model.config)
    for name in ('loss_seg', 'loss_bc', 'loss_center_ref', 'loss_angle_ref'):
        assert loss[name].item() == 0., name
    assert loss['loss_center_aux'].item() > 0.
    assert loss['loss_center_motion'].item() > 0.
    loss['loss_total'].backward()
    for name in ('seg_logits', 'pred_bc', 'updated_ref_boxs'):
        assert torch.count_nonzero(output[name].grad) == 0, name


def test_motion_speed_uses_each_history_to_current_real_span(full_model_runtime):
    model = construct30(full_model_runtime, 'b0')
    sampler, _, sequence, state, payload, _, _ = _case(full_model_runtime[0], 'b0')
    # 每0.5秒0.1米：三段历史位移不同，但速度全为0.2m/s。
    for i, frame in enumerate(sequence):
        frame['3d_bbox'].center[:] = [i * .1, 0., 0.]
    for key in ('online_recursive_state', 'online_motion_aux_state', 'motion_prediction', 'candidate_shared_transform'):
        payload.pop(key, None)
    payload['candidate_offsets'] = np.zeros((3, 3), dtype=np.float32)
    row = sampler.motion_processing_mf(payload, model.config)
    np.testing.assert_array_equal(row['motion_state_label'], [0, 0, 0])
    np.testing.assert_allclose(np.abs(row['delta_T_real']), [.5, 1., 1.5])
    # fixed负控只改变B1时间输入，物理速度标签仍然不变。
    changed = copy.deepcopy(model.config)
    changed.dynamics_time_mode = 'fixed'
    changed.dynamics_fixed_delta_t = .1
    for index, frame in enumerate(sequence):
        frame['_ct_effective_timestamp'] = index * .1
        frame['_ct_dynamics_time_mode'] = 'fixed'
    fixed = sampler.motion_processing_mf(payload, changed)
    np.testing.assert_array_equal(fixed['motion_state_label'], row['motion_state_label'])


def test_long_rollin_visits_eight_predicted_states_then_one_gradient_endpoint(full_model_runtime):
    model = construct30(full_model_runtime, 'b0').train()
    _, _, sequence, _, _, _, _ = _case(full_model_runtime[0], 'b0')
    frames = []
    for i in range(10):
        frame = copy.deepcopy(sequence[min(i, 8)])
        frame['frame_id'] = i
        frame['timestamp'] = 1_500_000_000_000_000 + i * 500_000
        frame['3d_bbox'].center[0] = i * .4
        frames.append(frame)
    item = dict(mode='rollin', index=7, candidate=3, epoch=0, start=0, endpoint=9,
        tracklet_id=0, tracklet_key='long/test', frames=frames, first=frames[0], max_steps=8)
    before = torch.get_rng_state().clone()
    observed = []
    handle = model.register_forward_hook(lambda module, args, output:
        observed.append((torch.is_grad_enabled(), module.training)))
    try:
        data = prepare_observation_batch(model, [item, copy.deepcopy(item)])
        assert model._ct_v29_rollin_diagnostics['sample_forwards'] == 16
        assert observed == [(False, False)] * 8
        assert torch.equal(before, torch.get_rng_state())
        loss = model.compute_loss(data, model(data))['loss_b0_transaction']
        loss.backward()
        assert observed[-1] == (True, True)
        assert len(observed) == 9
        assert torch.isfinite(loss)
    finally:
        handle.remove()


def test_teacher_empty_history_keeps_requested_endpoint(full_model_runtime):
    model = construct30(full_model_runtime, 'b0')
    sampler, _, sequence, _, payload, _, _ = _case(full_model_runtime[0], 'b0')
    for frame in sequence:
        frame['pc'].points = np.empty((3, 0), dtype=np.float64)
        if hasattr(frame['pc'], 'point_ids'):
            frame['pc'].point_ids = np.empty(0, dtype=np.int64)
    for key in ('online_recursive_state', 'online_motion_aux_state', 'motion_prediction', 'candidate_shared_transform'):
        payload.pop(key, None)
    payload['candidate_offsets'] = np.zeros((3, 3), dtype=np.float32)
    # 旧teacher会拒绝三个空历史；v30必须返回原端点的标签与零真实点mask。
    row = sampler.motion_processing_mf(payload, model.config)
    assert row['b0_point_valid_mask'].sum() == 0
    assert row['motion_state_label'][0] == 1
    assert row['box_label'].shape == (4,)
