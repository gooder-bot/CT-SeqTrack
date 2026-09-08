"""使用冻结 SeqTrack 算式和真实网络，验证 v28 的整批观测及梯度所有权。"""
import copy
import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from tests.test_ct_v27_full_model import full_model_runtime, _training_batch
from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v27_b0_alignment_real import _digest_tensors
from utils.config import load_yaml_config
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state
from utils.v27_input import build_v27_eval_input


def construct(runtime, arm):
    path = ('cfgs/28_seqtrack_reference.yaml' if arm == 'reference'
            else f'cfgs/ct_seqtrack/28_{arm}.yaml')
    config = runtime[0][3](load_yaml_config(Path(__file__).resolve().parents[1] / path))
    torch.manual_seed(42)
    return runtime[2](config)


@pytest.mark.parametrize('moving_count', [0, 1, 3, 4])
def test_reference_complete_batch_losses_bc_gradient_and_adam_update(full_model_runtime, moving_count):
    from tests.fixtures.seqtrack_v28_reference import reference_forward, reference_compute_loss
    current = construct(full_model_runtime, 'b0').train()
    reference = copy.deepcopy(current)
    data, _, _ = _training_batch(full_model_runtime, current, batch_size=4)
    data['candidate_id'] = torch.tensor([0, 0, 1, 3])  # unequal candidate population
    data['motion_state_label'][:, 0] = 0
    data['motion_state_label'][:moving_count, 0] = 1
    # 固定原 B0 时间标记，独立oracle的时间函数也来自原SeqTrack。
    data['delta_T'][:] = torch.tensor([-.1, -.2, -.3])
    current.config.main_time_current = reference.config.main_time_current = .1
    initial_rng = torch.get_rng_state().clone()
    output = current(data)
    losses = current.compute_loss(data, output)
    torch.set_rng_state(initial_rng)
    expected_output = reference_forward(reference, data)
    expected = reference_compute_loss(reference, data, expected_output)
    for key, value in expected_output.items():
        assert torch.equal(output[key], value), key
    for key, value in expected.items():
        assert torch.equal(losses[key], value), key
    # BC 不通过seg argmax反传；直接检验原BC梯度系数，防止别名 += 再次加入。
    bc_gradient = torch.autograd.grad(losses['loss_b0_transaction'], output['pred_bc'], retain_graph=True)[0]
    expected_bc = torch.autograd.grad(expected['loss_total'], expected_output['pred_bc'], retain_graph=True)[0]
    assert torch.equal(bc_gradient, expected_bc)
    configured = current.configure_optimizers()['optimizer']
    original_adam = torch.optim.Adam(reference.parameters(), lr=1e-4,
        betas=(.5, .999), eps=1e-6, weight_decay=0., foreach=False, fused=False)
    losses['loss_total'].backward()
    expected['loss_total'].backward()
    for (name, a), (_, b) in zip(current.named_parameters(), reference.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), name
    configured.step()
    original_adam.step()
    for (name, a), (_, b) in zip(current.named_parameters(), reference.named_parameters()):
        assert torch.equal(a, b), name


def test_v28_actual_five_arm_two_update_b0_adam_bn_rng_identity(full_model_runtime):
    base = construct(full_model_runtime, 'b0')
    shared, _, _ = _training_batch(full_model_runtime, base, batch_size=4)
    shared['candidate_id'] = torch.tensor([0, 0, 2, 3])
    common = None
    for arm in ('b0', 'b1_gru', 'b1_cfc', 'full_minus_b3', 'full', 'reference'):
        random.seed(42)
        np.random.seed(42)
        model = construct(full_model_runtime, arm).train()
        optimizer = model.configure_optimizers()['optimizer']
        history = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            routing = (model.use_ct_joint_full, model.use_b1motion_v3,
                       model.ct_enable_b1, model.ct_enable_b2, model.ct_enable_b3)
            try:
                model.use_ct_joint_full = model.use_b1motion_v3 = False
                model.ct_enable_b1 = model.ct_enable_b2 = model.ct_enable_b3 = False
                output = model(shared)
                losses = model.compute_loss(shared, output)
                total = model._ct_candidate_weighted_observation_loss(shared, output, losses)
            finally:
                (model.use_ct_joint_full, model.use_b1motion_v3,
                 model.ct_enable_b1, model.ct_enable_b2, model.ct_enable_b3) = routing
            if model.use_b1motion_v3:
                rng = capture_global_rng_state()
                try:
                    batch, _, _ = _training_batch(full_model_runtime, model, batch_size=1)
                    mechanism = model._forward_safe_mechanism(batch)
                    plugin_loss = model.compute_loss(batch, mechanism)['loss_plugin_transaction']
                    plugin_grads = torch.autograd.grad(plugin_loss,
                        [p for n, p in model.named_parameters() if not model._ct_any_plugin_parameter(n)],
                        retain_graph=True, allow_unused=True)
                    assert all(gradient is None for gradient in plugin_grads)
                    total = total + plugin_loss
                finally:
                    restore_global_rng_state(rng)
            total.backward()
            b0 = [(name, parameter) for name, parameter in model.named_parameters()
                  if not model._ct_any_plugin_parameter(name)]
            gradients = _digest_tensors((name, parameter.grad) for name, parameter in b0
                                        if parameter.grad is not None)
            optimizer.step()
            history.append((gradients, _digest_tensors(b0),
                _digest_tensors((name, value) for name, value in model.named_buffers()
                    if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))),
                _digest_tensors((f'{name}.{key}', value) for name, parameter in b0
                    for key, value in sorted(optimizer.state[parameter].items()) if torch.is_tensor(value)),
                hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
        if common is None:
            common = history
        else:
            assert history == common, arm


def test_v28_current_empty_keeps_history_prediction_and_whole_empty_holds(full_model_runtime):
    model = construct(full_model_runtime, 'b0').eval()
    _, _, sequence, state, _, _, _ = _case(full_model_runtime[0], 'b0')
    sequence = copy.deepcopy(sequence)
    sequence[8]['pc'] = full_model_runtime[0][1].PointCloud(np.empty((3, 0)))
    batch, _ = build_v27_eval_input(model, sequence, 8, state.results_bbs, recursive_state=state)
    assert not batch['b0_point_valid_mask'][:, -1].any()
    assert batch['b0_point_valid_mask'][:, :-1].any()
    with torch.no_grad():
        output = model(batch)
    assert output['aux_estimation_boxes'].abs().sum() > 0
    batch['b0_point_valid_mask'].fill_(False)
    with torch.no_grad():
        held = model(batch)
    assert torch.equal(held['aux_estimation_boxes'], torch.zeros(1, 4))


def test_v28_seg_feature_alignment_export_and_b2_memory_source(full_model_runtime):
    model = construct(full_model_runtime, 'full_minus_b3').eval()
    batch, _, _ = _training_batch(full_model_runtime, model, batch_size=2)
    exported = {}
    handle = model.seg_pointnet.register_forward_hook(
        lambda module, inputs, output: exported.update(feature=output[1]) if isinstance(output, tuple) else None)
    with torch.no_grad():
        output = model(batch)
    handle.remove()
    expected = exported['feature'].transpose(1, 2).reshape(2, 4, 1024, 64)
    assert torch.equal(output['b0_point_aligned_features'], expected)
    assert torch.equal(output['ct_base_evidence_points'], batch['points'].reshape(2, 4, 1024, -1)[:, -1])
