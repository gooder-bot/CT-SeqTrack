"""实际 PointNet/Transformer/B1/B2/B3 多 tick 累积反传，无 CUDA 替代算子。"""
import pytest
import torch

from tests.test_ct_v27_full_model import full_model_runtime, _construct, _training_batch
from tests.test_ct_v27_input_flow import sampler_runtime
from utils.training_isolation import freeze_batchnorm_running_stats


@pytest.mark.parametrize('arm', ['full_minus_b3', 'full'])
def test_real_multitick_plugins_do_not_mutate_b0_bn_or_saved_backward_tensors(full_model_runtime, arm):
    model = _construct(full_model_runtime, arm).train()
    observation, _, _ = _training_batch(full_model_runtime, model, batch_size=4)
    observation['candidate_id'] = torch.arange(4, dtype=torch.long)
    routing = (model.use_ct_joint_full, model.use_b1motion_v3,
               model.ct_enable_b1, model.ct_enable_b2, model.ct_enable_b3)
    try:
        model.use_ct_joint_full = model.use_b1motion_v3 = False
        model.ct_enable_b1 = model.ct_enable_b2 = model.ct_enable_b3 = False
        observation_output = model(observation)
        observation_losses = model.compute_loss(observation, observation_output)
        observation_loss = model._ct_candidate_weighted_observation_loss(
            observation, observation_output, observation_losses)
    finally:
        (model.use_ct_joint_full, model.use_b1motion_v3,
         model.ct_enable_b1, model.ct_enable_b2, model.ct_enable_b3) = routing
    b0_named = [(name, parameter) for name, parameter in model.named_parameters()
                if not model._ct_any_plugin_parameter(name)]
    expected_gradients = torch.autograd.grad(observation_loss, [parameter for _, parameter in b0_named],
                                             retain_graph=True, allow_unused=True)
    bn_buffers = {name: (value.detach().clone(), value._version)
                  for name, value in model.named_buffers()
                  if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    assert bn_buffers
    losses = []
    motion_diagnostics = []
    row_counts = [1, 2, 1]
    modes = [(module, module.training) for module in model.modules()]
    with freeze_batchnorm_running_stats(model, excluded_prefixes=(
            'physical_motion_encoder', 'ct_joint_search_refiner', 'ct_joint_router')):
        for tick_index, count in enumerate(row_counts):
            batch, _, state = _training_batch(full_model_runtime, model, batch_size=count)
            # The sampler helper bypasses _process_online_raw, which normally
            # attaches these verified causal-age fields before loss reduction.
            batch['ct_recursive_state_age'] = torch.full((count,), float(state.rollout_age(8)))
            batch['ct_recursive_state_age_valid'] = torch.ones(count)
            # Different label populations force independent cumulative class-balance updates.
            if tick_index == 1:
                batch['ct_extension_labels'] = batch['ct_extension_valid_mask'].clone()
            output = model._forward_safe_mechanism(batch)
            tick_losses = model.compute_loss(batch, output)
            motion_diagnostics.append({key: float(value.detach()) for key, value in tick_losses.items()
                                       if key.startswith(('loss_motion_v3', 'loss_b1', 'loss_ct_acquisition'))})
            losses.append(tick_losses['loss_plugin_transaction'] * (count / sum(row_counts)))
    assert all(module.training == prior for module, prior in modes)
    actual_buffers = dict(model.named_buffers())
    for name, (value, version) in bn_buffers.items():
        assert torch.equal(value, actual_buffers[name]), name
        assert version == actual_buffers[name]._version, name
    total = observation_loss + sum(losses)
    assert torch.isfinite(total)
    total.backward()
    for (name, parameter), expected in zip(b0_named, expected_gradients):
        if expected is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0, msg=name)
    for prefix in ('physical_motion_encoder.', 'ct_joint_search_refiner.', 'ct_joint_router.'):
        parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]
        if parameters:
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients), prefix
            assert any(torch.count_nonzero(gradient) for gradient in gradients), (prefix, motion_diagnostics)
    optimizer = model.configure_optimizers()['optimizer']
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters())
