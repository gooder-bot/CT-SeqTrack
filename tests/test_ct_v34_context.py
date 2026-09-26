"""v34 query 上下文的真实支持、梯度边界与旧版初始化兼容性。"""
from dataclasses import replace
import math

import pytest
import torch
from torch import nn

from models.ct_v31.decoder import SharedHypothesisDecoder
from models.ct_v31.observation import B0Observation
from models.ct_v31.model import JointTracker
from tests.test_ct_v31_joint import make_batch, model_config
from tests.test_ct_v31_observation_decoder import _decoder_inputs


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(34)
    yield
    torch.set_num_threads(previous)


def context_inputs():
    observation, evidence, batch, prior = _decoder_inputs()
    batch['frame_times'] = torch.cat((batch['history_times'], torch.zeros(1, 1)), dim=1)
    observation = replace(observation, coarse_features=torch.randn(1, 256, requires_grad=True),
        history_support=torch.rand(1, 3, 3, requires_grad=True),
        current_support=torch.rand(1, 3, requires_grad=True))
    return observation, evidence, batch, prior


def activate_context(decoder):
    with torch.no_grad():
        nn.init.normal_(decoder.context_fusion.weight, std=.03)
        nn.init.normal_(decoder.pose_head.weight, std=.03)
        nn.init.normal_(decoder.quality_head.weight, std=.03)


def test_support_distinguishes_empty_singleton_and_background_without_gt():
    clean = torch.zeros(3, 4, 4, 5, requires_grad=True)
    valid = torch.zeros(3, 4, 4, dtype=torch.bool)
    valid[1:, :, 0] = True
    foreground = torch.zeros(3, 4, 4, requires_grad=True)
    with torch.no_grad():
        foreground[1, :, 0] = 1.
    history, current = B0Observation.observation_support(clean, valid, foreground,
        torch.zeros(3, 3, 4), torch.ones(3, 3))
    assert not history.requires_grad and not current.requires_grad
    torch.testing.assert_close(current[0], torch.zeros(3))
    expected_count = math.log(2.) / math.log(5.)
    torch.testing.assert_close(current[1, :2], torch.full((2,), expected_count))
    torch.testing.assert_close(current[2, 0], current[1, 0])
    assert current[2, 1] == 0  # 一真实背景点与零点有不同点量。
    torch.testing.assert_close(history[:, 0], current)


def test_history_mass_uses_input_box_geometry_but_current_uses_whole_crop():
    clean = torch.zeros(1, 4, 2, 5)
    clean[..., :3] = torch.tensor([[[[0., 1.5, 0.], [1.5, 0., 0.]]]]).expand(1, 4, 2, 3)
    boxes = torch.zeros(1, 3, 4)
    boxes[..., 3] = math.pi / 2
    history, current = B0Observation.observation_support(clean,
        torch.ones(1, 4, 2, dtype=torch.bool), torch.ones(1, 4, 2), boxes,
        torch.tensor([[4., 2., 1.]]))
    torch.testing.assert_close(history[..., 1], torch.full((1, 3), math.log(2.) / math.log(3.)))
    torch.testing.assert_close(current[:, 1], torch.ones(1))


def test_observation_support_uses_unique_mask_and_invalid_padding_is_inert():
    batch = make_batch(1, 8)
    batch['point_ids'][..., 1] = batch['point_ids'][..., 0]
    batch['history_valid'][:, 0] = False
    observation = B0Observation(4, query_context=True).eval()
    expected = observation(batch)
    batch['points'][..., 1, :] = float('nan')
    batch['points'][:, 0] = float('nan')
    batch['history_boxes'][:, 0] = float('nan')
    batch['segmentation_labels'].zero_()  # 标签不能改变线上支持。
    actual = observation(batch)
    for name in ('coarse_features', 'history_support', 'current_support'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert actual.coarse_features.requires_grad
    assert not actual.history_support.requires_grad and not actual.current_support.requires_grad
    assert torch.count_nonzero(actual.history_support[:, 0]) == 0
    torch.testing.assert_close(actual.current_support[:, 0], torch.tensor([math.log(8.) / math.log(9.)]))


def test_v33_does_not_construct_context_and_v34_preserves_old_parameters_and_rng():
    torch.manual_seed(112)
    legacy = SharedHypothesisDecoder().eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(112)
    upgraded = SharedHypothesisDecoder(query_context=True).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    assert not hasattr(legacy, 'history_context')
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[name], value, rtol=0, atol=0)
    assert sum(p.numel() for p in upgraded.parameters()) - sum(p.numel() for p in legacy.parameters()) == 27136
    inputs = context_inputs()
    before, after = legacy(*inputs), upgraded(*inputs)
    for name in ('hypothesis_boxes', 'quality_logits', 'history_boxes', 'decoder_features'):
        torch.testing.assert_close(getattr(after, name), getattr(before, name), rtol=0, atol=0)
    assert before.query_context_norm is None
    torch.testing.assert_close(after.query_context_norm, torch.zeros(1))
    assert not after.query_context_norm.requires_grad


def test_fixed_history_slots_keep_existing_empty_boxes_and_use_physical_age():
    observation, evidence, batch, prior = context_inputs()
    batch['history_valid'][:] = torch.tensor([[False, False, True]])
    batch['history_boxes'][:, :2] = float('nan')
    observation.history_support = torch.zeros(1, 3, 3)
    observation.history_support[:, :2] = float('nan')
    decoder = SharedHypothesisDecoder(query_context=True).eval()
    descriptors = []
    hook = decoder.history_context.register_forward_pre_hook(
        lambda module, args: descriptors.append(args[0].detach().clone()))
    decoder(observation, evidence, batch, prior)
    batch['history_times'] *= 9.  # B1 时间干预不能改变这里的真实 age。
    decoder(observation, evidence, batch, prior)
    batch['frame_times'][0, 2] = -.25
    decoder(observation, evidence, batch, prior)
    hook.remove()
    torch.testing.assert_close(descriptors[0], descriptors[1], rtol=0, atol=0)
    descriptor = descriptors[0].reshape(1, 3, 10)
    assert torch.count_nonzero(descriptor[:, :2]) == 0
    assert descriptor[0, 2, -1] == 1
    assert torch.count_nonzero(descriptor[0, 2, 6:9]) == 0
    torch.testing.assert_close(descriptor[0, 2, 5], torch.tensor(math.log(2.)))
    torch.testing.assert_close(descriptors[2].reshape(1, 3, 10)[0, 2, 5], torch.tensor(math.log(1.5)))
    # 改真实时间会改变描述，不能从 effective history_times 回填。
    del batch['frame_times']
    with pytest.raises(ValueError, match='physical frame_times'):
        decoder(observation, evidence, batch, prior)


def test_context_only_changes_current_queries_and_keeps_support_and_mode_geometry_detached():
    observation, evidence, batch, prior = context_inputs()
    evidence.centers_xyz.requires_grad_()
    batch['history_boxes'].requires_grad_()
    batch['frame_times'].requires_grad_()
    decoder = SharedHypothesisDecoder(query_context=True).eval()
    activate_context(decoder)
    output = decoder(observation, evidence, batch, prior)
    variables = (observation.coarse_features, observation.history_support,
        observation.current_support, batch['history_boxes'], batch['frame_times'], evidence.centers_xyz)
    gradients = torch.autograd.grad(output.decoder_features.square().sum(), variables,
        retain_graph=True, allow_unused=True)
    assert gradients[0] is not None and gradients[0].abs().sum() > 0
    assert all(value is None for value in gradients[1:])
    # 历史重建没有新的 pooled 分支；其原有 source/共享头路径不变。
    history_gradient = torch.autograd.grad(output.history_boxes.sum(), observation.coarse_features,
        retain_graph=True, allow_unused=True)[0]
    assert history_gradient is None or torch.count_nonzero(history_gradient) == 0
    mode_gradient = torch.autograd.grad(output.hypothesis_boxes[:, 1:, :3].sum(),
        evidence.centers_xyz, retain_graph=True)[0]
    torch.testing.assert_close(mode_gradient, torch.ones_like(mode_gradient))
    quality_gradient = torch.autograd.grad(output.quality_logits.sum(), observation.coarse_features)[0]
    assert quality_gradient.abs().sum() > 0
    changed = decoder(replace(observation, coarse_features=observation.coarse_features + 1.),
                      evidence, batch, prior)
    torch.testing.assert_close(changed.history_boxes, output.history_boxes, rtol=0, atol=0)
    assert not torch.allclose(changed.decoder_features, output.decoder_features)


def test_all_empty_context_keeps_existing_fallback_and_full_mode_rules():
    observation, evidence, batch, prior = context_inputs()
    observation = replace(observation, sequence_valid=torch.zeros(1, dtype=torch.bool),
        current_valid=torch.zeros(1, dtype=torch.bool), source_valid=torch.zeros_like(observation.source_valid),
        coarse_features=torch.full((1, 256), float('nan')), current_support=torch.zeros(1, 3),
        history_support=torch.zeros(1, 3, 3))
    decoder = SharedHypothesisDecoder(query_context=True).eval()
    activate_context(decoder)
    output = decoder(observation, evidence, batch, prior)
    torch.testing.assert_close(output.hypothesis_boxes[:, 0], prior.box)
    assert output.hypothesis_valid.all()
    # 零点不删除合法历史 pose；Full 模式仍可读该上下文，q0 的空序列 fallback 不变。
    batch['history_boxes'] = batch['history_boxes'].clone()
    batch['history_boxes'][:, 0, 0] -= 2.
    changed = decoder(observation, evidence, batch, prior)
    torch.testing.assert_close(changed.hypothesis_boxes[:, 0], prior.box)
    assert not torch.allclose(changed.decoder_features[:, 1:], output.decoder_features[:, 1:])
    empty = replace(evidence, point_valid=torch.zeros_like(evidence.point_valid))
    output = decoder(observation, empty, batch, prior)
    torch.testing.assert_close(output.hypothesis_boxes, prior.box[:, None].expand(-1, 4, -1))
    assert output.hypothesis_valid.tolist() == [[True, False, False, False]]


@pytest.mark.parametrize('arm,backend', [('b0', 'cfc'), ('b1', 'cfc'), ('b1_b2', 'cfc'),
                                       ('full', 'cfc'), ('full', 'gru')])
def test_v34_model_path_and_full_contracts_have_finite_forward_backward(arm, backend):
    config = dict(model_config(arm, backend), net_model='ctseqtrackv34', experiment_family='ct_seqtrack_v34')
    torch.manual_seed(22)
    model = JointTracker(config).eval()
    assert model.observation.query_context and model.decoder.query_context
    batch = make_batch(1)
    output = model(batch)
    loss = model.compute_losses(batch, output)['loss_total']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert output.decoder.query_context_norm.shape == (1,)
    assert output.observation.coarse_features.shape == (1, 256)
