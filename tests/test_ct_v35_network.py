"""v35历史条件、真实点局部读取及Full梯度所有权的CPU验证。"""
from dataclasses import replace

import pytest
import torch
from torch import nn

from models.ct_v31.contracts import EvidenceHypotheses
from models.ct_v31.decoder import SharedHypothesisDecoder
from models.ct_v31.geometry import rotate_xyz, transform_boxes
from models.ct_v31.history_context import history_descriptor
from models.ct_v31.local_observation import LocalObservationReader
from models.ct_v31.model import JointTracker
from models.ct_v31.observation import B0Observation
from models.ct_v31.prior import empty_prior
from tests.test_ct_v31_joint import make_batch, model_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(35)
    yield
    torch.set_num_threads(previous)


def inputs(n=24):
    batch = make_batch(1, n)
    batch['history_is_initial'] = torch.tensor([[True, False, False]])
    observation = B0Observation(8, query_context=True, coarse_condition=True).eval()(batch)
    evidence = EvidenceHypotheses.empty(observation.coarse_box)
    evidence.point_valid[:, :24] = True
    evidence.point_ids[:, :24] = torch.arange(1000, 1024)
    evidence.point_xyz[:, :24] = batch['points'][:, -1, :24, :3]
    evidence.point_features[:, :24] = torch.randn(1, 24, 64)
    evidence.selected_identity_logits[:, :24] = torch.linspace(-1., 1., 24)
    evidence.mode_valid[:] = True
    for mode in range(3):
        evidence.members[:, mode, mode * 8:(mode + 1) * 8] = True
    seeds = torch.zeros(1, 4, 4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    return observation, evidence, batch, seeds, valid


def active_reader():
    reader = LocalObservationReader().eval()
    nn.init.normal_(reader.output.weight, std=.03)
    nn.init.constant_(reader.output.bias, .2)
    return reader


def test_shared_history_descriptor_preserves_v34_and_marks_only_existing_initial_slots():
    _, _, batch, _, _ = inputs()
    history = batch['history_boxes'].clone().requires_grad_()
    support = torch.rand(1, 3, 3, requires_grad=True)
    old = history_descriptor(batch, history, batch['history_valid'], batch['box_size'], support, .5)
    new = history_descriptor(batch, history, batch['history_valid'], batch['box_size'], support, .5,
                             include_initial=True)
    torch.testing.assert_close(new.reshape(1, 3, 11)[..., :10], old.reshape(1, 3, 10), rtol=0, atol=0)
    torch.testing.assert_close(new.reshape(1, 3, 11)[..., 10], torch.tensor([[1., 0., 0.]]))
    assert not new.requires_grad
    batch['history_valid'][:, 0] = False
    history = history.detach()
    history[:, 0] = float('nan')
    masked = history_descriptor(batch, history, batch['history_valid'], batch['box_size'], support, .5,
                                include_initial=True)
    assert torch.count_nonzero(masked.reshape(1, 3, 11)[:, 0]) == 0
    with pytest.raises(ValueError, match='history_is_initial'):
        history_descriptor({k: v for k, v in batch.items() if k != 'history_is_initial'},
            history, batch['history_valid'], batch['box_size'], support, .5, include_initial=True)


def test_new_branches_preserve_common_parameters_rng_and_zero_initial_function():
    observation, evidence, batch, _, _ = inputs()
    torch.manual_seed(113)
    old = B0Observation(8, query_context=True).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(113)
    new = B0Observation(8, query_context=True, coarse_condition=True).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    before, after = old(batch), new(batch)
    for name in ('coarse_box', 'coarse_features', 'source_tokens', 'point_features'):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
    torch.manual_seed(114)
    old_decoder = SharedHypothesisDecoder(query_context=True).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(114)
    new_decoder = SharedHypothesisDecoder(query_context=True, initial_context=True, local_observation=True).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in old_decoder.state_dict().items():
        if not name.startswith(('history_context.', 'coarse_context.', 'context_fusion.')):
            torch.testing.assert_close(new_decoder.state_dict()[name], value, rtol=0, atol=0)
    before = old_decoder(observation, evidence, batch, empty_prior(batch))
    after = new_decoder(observation, evidence, batch, empty_prior(batch))
    for name in ('hypothesis_boxes', 'quality_logits', 'history_boxes', 'decoder_features'):
        torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
    assert after.local_delta_norm.count_nonzero() == 0
    assert before.local_neighbor_count is None


def test_coarse_condition_reads_shared_36_features_but_returns_original_pooled_semantics():
    _, _, batch, _, _ = inputs()
    model = B0Observation(8, query_context=True, coarse_condition=True).eval()
    nn.init.normal_(model.coarse_history_condition[-1].weight, std=.03)
    seen = []
    hook = model.coarse_history_condition.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    output = model(batch)
    hook.remove()
    expected = history_descriptor(batch, batch['history_boxes'], batch['history_valid'], batch['box_size'],
                                  output.history_support, .5, include_initial=True)
    torch.testing.assert_close(seen[0], torch.cat((expected, output.current_support), dim=-1), rtol=0, atol=0)
    changed = dict(batch, history_is_initial=torch.zeros_like(batch['history_is_initial']))
    other = model(changed)
    torch.testing.assert_close(other.coarse_features, output.coarse_features, rtol=0, atol=0)
    assert not torch.equal(other.coarse_box, output.coarse_box)
    output.coarse_box.square().sum().backward()
    assert model.coarse_history_condition[0].weight.grad.abs().sum() > 0


def test_v35_new_modules_do_not_shift_full_prior_evidence_or_common_initialization():
    torch.manual_seed(116)
    old = JointTracker(dict(model_config('full'), net_model='ctseqtrackv34',
                            experiment_family='ct_seqtrack_v34'))
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(116)
    new = JointTracker(dict(model_config('full'), net_model='ctseqtrackv35',
                            experiment_family='ct_seqtrack_v35'))
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        if not name.startswith(('decoder.history_context.', 'decoder.coarse_context.', 'decoder.context_fusion.')):
            torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)


def test_local_radius_unique_count_truncation_and_support_use_full_lwh():
    observation, evidence, batch, seeds, valid = inputs(24)
    batch['points'][:, -1, :, :3] = 0.
    observation = replace(observation, foreground_probability=torch.full_like(observation.foreground_probability, .25))
    evidence.point_valid.zero_()
    # 中心到各角点的归一化距离sqrt(.75)<1；若误用半尺寸将全部排除。
    reader = active_reader()
    increment, diag = reader(observation, evidence, batch, seeds, valid, evidence.members)
    torch.testing.assert_close(diag['local_neighbor_count'], torch.full((1, 4, 8), 24))
    torch.testing.assert_close(diag['local_selected_count'], torch.full((1, 4, 8), 16))
    torch.testing.assert_close(diag['local_mean_support'], torch.full((1, 4, 8), .25))
    assert increment.abs().sum() > 0
    assert all(not value.requires_grad for value in diag.values())


def test_local_permutation_and_equal_distance_ties_use_raw_ids():
    observation, evidence, batch, seeds, valid = inputs()
    # 全部等距且超过16点，排序不能依赖槽序。
    batch['points'][:, -1, :, :3] = 0.
    evidence.point_xyz.zero_()
    reader = active_reader()
    before, before_diag = reader(observation, evidence, batch, seeds, valid, evidence.members)
    order = torch.randperm(24)
    changed = dict(batch)
    for key in ('points', 'point_valid', 'point_ids'):
        changed[key] = batch[key][:, :, order]
    changed_observation = replace(observation, point_features=observation.point_features[:, :, order],
        foreground_probability=observation.foreground_probability[:, :, order])
    extension_order = torch.randperm(256)
    changed_evidence = replace(evidence,
        point_xyz=evidence.point_xyz[:, extension_order], point_features=evidence.point_features[:, extension_order],
        point_valid=evidence.point_valid[:, extension_order], point_ids=evidence.point_ids[:, extension_order],
        selected_identity_logits=evidence.selected_identity_logits[:, extension_order],
        members=evidence.members[:, :, extension_order])
    after, after_diag = reader(changed_observation, changed_evidence, changed, seeds, valid, changed_evidence.members)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for key in before_diag:
        torch.testing.assert_close(after_diag[key], before_diag[key], rtol=0, atol=0)


def test_duplicate_raw_ids_and_invalid_nan_do_not_become_local_measurements():
    observation, evidence, batch, seeds, valid = inputs()
    evidence.point_valid.zero_()
    reader = active_reader()
    before, before_diag = reader(observation, evidence, batch, seeds, valid, evidence.members)
    evidence.point_valid[:, 0] = True
    evidence.point_ids[:, 0] = batch['point_ids'][0, -1, 0]
    evidence.point_xyz[:] = float('nan')
    evidence.point_features[:] = float('nan')
    evidence.selected_identity_logits[:] = float('nan')
    after, after_diag = reader(observation, evidence, batch, seeds, valid, evidence.members)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for key in before_diag:
        torch.testing.assert_close(after_diag[key], before_diag[key], rtol=0, atol=0)


def test_empty_or_distant_local_neighborhood_remains_zero_after_bias_training():
    observation, evidence, batch, seeds, valid = inputs()
    evidence.point_valid.zero_()
    batch['point_valid'][:, -1] = False
    batch['points'][:, -1] = float('nan')
    reader = active_reader()
    increment, diag = reader(observation, evidence, batch, seeds, valid, evidence.members)
    assert increment.count_nonzero() == 0
    assert all(value.count_nonzero() == 0 for value in diag.values())
    batch['point_valid'][:, -1] = True
    batch['points'][:, -1] = 1000.
    increment, diag = reader(observation, evidence, batch, seeds, valid, evidence.members)
    assert increment.count_nonzero() == diag['local_neighbor_count'].count_nonzero() == 0


def test_local_world_rotation_and_gt_counterfactual_leave_evidence_unchanged():
    observation, evidence, batch, seeds, valid = inputs()
    reader = active_reader()
    before, _ = reader(observation, evidence, batch, seeds, valid, evidence.members)
    angle = torch.tensor([.83])
    changed = dict(batch)
    changed['points'] = torch.cat((rotate_xyz(batch['points'][..., :3], angle, to_world=True),
                                   batch['points'][..., 3:]), dim=-1)
    changed['target_box'] = torch.randn_like(batch['target_box']) * 100
    changed['segmentation_labels'] = torch.zeros_like(batch['segmentation_labels'])
    changed['bc_targets'] = torch.randn_like(batch['bc_targets'])
    rotated_evidence = replace(evidence, point_xyz=rotate_xyz(evidence.point_xyz, angle, to_world=True))
    after, _ = reader(observation, rotated_evidence, changed, transform_boxes(seeds, angle, to_world=True),
                       valid, evidence.members)
    torch.testing.assert_close(after, before, rtol=2e-5, atol=2e-6)


def test_local_features_are_live_while_geometry_support_votes_and_reliability_are_detached():
    observation, evidence, batch, seeds, valid = inputs()
    observation = replace(observation, point_features=observation.point_features.detach().requires_grad_(),
        foreground_probability=observation.foreground_probability.detach().requires_grad_())
    evidence = replace(evidence, point_features=evidence.point_features.requires_grad_(),
        point_xyz=evidence.point_xyz.requires_grad_(), selected_identity_logits=evidence.selected_identity_logits.requires_grad_(),
        vote_xyz=evidence.vote_xyz.requires_grad_(), reliability_logits=evidence.reliability_logits.requires_grad_())
    seeds.requires_grad_()
    batch['points'].requires_grad_()
    reader = active_reader()
    increment, _ = reader(observation, evidence, batch, seeds, valid, evidence.members)
    variables = (observation.point_features, evidence.point_features, seeds, batch['points'],
        evidence.point_xyz, observation.foreground_probability, evidence.selected_identity_logits,
        evidence.vote_xyz, evidence.reliability_logits)
    gradients = torch.autograd.grad(increment.square().sum(), variables, allow_unused=True)
    assert all(g is not None and g.abs().sum() > 0 for g in gradients[:2])
    assert all(g is None for g in gradients[2:])


def test_local_fixed_members_prevent_other_mode_feature_coordinate_bypass():
    observation, evidence, batch, seeds, valid = inputs()
    reader = active_reader()
    before, _ = reader(observation, evidence, batch, seeds, valid, evidence.members)
    changed_features = evidence.point_features.clone()
    changed_xyz = evidence.point_xyz.clone()
    changed_features[:, 8:24] += 100
    changed_xyz[:, 8:24] *= 10
    changed = replace(evidence, point_features=changed_features, point_xyz=changed_xyz,
                       vote_xyz=evidence.vote_xyz + 100, reliability_logits=evidence.reliability_logits + 100)
    after, _ = reader(observation, changed, batch, seeds, valid, evidence.members)
    torch.testing.assert_close(after[:, 1], before[:, 1], rtol=0, atol=0)
    assert not torch.equal(after[:, 0], before[:, 0])


def test_v35_decoder_preserves_mode_output_live_center_and_q0_coarse_query_gradient():
    observation, evidence, batch, _, _ = inputs()
    observation = replace(observation, coarse_box=observation.coarse_box.detach().requires_grad_())
    evidence.centers_xyz.requires_grad_()
    decoder = SharedHypothesisDecoder(query_context=True, initial_context=True, local_observation=True).eval()
    nn.init.normal_(decoder.local_reader.output.weight, std=.03)
    output = decoder(observation, evidence, batch, empty_prior(batch))
    feature_grad = torch.autograd.grad(output.decoder_features.square().sum(),
        (observation.coarse_box, evidence.centers_xyz), allow_unused=True, retain_graph=True)
    assert feature_grad[0] is not None and feature_grad[0].abs().sum() > 0
    assert feature_grad[1] is None
    center_grad, = torch.autograd.grad(output.hypothesis_boxes[:, 1:, :3].sum(), evidence.centers_xyz)
    torch.testing.assert_close(center_grad, torch.ones_like(center_grad), rtol=0, atol=0)


@pytest.mark.parametrize('arm', ['b0', 'full'])
def test_v35_actual_network_forward_backward_and_public_interfaces(arm):
    config = dict(model_config(arm), net_model='ctseqtrackv35', experiment_family='ct_seqtrack_v35')
    model = JointTracker(config).train()
    batch = make_batch(2, 24)
    batch['history_is_initial'] = torch.tensor([[True, False, False]]).expand(2, -1)
    batch['extension_ids'] += 1000
    output = model(batch)
    assert output.accepted_box.shape == (2, 4)
    assert model.decoder.history_context[0].in_features == 33
    assert model.observation.coarse_history_condition[0].in_features == 36
    for field in ('local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'):
        value = getattr(output.decoder, field)
        assert value.shape == (2, 4, 8) and not value.requires_grad
        if arm == 'b0':
            assert value[:, 1:].count_nonzero() == 0
    loss = model.compute_losses(batch, output)['loss_total']
    assert torch.isfinite(loss)
    loss.backward()
    # 零初始化pose/quality头先学习；第二次更新才检验新增局部出口的live梯度。
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-5)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    model.compute_losses(batch, output)['loss_total'].backward()
    assert model.decoder.local_reader.output.weight.grad.abs().sum() > 0
    assert model.observation.coarse_history_condition[-1].weight.grad.abs().sum() > 0
    assert all(p.requires_grad for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
