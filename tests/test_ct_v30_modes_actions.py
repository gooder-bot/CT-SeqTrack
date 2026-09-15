"""v30 模式预算、同状态动作、梯度所有权与可复现策略的 CPU 合同。"""
from dataclasses import replace

import pytest
import torch

from models.ct_v2.action_v30 import B3ModeUtilityUpdater, mode_action_geometry
from models.ct_v2.evidence_v30 import ModeEvidenceBuilder, extract_mode_geometry
from models.ct_v2.evidence_memory import B2EvidenceAcquirer
from utils.v30_policy import choose_mode_action, mechanism_behavior_policy
from utils.v30_training import (compute_b3_mode_utility_loss, mode_quality_loss,
                               legal_action_row_mean)


def _mode_inputs():
    votes = torch.tensor([[[.9, 0.], [1.1, 0.], [-1.5, 0.], [-1.6, .1], [0., 3.]]], requires_grad=True)
    weights = torch.tensor([[.9, .8, .7, .6, .2]], requires_grad=True)
    return dict(votes=votes, weights=weights, valid=torch.ones(1, 5, dtype=torch.bool),
                features=torch.randn(1, 5, 64, requires_grad=True), observation_xy=torch.zeros(1, 2),
                identity_margin=torch.tensor([[.5, .6, .3, .4, -.2]], requires_grad=True),
                point_ids=torch.arange(5)[None], b1_center_xy=torch.zeros(1, 2, requires_grad=True),
                b1_direction_xy=torch.tensor([[1., 0.]]),
                b1_sigma_parallel_perp=torch.ones(1, 2, requires_grad=True),
                support_half_size_parallel_perp=torch.tensor([[4., 3.]], requires_grad=True))


def _action_inputs(modes):
    return dict(observation_box=torch.tensor([[0., 0., .3, .7]], requires_grad=True),
                evidence_modes=modes, base_evidence=torch.randn(1, 64, requires_grad=True),
                base_presence_probability=torch.zeros(1, requires_grad=True),
                extension_presence_probability=torch.zeros(1, requires_grad=True),
                observation_stats=torch.zeros(1, 5, requires_grad=True),
                b1_sigma_parallel_perp=torch.ones(1, 2, requires_grad=True),
                query_delta_t=torch.tensor([.5], requires_grad=True), gap_ratio=torch.ones(1, requires_grad=True))


def test_local_weighted_support_beats_isolated_largest_weight_and_raw_count():
    votes = torch.tensor([[[0., 0.], [.05, 0.], [.1, 0.], [3., 0.]]])
    weights = torch.tensor([[.4, .4, .4, .9]])
    geo = extract_mode_geometry(votes, weights, torch.ones(1, 4, dtype=torch.bool),
        torch.zeros(1, 2), torch.zeros(1, 4), torch.tensor([[10, 11, 12, 13]]))
    assert geo['seed_point_ids'][0, 0] == 10
    assert geo['unique_count'][0, 0] == 3
    assert geo['centers_xy'][0, 0, 0] == pytest.approx(.05)
    assert geo['valid'].sum() == 2
    # 大量低 targetness 点不能通过旧 unweighted inlier-ratio 乘子反超。
    votes = torch.tensor([[[0., 0.]] * 2 + [[3., 0.]] * 8])
    weights = torch.tensor([[.9] * 2 + [.1] * 8])
    geo = extract_mode_geometry(votes, weights, torch.ones(1, 10, dtype=torch.bool),
        torch.zeros(1, 2), torch.zeros(1, 10), torch.arange(10)[None])
    assert geo['support_score'][0, 0] > geo['support_score'][0, 1]
    assert geo['support_score'][0, 0] == pytest.approx(1.8 / 2.6)


def test_mode_set_is_permutation_invariant_and_quality_does_not_reorder_modes():
    torch.manual_seed(30)
    builder, inputs = ModeEvidenceBuilder(), _mode_inputs()
    first = builder(**inputs)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    moved = dict(inputs)
    for key in ('votes', 'weights', 'valid', 'features', 'identity_margin', 'point_ids'):
        moved[key] = moved[key][:, permutation]
    second = builder(**moved)
    assert torch.allclose(first.centers_xy, second.centers_xy)
    assert torch.allclose(first.features, second.features, atol=1e-6)
    assert torch.equal(first.seed_point_ids, second.seed_point_ids)
    assert torch.equal(first.member_mask, second.member_mask[:, :, torch.argsort(permutation)])
    with torch.no_grad():
        builder.quality_head[-1].weight.mul_(-50)
    changed_quality = builder(**inputs)
    assert torch.equal(first.seed_point_ids, changed_quality.seed_point_ids)
    assert torch.equal(first.evidence_top_index, torch.tensor([0]))
    assert torch.all(first.support_score[:, :-1] >= first.support_score[:, 1:])


def test_single_mode_ablation_keeps_shapes_and_only_two_actions():
    modes = ModeEvidenceBuilder(mode_count=1)(**_mode_inputs())
    geometry = mode_action_geometry(torch.zeros(1, 4), modes, torch.tensor([.5]))
    assert modes.centers_xy.shape == (1, 3, 2)
    assert modes.valid.tolist() == [[True, False, False]]
    assert geometry['action_valid'].tolist() == [[True, True, False, False, False, False]]
    with pytest.raises(ValueError, match='mode_count'):
        ModeEvidenceBuilder(mode_count=2)


def test_quality_target_and_gradient_are_detached_from_geometry_and_targetness():
    inputs, builder = _mode_inputs(), ModeEvidenceBuilder()
    modes = builder(**inputs)
    data = {'ct_extension_labels': torch.tensor([[1., 1., 0., 0., 0.]], requires_grad=True),
            'ct_extension_valid_mask': torch.ones(1, 5),
            'box_label': torch.zeros(1, 4, requires_grad=True),
            'bbox_size': torch.tensor([[.2, .4, .3]], requires_grad=True)}
    output = {'ct_evidence_modes': modes, 'ct_extension_selected_indices': torch.arange(5)[None],
              'ct_extension_selected_valid_mask': torch.ones(1, 5)}
    result = mode_quality_loss(data, output)
    scale = .5 * torch.linalg.norm(data['bbox_size'][0, :2])
    expected = torch.exp(-modes.centers_xy[0, 0].square().sum() / (2 * scale.square()))
    assert result['purity'][0, 0] == 1
    assert result['target'][0, 0] == pytest.approx(float(expected))
    assert result['target'][0, 1] == 0
    result['loss'].backward()
    assert inputs['features'].grad.abs().sum() > 0
    assert builder.quality_head[-1].weight.grad.abs().sum() > 0
    for key in ('votes', 'weights', 'identity_margin', 'b1_center_xy', 'b1_sigma_parallel_perp', 'support_half_size_parallel_perp'):
        assert inputs[key].grad is None, key
    for value in data.values():
        assert value.grad is None
    assert not modes.centers_xy.requires_grad and not modes.covariance_xy.requires_grad


def test_six_actions_share_exact_observation_z_yaw_and_time_radius():
    modes = ModeEvidenceBuilder()(**_mode_inputs())
    obs = torch.tensor([[0., 0., .3, .7]])
    for dt, radius in ((0., .5), (.1, .55), (.5, .75), (7., 2.)):
        geometry = mode_action_geometry(obs, modes, torch.tensor([dt]))
        residual = geometry['residual'].reshape(1, 3, 2, 2)
        assert torch.allclose(residual[:, :, 0] * 2, residual[:, :, 1])
        assert torch.linalg.norm(residual[:, :, 1], dim=-1).max() <= radius + 1e-6
        assert geometry['radius'].item() == pytest.approx(radius)
        assert torch.equal(geometry['action_boxes'][..., 2:], obs[:, None, 2:].expand(-1, 6, -1))
    invalid = mode_action_geometry(obs, modes, torch.tensor([float('nan')]))
    assert not invalid['action_valid'].any()
    assert torch.equal(invalid['action_boxes'], obs[:, None].expand(-1, 6, -1))


def test_policy_q_tie_smaller_motion_then_mode_and_always_ignores_q():
    obs = torch.tensor([[0., 0., .2, .8]])
    actions = obs[:, None].expand(-1, 6, -1).clone()
    actions[0, :, 0] = torch.tensor([.375, .75, -.1, -.2, .1, .2])
    valid = torch.ones(1, 6, dtype=torch.bool)
    scores = torch.ones(1, 6) * .4
    result = choose_mode_action(obs, actions, valid, scores, {'kind': 'threshold', 'threshold': .3})
    assert result['chosen_action_index'].item() == 2
    assert result['best_action_index'].item() == 2
    result = choose_mode_action(obs, actions, valid, scores, {'kind': 'threshold', 'threshold': .4})
    assert result['chosen_action_index'].item() == -1
    assert torch.equal(result['final_box'], obs)
    scores[:] = float('nan')
    result = choose_mode_action(obs, actions, valid, scores, {'kind': 'always'})
    assert result['chosen_action_index'].item() == 1
    assert result['final_box'][0, 0] == .75
    result = choose_mode_action(obs, actions, valid, scores, {'kind': 'threshold', 'threshold': -1})
    assert not result['applied'].any()
    assert torch.equal(result['final_box'], obs)


def test_behavior_is_track_fixed_exploration_is_frame_specific_and_rng_unchanged():
    rng = torch.get_rng_state().clone()
    policies = [mechanism_behavior_policy(42, 0, 'scene/track' + str(i), 0) for i in range(500)]
    kinds = {p['kind'] for p in policies}
    assert kinds == {'never', 'always', 'explore', 'threshold'}
    key = next('scene/track' + str(i) for i, p in enumerate(policies) if p['kind'] == 'explore')
    p0 = mechanism_behavior_policy(42, 0, key, 0)
    p1 = mechanism_behavior_policy(42, 0, key, 1)
    assert p0['kind'] == p1['kind'] == 'explore'
    assert p0['action_seed'] != p1['action_seed']
    assert p0 == mechanism_behavior_policy(42, 0, key, 0)
    assert torch.equal(rng, torch.get_rng_state())


def test_b3_loss_has_geometric_signal_beyond_sp_dead_zone_and_detaches_all_sources():
    builder_inputs, builder = _mode_inputs(), ModeEvidenceBuilder()
    modes = builder(**builder_inputs)
    # 所有候选都朝远处目标走，但仍停留在 P 的 2m 饱和区且框不重叠。
    modes = replace(modes, centers_xy=torch.tensor([[[3., 0.], [4., 0.], [5., 0.]]]))
    inputs, updater = _action_inputs(modes), B3ModeUtilityUpdater(require_calibration=True)
    final, output = updater(**inputs)
    output['observation_aux_estimation_boxes'] = inputs['observation_box']
    data = dict(box_label=torch.tensor([[4., 0., .3, .7]], requires_grad=True),
                bbox_size=torch.tensor([[.2, .4, .3]]), target_bbox_size=torch.tensor([[.2, .4, .3]]))
    result = compute_b3_mode_utility_loss(data, output, {})
    assert torch.equal(final, inputs['observation_box'])
    assert not result['action_h1_success_gain'].any()
    assert not result['action_h1_precision_gain'].any()
    assert torch.allclose(result['action_h1_distance_gain'], torch.tensor([[.5, 1., .5, 1., .5, 1.]]))
    assert result['loss_distance'] > 0
    assert result['loss_iou'] == 0
    result['loss'].backward()
    assert updater.expected_distance_gain_head.weight.grad.abs().sum() > 0
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            assert value.grad is None, key
    assert builder_inputs['features'].grad is None
    assert builder.quality_head[-1].weight.grad is None
    assert data['box_label'].grad is None


def test_row_reduction_gives_each_legal_row_equal_weight_and_empty_has_finite_gradient():
    error = torch.tensor([[1., 2., 3., 4., 5., 6.], [10., 999., 999., 999., 999., 999.]], requires_grad=True)
    valid = torch.tensor([[True] * 6, [True, False, False, False, False, False]])
    value = legal_action_row_mean(error, valid)
    assert value.item() == pytest.approx(6.75)
    value.backward()
    assert error.grad[0, 0] == pytest.approx(1 / 12)
    assert error.grad[1, 0] == .5
    empty = error.detach().clone().requires_grad_()
    loss = legal_action_row_mean(empty, torch.zeros_like(valid))
    loss.backward()
    assert loss.item() == 0 and not empty.grad.any()


def _b2_inputs():
    n = 8
    ids = torch.full((1, 768), -1, dtype=torch.long)
    ids[0, :n] = torch.arange(n) + 10000
    base_ids = torch.full((1, 1024), -1, dtype=torch.long)
    base_ids[0, :4] = torch.arange(4)
    points = torch.zeros(1, 768, 5)
    points[0, :n, 0] = torch.linspace(.5, 2, n)
    return dict(extension_points=points, extension_valid_mask=ids >= 0,
        extension_source=(ids >= 0).long(), extension_point_ids=ids,
        current_base_features=torch.randn(1, 1024, 64, requires_grad=True),
        current_base_valid_mask=base_ids >= 0, current_base_point_ids=base_ids,
        memory_tokens=torch.zeros(1, 36, 64), memory_valid_mask=torch.zeros(1, 36, dtype=torch.bool),
        observation_box=torch.zeros(1, 4), observation_stats=torch.zeros(1, 5),
        b1_center_xy=torch.zeros(1, 2), b1_sigma_parallel_perp=torch.tensor([[.5, 2.]]),
        b1_direction_xy=torch.tensor([[1., 0.]]), b1_valid=torch.ones(1),
        query_delta_t=torch.tensor([.5]), gap_ratio=torch.ones(1),
        first_box_size_wlh=torch.tensor([[1.8, 4.2, 1.6]]),
        support_half_size_parallel_perp=torch.tensor([[4., 3.]]))


def test_b2_real_forward_uses_support_for_geometry_and_separate_log_sigma():
    torch.manual_seed(30)
    module = B2EvidenceAcquirer(v27_enabled=True, v28_enabled=True, v30_enabled=True,
        relation_aware_sampling=True, robust_consensus_voting=True).eval()
    inputs = _b2_inputs()
    first = module(**inputs)
    geometry = first['ct_b2_geometry_features_prepool']
    assert geometry.shape == (1, 768, 7)
    assert torch.allclose(geometry[0, :8, 0], inputs['extension_points'][0, :8, 0] / 4)
    assert torch.allclose(geometry[0, :8, 5], torch.full((8,), .5).log())
    assert first['ct_mode_features'].shape == (1, 3, 64)
    assert first['ct_mode_valid'].any()
    first['ct_mode_quality_logit'].sum().backward()
    assert module.mode_builder.quality_head[-1].weight.grad.abs().sum() > 0
    assert inputs['current_base_features'].grad is None
    assert module.vote_head[-1].weight.grad is None
    assert module.targetness_head[-1].weight.grad is None
    changed = dict(inputs, b1_sigma_parallel_perp=torch.tensor([[3., 4.]]))
    second = module(**changed)
    assert torch.equal(geometry[..., :2], second['ct_b2_geometry_features_prepool'][..., :2])
    assert not torch.equal(geometry[..., 5:], second['ct_b2_geometry_features_prepool'][..., 5:])
    with pytest.raises(ValueError, match='actual support'):
        module(**dict(inputs, support_half_size_parallel_perp=None))


def test_empty_and_nonfinite_modes_cannot_execute_and_have_finite_quality_loss():
    inputs = _mode_inputs()
    inputs['valid'].zero_()
    modes = ModeEvidenceBuilder()(**inputs)
    assert not modes.valid.any() and modes.evidence_top_index.item() == -1
    updater = B3ModeUtilityUpdater()
    updater.install_policy({'kind': 'always'})
    action_inputs = _action_inputs(modes)
    final, output = updater(**action_inputs)
    assert torch.equal(final, action_inputs['observation_box'])
    assert not output['ct_b3_action_valid'].any()
    assert torch.isfinite(output['ct_b3_action_scores']).all()
    data = dict(ct_extension_labels=torch.zeros(1, 5), ct_extension_valid_mask=torch.zeros(1, 5),
                box_label=torch.zeros(1, 4), bbox_size=torch.ones(1, 3))
    result = mode_quality_loss(data, dict(ct_evidence_modes=modes,
        ct_extension_selected_indices=torch.arange(5)[None], ct_extension_selected_valid_mask=torch.zeros(1, 5)))
    assert result['loss'] == 0
    result['loss'].backward()
    assert inputs['features'].grad is not None and not inputs['features'].grad.any()


def test_fixed_slot_mode_extraction_matches_batched_and_separate_strict_runs():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        generator = torch.Generator().manual_seed(30)
        votes = torch.randn(3, 256, 2, generator=generator)
        weights = torch.rand(3, 256, generator=generator)
        valid = torch.arange(256)[None] < torch.tensor([[256], [7], [0]])
        ids = torch.arange(256)[None].expand(3, -1).masked_fill(~valid, -1)
        args = (votes, weights, valid, torch.zeros(3, 2), torch.zeros(3, 256), ids)
        first = extract_mode_geometry(*args)
        repeated = extract_mode_geometry(*args)
        for name, value in first.items():
            assert torch.equal(value, repeated[name]), name
            for row in range(3):
                single = extract_mode_geometry(*(arg[row:row + 1] for arg in args))
                assert torch.equal(value[row:row + 1], single[name]), name
        assert not first['valid'][2].any()
        assert torch.isfinite(first['covariance_xy']).all()
    finally:
        torch.use_deterministic_algorithms(previous)
