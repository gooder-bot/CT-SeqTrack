"""B2 重心权重开关：默认合同、相同前向及明确的直接梯度边界。"""
from dataclasses import fields, is_dataclass

import pytest
import torch

from models.ct_v36.evidence import B2IdentityEvidence, build_live_modes
from models.ct_v37.model import JointTracker
from tests.test_ct_v37_network import batch, config


@pytest.fixture(autouse=True)
def cpu_rng_and_threads():
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(37)
    yield
    torch.set_rng_state(previous_rng)
    torch.set_num_threads(previous_threads)


def _case(name, dtype):
    votes = torch.tensor([[[.2, .1, .3], [.45, -.2, .4],
                           [3., .1, -.2], [3.3, .1, .1]]], dtype=dtype)
    weights = torch.tensor([[.2, .8, .3, .6]], dtype=dtype)
    valid = torch.ones(1, 4, dtype=torch.bool)
    ids = torch.tensor([[9, 3, 7, 2]])
    if name == 'one_point':
        valid[:, 1:] = False
        ids[:, 1:] = -1
    elif name == 'two_points':
        valid[:, 2:] = False
        ids[:, 2:] = -1
    elif name == 'empty_slots':
        valid.zero_()
        ids.fill_(-1)
    elif name == 'duplicate_ids':
        ids[:] = torch.tensor([[9, 9, 7, 2]])
    elif name == 'nonfinite':
        votes[0, 0, 0] = float('nan')
        weights[0, 1] = float('inf')
        weights[0, 2] = float('nan')
    elif name == 'zero_and_tiny_mass':
        votes[0, :2] = torch.tensor([[.2, .1, .3], [.3, .2, .4]], dtype=dtype)
        weights[:] = torch.tensor([[1e-12, 1e-10, 0., -.1]], dtype=dtype)
    else:
        assert name == 'two_clusters'
    return votes, weights, valid, ids


def _pure_run(values, kwargs):
    votes, weights, valid, ids = values
    votes = votes.clone().requires_grad_()
    weights = weights.clone().requires_grad_()
    output = build_live_modes(votes, weights, valid, ids, **kwargs)
    loss = output[0].square().sum() + (output[0] * votes.new_tensor([1.3, -.7, .4])).sum()
    gradients = torch.autograd.grad(loss, (votes, weights), allow_unused=True)
    return output, gradients


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('name', ['two_clusters', 'one_point', 'two_points', 'empty_slots',
                                  'duplicate_ids', 'nonfinite', 'zero_and_tiny_mass'])
def test_seven_pool_cases_preserve_forward_and_independent_vote_gradient(name, dtype):
    values = _case(name, dtype)
    default, default_grad = _pure_run(values, {})
    live, live_grad = _pure_run(values, {'detach_mass_weights': False})
    detached, detached_grad = _pure_run(values, {'detach_mass_weights': True})
    for actual in (live, detached):
        for expected, value in zip(default, actual):
            assert torch.equal(expected, value)
            assert torch.isfinite(value).all()
    assert torch.equal(default_grad[0], live_grad[0])
    assert torch.equal(default_grad[1], live_grad[1])
    assert torch.equal(live_grad[0], detached_grad[0])
    assert torch.isfinite(detached_grad[0]).all()
    assert detached_grad[1] is None
    if name == 'two_clusters':
        # 非退化对照，避免旧权重梯度本来就为零而虚假通过。
        assert live_grad[0].abs().sum() > 0
        assert live_grad[1].abs().sum() > 0


@pytest.mark.parametrize('value', [None, 0, 1, 'False'])
def test_switch_rejects_non_boolean_values(value):
    with pytest.raises(TypeError, match='detach_mass_weights must be bool'):
        B2IdentityEvidence(detach_mass_weights=value)
    with pytest.raises(TypeError, match='detach_mass_weights must be bool'):
        build_live_modes(*_case('two_clusters', torch.float32), detach_mass_weights=value)


def test_constructor_switch_preserves_initialization_rng_and_state_dict():
    models, states = [], []
    for kwargs in ({}, {'detach_mass_weights': False}, {'detach_mass_weights': True}):
        torch.manual_seed(425)
        model = B2IdentityEvidence(**kwargs)
        models.append(model)
        states.append(torch.get_rng_state().clone())
    assert [m.detach_mass_weights for m in models] == [False, False, True]
    assert all(torch.equal(states[0], state) for state in states[1:])
    reference = models[0].state_dict()
    for model in models[1:]:
        assert reference.keys() == model.state_dict().keys()
        assert dict(models[0].named_buffers()).keys() == dict(model.named_buffers()).keys()
        for name, value in reference.items():
            assert torch.equal(value, model.state_dict()[name]), name
    assert not any('detach_mass_weights' in name for name in reference)


def _model_pair(arm='full', training=False, nonzero_heads=False):
    cfg = config(arm)
    initial_rng = torch.get_rng_state().clone()
    live = JointTracker(cfg).train(training)
    live_rng = torch.get_rng_state().clone()
    assert live.evidence.detach_mass_weights is False
    torch.set_rng_state(initial_rng)
    detached = JointTracker(dict(cfg, v37_centroid_recipe=arm + '_mass_detach')).train(training)
    assert detached.evidence.detach_mass_weights is True
    assert torch.equal(live_rng, torch.get_rng_state())
    assert live.state_dict().keys() == detached.state_dict().keys()
    for name, value in live.state_dict().items():
        assert torch.equal(value, detached.state_dict()[name]), name
    if nonzero_heads:
        torch.nn.init.normal_(live.decoder.mode_pose_head.weight, std=.03)
        torch.nn.init.normal_(live.decoder.local_reader.output.weight, std=.03)
        torch.nn.init.normal_(live.verifier.output.weight, std=.02)
        detached.load_state_dict(live.state_dict())
    return live, detached


def _assert_same_values(left, right, path='output'):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and torch.equal(left, right), path
    elif is_dataclass(left):
        assert type(left) is type(right), path
        for field in fields(left):
            _assert_same_values(getattr(left, field.name), getattr(right, field.name), path + '.' + field.name)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_same_values(left[key], right[key], path + '.' + key)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right)):
            _assert_same_values(first, second, path + '.' + str(index))
    else:
        assert left == right, path


@pytest.mark.parametrize('arm', ['b1_b2', 'full'])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('nonzero_heads', [False, True])
def test_complete_forward_losses_selection_and_rng_remain_identical(arm, training, nonzero_heads):
    models = _model_pair(arm, training, nonzero_heads)
    values = batch()
    initial_rng = torch.get_rng_state().clone()
    outputs, losses, end_rng = [], [], []
    for model in models:
        torch.set_rng_state(initial_rng)
        output = model(values)
        output.loss_statistics = {}
        outputs.append(output)
        losses.append(model.compute_losses(values, output))
        end_rng.append(torch.get_rng_state().clone())
    _assert_same_values(outputs[0], outputs[1])
    _assert_same_values(losses[0], losses[1], 'losses')
    _assert_same_values(outputs[0].loss_statistics, outputs[1].loss_statistics, 'loss_statistics')
    assert torch.equal(*end_rng)
    for name, value in models[0].named_buffers():
        assert torch.equal(value, dict(models[1].named_buffers())[name]), name


@pytest.mark.parametrize('nonzero_heads', [False, True])
def test_mode_geometry_cuts_only_direct_score_head_paths_and_keeps_vote_b0_live(nonzero_heads):
    values = batch(1)
    models = _model_pair(nonzero_heads=nonzero_heads)
    for detached, model in zip((False, True), models):
        output = model(values)
        losses = model.compute_losses(values, output)
        assert losses['loss_modes_center'] > 0
        variables = (model.evidence.identity_head[-1].weight,
                     model.evidence.reliability_head[-1].weight,
                     model.evidence.vote_head[-1].weight,
                     model.evidence.point_encoder[0].weight,
                     output.observation.point_features)
        gradients = torch.autograd.grad(losses['loss_modes'], variables, allow_unused=True, retain_graph=True)
        for gradient in gradients[2:]:
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        if detached:
            assert gradients[:2] == (None, None)
        else:
            assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients[:2])
        # 两个显式任务仍可训练自己的头，不通过 detach 冻结参数。
        for loss_name, head in (('loss_identity', model.evidence.identity_head),
                                ('loss_reliability', model.evidence.reliability_head)):
            gradient = torch.autograd.grad(losses[loss_name], head[-1].weight, retain_graph=True)[0]
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize('empty', [False, True])
def test_background_and_empty_batches_keep_finite_explicit_supervision(empty):
    model = _model_pair(training=True)[1]
    values = batch(1)
    values['extension_labels'].zero_()
    if empty:
        values['point_valid'].zero_()
        values['extension_valid'].zero_()
    output = model(values)
    losses = model.compute_losses(values, output)
    assert losses['loss_modes'] == 0 and losses['loss_vote'] == 0
    if not empty:
        assert losses['loss_identity'] > 0 and losses['loss_reliability'] > 0
    losses['loss_total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
