"""事件起点、时间积分、梯度隔离与 E 分支语义的 CPU 合同。"""
import copy
import torch
import pytest

from models.ct_v36.prior import EventPrior, LegacyMotionCore, decayed_acceleration_displacement
from models.ct_v31.prior import PhysicalTimePrior


def event_batch():
    world = torch.tensor([[[1e8, -1e8, 4., .3], [1e8 + 1., -1e8, 4., .3],
                           [1e8 + 2., -1e8, 4., .3]]], dtype=torch.float64)
    anchor = torch.tensor([[1e8 + 12., -1e8 - 5., 3., -.8]], dtype=torch.float64)
    history = world.clone()
    history[..., :3] -= anchor[:, None, :3]
    return dict(event_boxes=world, event_times=torch.tensor([[0., .5, 1.]], dtype=torch.float64),
        event_valid=torch.ones(1, 3, dtype=torch.bool),
        event_edge_valid=torch.tensor([[False, True, True]]), event_segment=torch.zeros(1, 3, dtype=torch.long),
        event_geometry=torch.zeros(1, 3, 8), event_source=torch.zeros(1, 3, dtype=torch.long),
        event_initial=torch.tensor([[True, False, False]]), event_frame_ids=torch.tensor([[0, 1, 2]]),
        current_time=torch.tensor([1.5], dtype=torch.float64), current_frame=torch.tensor([3]),
        anchor_box=anchor, box_size=torch.tensor([[4., 2., 2.]]), points=torch.zeros(1, 4, 8, 5),
        history_boxes=history.float(), history_times=torch.tensor([[-1.5, -1., -.5]]),
        history_valid=torch.ones(1, 3, dtype=torch.bool), history_pair_valid=torch.ones(1, 2, dtype=torch.bool),
        current_dt=torch.tensor([.5]), history_geometry=torch.zeros(1, 3, 8),
        history_is_initial=torch.tensor([[True, False, False]]))


@pytest.mark.parametrize('backend', ['learned', 'cv'])
def test_world_forecast_uses_event_origin_and_ignores_unsupported_anchor_and_queries(backend):
    model = EventPrior(backend=backend).eval()
    batch = event_batch()
    first = model(batch)
    torch.testing.assert_close(first.world_box[0, :3], torch.tensor([1e8 + 3., -1e8, 4.], dtype=torch.float64))
    shifted = copy.deepcopy(batch)
    shifted['anchor_box'][:, :3] += torch.tensor([20., -12., 7.])
    shifted['current_time'][:] = 1.25
    model(shifted)  # 额外查询不得推进永久 hidden。
    shifted['current_time'][:] = 1.5
    second = model(shifted)
    torch.testing.assert_close(second.world_box, first.world_box, rtol=0, atol=0)
    torch.testing.assert_close(second.range_xy, first.range_xy, rtol=0, atol=0)
    torch.testing.assert_close(second.event_feature, first.event_feature, rtol=0, atol=0)
    torch.testing.assert_close(second.box[:, :3], first.box[:, :3] - torch.tensor([20., -12., 7.]))


def test_single_event_and_new_segment_have_position_without_phantom_velocity():
    model = EventPrior().eval()
    batch = event_batch()
    batch['event_valid'][:, :2] = False
    batch['event_edge_valid'][:] = False
    out = model(batch)
    assert out.prior_valid.item() and not out.motion_valid.item()
    torch.testing.assert_close(out.world_box, batch['event_boxes'][:, -1])
    assert out.range_xy.min() > 0
    batch = event_batch()
    batch['event_segment'][:, -1] = 1
    batch['event_edge_valid'][:, -1] = False
    out = model(batch)
    assert not out.motion_valid.item()
    torch.testing.assert_close(out.velocity, torch.zeros_like(out.velocity))


def test_integral_zero_small_and_long_time_and_gradients():
    v = torch.tensor([[3., -1.]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[2., .5]], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(decayed_acceleration_displacement(v, a, torch.tensor([0.]), tau), torch.zeros_like(v))
    small = torch.tensor([1e-7], dtype=torch.float64)
    actual = decayed_acceleration_displacement(v, a, small, tau)
    torch.testing.assert_close(actual, v * small + .5 * a * small.square(), rtol=1e-12, atol=1e-25)
    long = decayed_acceleration_displacement(v, a, torch.tensor([100.]), tau)
    torch.testing.assert_close(long, v * 100 + a * 99)
    long.sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in (v, a, tau))


def test_range_current_loss_only_reaches_independent_rate_head():
    model = EventPrior().train()
    out = model(event_batch())
    out.range_for_loss.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.range_rate_head.parameters())
    assert all(p.grad is None for p in model.range_anchor_head.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.velocity_head.parameters())
    assert not out.world_box.requires_grad and not out.box.requires_grad and not out.range_xy.requires_grad
    model.zero_grad(set_to_none=True)
    out = model(event_batch())
    out.range_anchor.sum().backward()
    assert any(p.grad is not None for p in model.range_anchor_head.parameters())
    assert all(p.grad is None for p in model.range_rate_head.parameters())


def test_motion_is_live_but_never_backpropagates_into_event_state():
    model = EventPrior()
    batch = event_batch()
    batch['event_boxes'].requires_grad_()
    model(batch).mean_xy.square().sum().backward()
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert model.acceleration_head.weight.grad.abs().sum() > 0
    assert batch['event_boxes'].grad is None


def test_common_component_initialization_is_independent_of_arm_and_global_rng():
    torch.manual_seed(17)
    learned = EventPrior(seed=42)
    torch.rand(80)
    cv = EventPrior(backend='cv', seed=42)
    for name, value in cv.state_dict().items():
        torch.testing.assert_close(value, learned.state_dict()[name], rtol=0, atol=0)
    assert not any(name.startswith(('velocity_head', 'acceleration_head', 'tau_parameter')) for name in cv.state_dict())
    legacy = EventPrior(backend='legacy', seed=42)
    assert not hasattr(legacy, 'encoder') and not hasattr(legacy, 'query_context')
    for name, value in legacy.state_dict().items():
        if name.startswith(('range_anchor_head.', 'range_rate_head.')):
            torch.testing.assert_close(value, learned.state_dict()[name], rtol=0, atol=0)
    assert legacy.range_anchor_head[0].in_features == 128


def test_range_grows_at_fixed_event_and_model():
    model = EventPrior()
    batch = event_batch()
    first = model(batch)
    batch['current_time'][:] = 5.
    last = model(batch)
    torch.testing.assert_close(first.range_anchor, last.range_anchor)
    torch.testing.assert_close(first.range_rate, last.range_rate)
    assert (last.range_xy >= first.range_xy).all()


def test_trained_event_parameters_are_query_invariant_and_discard_old_segments():
    model = EventPrior().eval()
    generator = torch.Generator().manual_seed(901)
    # 初始零出口会掩盖错误的 T/anchor 依赖；使用非零学习后参数检验合同。
    with torch.no_grad():
        for head in (model.velocity_head, model.acceleration_head,
                     model.range_anchor_head[-1], model.range_rate_head[-1]):
            head.weight.copy_(.01 * torch.randn(head.weight.shape, generator=generator))
    batch = event_batch()
    first = model(batch)
    future = copy.deepcopy(batch)
    future['current_time'][:] = 12.
    future['anchor_box'][:, :3] += 123.
    later = model(future)
    for field in ('event_feature', 'velocity', 'acceleration', 'tau', 'range_anchor', 'range_rate'):
        torch.testing.assert_close(getattr(first, field), getattr(later, field), rtol=0, atol=0)
    assert (later.range_xy >= first.range_xy).all()
    future['current_time'][:] = batch['current_time']
    queried_again = model(future)
    torch.testing.assert_close(queried_again.world_box, first.world_box, rtol=0, atol=0)
    batch['event_segment'][:, -1] = 1
    batch['event_edge_valid'][:, -1] = False
    reset = model(batch)
    batch['event_boxes'][:, :2, :3] += torch.tensor([1000., -2000., 30.])
    batch['event_geometry'][:, :2] += 11.
    changed_old_segment = model(batch)
    torch.testing.assert_close(changed_old_segment.event_feature, reset.event_feature, rtol=0, atol=0)
    torch.testing.assert_close(changed_old_segment.world_box, reset.world_box, rtol=0, atol=0)


def test_legacy_valid_and_fallback_use_their_actual_origins():
    model = EventPrior(backend='legacy')
    batch = event_batch()
    out = model(batch)
    assert out.legacy_valid.item()
    torch.testing.assert_close(out.origin_world, batch['anchor_box'])
    assert out.origin_frame.item() == 2 and out.horizon.item() == .5
    torch.testing.assert_close(out.world_box[:, :2], batch['anchor_box'][:, :2] + out.mean_xy.detach().double())
    torch.testing.assert_close(out.world_box[:, 2:], batch['event_boxes'][:, -1, 2:])
    batch['history_pair_valid'][:] = False
    batch['current_time'][:] = 3.
    batch['current_frame'][:] = 7
    out = model(batch)
    assert not out.legacy_valid.item()
    torch.testing.assert_close(out.feature, model.legacy(batch)['feature'])
    torch.testing.assert_close(out.origin_world, batch['event_boxes'][:, -1])
    assert out.origin_frame.item() == 2 and out.horizon.item() == 2.
    torch.testing.assert_close(out.world_box[:, :2], batch['event_boxes'][:, -1, :2] + torch.tensor([[4., 0.]], dtype=torch.float64))


@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_legacy_mean_math_matches_old_core_when_weights_are_equal(backend):
    batch = event_batch()
    old = PhysicalTimePrior(temporal_backend=backend)
    new = LegacyMotionCore(.5, backend, 42)
    for name in ('step_projection', 'temporal_cell', 'context', 'mean_head'):
        getattr(new, name).load_state_dict(getattr(old, name).state_dict())
    batch.update(point_valid=torch.ones(1, 4, 8, dtype=torch.bool), fallback_box=torch.zeros(1, 4))
    a, b = old(batch), new(batch)
    torch.testing.assert_close(a.mean_xy, b['mean'], rtol=0, atol=0)
    torch.testing.assert_close(a.feature, b['feature'], rtol=0, atol=0)
    torch.testing.assert_close(a.kinematic_xy, b['kinematic'], rtol=0, atol=0)


def test_nonfinite_and_reversed_event_time_fail_explicitly():
    model = EventPrior()
    batch = event_batch()
    batch['event_times'][0, -1] = .4
    with pytest.raises(ValueError, match='strictly increase'):
        model(batch)
    batch = event_batch()
    batch['event_boxes'][0, -1, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='event boxes'):
        model(batch)
    batch = event_batch()
    batch['anchor_box'][0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='accepted anchor'):
        model(batch)


def test_mean_does_not_use_a_query_displacement_cap():
    model = EventPrior(backend='cv')
    batch = event_batch()
    batch['current_time'][:] = 100.
    out = model(batch)
    torch.testing.assert_close(out.mean_xy, torch.tensor([[198., 0.]]))
    assert out.mean_xy.norm() > 12.
