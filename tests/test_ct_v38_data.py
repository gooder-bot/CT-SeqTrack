"""v38 强 B1 数据等价、无身份记忆和三候选可信状态合同。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v38.config import normalize_config, recipe_scope
from models.ct_v38.data import BatchBuilder, RawEndpointDataset
from tests.helpers.raw_tracking import TinySource, request, fake_prior, fake_output
from tests.helpers.assertions import _assert_identical


RISK_KEYS = {'previous_is_propagation', 'consecutive_propagations',
            'consecutive_untrusted_outputs', 'trusted_velocity_age'}
MEMORY_KEYS = {'memory_points', 'memory_ids', 'memory_valid', 'memory_metadata'}


def cfg(arm='full'):
    return normalize_config(dict(v38_arm=arm, ct_engineering_check=True,
        point_sample_size=8, batch_size=4, workers=0, v32_reserve_windows=0, epoch=11))


def output_for(batch, *, selected=0, trusted=.9, legacy=.8, delta=0., probabilities=None):
    from models.ct_v38.verification import current_q0_support
    out = fake_output(batch, quality=legacy)
    out.observation.point_features = batch['points'].new_zeros((*batch['point_valid'].shape, 64))
    boxes = batch['fallback_box'][:, None].repeat(1, 3, 1)
    boxes[:, 0, 0] += delta
    out.candidates = SimpleNamespace(boxes=boxes, valid=torch.ones(len(boxes), 3, dtype=torch.bool))
    out.q0_is_fallback = torch.zeros(len(boxes), dtype=torch.bool)
    out.selected_index.fill_(selected)
    out.accepted_box = boxes[:, selected]
    out.legacy_q0_quality = torch.full((len(boxes),), legacy)
    out.selected_quality = out.legacy_q0_quality.clone() if selected == 0 else torch.zeros(len(boxes))
    if probabilities is not None:
        out.observation.foreground_probability[:, -1] = probabilities
    support = current_q0_support(out.observation, batch, boxes[:, 0])['mask']
    support = support & (selected == 0)
    score = torch.full((len(boxes),), trusted if selected == 0 else 0.)
    out.commit_decision = SimpleNamespace(trusted_score=score, support_mask=support,
        eligible=(support.sum(-1) >= 3) & (score >= .5), source=out.selected_index.clone())
    return out


def next_batch(builder, raw, frame):
    row = raw[request(frame)]
    batch = builder.prepare([row])
    builder.supervise_prior(batch, fake_prior(batch))
    return row, batch






def test_q0_prior_trusted_propagation_q0_separates_position_and_velocity():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    state = builder.states[row['request'].state_key]
    record = builder.commit(output_for(batch), batch, diagnostics=True)[0]
    assert record['supported'] and record['trusted_velocity_valid']
    trusted = deepcopy(state.trusted)
    strong_time, supported_time = state.last_strong_time, state.last_supported_time
    velocity_time = state.last_velocity_update_time
    accepted_world = state.boxes[1].copy()
    for frame, index in ((2, 1), (3, 2)):
        _, batch = next_batch(builder, raw, frame)
        np.testing.assert_allclose(batch['anchor_box'][0].numpy(), accepted_world)
        out = output_for(batch, selected=index)
        # 当前框内点很多也不能使传播候选伪装成观测。
        assert batch['point_valid'][0, -1].sum() >= 3
        record = builder.commit(out, batch, diagnostics=True)[0]
        assert record['propagation_only'] and not record['strong'] and not record['supported']
        assert state.last_strong_time == strong_time
        assert state.last_supported_time == supported_time
        assert state.last_velocity_update_time == velocity_time
        assert state.previous_quality == 0.
        _assert_identical(state.trusted, trusted)
        assert state.transitions[frame]  # 原 motion-pair 数学没有按来源重写。
        assert record['consecutive_propagations'] == frame - 1
        accepted_world = state.boxes[frame].copy()
    _, batch = next_batch(builder, raw, 4)
    assert batch['previous_is_propagation'].item()
    assert batch['consecutive_propagations'].item() == 2
    assert batch['trusted_velocity_age'].item() == pytest.approx(1.5)
    record = builder.commit(output_for(batch, legacy=.73), batch, diagnostics=True)[0]
    assert record['trusted_position_updated'] and not record['trusted_velocity_updated']
    assert not state.trusted_velocity_valid and np.array_equal(state.trusted_velocity, np.zeros(3))
    assert state.previous_quality == pytest.approx(.73)
    assert state.last_velocity_update_time == velocity_time
    assert record['trusted_velocity_age'] == 0.
    assert record['consecutive_propagations'] == record['consecutive_untrusted_outputs'] == 0
    _, batch = next_batch(builder, raw, 5)
    record = builder.commit(output_for(batch), batch, diagnostics=True)[0]
    assert record['trusted_velocity_updated'] and state.trusted_velocity_valid


def test_rejected_q0_updates_only_strong_time_and_never_uses_legacy_quality_as_T():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    state = builder.states[row['request'].state_key]
    old_position_time, old_velocity_time = state.last_supported_time, state.last_velocity_update_time
    record = builder.commit(output_for(batch, legacy=.99, trusted=.1), batch, diagnostics=True)[0]
    assert record['strong'] and not record['supported']
    assert state.last_strong_time == 100.5
    assert state.last_supported_time == old_position_time
    assert state.last_velocity_update_time == old_velocity_time
    assert state.previous_quality == pytest.approx(.99)
    assert state.consecutive_untrusted_outputs == 1 and state.consecutive_propagations == 0


@pytest.mark.parametrize('support_count', [0, 1, 2])
def test_high_T_cannot_write_with_insufficient_current_unique_support(support_count):
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    _, batch = next_batch(builder, raw, 1)
    probability = torch.zeros_like(batch['point_valid'][:, -1], dtype=torch.float)
    probability[:, :support_count] = 1.
    record = builder.commit(output_for(batch, trusted=.99, probabilities=probability),
                            batch, diagnostics=True)[0]
    assert record['geometric_fg_count'] == support_count
    assert not record['supported']


def test_duplicate_ids_do_not_create_support_and_wrong_commit_mask_is_rejected():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    batch['point_ids'][0, -1, 1:4] = batch['point_ids'][0, -1, 0]
    out = output_for(batch, trusted=.99)
    assert out.commit_decision.support_mask.sum() == 1
    out.commit_decision.support_mask[0, 1] = True
    state = builder.states[row['request'].state_key]
    with pytest.raises(ValueError, match='support differs'):
        builder.commit(out, batch)
    assert state.next_frame == 1 and 1 not in state.boxes


def test_tampered_eligibility_rejected_before_state_write():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    out = output_for(batch, trusted=.1)
    out.commit_decision.eligible[:] = True
    state = builder.states[row['request'].state_key]
    with pytest.raises(ValueError, match='eligibility disagrees'):
        builder.commit(out, batch)
    assert state.next_frame == 1


def test_current_GT_never_enters_commit_and_next_crop_follows_selected_output():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    alternative = deepcopy(builder)
    alternative._pending[0]['current_gt'][:3] += 100.
    alternative._pending[0]['row']['frames'][1]['3d_bbox'][:3] += 100.
    out = output_for(batch, delta=1., legacy=.9, trusted=.1)
    left = builder.commit(out, batch, diagnostics=True)
    right = alternative.commit(deepcopy(out), deepcopy(batch), diagnostics=True)
    assert left == right
    _assert_identical(vars(builder.states[row['request'].state_key]),
                      vars(alternative.states[row['request'].state_key]))
    next_row = raw[request(2)]
    next_prepared = builder.prepare([next_row])
    expected = batch['anchor_box'][0].numpy().copy()
    expected[:3] += out.accepted_box[0, :3].numpy()
    expected[3] = out.accepted_box[0, 3].item()
    np.testing.assert_allclose(next_prepared['anchor_box'][0].numpy(), expected)
    assert next_prepared['consecutive_untrusted_outputs'].item() == 1


def test_invalid_transition_writes_position_but_resets_velocity():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    out = output_for(batch)
    out.prior.box[:, 0] += 10.
    record = builder.commit(out, batch, diagnostics=True)[0]
    state = builder.states[row['request'].state_key]
    assert record['trusted_position_updated'] and not state.trusted_velocity_valid
    assert not state.transitions[1]
    assert record['trusted_velocity_reset_reason'] == 'invalid_transition'
    with pytest.raises(RuntimeError, match='duplicate commit'):
        builder.commit(out, batch)


def test_new_T_can_accept_q0_without_replacing_legacy_previous_quality():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    record = builder.commit(output_for(batch, legacy=.1, trusted=.9), batch, diagnostics=True)[0]
    assert record['supported'] and record['commit_quality'] == pytest.approx(.9)
    assert builder.states[row['request'].state_key].previous_quality == pytest.approx(.1)


@pytest.mark.parametrize('arm', ['b1', 'full'])
def test_empty_q0_alias_is_propagation_but_b1_preserves_old_quality(arm):
    builder, raw = BatchBuilder(cfg(arm)), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    batch['point_valid'].zero_()
    out = output_for(batch, selected=1, legacy=.83)
    out.q0_is_fallback[:] = True
    out.candidates.valid[:, 0] = False
    if arm == 'b1':
        out.selected_quality[:] = .83  # 旧 q0 fallback 质量保留在 B1 状态输入中。
    record = builder.commit(out, batch, diagnostics=True)[0]
    state = builder.states[row['request'].state_key]
    assert record['propagation_only'] and not record['strong'] and not record['supported']
    assert state.last_supported_time == state.initialization_time
    assert state.previous_quality == pytest.approx(.83 if arm == 'b1' else 0.)


def test_full_rejects_observation_alias_before_commit():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    out = output_for(batch)
    out.q0_is_fallback[:] = True
    with pytest.raises(ValueError, match='propagation-alias q0'):
        builder.commit(out, batch)
    assert builder.states[row['request'].state_key].next_frame == 1


def test_bad_second_row_does_not_partially_commit_first_row():
    builder, raw = BatchBuilder(cfg()), RawEndpointDataset(TinySource())
    rows = [raw[request(1, track=track)] for track in (0, 1)]
    batch = builder.prepare(rows)
    builder.supervise_prior(batch, fake_prior(batch))
    out = output_for(batch)
    out.commit_decision.eligible[1] = False
    with pytest.raises(ValueError, match='eligibility disagrees'):
        builder.commit(out, batch)
    assert all(state.next_frame == 1 and 1 not in state.boxes for state in builder.states.values())


@pytest.mark.parametrize('arm', ['b1', 'full'])
@pytest.mark.parametrize('bad', ['prior_fewer_rows', 'prior_wrong_width', 'prior_nan_second',
    'prior_inf_second', 'probability_fewer_rows', 'probability_wrong_width', 'probability_wrong_time'])
def test_malformed_commit_input_is_atomic_and_retryable(arm, bad):
    builder, raw = BatchBuilder(cfg(arm)), RawEndpointDataset(TinySource())
    rows = [raw[request(1, track=track)] for track in (0, 1)]
    batch = builder.prepare(rows)
    builder.supervise_prior(batch, fake_prior(batch))
    pending = builder._pending
    before = {key: deepcopy(vars(state)) for key, state in builder.states.items()}
    good = output_for(batch)
    out = deepcopy(good)
    if bad == 'prior_fewer_rows':
        out.prior.box = out.prior.box[:1]
    elif bad == 'prior_wrong_width':
        out.prior.box = out.prior.box[:, :3]
    elif bad == 'prior_nan_second':
        out.prior.box[1, 0] = float('nan')
    elif bad == 'prior_inf_second':
        out.prior.box[1, 3] = float('inf')
    elif bad == 'probability_fewer_rows':
        out.observation.foreground_probability = out.observation.foreground_probability[:1]
    elif bad == 'probability_wrong_width':
        out.observation.foreground_probability = out.observation.foreground_probability[..., :-1]
    else:
        out.observation.foreground_probability = out.observation.foreground_probability[:, :-1]
    error = FloatingPointError if bad in ('prior_nan_second', 'prior_inf_second') else ValueError
    with pytest.raises(error, match='prior|probability'):
        builder.commit(out, batch)
    assert builder._pending is pending
    assert builder.states.keys() == before.keys()
    for key, state in builder.states.items():
        _assert_identical(vars(state), before[key])
    # 异常整批不消费 pending；同一事务仍可用完整合法输出提交。
    records = builder.commit(good, batch, diagnostics=True)
    assert len(records) == 2 and builder._pending is None
    assert all(state.next_frame == 2 and 1 in state.boxes for state in builder.states.values())


def test_b1_nonfinite_foreground_probability_remains_unsupported():
    builder, raw = BatchBuilder(cfg('b1')), RawEndpointDataset(TinySource())
    row, batch = next_batch(builder, raw, 1)
    out = fake_output(batch, quality=.9)
    out.observation.foreground_probability[:, -1] = float('nan')
    record = builder.commit(out, batch, diagnostics=True)[0]
    assert record['geometric_fg_count'] == 0 and not record['supported']
    assert builder.states[row['request'].state_key].next_frame == 2


def test_thirty_two_low_confidence_propagations_keep_selected_crop_and_age_without_hold():
    """长期传播不刷新测量，也不因连数/低分触发额外 hold 或停止规则。"""
    dt = .37
    builder = BatchBuilder(cfg())
    raw = RawEndpointDataset(TinySource(lengths=(40,), dt=dt))

    def prepare_frame(frame):
        row = raw[request(frame, branch=4, end=35)]
        batch = builder.prepare([row])
        builder.supervise_prior(batch, fake_prior(batch), training=False)
        return row, batch

    row, batch = prepare_frame(1)
    # 第一次可靠 q0 建立非零可信速度；此后全程只有传播输出。
    builder.commit(output_for(batch, delta=.2), batch)
    state = builder.states[row['request'].state_key]
    assert state.trusted_velocity_valid and np.linalg.norm(state.trusted_velocity) > 0.
    trusted = deepcopy(state.trusted)
    velocity = state.trusted_velocity.copy()
    supported_time = state.last_supported_time
    strong_time = state.last_strong_time
    velocity_time = state.last_velocity_update_time
    previous_box = state.boxes[1].copy()

    for streak, frame in enumerate(range(2, 34), 1):
        _, batch = prepare_frame(frame)
        np.testing.assert_allclose(batch['anchor_box'][0].numpy(), previous_box, atol=1e-6)
        assert batch['consecutive_propagations'].item() == streak - 1
        assert batch['consecutive_untrusted_outputs'].item() == streak - 1
        assert batch['supported_age'].item() == pytest.approx(streak * dt)
        assert batch['trusted_velocity_age'].item() == pytest.approx(streak * dt)
        selected = 1 if frame % 2 == 0 else 2
        out = output_for(batch, selected=selected, trusted=.01, legacy=.01)
        # 原 prior 与现有 trusted propagation 保持不同来源、略有不同位置。
        out.prior.box[:, 0] += .025
        out.candidates.boxes[:, 1] = out.prior.box
        out.output_utility = torch.full((1,), 1e-6)
        expected = out.accepted_box[0].numpy().astype(np.float64, copy=True)
        expected[:3] += batch['anchor_box'][0, :3].double().numpy()
        record = builder.commit(out, batch, diagnostics=True)[0]
        np.testing.assert_allclose(state.boxes[frame], expected, atol=1e-6)
        assert np.linalg.norm(state.boxes[frame][:3] - previous_box[:3]) > .1
        assert record['propagation_only'] and not record['strong'] and not record['supported']
        assert not record['trusted_position_updated'] and not record['trusted_velocity_updated']
        assert record['consecutive_propagations'] == streak
        assert record['consecutive_untrusted_outputs'] == streak
        assert record['weak_age'] == pytest.approx(streak * dt)
        assert record['supported_age'] == pytest.approx(streak * dt)
        assert record['trusted_velocity_age'] == pytest.approx(streak * dt)
        assert state.last_supported_time == supported_time
        assert state.last_strong_time == strong_time
        assert state.last_velocity_update_time == velocity_time
        assert state.trusted_velocity_valid and state.previous_quality == 0.
        _assert_identical(state.trusted, trusted)
        np.testing.assert_array_equal(state.trusted_velocity, velocity)
        previous_box = state.boxes[frame].copy()

    assert state.next_frame == 34 and len(builder.states) == 1
    _, following = prepare_frame(34)
    np.testing.assert_allclose(following['anchor_box'][0].numpy(), previous_box, atol=1e-6)
    assert following['consecutive_propagations'].item() == 32
    assert following['consecutive_untrusted_outputs'].item() == 32
    assert following['supported_age'].item() == pytest.approx(33 * dt)


@pytest.mark.parametrize('arm',['b1','full'])
def test_transaction_has_no_b2_storage_or_compute_fields(arm):
    builder=BatchBuilder(cfg(arm))
    row,batch=next_batch(builder,RawEndpointDataset(TinySource()),1)
    assert not any(key.startswith(('memory','extension')) for key in batch)
    assert not hasattr(builder.states[row['request'].state_key], 'memory')
    assert batch['acquisition_valid'].all()
    out=fake_output(batch) if arm=='b1' else output_for(batch)
    record=builder.commit(out,batch,diagnostics=True)[0]
    assert not any(key.startswith(('memory','extension')) for key in record)
