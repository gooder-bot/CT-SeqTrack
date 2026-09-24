"""可信状态与原始点计数：观测支持、记忆写入和速度生命周期。"""

from copy import deepcopy

import numpy as np
import pytest
import torch

from models.ct_v31.data import BatchBuilder, RawEndpointDataset, inside_box
from tests.test_ct_v31_runtime import TinySource, cfg, request, fake_prior, fake_output


def prepare(builder, raw, frame, *, start=3):
    rows = [raw[request(frame, branch=0, start=start, end=9)]]
    batch = builder.prepare(rows)
    builder.acquire(batch, fake_prior(batch), training=False)
    return batch, next(iter(builder.states.values()))


def observed_output(batch, *, quality=.9, foreground_count=None):
    output = fake_output(batch, quality=quality)
    output.accepted_box = batch['target_box'].clone()
    if foreground_count is not None:
        output.observation.foreground_probability.zero_()
        output.observation.foreground_probability[0, -1, :foreground_count] = .5
    return output


@pytest.mark.parametrize('count,quality,strong,supported', [
    (0, .9, False, False), (2, .9, False, False),
    (3, .49, True, False), (3, .5, True, True),
])
def test_geometric_foreground_and_quality_have_separate_state_roles(count, quality, strong, supported):
    builder, raw = BatchBuilder(cfg(v31_arm='b0')), RawEndpointDataset(TinySource())
    batch, state = prepare(builder, raw, 3)
    previous_time, previous_velocity = state.trusted[-1][1], state.trusted_velocity.copy()
    record, = builder.commit(observed_output(batch, quality=quality, foreground_count=count),
                             batch, diagnostics=True)
    assert record['geometric_fg_count'] == count
    assert record['strong'] is strong and record['supported'] is supported
    assert record['memory_write'] is supported
    assert state.last_strong_time == (101.5 if strong else 101.)
    assert state.last_supported_time == (101.5 if supported else 101.)
    assert state.trusted[-1][1] == (101.5 if supported else previous_time)
    np.testing.assert_allclose(state.trusted_velocity, previous_velocity, atol=1e-7)
    assert state.trusted_velocity_valid
    # 相邻 accepted 几何一致性保持旧定义，不额外改成前景或 quality gate。
    following = builder.prepare([raw[request(4, branch=0, start=3, end=9)]])
    assert following['history_pair_valid'][0, -1]


def test_supported_pose_does_not_depend_on_memory_write_success(monkeypatch):
    builder, raw = BatchBuilder(cfg(v31_arm='b0')), RawEndpointDataset(TinySource())
    batch, state = prepare(builder, raw, 3)
    monkeypatch.setattr(state.memory, 'update', lambda *args: False)
    record, = builder.commit(observed_output(batch), batch, diagnostics=True)
    assert record['strong'] and record['supported'] and not record['memory_write']
    assert state.trusted[-1][1] == 101.5 and state.last_supported_time == 101.5
    assert state.trusted_velocity_valid


def test_memory_excludes_outside_high_foreground_and_keeps_outside_background():
    raw = RawEndpointDataset(TinySource())
    row = raw[request(3, branch=0, start=3, end=9)]
    current = row['frames'][3]
    current['pc'][-2:] = current['3d_bbox'][:3] + np.asarray([[3., 0., 0.], [3.5, 0., 0.]])
    builder = BatchBuilder(cfg(v31_arm='b0'))
    batch = builder.prepare([row])
    builder.acquire(batch, fake_prior(batch), training=False)
    output = observed_output(batch)
    output.observation.foreground_probability[0, -1, batch['point_ids'][0, -1] == 5] = .1
    record, = builder.commit(output, batch, diagnostics=True)
    state = next(iter(builder.states.values()))
    saved = state.memory.recent[-1]
    assert record['geometric_fg_count'] == 4 and record['memory_fg_count'] == 4
    assert set(saved['ids'][saved['fg']]) == {0, 1, 2, 3}
    assert set(saved['ids'][~saved['fg']]) == {5}
    assert 4 not in saved['ids']


def test_repeated_identity_cannot_manufacture_strong_support():
    builder, raw = BatchBuilder(cfg(v31_arm='b0')), RawEndpointDataset(TinySource())
    batch, _ = prepare(builder, raw, 3)
    batch['point_valid'][0, -1] = False
    batch['point_valid'][0, -1, :3] = True
    batch['point_ids'][0, -1, :3] = 0
    record, = builder.commit(observed_output(batch), batch, diagnostics=True)
    assert record['geometric_fg_count'] == 1 and not record['strong']
    assert not record['supported'] and not record['memory_write']


def test_nonadjacent_new_trusted_anchor_discards_old_velocity_then_relearns_adjacent_velocity():
    builder, raw = BatchBuilder(cfg(v31_arm='b0')), RawEndpointDataset(TinySource())
    first, state = prepare(builder, raw, 3)
    velocity = state.trusted_velocity.copy()
    record, = builder.commit(observed_output(first, quality=.1), first, diagnostics=True)
    assert record['trusted_velocity_reset_reason'] == 'unsupported_retained'
    np.testing.assert_array_equal(state.trusted_velocity, velocity)
    assert state.trusted_velocity_valid
    second, state = prepare(builder, raw, 4)
    record, = builder.commit(observed_output(second), second, diagnostics=True)
    assert record['supported'] and record['trusted_velocity_reset_reason'] == 'nonadjacent_supported'
    assert not state.trusted_velocity_valid
    np.testing.assert_array_equal(state.trusted_velocity, np.zeros(3))
    third, state = prepare(builder, raw, 5)
    torch.testing.assert_close(third['fallback_box'][0, :3], torch.zeros(3), atol=1e-7, rtol=0.)
    assert not third['trusted_velocity_valid'][0]
    record, = builder.commit(observed_output(third), third, diagnostics=True)
    assert record['trusted_velocity_reset_reason'] == 'supported_adjacent'
    assert state.trusted_velocity_valid
    np.testing.assert_allclose(state.trusted_velocity, velocity, atol=1e-7)


def test_supported_relocalization_resets_velocity_but_keeps_accepted_transition_semantics():
    builder, raw = BatchBuilder(cfg(v31_arm='b0')), RawEndpointDataset(TinySource())
    batch, state = prepare(builder, raw, 3)
    output = observed_output(batch)
    output.prior.box[:, 0] -= 10
    record, = builder.commit(output, batch, diagnostics=True)
    assert record['supported'] and record['trusted_velocity_reset_reason'] == 'invalid_transition'
    assert state.trusted[-1][1] == 101.5 and not state.transitions[3]
    assert not state.trusted_velocity_valid
    np.testing.assert_array_equal(state.trusted_velocity, np.zeros(3))
    following, _ = prepare(builder, raw, 4)
    assert not following['history_pair_valid'][0, -1]
    builder.commit(observed_output(following, foreground_count=0), following)
    assert not state.trusted_velocity_valid
    np.testing.assert_array_equal(state.trusted_velocity, np.zeros(3))


def test_sparse_seed_preserves_legal_pose_velocity_without_inventing_strong_observations():
    raw = RawEndpointDataset(TinySource())
    row = raw[request(4, branch=0, start=4, end=9)]
    for frame in (1, 2, 3):
        row['frames'][frame]['pc'] = np.empty((0, 3))
        row['frames'][frame]['point_ids'] = np.empty(0, dtype=np.int64)
    builder = BatchBuilder(cfg(v31_arm='b0'))
    batch = builder.prepare([row])
    state = next(iter(builder.states.values()))
    assert state.last_strong_time is None
    assert state.initialization_time == 101.5 and state.last_supported_time == 101.5
    assert batch['weak_age'].item() == .5 and batch['supported_age'].item() == .5
    assert state.trusted_velocity_valid
    np.testing.assert_allclose(state.trusted_velocity, [.4, 0., 0.])
    builder.acquire(batch, fake_prior(batch), training=False)
    builder.commit(observed_output(batch, foreground_count=0), batch)
    following, _ = prepare(builder, raw, 5, start=4)
    assert following['weak_age'].item() == 1.


def test_seed_strong_timestamp_uses_actual_input_box_and_latest_observed_seed():
    raw = RawEndpointDataset(TinySource())
    row = raw[request(4, branch=1, start=4, end=9)]
    builder = BatchBuilder(cfg(v31_arm='b0'))
    input_box = builder._new_state(row).boxes[3]
    current = row['frames'][3]
    # 这些点仍在合法 GT 内，却落在实际带共同偏移的输入 seed 框外。
    x, y = np.meshgrid(np.linspace(-1.99, 1.99, 21), np.linspace(-.99, .99, 11))
    candidates = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)] + current['3d_bbox'][:3]
    boundary = (inside_box(candidates, current['3d_bbox'], current['box_size'])
                & ~inside_box(candidates, input_box, current['box_size']))
    assert boundary.sum() >= 3
    current['pc'] = candidates[boundary][:3]
    current['point_ids'] = np.arange(3)
    batch = builder.prepare([row])
    state = next(iter(builder.states.values()))
    assert state.last_strong_time == 101.
    assert batch['weak_age'].item() == 1.


def test_single_seed_has_zero_velocity_without_claiming_a_measured_pair():
    builder, raw = BatchBuilder(cfg(v31_arm='b0')), RawEndpointDataset(TinySource())
    batch, state = prepare(builder, raw, 1, start=1)
    assert not state.trusted_velocity_valid and not batch['trusted_velocity_valid'][0]
    np.testing.assert_array_equal(state.trusted_velocity, np.zeros(3))


def test_eval_gt_counts_are_readonly_and_distinguish_raw_crop_and_sampled_points():
    raw = RawEndpointDataset(TinySource())
    row = raw[request(3, branch=0, start=3, end=9)]
    builder = BatchBuilder(cfg(v31_arm='b0', point_sample_size=3))
    batch = builder.prepare([row])
    before = {key: value.clone() for key, value in batch.items()}
    builder.acquire(batch, fake_prior(batch), training=False)
    assert batch['diagnostic_raw_point_count'].item() == 6
    assert batch['diagnostic_target_count'].item() == 4
    assert batch['diagnostic_crop_point_count'].item() == 4
    assert batch['diagnostic_crop_target_count'].item() == 4
    assert batch['diagnostic_sampled_point_count'].item() == 3
    assert batch['diagnostic_sampled_target_count'].item() == 3
    assert batch['diagnostic_gt_xy_displacement'].item() == pytest.approx(.2)
    for key, value in before.items():
        assert torch.equal(batch[key], value), key
    alternate = deepcopy(row)
    alternate['frames'][3]['3d_bbox'][:3] += 100.
    other_builder = BatchBuilder(cfg(v31_arm='b0', point_sample_size=3))
    other = other_builder.prepare([alternate])
    other_builder.acquire(other, fake_prior(other), training=False)
    assert other['diagnostic_target_count'].item() == 0
    for key in ('points', 'point_valid', 'point_ids', 'history_boxes', 'history_pair_valid',
                'fallback_box', 'box_size', 'extension_points', 'extension_valid', 'extension_ids'):
        assert torch.equal(batch[key], other[key]), key
