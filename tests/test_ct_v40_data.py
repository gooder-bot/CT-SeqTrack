"""V40 搜索/可信状态及 worker-only 多时域标签；服务器检查时执行。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v40.config import normalize_config
from models.ct_v40.data import BatchBuilder, RawEndpointDataset, EndpointRequest, inside_box
from models.ct_v40.search import make_search_plan, sample_search_points
from tests.helpers.raw_tracking import TinySource, fake_output, fake_prior


def cfg(arm='full'):
    return normalize_config(dict(v40_arm=arm, ct_engineering_check=True,
        batch_size=4, workers=0, v32_reserve_windows=0, epoch=11))


def request(frame, *, branch=4, end=9):
    return EndpointRequest(9, 0, branch, 1, end, frame)


class MetadataSource(TinySource):
    """测试中 metadata 实现刻意不调用 get_frames，检测未来点云回退。"""
    def __init__(self, lengths=(10,), dt=.5):
        super().__init__(lengths, dt)
        self.metadata_reads, self.point_reads = [], []
        self.missing, self.broken = set(), set()

    def get_frame_metadata(self, index, frame):
        self.metadata_reads.append(frame)
        if frame in self.broken:
            raise OSError('annotation I/O failure')
        if frame in self.missing:
            raise KeyError('missing annotation')
        return dict(timestamp=100. + frame * self.dt, scene_id='tiny_scene',
            box_size=np.asarray([4., 2., 2.]),
            **{'3d_bbox': np.asarray([frame * .2, index * 20., 0., .1])})

    def get_frames(self, index, frame_ids):
        self.point_reads.extend(frame_ids)
        return super().get_frames(index, frame_ids)


def output_for(batch, *, selected=0, trusted=.9, delta=0.):
    from models.ct_v40.verification import current_q0_support
    out = fake_output(batch, quality=.8)
    out.observation.point_features = batch['points'].new_zeros((*batch['point_valid'].shape, 64))
    boxes = batch['fallback_box'][:, None].repeat(1, 3, 1)
    boxes[:, 0, 0] += delta
    out.candidates = SimpleNamespace(boxes=boxes, valid=torch.ones(len(boxes), 3, dtype=torch.bool))
    out.q0_is_fallback = torch.zeros(len(boxes), dtype=torch.bool)
    out.selected_index.fill_(selected)
    out.accepted_box = boxes[:, selected]
    out.legacy_q0_quality = torch.full((len(boxes),), .8)
    out.selected_quality = out.legacy_q0_quality.clone() if selected == 0 else torch.zeros(len(boxes))
    support = current_q0_support(out.observation, batch, boxes[:, 0])['mask'] & (selected == 0)
    score = torch.full((len(boxes),), trusted if selected == 0 else 0.)
    out.commit_decision = SimpleNamespace(trusted_score=score, support_mask=support,
        eligible=(support.sum(-1) >= 3) & (score >= .5), source=out.selected_index.clone())
    return out


def plan(previous=False, *, anchor=(0., 0., 0., 0.), backup=(10., 0., 0., 0.),
         velocity=(0., 0., 0.), velocity_valid=False, now=2., trusted_time=0.):
    return make_search_plan(anchor=anchor, trusted_box=backup, trusted_time=trusted_time,
        velocity=velocity, velocity_valid=velocity_valid, timestamp=now,
        previous_measurement_eligible=previous, policy='dual_roi_guard_v1', max_age=1.5)


def test_search_cap_is_search_only_and_invalid_velocity_is_not_used():
    trusted = np.asarray([1., 2., 3., .2])
    velocity = np.asarray([2., 1., -1.])
    before = trusted.copy()
    p = plan(backup=trusted, velocity=velocity, velocity_valid=True, now=5.)
    np.testing.assert_array_equal(p.backup_box, before + np.r_[1.5 * velocity, 0.])
    np.testing.assert_array_equal(trusted, before)
    assert p.trusted_age == 5. and p.backup_elapsed == 1.5
    assert p.reason == 'accepted_untrusted' and (p.main_quota, p.backup_quota) == (768, 256)
    np.testing.assert_array_equal(plan(backup=trusted, velocity=velocity).backup_box, trusted)
    assert not plan(None).triggered and not plan(True).triggered
    assert plan(None).reason == 'initialization' and plan(True).reason == 'accepted_eligible'


@pytest.mark.parametrize('main_n,backup_n,expected', [(900, 600, (768, 256)),
    (900, 100, (900, 100)), (100, 1000, (100, 924)), (1000, 100, (924, 100))])
def test_dual_roi_quota_spillback_and_no_intervening_background(main_n, backup_n, expected):
    points = np.concatenate((np.zeros((main_n, 3)), np.tile([10., 0., 0.], (backup_n, 1)),
                             np.asarray([[5., 0., 0.]])))
    ids = np.arange(len(points))
    sampled = sample_search_points(points, ids, plan(), np.ones(3), scale=1., offset=0.)
    assert (sampled.sampled_main, sampled.sampled_backup) == expected
    assert len(sampled.indices) == min(main_n + backup_n, 1024)
    assert len(np.unique(ids[sampled.indices])) == len(sampled.indices)
    assert len(points) - 1 not in sampled.union_indices


def test_overlap_and_duplicate_ids_never_create_extra_support():
    points = np.zeros((20, 3))
    ids = np.repeat(np.arange(10), 2)
    sampled = sample_search_points(points, ids, plan(backup=(0., 0., 0., 0.)), np.ones(3))
    np.testing.assert_array_equal(sampled.indices, np.arange(0, 20, 2))
    assert sampled.backup_only_count == 0


def test_no_new_backup_preserves_original_sample_and_rng_exactly():
    points = np.random.default_rng(8).uniform(-.1, .1, (1300, 3))
    ids, seed = np.arange(len(points)), 42
    sampled = sample_search_points(points, ids, plan(backup=(0., 0., 0., 0.)), np.ones(3), seed=seed)
    original = np.flatnonzero(inside_box(points, np.zeros(4), np.ones(3), scale=1.25, offset=2.))
    expected = np.sort(np.random.default_rng(seed).choice(original, 1024, replace=False))
    np.testing.assert_array_equal(sampled.indices, expected)


def test_rejected_accepted_still_commits_but_next_search_can_see_backup():
    raw, builder = RawEndpointDataset(MetadataSource()), BatchBuilder(cfg())
    row = raw[request(1)]
    first = builder.prepare([row])
    state = builder.states[row['request'].state_key]
    assert state.previous_measurement_eligible is None
    assert not first['diagnostic_search_triggered'].item()
    builder.commit(output_for(first, trusted=.1, delta=20.), first)
    assert state.boxes[1][0] == pytest.approx(20.)
    assert state.previous_measurement_eligible is False
    second = builder.prepare([raw[request(2)]])
    assert second['anchor_box'][0, 0].item() == pytest.approx(20.)
    assert second['diagnostic_search_triggered'].item()
    assert second['diagnostic_search_sampled_backup'].item() >= 3
    assert (second['segmentation_labels'][0, -1] == 1).sum().item() >= 3
    assert second['diagnostic_search_main_target_count'].item() == 0
    assert second['diagnostic_search_union_target_count'].item() >= 3
    assert second['diagnostic_search_sampled_main_target_count'].item() == 0
    assert second['diagnostic_search_sampled_backup_target_count'].item() >= 3
    assert state.last_supported_time == 100.


def test_eligible_commit_disables_guard_and_propagation_never_grants_measurement():
    raw, builder = RawEndpointDataset(MetadataSource()), BatchBuilder(cfg())
    row = raw[request(1)]
    first = builder.prepare([row])
    builder.commit(output_for(first), first)
    state = builder.states[row['request'].state_key]
    assert state.previous_measurement_eligible is True
    second = builder.prepare([raw[request(2)]])
    assert not second['diagnostic_search_triggered'].item()
    builder.commit(output_for(second, selected=1), second)
    assert state.previous_measurement_eligible is False


def test_capped_search_does_not_change_fallback_or_acquisition_recovery(monkeypatch):
    import models.ct_v40.data as data
    raw, builder = RawEndpointDataset(MetadataSource()), BatchBuilder(cfg())
    row = raw[request(1)]
    first = builder.prepare([row])
    builder.commit(output_for(first, trusted=.1, delta=20.), first)
    state = builder.states[row['request'].state_key]
    state.trusted[-1] = (np.zeros(4), 98.)
    state.trusted_velocity = np.asarray([2., 0., 0.])
    state.trusted_velocity_valid = True
    old = deepcopy(state)
    second = builder.prepare([raw[request(2)]])
    context = builder._pending[0]
    assert context['search_plan'].backup_box[0] == 3.
    assert context['recovery'][0] == 6.
    assert second['fallback_box'][0, 0].item() == -14.
    assert state.last_supported_time == old.last_supported_time
    np.testing.assert_array_equal(state.trusted_velocity, old.trusted_velocity)
    captured = {}
    def target(points, ids, **kwargs):
        captured.update(kwargs)
        return dict(acquisition_target=np.zeros(2), acquisition_valid=True, acquisition_demand=False)
    monkeypatch.setattr(data, 'band_grid_target', target)
    builder.supervise_prior(second, fake_prior(second), training=True)
    assert captured['recovery_center'][0] == 6.
    np.testing.assert_array_equal(captured['b0_raw_ids'], context['b0_ids'])


def test_b0_does_not_compute_or_materialize_acquisition(monkeypatch):
    import models.ct_v40.data as data
    builder = BatchBuilder(cfg('b0'))
    batch = builder.prepare([RawEndpointDataset(MetadataSource())[request(1)]])
    def forbidden(*args, **kwargs):
        raise AssertionError('B0 must not compute acquisition labels')
    monkeypatch.setattr(data, 'band_grid_target', forbidden)
    builder.supervise_prior(batch, None, training=True)
    assert not any(key.startswith('acquisition_') for key in batch)


def test_b0_cannot_select_learned_prior_but_can_commit_empty_observation_fallback():
    builder = BatchBuilder(cfg('b0'))
    row = RawEndpointDataset(MetadataSource())[request(1)]
    batch = builder.prepare([row])
    out = fake_output(batch)
    out.selected_index.fill_(1)
    with pytest.raises(ValueError, match='selection'):
        builder.commit(out, batch)
    out.selected_index.fill_(2)
    out.q0_is_fallback = torch.ones(1, dtype=torch.bool)
    batch['point_valid'].zero_()
    record = builder.commit(out, batch, diagnostics=True)[0]
    assert record['commit_source'] == 'trusted_propagation'
    assert record['propagation_only'] and not record['measurement_eligible']


def test_future_labels_use_worker_metadata_only_and_leave_endpoint_coverage():
    source = MetadataSource(lengths=(6,))
    raw = RawEndpointDataset(source, training=True, enable_m=True)
    source.metadata_reads.clear(); source.point_reads.clear()
    row = raw[request(1, branch=3, end=6)]
    targets = row['future_supervision']
    np.testing.assert_allclose(targets['future_displacement'], [[.4, 0.], [.8, 0.]])
    np.testing.assert_allclose(targets['future_dt'], [1., 2.])
    assert targets['future_valid'].all()
    assert source.metadata_reads == [2, 4] and max(source.point_reads) == 1
    tail = raw[request(4, branch=3, end=6)]
    np.testing.assert_array_equal(tail['future_supervision']['future_valid'], [True, False])
    assert raw[request(5, branch=3, end=6)]['request'].frame == 5


def test_missing_future_masks_one_horizon_but_io_failure_is_not_swallowed():
    source = MetadataSource()
    raw = RawEndpointDataset(source, training=True, enable_m=True)
    source.missing.add(2)
    row = raw[request(1, branch=3)]
    np.testing.assert_array_equal(row['future_supervision']['future_valid'], [False, True])
    source.broken.add(4)
    with pytest.raises(OSError, match='I/O'):
        raw[request(1, branch=3)]


@pytest.mark.parametrize('training,enable_m,branch', [(False, True, 4), (True, False, 3), (True, True, 4)])
def test_evaluation_and_non_m_arms_do_not_fetch_future_labels(training, enable_m, branch):
    source = MetadataSource()
    raw = RawEndpointDataset(source, training=training, enable_m=enable_m)
    source.metadata_reads.clear(); source.point_reads.clear()
    row = raw[request(1, branch=branch)]
    assert 'future_supervision' not in row
    assert not source.metadata_reads and max(source.point_reads) == 1
