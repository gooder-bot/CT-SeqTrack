import pickle

import numpy as np
import pytest
import torch

from models.ct_v41.config import normalize_config
from models.ct_v41.data import BatchBuilder, EndpointRequest
from tests.helpers.raw_tracking import fake_output
from tests.test_ct_v41_compact_frames import compact_source


def make_rows(tmp_path, branch):
    role = 'train' if branch < 4 else ('dev' if branch == 4 else 'auxiliary')
    source, _ = compact_source(tmp_path, role)
    rng = np.random.default_rng(21)
    xyz = rng.uniform(-1., 1., size=(3800, 4)).astype(np.float32)
    xyz[1::3, 0] += 25.
    for t in range(9):
        xyz[:, 1] += .01
        xyz.tofile(tmp_path / '0000' / f'{t:06d}.bin')
    start = 3 if branch < 4 else 1
    rows = []
    for t in range(start, 9):
        request = EndpointRequest(9, 0, branch, start, 9, t)
        history = tuple(max(0, i) for i in range(t - 3, t))
        ids = sorted(set(history + (t,)))
        frames = dict(zip(ids, source.endpoint_frames(0, ids)))
        row = dict(request=request, history_ids=history, frames=frames, tracklet_key='track')
        if request.starts_window:
            row['first_frame'] = frames[0] if 0 in frames else source.endpoint_metadata(0, 0)
        rows.append(row)
    return rows


def builder():
    return BatchBuilder(normalize_config(dict(v41_training_recipe='b1', ct_engineering_check=True,
                                               v41_attention_backend='math')))


@pytest.mark.parametrize('branch,detail', [(3, False), (3, True), (5, False), (4, True)])
def test_history_pool_matches_fresh_crop_through_motion_and_gt10_fallback(tmp_path, branch, detail):
    rows = make_rows(tmp_path, branch)
    cached, fresh = builder(), builder()
    cached.detailed_diagnostics = fresh.detailed_diagnostics = detail
    cold = fresh._history_crop
    fresh._history_crop = lambda f, i, b, s, sc, off, train, prev, keep: cold(f, i, b, s, sc, off, train, {}, None)
    observed, flags = [], set()
    for step, row in enumerate(rows):
        a, b = cached.prepare([row]), fresh.prepare([row])
        assert set(a) == set(b)
        for key in a:
            assert torch.equal(a[key], b[key]), (step, key)
        if detail and branch == 3:
            flags.update(a['diagnostic_train_cache_used'][0].tolist())
        # 连续端点重复使用同一历史帧，但以不同 seed 独立抽取 1024 个点。
        if row['request'].frame in (3, 4):
            slot = 2 - (row['request'].frame - 3)
            observed.append(a['point_ids'][0, slot].clone())
        for model, batch in ((cached, a), (fresh, b)):
            output = fake_output(batch, delta=25. if step == 0 else .2)
            model.commit(output, batch)
        if not row['request'].ends_window:
            pools = cached._history_pools[row['request'].state_key]
            assert len(pools) <= 2  # 只有当前历史的两个较新帧会再次出现。
            assert all(len(pool.xyz) == len(pool.ids) for pool in pools.values())
            assert {'_history_pools', '_next_history_pools'}.isdisjoint(cached.__getstate__())
            cloned = pickle.loads(pickle.dumps(cached))
            assert not cloned._history_pools and not cloned._next_history_pools
            assert vars(cached.states[row['request'].state_key]).keys() == vars(fresh.states[row['request'].state_key]).keys()
    if branch == 3:
        assert not torch.equal(*observed)
        if detail:
            assert flags == {True, False}
    assert not cached.states and not cached._history_pools and not cached._next_history_pools


def test_failed_commit_does_not_publish_historical_points(tmp_path):
    rows, model = make_rows(tmp_path, 3), builder()
    first = model.prepare([rows[0]])
    model.commit(fake_output(first), first)
    key = rows[0]['request'].state_key
    saved = model._history_pools[key]
    second = model.prepare([rows[1]])
    assert model._next_history_pools[key]
    failed = fake_output(second)
    failed.accepted_box[0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='accepted box'):
        model.commit(failed, second)
    assert model._history_pools[key] is saved and not model._next_history_pools
    model._pending = None  # host 的异常路径只保留已提交的前驱
    retry = model.prepare([rows[1]])
    for field in second:
        assert torch.equal(second[field], retry[field])
    model.reset()
    assert not model.states and not model._history_pools and not model._next_history_pools


def test_changed_historical_box_cannot_hit_a_stale_crop(tmp_path):
    rows, model = make_rows(tmp_path, 3), builder()
    first = model.prepare([rows[0]])
    model.commit(fake_output(first), first)
    key = rows[0]['request'].state_key
    near = model._history_pools[key][2]
    assert near.gt10 and not np.any(near.ids % 3 == 1)
    model.states[key].boxes[2][0] += 25.
    model.prepare([rows[1]])
    far = model._next_history_pools[key][2]
    assert not far.gt10 and np.all(far.ids % 3 == 1)
