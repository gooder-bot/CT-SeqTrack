"""真实 KITTI bin、可 spawn 的适配器和训练首帧准备的事务回归。"""
import pickle

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from datasets.data_classes import Box
from datasets.kitti_v41 import KITTIRawFrame, KITTIV41Dataset, normal_train_point_view
from models.ct_v41.config import normalize_config
from models.ct_v41.data import (BatchBuilder, EndpointRequest, RawEndpointDataset, build_loaders,
                                full_epoch_diagnostics, point_arrays)
from tests.helpers.raw_tracking import fake_output


class WorkerKITTI(KITTIV41Dataset):
    def __init__(self, root, role='train'):
        self.KITTI_velo = root
        self.v41_data_role = role
        self.ct_pointcloud_cache_bytes = 4 * 1024 ** 2
        self.input_metadata_sha256 = 'worker-transport-test'

    def get_num_tracklets(self):
        return 1

    def get_num_frames_tracklet(self, track):
        return 6

    def get_tracklet_key(self, track):
        return str(track)

    def get_endpoint_key(self, track, index):
        return f'{track}:{index}'

    def endpoint_metadata(self, track, index):
        center = np.array((index * .03, 0., 0.))
        return {'3d_bbox': Box(center, [2., 4., 2.], Quaternion(axis=[0, 0, 1], radians=.05)),
                'meta': {'track_id': track}, 'scene_id': '0000', 'frame_id': index,
                'timestamp': index * .1, 'input_metadata_sha256': self.input_metadata_sha256,
                'v41_data_role': self.v41_data_role}


@pytest.fixture
def kitti(tmp_path):
    scene = tmp_path / '0000'
    scene.mkdir()
    rng = np.random.default_rng(713)
    for index in range(6):
        raw = rng.uniform(-2, 2, (1600, 4)).astype(np.float32)
        raw[1::5, 0] += 24.
        raw.tofile(scene / f'{index:06}.bin')
    return WorkerKITTI(tmp_path)


def config(workers=1):
    return normalize_config(dict(ct_engineering_check=True, v41_training_recipe='b1',
                                 v41_attention_backend='math', batch_size=2, workers=workers,
                                 v32_reserve_windows=0, full_diagnostic_epochs=[39]))


def assert_batches_equal(a, b):
    assert a.keys() == b.keys()
    for field in a:
        assert torch.equal(a[field], b[field]), field


def test_spawn_first_rows_continuations_and_full_epoch(kitti):
    cfg = config()
    loader = build_loaders(cfg, roles=('train',), sources={'train': kitti})['train']
    loader.multiprocessing_context = 'spawn'
    raw = RawEndpointDataset(kitti, training=True)
    live, reference = BatchBuilder(cfg), BatchBuilder(cfg)
    with torch.no_grad():
        for epoch in (0, 37, 38):
            live.detailed_diagnostics = reference.detailed_diagnostics = full_epoch_diagnostics(cfg, epoch + 1)
            loader.batch_sampler.set_epoch(epoch)
            plan = list(loader.batch_sampler)
            seen = []
            for rows in loader:
                reqs = tuple(row['request'] for row in rows)
                seen.append(reqs)
                for row in rows:
                    r = row['request']
                    assert ('_prepared_start' in row) == (r.starts_window and epoch != 38)
                    if '_prepared_start' in row:
                        assert all('points' not in f['_v41_normal_train_cache'] for f in row['frames'].values())
                wanted = reference.prepare(raw.__getitems__(reqs))
                got = live.prepare(rows)
                assert_batches_equal(got, wanted)
                for model, batch in ((live, got), (reference, wanted)):
                    model.commit(fake_output(batch, delta=.15), batch, diagnostics=epoch == 38)
            assert seen == list(map(tuple, plan))
            assert not live.states and not live._history_pools


def test_prepared_start_retry_detailed_fallback_and_history_cache_miss(kitti, monkeypatch):
    cfg = config()
    loader = build_loaders(cfg, roles=('train',), sources={'train': kitti})['train']
    start = EndpointRequest(37, 0, 3, 3, 6, 3)
    step = EndpointRequest(37, 0, 3, 3, 6, 4)
    mixed_start = EndpointRequest(37, 0, 2, 4, 6, 4)
    old = RawEndpointDataset(kitti, training=True)
    row, = loader.collate_fn(loader.dataset.__getitems__([start]))
    model, ref = BatchBuilder(cfg), BatchBuilder(cfg)
    # 独立诊断工具可以用同一首行回读完整字段。
    assert_batches_equal(model.prepare([row]), ref.prepare([old[start]]))
    model.reset()
    ref.reset()
    model.detailed_diagnostics = ref.detailed_diagnostics = False
    with monkeypatch.context() as m:
        m.setattr(model, '_prepare_row', lambda *args: pytest.fail('materialized start twice'))
        a, b = model.prepare([row]), ref.prepare([old[start]])
        assert_batches_equal(a, b)
        bad = fake_output(a)
        bad.accepted_box[0, 0] = float('nan')
        with pytest.raises(FloatingPointError, match='accepted box'):
            model.commit(bad, a)
        assert not model.states and not model._history_pools
        model.reset()
        assert_batches_equal(a, model.prepare([row]))
        model.commit(fake_output(a), a)
        ref.commit(fake_output(b), b)
    # 同一标注帧在混合批中可同时为旧历史及首行输入。丢弃已提交池强制惰性回读。
    model._history_pools.clear()
    reqs = [step, mixed_start]
    rows = loader.collate_fn(loader.dataset.__getitems__(reqs))
    assert '_prepared_start' not in rows[0] and '_prepared_start' in rows[1]
    for index in (1, 2):
        assert 'points' not in rows[0]['frames'][index]['_v41_normal_train_cache']
    assert 'points' in rows[0]['frames'][3]['_v41_normal_train_cache']
    assert_batches_equal(model.prepare(rows), ref.prepare(old.__getitems__(reqs)))
    model.reset()
    assert_batches_equal(model.prepare([row]), a)
    assert row['_prepared_start'][3]['state'].next_frame == start.frame


@pytest.mark.parametrize('role', ['auxiliary', 'dev', 'test'])
def test_closed_loop_wire_carries_only_physical_references(kitti, role, monkeypatch):
    source = WorkerKITTI(kitti.KITTI_velo, role)
    req = EndpointRequest(0, 0, 5 if role == 'auxiliary' else 4, 1, 6, 1)
    baseline = RawEndpointDataset(source)[req]
    with monkeypatch.context() as m:
        m.setattr(KITTIRawFrame, 'arrays', lambda *a: pytest.fail('worker loaded a whole cloud'))
        compact = RawEndpointDataset(source, worker_transport=True)[req]
    assert len(pickle.dumps(compact)) < len(pickle.dumps(baseline)) // 4
    for index, f in compact['frames'].items():
        assert 'pc' not in f and '_v41_normal_train_cache' not in f
        got = point_arrays(f)
        want = point_arrays(baseline['frames'][index])
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)


def test_lazy_gt10_retains_exact_raw_fallback(kitti):
    ref = kitti.endpoint_frames(0, [1])[0]
    lazy = kitti.endpoint_frames(0, [1], worker_required=set())[0]
    for box, contained in [([0., 0., 0., .1], True), ([24., 0., 0., .1], False)]:
        a, b = [normal_train_point_view(f, [box], [4., 2., 2.]) for f in (ref, lazy)]
        assert a['cache_used'] == b['cache_used'] == contained
        for field in ('points', 'ids', 'raw_indices'):
            np.testing.assert_array_equal(a[field], b[field])


def test_training_loader_with_dev_source_keeps_parent_preparation(kitti):
    source = WorkerKITTI(kitti.KITTI_velo, 'dev')
    cfg = config()
    loader = build_loaders(cfg, roles=('train',), sources={'train': source})['train']
    reqs = next(iter(loader.batch_sampler))
    rows = loader.collate_fn(loader.dataset.__getitems__(reqs))
    assert all('_prepared_start' not in row for row in rows)
    expected = RawEndpointDataset(source, training=True).__getitems__(reqs)
    assert_batches_equal(BatchBuilder(cfg).prepare(rows), BatchBuilder(cfg).prepare(expected))


def test_worker_sends_only_used_slot_prefix_and_restores_mixed_batches(kitti):
    cfg = config()
    for index, count in enumerate((0, 0, 13, 1200, 5, 7)):
        raw = np.zeros((count, 4), dtype=np.float32)
        raw[:, 0] = np.linspace(-1., 1., count)
        raw.tofile(kitti.KITTI_velo / '0000' / f'{index:06}.bin')
    loader = build_loaders(cfg, roles=('train',), sources={'train': kitti})['train']
    raw = RawEndpointDataset(kitti, training=True)
    main, reference = BatchBuilder(cfg), BatchBuilder(cfg)
    main.detailed_diagnostics = reference.detailed_diagnostics = False
    for start, width in ((1, 0), (2, 13), (3, cfg.point_sample_size)):
        reqs = [EndpointRequest(37, 0, start, start, 5, start)]
        if start > 1:
            reqs.append(EndpointRequest(37, 0, start-1, start-1, 5, start))
        rows = pickle.loads(pickle.dumps(loader.collate_fn(loader.dataset.__getitems__(reqs))))
        fields = rows[0]['_prepared_start'][2]
        for key in ('points', 'point_ids', 'point_valid', 'segmentation_labels', 'bc_targets'):
            assert fields[key].shape[1] == width
        if len(rows) > 1:
            assert '_prepared_start' not in rows[1]
        a, b = main.prepare(rows), reference.prepare(raw.__getitems__(reqs))
        assert_batches_equal(a, b)
        assert main.commit(fake_output(a), a, diagnostics=True) == reference.commit(
            fake_output(b), b, diagnostics=True)
