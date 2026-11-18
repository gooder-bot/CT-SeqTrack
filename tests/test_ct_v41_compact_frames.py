"""KITTI endpoint IPC、按需回原云和数据角色的独立回归。"""
import os
import pickle

import numpy as np
import pytest
from pyquaternion import Quaternion

from datasets.data_classes import Box
from datasets.kitti_v41 import KITTIRawFrame, KITTIV41Dataset, normal_train_point_view
from models.ct_v41.config import normalize_config
from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset, build_loaders, inside_box
from tests.helpers.raw_tracking import TinySource


def compact_source(tmp_path, role):
    scene = tmp_path / '0000'
    scene.mkdir()
    cloud = np.zeros((1027, 4), dtype=np.float32)
    cloud[:, 0] = 20.
    cloud[[1, 5, 8], 0] = [-1., 0., 1.]
    cloud.tofile(scene / '000000.bin')
    source = KITTIV41Dataset.__new__(KITTIV41Dataset)
    source.KITTI_velo, source.ct_pointcloud_cache_bytes = tmp_path, 1024 * 1024
    source.input_metadata_sha256 = 'test-metadata'
    source.v41_data_role = role
    def metadata(track, index):
        center = np.asarray((0. if track == 0 else 20., 0., 0.))
        box = Box(center, [2., 4., 2.], Quaternion(axis=[0, 0, 1], radians=0))
        return {'3d_bbox': box, 'meta': {'track_id': track}, 'scene_id': '0000',
                'frame_id': index, 'timestamp': index * .1, 'v41_data_role': role}
    source.endpoint_metadata = metadata
    return source, cloud


def test_small_train_packet_and_raw_fallback_have_physical_ids(tmp_path, monkeypatch):
    source, raw = compact_source(tmp_path, 'train')
    with monkeypatch.context() as patch:
        def no_full_row_ids(*args):
            pytest.fail('building GT10 allocated the full bin row IDs')
        patch.setattr(KITTIRawFrame, 'arrays', no_full_row_ids)
        first = source.endpoint_frames(0, [0])[0]
    assert 'pc' not in first
    assert len(pickle.dumps(first)) < raw.nbytes // 2
    frame = pickle.loads(pickle.dumps(first))
    with monkeypatch.context() as patch:
        def forbidden(*args):
            pytest.fail('contained ROI materialized its full raw cloud')
        patch.setattr(KITTIRawFrame, 'arrays', forbidden)
        view = normal_train_point_view(frame, [[0., 0., 0., 0.]], [4., 2., 2.])
        np.testing.assert_array_equal(view['ids'], [1, 5, 8])
        np.testing.assert_array_equal(view['points'], raw[view['raw_indices'], :3])
    far = normal_train_point_view(frame, [[0., 0., 0., 0.], [20., 0., 0., 0.]], [4., 2., 2.])
    assert not far['cache_used']
    np.testing.assert_array_equal(far['points'], raw[:, :3])
    np.testing.assert_array_equal(far['ids'], np.arange(len(raw)))
    other = source.endpoint_frames(1, [0])[0]
    assert 5 not in other['_v41_normal_train_cache']['ids']
    assert other['_v41_normal_train_cache']['raw_count'] == len(raw)


@pytest.mark.parametrize('role', ['auxiliary', 'dev', 'test'])
def test_closed_loop_worker_uses_only_raw_points(tmp_path, role):
    source, raw = compact_source(tmp_path, role)
    (frame,) = source.endpoint_frames(0, [0])
    assert '_v41_normal_train_cache' not in frame
    np.testing.assert_array_equal(frame['pc'].points.T, raw[:, :3])


def test_public_annotation_with_same_physical_id_has_its_own_view(tmp_path):
    source, _ = compact_source(tmp_path, 'train')
    def metadata(anno):
        frame = source.endpoint_metadata(anno['track_id'], anno['frame'])
        frame['3d_bbox'].center[0] = anno['x']
        return frame
    source._frame_metadata_from_anno = metadata
    near = source._get_frame_from_anno(dict(track_id=0, scene='0000', frame=0, x=0.))
    far = source._get_frame_from_anno(dict(track_id=0, scene='0000', frame=0, x=20.))
    np.testing.assert_array_equal(near['_v41_normal_train_cache']['ids'], [1, 5, 8])
    assert 5 not in far['_v41_normal_train_cache']['ids']
    assert 0 in far['_v41_normal_train_cache']['ids']


def test_train_view_keeps_strict_float64_aabb_boundaries(tmp_path):
    source, _ = compact_source(tmp_path, 'train')
    ref = KITTIRawFrame(tmp_path, '0000', 0, 0)
    frame = source.endpoint_metadata(0, 0)
    # 亚 float32 ULP 的移动可使一个恰在舍入边界的点由外变内。
    frame['3d_bbox'].center[0] += 1e-8
    raw = np.zeros((4, 4), np.float32)
    raw[:, 0] = [-12., np.nextafter(np.float32(-12.), np.float32(0.)), 12.,
                 np.nextafter(np.float32(12.), np.float32(20.))]
    raw.tofile(tmp_path / '0000' / '000000.bin')
    view = source._train_view(frame, ref)
    expected = np.flatnonzero(((raw[:, :3] > view['lower']) & (raw[:, :3] < view['upper'])).all(-1))
    np.testing.assert_array_equal(view['ids'], expected)
    np.testing.assert_array_equal(expected, [1, 2])


def test_first_size_is_metadata_only_and_batch_reuses_annotations():
    class Reads(TinySource):
        def get_frame_metadata(self, tracklet, frame):
            self.meta_reads.append(frame)
            return TinySource.get_frames(self, tracklet, [frame])[0]

        def get_frames(self, tracklet, frames):
            self.raw_reads.extend(frames)
            return super().get_frames(tracklet, frames)

    source = Reads((12,))
    source.raw_reads, source.meta_reads = [], []
    dataset = RawEndpointDataset(source)
    source.meta_reads.clear()
    start = EndpointRequest(1, 0, 1, 5, 10, 5)
    another_branch = EndpointRequest(1, 0, 2, 5, 10, 5)
    first, second = dataset.__getitems__([start, another_branch])
    assert first['frames'][5] is second['frames'][5]
    assert source.raw_reads == [2, 3, 4, 5]
    assert source.meta_reads == [0]
    assert 0 not in first['frames'] and first['first_frame']['timestamp'] == 100.
    source.meta_reads.clear()
    successor = dataset[EndpointRequest(1, 0, 1, 5, 10, 6)]
    assert not source.meta_reads and 'first_frame' not in successor


class WorkerSource(TinySource):
    def get_frames(self, tracklet, frame_ids):
        rows = super().get_frames(tracklet, frame_ids)
        for row in rows:
            row['reader_pid'] = os.getpid()
        return rows


def test_persistent_worker_obeys_new_epoch_requests():
    config = normalize_config(dict(ct_engineering_check=True, v41_attention_backend='math',
                                   batch_size=2, workers=1, v32_reserve_windows=0))
    loader = build_loaders(config, roles=('train',), sources={'train': WorkerSource((5,))})['train']
    pids = set()
    for epoch in (0, 1):
        loader.batch_sampler.set_epoch(epoch)
        expected = list(loader.batch_sampler)
        observed = []
        for rows in loader:
            observed.append(tuple(row['request'] for row in rows))
            pids.update(row['frames'][row['request'].frame]['reader_pid'] for row in rows)
        assert observed == expected
        assert all(r.epoch == epoch for requests in observed for r in requests)
    assert len(pids) == 1 and os.getpid() not in pids


@pytest.mark.parametrize('branch,detailed,label_slots', [(0, False, 4), (5, False, 1),
                                                          (4, True, 1), (4, False, 0)])
def test_only_consumed_point_labels_are_prepared(branch, detailed, label_slots):
    source = RawEndpointDataset(TinySource((5,)), training=True, enable_m=True)
    config = normalize_config(dict(ct_engineering_check=True, v41_attention_backend='math'))
    builder = BatchBuilder(config)
    builder.detailed_diagnostics = detailed
    batch = builder.prepare([source[EndpointRequest(0, 0, branch, 1, 4, 1)]])
    if label_slots:
        assert batch['segmentation_labels'].shape[1] == label_slots
    else:
        assert 'segmentation_labels' not in batch
    assert ('bc_targets' in batch) == (branch == 0)
    assert ('history_target_boxes' in batch) == (branch == 0)
    assert not any(key.startswith('acquisition_') for key in batch)


@pytest.mark.parametrize('point_dtype,box_dtype', [(np.float32, np.float64), (np.float64, np.float32),
                                                  (np.float32, np.float32), (np.int16, np.int16)])
def test_box_membership_matches_stacked_xyz_at_boundaries(point_dtype, box_dtype):
    edge = np.float32(1.)
    points = np.array([[edge, 0, 0], [np.nextafter(edge, np.float32(2.)), 0, 0],
                       [0, -1, 0], [0, 0, 1], [.4, .7, -1.1]], dtype=point_dtype)
    for yaw in (0., .71):
        box = np.array([0., 0., 0., yaw], dtype=box_dtype)
        size = np.array([2., 2., 2.], dtype=np.float64)
        delta = points - box[:3]
        c, s = np.cos(box[3]), np.sin(box[3])
        local = np.stack((c * delta[:, 0] + s * delta[:, 1],
                          -s * delta[:, 0] + c * delta[:, 1], delta[:, 2]), -1)
        expected = (np.abs(local) <= size * .5).all(-1)
        np.testing.assert_array_equal(inside_box(points, box, size), expected)
