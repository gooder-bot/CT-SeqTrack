"""V41 KITTI 输入边界用例；本轮仅静态审阅，运行留给服务器。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datasets.kitti_v41 import (GEOMETRY, KITTIV41Dataset, build_v41_kitti_manifest,
                               kitti_input_metadata_identity, normal_train_point_view,
                               validate_v41_kitti_manifest)


def _frame(role='train'):
    points = np.asarray([[0., 0., 0.], [1., 0., 0.], [20., 0., 0.]], dtype=np.float32)
    return {'pc': SimpleNamespace(points=points.T, point_ids=np.asarray([10, 20, 30])),
            'v41_data_role': role,
            '_v41_normal_train_cache': dict(points=points[:2], ids=np.asarray([10, 20]),
                raw_indices=np.asarray([0, 1]), lower=np.full(3, -10.), upper=np.full(3, 10.))}


def test_three_roles_are_explicit_and_immutable():
    manifest = build_v41_kitti_manifest()
    assert set(manifest['roles']) == {'train', 'dev', 'test'}
    assert manifest['roles']['dev'] == ['0017', '0018']
    assert manifest['geometry'] == GEOMETRY
    validate_v41_kitti_manifest(manifest)
    manifest['roles']['test'] = ['0018', '0019']
    with pytest.raises(ValueError):
        validate_v41_kitti_manifest(manifest)


def test_cache_preserves_raw_ids_and_mapping():
    frame = _frame()
    view = normal_train_point_view(frame, [[0, 0, 0, 0]], [4, 2, 2])
    assert view['cache_used']
    assert np.array_equal(view['ids'], frame['pc'].point_ids[view['raw_indices']])
    assert np.array_equal(view['points'], frame['pc'].points.T[view['raw_indices']])
    assert frame['pc'].points.shape == (3, 3)


def test_any_backup_roi_outside_cache_falls_back_to_raw():
    view = normal_train_point_view(_frame(), [[0, 0, 0, 0], [9, 0, 0, 0]], [4, 2, 2])
    assert not view['cache_used']
    assert view['ids'].tolist() == [10, 20, 30]


@pytest.mark.parametrize('role', ['dev', 'test', 'auxiliary'])
def test_non_normal_train_never_uses_gt_cache(role):
    assert not normal_train_point_view(_frame(role), [[0, 0, 0, 0]], [4, 2, 2])['cache_used']
    assert not normal_train_point_view(_frame(), [[0, 0, 0, 0]], [4, 2, 2],
                                       auxiliary=True)['cache_used']


def _reader(tmp_path):
    dataset = KITTIV41Dataset.__new__(KITTIV41Dataset)
    dataset.data_root = Path(tmp_path)
    dataset.KITTI_velo = Path(tmp_path) / 'velodyne'
    dataset.ct_pointcloud_cache_bytes = 0
    return dataset


def test_only_known_missing_files_are_empty_not_one_fake_point(tmp_path):
    dataset = _reader(tmp_path)
    cloud = dataset._pointcloud_for_frame('0001', 177)
    assert cloud.points.shape == (3, 0)
    assert len(cloud.point_ids) == 0
    with pytest.raises(FileNotFoundError):
        dataset._pointcloud_for_frame('0001', 181)


def test_existing_empty_bin_is_legal_but_malformed_bin_fails(tmp_path):
    dataset = _reader(tmp_path)
    scene = dataset.KITTI_velo / '0000'
    scene.mkdir(parents=True)
    (scene / '000000.bin').write_bytes(b'')
    assert dataset._pointcloud_for_frame('0000', 0).points.shape == (3, 0)
    (scene / '000001.bin').write_bytes(b'bad')
    with pytest.raises(ValueError):
        dataset._pointcloud_for_frame('0000', 1)


def test_empty_cache_keeps_empty_observation():
    frame = _frame()
    frame['pc'].points = np.empty((3, 0), np.float32)
    frame['pc'].point_ids = np.empty(0, np.int64)
    cache = frame['_v41_normal_train_cache']
    cache.update(points=np.empty((0, 3), np.float32), ids=np.empty(0, np.int64),
                 raw_indices=np.empty(0, np.int64))
    view = normal_train_point_view(frame, [[0, 0, 0, 0]], [4, 2, 2])
    assert view['cache_used'] and len(view['points']) == 0


def test_label_or_calibration_change_updates_input_identity(tmp_path):
    for folder in ('label_02', 'calib'):
        (tmp_path / folder).mkdir()
        for scene in range(21):
            (tmp_path / folder / f'{scene:04d}.txt').write_text('original\n', encoding='utf-8')
    before = kitti_input_metadata_identity(tmp_path)
    (tmp_path / 'calib/0019.txt').write_text('changed calibration\n', encoding='utf-8')
    after = kitti_input_metadata_identity(tmp_path)
    assert before['content_sha256'] != after['content_sha256']
