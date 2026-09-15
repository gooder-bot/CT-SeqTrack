"""v30 实际数据接口：标定、缺测、身份、尾部覆盖与整云缓存隔离。"""
import copy
import importlib
import sys
import types

import numpy as np
import pytest
from pyquaternion import Quaternion

from tests.test_ct_v27_input_flow import sampler_runtime
from tests.test_ct_v28_preflight_data_entry import metadata_dataset_entry
from utils.data_cache_v30 import PointCloudArrayLRU, process_pointcloud_cache
from utils.dataset_protocol_v30 import (
    build_dataset_manifest, select_dataset_protocol, validate_dataset_manifest,
)
from utils.recursive_state import OnlineRecursiveBatchSampler


def test_cache_eviction_and_mutation_are_bounded_and_independent(monkeypatch):
    cache = PointCloudArrayLRU(80)
    xyz = np.arange(6, dtype=np.float32).reshape(3, 2)
    ids = np.asarray([8, 2], dtype=np.int64)
    cache.put('a', xyz, ids)  # 40 bytes
    xyz[:] = -7
    hit, hit_ids = cache.get('a')
    np.testing.assert_array_equal(hit, np.arange(6, dtype=np.float32).reshape(3, 2))
    hit[:] = -9
    hit_ids[:] = -3
    assert cache.get('a')[0][0, 0] == 0 and cache.get('a')[1][0] == 8
    cache.put('b', xyz, ids)
    cache.get('a')
    cache.put('c', xyz, ids)
    assert cache.get('b') is None and cache.bytes_used == 80
    cache.put('oversize', np.zeros((3, 30), np.float64), np.arange(30))
    assert cache.get('oversize') is None and cache.bytes_used == 80
    zero = PointCloudArrayLRU(0)
    zero.put('a', xyz, ids)
    assert zero.get('a') is None
    process = process_pointcloud_cache(80)
    process.put('a', xyz, ids)
    import utils.data_cache_v30 as module
    monkeypatch.setattr(module.os, 'getpid', lambda: -1234)
    assert len(process_pointcloud_cache(80)) == 0  # a fork must not inherit warm clouds


def test_manifest_roles_validation_and_nuscenes_legacy_scene_selection():
    kitti = dict(dataset='kitti_mf', version='kitti_tracking', ct_frame_stride=1)
    manifest = build_dataset_manifest(kitti)
    assert manifest['scenes']['train'] == [f'{i:04d}' for i in range(17)]
    assert select_dataset_protocol(kitti, 'val')[1:] == ('dev', ['0018'])
    assert select_dataset_protocol(kitti, 'eval')[1:] == ('test', ['0019', '0020'])
    altered = copy.deepcopy(manifest)
    altered['scenes']['calibration'] = ['0000']
    with pytest.raises(ValueError, match='sha256'):
        validate_dataset_manifest(altered)
    assert build_dataset_manifest({**kitti, 'ct_frame_stride': 2})['content_sha256'] != manifest['content_sha256']
    for key, value in [('kitti_frame_period', .5), ('ct_coordinate_mode', 'global'), ('ct_frame_stride', 1.5)]:
        with pytest.raises(ValueError):
            build_dataset_manifest({**kitti, key: value})
    splits = dict(mini_train=[f's{i}' for i in range(8)], mini_val=['v0', 'v1'])
    nu = build_dataset_manifest(dict(dataset='nuscenes_mf', version='v1.0-mini'), splits)
    from utils.v28_protocol import build_scene_manifest
    assert nu['scenes'] == build_scene_manifest(splits, 'v1.0-mini')['scenes']
    assert nu['parameter_training_overlap'] is True


def test_v30_factory_precedes_legacy_and_kitti_does_not_import_splits(metadata_dataset_entry, monkeypatch):
    from pathlib import Path
    from utils.config import load_yaml_config
    _, calls = metadata_dataset_entry
    package = sys.modules['datasets']
    package.kitti_mf.KITTIMFDataset = package.nuscenes_lidar_mf.NuScenesMFDataset
    cfg = types.SimpleNamespace(**load_yaml_config(Path(__file__).resolve().parents[1] /
        'cfgs/ct_seqtrack/29_full_cfc_nuscenes_full.yaml'))
    cfg.ct_enable_v30 = True
    cfg.preloading = False
    package.get_dataset(cfg, type='train_motion_mf', protocol_role='train')
    assert calls[-1]['ct_scene_manifest']['schema'] == 'ct_seqtrack.dataset_protocol.v30'
    assert calls[-1]['preload_offset'] == -1 and calls[-1]['ct_pointcloud_cache_bytes'] == 268435456
    cfg.dataset, cfg.version, cfg.ct_coordinate_mode = 'kitti_mf', 'kitti_tracking', 'sensor_relative'
    monkeypatch.setattr(sys.modules['nuscenes.utils.splits'], 'create_splits_scenes',
                        lambda: pytest.fail('KITTI attempted to build a nuScenes manifest'))
    for role, expected in [('train', [f'{i:04d}' for i in range(17)]),
                           ('calibration', ['0017']), ('val', ['0018']), ('test', ['0019', '0020'])]:
        package.get_dataset(cfg, type='test', protocol_role=role)
        assert calls[-1]['ct_scene_names'] == expected
        assert calls[-1]['frame_period'] == .1 and calls[-1]['preload_offset'] == -1


def _make_kitti(root, frames=(0, 1, 2, 3, 4), scenes=('0000',)):
    for directory in ('velodyne', 'label_02', 'calib'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    # Nontrivial rotations expose omission of rectification or point/box mismatch.
    velo_to_camera = Quaternion(axis=[0, 0, 1], radians=.12).rotation_matrix @ np.asarray(
        [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    rect = Quaternion(axis=[1, 0, 0], radians=.07).rotation_matrix
    translation = np.asarray([.2, -.3, .4])
    transform = np.column_stack((velo_to_camera, translation))
    cloud = np.asarray([[2., .1, -.3, .7], [3., -2., .2, .9], [150., 70., 8., 1.]], np.float32)
    for scene in scenes:
        (root / 'velodyne' / scene).mkdir()
        (root / 'calib' / (scene + '.txt')).write_text(
            'R_rect: ' + ' '.join(map(str, rect.ravel())) + '\nTr_velo_cam: ' +
            ' '.join(map(str, transform.ravel())) + '\n', encoding='utf-8')
        (root / 'label_02' / (scene + '.txt')).write_text('\n'.join(
            f'{frame} 0 Car 0 0 0 0 0 10 10 2 2 4 1 2 {10+frame} 0.35'
            for frame in frames), encoding='utf-8')
        for frame in frames:
            cloud.tofile(root / 'velodyne' / scene / f'{frame:06d}.bin')
    return rect, transform, cloud


def test_kitti_full_calibration_box_cloud_roundtrip_and_no_gt_precrop(sampler_runtime, tmp_path):
    module = importlib.import_module('datasets.kitti_mf')
    rect, transform, raw = _make_kitti(tmp_path)
    dataset = module.KITTIMFDataset(tmp_path, 'train', scene_ids=[0], ct_enable_v30=True,
        preload_offset=10, preloading=False, frame_period=.1, ct_pointcloud_cache_bytes=1024)
    assert dataset.preload_offset == -1
    actual = dataset.get_frames(0, [0])[0]
    camera_to_tracking = np.asarray([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    level = camera_to_tracking @ rect @ transform[:, :3]
    np.testing.assert_allclose(actual['pc'].points, level @ raw[:, :3].T, rtol=1e-6, atol=1e-6)
    assert actual['pc'].nbr_points() == 3  # includes the far-away point outside GT precrop
    assert actual['pc'].points.dtype == np.float32
    np.testing.assert_array_equal(actual['pc'].point_ids, [0, 1, 2])
    full = np.eye(4)
    full[:3, :4] = rect @ transform
    center_sensor = np.linalg.solve(level, actual['3d_bbox'].center)
    np.testing.assert_allclose((full @ np.r_[center_sensor, 1])[:3], [1, 1, 10], atol=1e-12)
    # Recover rectified-camera box corners via the exact same transform.
    corners_sensor = np.linalg.solve(level, actual['3d_bbox'].corners())
    corners_camera = (full @ np.vstack((corners_sensor, np.ones(8))))[:3]
    ry = Quaternion(axis=[0, 1, 0], radians=.35).rotation_matrix
    local = ry.T @ (corners_camera - np.asarray([1, 1, 10])[:, None])
    np.testing.assert_allclose(np.sort(np.abs(local), axis=1), np.repeat([[2.], [1.], [1.]], 8, axis=1), atol=1e-11)
    changed = copy.deepcopy(dataset.tracklet_anno_list[0][0])
    changed['x'] += 100
    changed['width'] *= 3
    other = dataset._get_frame_from_anno(changed)
    np.testing.assert_array_equal(other['pc'].points, actual['pc'].points)
    actual['pc'].points[:] = 999
    np.testing.assert_array_equal(dataset.get_frames(0, [0])[0]['pc'].points, other['pc'].points)
    assert actual['coordinate_mode'] == 'sensor_relative' and actual['sensor_to_sequence_world'] is None
    assert actual['scene_id'] == '0000'
    assert dataset.velos == {}  # v30 never fills the old unbounded dictionary


def test_kitti_metadata_stride_empty_and_legacy_branch(sampler_runtime, tmp_path, monkeypatch):
    module = importlib.import_module('datasets.kitti_mf')
    _make_kitti(tmp_path, frames=(0, 1, 3, 4, 7))
    dataset = module.KITTIMFDataset(tmp_path, 'train', scene_ids=[0], ct_enable_v30=True,
        frame_period=.1, ct_frame_stride=2, allow_missing_pointcloud=True)
    metadata = dataset.get_frames_metadata(0, [0, 1, 2])
    assert [row['frame_id'] for row in metadata] == [0, 3, 7]
    np.testing.assert_allclose([row['timestamp'] for row in metadata], [0, .3, .7])
    assert all('pc' not in row for row in metadata)
    assert '/stride/2/' in dataset.get_endpoint_key(0, 1)
    # Metadata is available even if the corresponding cloud is missing or empty.
    (tmp_path / 'velodyne' / '0000' / '000003.bin').unlink()
    (tmp_path / 'velodyne' / '0000' / '000007.bin').write_bytes(b'')
    for index in (1, 2):
        row = dataset.get_frames(0, [index])[0]
        assert row['pc'].points.shape == (3, 0) and row['pc'].point_ids.shape == (0,)
    monkeypatch.setattr(dataset, '_pointcloud_for_frame', lambda *args: pytest.fail('metadata loaded cloud'))
    assert dataset.get_frame_metadata(0, 2)['raw_frame_token'] == '0000/000007'
    legacy = module.KITTIMFDataset(tmp_path, 'train', scene_ids=[0], preload_offset=10,
                                    frame_period=.1, ct_enable_v30=False)
    assert legacy.preload_offset == 10
    assert legacy._calibration_for_scene('0000').shape == (3, 4)
    assert len(legacy._level_rotations) == 0


def test_kitti_manifest_emits_all_endpoints_and_partial_tail(sampler_runtime, tmp_path):
    module = importlib.import_module('datasets.kitti_mf')
    _make_kitti(tmp_path, scenes=tuple(f'{i:04d}' for i in range(17)))
    manifest = build_dataset_manifest(dict(dataset='kitti_mf', version='kitti_tracking'))
    dataset = module.KITTIMFDataset(tmp_path, 'train', ct_enable_v30=True,
        ct_scene_manifest=manifest, ct_scene_names=manifest['scenes']['train'], ct_scene_role='train')
    wrapped = types.SimpleNamespace(dataset=dataset, num_candidates=1)
    sampler = OnlineRecursiveBatchSampler(wrapped, slots=16, candidate_views=1, shadow_enabled=False)
    batches = list(sampler)
    endpoints = [(row[3], row[4]) for batch in batches for row in batch]
    assert len(endpoints) == len(set(endpoints)) == 17 * 4
    assert set(endpoints) == {(track, frame) for track in range(17) for frame in range(1, 5)}
    assert any(len(batch) < 16 for batch in batches)
    assert sampler.dataset_manifest_sha256 == manifest['content_sha256']


@pytest.fixture
def nuscenes_dataset_runtime(sampler_runtime, monkeypatch):
    classes = sampler_runtime[1]
    tables = {}
    class FakeNuScenes:
        calls = 0
        def __init__(self, **kwargs):
            type(self).calls += 1
            self.tables = copy.deepcopy(tables)
            self.instance = list(self.tables['instance'].values())
        def get(self, table, token):
            return self.tables[table][token]
    class NuBox(classes.Box):
        def __init__(self, *args, token=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.token = token
    class LidarPointCloud(classes.PointCloud):
        calls = 0
        @classmethod
        def from_file(cls, path):
            cls.calls += 1
            return cls(np.fromfile(path, np.float32).reshape(-1, 5)[:, :4].T)
    for name, attributes in {
        'nuscenes.nuscenes': dict(NuScenes=FakeNuScenes),
        'nuscenes.utils.data_classes': dict(LidarPointCloud=LidarPointCloud, Box=NuBox),
        'nuscenes.utils.splits': dict(create_splits_scenes=lambda: {'train_track': ['scene-0001'], 'val': ['scene-0001']}),
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        parent, child = name.rsplit('.', 1)
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    dataset_module = importlib.import_module('datasets.nuscenes_lidar_mf')
    return dataset_module, tables, FakeNuScenes, LidarPointCloud


def test_nuscenes_shared_metadata_cached_world_cloud_and_stride(nuscenes_dataset_runtime, tmp_path):
    module, tables, FakeNuScenes, LidarPointCloud = nuscenes_dataset_runtime
    tables.update(category={'car': {'name': 'vehicle.car'}},
        instance={'inst': dict(token='inst', first_annotation_token='a0', category_token='car')},
        scene={'scene': {'name': 'scene-0001'}}, sample={}, sample_data={}, sample_annotation={},
        calibrated_sensor={'cal': dict(rotation=Quaternion(axis=[0, 0, 1], radians=.2).elements,
                                       translation=[.2, .3, 1.])},
        ego_pose={'ego': dict(rotation=Quaternion(axis=[0, 0, 1], radians=-.3).elements,
                              translation=[100., 20., 0.])})
    raw = np.arange(15, dtype=np.float32).reshape(3, 5)
    for i in range(3):
        raw.tofile(tmp_path / f'{i}.bin')
        tables['sample'][f's{i}'] = dict(scene_token='scene', data={'LIDAR_TOP': f'd{i}'})
        tables['sample_data'][f'd{i}'] = dict(token=f'd{i}', timestamp=1_500_000_000_000_000+i*500_000,
            is_key_frame=True, calibrated_sensor_token='cal', ego_pose_token='ego', filename=f'{i}.bin')
        tables['sample_annotation'][f'a{i}'] = dict(token=f'a{i}', sample_token=f's{i}',
            translation=[i, 0., 0.], size=[2., 4., 2.], rotation=[1., 0., 0., 0.],
            category_name='vehicle.car', num_lidar_pts=1, next=f'a{i+1}' if i < 2 else '')
    args = dict(path=tmp_path, split='train_track', category_name='Car', version='v1.0-mini',
                preloading=False, key_frame_only=True, ct_enable_v30=True, ct_frame_stride=2)
    left = module.NuScenesMFDataset(**args)
    right = module.NuScenesMFDataset(**args)
    assert left.nusc is right.nusc and FakeNuScenes.calls == 1
    metadata = left.get_frames_metadata(0, [0, 1])
    assert LidarPointCloud.calls == 0
    assert metadata[1]['raw_frame_token'] == 'd2'
    assert metadata[1]['timestamp'] - metadata[0]['timestamp'] == 1.
    metadata[0]['meta']['box_anno']['translation'][0] = 999
    assert right.get_frame_metadata(0, 0)['meta']['box_anno']['translation'][0] == 0
    first = left.get_frames(0, [0])[0]
    second = right.get_frames(0, [0])[0]
    assert LidarPointCloud.calls == 1
    np.testing.assert_array_equal(first['pc'].points, second['pc'].points)
    assert first['pc'].points.dtype == np.float32
    first['pc'].points[:] = 999
    first['pc'].point_ids[:] = -1
    third = left.get_frames(0, [0])[0]
    np.testing.assert_array_equal(third['pc'].points, second['pc'].points)
    np.testing.assert_array_equal(third['pc'].point_ids, [0, 1, 2])
    # Legacy decoding/transform and v30 cache produce exactly the same arrays.
    legacy = module.NuScenesMFDataset(**{**args, 'ct_enable_v30': False})
    old = legacy.get_frames(0, [0])[0]
    np.testing.assert_array_equal(old['pc'].points, third['pc'].points)
    np.testing.assert_array_equal(old['pc'].point_ids, third['pc'].point_ids)
