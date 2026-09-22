"""活动数据合同：工厂身份、raw ID、标定、缓存与三种时间控制。

仅使用合成数据及 nuScenes SDK stub；不依赖历史源码快照或真实数据根。
"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
from pyquaternion import Quaternion

from datasets.cache import PointCloudArrayLRU, process_pointcloud_cache
from datasets.protocol import build_dataset_manifest, select_dataset_protocol, validate_dataset_manifest
from datasets.protocol_utils import canonical_sha256, VIRTUAL_RATE_DEFAULTS




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
    import datasets.cache as module
    monkeypatch.setattr(module.os, 'getpid', lambda: -1234)
    assert len(process_pointcloud_cache(80)) == 0


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


def test_kitti_full_calibration_box_cloud_roundtrip_and_no_gt_precrop(tmp_path):
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
    assert not hasattr(dataset, "velos")


def test_kitti_metadata_stride_empty(tmp_path, monkeypatch):
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



@pytest.fixture
def nuscenes_dataset_runtime(monkeypatch):
    from datasets import data_classes as classes
    for name in ('nuscenes', 'nuscenes.utils'):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    monkeypatch.delitem(sys.modules, 'datasets.nuscenes_lidar_mf', raising=False)
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
    # Restore the module cache after the SDK stub leaves scope.
    import datasets
    monkeypatch.delattr(datasets, 'nuscenes_lidar_mf', raising=False)
    dataset_module = importlib.import_module('datasets.nuscenes_lidar_mf')
    yield dataset_module, tables, FakeNuScenes, LidarPointCloud
    sys.modules.pop('datasets.nuscenes_lidar_mf', None)
    if getattr(datasets, 'nuscenes_lidar_mf', None) is dataset_module:
        delattr(datasets, 'nuscenes_lidar_mf')


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
    sensor = Quaternion(tables['calibrated_sensor']['cal']['rotation']).rotation_matrix
    ego = Quaternion(tables['ego_pose']['ego']['rotation']).rotation_matrix
    # Two-stage in-place float32 transforms are the original SDK arithmetic.
    expected = raw[:, :3].T.copy()
    expected[:] = sensor @ expected
    expected[:] = expected + np.asarray([.2, .3, 1.])[:, None]
    expected[:] = ego @ expected
    expected[:] = expected + np.asarray([100., 20., 0.])[:, None]
    np.testing.assert_array_equal(third['pc'].points, expected)



# 2026-09-22: frozen v31 raw_factory -> old get_dataset SDK constructor arguments.
FACTORY_SHA256 = {
    ('v1.0-mini', 'train'): 'b12f9b6961f4b80f4ea110a8fbe688592fb44c66a686a7db4c3b460604df516b',
    ('v1.0-mini', 'calibration'): 'b4e4472dc6d042cf17c183bcf59ec1ded28b85ce0171a23019d8ab929695a0ee',
    ('v1.0-mini', 'dev'): 'c93fe4bee3320e336417b651f4e434a3bb978a3820c4b873670a1d224cc4a775',
    ('v1.0-mini', 'test'): '617550522c40e77c9d724d0087b8b75e690f65ef6e52a290f65b1f893cbf1880',
    ('v1.0-trainval', 'train'): 'ac45561605737b81f9e9d8921e799eed7f3157d2f1ea07c5ed5ceeabc366ba4e',
    ('v1.0-trainval', 'calibration'): 'ed2ea056e5d6f3345bb34b7138602dbc0174e7e3d6e8dbb10a998e7dd1abbea3',
    ('v1.0-trainval', 'dev'): 'ac5eec4edfc6f98c4982aeb6c09dd89380fa91ba646b473a0c892843258e49c3',
    ('v1.0-trainval', 'test'): '9947f903c57df2361ca7fd441d158a0c297312f2c905a4cc47ab2113b3f9b43b',
    ('kitti_tracking', 'train'): 'e2cd9c62f7a0de750b6b60387b13be4801a70b3edf8df31d92a7e529de750432',
    ('kitti_tracking', 'calibration'): '5ca544f908a7a19acb815ccb71f781df7afeb9c2aedfd1a5555949e0474c067f',
    ('kitti_tracking', 'dev'): '4a4549ba9ebf488d046baef524434c1c9dc175302d915a7795bcd6638010fabc',
    ('kitti_tracking', 'test'): 'ec9dd63653300d3c0e938c0989e24b2b9e023ef9f5d0c209144b4d74b31a0a2a',
}


def _scene_splits():
    return dict(mini_train=[f's{i}' for i in range(8)], mini_val=['v0', 'v1'],
                train_track=[f's{i:03d}' for i in range(350)], val=[f'v{i:03d}' for i in range(150)])


@pytest.mark.parametrize('version,role', FACTORY_SHA256)
def test_raw_factory_matches_frozen_constructor_arguments(monkeypatch, version, role):
    from models.ct_v31.config import normalize_config
    from datasets import get_raw_dataset
    import datasets.protocol as protocol
    original = protocol.build_dataset_manifest
    monkeypatch.setattr(protocol, 'build_dataset_manifest',
                        lambda config, scene_splits=None: original(config, _scene_splits()))
    kind = 'kitti_mf' if version == 'kitti_tracking' else 'nuscenes_mf'
    module = types.ModuleType('datasets.kitti_mf' if kind == 'kitti_mf' else 'datasets.nuscenes_lidar_mf')
    setattr(module, 'KITTIMFDataset' if kind == 'kitti_mf' else 'NuScenesMFDataset', lambda **kw: kw)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = normalize_config(dict(dataset=kind, version=version, path='DATA_ROOT',
        ct_coordinate_mode='sensor_relative' if kind == 'kitti_mf' else 'global'))
    actual = get_raw_dataset(config, role)
    assert canonical_sha256(actual) == FACTORY_SHA256[(version, role)]
    assert actual['ct_scene_manifest']['content_sha256'] == build_dataset_manifest(
        config, _scene_splits())['content_sha256']


@pytest.mark.parametrize('version', ['v1.0-mini', 'v1.0-trainval'])
def test_scene_order_and_registered_manifest_identity(version):
    splits = _scene_splits()
    mini = version == 'v1.0-mini'
    source, evaluation = ('mini_train', 'mini_val') if mini else ('train_track', 'val')
    ranked = sorted(splits[source], key=lambda scene: hashlib.sha256(
        ('ct27|42|' + scene).encode()).hexdigest())
    expected = dict(schema='ct_seqtrack.dataset_protocol.v30', dataset='nuscenes_mf',
        version=version, split_seed=42, training_source=source, evaluation_source=evaluation,
        parameter_training_overlap=True,
        scenes=dict(train=sorted(splits[source]), calibration=ranked[6:7] if mini else ranked[315:332],
                    dev=ranked[7:] if mini else ranked[332:], test=sorted(splits[evaluation])),
        coordinate_mode='global', coordinate_axes='nuscenes_global_z_up',
        time_source='lidar_sample_data_microseconds', frame_stride=1, full_endpoint_coverage=True)
    expected['content_sha256'] = canonical_sha256(expected)
    actual = build_dataset_manifest(dict(dataset='nuscenes_mf', version=version), splits)
    assert actual == expected
    changed = copy.deepcopy(actual)
    changed['scenes']['calibration'][0] = 'bad'
    with pytest.raises(ValueError, match='sha256'):
        validate_dataset_manifest(changed)


def test_kitti_roles_and_manifest_validation():
    config = dict(dataset='kitti_mf', version='kitti_tracking')
    assert select_dataset_protocol(config, 'val')[1:] == ('dev', ['0018'])
    assert select_dataset_protocol(config, 'eval')[1:] == ('test', ['0019', '0020'])
    original = build_dataset_manifest(config)
    assert original['scenes']['train'] == [f'{i:04d}' for i in range(17)]
    assert build_dataset_manifest({**config, 'ct_frame_stride': 2})['content_sha256'] != original['content_sha256']
    for key, value in [('ct_frame_stride', 1.5), ('ct_frame_stride', True),
                       ('kitti_frame_period', .5), ('ct_coordinate_mode', 'global')]:
        with pytest.raises(ValueError):
            build_dataset_manifest({**config, key: value})


def _populate_nuscenes(tables, root):
    tables.update(category={'car': {'name': 'vehicle.car'}},
        instance={'inst': dict(token='inst', first_annotation_token='a0', category_token='car')},
        scene={'scene': {'name': 'scene-0001'}}, sample={}, sample_data={}, sample_annotation={},
        calibrated_sensor={'cal': dict(rotation=[1., 0., 0., 0.], translation=[.2, .3, 1.])},
        ego_pose={'ego': dict(rotation=[1., 0., 0., 0.], translation=[100., 20., 0.])})
    raw = np.arange(15, dtype=np.float32).reshape(3, 5)
    for i, stamp in enumerate([1_000_000, 1_300_000, 1_900_000, 2_400_000]):
        raw.tofile(root / f'{i}.bin')
        tables['sample'][f's{i}'] = dict(scene_token='scene', data={'LIDAR_TOP': f'd{i}'})
        tables['sample_data'][f'd{i}'] = dict(token=f'd{i}', timestamp=stamp,
            is_key_frame=True, calibrated_sensor_token='cal', ego_pose_token='ego', filename=f'{i}.bin')
        tables['sample_annotation'][f'a{i}'] = dict(token=f'a{i}', sample_token=f's{i}',
            translation=[i, 0., 0.], size=[2., 4., 2.], rotation=[1., 0., 0., 0.],
            category_name='vehicle.car', num_lidar_pts=1, next=f'a{i+1}' if i < 3 else '')


@pytest.mark.parametrize('kind', ['kitti', 'nuscenes'])
def test_true_fixed_shuffled_time_and_default_selection(kind, nuscenes_dataset_runtime, tmp_path, monkeypatch):
    nu, tables, _, _ = nuscenes_dataset_runtime
    if kind == 'kitti':
        from datasets.kitti_mf import KITTIMFDataset
        _make_kitti(tmp_path, frames=(0, 3, 9, 14))
        constructor = KITTIMFDataset
        kwargs = dict(path=tmp_path, split='train', version='kitti_tracking', scene_ids=[0])
    else:
        _populate_nuscenes(tables, tmp_path)
        constructor = nu.NuScenesMFDataset
        kwargs = dict(path=tmp_path, split='train_track', version='v1.0-mini', key_frame_only=True)
    kwargs.update(ct_enable_v30=True, dynamics_fixed_delta_t=.5, hist_num=3)
    real = constructor(**kwargs)
    fixed = constructor(**kwargs, dynamics_time_mode='fixed')
    key = real.get_tracklet_key(0)
    expected_key = ('kitti_mf/kitti_tracking/train/0000/0/Car/interval/1/phase/0/v30/sensor_relative/stride/1'
                    if kind == 'kitti' else 'nuscenes_mf/v1.0-mini/train_track/scene/inst/v30/global/stride/1')
    assert key == expected_key
    selection = [dict(tracklet_key=key, included=True, keep_indices=[0, 1, 2, 3])]
    assert real.virtual_rate_selection_sha256 == canonical_sha256(selection)
    assert real._virtual_rate_protocol() == {
        key.removeprefix('virtual_rate_'): value for key, value in VIRTUAL_RATE_DEFAULTS.items()}
    real_rows = real.get_frames_metadata(0, range(4))
    fixed_rows = fixed.get_frames_metadata(0, range(4))
    assert [r['_ct_effective_timestamp'] for r in real_rows] == [r['timestamp'] for r in real_rows]
    assert [r['_ct_effective_timestamp'] for r in fixed_rows] == [0., .5, 1., 1.5]
    manifest_path = tmp_path / 'shuffled.json'
    monkeypatch.setattr(real, '_git_state', lambda: dict(commit='synthetic', dirty=False))
    real.build_dynamics_time_manifest(manifest_path, seed=42)
    shuffled = constructor(**kwargs, dynamics_time_mode='shuffled', dynamics_time_manifest=str(manifest_path))
    rows = shuffled.get_frames_metadata(0, range(4))
    np.testing.assert_allclose(sorted(np.diff([r['_ct_effective_timestamp'] for r in rows])),
                               sorted(np.diff([r['timestamp'] for r in real_rows])))
    assert [r['timestamp'] for r in rows] == [r['timestamp'] for r in real_rows]
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    payload['entries'][1]['effective_timestamp'] += 1
    manifest_path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256'):
        constructor(**kwargs, dynamics_time_mode='shuffled', dynamics_time_manifest=str(manifest_path))


def test_dataset_package_import_is_lightweight():
    script = "import datasets, sys; assert not any(x in sys.modules for x in ('nuscenes','torch','pytorch_lightning','pointnet2_ops'))"
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
