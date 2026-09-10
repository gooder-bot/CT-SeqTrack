"""性能分支只消除未使用的 IO/全云复制，保持输入和监督逐位一致。"""
import copy
import importlib
import sys
import types

import numpy as np
import pytest
from pyquaternion import Quaternion

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v29_rollin import setup_case
from utils.v29_rollin import process_query, _initial_box


@pytest.mark.parametrize('oriented,canonical', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('count', [0, 1, 2, 3, 3000])
def test_crop_performance_preserves_points_ids_and_source(sampler_runtime, oriented, canonical, count):
    sampler, classes, _, _ = sampler_runtime
    rng = np.random.default_rng(17)
    xyz = rng.normal(size=(3, count)).astype(np.float32)
    pc = classes.PointCloud(xyz, point_ids=np.arange(count, dtype=np.int64) * 7 + 10)
    box = classes.Box([.3, -.2, .1], [2., 4., 2.],
                      Quaternion(axis=[0, 0, 1], radians=.43))
    ref = classes.Box([-.1, .6, .2], [2., 4., 2.],
                      Quaternion(axis=[0, 0, 1], radians=-.19))
    source = copy.deepcopy(pc)
    old = sampler.points_utils.generate_subwindow_with_aroundboxs(
        pc, box, ref, scale=1.2, offset=.4, oriented=oriented, canonicalize=canonical)
    new = sampler.points_utils.generate_subwindow_with_aroundboxs(
        pc, box, ref, scale=1.2, offset=.4, oriented=oriented, canonicalize=canonical,
        copy_selected_only=True)
    np.testing.assert_array_equal(new.points, old.points)
    np.testing.assert_array_equal(new.point_ids, old.point_ids)
    for actual, expected in ((new.points, old.points), (new.point_ids, old.point_ids)):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.strides == expected.strides
        assert actual.flags.writeable == expected.flags.writeable
    np.testing.assert_array_equal(pc.points, source.points)
    np.testing.assert_array_equal(pc.point_ids, source.point_ids)
    assert not np.shares_memory(new.points, pc.points)


@pytest.mark.parametrize('crop_name', ['crop_pc_axis_aligned', 'crop_pc_axis_aligned_with_aroundboxs'])
@pytest.mark.parametrize('layout', ['C', 'F', 'sliced', 'reversed', 'readonly'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_crop_preserves_stride_writeability_shape_and_dtype(sampler_runtime, crop_name, layout, dtype):
    sampler, classes, _, _ = sampler_runtime
    xyz = np.arange(3 * 11, dtype=dtype).reshape(3, 11) / 12 - 1
    ids = np.arange(11, dtype=np.int64) * 3 + 71
    if layout == 'F':
        xyz = np.asfortranarray(xyz)
    elif layout == 'sliced':
        xyz, ids = xyz[:, ::2], ids[::2]
    elif layout == 'reversed':
        xyz, ids = xyz[:, ::-1], ids[::-1]
    elif layout == 'readonly':
        xyz.flags.writeable = False
        ids.flags.writeable = False
    pc = classes.PointCloud(xyz, point_ids=ids)
    source_xyz, source_ids = pc.points.copy(), pc.point_ids.copy()
    source_layout = (pc.points.shape, pc.points.dtype, pc.points.strides, pc.points.flags.writeable,
                     pc.point_ids.shape, pc.point_ids.dtype, pc.point_ids.strides, pc.point_ids.flags.writeable)
    box = classes.Box([0., 0., .5], [2., 2., 3.], Quaternion())
    crop = getattr(sampler.points_utils, crop_name)
    extra = {'around_boxs': [copy.deepcopy(box)]} if crop_name.endswith('aroundboxs') else {}
    old, old_mask = crop(pc, box, return_mask=True, **extra)
    new, new_mask = crop(pc, box, return_mask=True, copy_selected_only=True, **extra)
    np.testing.assert_array_equal(new_mask, old_mask)
    np.testing.assert_array_equal(new.point_ids, source_ids[new_mask])
    for actual, expected in ((new.points, old.points), (new.point_ids, old.point_ids)):
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.strides == expected.strides
        assert actual.flags.writeable == expected.flags.writeable
        assert actual.flags.c_contiguous == expected.flags.c_contiguous
        assert actual.flags.f_contiguous == expected.flags.f_contiguous
    assert new.points.size and new.points.flags.writeable and new.point_ids.flags.writeable
    new.points[:] = 100
    new.point_ids[:] = -1
    np.testing.assert_array_equal(pc.points, source_xyz)
    np.testing.assert_array_equal(pc.point_ids, source_ids)
    assert source_layout == (
        pc.points.shape, pc.points.dtype, pc.points.strides, pc.points.flags.writeable,
        pc.point_ids.shape, pc.point_ids.dtype, pc.point_ids.strides, pc.point_ids.flags.writeable)


def test_crop_strict_boundaries_and_extra_attribute_aliases(sampler_runtime):
    sampler, classes, _, _ = sampler_runtime

    class AnnotatedCloud(classes.PointCloud):
        pass

    xyz = np.array([[-1., np.nextafter(-1., 0.), 0., np.nextafter(1., 0.), 1.],
                    [0.] * 5, [0.] * 5])
    pc = AnnotatedCloud(xyz, point_ids=np.array([71, 42, 13, 24, 99]))
    pc.extra = {'nested': [pc.points, pc.point_ids], 'note': ['preserved']}
    box = classes.Box([0., 0., 0.], [2., 2., 2.], Quaternion())
    old, mask = sampler.points_utils.crop_pc_axis_aligned(pc, box, return_mask=True)
    new, new_mask = sampler.points_utils.crop_pc_axis_aligned(
        pc, box, return_mask=True, copy_selected_only=True)
    np.testing.assert_array_equal(mask, [False, True, True, True, False])
    np.testing.assert_array_equal(mask, new_mask)
    np.testing.assert_array_equal(old.points, new.points)
    np.testing.assert_array_equal(new.point_ids, [42, 13, 24])
    assert type(new) is AnnotatedCloud
    for original, expected, actual in zip(pc.extra['nested'], old.extra['nested'], new.extra['nested']):
        np.testing.assert_array_equal(expected, actual)
        assert not np.shares_memory(original, actual)
    new.extra['note'].append('changed')
    new.extra['nested'][0][0, 0] = 100
    assert pc.extra['note'] == ['preserved'] and pc.points[0, 0] == -1.


def test_standard_crop_does_not_deepcopy_replaced_arrays(sampler_runtime):
    sampler, classes, _, _ = sampler_runtime

    class NoDeepcopy(np.ndarray):
        def __deepcopy__(self, memo):
            raise AssertionError('full source array must not be copied')

    pc = classes.PointCloud(np.zeros((3, 4096)).view(NoDeepcopy))
    box = classes.Box([0., 0., 0.], [2., 2., 2.], Quaternion())
    result = sampler.points_utils.crop_pc_axis_aligned(pc, box, copy_selected_only=True)
    assert result.points.shape == (3, 4096)
    with pytest.raises(AssertionError, match='full source array'):
        sampler.points_utils.crop_pc_axis_aligned(pc, box)


@pytest.mark.parametrize('mode', ['true', 'fixed', 'shuffled'])
def test_metadata_matches_full_frame_without_pointcloud_io(sampler_runtime, monkeypatch, mode):
    _, classes, _, _ = sampler_runtime
    nu_main = types.ModuleType('nuscenes.nuscenes')
    nu_main.NuScenes = object
    nu_data = types.ModuleType('nuscenes.utils.data_classes')

    class NuBox(classes.Box):
        def __init__(self, *args, token=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.token = token

    calls = []

    class Lidar:
        @staticmethod
        def from_file(path):
            calls.append(path)
            return classes.PointCloud(np.array([[1.], [2.], [3.]]))

    nu_data.Box, nu_data.LidarPointCloud = NuBox, Lidar
    nu_splits = types.ModuleType('nuscenes.utils.splits')
    nu_splits.create_splits_scenes = lambda: {}
    for key, module in [('nuscenes.nuscenes', nu_main),
                        ('nuscenes.utils.data_classes', nu_data),
                        ('nuscenes.utils.splits', nu_splits)]:
        monkeypatch.setitem(sys.modules, key, module)
    module = importlib.import_module('datasets.nuscenes_lidar_mf')
    dataset = module.NuScenesMFDataset.__new__(module.NuScenesMFDataset)
    anno = {
        'sample_data_lidar': dict(filename='lidar.bin', timestamp=1500000000500000,
                                 token='frame', calibrated_sensor_token='sensor', ego_pose_token='pose'),
        'box_anno': dict(translation=[.1, .2, .3], size=[2., 4., 2.],
                         rotation=Quaternion(axis=[0, 0, 1], radians=.34).elements.tolist(),
                         category_name='vehicle.car', token='box'),
    }
    dataset.tracklet_anno_list = [[anno, anno]]
    dataset.path, dataset.preloading = 'unused', False
    dataset.get_endpoint_key = lambda seq, frame: f'{seq}/{frame}'
    dataset.dynamics_time_mode, dataset.dynamics_fixed_delta_t = mode, .5
    dataset._shuffled_effective_timestamps = {'0/0': .2, '0/1': .73}
    sensor_calls = []

    def sensor_record(kind, token):
        sensor_calls.append((kind, token))
        return {'rotation': [1., 0., 0., 0.], 'translation': [0., 0., 0.]}

    dataset.nusc = types.SimpleNamespace(get=sensor_record)
    metadata = dataset.get_frame_metadata(0, 1)
    assert calls == [] and sensor_calls == [] and 'pc' not in metadata
    full = dataset.get_frames(0, [1])[0]
    assert len(calls) == 1 and len(sensor_calls) == 2
    assert metadata['meta'] is anno and full['meta'] is anno
    for key in ('timestamp', 'frame_id', '_ct_endpoint_key', '_ct_effective_timestamp',
                '_ct_dynamics_time_mode'):
        assert metadata[key] == full[key]
    for name in ('center', 'wlh', 'rotation_matrix'):
        np.testing.assert_array_equal(getattr(metadata['3d_bbox'], name), getattr(full['3d_bbox'], name))
    dataset.preloading, dataset.training_samples = True, [[full, full]]
    cached = dataset.get_frames(0, [1])[0]
    np.testing.assert_array_equal(metadata['3d_bbox'].center, cached['3d_bbox'].center)
    assert len(calls) == 1
    # 通用读取继续返回完整帧，保留重复请求的顺序、预加载别名及扩展字段。
    full['custom_metadata'] = {'marker': object()}
    cached_rows = dataset.get_frames(0, [1, 0, 1])
    assert [frame['_ct_endpoint_key'] for frame in cached_rows] == ['0/1', '0/0', '0/1']
    assert len(calls) == 1
    for frame in cached_rows:
        assert frame is not full and frame['pc'] is full['pc']
        assert frame['3d_bbox'] is full['3d_bbox']
        assert frame['custom_metadata'] is full['custom_metadata']
    assert len({id(frame) for frame in cached_rows}) == 3
    dataset.preloading = False
    fresh_rows = dataset.get_frames(0, [1, 0, 1])
    assert [frame['_ct_endpoint_key'] for frame in fresh_rows] == ['0/1', '0/0', '0/1']
    assert len(calls) == 4 and len(sensor_calls) == 8
    assert len({id(frame['pc']) for frame in fresh_rows}) == 3
    assert len({id(frame['3d_bbox']) for frame in fresh_rows}) == 3
    second_metadata = dataset.get_frame_metadata(0, 1)
    assert 'pc' not in second_metadata and len(calls) == 4 and len(sensor_calls) == 8
    assert second_metadata['3d_bbox'] is not metadata['3d_bbox']
    np.testing.assert_array_equal(second_metadata['3d_bbox'].center, anno['box_anno']['translation'])


@pytest.mark.parametrize('recursive', [False, True])
@pytest.mark.parametrize('perf', [False, True])
def test_empty_scan_only_omitted_when_result_is_unused(sampler_runtime, monkeypatch, recursive, perf):
    sampler, config, _, _, payload, _, _ = _case(sampler_runtime, 'b0')
    config.ct_enable_v28 = config.ct_enable_v29 = True
    config.ct_runtime_optimization = 'equivalent_v1' if perf else 'legacy'
    if not recursive:
        payload.pop('online_recursive_state')
    calls = []
    original = sampler.geometry_utils.points_in_box

    def count_scan(box, points):
        calls.append(points.shape[1])
        return original(box, points)

    class StopBeforeCrop(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopBeforeCrop

    monkeypatch.setattr(sampler.geometry_utils, 'points_in_box', count_scan)
    monkeypatch.setattr(sampler, 'apply_shared_se2_to_boxes', stop)
    with pytest.raises(StopBeforeCrop):
        sampler.motion_processing_mf(payload, config)
    assert len(calls) == (0 if recursive and perf else 3)


@pytest.mark.parametrize('candidate', [1, 2, 3])
@pytest.mark.parametrize('count', [0, 1, 2, 3])
def test_rollin_processing_equivalent_including_sparse_labels(sampler_runtime, candidate, count):
    config, item = setup_case(sampler_runtime, start=0, endpoint=2)
    _, classes, _, _ = sampler_runtime
    item['candidate'] = candidate
    for frame in item['frames']:
        xyz = np.repeat(frame['3d_bbox'].center[:, None], count, axis=1)
        frame['pc'] = classes.PointCloud(xyz, point_ids=np.arange(count) + 100)
    predictions = {0: _initial_box(item, config)}
    before, _ = process_query(item, predictions, 1, config)
    config.ct_runtime_optimization = 'equivalent_v1'
    after, _ = process_query(item, predictions, 1, config)
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)


@pytest.mark.parametrize('frame_id', [0, 8])
def test_mf_first_frame_uses_metadata_only_in_perf(sampler_runtime, frame_id):
    sampler, _, _, EasyDict = sampler_runtime
    calls = []
    frames = [{'pc': object(), '3d_bbox': object()} for _ in range(9)]

    def get_frames(tracklet, frame_ids):
        calls.append(tuple(frame_ids))
        return [frames[index] for index in frame_ids]

    metadata = {'3d_bbox': frames[0]['3d_bbox']}
    instance = sampler.MotionTrackingSamplerMF.__new__(sampler.MotionTrackingSamplerMF)
    instance.dataset = types.SimpleNamespace(get_frames=get_frames,
                                             get_frame_metadata=lambda *_: metadata)
    instance.config = EasyDict(ct_enable_v29=True, ct_runtime_optimization='equivalent_v1')
    first, current = instance._first_and_current_frames(0, frame_id)
    assert first['3d_bbox'] is frames[0]['3d_bbox'] and current is frames[frame_id]
    assert 'pc' not in first and 'pc' in current
    assert calls == [(frame_id,)]
    instance.config.ct_runtime_optimization = 'legacy'
    instance._first_and_current_frames(0, frame_id)
    assert calls[-1] == (0, frame_id)
