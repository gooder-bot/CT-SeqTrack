"""V41 独立 KITTI 两臂协议；文献坐标约定、完整原云与安全训练缓存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from datasets.cache import DEFAULT_POINTCLOUD_CACHE_BYTES, load_pointcloud_arrays
from datasets.data_classes import Box, PointCloud
from datasets.kitti_mf import KITTIMFDataset


GEOMETRY = 'open3dsot_velodyne_v1'
KNOWN_MISSING = frozenset(('0001', frame) for frame in range(177, 181))
SCENE_SPLITS = {'train': tuple(f'{i:04d}' for i in range(17)),
                'dev': ('0017', '0018'), 'test': ('0019', '0020')}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def build_v41_kitti_manifest():
    """算法两臂共用的数据合同；不借用历史 calibration/dev 四角色。"""
    splits = {key: list(value) for key, value in SCENE_SPLITS.items()}
    result = dict(schema='ct_kitti_two_arm_v41', category='Car', version='kitti_tracking',
                  roles=splits, split_sha256=_sha(splits), geometry=GEOMETRY,
                  coordinate_mode='sensor_relative', frame_period=.1, frame_stride=1,
                  known_missing=[f'{scene}/{frame:06d}.bin' for scene, frame in sorted(KNOWN_MISSING)],
                  missing_policy='whitelist_empty_preserve_endpoint_otherwise_error',
                  evaluation_size='first_frame', evaluation_preload_offset=-1,
                  initialization='first_frame_gt_once', evaluation='complete_tracklet_ope')
    result['content_sha256'] = _sha(result)
    return result


def validate_v41_kitti_manifest(value):
    if value != build_v41_kitti_manifest():
        raise ValueError('v41 requires the registered three-role KITTI Car protocol')
    return dict(value)


def kitti_input_metadata_identity(path):
    root = KITTIMFDataset._resolve_data_root(path)
    files = {f'{folder}/{scene:04d}.txt': hashlib.sha256(
        (root / folder / f'{scene:04d}.txt').read_bytes()).hexdigest()
        for folder in ('label_02', 'calib') for scene in range(21)}
    return dict(files=files, content_sha256=_sha(files))


def _role(value):
    role = str(value).strip().lower()
    role = {'val': 'dev', 'valid': 'dev', 'validation': 'dev'}.get(role, role)
    if role not in (*SCENE_SPLITS, 'auxiliary'):
        raise ValueError('v41 KITTI role must be train/dev/test/auxiliary')
    return role


def _readonly(value):
    value = np.asarray(value).copy()
    value.flags.writeable = False
    return value


def normal_train_point_view(frame, roi_boxes, box_size, *, scale=1.25, offset=2.,
                            enabled=True, auxiliary=False):
    """返回完整包含实际 ROI 的缓存视图；raw_indices 必须映射回完整原云。

    frame['pc'] 永不替换。候选不完整落入训练 GT10 AABB 时直接回原云；
    auxiliary/dev/test 永远原云。训练 GT 只决定可证明等价的存储视图。
    """
    pc = frame['pc']
    points = np.asarray(pc.points[:3].T)
    ids = np.asarray(pc.point_ids, dtype=np.int64)
    indices = np.arange(len(points), dtype=np.int64)
    result = dict(points=points, ids=ids, raw_indices=indices, cache_used=False)
    cache = frame.get('_v41_normal_train_cache')
    if not enabled or auxiliary or frame.get('v41_data_role') != 'train' or cache is None:
        return result
    boxes = np.asarray(roi_boxes, dtype=np.float64).reshape(-1, 4)
    size = np.asarray(box_size, dtype=np.float64)
    if (not len(boxes) or size.shape != (3,) or not np.isfinite(size).all()
            or not (size > 0).all() or not np.isfinite(boxes).all()
            or not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0 or offset < 0):
        raise ValueError('normal train cache requires finite actual ROI boxes and positive LWH')
    half = size * (float(scale) / 2) + float(offset)
    signs = np.asarray([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    local = signs * half
    for box in boxes:
        c, s = np.cos(box[3]), np.sin(box[3])
        corners = local @ np.asarray([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]) + box[:3]
        # 保守收紧边界；浮点接近边界时回 raw，不增加 GT 依赖的漏点风险。
        if not ((corners >= cache['lower'] + 1e-7).all()
                and (corners <= cache['upper'] - 1e-7).all()):
            return result
    return dict(points=cache['points'], ids=cache['ids'],
                raw_indices=cache['raw_indices'], cache_used=True)


class KITTIV41Dataset(KITTIMFDataset):
    """只覆盖新协议和文献几何；原 KITTI adapter 与旧四角色合同不变。"""

    def __init__(self, path, split, category_name='Car', version='kitti_tracking', *,
                 protocol_manifest=None, usage=None, train_preload_offset=10., **kwargs):
        role = _role(usage or split)
        source_role = 'train' if role == 'auxiliary' else role
        if category_name != 'Car' or version != 'kitti_tracking':
            raise ValueError('v41 formal KITTI protocol is Car/kitti_tracking')
        if float(train_preload_offset) != 10.:
            raise ValueError('v41 ordinary training preload must be the registered 10 m')
        self.v41_manifest = validate_v41_kitti_manifest(
            build_v41_kitti_manifest() if protocol_manifest is None else protocol_manifest)
        self.v41_data_role = role
        self.v41_source_role = source_role
        self.v41_train_preload_offset = 10.
        self.input_metadata_identity = kitti_input_metadata_identity(path)
        self.input_metadata_sha256 = self.input_metadata_identity['content_sha256']
        for name in ('ct_scene_manifest', 'ct_scene_names', 'ct_scene_role', 'scene_ids',
                     'protocol_role', 'preloading', 'preload_offset', 'allow_missing_pointcloud'):
            kwargs.pop(name, None)
        if int(kwargs.get('ct_frame_stride', 1)) != 1 or kwargs.get('kitti_hv_interval', 1) != 1:
            raise ValueError('v41 first KITTI experiment uses all source frames, stride 1')
        super().__init__(path, source_role, category_name, version,
                         ct_scene_names=list(SCENE_SPLITS[source_role]),
                         ct_scene_role=source_role, protocol_role=source_role,
                         ct_scene_manifest=None, preloading=False, preload_offset=-1,
                         allow_missing_pointcloud=False, **kwargs)
        self.ct_scene_manifest = self.v41_manifest
        self.preload_offset = -1.  # 原云永不按 GT 裁掉；缓存单独挂载在 train 帧。

    def _configure_temporal_protocol(self, kwargs, **parameters):
        parameters['dataset_name'] = 'kitti_v41'
        return super()._configure_temporal_protocol(kwargs, **parameters)

    def _tracklet_identity(self, source_idx, tracklet):
        result = super()._tracklet_identity(source_idx, tracklet)
        first = tracklet[0]
        result['tracklet_key'] = (f'kitti_v41/{self.version}/{self.v41_source_role}/'
                                 f'{first["scene"]}/{int(first["track_id"])}/Car/'
                                 f'{GEOMETRY}/stride/1/{self.v41_manifest["content_sha256"]}')
        return result

    def _manifest_header(self):
        result = super()._manifest_header()
        result['dataset_manifest_sha256'] = self.v41_manifest['content_sha256']
        result['kitti_geometry'] = GEOMETRY
        return result

    def _calibration_for_scene(self, scene_id):
        scene_id = str(scene_id)
        if scene_id not in self.calibs:
            path = self.KITTI_calib / f'{scene_id}.txt'
            data = self._read_calib_file(path)
            matrix = next((data[k] for k in ('Tr_velo_cam', 'Tr_velo_to_cam') if k in data), None)
            if matrix is None or np.asarray(matrix).shape != (3, 4):
                raise ValueError(f'{path} requires a finite 3x4 Tr_velo_cam')
            matrix = np.asarray(matrix, dtype=np.float64)
            if (not np.isfinite(matrix).all()
                    or not np.allclose(matrix[:, :3].T @ matrix[:, :3], np.eye(3), atol=2e-3)):
                raise ValueError(f'{path} has invalid rigid Tr_velo_cam')
            self.calibs[scene_id] = matrix
        return self.calibs[scene_id]

    def _pointcloud_for_frame(self, scene_id, frame_id):
        scene_id, frame_id = str(scene_id), int(frame_id)
        path = self.KITTI_velo / scene_id / f'{frame_id:06d}.bin'

        def read():
            if not path.is_file():
                if (scene_id, frame_id) in KNOWN_MISSING:
                    return np.empty((3, 0), np.float32), np.empty(0, np.int64)
                raise FileNotFoundError(f'Unexpected missing KITTI point cloud: {path}')
            if path.stat().st_size % 16:
                raise ValueError(f'Malformed KITTI point cloud byte length: {path}')
            raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            if not np.isfinite(raw).all():
                raise ValueError(f'Nonfinite KITTI point cloud: {path}')
            return raw[:, :3].T.copy(), np.arange(len(raw), dtype=np.int64)

        key = ('kitti_v41', str(self.data_root), GEOMETRY, scene_id, frame_id)
        points, ids = load_pointcloud_arrays(key, read, self.ct_pointcloud_cache_bytes)
        return PointCloud(points, point_ids=ids)

    def _frame_metadata_from_anno(self, anno):
        scene_id, frame_id = str(anno['scene']), int(anno['frame'])
        transform = np.eye(4, dtype=np.float64)
        transform[:3] = self._calibration_for_scene(scene_id)
        center = np.linalg.solve(transform, np.asarray([float(anno['x']),
            float(anno['y']) - float(anno['height']) / 2, float(anno['z']), 1.]))[:3]
        size = [float(anno[key]) for key in ('width', 'length', 'height')]
        if not np.isfinite(center).all() or not np.isfinite(size).all() or min(size) <= 0:
            raise ValueError('KITTI box center/dimensions must be finite and positive')
        orientation = (Quaternion(axis=[0, 0, -1], radians=float(anno['rotation_y']))
                       * Quaternion(axis=[0, 0, -1], degrees=90))
        box = Box(center, size, orientation, name=str(anno['type']))
        return dict(**{'3d_bbox': box}, meta=dict(anno), frame_id=frame_id,
                    timestamp=frame_id * .1, scene_id=scene_id, sequence_id=scene_id,
                    raw_frame_id=frame_id, raw_frame_token=f'{scene_id}/{frame_id:06d}',
                    coordinate_mode='sensor_relative', sensor_to_sequence_world=None,
                    sensor_to_tracking_rotation=np.eye(3),
                    dataset_manifest_sha256=self.v41_manifest['content_sha256'],
                    input_metadata_sha256=self.input_metadata_sha256,
                    v41_data_role=self.v41_data_role, kitti_geometry=GEOMETRY,
                    pointcloud_missing=(scene_id, frame_id) in KNOWN_MISSING
                    and not (self.KITTI_velo / scene_id / f'{frame_id:06d}.bin').is_file())

    def _get_frame_from_anno(self, anno):
        frame = self._frame_metadata_from_anno(anno)
        frame['pc'] = self._pointcloud_for_frame(anno['scene'], anno['frame'])
        if self.v41_data_role == 'train':
            # 与文献 crop_pc_axis_aligned(scale=1, offset=10) 相同的宽 AABB。
            corners = frame['3d_bbox'].corners()
            lower, upper = corners.min(axis=1) - 10., corners.max(axis=1) + 10.
            points, ids = frame['pc'].points[:3].T, frame['pc'].point_ids
            selected = np.flatnonzero(((points > lower) & (points < upper)).all(-1))
            frame['_v41_normal_train_cache'] = dict(
                points=_readonly(points[selected]), ids=_readonly(ids[selected]),
                raw_indices=_readonly(selected), lower=_readonly(lower), upper=_readonly(upper))
        return frame


def get_v41_raw_dataset(config, role):
    """新两臂入口；auxiliary 共享 train split，但不使用训练 GT 缓存。"""
    def get(key, default=None):
        return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)
    if get('dataset') not in ('kitti', 'kitti_mf', 'kitti_v41'):
        raise ValueError('get_v41_raw_dataset requires KITTI')
    if get('v41_kitti_geometry', GEOMETRY) != GEOMETRY:
        raise ValueError('v41 two-arm geometry must match the registered baseline convention')
    usage = _role(role)
    return KITTIV41Dataset(get('path'), 'train' if usage == 'auxiliary' else usage,
        category_name=get('category_name', 'Car'), version=get('version', 'kitti_tracking'),
        usage=usage, hist_num=3, coordinate_mode='sensor_relative',
        train_preload_offset=get('v41_train_preload_offset', 10.),
        ct_pointcloud_cache_bytes=get('ct_pointcloud_cache_bytes', DEFAULT_POINTCLOUD_CACHE_BYTES),
        ct_frame_stride=get('ct_frame_stride', 1),
        dynamics_time_mode=get('dynamics_time_mode', 'true'),
        dynamics_fixed_delta_t=get('dynamics_fixed_delta_t', .1),
        dynamics_time_manifest=get('dynamics_time_manifest', ''))
