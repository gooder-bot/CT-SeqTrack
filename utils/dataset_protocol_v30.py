"""v30 数据集身份与场景用途；不加载点云，也不依赖 nuScenes 的导入副作用。"""
from __future__ import annotations

import copy
import hashlib
import json
import math


SCHEMA = 'ct_seqtrack.dataset_protocol.v30'
ROLES = ('train', 'calibration', 'dev', 'test')


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def frame_stride(value):
    """拒绝把小数或 bool 静默转换成另一套采样协议。"""
    if isinstance(value, bool) or int(value) != float(value) or int(value) < 1:
        raise ValueError('ct_frame_stride must be a positive integer')
    return int(value)


def build_dataset_manifest(config, scene_splits=None):
    dataset = str(_get(config, 'dataset'))
    if dataset == 'kitti':
        dataset = 'kitti_mf'
    version = str(_get(config, 'version', 'kitti_tracking'))
    seed = int(_get(config, 'ct_partition_seed', 42))
    stride = frame_stride(_get(config, 'ct_frame_stride', 1))
    if dataset == 'kitti_mf':
        if version != 'kitti_tracking':
            raise ValueError('v30 KITTI requires version=kitti_tracking')
        if _get(config, 'kitti_scene_ids') is not None:
            raise ValueError('v30 KITTI scene roles are fixed; kitti_scene_ids is not an override')
        coordinate_mode = str(_get(config, 'ct_coordinate_mode', 'sensor_relative'))
        if coordinate_mode != 'sensor_relative':
            raise ValueError('KITTI without registered ego poses requires sensor_relative coordinates')
        period = float(_get(config, 'kitti_frame_period', .1))
        if period != .1:
            raise ValueError('v30 KITTI source frame period is 0.1 seconds; use ct_frame_stride for gaps')
        roles = {'train': [f'{i:04d}' for i in range(17)],
                 'calibration': ['0017'], 'dev': ['0018'], 'test': ['0019', '0020']}
        source, evaluation, overlap = 'train', 'test', False
        time_source = 'original_frame_index_times_0.1_seconds'
        axes = 'rectified_camera_level_at_lidar_origin_x_forward_y_left_z_up'
    elif dataset == 'nuscenes_mf':
        if scene_splits is None:
            from nuscenes.utils.splits import create_splits_scenes
            scene_splits = create_splits_scenes()
        from utils.v28_protocol import build_scene_manifest
        legacy = build_scene_manifest(scene_splits, version, seed)
        roles = copy.deepcopy(legacy['scenes'])
        source, evaluation = legacy['training_source'], legacy['evaluation_source']
        overlap = legacy['parameter_training_overlap']
        coordinate_mode = str(_get(config, 'ct_coordinate_mode', 'global'))
        if coordinate_mode != 'global':
            raise ValueError('v30 nuScenes requires calibrated global coordinates')
        time_source, axes = 'lidar_sample_data_microseconds', 'nuscenes_global_z_up'
    else:
        raise ValueError(f'Unsupported v30 dataset {dataset!r}')
    result = dict(schema=SCHEMA, dataset=dataset, version=version, split_seed=seed,
        training_source=source, evaluation_source=evaluation,
        parameter_training_overlap=overlap, scenes=roles,
        coordinate_mode=coordinate_mode, coordinate_axes=axes, time_source=time_source,
        frame_stride=stride, full_endpoint_coverage=True)
    result['content_sha256'] = _digest(result)
    validate_dataset_manifest(result)
    return result


def validate_dataset_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
        raise ValueError('Expected a v30 dataset manifest')
    payload = {key: value for key, value in manifest.items() if key != 'content_sha256'}
    if manifest.get('content_sha256') != _digest(payload):
        raise ValueError('v30 dataset manifest content_sha256 mismatch')
    scenes = manifest.get('scenes', {})
    if set(scenes) != set(ROLES):
        raise ValueError('v30 dataset manifest must define all four scene roles')
    for role in ROLES:
        rows = scenes[role]
        if not isinstance(rows, list) or not rows or len(set(rows)) != len(rows):
            raise ValueError(f'Invalid or repeated scenes for {role}')
    if set(scenes['calibration']) & set(scenes['dev']):
        raise ValueError('Calibration and diagnostic scenes must be disjoint')
    if set(scenes['test']) & set().union(*(set(scenes[r]) for r in ROLES[:-1])):
        raise ValueError('Final evaluation scenes overlap training/fitting/diagnostics')
    overlap = bool(set(scenes['train']) & (set(scenes['calibration']) | set(scenes['dev'])))
    if overlap != manifest.get('parameter_training_overlap'):
        raise ValueError('parameter_training_overlap does not describe the scene roles')
    if manifest.get('full_endpoint_coverage') is not True:
        raise ValueError('v30 requires complete mechanism endpoint coverage')
    frame_stride(manifest.get('frame_stride', 0))
    if manifest.get('dataset') == 'kitti_mf':
        expected = {'train': [f'{i:04d}' for i in range(17)],
                    'calibration': ['0017'], 'dev': ['0018'], 'test': ['0019', '0020']}
        if scenes != expected or manifest.get('coordinate_mode') != 'sensor_relative':
            raise ValueError('KITTI manifest roles/coordinates differ from the registered protocol')
    elif manifest.get('dataset') == 'nuscenes_mf' and manifest.get('coordinate_mode') == 'global':
        counts = {'v1.0-mini': (8, 1, 1, 2), 'v1.0-trainval': (350, 17, 18, 150)}
        expected = counts.get(manifest.get('version'))
        if expected is None or tuple(len(scenes[role]) for role in ROLES) != expected:
            raise ValueError('nuScenes manifest scene counts/version differ from v30 protocol')
        if not set(scenes['calibration'] + scenes['dev']).issubset(scenes['train']):
            raise ValueError('nuScenes v30 fitting/diagnostics must disclose training overlap')
    else:
        raise ValueError('Unknown dataset coordinate contract')
    return manifest


def select_dataset_protocol(config, requested_role, scene_splits=None):
    manifest = build_dataset_manifest(config, scene_splits)
    role = str(_get(config, 'ct_protocol_role', '') or requested_role).lower()
    role = {'training': 'train', 'validation': 'val', 'valid': 'val',
            'evaluation': 'eval'}.get(role, role)
    if role == 'val':
        # KITTI 0018 serves development; 0019/20 are never training validation.
        role = 'dev' if manifest['dataset'] == 'kitti_mf' else 'test'
    elif role == 'eval':
        role = 'test'
    if role not in ROLES:
        raise ValueError(f'Unknown v30 dataset role {role!r}')
    return manifest, role, list(manifest['scenes'][role])


def validate_dataset_selection(manifest, role, scene_names):
    validate_dataset_manifest(manifest)
    if role not in ROLES or list(scene_names) != list(manifest['scenes'][role]):
        raise ValueError('Dataset scene selection does not match its v30 manifest role')


def apply_frame_stride(dataset, value):
    """均匀子采样保留原 annotation/frame/token/timestamp 及首帧，不重编号。"""
    stride = frame_stride(value)
    if stride != 1:
        dataset.tracklet_anno_list = [rows[::stride] for rows in dataset.tracklet_anno_list]
        dataset.tracklet_len_list = [len(rows) for rows in dataset.tracklet_anno_list]
    dataset.ct_frame_stride = stride


def validate_tracklet_timestamps(dataset):
    """只验证真实 annotation 序列；sampler 首部 padding 不在此表中。"""
    for tracklet_id, annotations in enumerate(dataset.tracklet_anno_list):
        previous = None
        for annotation in annotations:
            current = float(dataset._anno_timestamp(annotation))
            if not math.isfinite(current) or (previous is not None and current <= previous):
                raise ValueError(f'Non-finite or unordered physical timestamp in tracklet {tracklet_id}')
            previous = current
