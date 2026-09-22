"""原始数据集工厂；导入包不加载 SDK、旧 sampler 或模型。"""
from .protocol import select_dataset_protocol
from .protocol_utils import resolve_virtual_rate_kwargs, resolve_dynamics_time_kwargs
from .cache import DEFAULT_POINTCLOUD_CACHE_BYTES


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def get_raw_dataset(config, role):
    """直接返回 v31 原始帧；裁剪和状态由 BatchBuilder 管理。"""
    manifest, role, scenes = select_dataset_protocol(config, role)
    common = dict(path=_get(config, 'path'), category_name=_get(config, 'category_name'),
        version=_get(config, 'version'), preloading=False, preload_offset=-1, hist_num=3,
        ct_enable_v30=True, ct_frame_stride=_get(config, 'ct_frame_stride', 1),
        ct_pointcloud_cache_bytes=_get(config, 'ct_pointcloud_cache_bytes', DEFAULT_POINTCLOUD_CACHE_BYTES),
        ct_scene_manifest=manifest, ct_scene_names=scenes, ct_scene_role=role,
        coordinate_mode=manifest['coordinate_mode'],
        **resolve_virtual_rate_kwargs(config, role), **resolve_dynamics_time_kwargs(config, role))
    if manifest['dataset'] == 'nuscenes_mf':
        from .nuscenes_lidar_mf import NuScenesMFDataset
        return NuScenesMFDataset(
            split=manifest['training_source'] if role != 'test' else manifest['evaluation_source'],
            key_frame_only=True, min_points=1 if role != 'train' else -1, **common)
    from .kitti_mf import KITTIMFDataset
    return KITTIMFDataset(split='train', frame_period=_get(config, 'kitti_frame_period', .1),
        kitti_hv_interval=1, scene_ids=None, allow_missing_pointcloud=False, **common)
