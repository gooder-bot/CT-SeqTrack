from datasets import sampler, \
                    kitti_mf, \
                    nuscenes_lidar_mf,  \
                    waymo_data_mf
from datasets.protocol_utils import (
    normalize_protocol_role,
    resolve_dynamics_time_kwargs,
    resolve_virtual_rate_kwargs,
)
                    



def get_dataset(config, type='train', **kwargs):
    if config.dataset == 'nuscenes_mf':
        split = kwargs.get('split', 'train_track')
        role = kwargs.get('protocol_role')
        if role is None:
            role = 'train' if type.lower().startswith('train') else 'eval'
        role = normalize_protocol_role(role)
        scene_kwargs = {}
        if bool(getattr(config, 'ct_enable_v27', False)):
            from nuscenes.utils.splits import create_splits_scenes
            from utils.v27_protocol import select_scene_protocol
            manifest, role, scene_names = select_scene_protocol(
                config, role, create_splits_scenes())
            split = (manifest['training_source'] if role != 'test'
                     else manifest['evaluation_source'])
            scene_kwargs = dict(ct_scene_manifest=manifest,
                                ct_scene_names=scene_names, ct_scene_role=role)
        virtual_rate_kwargs = resolve_virtual_rate_kwargs(config, role)
        dynamics_time_kwargs = resolve_dynamics_time_kwargs(config, role)
        data = nuscenes_lidar_mf.NuScenesMFDataset(path=config.path,
                                             split=split,
                                             category_name=config.category_name,
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             # can only use keyframes for training
                                             preloading=config.preloading,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             min_points=(1 if role != 'train' else -1)
                                             if scene_kwargs else (1 if split in
                                                 [config.val_split, config.test_split] else -1),
                                             hist_num=config.hist_num,
                                             **scene_kwargs,
                                             **virtual_rate_kwargs,
                                             **dynamics_time_kwargs)
    elif config.dataset in ('kitti_mf', 'kitti'):
        role = kwargs.get('protocol_role')
        if role is None:
            role = 'train' if type.lower().startswith('train') else 'eval'
        role = normalize_protocol_role(role)
        virtual_rate_kwargs = resolve_virtual_rate_kwargs(config, role)
        dynamics_time_kwargs = resolve_dynamics_time_kwargs(config, role)
        data = kitti_mf.KITTIMFDataset(
            path=config.path,
            split=kwargs.get('split', 'train'),
            category_name=config.category_name,
            version=getattr(config, 'version', 'kitti_tracking'),
            preloading=config.preloading,
            preload_offset=(
                config.preload_offset if type != 'test' else -1),
            hist_num=config.hist_num,
            frame_period=getattr(
                config, 'kitti_frame_period',
                getattr(config, 'default_time_step', 0.1)),
            kitti_hv_interval=getattr(
                config, f'{role}_kitti_hv_interval',
                getattr(config, 'kitti_hv_interval', 1)),
            scene_ids=getattr(config, 'kitti_scene_ids', None),
            allow_missing_pointcloud=getattr(
                config, 'kitti_allow_missing_pointcloud', False),
            **virtual_rate_kwargs,
            **dynamics_time_kwargs)
    elif config.dataset == 'waymo_mf':
        data = waymo_data_mf.WaymoDataset(path=config.path,
                                       split=kwargs.get('split', 'train'),
                                       category_name=config.category_name,
                                       preloading=config.preloading,
                                       preload_offset=config.preload_offset,
                                       tiny=config.tiny,
                                       hist_num = config.hist_num)
    else:
        raise ValueError(
            f"Unsupported dataset {config.dataset!r}; expected one of "
            "'nuscenes_mf', 'kitti_mf', or 'waymo_mf'.")
  
    if type.lower() == 'train_motion_mf':
        return sampler.MotionTrackingSamplerMF(dataset=data,
                                             config=config)
    
    else:
        return sampler.TestTrackingSampler(dataset=data, config=config)
