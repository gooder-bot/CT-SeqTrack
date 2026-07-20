from datasets import sampler, \
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
                                             min_points=1 if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1,
                                             hist_num=config.hist_num,
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
        data = None
  
    if type.lower() == 'train_motion_mf':
        return sampler.MotionTrackingSamplerMF(dataset=data,
                                             config=config)
    
    else:
        return sampler.TestTrackingSampler(dataset=data, config=config)
