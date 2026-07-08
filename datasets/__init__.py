from datasets import sampler, \
                    nuscenes_lidar_mf,  \
                    waymo_data_mf
                    



def get_dataset(config, type='train', **kwargs):
    if config.dataset == 'nuscenes_mf':
        virtual_rate_kwargs = {
            'virtual_rate_mode': getattr(config, 'virtual_rate_mode', 'none'),
            'virtual_rate_gap_pattern': getattr(config, 'virtual_rate_gap_pattern', [1, 1, 2, 4]),
            'virtual_rate_stride': getattr(config, 'virtual_rate_stride', 2),
            'virtual_rate_drop_every': getattr(config, 'virtual_rate_drop_every', 5),
            'virtual_rate_drop_prob': getattr(config, 'virtual_rate_drop_prob', 0.0),
            'virtual_rate_seed': getattr(config, 'virtual_rate_seed', 42),
            'virtual_rate_max_gap': getattr(config, 'virtual_rate_max_gap', 5),
            'virtual_rate_manifest': getattr(config, 'virtual_rate_manifest', ''),
            'virtual_rate_keep_first': getattr(config, 'virtual_rate_keep_first', True),
            'virtual_rate_keep_last': getattr(config, 'virtual_rate_keep_last', True),
            'virtual_rate_min_tracklet_len': getattr(config, 'virtual_rate_min_tracklet_len', 0),
            'virtual_rate_burst_keep_lengths': getattr(config, 'virtual_rate_burst_keep_lengths', [3, 2, 3]),
            'virtual_rate_burst_skip_lengths': getattr(config, 'virtual_rate_burst_skip_lengths', [2, 3, 3]),
        }
        data = nuscenes_lidar_mf.NuScenesMFDataset(path=config.path,
                                             split=kwargs.get('split', 'train_track'),
                                             category_name=config.category_name,
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             # can only use keyframes for training
                                             preloading=config.preloading,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             min_points=1 if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1,
                                             hist_num=config.hist_num,
                                             **virtual_rate_kwargs)
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
