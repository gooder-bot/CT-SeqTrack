"""v31 独立配置白名单；历史 formal 字段不能静默进入联合模型。"""

import hashlib
import json
from pathlib import Path

from utils.config import load_yaml_config
from .contracts import SCHEMA


DEFAULTS = dict(
    net_model='ctseqtrackv31', experiment_family='ct_seqtrack_v31',
    experiment_name='ct31_full_mini_car_scratch_seed42', v31_arm='full',
    dataset='nuscenes_mf', version='v1.0-mini', category_name='Car',
    path='/home/lishengjie/data/nuscenes-mini', ct_coordinate_mode='global',
    ct_partition_seed=42, ct_frame_stride=1, ct_pointcloud_cache_bytes=268435456,
    kitti_frame_period=.1, hist_num=3, point_sample_size=1024,
    bb_scale=1.25, bb_offset=2., time_scale=.5,
    v31_short_window=3, v31_long_window=8, v31_curriculum_epochs=10,
    batch_size=16, workers=4, seed=42, epoch=60, lr=.0001, wd=0.,
    lr_decay_step=20, lr_decay_rate=.1, check_val_every_n_epoch=5,
    trainer_devices=1, accelerator='auto', precision=32,
    dynamics_time_mode='true', dynamics_time_manifest=None,
    v31_evaluate_late3=True, ct_engineering_check=False,
    limit_train_batches=1., limit_val_batches=1.,
    checkpoint=None, init_checkpoint=None, test=False, log_dir=None, tag='v31',
    cfg=None, eval_checkpoint_epoch=None,
)


class V31Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    unknown = sorted(set(supplied) - set(DEFAULTS))
    if unknown:
        raise ValueError('v31 unknown/inactive configuration keys: ' + ', '.join(unknown))
    cfg = V31Config(DEFAULTS)
    cfg.update(supplied)
    if cfg.net_model != 'ctseqtrackv31' or cfg.experiment_family != 'ct_seqtrack_v31':
        raise ValueError('v31 model and experiment identity must match')
    if cfg.v31_arm not in ('b0', 'b1', 'b1_b2', 'full'):
        raise ValueError('v31_arm must be b0, b1, b1_b2 or full')
    if cfg.dataset not in ('nuscenes_mf', 'kitti_mf'):
        raise ValueError('v31 supports registered nuScenes and KITTI datasets')
    if cfg.dataset == 'nuscenes_mf':
        if cfg.version not in ('v1.0-mini', 'v1.0-trainval') or cfg.ct_coordinate_mode != 'global':
            raise ValueError('nuScenes requires registered version and global coordinates')
    elif cfg.version != 'kitti_tracking' or cfg.ct_coordinate_mode != 'sensor_relative':
        raise ValueError('KITTI requires kitti_tracking/sensor_relative')
    if cfg.init_checkpoint is not None:
        raise ValueError('v31 scratch_only forbids --init_checkpoint')
    if cfg.test and not cfg.checkpoint:
        raise ValueError('v31 evaluation requires --checkpoint')
    if cfg.dynamics_time_mode not in ('true', 'fixed', 'shuffled'):
        raise ValueError('unknown dynamics_time_mode')
    if cfg.dynamics_time_mode == 'shuffled' and not cfg.dynamics_time_manifest:
        raise ValueError('shuffled time requires an explicit manifest')
    if cfg.trainer_devices != 1:
        raise ValueError('v31 serial per-branch state currently requires one device per run')
    for name in ('hist_num', 'point_sample_size', 'v31_short_window', 'v31_long_window',
                 'v31_curriculum_epochs', 'batch_size', 'epoch', 'lr_decay_step', 'ct_frame_stride'):
        value = cfg[name]
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if cfg.hist_num != 3 or cfg.precision != 32 or cfg.workers < 0:
        raise ValueError('v31 requires hist_num=3, FP32 and nonnegative workers')
    if cfg.time_scale <= 0 or cfg.lr <= 0 or cfg.bb_scale <= 0 or cfg.bb_offset < 0:
        raise ValueError('v31 time, optimization and crop scales must be valid')
    if not cfg.ct_engineering_check:
        fixed = dict(epoch=60, batch_size=16, workers=4, lr=.0001, wd=0.,
                     lr_decay_step=20, lr_decay_rate=.1, point_sample_size=1024,
                     v31_short_window=3, v31_long_window=8, v31_curriculum_epochs=10,
                     limit_train_batches=1., limit_val_batches=1.)
        bad = [key for key, value in fixed.items()
               if cfg[key] != value or (key.startswith('limit_') and isinstance(cfg[key], int))]
        if bad:
            raise ValueError('formal v31 budget mismatch; use ct_engineering_check for smoke: ' + ', '.join(bad))
    elif cfg.log_dir:
        root = Path(__file__).resolve().parents[2] / 'artifacts' / 'ct_checks'
        destination = Path(cfg.log_dir).resolve()
        if root.resolve() not in (destination, *destination.parents):
            raise ValueError('engineering outputs must be under artifacts/ct_checks')
    return cfg


def load_config(path, overrides=None):
    data = load_yaml_config(path)
    data.update(overrides or {})
    data['cfg'] = str(path)
    return normalize_config(data)


def config_identity(config):
    """迁移路径/日志/评测方式不改变权重身份；训练与时间语义均进入摘要。"""
    cfg = normalize_config(config)
    excluded = {'cfg', 'path', 'log_dir', 'tag', 'checkpoint', 'init_checkpoint',
                'test', 'eval_checkpoint_epoch', 'accelerator', 'v31_evaluate_late3'}
    payload = {key: value for key, value in cfg.items() if key not in excluded}
    # manifest 文件内容决定 time control，而不是可迁移的绝对文件名。
    manifest = payload.pop('dynamics_time_manifest', None)
    if manifest:
        payload['dynamics_time_manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    payload['schema'] = SCHEMA
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()
