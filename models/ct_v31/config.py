"""v33 综合 B0 配置与权重身份；旧实验通过冻结版本复现。"""

import hashlib
import json
from pathlib import Path

from utils.config import load_yaml_config
from .contracts import SCHEMA


DEFAULTS = dict(
    net_model='ctseqtrackv33', experiment_family='ct_seqtrack_v33',
    experiment_name='ct33_b0_original_mini_car_scratch_seed42', v31_arm='b0',
    v31_temporal_backend='cfc',
    dataset='nuscenes_mf', version='v1.0-mini', category_name='Car',
    path='/home/lishengjie/data/nuscenes-mini', ct_coordinate_mode='global',
    ct_partition_seed=42, ct_frame_stride=1, ct_pointcloud_cache_bytes=268435456,
    kitti_frame_period=.1, hist_num=3, point_sample_size=1024,
    bb_scale=1.25, bb_offset=2., time_scale=.5,
    v31_short_window=3, v31_long_window=8, v31_curriculum_epochs=10,
    v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
    batch_size=16, workers=4, seed=42, epoch=60, lr=.0001, wd=0.,
    lr_decay_step=20, lr_decay_rate=.1, check_val_every_n_epoch=5,
    lr_schedule='step', lr_milestones=(), lr_warmup_steps=0,
    trainer_devices=1, accelerator='auto', precision=32,
    dynamics_time_mode='true', dynamics_time_manifest=None,
    v31_evaluate_late3=True, ct_engineering_check=False,
    limit_train_batches=1., limit_val_batches=1.,
    checkpoint=None, init_checkpoint=None, test=False, log_dir=None, tag='v33',
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
        raise ValueError('v33 unknown/inactive configuration keys: ' + ', '.join(unknown))
    cfg = V31Config(DEFAULTS)
    cfg.update(supplied)
    if (cfg.net_model not in ('ctseqtrackv33', 'seqtrack_reference')
            or cfg.experiment_family != 'ct_seqtrack_v33'):
        raise ValueError('v33 model and experiment identity must match; '
                         'reproduce frozen v32 with Git ddcb1a1, '
                         'or reproduce frozen v31 with Git b1d886e and its original configs')
    if cfg.net_model == 'seqtrack_reference' and cfg.v31_arm != 'b0':
        raise ValueError('SeqTrack reference requires v31_arm=b0')
    if cfg.net_model == 'seqtrack_reference' and cfg.dynamics_time_mode != 'true':
        raise ValueError('SeqTrack reference retains its original pseudo-time protocol')
    if cfg.v31_arm not in ('b0', 'b1', 'b1_b2', 'full'):
        raise ValueError('v31_arm must be b0, b1, b1_b2 or full')
    if cfg.v31_temporal_backend not in ('cfc', 'gru'):
        raise ValueError('v31_temporal_backend must be cfc or gru')
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
    if (isinstance(cfg.v32_reserve_windows, bool)
            or int(cfg.v32_reserve_windows) != cfg.v32_reserve_windows
            or cfg.v32_reserve_windows < 0):
        raise ValueError('v32_reserve_windows must be a nonnegative integer')
    if (isinstance(cfg.lr_warmup_steps, bool) or not isinstance(cfg.lr_warmup_steps, int)
            or cfg.lr_warmup_steps < 0):
        raise ValueError('lr_warmup_steps must be a nonnegative integer')
    import math
    if (cfg.lr_schedule not in ('step', 'multistep')
            or not isinstance(cfg.lr_milestones, (list, tuple))):
        raise ValueError('v33 requires step or multistep and a list of lr_milestones')
    if cfg.lr_warmup_steps > 0 and cfg.lr_schedule != 'multistep':
        raise ValueError('positive lr_warmup_steps requires lr_schedule=multistep')
    milestones = cfg.lr_milestones
    if (any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in milestones) or list(milestones) != sorted(set(milestones))):
        raise ValueError('lr_milestones must be strictly increasing positive integers')
    cfg.lr_milestones = list(milestones)
    if (cfg.lr_schedule == 'step' and milestones) or (cfg.lr_schedule == 'multistep' and not milestones):
        raise ValueError('step uses no lr_milestones; multistep requires lr_milestones')
    if not math.isfinite(cfg.lr) or not math.isfinite(cfg.lr_decay_rate) or not 0 < cfg.lr_decay_rate < 1:
        raise ValueError('learning rate must be finite and decay rate must be in (0,1)')
    if any(not math.isfinite(float(cfg[key])) or cfg[key] < 0
           for key in ('v32_seed_translation', 'v32_seed_yaw_degrees')):
        raise ValueError('v32 seed perturbation bounds must be finite and nonnegative')
    if not cfg.ct_engineering_check:
        fixed = dict(epoch=60, batch_size=16, workers=4, wd=0.,
                     lr_decay_step=20, lr_decay_rate=.1, point_sample_size=1024,
                     v31_short_window=3, v31_long_window=8, v31_curriculum_epochs=10,
                     v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
                     limit_train_batches=1., limit_val_batches=1.)
        bad = [key for key, value in fixed.items()
               if cfg[key] != value or (key.startswith('limit_') and isinstance(cfg[key], int))]
        if bad:
            raise ValueError('formal v33 budget mismatch; use ct_engineering_check for smoke: ' + ', '.join(bad))
        recipe = (cfg.lr, cfg.lr_schedule, tuple(cfg.lr_milestones), cfg.lr_warmup_steps)
        registered = {(.0001, 'step', (), 0)}
        if cfg.net_model == 'ctseqtrackv33' and cfg.v31_arm == 'b0':
            registered.update({(.0001, 'multistep', (20, 50), 0), (.00005, 'multistep', (20, 50), 0),
                               (.00015, 'multistep', (20, 50), 0), (.0003, 'multistep', (20, 50), 2000)})
        if recipe not in registered:
            raise ValueError('unregistered formal v33 learning-rate recipe')
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
    # 未启用 warmup 的旧 R/A/B/C/D 保留原摘要；正数进入新配方身份。
    if cfg.lr_warmup_steps == 0:
        payload.pop('lr_warmup_steps')
    # manifest 文件内容决定 time control，而不是可迁移的绝对文件名。
    manifest = payload.pop('dynamics_time_manifest', None)
    if manifest:
        payload['dynamics_time_manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    payload['schema'] = SCHEMA
    payload['evaluation_rng_policy'] = 'per_checkpoint_seed_v1'
    if cfg.net_model == 'seqtrack_reference':
        from models.seqtrack_reference.protocol import protocol_identity
        payload['reference_protocol'] = protocol_identity()
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()
