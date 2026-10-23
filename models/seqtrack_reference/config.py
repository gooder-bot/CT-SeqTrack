"""独立 SeqTrack 工程配置；不继承已归档的 CT 算法配方。"""
import hashlib
import json
import math
from pathlib import Path

from utils.config import load_yaml_config
from utils.time_config import TIME_ENGINEERING_KEYS
from .identity import model_schema
from .protocol import CANONICAL, protocol_identity


DEFAULTS = dict(
    net_model='seqtrack_reference', experiment_family='seqtrack_reference',
    experiment_name='seqtrack_reference_engineering',
    dataset='nuscenes_mf', version='v1.0-mini', category_name='Car',
    path='/home/lishengjie/data/nuscenes-mini', ct_coordinate_mode='global',
    ct_partition_seed=42, ct_frame_stride=1, ct_pointcloud_cache_bytes=268435456,
    kitti_frame_period=.1, hist_num=3, point_sample_size=1024,
    bb_scale=1.25, bb_offset=2.,
    batch_size=2, workers=0, seed=42, epoch=2, lr=.0001, wd=0.,
    lr_decay_step=20, lr_decay_rate=.1, check_val_every_n_epoch=1,
    lr_schedule='step', lr_milestones=(), lr_warmup_steps=0,
    trainer_devices=1, accelerator='auto', precision=32,
    dynamics_time_mode='true', dynamics_time_manifest=None,
    evaluate_late3=False, ct_engineering_check=True,
    limit_train_batches=1., limit_val_batches=1.,
    checkpoint=None, init_checkpoint=None, test=False, log_dir=None, tag='reference_engineering',
    cfg=None, eval_checkpoint_epoch=None,
)


class ReferenceConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    if set(supplied).intersection(TIME_ENGINEERING_KEYS):
        raise ValueError('reference retains its original pseudo-time protocol; role-specific dynamics controls are not supported')
    unknown = sorted(set(supplied) - set(DEFAULTS))
    if unknown:
        raise ValueError('reference unknown/inactive configuration keys: ' + ', '.join(unknown))
    cfg = ReferenceConfig(DEFAULTS)
    cfg.update(supplied)
    if cfg.net_model != 'seqtrack_reference' or cfg.experiment_family != 'seqtrack_reference':
        raise ValueError('reference model and experiment identity must match')
    if cfg.ct_engineering_check is not True:
        raise ValueError('current reference entry is engineering-only; reproduce historical formal runs with frozen source')
    if cfg.dataset not in ('nuscenes_mf', 'kitti_mf'):
        raise ValueError('reference supports nuScenes and KITTI raw datasets')
    if cfg.dataset == 'nuscenes_mf':
        if cfg.version not in ('v1.0-mini', 'v1.0-trainval') or cfg.ct_coordinate_mode != 'global':
            raise ValueError('nuScenes requires a supported version and global coordinates')
    elif cfg.version != 'kitti_tracking' or cfg.ct_coordinate_mode != 'sensor_relative':
        raise ValueError('KITTI requires kitti_tracking/sensor_relative')
    if cfg.init_checkpoint is not None:
        raise ValueError('reference scratch_only forbids --init_checkpoint')
    if cfg.test and not cfg.checkpoint:
        raise ValueError('reference evaluation requires --checkpoint')
    if cfg.dynamics_time_mode != 'true' or cfg.dynamics_time_manifest is not None:
        raise ValueError('reference retains its original pseudo-time protocol')
    if cfg.trainer_devices != 1 or cfg.precision != 32:
        raise ValueError('reference requires one device and FP32')
    if cfg.accelerator not in ('auto', 'cpu', 'gpu'):
        raise ValueError('unsupported accelerator')
    for name in ('hist_num', 'point_sample_size', 'batch_size', 'epoch', 'lr_decay_step',
                 'check_val_every_n_epoch', 'ct_frame_stride'):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(name + ' must be a positive integer')
    for name in ('workers', 'seed', 'ct_partition_seed', 'ct_pointcloud_cache_bytes', 'lr_warmup_steps'):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    for name in ('hist_num', 'point_sample_size', 'bb_scale', 'bb_offset'):
        if isinstance(cfg[name], bool) or cfg[name] != CANONICAL[name]:
            raise ValueError('fixed reference algorithm setting mismatch: ' + name)
    if (not math.isfinite(cfg.lr) or cfg.lr <= 0 or not math.isfinite(cfg.wd) or cfg.wd < 0
            or not math.isfinite(cfg.lr_decay_rate) or not 0 < cfg.lr_decay_rate < 1
            or not math.isfinite(cfg.kitti_frame_period) or cfg.kitti_frame_period <= 0):
        raise ValueError('reference optimizer/time settings must be finite and valid')
    if cfg.lr_schedule not in ('step', 'multistep') or not isinstance(cfg.lr_milestones, (list, tuple)):
        raise ValueError('reference supports step or multistep schedules')
    milestones = cfg.lr_milestones
    if (any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in milestones)
            or list(milestones) != sorted(set(milestones))):
        raise ValueError('lr_milestones must be strictly increasing positive integers')
    cfg.lr_milestones = list(milestones)
    if (cfg.lr_schedule == 'step' and (milestones or cfg.lr_warmup_steps)) or (cfg.lr_schedule == 'multistep' and not milestones):
        raise ValueError('step uses no milestones/warmup; multistep requires milestones')
    for name in ('test', 'evaluate_late3'):
        if type(cfg[name]) is not bool:
            raise ValueError(name + ' must be bool')
    for name in ('limit_train_batches', 'limit_val_batches'):
        value = cfg[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (name == 'limit_train_batches' and value == 0)):
            raise ValueError(name + ' must be valid (validation may be disabled with zero)')
        if isinstance(value, float) and value > 1:
            raise ValueError(name + ' fraction must be <= 1; use an integer for a batch count')
    if (not cfg.test and cfg.evaluate_late3
            and not (type(cfg.limit_train_batches) is float and cfg.limit_train_batches == 1.)):
        raise ValueError('partial/integer train limits require --no_late3; automatic evaluation requires complete epochs')
    if cfg.log_dir:
        root = (Path(__file__).resolve().parents[2] / 'artifacts' / 'ct_checks').resolve()
        destination = Path(cfg.log_dir).resolve()
        if root not in (destination, *destination.parents):
            raise ValueError('engineering outputs must be under artifacts/ct_checks')
    return cfg


def load_config(path, overrides=None):
    data = load_yaml_config(path)
    data.update(overrides or {})
    data['cfg'] = str(path)
    return normalize_config(data)


def config_identity(config):
    cfg = normalize_config(config)
    excluded = {'cfg', 'path', 'log_dir', 'tag', 'checkpoint', 'init_checkpoint',
                'test', 'eval_checkpoint_epoch', 'accelerator', 'evaluate_late3'}
    payload = {key: value for key, value in cfg.items() if key not in excluded}
    payload.update(schema=model_schema(cfg), evaluation_rng_policy='per_checkpoint_seed_v1',
                   reference_protocol=protocol_identity())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()
