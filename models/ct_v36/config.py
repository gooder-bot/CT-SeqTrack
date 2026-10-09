"""v36 独立配置身份；正式预算仅 A/C/D/E/F 五组。"""
import hashlib
import json
import math
from pathlib import Path
from models.ct_v31.config import DEFAULTS as OLD_DEFAULTS, V31Config
from utils.config import load_yaml_config
from utils.execution_profile import validate_execution_profile
from .contracts import SCHEMA

DEFAULTS = dict(OLD_DEFAULTS, net_model='ctseqtrackv36', experiment_family='ct_seqtrack_v36',
    experiment_name='ct36_A_full_mini_car_scratch_seed42', tag='A_full_mini_car_seed42',
    v36_arm='A', v31_arm='full', v31_short_window=4, lr=.00015,
    lr_schedule='multistep', lr_milestones=[20, 50], v36_diagnostics=True,
    v36_initialization_policy='semantic_components_v1',
    v36_sampling_policy='endpoint_streams_v1', v36_state_policy='independent_measurement_v1',
    v36_schedule_policy='continuous_lanes_v1')


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(DEFAULTS) - {'execution_profile'}
    if unknown:
        raise ValueError('v36 unknown/inactive keys: ' + ', '.join(sorted(unknown)))
    cfg = V31Config(DEFAULTS)
    cfg.update(supplied)
    validate_execution_profile(cfg)
    if cfg.net_model != 'ctseqtrackv36' or cfg.experiment_family != 'ct_seqtrack_v36':
        raise ValueError('v36 model/family identity mismatch')
    if cfg.v36_arm not in ('A', 'C', 'D', 'E', 'F') or cfg.v31_arm != 'full':
        raise ValueError('v36 registered arms are A/C/D/E/F; legacy arm key must be full')
    if cfg.init_checkpoint is not None:
        raise ValueError('scratch_only forbids --init_checkpoint')
    if cfg.test and not cfg.checkpoint:
        raise ValueError('evaluation requires --checkpoint')
    if cfg.v31_temporal_backend not in ('cfc', 'gru'):
        raise ValueError('unsupported temporal backend')
    if cfg.dynamics_time_mode not in ('true', 'fixed', 'shuffled'):
        raise ValueError('unsupported time mode')
    if cfg.dynamics_time_mode == 'shuffled' and not cfg.dynamics_time_manifest:
        raise ValueError('shuffled time requires manifest')
    if cfg.dataset not in ('nuscenes_mf', 'kitti_mf'):
        raise ValueError('unsupported dataset')
    if cfg.dataset == 'nuscenes_mf' and (cfg.version not in ('v1.0-mini', 'v1.0-trainval') or cfg.ct_coordinate_mode != 'global'):
        raise ValueError('nuScenes requires registered global coordinates')
    if cfg.dataset == 'kitti_mf' and (cfg.version != 'kitti_tracking' or cfg.ct_coordinate_mode != 'sensor_relative'):
        raise ValueError('KITTI requires sensor_relative coordinates')
    for key in ('batch_size', 'epoch', 'hist_num', 'point_sample_size', 'v31_short_window',
                'v31_long_window', 'v31_curriculum_epochs', 'ct_frame_stride', 'check_val_every_n_epoch'):
        if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or cfg[key] <= 0:
            raise ValueError('positive integer required: ' + key)
    if cfg.hist_num != 3 or cfg.precision != 32 or cfg.trainer_devices != 1 or cfg.workers < 0:
        raise ValueError('v36 requires three history frames, FP32, one device and valid workers')
    for key in ('time_scale', 'lr', 'bb_scale'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('positive finite scale required: ' + key)
    for key in ('bb_offset', 'v32_seed_translation', 'v32_seed_yaw_degrees', 'wd'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('nonnegative finite scale required: ' + key)
    if (isinstance(cfg.v32_reserve_windows, bool) or int(cfg.v32_reserve_windows) != cfg.v32_reserve_windows
            or cfg.v32_reserve_windows < 0):
        raise ValueError('reserve_windows must be a nonnegative integer')
    if cfg.lr_schedule != 'multistep' or list(cfg.lr_milestones) != [20, 50] or cfg.lr_warmup_steps != 0:
        raise ValueError('v36 uses fixed 20/50 multistep without warmup')
    cfg.lr_milestones = list(cfg.lr_milestones)
    for key in ('v36_initialization_policy', 'v36_sampling_policy', 'v36_state_policy', 'v36_schedule_policy'):
        if cfg[key] != DEFAULTS[key]:
            raise ValueError('v36 policy identity mismatch: ' + key)
    if type(cfg.v36_diagnostics) is not bool:
        raise ValueError('v36_diagnostics must be bool')
    if not cfg.ct_engineering_check:
        fixed = dict(epoch=60, batch_size=16, workers=4, seed=42, lr=.00015, wd=0.,
            lr_decay_rate=.1, point_sample_size=1024, v31_short_window=4, v31_long_window=8,
            v31_curriculum_epochs=10, v32_reserve_windows=112, v32_seed_translation=.3,
            v32_seed_yaw_degrees=1.5, dataset='nuscenes_mf', version='v1.0-mini',
            category_name='Car', ct_partition_seed=42, ct_frame_stride=1,
            time_scale=.5, bb_scale=1.25, bb_offset=2., check_val_every_n_epoch=5,
            v31_temporal_backend='cfc',
            dynamics_time_mode='true', v31_evaluate_late3=True, v36_diagnostics=True,
            limit_train_batches=1., limit_val_batches=1.)
        bad = [k for k, v in fixed.items() if cfg[k] != v or (k.startswith('limit_') and isinstance(cfg[k], int))]
        if bad:
            raise ValueError('unregistered formal v36 recipe: ' + ', '.join(bad))
    elif cfg.log_dir:
        root = Path(__file__).resolve().parents[2] / 'artifacts' / 'ct_checks'
        dest = Path(cfg.log_dir).resolve()
        if root.resolve() not in (dest, *dest.parents):
            raise ValueError('engineering outputs must be under artifacts/ct_checks')
    return cfg


def load_config(path, overrides=None):
    data = load_yaml_config(path)
    data.update(overrides or {})
    data['cfg'] = str(path)
    return normalize_config(data)


def config_identity(config):
    cfg = normalize_config(config)
    excluded = {'cfg', 'path', 'log_dir', 'tag', 'checkpoint', 'init_checkpoint', 'test',
                'eval_checkpoint_epoch', 'accelerator', 'v31_evaluate_late3'}
    payload = {k: v for k, v in cfg.items() if k not in excluded}
    if payload.get('execution_profile') == 'optimized_fp32':
        payload.pop('execution_profile')
    manifest = payload.pop('dynamics_time_manifest', None)
    if manifest:
        payload['dynamics_time_manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    payload.update(schema=SCHEMA, evaluation_rng_policy='endpoint_eval0_v1')
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
