"""v38 三组固定 scratch 配方；全部复用无 B2 的强 concat B1。"""
import hashlib
import json
import math
from pathlib import Path

from utils.config import load_yaml_config
from utils.execution_lab import execution_scope
from utils.execution_profile import validate_execution_profile
from utils.time_config import engineering_time_defaults, validate_time_config, hash_time_manifest_paths
from .identity import SCHEMA, EXECUTION_REVISION, REGISTRATION_FILE, REGISTRATION_SCHEMA


# 数值和字段名沿用已验证的强 B1；定义由本版本持有，不再适配旧模型。
SHARED_DEFAULTS = dict(
    v31_temporal_backend='cfc',
    dataset='nuscenes_mf', version='v1.0-mini', category_name='Car',
    path='/home/lishengjie/data/nuscenes-mini', ct_coordinate_mode='global',
    ct_partition_seed=42, ct_frame_stride=1, ct_pointcloud_cache_bytes=268435456,
    kitti_frame_period=.1, hist_num=3, point_sample_size=1024,
    bb_scale=1.25, bb_offset=2., time_scale=.5,
    v31_long_window=8, v31_curriculum_epochs=10,
    v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
    batch_size=16, workers=4, seed=42, epoch=60, wd=0.,
    lr_decay_step=20, lr_decay_rate=.1, lr_warmup_steps=0,
    trainer_devices=1, accelerator='auto', precision=32,
    dynamics_time_mode='true', dynamics_time_manifest=None,
    v31_evaluate_late3=True, ct_engineering_check=False,
    limit_train_batches=1., limit_val_batches=1.,
    checkpoint=None, init_checkpoint=None, test=False, log_dir=None,
    cfg=None, eval_checkpoint_epoch=None)
V35_DEFAULTS = dict(v35_seed_policy='initial_exact_other_perturbed_v1',
    v35_history_descriptor_dim=33, v35_local_neighbors=16, v35_local_radius=1.0,
    v35_local_input_dim=69, v35_train_diagnostics=True)


class TrackingConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


RECIPES = {
    'b1_standard': dict(arm='b1', lr_milestones=[20, 50]),
    'b1_hold': dict(arm='b1', lr_milestones=[20]),
    'full_standard': dict(arm='full', lr_milestones=[20, 50]),
}
DEFAULTS = dict(SHARED_DEFAULTS, **V35_DEFAULTS,
    net_model='ctseqtrackv38', experiment_family='ct_seqtrack_v38',
    experiment_name='ct38_full_standard_mini_car_scratch_seed42',
    tag='v38_full_standard_mini_car_seed42', v31_arm='b1', v38_arm='full',
    v38_training_recipe='full_standard', v31_short_window=4,
    lr=.00015, lr_schedule='multistep', lr_milestones=[20, 50],
    check_val_every_n_epoch=1, execution_profile='optimized_fp32', devices=None,
    v38_diagnostics=True, v38_execution_revision=EXECUTION_REVISION,
    v38_selection_policy='q0_prior_trusted_cv_gain_v1',
    v38_state_policy='selected_q0_trusted_only_v1',
    v38_main_policy='strong_concat_b1_v1',
    v38_activation_policy='active_from_epoch1_v1',
    v38_gain_margin=0., v38_trust_threshold=.5, v38_min_support=3,
    v38_duplicate_xyz_tolerance=1e-4, v38_duplicate_yaw_tolerance=1e-5)
POLICY_KEYS = tuple(key for key in DEFAULTS if key.startswith('v38_') and key.endswith('_policy'))
FIXED_B3_KEYS = ('v38_gain_margin', 'v38_trust_threshold', 'v38_min_support',
                'v38_duplicate_xyz_tolerance', 'v38_duplicate_yaw_tolerance')


def recipe_identity(config):
    name = config.get('v38_training_recipe')
    if not isinstance(name, str) or name not in RECIPES:
        raise ValueError('unknown v38 training recipe: ' + str(name))
    options = list(config.get('v38_engineering_options', ('unique_mask',)))
    projection = 'factorized' if 'factorized_projection' in options else 'concat'
    return dict(recipe=name, implementation_id=f'v38_clean1_{projection}_b1_selective_update_v1',
        execution_revision=EXECUTION_REVISION,
        projection=projection, options=options, prior_read_applicable=False,
        arm=RECIPES[name]['arm'], lr_milestones=list(RECIPES[name]['lr_milestones']),
        active_from_epoch=1, candidates=['q0', 'prior', 'trusted_propagation'],
        duplicate_xyz_tolerance=1e-4, duplicate_yaw_tolerance=1e-5)


def recipe_scope(config):
    recipe_identity(config)
    # 新路径不继承环境/外层工程变量；共同 B1 仅启用已验证的唯一点去重。
    return execution_scope(config.get('v38_engineering_options', ('unique_mask',)),
                           registered_recipe=True)


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    if any(not isinstance(key, str) for key in supplied):
        raise ValueError('configuration keys must be strings')
    defaults = dict(DEFAULTS)
    defaults.update(engineering_time_defaults(supplied))
    if 'v38_engineering_options' in supplied:
        options = supplied['v38_engineering_options']
        if (supplied.get('ct_engineering_check') is not True or not isinstance(options, (list, tuple))
                or not all(isinstance(value, str) for value in options)
                or len(options) != len(set(options)) or 'unique_mask' not in options
                or set(options) - {'unique_mask', 'factorized_projection'}):
            raise ValueError('v38_engineering_options requires engineering mode and unique_mask/optional factorized_projection')
        defaults['v38_engineering_options'] = ['unique_mask']
    if supplied.get('lr_schedule') == 'piecewise':
        defaults['lr_stage_values'] = ()
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError('v38 unknown/inactive configuration keys: ' + ', '.join(sorted(unknown)))
    cfg = TrackingConfig(defaults)
    cfg.update(supplied)
    if 'v38_arm' in supplied and 'v38_training_recipe' not in supplied:
        cfg.v38_training_recipe = 'b1_standard' if cfg.v38_arm == 'b1' else 'full_standard'
    elif 'v38_training_recipe' in supplied and 'v38_arm' not in supplied:
        if isinstance(cfg.v38_training_recipe, str) and cfg.v38_training_recipe in RECIPES:
            cfg.v38_arm = RECIPES[cfg.v38_training_recipe]['arm']
    spec = RECIPES.get(cfg.v38_training_recipe) if isinstance(cfg.v38_training_recipe, str) else None
    if spec is None or cfg.v38_arm != spec['arm']:
        raise ValueError('v38 arm/recipe identity mismatch')
    if 'lr_milestones' not in supplied:
        cfg.lr_milestones = list(spec['lr_milestones'])
    if cfg.net_model != 'ctseqtrackv38' or cfg.experiment_family != 'ct_seqtrack_v38':
        raise ValueError('v38 model/family identity mismatch')
    if cfg.v38_execution_revision != EXECUTION_REVISION:
        raise ValueError('v38 execution revision mismatch; use the original frozen source')
    if cfg.v31_arm != 'b1':
        raise ValueError('all v38 arms must use the strong B1 observation and prior')
    validate_execution_profile(cfg)
    if cfg.execution_profile != 'optimized_fp32' and not cfg.ct_engineering_check:
        raise ValueError('v38 registers optimized_fp32 concat only')
    if cfg.init_checkpoint is not None:
        raise ValueError('v38 scratch_only forbids --init_checkpoint')
    if cfg.test and not cfg.checkpoint:
        raise ValueError('v38 evaluation requires --checkpoint')
    for key in ('ct_engineering_check', 'v38_diagnostics', 'v35_train_diagnostics', 'v31_evaluate_late3', 'test'):
        if type(cfg[key]) is not bool:
            raise ValueError(key + ' must be bool')
    if cfg.devices is not None and (type(cfg.devices) is not int or cfg.devices < 0):
        raise ValueError('devices selects one nonnegative physical GPU ID')
    for key in POLICY_KEYS + FIXED_B3_KEYS:
        if type(cfg[key]) is not type(DEFAULTS[key]) or cfg[key] != DEFAULTS[key]:
            raise ValueError('v38 fixed policy identity mismatch: ' + key)
    _validate_shared_configuration(cfg)
    validate_time_config(cfg)
    if (not cfg.test and cfg.v31_evaluate_late3
            and not (type(cfg.limit_train_batches) is float and cfg.limit_train_batches == 1.)):
        raise ValueError('partial/integer train limits require --no_late3; automatic evaluation requires complete epochs')
    if not cfg.ct_engineering_check and (cfg.lr_schedule != 'multistep' or list(cfg.lr_milestones) != spec['lr_milestones']
            or cfg.lr_warmup_steps != 0):
        raise ValueError('v38 requires its registered 20/50 or 20-only schedule without warmup')
    cfg.lr_milestones = list(cfg.lr_milestones)
    for key in ('time_scale', 'lr', 'bb_scale', 'bb_offset', 'wd'):
        if not math.isfinite(float(cfg[key])):
            raise ValueError('nonfinite v38 configuration value: ' + key)
    if type(cfg.workers) is not int or cfg.workers < 0:
        raise ValueError('workers must be a nonnegative integer')
    if type(cfg.check_val_every_n_epoch) is not int or cfg.check_val_every_n_epoch < 1:
        raise ValueError('check_val_every_n_epoch must be a positive integer')
    if not cfg.ct_engineering_check:
        fixed = dict(epoch=60, batch_size=16, workers=4, seed=42, lr=.00015, wd=0.,
            lr_decay_step=20, lr_decay_rate=.1, point_sample_size=1024,
            v31_short_window=4, v31_long_window=8, v31_curriculum_epochs=10,
            v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
            dataset='nuscenes_mf', version='v1.0-mini', category_name='Car',
            ct_partition_seed=42, ct_frame_stride=1, time_scale=.5,
            bb_scale=1.25, bb_offset=2., check_val_every_n_epoch=1,
            v31_temporal_backend='cfc', dynamics_time_mode='true',
            v31_evaluate_late3=True, v35_train_diagnostics=True, v38_diagnostics=True,
            precision=32, trainer_devices=1, limit_train_batches=1., limit_val_batches=1.)
        bad = [key for key, value in fixed.items() if cfg[key] != value
               or (key.startswith('limit_') and type(cfg[key]) is not float)]
        if bad:
            raise ValueError('unregistered formal v38 recipe: ' + ', '.join(bad))
    elif cfg.log_dir:
        root = (Path(__file__).resolve().parents[2] / 'artifacts/ct_checks').resolve()
        dest = Path(cfg.log_dir).resolve()
        if root not in (dest, *dest.parents):
            raise ValueError('v38 engineering outputs must be under artifacts/ct_checks')
    return cfg


def _validate_shared_configuration(cfg):
    """当前数据、时间、结构和可复用工程能力的校验；正式配方另做精确匹配。"""
    if cfg.v31_temporal_backend not in ('cfc', 'gru'):
        raise ValueError('v31_temporal_backend must be cfc or gru')
    if cfg.dataset not in ('nuscenes_mf', 'kitti_mf'):
        raise ValueError('v38 supports nuScenes and KITTI datasets')
    if cfg.dataset == 'nuscenes_mf':
        if cfg.version not in ('v1.0-mini', 'v1.0-trainval') or cfg.ct_coordinate_mode != 'global':
            raise ValueError('nuScenes requires registered version and global coordinates')
    elif cfg.version != 'kitti_tracking' or cfg.ct_coordinate_mode != 'sensor_relative':
        raise ValueError('KITTI requires kitti_tracking/sensor_relative')
    if type(cfg.trainer_devices) is not int or cfg.trainer_devices != 1:
        raise ValueError('v38 serial per-branch state requires one device per run')
    for name in ('hist_num', 'point_sample_size', 'v31_short_window', 'v31_long_window',
                 'v31_curriculum_epochs', 'batch_size', 'epoch', 'lr_decay_step', 'ct_frame_stride'):
        value = cfg[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('seed', 'ct_partition_seed', 'ct_pointcloud_cache_bytes', 'v32_reserve_windows', 'workers'):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    for name in ('time_scale', 'lr', 'bb_scale', 'bb_offset', 'wd', 'lr_decay_rate',
                 'v32_seed_translation', 'v32_seed_yaw_degrees', 'kitti_frame_period'):
        if type(cfg[name]) not in (int, float) or not math.isfinite(cfg[name]):
            raise ValueError(name + ' must be a finite number')
    if cfg.hist_num != 3 or type(cfg.precision) is not int or cfg.precision != 32:
        raise ValueError('v38 requires hist_num=3, Trainer FP32 and nonnegative workers')
    if cfg.time_scale <= 0 or cfg.lr <= 0 or cfg.bb_scale <= 0 or cfg.bb_offset < 0:
        raise ValueError('time, optimization and crop scales must be valid')
    if cfg.wd < 0 or cfg.kitti_frame_period <= 0:
        raise ValueError('weight decay must be nonnegative and KITTI frame period positive')
    for name in ('limit_train_batches', 'limit_val_batches'):
        value = cfg[name]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (name == 'limit_train_batches' and value == 0)):
            raise ValueError(name + ' must be finite and valid (validation may be disabled with zero)')
        if type(value) is float and value > 1:
            raise ValueError(name + ' fractions must be <= 1; use an integer batch count')
    if (isinstance(cfg.v32_reserve_windows, bool)
            or int(cfg.v32_reserve_windows) != cfg.v32_reserve_windows or cfg.v32_reserve_windows < 0):
        raise ValueError('v32_reserve_windows must be a nonnegative integer')
    if type(cfg.lr_warmup_steps) is not int or cfg.lr_warmup_steps < 0:
        raise ValueError('lr_warmup_steps must be a nonnegative integer')
    if cfg.lr_schedule not in ('step', 'multistep', 'piecewise') or not isinstance(cfg.lr_milestones, (list, tuple)):
        raise ValueError('requires step, multistep or piecewise and a list of lr_milestones')
    if cfg.lr_warmup_steps > 0 and cfg.lr_schedule != 'multistep':
        raise ValueError('positive lr_warmup_steps requires lr_schedule=multistep')
    milestones = cfg.lr_milestones
    if (any(type(value) is not int or value <= 0 for value in milestones)
            or list(milestones) != sorted(set(milestones))):
        raise ValueError('lr_milestones must be strictly increasing positive integers')
    if (cfg.lr_schedule == 'step' and milestones) or (cfg.lr_schedule != 'step' and not milestones):
        raise ValueError('step uses no lr_milestones; multistep/piecewise requires lr_milestones')
    if not math.isfinite(cfg.lr) or not math.isfinite(cfg.lr_decay_rate) or not 0 < cfg.lr_decay_rate < 1:
        raise ValueError('learning rate must be finite and decay rate must be in (0,1)')
    if cfg.lr_schedule == 'piecewise':
        values = cfg.lr_stage_values
        if (not isinstance(values, (list, tuple)) or len(values) != len(milestones) + 1
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v <= 0 for v in values)):
            raise ValueError('piecewise requires one positive finite lr_stage_values entry per stage')
        if values[0] != cfg.lr:
            raise ValueError('piecewise lr must equal the first lr_stage_values entry')
        cfg.lr_stage_values = list(values)
    if any(not math.isfinite(float(cfg[key])) or cfg[key] < 0
           for key in ('v32_seed_translation', 'v32_seed_yaw_degrees')):
        raise ValueError('seed perturbation bounds must be finite and nonnegative')
    for key, value in V35_DEFAULTS.items():
        if key != 'v35_train_diagnostics' and (isinstance(cfg[key], bool) or cfg[key] != value):
            raise ValueError('strong B1 fixed architecture identity mismatch: ' + key)


def load_config(path, overrides=None):
    values = load_yaml_config(path)
    values.update(overrides or {})
    values['cfg'] = str(path)
    return normalize_config(values)


def config_identity(config):
    cfg = normalize_config(config)
    excluded = {'cfg', 'path', 'log_dir', 'tag', 'checkpoint', 'init_checkpoint',
        'test', 'eval_checkpoint_epoch', 'accelerator', 'v31_evaluate_late3', 'devices'}
    payload = {key: value for key, value in cfg.items() if key not in excluded}
    hash_time_manifest_paths(payload)
    payload.update(schema=SCHEMA, evaluation_rng_policy='per_checkpoint_seed_v1',
                   training_execution_recipe=recipe_identity(cfg))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()
