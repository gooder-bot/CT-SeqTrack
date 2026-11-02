"""V41 Full KITTI 固定 scratch 配方；恢复链与独立长轨迹辅助训练。"""
import hashlib
import json
import math
from pathlib import Path

from utils.config import load_yaml_config
from utils.execution_lab import execution_scope
from utils.execution_profile import validate_execution_profile
from utils.time_config import engineering_time_defaults, validate_time_config, hash_time_manifest_paths
from .identity import SCHEMA, EXECUTION_REVISION


# 数值和字段名沿用已验证的强 B1；定义由本版本持有，不再适配旧模型。
SHARED_DEFAULTS = dict(
    v31_temporal_backend='cfc',
    dataset='kitti_mf', version='kitti_tracking', category_name='Car',
    path='/home/lishengjie/data/cxtrack/training', ct_coordinate_mode='sensor_relative',
    ct_partition_seed=42, ct_frame_stride=1, ct_pointcloud_cache_bytes=268435456,
    kitti_frame_period=.1, hist_num=3, point_sample_size=1024,
    bb_scale=1.25, bb_offset=2., time_scale=.5,
    v31_long_window=8, v31_curriculum_epochs=10,
    v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
    batch_size=16, workers=4, seed=42, epoch=60, wd=0.,
    lr_decay_step=20, lr_decay_rate=.2, lr_warmup_steps=0,
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
    'b0': dict(arm='b0', enable_b1=False, enable_m=False, enable_b3=False),
    'b1': dict(arm='b1', enable_b1=True, enable_m=False, enable_b3=False),
    'b1_mtp': dict(arm='b1_mtp', enable_b1=True, enable_m=True, enable_b3=False),
    'full': dict(arm='full', enable_b1=True, enable_m=True, enable_b3=True),
}
for _spec in RECIPES.values():
    _spec.update(lr_milestones=[20, 40],
                 search_policy='dual_roi_guard_v1' if _spec['enable_b3'] else 'accepted_only')
DEFAULTS = dict(SHARED_DEFAULTS, **V35_DEFAULTS,
    net_model='ctseqtrackv41', experiment_family='ct_seqtrack_v41',
    experiment_name='ct41_full_kitti_car_scratch_seed42',
    tag='v41_full_kitti_car_seed42', v31_arm='b1', v41_arm='full',
    v41_training_recipe='full', v31_short_window=4,
    lr=.00015, lr_schedule='multistep', lr_milestones=[20, 40],
    check_val_every_n_epoch=1, execution_profile='optimized_fp32', devices=None,
    v41_diagnostics=True, v41_execution_revision=EXECUTION_REVISION,
    v41_selection_policy='q0_recovery_prior_fallback_tie_v1',
    v41_state_policy='selected_current_measurement_trusted_v1',
    v41_main_policy='dense_b0_optional_concat_b1_v1',
    v41_activation_policy='active_from_epoch1_v1',
    v41_gain_margin=0., v41_trust_threshold=.5, v41_min_support=3,
    v41_duplicate_xyz_tolerance=1e-4, v41_duplicate_yaw_tolerance=1e-5,
    enable_b1=True, enable_m=True, enable_b3=True,
    v41_search_policy='dual_roi_guard_v1',
    v41_search_main_quota=768, v41_search_backup_quota=256, v41_search_max_age=1.5,
    v41_m_weight=.05, v41_m_horizons=[2, 4], v41_m_horizon_weights=[1., .5],
    v41_attention_backend='fa2_varlen', v41_attention_precision='qkv_bf16_only',
    v41_cpu_acquisition='chunked_exact_v1', v41_memory_logging=True,
    v41_recovery_weight=.1, v41_aux_fraction=.25, v41_aux_loss_weight=.25,
    v41_train_preload_offset=10., v41_kitti_geometry='open3dsot_velodyne_v1')
POLICY_KEYS = tuple(key for key in DEFAULTS if key.startswith('v41_') and key.endswith('_policy'))
FIXED_B3_KEYS = ('v41_gain_margin', 'v41_trust_threshold', 'v41_min_support',
                'v41_duplicate_xyz_tolerance', 'v41_duplicate_yaw_tolerance')


def recipe_identity(config):
    name = config.get('v41_training_recipe')
    if not isinstance(name, str) or name not in RECIPES:
        raise ValueError('unknown v41 training recipe: ' + str(name))
    options = list(config.get('v41_engineering_options', ('unique_mask',)))
    return dict(recipe=name, implementation_id='v41_current_point_xyz_recovery_v1',
        execution_revision=EXECUTION_REVISION,
        projection='concat', options=options, prior_read_applicable=False,
        arm=RECIPES[name]['arm'], lr_milestones=list(RECIPES[name]['lr_milestones']),
        enabled={key: RECIPES[name][key] for key in ('enable_b1', 'enable_m', 'enable_b3')},
        search_policy=RECIPES[name]['search_policy'],
        attention_backend=config.get('v41_attention_backend', 'fa2_varlen'),
        attention_precision='qkv_bf16_only', m_weight=.05, m_horizons=[2, 4],
        search_quota=[768, 256], search_max_age_seconds=1.5,
        active_from_epoch=1, candidates=['q0', 'prior', 'trusted_propagation', 'recovery_measurement'],
        tie_priority=[0, 3, 1, 2], measurement_sources=[0, 3],
        recovery=dict(type='current_local_xyz_residual_v1', bound='one_box_size', weight=.1),
        auxiliary=dict(fraction=.25, loss_weight=.25, main_optimizer_steps_unchanged=True),
        kitti_geometry=config.get('v41_kitti_geometry', 'open3dsot_velodyne_v1'), train_preload_offset=10.,
        duplicate_xyz_tolerance=1e-4, duplicate_yaw_tolerance=1e-5)


def recipe_scope(config):
    recipe_identity(config)
    # 新路径不继承环境/外层工程变量；共同 B1 仅启用已验证的唯一点去重。
    return execution_scope(config.get('v41_engineering_options', ('unique_mask',)),
                           registered_recipe=True)


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    if any(not isinstance(key, str) for key in supplied):
        raise ValueError('configuration keys must be strings')
    defaults = dict(DEFAULTS)
    defaults.update(engineering_time_defaults(supplied))
    if 'v41_engineering_options' in supplied:
        options = supplied['v41_engineering_options']
        if (supplied.get('ct_engineering_check') is not True or not isinstance(options, (list, tuple))
                or not all(isinstance(value, str) for value in options)
                or len(options) != len(set(options)) or 'unique_mask' not in options
                or set(options) != {'unique_mask'}):
            raise ValueError('v41_engineering_options supports unique_mask only; concat is fixed')
        defaults['v41_engineering_options'] = ['unique_mask']
    if supplied.get('lr_schedule') == 'piecewise':
        defaults['lr_stage_values'] = ()
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError('v41 unknown/inactive configuration keys: ' + ', '.join(sorted(unknown)))
    cfg = TrackingConfig(defaults)
    cfg.update(supplied)
    if 'v41_arm' in supplied and 'v41_training_recipe' not in supplied:
        cfg.v41_training_recipe = cfg.v41_arm
    elif 'v41_training_recipe' in supplied and 'v41_arm' not in supplied:
        if isinstance(cfg.v41_training_recipe, str) and cfg.v41_training_recipe in RECIPES:
            cfg.v41_arm = RECIPES[cfg.v41_training_recipe]['arm']
    spec = RECIPES.get(cfg.v41_training_recipe) if isinstance(cfg.v41_training_recipe, str) else None
    if spec is None or cfg.v41_arm != spec['arm']:
        raise ValueError('v41 arm/recipe identity mismatch')
    derived = {key: spec[key] for key in ('enable_b1', 'enable_m', 'enable_b3')}
    derived.update(v41_search_policy=spec['search_policy'], v31_arm='b1' if spec['enable_b1'] else 'b0')
    for key, value in derived.items():
        if key in supplied and (type(supplied[key]) is not type(value) or supplied[key] != value):
            raise ValueError('v41 arm-derived field cannot be overridden: ' + key)
        cfg[key] = value
    if 'lr_milestones' not in supplied:
        cfg.lr_milestones = list(spec['lr_milestones'])
    if cfg.net_model != 'ctseqtrackv41' or cfg.experiment_family != 'ct_seqtrack_v41':
        raise ValueError('v41 model/family identity mismatch')
    if cfg.v41_execution_revision != EXECUTION_REVISION:
        raise ValueError('v41 execution revision mismatch; use the original frozen source')
    validate_execution_profile(cfg)
    if cfg.execution_profile != 'optimized_fp32':
        raise ValueError('v41 requires FP32 backbone/projections; only attention QKV may use BF16')
    if cfg.init_checkpoint is not None:
        raise ValueError('v41 scratch_only forbids --init_checkpoint')
    if cfg.test and not cfg.checkpoint:
        raise ValueError('v41 evaluation requires --checkpoint')
    for key in ('ct_engineering_check', 'v41_diagnostics', 'v35_train_diagnostics', 'v31_evaluate_late3', 'test', 'v41_memory_logging'):
        if type(cfg[key]) is not bool:
            raise ValueError(key + ' must be bool')
    if cfg.devices is not None and (type(cfg.devices) is not int or cfg.devices < 0):
        raise ValueError('devices selects one nonnegative physical GPU ID')
    for key in POLICY_KEYS + FIXED_B3_KEYS:
        if key == 'v41_search_policy':
            continue
        if type(cfg[key]) is not type(DEFAULTS[key]) or cfg[key] != DEFAULTS[key]:
            raise ValueError('v41 fixed policy identity mismatch: ' + key)
    for key in ('v41_search_main_quota', 'v41_search_backup_quota', 'v41_search_max_age',
                'v41_m_weight', 'v41_m_horizons', 'v41_m_horizon_weights',
                'v41_attention_precision', 'v41_cpu_acquisition', 'v41_recovery_weight',
                'v41_aux_fraction', 'v41_aux_loss_weight', 'v41_train_preload_offset', 'v41_kitti_geometry'):
        if type(cfg[key]) is not type(DEFAULTS[key]) or cfg[key] != DEFAULTS[key]:
            raise ValueError('v41 fixed structural contract mismatch: ' + key)
    if cfg.v41_attention_backend not in ('fa2_varlen', 'math'):
        raise ValueError('v41 attention backend must be fa2_varlen or engineering math')
    if not cfg.ct_engineering_check and cfg.v41_attention_backend != 'fa2_varlen':
        raise ValueError('formal v41 requires actual FA2; math fallback is forbidden')
    _validate_shared_configuration(cfg)
    validate_time_config(cfg)
    if (not cfg.test and cfg.v31_evaluate_late3
            and not (type(cfg.limit_train_batches) is float and cfg.limit_train_batches == 1.)):
        raise ValueError('partial/integer train limits require --no_late3; automatic evaluation requires complete epochs')
    if not cfg.ct_engineering_check and (cfg.lr_schedule != 'multistep' or list(cfg.lr_milestones) != spec['lr_milestones']
            or cfg.lr_warmup_steps != 0):
        raise ValueError('v41 requires registered 20/40 decay without warmup')
    cfg.lr_milestones = list(cfg.lr_milestones)
    for key in ('time_scale', 'lr', 'bb_scale', 'bb_offset', 'wd'):
        if not math.isfinite(float(cfg[key])):
            raise ValueError('nonfinite v41 configuration value: ' + key)
    if type(cfg.workers) is not int or cfg.workers < 0:
        raise ValueError('workers must be a nonnegative integer')
    if type(cfg.check_val_every_n_epoch) is not int or cfg.check_val_every_n_epoch < 1:
        raise ValueError('check_val_every_n_epoch must be a positive integer')
    if not cfg.ct_engineering_check:
        fixed = dict(epoch=60, batch_size=16, workers=4, seed=42, lr=.00015, wd=0.,
            lr_decay_step=20, lr_decay_rate=.2, point_sample_size=1024,
            v31_short_window=4, v31_long_window=8, v31_curriculum_epochs=10,
            v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
            dataset='kitti_mf', version='kitti_tracking', category_name='Car',
            ct_coordinate_mode='sensor_relative', v41_arm='full', v41_training_recipe='full',
            ct_partition_seed=42, ct_frame_stride=1, time_scale=.5,
            bb_scale=1.25, bb_offset=2., check_val_every_n_epoch=1,
            v31_temporal_backend='cfc', dynamics_time_mode='true',
            v31_evaluate_late3=True, v35_train_diagnostics=True, v41_diagnostics=True, v41_memory_logging=True,
            precision=32, trainer_devices=1, limit_train_batches=1., limit_val_batches=1.)
        bad = [key for key, value in fixed.items() if cfg[key] != value
               or (key.startswith('limit_') and type(cfg[key]) is not float)]
        if bad:
            raise ValueError('V41 experiment recipe differs from the agreed configuration: ' + ', '.join(bad))
    elif cfg.log_dir:
        root = (Path(__file__).resolve().parents[2] / 'artifacts/ct_checks').resolve()
        dest = Path(cfg.log_dir).resolve()
        if root not in (dest, *dest.parents):
            raise ValueError('v41 engineering outputs must be under artifacts/ct_checks')
    return cfg


def _validate_shared_configuration(cfg):
    """当前数据、时间、结构和可复用工程能力的校验；正式配方另做精确匹配。"""
    if cfg.v31_temporal_backend not in ('cfc', 'gru'):
        raise ValueError('v31_temporal_backend must be cfc or gru')
    if cfg.dataset not in ('nuscenes_mf', 'kitti_mf'):
        raise ValueError('v41 supports nuScenes and KITTI datasets')
    if cfg.dataset == 'nuscenes_mf':
        if cfg.version not in ('v1.0-mini', 'v1.0-trainval') or cfg.ct_coordinate_mode != 'global':
            raise ValueError('nuScenes requires registered version and global coordinates')
    elif cfg.version != 'kitti_tracking' or cfg.ct_coordinate_mode != 'sensor_relative':
        raise ValueError('KITTI requires kitti_tracking/sensor_relative')
    if type(cfg.trainer_devices) is not int or cfg.trainer_devices != 1:
        raise ValueError('v41 serial per-branch state requires one device per run')
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
        raise ValueError('v41 requires hist_num=3, Trainer FP32 and nonnegative workers')
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
