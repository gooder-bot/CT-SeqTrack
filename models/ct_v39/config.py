"""v39 两组 Full：相同 E/G/M 与状态协议，仅学习率配方不同。"""
import hashlib
import json
import math
from pathlib import Path

from utils.config import load_yaml_config
from utils.execution_lab import execution_scope
from utils.time_config import engineering_time_defaults, validate_time_config, hash_time_manifest_paths
from .identity import SCHEMA, EXECUTION_REVISION, REGISTRATION_FILE, REGISTRATION_SCHEMA


class TrackingConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


RECIPES = {
    'full_a': dict(lr=1.5e-4, lr_schedule='multistep', lr_milestones=[20, 50], lr_decay_rate=.1),
    'full_b': dict(lr=1e-4, lr_schedule='step', lr_milestones=[], lr_decay_rate=.2),
}
G_DEFAULTS = dict(
    v39_template_points=64, v39_template_count=3, v39_pool_points=512, v39_crop_points=256,
    v39_local_neighbors=8, v39_sinkhorn_iterations=5, v39_g_feature_dim=64,
    v39_g_match_radius_min=.15, v39_g_match_radius_max=.5, v39_g_match_radius_fraction=.1,
    v39_g_match_probability_threshold=.2,
    v39_g_trust_threshold=.8, v39_g_min_support=3,
    v39_route_threshold=.9, v39_route_min_support=5, v39_route_nomatch_max=.1,
    v39_route_margin=.05, v39_max_cheap_frames=3, v39_max_full_gap=1.5,
    v39_route_window_fraction=.25,
    v39_cpu_raw_cache_bytes=536870912, v39_gpu_raw_cache_bytes=268435456,
    v39_recovery_local_points=2048, v39_recovery_global_points=4096,
    v39_refine_crop_points=256, v39_coarse_voxel_size=.5, v39_coarse_peaks=2,
    v39_search_nomatch_min=.5, v39_search_untrusted_seconds=1.,
    v39_global_search_after_failures=2, v39_search_radius_min=8., v39_search_radius_max=24.,
    v39_search_radius_base=4., v39_search_sigma_factor=3., v39_search_age_rate=2.,
    v39_recovery_crop_trust_threshold=.5,
    v39_loss_success=.125, v39_loss_precision=.125, v39_loss_trusted=.5,
    v39_loss_ranking=.1, v39_loss_match=.5, v39_loss_correction=.5,
    v39_loss_targetness=.2, v39_loss_nomatch=.2, v39_loss_qg=.5)
DEFAULTS = dict(
    net_model='ctseqtrackv39', experiment_family='ct_seqtrack_v39',
    experiment_name='ct39_full_a_mini_car_scratch_seed42', tag='v39_full_a_mini_car_seed42',
    v39_arm='full', v39_training_recipe='full_a', v31_arm='b1',
    v39_execution_revision=EXECUTION_REVISION, v39_activation_policy='active_from_epoch1_v1',
    v39_state_policy='accepted_separate_trusted_geometry_v1',
    v39_main_policy='strong_concat_b1_geometry_multistep_v1',
    v39_inference_mode='accuracy', v39_diagnostics=True,
    v39_point_packing=True, v39_attention_backend='fa2_varlen',
    v39_feature_bf16=True, v39_attention_deterministic=True,
    v39_m_horizons=[2, 4], v39_m_weights=[1., .5], v39_m_loss_weight=.05,
    v31_temporal_backend='cfc', dataset='nuscenes_mf', version='v1.0-mini', category_name='Car',
    path='/home/lishengjie/data/nuscenes-mini', ct_coordinate_mode='global',
    ct_partition_seed=42, ct_frame_stride=1, ct_pointcloud_cache_bytes=268435456,
    kitti_frame_period=.1, hist_num=3, point_sample_size=1024,
    bb_scale=1.25, bb_offset=2., time_scale=.5,
    v31_short_window=4, v31_long_window=8, v31_curriculum_epochs=10,
    v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
    v35_seed_policy='initial_exact_other_perturbed_v1', v35_history_descriptor_dim=33,
    v35_local_neighbors=16, v35_local_radius=1., v35_local_input_dim=69, v35_train_diagnostics=True,
    batch_size=16, workers=4, seed=42, epoch=60, wd=0.,
    lr=1.5e-4, lr_schedule='multistep', lr_milestones=[20, 50],
    lr_decay_step=20, lr_decay_rate=.1, lr_warmup_steps=0,
    trainer_devices=1, accelerator='auto', precision=32, execution_profile='optimized_bf16', devices=None,
    dynamics_time_mode='true', dynamics_time_manifest=None,
    v31_evaluate_late3=True, ct_engineering_check=False,
    limit_train_batches=1., limit_val_batches=1., check_val_every_n_epoch=1,
    checkpoint=None, init_checkpoint=None, test=False, log_dir=None, cfg=None, eval_checkpoint_epoch=None,
    **G_DEFAULTS)


def recipe_identity(config):
    name = config.get('v39_training_recipe')
    if name not in RECIPES:
        raise ValueError('unknown v39 training recipe: ' + str(name))
    return dict(recipe=name, arm='full', implementation_id='v39_geometry_multistep_v1',
        execution_revision=EXECUTION_REVISION, options=['unique_mask'], active_from_epoch=1,
        lr=config['lr'], lr_schedule=config['lr_schedule'], lr_milestones=list(config['lr_milestones']),
        lr_decay_step=config['lr_decay_step'], lr_decay_rate=config['lr_decay_rate'],
        acquisition=False, multistep_horizons=list(config['v39_m_horizons']),
        attention_backend=config['v39_attention_backend'], feature_bf16=config['v39_feature_bf16'])


def recipe_scope(config):
    recipe_identity(config)
    return execution_scope(('unique_mask',), registered_recipe=True)


def inference_identity(config):
    return dict(mode=config['v39_inference_mode'], threshold=config['v39_route_threshold'],
                policy={key: config[key] for key in G_DEFAULTS if not key.startswith('v39_loss_')})


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    if any(not isinstance(key, str) for key in supplied):
        raise ValueError('configuration keys must be strings')
    defaults = dict(DEFAULTS)
    defaults.update(engineering_time_defaults(supplied))
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError('v39 unknown/inactive configuration keys: ' + ', '.join(sorted(unknown)))
    cfg = TrackingConfig(defaults)
    cfg.update(supplied)
    spec = RECIPES.get(cfg.v39_training_recipe) if isinstance(cfg.v39_training_recipe, str) else None
    if spec is None or cfg.v39_arm != 'full':
        raise ValueError('v39 requires full_a/full_b and arm=full')
    for key, value in spec.items():
        if key not in supplied:
            cfg[key] = list(value) if isinstance(value, list) else value
    if cfg.net_model != 'ctseqtrackv39' or cfg.experiment_family != 'ct_seqtrack_v39':
        raise ValueError('v39 model/family identity mismatch')
    if cfg.v39_execution_revision != EXECUTION_REVISION or cfg.v31_arm != 'b1':
        raise ValueError('v39 execution revision or strong B1 identity mismatch')
    if cfg.init_checkpoint is not None:
        raise ValueError('v39 scratch_only forbids --init_checkpoint')
    if cfg.test and not cfg.checkpoint:
        raise ValueError('v39 evaluation requires --checkpoint')
    bool_keys = ('ct_engineering_check', 'test', 'v39_diagnostics', 'v35_train_diagnostics',
                 'v31_evaluate_late3', 'v39_point_packing', 'v39_feature_bf16', 'v39_attention_deterministic')
    for key in bool_keys:
        if type(cfg[key]) is not bool:
            raise ValueError(key + ' must be bool')
    if cfg.devices is not None and (type(cfg.devices) is not int or cfg.devices < 0):
        raise ValueError('devices selects one nonnegative physical GPU ID')
    if cfg.v39_inference_mode not in ('accuracy', 'adaptive'):
        raise ValueError('v39_inference_mode must be accuracy or adaptive')
    if not cfg.test and cfg.v39_inference_mode != 'accuracy':
        raise ValueError('training uses the registered common rollout; inference mode is evaluation-only')
    if cfg.v39_attention_backend not in ('fa2_varlen', 'math_reference'):
        raise ValueError('v39 attention backend must explicitly be fa2_varlen or math_reference')
    if cfg.execution_profile not in ('optimized_fp32', 'optimized_bf16'):
        raise ValueError('v39 requires an optimized execution profile')
    if cfg.v39_feature_bf16 != (cfg.execution_profile == 'optimized_bf16'):
        raise ValueError('v39 feature precision and execution_profile disagree')
    if cfg.v39_attention_backend == 'math_reference' and cfg.v39_feature_bf16:
        raise ValueError('math_reference is the engineering FP32 path')
    _validate_values(cfg)
    validate_time_config(cfg)
    if not cfg.test and cfg.v31_evaluate_late3 and not (type(cfg.limit_train_batches) is float and cfg.limit_train_batches == 1.):
        raise ValueError('partial train limits require --no_late3; automatic evaluation requires complete epochs')
    if not cfg.ct_engineering_check:
        # 文件路径、输出身份和评测阈值可变；训练/结构配方全部固定。
        exempt = {'experiment_name', 'tag', 'path', 'log_dir', 'cfg', 'checkpoint', 'test',
                  'eval_checkpoint_epoch', 'accelerator', 'devices', 'v39_training_recipe',
                  'v39_inference_mode', 'v39_route_threshold'}
        expected = dict(DEFAULTS)
        expected.update(spec)
        bad = [key for key, value in expected.items() if key not in exempt
               and (cfg[key] != value or (key.startswith('limit_') and type(cfg[key]) is not float))]
        if not cfg.test and cfg.v39_route_threshold != DEFAULTS['v39_route_threshold']:
            bad.append('v39_route_threshold')
        if bad:
            raise ValueError('unregistered formal v39 recipe: ' + ', '.join(sorted(bad)))
    elif cfg.log_dir:
        root = (Path(__file__).resolve().parents[2] / 'artifacts/ct_checks').resolve()
        dest = Path(cfg.log_dir).resolve()
        if root not in (dest, *dest.parents):
            raise ValueError('v39 engineering outputs must be under artifacts/ct_checks')
    cfg.lr_milestones = list(cfg.lr_milestones)
    cfg.v39_m_horizons, cfg.v39_m_weights = list(cfg.v39_m_horizons), list(cfg.v39_m_weights)
    return cfg


def _validate_values(cfg):
    if cfg.hist_num != 3 or cfg.precision != 32 or type(cfg.precision) is not int:
        raise ValueError('v39 requires three history frames and Trainer precision=32; features manage BF16 locally')
    if cfg.trainer_devices != 1 or type(cfg.trainer_devices) is not int:
        raise ValueError('v39 accepted transactions require one device')
    if cfg.v31_temporal_backend not in ('cfc', 'gru'):
        raise ValueError('temporal backend must be cfc or gru')
    if cfg.dataset not in ('nuscenes_mf', 'kitti_mf'):
        raise ValueError('unsupported v39 dataset')
    if cfg.dataset == 'nuscenes_mf' and (cfg.version not in ('v1.0-mini', 'v1.0-trainval') or cfg.ct_coordinate_mode != 'global'):
        raise ValueError('nuScenes requires registered version/global coordinates')
    if cfg.dataset == 'kitti_mf' and (cfg.version != 'kitti_tracking' or cfg.ct_coordinate_mode != 'sensor_relative'):
        raise ValueError('KITTI requires kitti_tracking/sensor_relative')
    for key, default in DEFAULTS.items():
        value = cfg[key]
        if type(default) is int and key not in ('precision',):
            if type(value) is not int or value < 0:
                raise ValueError(key + ' must be a nonnegative integer')
        if type(default) is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(key + ' must be finite')
    positive = ('time_scale', 'lr', 'bb_scale', 'kitti_frame_period', 'batch_size', 'epoch',
                'point_sample_size', 'v31_short_window', 'v31_long_window', 'v31_curriculum_epochs',
                'lr_decay_step', 'ct_frame_stride', 'check_val_every_n_epoch', 'v39_template_points',
                'v39_template_count', 'v39_pool_points', 'v39_crop_points', 'v39_local_neighbors',
                'v39_sinkhorn_iterations', 'v39_g_feature_dim', 'v39_max_full_gap')
    if any(cfg[key] <= 0 for key in positive):
        raise ValueError('v39 sizes/time/learning rate must be positive')
    if any(cfg[key] < 0 for key in ('wd', 'bb_offset', 'v32_seed_translation', 'v32_seed_yaw_degrees')):
        raise ValueError('weight decay/crop/seed bounds must be nonnegative')
    probabilities = ('v39_g_trust_threshold', 'v39_route_threshold', 'v39_route_nomatch_max',
                     'v39_route_window_fraction', 'v39_search_nomatch_min', 'v39_recovery_crop_trust_threshold',
                     'v39_g_match_probability_threshold')
    if any(not 0 <= cfg[key] <= 1 for key in probabilities):
        raise ValueError('v39 probabilities must be in [0,1]')
    if not 0 < cfg.lr_decay_rate < 1 or cfg.lr_warmup_steps != 0:
        raise ValueError('v39 requires decay in (0,1) and no warmup')
    if cfg.lr_schedule not in ('step', 'multistep') or not isinstance(cfg.lr_milestones, (list, tuple)):
        raise ValueError('v39 requires step or multistep schedule')
    milestones = cfg.lr_milestones
    if any(type(v) is not int or v <= 0 for v in milestones) or list(milestones) != sorted(set(milestones)):
        raise ValueError('milestones must be strictly increasing positive integers')
    if (cfg.lr_schedule == 'step' and milestones) or (cfg.lr_schedule == 'multistep' and not milestones):
        raise ValueError('step has no milestones; multistep requires milestones')
    if list(cfg.v39_m_horizons) != [2, 4] or len(cfg.v39_m_weights) != 2:
        raise ValueError('v39 M requires horizons [2,4] and two weights')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in cfg.v39_m_weights):
        raise ValueError('M weights must be finite and positive')
    if cfg.v39_m_loss_weight < 0 or any(cfg[key] < 0 for key in G_DEFAULTS if key.startswith('v39_loss_')):
        raise ValueError('loss weights must be nonnegative')
    for key in ('limit_train_batches', 'limit_val_batches'):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'limit_train_batches' and value == 0):
            raise ValueError(key + ' must be finite and valid')
        if type(value) is float and value > 1:
            raise ValueError('fractional batch limits must be <=1')


def load_config(path, overrides=None):
    values = load_yaml_config(path)
    values.update(overrides or {})
    values['cfg'] = str(path)
    return normalize_config(values)


def config_identity(config):
    cfg = normalize_config(config)
    excluded = {'cfg', 'path', 'log_dir', 'tag', 'checkpoint', 'init_checkpoint', 'test',
        'eval_checkpoint_epoch', 'accelerator', 'devices', 'v31_evaluate_late3',
        'v39_inference_mode', 'v39_route_threshold'}
    payload = {key: value for key, value in cfg.items() if key not in excluded}
    hash_time_manifest_paths(payload)
    payload.update(schema=SCHEMA, evaluation_rng_policy='per_checkpoint_seed_v1',
                   training_execution_recipe=recipe_identity(cfg))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()
