"""v37 五组独立登记；复用 v35 组件不改变旧配置及摘要。"""
import hashlib
import json
import math
from pathlib import Path

from models.ct_v31.config import DEFAULTS as OLD_DEFAULTS, V35_DEFAULTS, V31Config
from utils.config import load_yaml_config
from .identity import SCHEMA
from utils.b1_training_recipes import RECIPE_KEY, recipe_spec, recipe_identity

ARMS = ('b0', 'b1', 'b1_b2', 'full')
EXECUTION_PROFILES = ('legacy', 'optimized_fp32', 'optimized_bf16')
DEFAULTS = dict(OLD_DEFAULTS, **V35_DEFAULTS,
    net_model='ctseqtrackv37', experiment_family='ct_seqtrack_v37',
    experiment_name='ct37_full_mini_car_scratch_seed42', tag='full_mini_car_seed42',
    v37_arm='full', v31_arm='full', v31_short_window=4,
    lr=.00015, lr_schedule='multistep', lr_milestones=[20, 50],
    execution_profile='optimized_fp32', v37_diagnostics=True,
    v37_observation_policy='v35_lwh_q0_v1',
    v37_state_policy='v35_accepted_window_v1',
    v37_evidence_policy='current_geometry_identity_only_v1',
    v37_selection_policy='final_box_sp_utility_v1')
POLICY_KEYS = tuple(key for key in DEFAULTS if key.startswith('v37_') and key.endswith('_policy'))


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(DEFAULTS) - {RECIPE_KEY}
    if unknown:
        raise ValueError('v37 unknown/inactive configuration keys: ' + ', '.join(sorted(unknown)))
    cfg = V31Config(DEFAULTS)
    cfg.update(supplied)
    recipe = recipe_spec(cfg)
    if 'v37_arm' in supplied and 'v31_arm' not in supplied:
        cfg.v31_arm = cfg.v37_arm
    if 'v31_arm' in supplied and 'v37_arm' not in supplied:
        cfg.v37_arm = cfg.v31_arm
    if cfg.net_model != 'ctseqtrackv37' or cfg.experiment_family != 'ct_seqtrack_v37':
        raise ValueError('v37 model/family identity mismatch')
    if cfg.v37_arm not in ARMS or cfg.v31_arm != cfg.v37_arm:
        raise ValueError('v37_arm and compatible v31_arm must be the same registered arm')
    if cfg.execution_profile not in EXECUTION_PROFILES:
        raise ValueError('unknown v37 execution_profile')
    if cfg.init_checkpoint is not None:
        raise ValueError('scratch_only forbids --init_checkpoint')
    if cfg.test and not cfg.checkpoint:
        raise ValueError('evaluation requires --checkpoint')
    if type(cfg.ct_engineering_check) is not bool:
        raise ValueError('ct_engineering_check must be bool')
    for key in ('v37_diagnostics', 'v35_train_diagnostics', 'v31_evaluate_late3'):
        if type(cfg[key]) is not bool:
            raise ValueError(key + ' must be bool')
    for key in POLICY_KEYS:
        if cfg[key] != DEFAULTS[key]:
            raise ValueError('v37 policy identity mismatch: ' + key)
    # 旧校验器只看到独立适配后的旧字段，绝不接收 v37 身份或改写其默认值。
    legacy = _legacy_values(cfg)
    from models.ct_v31.config import normalize_config as normalize_v35
    normalize_v35(legacy)
    for key in ('time_scale', 'lr', 'bb_scale', 'bb_offset', 'wd'):
        if not math.isfinite(float(cfg[key])):
            raise ValueError('nonfinite configuration value: ' + key)
    cfg.lr_milestones = list(cfg.lr_milestones)
    milestones = [20, 50] if recipe is None else recipe['lr_milestones']
    if (cfg.lr_schedule != 'multistep' or cfg.lr_milestones != milestones
            or cfg.lr_warmup_steps != 0):
        raise ValueError('v37 requires its registered multistep schedule without warmup')
    if recipe is not None and (cfg.v37_arm != 'b1' or cfg.execution_profile != 'optimized_fp32'):
        raise ValueError('registered execution recipes require B1 FP32')
    if not cfg.ct_engineering_check:
        fixed = dict(epoch=60, batch_size=16, workers=4, seed=42, lr=.00015, wd=0.,
            lr_decay_step=20, lr_decay_rate=.1, point_sample_size=1024,
            v31_short_window=4, v31_long_window=8, v31_curriculum_epochs=10,
            v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
            dataset='nuscenes_mf', version='v1.0-mini', category_name='Car',
            ct_partition_seed=42, ct_frame_stride=1, time_scale=.5,
            bb_scale=1.25, bb_offset=2., check_val_every_n_epoch=5,
            v31_temporal_backend='cfc', dynamics_time_mode='true',
            v31_evaluate_late3=True, v35_train_diagnostics=True, v37_diagnostics=True,
            precision=32, trainer_devices=1, limit_train_batches=1., limit_val_batches=1.)
        if recipe is not None:
            fixed.update(batch_size=recipe['batch_size'], check_val_every_n_epoch=1)
        bad = [key for key, value in fixed.items() if cfg[key] != value
               or (key.startswith('limit_') and isinstance(cfg[key], int))]
        if bad:
            raise ValueError('unregistered formal v37 recipe: ' + ', '.join(bad))
        if cfg.execution_profile == 'legacy':
            raise ValueError('legacy execution is engineering-only for v37')
        if cfg.execution_profile == 'optimized_bf16' and cfg.v37_arm != 'full':
            raise ValueError('only v37 Full registers formal local BF16')
    elif cfg.log_dir:
        root = Path(__file__).resolve().parents[2] / 'artifacts' / 'ct_checks'
        dest = Path(cfg.log_dir).resolve()
        if root.resolve() not in (dest, *dest.parents):
            raise ValueError('engineering outputs must be under artifacts/ct_checks')
    return cfg


def _legacy_values(config, arm=None):
    keys = set(OLD_DEFAULTS) | set(V35_DEFAULTS)
    value = {key: item for key, item in config.items() if key in keys}
    arm = arm or ('b0' if config.get('v37_arm', config.get('v31_arm')) == 'b0' else 'b1')
    value.update(net_model='ctseqtrackv35', experiment_family='ct_seqtrack_v35',
        v31_arm=arm, ct_engineering_check=True, log_dir=None,
        execution_profile='legacy' if config.get('execution_profile') == 'legacy' else 'optimized_fp32')
    return value


def as_v35_config(config, arm=None):
    """仅用于旧组件构造；不能用于 v37 manifest/checkpoint 的身份。"""
    from models.ct_v31.config import normalize_config as normalize_v35
    return normalize_v35(_legacy_values(normalize_config(config), arm))


def load_config(path, overrides=None):
    values = load_yaml_config(path)
    values.update(overrides or {})
    values['cfg'] = str(path)
    return normalize_config(values)


def config_identity(config):
    cfg = normalize_config(config)
    excluded = {'cfg', 'path', 'log_dir', 'tag', 'checkpoint', 'init_checkpoint',
                'test', 'eval_checkpoint_epoch', 'accelerator', 'v31_evaluate_late3'}
    payload = {key: value for key, value in cfg.items() if key not in excluded}
    manifest = payload.pop('dynamics_time_manifest', None)
    if manifest:
        payload['dynamics_time_manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    payload.update(schema=SCHEMA, evaluation_rng_policy='per_checkpoint_seed_v1')
    if RECIPE_KEY in cfg:
        payload['training_execution_recipe'] = recipe_identity(cfg)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()
