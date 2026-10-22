"""v38 三组固定 scratch 配方；全部复用无 B2 的强 concat B1。"""
import hashlib
import json
import math
from pathlib import Path

from models.ct_v31.config import DEFAULTS as OLD_DEFAULTS, V35_DEFAULTS, V31Config
from utils.config import load_yaml_config
from utils.execution_lab import execution_scope
from .identity import SCHEMA


REGISTRATION_FILE = '38_registration.json'
REGISTRATION_SCHEMA = 'ct_seqtrack.v38.registration.v1'
RECIPES = {
    'b1_standard': dict(arm='b1', lr_milestones=[20, 50]),
    'b1_hold': dict(arm='b1', lr_milestones=[20]),
    'full_standard': dict(arm='full', lr_milestones=[20, 50]),
}
DEFAULTS = dict(OLD_DEFAULTS, **V35_DEFAULTS,
    net_model='ctseqtrackv38', experiment_family='ct_seqtrack_v38',
    experiment_name='ct38_full_standard_mini_car_scratch_seed42',
    tag='v38_full_standard_mini_car_seed42', v31_arm='b1', v38_arm='full',
    v38_training_recipe='full_standard', v31_short_window=4,
    lr=.00015, lr_schedule='multistep', lr_milestones=[20, 50],
    check_val_every_n_epoch=1, execution_profile='optimized_fp32', devices=None,
    v38_diagnostics=True,
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
    if name not in RECIPES:
        raise ValueError('unknown v38 training recipe: ' + str(name))
    return dict(recipe=name, implementation_id='v38_concat_b1_selective_update_v1',
        projection='concat', options=['unique_mask'], prior_read_applicable=False,
        arm=RECIPES[name]['arm'], lr_milestones=list(RECIPES[name]['lr_milestones']),
        active_from_epoch=1, candidates=['q0', 'prior', 'trusted_propagation'],
        duplicate_xyz_tolerance=1e-4, duplicate_yaw_tolerance=1e-5)


def recipe_scope(config):
    recipe_identity(config)
    # 新路径不继承环境/外层工程变量；共同 B1 仅启用已验证的唯一点去重。
    return execution_scope(('unique_mask',), registered_recipe=True)


def normalize_config(config=None):
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError('v38 unknown/inactive configuration keys: ' + ', '.join(sorted(unknown)))
    cfg = V31Config(DEFAULTS)
    cfg.update(supplied)
    if 'v38_arm' in supplied and 'v38_training_recipe' not in supplied:
        cfg.v38_training_recipe = 'b1_standard' if cfg.v38_arm == 'b1' else 'full_standard'
    elif 'v38_training_recipe' in supplied and 'v38_arm' not in supplied:
        if cfg.v38_training_recipe in RECIPES:
            cfg.v38_arm = RECIPES[cfg.v38_training_recipe]['arm']
    spec = RECIPES.get(cfg.v38_training_recipe)
    if spec is None or cfg.v38_arm != spec['arm']:
        raise ValueError('v38 arm/recipe identity mismatch')
    if 'lr_milestones' not in supplied:
        cfg.lr_milestones = list(spec['lr_milestones'])
    if cfg.net_model != 'ctseqtrackv38' or cfg.experiment_family != 'ct_seqtrack_v38':
        raise ValueError('v38 model/family identity mismatch')
    if cfg.v31_arm != 'b1':
        raise ValueError('all v38 arms must use legacy B1 without extension acquisition')
    if cfg.execution_profile != 'optimized_fp32':
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
    from models.ct_v31.config import normalize_config as normalize_v35
    normalize_v35(_legacy_values(cfg))
    if (cfg.lr_schedule != 'multistep' or list(cfg.lr_milestones) != spec['lr_milestones']
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


def _legacy_values(config):
    keys = set(OLD_DEFAULTS) | set(V35_DEFAULTS)
    values = {key: value for key, value in config.items() if key in keys}
    values.update(net_model='ctseqtrackv35', experiment_family='ct_seqtrack_v35',
        v31_arm='b1', ct_engineering_check=True, log_dir=None,
        execution_profile='optimized_fp32')
    return values


def as_v35_config(config, arm=None):
    """所有新臂只适配 B1；绝不把 Full 传成旧 B2/Full。"""
    if arm not in (None, 'b1', 'full'):
        raise ValueError('v38 legacy adapter supports only its B1/full arms')
    from models.ct_v31.config import normalize_config as normalize_v35
    return normalize_v35(_legacy_values(normalize_config(config)))


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
    manifest = payload.pop('dynamics_time_manifest', None)
    if manifest:
        payload['dynamics_time_manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    payload.update(schema=SCHEMA, evaluation_rng_policy='per_checkpoint_seed_v1',
                   training_execution_recipe=recipe_identity(cfg))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()
