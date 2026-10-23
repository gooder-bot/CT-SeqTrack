"""当前工程时间对照的 split-aware manifest 配置和内容身份。"""
import hashlib
import math
from pathlib import Path

from datasets.protocol_utils import resolve_dynamics_time_kwargs


MANIFEST_ROLES = ('train', 'val', 'test', 'eval', 'calibration', 'dev')
TIME_MANIFEST_KEYS = ('dynamics_time_manifest',) + tuple(
    key for role in MANIFEST_ROLES
    for key in (f'dynamics_time_manifest_{role}', f'{role}_dynamics_time_manifest'))
TIME_ENGINEERING_KEYS = TIME_MANIFEST_KEYS[1:] + ('dynamics_fixed_delta_t',)


def engineering_time_defaults(supplied):
    selected = set(supplied).intersection(TIME_ENGINEERING_KEYS)
    if selected and supplied.get('ct_engineering_check') is not True:
        raise ValueError('role-specific time manifests/fixed delta require ct_engineering_check')
    return {key: None for key in selected}


def validate_time_config(config):
    mode = config['dynamics_time_mode']
    if mode not in ('true', 'fixed', 'shuffled'):
        raise ValueError('unknown dynamics_time_mode')
    for key in TIME_MANIFEST_KEYS:
        if key not in config:
            continue
        value = config[key]
        if value is not None and not isinstance(value, (str, Path)):
            raise ValueError(key + ' must be a manifest path or None')
        if isinstance(value, Path):
            config[key] = str(value)
        if value and mode != 'shuffled':
            raise ValueError(key + ' is valid only for shuffled time')
    if 'dynamics_fixed_delta_t' in config:
        value = config['dynamics_fixed_delta_t']
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('dynamics_fixed_delta_t must be a positive finite number')
    if mode == 'shuffled':
        # 与 BatchBuilder 的实际 dataset role 映射相同；不是 CLI 的 val 别名。
        roles = {'test'} if config['test'] else {'train'}
        if not config['test'] and config['limit_val_batches'] > 0:
            roles.add('dev' if config['dataset'] == 'kitti_mf' else 'test')
        if not config['test'] and config['v31_evaluate_late3']:
            roles.add('test')
        missing = [role for role in sorted(roles)
                   if not resolve_dynamics_time_kwargs(config, role)['dynamics_time_manifest']]
        if missing:
            raise ValueError('shuffled time requires an explicit manifest for dataset roles: ' + ', '.join(missing))


def hash_time_manifest_paths(payload):
    """所有声明的 manifest 内容进入身份，路径只作运行位置，不进入算法身份。"""
    for key in TIME_MANIFEST_KEYS:
        path = payload.pop(key, None)
        if path:
            payload[key + '_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return payload
