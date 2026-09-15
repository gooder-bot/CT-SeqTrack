"""v29可选等价执行优化；不改变网络算子、归约或训练随机流。"""

from collections import defaultdict
from contextlib import contextmanager

import torch

from utils.training_isolation import capture_global_rng_state, restore_global_rng_state


PERFORMANCE_DEFAULTS = {
    'ct_runtime_optimization': 'legacy',
    'ct_diagnostic_policy': 'full',
    'ct_scalar_log_every_n_steps': 50,
    'ct_diagnostic_every_n_steps': 100,
    'ct_h3_diagnostic_keep_ratio': 0.1,
}


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def performance_enabled(config):
    return bool(_get(config, 'ct_enable_v29', False)) and _get(
        config, 'ct_runtime_optimization', 'legacy') == 'equivalent_v1'


def diagnostics_sampled(config):
    return performance_enabled(config) and _get(config, 'ct_diagnostic_policy', 'full') == 'sampled_v1'


def validate_performance_contract(config):
    """旧配置不添加字段；显式性能配置及其恢复身份拒绝拼错/非法频率。"""
    if not any(_get(config, name) is not None for name in PERFORMANCE_DEFAULTS):
        return
    if not bool(_get(config, 'ct_enable_v29', False)):
        raise ValueError('performance configuration requires v29')
    mode = _get(config, 'ct_runtime_optimization', 'legacy')
    policy = _get(config, 'ct_diagnostic_policy', 'full')
    if mode not in ('legacy', 'equivalent_v1'):
        raise ValueError('unknown v29 runtime optimization')
    if policy not in ('full', 'sampled_v1'):
        raise ValueError('unknown v29 diagnostic policy')
    if policy == 'sampled_v1' and mode != 'equivalent_v1':
        raise ValueError('sampled diagnostics require equivalent_v1 runtime')
    for name in ('ct_scalar_log_every_n_steps', 'ct_diagnostic_every_n_steps'):
        value = _get(config, name, PERFORMANCE_DEFAULTS[name])
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    ratio = _get(config, 'ct_h3_diagnostic_keep_ratio', 0.1)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
        raise ValueError('ct_h3_diagnostic_keep_ratio must be in (0, 1]')
    if policy == 'sampled_v1':
        target = ('same_state_six_action_h1_aux_geometry_v30'
                  if _get(config, 'ct_enable_v30', False) else 'instantaneous_sp_gain_v1')
        if _get(config, 'ct_b3_target_contract') != target:
            raise ValueError('sampled H3 requires the registered instantaneous B3 target contract'
                             if _get(config, 'ct_enable_v30', False)
                             else 'sampled H3 requires the v29 H1-only B3 target contract')
        # 抽样指标不允许参与保存、调度、早停或模型选择。
        forbidden = ('sampled_', 'relation_ap', 'relation_auroc', 'relation_auprc',
                     'relation_ece', 'ct_epoch_calibration', 'h3')
        for key in ('checkpoint_monitor', 'scheduler_monitor', 'early_stopping_monitor',
                    'model_selection_metric', 'monitor'):
            monitor = str(_get(config, key, '')).lower()
            if any(name in monitor for name in forbidden):
                raise ValueError(f'{key} cannot consume sampled training diagnostics')


def scalar_items_to_python(mapping):
    """按device/dtype一次读回标量，不改每个值的原计算与精度。"""
    groups = defaultdict(list)
    result = {}
    for name, value in mapping.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f'{name} is not a scalar')
            groups[(value.device, value.dtype)].append((name, value.detach().reshape(())))
        else:
            result[name] = value
    for entries in groups.values():
        values = torch.stack([value for _, value in entries]).cpu().tolist()
        result.update((name, value) for (name, _), value in zip(entries, values))
    return {name: result[name] for name in mapping}


@torch.no_grad()
def restore_changed_buffers(module, snapshots):
    """一次读取每个设备的比较结果；仅恢复变化buffer，保留未变版本计数。"""
    buffers = dict(module.named_buffers())
    if buffers.keys() != snapshots.keys():
        raise RuntimeError('buffer registration changed inside isolated forward')
    storages = set()
    for value in buffers.values():
        if value.layout == torch.strided and value.numel():
            identity = (value.device, value.untyped_storage().data_ptr())
            if identity in storages:
                # 别名 buffer 的恢复会改变后续比较结果；保留原逐项比较/恢复顺序。
                for name, current in buffers.items():
                    if not torch.equal(current, snapshots[name]):
                        current.copy_(snapshots[name])
                return
            storages.add(identity)
    groups = defaultdict(list)
    equal_by_name = {}
    for name, value in buffers.items():
        before = snapshots[name]
        if (value.device.type == 'cuda' and value.device == before.device
                and value.layout == before.layout == torch.strided
                and not value.is_quantized and not before.is_quantized
                and value.shape == before.shape and value.dtype == before.dtype):
            groups[value.device].append((name, torch.eq(value, before).all()))
        else:
            # CPU、特殊layout、不同shape/dtype保留原torch.equal语义和错误行为。
            equal_by_name[name] = torch.equal(value, before)
    for entries in groups.values():
        equal = torch.stack([flag for _, flag in entries]).cpu().tolist()
        equal_by_name.update((name, flag) for (name, _), flag in zip(entries, equal))
    # 不按设备分组恢复：所有变化buffer仍按named_buffers的原顺序提交。
    for name, value in buffers.items():
        if not equal_by_name[name]:
            value.copy_(snapshots[name])


@contextmanager
def preserved_model_state(host):
    """纯诊断执行区：完整恢复BN/其他buffers、模式和Python/NumPy/torch RNG。"""
    flags = [(module, module.training) for module in host.modules()]
    buffers = {name: value.detach().clone() for name, value in host.named_buffers()}
    rng = capture_global_rng_state()
    try:
        yield
    finally:
        try:
            restore_changed_buffers(host, buffers)
        finally:
            for module, flag in flags:
                module.training = flag
            restore_global_rng_state(rng)
