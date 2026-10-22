"""执行优化身份；不改变模型版本、旧配置默认值或参数字典。"""
from collections.abc import Mapping


EXECUTION_PROFILES = ('legacy', 'optimized_fp32', 'optimized_bf16')
EXECUTION_SCHEMA = 'ct_seqtrack.execution.v1'
_SUPPORTED_MODELS = ('ctseqtrackv35', 'ctseqtrackv36', 'ctseqtrackv37', 'ctseqtrackv38')


def validate_execution_profile(config):
    """旧版 BF16 限工程验证；v37 正式 Full 由其登记合同另行校验。"""
    if 'execution_profile' not in config:
        return
    if config.get('net_model') not in _SUPPORTED_MODELS:
        raise ValueError('execution_profile is supported only for v35/v36/v37/v38')
    profile = config['execution_profile']
    if profile not in EXECUTION_PROFILES:
        raise ValueError('unknown execution_profile: ' + str(profile))
    formal_v37_full = (config.get('net_model') == 'ctseqtrackv37'
                       and config.get('v37_arm') == 'full')
    if (profile == 'optimized_bf16' and not config.get('ct_engineering_check', False)
            and not formal_v37_full):
        raise ValueError('optimized_bf16 execution_profile requires ct_engineering_check')


def effective_execution_profile(config):
    validate_execution_profile(config)
    if config.get('net_model') not in _SUPPORTED_MODELS:
        return 'legacy'
    return config.get('execution_profile', 'optimized_fp32')


def optimized_execution(config):
    return effective_execution_profile(config) != 'legacy'


def execution_identity(config, source=None):
    """记录实际执行路径；源码 hash 继续覆盖全部生产 Python 文件。"""
    if source is None:
        from models.ct_v31.entry import source_identity
        source = source_identity()
    return dict(schema=EXECUTION_SCHEMA, profile=effective_execution_profile(config),
                source_sha256=source['sha256'])


def validate_execution_identity(value, config, source=None):
    if not isinstance(value, Mapping) or value.get('schema') != EXECUTION_SCHEMA:
        raise ValueError('checkpoint/run lacks execution identity; use its original source for legacy checkpoints')
    expected = execution_identity(config, source)
    if dict(value) != expected:
        raise ValueError('execution profile/source identity mismatch; use the matching profile and original source')
    return value
