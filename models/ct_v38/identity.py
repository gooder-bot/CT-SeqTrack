"""v38 独立身份；不接受旧版本 checkpoint 的跨源码恢复。"""

SCHEMA = 'ct_seqtrack.selective_update.v38'
RUNTIME_KEY = 'ct_v38_runtime'


def model_version(config=None):
    if config is not None and config.get('net_model') != 'ctseqtrackv38':
        raise ValueError('v38 identity requires ctseqtrackv38')
    return 'v38'


def model_schema(config=None):
    model_version(config)
    return SCHEMA


def runtime_key(config=None):
    model_version(config)
    return RUNTIME_KEY


def runtime_schema(config=None):
    model_version(config)
    return 'ct_seqtrack.v38.epoch_boundary.v1'


def training_audit_schema(config=None):
    model_version(config)
    return 'ct_seqtrack.v38.training_audit.v1'
