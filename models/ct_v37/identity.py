"""v37 独立身份；旧 v35/v36 的 schema 与分派表保持不变。"""

SCHEMA = 'ct_seqtrack.structural_evidence.v37'
RUNTIME_KEY = 'ct_v37_runtime'


def model_version(config=None):
    if config is not None and config.get('net_model') != 'ctseqtrackv37':
        raise ValueError('v37 identity requires ctseqtrackv37')
    return 'v37'


def model_schema(config=None):
    model_version(config)
    return SCHEMA


def runtime_key(config=None):
    model_version(config)
    return RUNTIME_KEY


def runtime_schema(config=None):
    model_version(config)
    return 'ct_seqtrack.v37.epoch_boundary.v1'


def training_audit_schema(config=None):
    model_version(config)
    return 'ct_seqtrack.v37.training_audit.v1'
