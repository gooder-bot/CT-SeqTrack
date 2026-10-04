"""v36 版本身份；不扩展旧版本的分派表。"""
from .contracts import SCHEMA, RUNTIME_KEY


def model_version(config=None):
    if config is not None and config.get('net_model') != 'ctseqtrackv36':
        raise ValueError('v36 identity requires ctseqtrackv36')
    return 'v36'


def model_schema(config=None):
    model_version(config)
    return SCHEMA


def runtime_key(config=None):
    model_version(config)
    return RUNTIME_KEY


def runtime_schema(config=None):
    model_version(config)
    return 'ct_seqtrack.v36.epoch_boundary.v1'


def training_audit_schema(config=None):
    model_version(config)
    return 'ct_seqtrack.v36.training_audit.v1'
