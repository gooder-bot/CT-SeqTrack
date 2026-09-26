"""按配置选择实验身份；v33/reference 的历史摘要及 checkpoint 键保持。"""

from .contracts import SCHEMA


def model_version(config=None):
    model = 'ctseqtrackv33' if config is None else config.get('net_model', 'ctseqtrackv33')
    versions = {'ctseqtrackv33': 'v33', 'seqtrack_reference': 'v33', 'ctseqtrackv34': 'v34'}
    if model not in versions:
        raise ValueError('unsupported model identity: ' + str(model))
    return versions[model]


def model_schema(config=None):
    return SCHEMA if model_version(config) == 'v33' else 'ct_seqtrack.joint_identity.v34'


def runtime_key(config=None):
    return 'ct_' + model_version(config) + '_runtime'


def runtime_schema(config=None):
    return 'ct_seqtrack.' + model_version(config) + '.epoch_boundary.v1'


def training_audit_schema(config=None):
    return 'ct_seqtrack.' + model_version(config) + '.training_audit.v1'
