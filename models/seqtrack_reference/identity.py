"""独立参考臂的新工程身份；旧正式运行使用其冻结源码。"""
SCHEMA = 'seqtrack_reference.clean1.v1'


def model_version(config=None):
    return 'reference_clean1'


def model_schema(config=None):
    return SCHEMA


def runtime_key(config=None):
    return 'seqtrack_reference_runtime'


def runtime_schema(config=None):
    return 'seqtrack_reference.clean1.epoch_boundary.v1'


def training_audit_schema(config=None):
    return 'seqtrack_reference.clean1.training_audit.v1'
