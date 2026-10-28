"""v39 独立源码、训练配方和完整 epoch 恢复身份。"""

EXECUTION_REVISION = 'v1'
SCHEMA = 'ct_seqtrack.geometry_motion.v39'
RUNTIME_KEY = 'ct_v39_runtime'
EPOCH_SCHEMA = 'ct_seqtrack.v39.epoch_boundary.v1'
TRAINING_AUDIT_SCHEMA = 'ct_seqtrack.v39.training_audit.v1'
RESUME_SCHEMA = 'ct_seqtrack.v39.resume.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v39.candidate_records.v1'
COMPARE_SCHEMA = 'ct_seqtrack.v39.comparison.v1'
FULL_BATCH_SCHEMA = 'ct_seqtrack.v39.full_batch_check.v1'
TRAINING_DIAGNOSTICS_SCHEMA = 'ct_seqtrack.v39.training_diagnostics.v1'
BENCHMARK_SCHEMA = 'ct_seqtrack.v39.engineering_benchmark.v1'
REGISTRATION_SCHEMA = 'ct_seqtrack.v39.registration.v1'
REGISTRATION_FILE = '39_registration.json'

# 远端允许保留已封存的退役版本；不得扩大为按版本号或目录名的通配排除。
SOURCE_INVENTORY_POLICY = 'v39_active_python_with_explicit_retired_v1'
RETIRED_SOURCE_DIRECTORIES = ('models/ct_v31', 'models/ct_v36', 'models/ct_v37')
RETIRED_SOURCE_FILES = ('models/ctseqtrackv31.py', 'models/ctseqtrackv36.py', 'models/ctseqtrackv37.py')


def retired_source_path(relative):
    """只排除明确退役的三套代码；未知 Python 文件仍进入 source hash。"""
    return relative in RETIRED_SOURCE_FILES or any(
        relative.startswith(directory + '/') for directory in RETIRED_SOURCE_DIRECTORIES)


def model_version(config=None):
    if config is not None and config.get('net_model') != 'ctseqtrackv39':
        raise ValueError('v39 identity requires ctseqtrackv39')
    return 'v39'


def model_schema(config=None):
    model_version(config)
    return SCHEMA


def runtime_key(config=None):
    model_version(config)
    return RUNTIME_KEY


def runtime_schema(config=None):
    model_version(config)
    return EPOCH_SCHEMA


def training_audit_schema(config=None):
    model_version(config)
    return TRAINING_AUDIT_SCHEMA
