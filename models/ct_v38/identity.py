"""v38 clean1 独立执行身份；旧运行必须使用对应的冻结源码。"""

EXECUTION_REVISION = 'clean1'
FROZEN_MODEL_SCHEMA = 'ct_seqtrack.selective_update.v38'
SCHEMA = 'ct_seqtrack.selective_update.v38_clean1'
RUNTIME_KEY = 'ct_v38_clean1_runtime'
EPOCH_SCHEMA = 'ct_seqtrack.v38_clean1.epoch_boundary.v1'
TRAINING_AUDIT_SCHEMA = 'ct_seqtrack.v38_clean1.training_audit.v1'
RESUME_SCHEMA = 'ct_seqtrack.v38_clean1.resume.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v38_clean1.candidate_records.v1'
COMPARE_SCHEMA = 'ct_seqtrack.v38_clean1.comparison.v1'
FULL_BATCH_SCHEMA = 'ct_seqtrack.v38_clean1.full_batch_check.v1'
RELIABILITY_PLOT_SCHEMA = 'ct_seqtrack.v38_clean1.reliability_plot.v1'
TRAINING_DIAGNOSTICS_SCHEMA = 'ct_seqtrack.v38_clean1.training_diagnostics.v1'
BENCHMARK_SCHEMA = 'ct_seqtrack.v38_clean1.engineering_benchmark.v1'
REGISTRATION_SCHEMA = 'ct_seqtrack.v38_clean1.registration.v1'
REGISTRATION_FILE = '38_clean1_registration.json'


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
    return EPOCH_SCHEMA


def training_audit_schema(config=None):
    model_version(config)
    return TRAINING_AUDIT_SCHEMA
