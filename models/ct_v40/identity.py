"""V40 独立执行身份；v38clean1 仅为源码基底，不接受跨版本恢复。"""

EXECUTION_REVISION = 'v1'
SCHEMA = 'ct_seqtrack.motion_search.v40'
RUNTIME_KEY = 'ct_v40_runtime'
EPOCH_SCHEMA = 'ct_seqtrack.v40.epoch_boundary.v1'
TRAINING_AUDIT_SCHEMA = 'ct_seqtrack.v40.training_audit.v1'
RESUME_SCHEMA = 'ct_seqtrack.v40.resume.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v40.candidate_records.v1'
COMPARE_SCHEMA = 'ct_seqtrack.v40.comparison.v1'
FULL_BATCH_SCHEMA = 'ct_seqtrack.v40.full_batch_check.v1'
RELIABILITY_PLOT_SCHEMA = 'ct_seqtrack.v40.reliability_plot.v1'
TRAINING_DIAGNOSTICS_SCHEMA = 'ct_seqtrack.v40.training_diagnostics.v1'
BENCHMARK_SCHEMA = 'ct_seqtrack.v40.engineering_benchmark.v1'
REGISTRATION_SCHEMA = 'ct_seqtrack.v40.registration.v1'
REGISTRATION_FILE = '40_registration.json'
BASELINE_COMMIT = '6dc7c4e'
BASELINE_IMPLEMENTATION = 'v38_clean1_concat_b1_selective_update_v1'
ALLOCATOR = 'backend:native,roundup_power2_divisions:2'
SOURCE_INVENTORY_POLICY = 'v40_active_python_explicit_retired_v1'
RETIRED_SOURCE_DIRECTORIES = ('models/ct_v31', 'models/ct_v36', 'models/ct_v37')
RETIRED_SOURCE_FILES = ('models/ctseqtrackv31.py', 'models/ctseqtrackv36.py', 'models/ctseqtrackv37.py')


def retired_source_path(relative):
    return relative in RETIRED_SOURCE_FILES or any(
        relative.startswith(directory + '/') for directory in RETIRED_SOURCE_DIRECTORIES)


def model_version(config=None):
    if config is not None and config.get('net_model') != 'ctseqtrackv40':
        raise ValueError('v40 identity requires ctseqtrackv40')
    return 'v40'


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
