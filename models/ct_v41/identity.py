"""V41 当前点恢复与 KITTI 两组协议；V40 仅为冻结源码基底。"""

EXECUTION_REVISION = 'v1'
SCHEMA = 'ct_seqtrack.current_point_recovery.v41'
RUNTIME_KEY = 'ct_v41_runtime'
EPOCH_SCHEMA = 'ct_seqtrack.v41.epoch_boundary.v1'
TRAINING_AUDIT_SCHEMA = 'ct_seqtrack.v41.training_audit.v1'
RESUME_SCHEMA = 'ct_seqtrack.v41.resume.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v41.candidate_records.v1'
COMPARE_SCHEMA = 'ct_seqtrack.v41.comparison.v1'
FULL_BATCH_SCHEMA = 'ct_seqtrack.v41.full_batch_check.v1'
RELIABILITY_PLOT_SCHEMA = 'ct_seqtrack.v41.reliability_plot.v1'
TRAINING_DIAGNOSTICS_SCHEMA = 'ct_seqtrack.v41.training_diagnostics.v1'
BENCHMARK_SCHEMA = 'ct_seqtrack.v41.engineering_benchmark.v1'
REGISTRATION_SCHEMA = 'ct_seqtrack.v41.registration.v1'
REGISTRATION_FILE = '41_registration.json'
BASELINE_COMMIT = 'ef1a9902a49e5f7f7cdfd3c00aa83a33d9b9895c'
BASELINE_IMPLEMENTATION = 'v40_dense_fa_b1_mtp_search_v1'
ALLOCATOR = 'backend:native,roundup_power2_divisions:2'
SOURCE_INVENTORY_POLICY = 'v41_active_python_explicit_retired_v1'
RETIRED_SOURCE_DIRECTORIES = ('models/ct_v31', 'models/ct_v36', 'models/ct_v37')
RETIRED_SOURCE_FILES = ('models/ctseqtrackv31.py', 'models/ctseqtrackv36.py', 'models/ctseqtrackv37.py')


def retired_source_path(relative):
    return relative in RETIRED_SOURCE_FILES or any(
        relative.startswith(directory + '/') for directory in RETIRED_SOURCE_DIRECTORIES)


def model_version(config=None):
    if config is not None and config.get('net_model') != 'ctseqtrackv41':
        raise ValueError('v41 identity requires ctseqtrackv41')
    return 'v41'


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
