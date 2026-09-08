"""v28 配置、工程隔离、矩阵和校准身份；不启动数据/GPU/训练。"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.online_contract import (
    build_online_resume_contract, validate_scratch_training_contract,
    validate_v28_observation_updates,
    validate_v28_evaluation_checkpoint,
)
from utils.action_calibration_v27 import (
    action_calibration_config_identity, calibrate_actions_v27,
    validate_action_calibration_v27, validate_scene_manifest, sha256_json,
)
from utils.v27_protocol import build_scene_manifest
from tools.run_ct_v28_matrix import build_matrix, execute_training
from tools.ct_action_v27_runtime import use_v27_runtime
from tools.preflight_ct_v28 import inspect_protocol

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('b0', 'b1_gru', 'b1_cfc', 'full_minus_b3', 'full')


def config(arm='b0', full=False):
    suffix = '_nuscenes_full' if full else ''
    return configure_ct_variant(load_yaml_config(ROOT / 'cfgs/ct_seqtrack' / f'28_{arm}{suffix}.yaml'))


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('full', (False, True))
def test_all_v28_arms_obey_scratch_shared_observation_and_numeric_contract(arm, full):
    cfg = config(arm, full)
    validate_scratch_training_contract(cfg)
    assert cfg['ct_b0_loss_reduction'] == 'reference_batch'
    assert cfg['workers'] == 12 and cfg['check_val_every_n_epoch'] == 5
    assert cfg['ct_batch_schema'] == 'ct_seqtrack.train.v4'
    assert not cfg['ct_adam_foreach'] and not cfg['ct_adam_fused']
    assert cfg['ct_observation_contract'] == 'seqtrack_reference_compatible_v1'
    assert cfg['ct_v28_expected_mini_car_updates_per_epoch'] == (None if full else 1262)


def test_reference_is_the_same_b0_implementation_and_training_definition():
    reference = configure_ct_variant(load_yaml_config(ROOT / 'cfgs/28_seqtrack_reference.yaml'))
    baseline = config()
    differences = {k for k in set(reference) | set(baseline) if reference.get(k) != baseline.get(k)}
    assert differences == {'experiment_name', 'ct_reference_baseline'}
    validate_scratch_training_contract(reference)


@pytest.mark.parametrize('field,value', (
    ('ct_allow_tf32', True), ('ct_deterministic_warn_only', True),
    ('ct_adam_foreach', None), ('workers', 4), ('epoch', 1),
    ('limit_train_batches', 100), ('ct_b0_loss_reduction', 'candidate_weighted')))
def test_formal_v28_rejects_numerical_and_training_drift(field, value):
    cfg = config(); cfg[field] = value
    with pytest.raises(ValueError):
        validate_scratch_training_contract(cfg)


def test_engineering_is_bounded_and_identity_separate_from_formal():
    cfg = config()
    formal = build_online_resume_contract(cfg)
    cfg.update(ct_engineering_check=True, epoch=2, workers=0,
               check_val_every_n_epoch=1, limit_train_batches=16,
               log_dir=str(ROOT / 'artifacts/ct_checks/test_v28_engineering_contract'))
    validate_scratch_training_contract(cfg)
    assert build_online_resume_contract(cfg) != formal
    for change in ({'log_dir': str(ROOT / 'output/bad')},
                   {'limit_train_batches': 101}, {'epoch': 60}):
        with pytest.raises(ValueError):
            validate_scratch_training_contract(dict(cfg, **change))


def test_v28_scene_manifest_and_preflight_keep_all_eight_training_scenes():
    splits = {'mini_train': [f'm{i}' for i in range(8)], 'mini_val': ['v0', 'v1']}
    cfg = config()
    report = inspect_protocol(SimpleNamespace(**cfg), splits)
    manifest = validate_scene_manifest(report['scene_manifest'])
    assert manifest['schema'].endswith('.v28')
    assert manifest['parameter_training_overlap']
    assert len(manifest['scenes']['train']) == 8
    assert set(manifest['scenes']['test']) == {'v0', 'v1'}
    bad = copy.deepcopy(manifest); bad['parameter_training_overlap'] = False
    bad.pop('content_sha256'); bad['content_sha256'] = sha256_json(bad)
    with pytest.raises(ValueError, match='overlap'):
        validate_scene_manifest(bad)


@pytest.mark.parametrize('stage,count', (('initial', 1), ('mini', 6), ('full', 30)))
def test_matrix_defaults_initial_b0_and_later_matrices_are_only_plans(tmp_path, stage, count):
    matrix, configs = build_matrix(stage, '/data/nuscenes', tmp_path / stage)
    assert matrix['run_count'] == len(configs) == count
    assert not matrix['execution_requested']
    for run in matrix['runs']:
        assert '--init_checkpoint' not in run['train_argv']
        assert [x['epoch'] for x in run['next_commands']] == [58, 59, 60]
    if stage == 'initial':
        assert matrix['runs'][0]['arm'] == 'b0'
    else:
        with pytest.raises(ValueError, match='initial B0'):
            execute_training(matrix, tmp_path)


def test_v28_config_auto_routes_calibration_and_environment_is_diagnostic():
    path = ROOT / 'cfgs/ct_seqtrack/28_full.yaml'
    assert use_v27_runtime(['--config', str(path)])
    assert use_v27_runtime(['--config=' + str(path)])
    assert use_v27_runtime(['--v27']) and use_v27_runtime(['--v28'])
    cfg = config('full'); changed = dict(cfg, ct_runtime_environment={'gpu': 'different'},
        ct_observation_update_count_check={'actual_updates_per_epoch': 1262})
    assert action_calibration_config_identity(cfg) == action_calibration_config_identity(changed)
    changed['ct_allow_tf32'] = True
    assert action_calibration_config_identity(cfg) != action_calibration_config_identity(changed)


def test_v28_calibration_schema_binds_training_overlap_and_cannot_be_downgraded():
    manifest = build_scene_manifest({'mini_train': [f'm{i}' for i in range(8)],
                                    'mini_val': ['v0', 'v1']}, 'v1.0-mini', enable_v28=True)
    def runner(role, policy):
        scene = manifest['scenes'][role][0]
        return [dict(tracklet_id=scene + '/track', scene_id=scene, frame_id=0,
                     is_initial=1, structural_available=0, action_score=0,
                     observation_success=1, observation_precision=1,
                     candidate_success=1, candidate_precision=1,
                     final_success=1, final_precision=1, action_applied=0)]
    artifact = calibrate_actions_v27(runner('calibration', {'kind': 'never'}), runner,
        checkpoint_sha256='c', config_sha256='f', scene_manifest=manifest, code_sha256='s')
    assert artifact['schema'] == 'ct_seqtrack.action_calibration.v28'
    assert artifact['selection_role'] == 'training_internal_threshold_fit'
    validate_action_calibration_v27(artifact, 'c', 'f', code_sha256='s')
    artifact['schema'] = 'ct_seqtrack.action_calibration.v27'
    with pytest.raises(ValueError, match='schema'):
        validate_action_calibration_v27(artifact, 'c', 'f', code_sha256='s')


def test_v28_eval_requires_matching_v28_weights_but_allows_deployment_metadata():
    cfg = config()
    saved = dict(cfg, ct_runtime_environment={'gpu': 'original'},
                 ct_observation_update_count_check={'updates_per_epoch': 1262})
    checkpoint = {'hyper_parameters': {'config': saved}}
    current = dict(cfg, test=True, checkpoint='epoch=060.ckpt',
                   ct_source_checkpoint_epoch=60, ct_runtime_environment={'gpu': 'evaluation'})
    validate_v28_evaluation_checkpoint(checkpoint, current)
    with pytest.raises(ValueError, match='v28 checkpoint'):
        validate_v28_evaluation_checkpoint({'hyper_parameters': {'config': dict(cfg, ct_enable_v28=False)}}, current)
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_v28_evaluation_checkpoint(checkpoint, dict(current, ct_b0_loss_reduction='candidate_weighted'))


def test_mini_car_update_budget_uses_natural_rows_and_is_identity_bound():
    cfg = config()
    report = validate_v28_observation_updates(cfg, sample_count=1262 * 16 + 7,
        batch_size=16, drop_last=True, observed_updates=1262)
    assert report['registered_formal_total_updates'] == 75720
    assert report['dropped_final_rows'] == 7
    for rows, updates, drop_last in ((1261 * 16, 1261, True), (1261 * 16, 1262, True),
                                     (1262 * 16 + 7, 1263, False)):
        with pytest.raises(ValueError):
            validate_v28_observation_updates(cfg, sample_count=rows, batch_size=16,
                                             drop_last=drop_last, observed_updates=updates)
    changed = dict(cfg, ct_v28_expected_mini_car_updates_per_epoch=1261)
    assert build_online_resume_contract(cfg) != build_online_resume_contract(changed)
    with pytest.raises(ValueError):
        validate_scratch_training_contract(changed)
    full = config(full=True)
    report = validate_v28_observation_updates(full, sample_count=999 * 16,
        batch_size=16, drop_last=True, observed_updates=999)
    assert report['expected_updates_per_epoch'] is None
