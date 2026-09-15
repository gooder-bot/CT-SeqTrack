"""v30：按每行最大动作 q 构造阈值，最多十个真实闭环策略。"""
import json
import math
from pathlib import Path

import numpy as np

from utils.action_calibration import sha256_file, sha256_json
from utils.action_calibration_v27 import (
    CODE_FILES as LEGACY_CODE_FILES, action_calibration_config_identity,
    normalize_policy, normalize_rows, summarize_rows, paired_scene_bootstrap)

SCHEMA = 'ct_seqtrack.action_calibration.v30'
ROWS_SCHEMA = 'ct_seqtrack.action_rows.v30'
SCORE_DEFINITION = 'mean(predicted_delta_success,predicted_delta_precision)'
ACTION_SET = 'three_modes_half_full_xy_v1'
POLICY_FITTING = 'row_max_q_closed_loop_ten_v1'
CODE_FILES = tuple(dict.fromkeys((*LEGACY_CODE_FILES,
    'utils/action_calibration_v30.py', 'utils/v30_contracts.py',
    'utils/acquisition_v30.py', 'utils/v30_crop.py', 'utils/v30_policy.py',
    'utils/v30_training.py', 'utils/v30_evaluation.py', 'utils/masked_observation.py',
    'utils/v30_action_output.py', 'utils/v30_funnel.py', 'utils/v30_reporting.py',
    'utils/dataset_protocol_v30.py', 'utils/data_cache_v30.py',
    'datasets/kitti_mf.py', 'datasets/temporal_protocol.py',
    'models/ct_v2/action_v30.py', 'models/ct_v2/evidence_v30.py')))


def code_file_hashes(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    return {name: sha256_file(root / name) for name in CODE_FILES}


def code_content_sha256(root=None):
    return sha256_json(code_file_hashes(root))


def _rows(rows, role, manifest):
    rows = normalize_rows(rows)
    if not {row['scene_id'] for row in rows} <= set(manifest['scenes'][role]):
        raise ValueError(f'v30 {role} endpoints contain another split')
    return rows


def threshold_candidates(rows):
    """只按阈值数值去重；不按初始轨迹执行 mask 去重。"""
    scores = []
    for row in rows:
        if row['structural_available'] and not row['is_initial']:
            score = float(row['max_action_q'])
            if not math.isfinite(score):
                raise ValueError('v30 max_action_q must be finite')
            scores.append(score)
    values = [0.]
    if scores:
        values.extend(np.quantile(scores, [.1, .25, .5, .75, .9]).tolist())
    return [{'kind': 'threshold', 'threshold': float(value)} for value in sorted(set(values))]


def _rank(item):
    metric = item['metrics']
    return metric['U'], metric['S'], metric['P'], -metric['actions']


def calibrate_actions_v30(calibration_rows, runner, *, checkpoint_sha256,
                          config_sha256, scene_manifest, code_sha256=None):
    from utils.dataset_protocol_v30 import validate_dataset_manifest
    validate_dataset_manifest(scene_manifest)
    baseline = _rows(calibration_rows, 'calibration', scene_manifest)
    if any(row['action_applied'] or row['final_success'] != row['observation_success']
           or row['final_precision'] != row['observation_precision'] for row in baseline):
        raise ValueError('v30 threshold candidates require a never-policy rollout')
    ids = {(row['tracklet_id'], row['frame_id']) for row in baseline}
    never = {'kind': 'never'}
    evaluated = [{'policy': never, 'metrics': summarize_rows(baseline)}]
    policy_rows = {sha256_json(never): baseline}

    def evaluate(policy):
        rows = _rows(runner('calibration', policy), 'calibration', scene_manifest)
        if {(row['tracklet_id'], row['frame_id']) for row in rows} != ids:
            raise ValueError('v30 closed loops require identical complete endpoints')
        evaluated.append({'policy': policy, 'metrics': summarize_rows(rows)})
        policy_rows[sha256_json(policy)] = rows

    thresholds = threshold_candidates(baseline)
    for policy in [{'kind': 'always'}, *thresholds]:
        evaluate(policy)
    # 只围绕最佳阈值候选细化，不用 dev 改阈值；两个中点最多增加两次闭环。
    threshold_results = [item for item in evaluated if item['policy']['kind'] == 'threshold']
    best_threshold = max(threshold_results, key=_rank)['policy']['threshold']
    values = [p['threshold'] for p in thresholds]
    index = values.index(best_threshold)
    refinement = []
    for neighbour in (index - 1, index + 1):
        if 0 <= neighbour < len(values):
            midpoint = .5 * (best_threshold + values[neighbour])
            if midpoint not in values:
                refinement.append({'kind': 'threshold', 'threshold': midpoint})
    for policy in refinement:
        evaluate(policy)
    if len(evaluated) > 10:
        raise RuntimeError('v30 calibration exceeded registered ten-policy budget')
    chosen = max(evaluated, key=_rank)
    dev = _rows(runner('dev', chosen['policy']), 'dev', scene_manifest)
    dev_never = (dev if chosen['policy'] == never else
                 _rows(runner('dev', never), 'dev', scene_manifest))
    artifact = dict(schema=SCHEMA, action_policy=chosen['policy'],
        checkpoint_sha256=str(checkpoint_sha256), config_sha256=str(config_sha256),
        code_content_sha256=code_sha256 or code_content_sha256(), source_files=list(CODE_FILES),
        action_set=ACTION_SET, policy_fitting_contract=POLICY_FITTING,
        score_definition=SCORE_DEFINITION, metric_mode='benchmark_compat',
        comparator='>', metric_threshold_count=21,
        scene_manifest=scene_manifest, scene_manifest_sha256=scene_manifest['content_sha256'],
        parameter_training_overlap=scene_manifest['parameter_training_overlap'],
        selection_role=('training_internal_policy_fit' if scene_manifest['parameter_training_overlap']
                        else 'held_out_policy_fit'),
        selection_complete=True, dev_refit=False,
        threshold_population='never_rollout_row_max_action_q',
        calibration_closed_loop=evaluated, calibration_policy_count=len(evaluated),
        calibration_selected_metrics=chosen['metrics'],
        dev_locked_metrics=summarize_rows(dev), dev_never_metrics=summarize_rows(dev_never),
        calibration_gain_interval=paired_scene_bootstrap(policy_rows[sha256_json(chosen['policy'])], baseline),
        dev_gain_interval=paired_scene_bootstrap(dev, dev_never),
        calibration_rows_sha256=sha256_json(baseline))
    artifact['artifact_sha256'] = sha256_json(artifact)
    return artifact


def validate_action_calibration_v30(artifact, checkpoint_sha256, config_sha256,
                                    scene_manifest_sha256=None, code_sha256=None):
    from utils.dataset_protocol_v30 import validate_dataset_manifest
    body = dict(artifact)
    digest = body.pop('artifact_sha256', None)
    if digest != sha256_json(body):
        raise ValueError('v30 calibration content hash mismatch')
    expected = dict(schema=SCHEMA, checkpoint_sha256=str(checkpoint_sha256),
        config_sha256=str(config_sha256), code_content_sha256=code_sha256 or code_content_sha256(),
        action_set=ACTION_SET, policy_fitting_contract=POLICY_FITTING,
        score_definition=SCORE_DEFINITION, metric_mode='benchmark_compat', comparator='>',
        metric_threshold_count=21, selection_complete=True, dev_refit=False,
        threshold_population='never_rollout_row_max_action_q')
    if scene_manifest_sha256 is not None:
        expected['scene_manifest_sha256'] = scene_manifest_sha256
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f'v30 calibration {key} mismatch')
    manifest = artifact['scene_manifest']
    validate_dataset_manifest(manifest)
    if (artifact['scene_manifest_sha256'] != manifest['content_sha256']
            or artifact['parameter_training_overlap'] != manifest['parameter_training_overlap']):
        raise ValueError('v30 calibration dataset provenance mismatch')
    if not 3 <= artifact.get('calibration_policy_count', 0) <= 10:
        raise ValueError('v30 calibration policy budget mismatch')
    normalize_policy(artifact['action_policy'])
    return artifact


def install_v30_action_calibration(model, config, *, scene_splits=None, code_sha256=None):
    from utils.dataset_protocol_v30 import build_dataset_manifest
    get = config.get if isinstance(config, dict) else lambda k, d=None: getattr(config, k, d)
    router = model.ct_joint_router
    router.install_policy({'kind': 'never'})
    router.calibrated.fill_(False)
    model._ct_action_calibration = None
    status = dict(schema=SCHEMA, loaded=False, fallback='observation', reason='missing_calibration_artifact')
    model._ct_action_calibration_status = status
    path = get('ct_action_calibration_path')
    if path:
        try:
            checkpoint = get('checkpoint')
            if not checkpoint:
                raise ValueError('v30 calibration requires evaluation checkpoint')
            manifest = build_dataset_manifest(config, scene_splits=scene_splits)
            artifact = json.loads(Path(path).read_text(encoding='utf-8'))
            validate_action_calibration_v30(artifact, sha256_file(checkpoint),
                sha256_json(action_calibration_config_identity(config)),
                manifest['content_sha256'], code_sha256)
            router.install_policy(artifact['action_policy'])
            model._ct_action_calibration = artifact
            status.update(loaded=True, fallback=None, reason='ok', action_policy=artifact['action_policy'],
                          artifact_sha256=artifact['artifact_sha256'])
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
            router.install_policy({'kind': 'never'})
            router.calibrated.fill_(False)
            status['reason'] = f'{type(error).__name__}:{error}'
    model._ct_action_calibration_error = None if status['loaded'] else status['reason']
    return status
