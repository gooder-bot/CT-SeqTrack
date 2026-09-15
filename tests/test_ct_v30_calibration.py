"""阈值以行最大q登记，验证真实闭环预算、KITTI来源与策略身份。"""
import copy

import pytest

from utils.action_calibration_v30 import (
    threshold_candidates, calibrate_actions_v30, validate_action_calibration_v30)
from utils.action_calibration import sha256_json
from utils.dataset_protocol_v30 import build_dataset_manifest


def endpoint(frame, *, scene='0017', q=.2, score=.4, applied=False):
    initial = frame == 0
    return dict(tracklet_id=scene + '-car', scene_id=scene, frame_id=frame,
        is_initial=initial, structural_available=not initial,
        action_score=q, max_action_q=q, action_applied=applied and not initial,
        observation_success=1. if initial else .4,
        observation_precision=1. if initial else .4,
        candidate_success=1. if initial else score,
        candidate_precision=1. if initial else score,
        final_success=1. if initial else (score if applied else .4),
        final_precision=1. if initial else (score if applied else .4))


def test_candidates_use_row_maxima_numeric_thresholds_not_action_pool_or_masks():
    rows = [endpoint(i, q=value) for i, value in enumerate([0., .1, .4, .9])]
    for row in rows:
        row['action_score'] = -.8
        row['action_scores'] = [-1.] * 5 + [row['max_action_q']]
    policies = threshold_candidates(rows)
    thresholds = [p['threshold'] for p in policies]
    assert thresholds == pytest.approx([0., .16, .25, .4, .65, .8])
    # Q10/Q25 在初始never行上执行同一集合，但两者必须都保留闭环比较。
    assert sum(q > .16 for q in (.1, .4, .9)) == sum(q > .25 for q in (.1, .4, .9))


def test_calibration_kitti_closed_loops_midpoints_and_identity():
    manifest = build_dataset_manifest(dict(dataset='kitti_mf', version='kitti_tracking'))
    baseline = [endpoint(i, q=value) for i, value in enumerate([0., .1, .4, .9])]
    calls = []

    def runner(role, policy):
        calls.append((role, copy.deepcopy(policy)))
        threshold = policy.get('threshold', 2.)
        score = .9 if policy['kind'] == 'threshold' and .3 <= threshold <= .5 else .5
        applied = policy['kind'] != 'never'
        return [endpoint(i, scene='0017' if role == 'calibration' else '0018',
            q=baseline[i]['max_action_q'], score=score, applied=applied) for i in range(4)]

    artifact = calibrate_actions_v30(baseline, runner, checkpoint_sha256='checkpoint',
        config_sha256='config', scene_manifest=manifest, code_sha256='code')
    assert artifact['action_policy'] == {'kind': 'threshold', 'threshold': .4}
    assert artifact['calibration_policy_count'] == 10
    assert len([call for call in calls if call[0] == 'calibration']) == 9
    assert artifact['parameter_training_overlap'] is False
    validate_action_calibration_v30(artifact, 'checkpoint', 'config', manifest['content_sha256'], 'code')
    artifact['action_set'] = 'old_single_candidate'
    artifact.pop('artifact_sha256')
    artifact['artifact_sha256'] = sha256_json(artifact)
    with pytest.raises(ValueError, match='action_set'):
        validate_action_calibration_v30(artifact, 'checkpoint', 'config', manifest['content_sha256'], 'code')


def test_calibration_rejects_initial_rows_from_an_executing_policy():
    manifest = build_dataset_manifest(dict(dataset='kitti_mf', version='kitti_tracking'))
    with pytest.raises(ValueError, match='never-policy'):
        calibrate_actions_v30([endpoint(0), endpoint(1, score=.8, applied=True)], None,
            checkpoint_sha256='c', config_sha256='f', scene_manifest=manifest, code_sha256='s')
