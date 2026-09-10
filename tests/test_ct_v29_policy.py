"""v29即时效用、共享动作转移与训练内部闭环策略拟合。"""
from copy import deepcopy
from collections import Counter
import random

import numpy as np
import pytest
import torch

from models.ct_v2.action_v27 import B3UtilityUpdater
from tests.test_ct_v27_actions import _inputs, _rows, _manifest
from tests.test_ct_v27_training import _loss_case
from utils.action_calibration_v27 import (
    action_calibration_config_identity, calibrate_actions_v27, policy_mask,
    shortlist_policies, validate_action_calibration_v27, sha256_json,
)
from utils.sampling_utils import stable_uint32_seed
from utils.v27_training import compute_b3_utility_loss
from utils.v29_contracts import V29_CONTRACTS
from utils.v29_policy import (
    mechanism_behavior_policy, policy_transition, BEHAVIOR_NAMESPACE,
    TRANSITION_CONTRACT, UTILITY_TARGET_CONTRACT, POLICY_FITTING_CONTRACT,
)


def test_behavior_assignment_is_tracklet_stable_and_does_not_consume_rng():
    torch.manual_seed(17); random.seed(17); np.random.seed(17)
    torch_before, py_before, np_before = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    counts, changed_epochs = Counter(), 0
    for i in range(400):
        key = f'nuscenes/train/scene-token/instance-{i}'
        policy = mechanism_behavior_policy(42, 0, key)
        bucket = stable_uint32_seed(42, 0, key, BEHAVIOR_NAMESPACE) % 4
        expected = ({'kind': 'never'} if bucket == 0 else {'kind': 'always'} if bucket == 1
                    else {'kind': 'threshold', 'threshold': 0.})
        assert policy == expected == mechanism_behavior_policy(42, 0, key)
        counts[policy['kind']] += 1
        changed_epochs += policy != mechanism_behavior_policy(42, 1, key)
    assert set(counts) == {'never', 'always', 'threshold'} and changed_epochs > 0
    assert torch.equal(torch_before, torch.get_rng_state()) and py_before == random.getstate()
    assert np_before[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(np_before[1], np.random.get_state()[1])
    for key in ('', 'unknown', 'eval', None):
        with pytest.raises(ValueError, match='tracklet key'):
            mechanism_behavior_policy(42, 0, key)


@pytest.mark.parametrize('policy', [{'kind': 'never'}, {'kind': 'always'},
                                  {'kind': 'threshold', 'threshold': 0.}])
def test_shared_transition_is_detached_finite_structural_and_strict(policy):
    obs = torch.tensor([[0., 0., 1., .3]] * 5, requires_grad=True)
    candidate = obs.detach().clone(); candidate[:, 0] = .75
    candidate[:, 2:] = -10  # 部署只用有界XY，Z/yaw仍是B0。
    candidate[4, 0] = float('nan')
    score = torch.tensor([-.3, 0., .4, .5, .8], requires_grad=True)
    valid = torch.tensor([1., 1., 1., 0., 1.])
    snapshot = obs.detach().clone()
    final, accepted = policy_transition(obs, candidate, valid, score, policy)
    expected = torch.tensor(policy_mask(score.detach().numpy(), valid.numpy() > 0, policy))
    expected[4] = False
    assert torch.equal(accepted, expected)
    assert torch.equal(final[:, 2:], obs[:, 2:])
    assert torch.equal(final[~accepted], obs[~accepted]) and torch.isfinite(final).all()
    assert torch.equal(obs, snapshot) and not final.requires_grad
    if policy['kind'] == 'always':
        assert accepted[0]  # 负q也可训练访问accepted状态，不能退化为全never。
    if policy['kind'] == 'threshold':
        assert not accepted[1]  # q==tau不执行。


@pytest.mark.parametrize('policy', [{'kind': 'never'}, {'kind': 'always'},
                                  {'kind': 'threshold', 'threshold': .1}])
def test_router_shared_transition_preserves_v28_math_and_low_presence_actions(policy):
    torch.manual_seed(42)
    legacy = B3UtilityUpdater(require_calibration=True)
    shared = B3UtilityUpdater(require_calibration=True, use_shared_policy_transition=True)
    shared.load_state_dict(legacy.state_dict())
    inputs = _inputs()
    assert torch.equal(legacy(**inputs)[0], shared(**inputs)[0])  # 未装策略必须同样fallback。
    legacy.install_policy(policy); shared.install_policy(policy)
    a, ao = legacy(**inputs); b, bo = shared(**inputs)
    assert torch.equal(a, b)
    for key in ('ct_b3_action_score', 'ct_b3_final_gate', 'ct_router_bounded_residual_xy'):
        assert torch.equal(ao[key], bo[key])
    if policy['kind'] == 'always':
        assert bo['ct_b3_final_gate'].all() and not inputs['extension_presence_probability'].any()
    bo['ct_b3_action_score'].sum().backward()
    assert inputs['base_evidence'].grad is None and inputs['extension_evidence'].grad is None
    assert shared.expected_success_gain_head.weight.grad is not None


def test_v29_q_h1_loss_ignores_h3_values_and_keeps_negative_unaccepted_candidates():
    data, output = _loss_case()
    output['ct_router_applied_gate'] = torch.zeros(2)
    output['ct_b2_extension_presence_probability'] = torch.zeros(2)
    cfg = dict(ct_enable_v29=True, degrees=False, up_axis=[0, 0, 1], IoU_space=3)
    result = compute_b3_utility_loss(data, output, cfg)
    assert torch.equal(result['loss_success'], result['h1_success_gain'].square().mean())
    assert torch.equal(result['loss_precision'], result['h1_precision_gain'].square().mean())
    assert result['valid'].tolist() == [1., 1.]
    assert result['help_label'].tolist() == [1., 0.] and result['harm_label'].tolist() == [0., 1.]
    variables = [output[k] for k in ('ct_b3_expected_success_gain', 'ct_b3_expected_precision_gain',
                                     'ct_b3_help_logit', 'ct_b3_harm_logit')]
    gradients = torch.autograd.grad(result['loss'], variables, retain_graph=True)
    changed = deepcopy(data)
    changed['ct_h3_valid'].fill_(1)
    changed['ct_h3_success_gain'].fill_(-.9); changed['ct_h3_precision_gain'].fill_(.9)
    other = compute_b3_utility_loss(changed, output, cfg)
    assert torch.equal(result['loss'], other['loss'])
    assert all(torch.equal(a, b) for a, b in zip(gradients, torch.autograd.grad(other['loss'], variables)))
    assert output['observation_aux_estimation_boxes'].grad is None
    assert output['ct_router_bounded_residual_xy'].grad is None
    # v28仍保留原H3目标，v29不静默改写旧版本。
    legacy = compute_b3_utility_loss(data, output, dict(cfg, ct_enable_v29=False))
    assert not torch.equal(result['loss'], legacy['loss'])


def test_v29_shortlist_always_retains_controls_and_at_most_three_thresholds():
    rows = _rows('fit')
    for row in rows[1:]:
        row['candidate_success'] = row['candidate_precision'] = .1
    shortlist, screen = shortlist_policies(rows, include_controls=True)
    assert shortlist[0]['policy'] == {'kind': 'always'}
    assert sum(x['policy']['kind'] == 'threshold' for x in shortlist) <= 3
    assert {x['policy']['kind'] for x in screen} >= {'never', 'always'}
    for row in rows:
        row['structural_available'] = 0
    shortlist, screen = shortlist_policies(rows, include_controls=True)
    assert [x['policy'] for x in shortlist] == [{'kind': 'always'}]
    assert [x['policy'] for x in screen] == [{'kind': 'never'}, {'kind': 'always'}]


def _fit_v29(tie=False):
    manifest = _manifest(False)
    calls = []
    def runner(role, policy):
        assert role in ('calibration', 'dev')
        calls.append((role, deepcopy(policy)))
        rows = _rows(manifest['scenes'][role][0])
        applied = policy_mask([r['action_score'] for r in rows],
                              [r['structural_available'] for r in rows], policy)
        for index in (1, 2):
            rows[index]['action_applied'] = int(applied[index])
            # one-step筛选看起来有害，但真实always闭环有益，必须实际比较。
            rows[index]['candidate_success'] = rows[index]['candidate_precision'] = .1
            score = .5 if tie or not applied[index] else .9 if policy['kind'] == 'always' else .6
            rows[index]['final_success'] = rows[index]['final_precision'] = score
        return rows
    baseline = runner('calibration', {'kind': 'never'})
    artifact = calibrate_actions_v27(baseline, runner, checkpoint_sha256='checkpoint',
        config_sha256='config', scene_manifest=manifest, code_sha256='source', enable_v29=True)
    return artifact, calls


def test_v29_internal_fit_uses_actual_closed_loops_and_locked_dev():
    artifact, calls = _fit_v29()
    assert artifact['schema'] == 'ct_seqtrack.action_calibration.v29'
    assert artifact['action_policy'] == {'kind': 'always'}
    assert artifact['utility_target'] == UTILITY_TARGET_CONTRACT
    assert artifact['transition_contract'] == TRANSITION_CONTRACT
    assert artifact['policy_fitting_contract'] == POLICY_FITTING_CONTRACT
    assert artifact['selection_role'] == 'training_internal_policy_fit'
    assert artifact['parameter_training_overlap'] is True and artifact['dev_refit'] is False
    cal_policies = [entry['policy'] for entry in artifact['calibration_closed_loop']]
    assert {'kind': 'never'} in cal_policies and {'kind': 'always'} in cal_policies
    assert len(cal_policies) <= 5
    assert [p for role, p in calls if role == 'dev'] == [artifact['action_policy'], {'kind': 'never'}]
    validate_action_calibration_v27(artifact, 'checkpoint', 'config', code_sha256='source', enable_v29=True)
    with pytest.raises(ValueError, match='schema'):
        validate_action_calibration_v27(artifact, 'checkpoint', 'config', code_sha256='source', enable_v29=False)
    modified = deepcopy(artifact); modified['utility_target'] = 'mixed_h3'
    modified.pop('artifact_sha256'); modified['artifact_sha256'] = sha256_json(modified)
    with pytest.raises(ValueError, match='utility_target'):
        validate_action_calibration_v27(modified, 'checkpoint', 'config', code_sha256='source', enable_v29=True)
    assert _fit_v29(tie=True)[0]['action_policy'] == {'kind': 'never'}


def test_v29_identity_binds_every_registered_contract_value():
    config = dict(V29_CONTRACTS, ct_enable_v29=True, ct_enable_v28=True, ct_enable_v27=True)
    identity = action_calibration_config_identity(config)
    assert identity['ct_enable_v29'] is True
    for key, value in V29_CONTRACTS.items():
        assert identity[key] == value
        changed = dict(config, **{key: str(value) + '_changed'})
        assert action_calibration_config_identity(changed) != identity
        missing = dict(config); missing.pop(key)
        assert action_calibration_config_identity(missing) != identity
    assert action_calibration_config_identity(dict(config, ct_runtime_environment={'gpu': 7})) == identity
