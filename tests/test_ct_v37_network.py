"""v37主路逐位保持、共同训练图与真实证据/梯度权限。"""
import copy
from dataclasses import replace

import pytest
import torch

from models.ct_v31.model import JointTracker as V35Tracker
from models.ct_v31.observation import B0Observation
from models.ct_v31.prior import PhysicalTimePrior
from models.ct_v36.contracts import EvidenceHypotheses
from models.ct_v37.config import normalize_config, as_v35_config
from models.ct_v37.model import JointTracker, select_candidates
from models.ct_v37.decoder import SharedHypothesisDecoder
from models.ct_v37.evidence import B2IdentityEvidence, build_live_modes
from models.ct_v37.losses import pairwise_gap_loss
from models.ct_v37.verification import FinalBoxVerifier, LocalBoxReader
from tests.test_ct_v31_joint import make_batch


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(37)
    yield
    torch.set_num_threads(previous)


def config(arm='full'):
    return normalize_config(dict(v37_arm=arm, ct_engineering_check=True,
        point_sample_size=16, batch_size=2, workers=0, epoch=2))


def batch(b=2, n=16):
    values = make_batch(b, n)
    values['history_is_initial'] = torch.tensor([[True, False, False]]).expand(b, -1).clone()
    values['extension_ids'] = torch.where(values['extension_valid'], values['extension_ids'] + 1000, -1)
    return values


@pytest.mark.parametrize('arm', ['b0', 'b1'])
def test_baseline_two_adam_steps_match_v35_output_loss_gradient_bn_rng(arm):
    cfg = config(arm)
    torch.manual_seed(73)
    old = V35Tracker(as_v35_config(cfg)).train()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    new = JointTracker(cfg).train()
    assert torch.equal(old_rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    optimizers = [torch.optim.Adam(model.parameters(), lr=1.5e-4) for model in (old, new)]
    values = batch()
    for _ in range(2):
        rng = torch.get_rng_state().clone()
        results = []
        for model, optimizer in zip((old, new), optimizers):
            torch.set_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
            output = model(values)
            losses = model.compute_losses(values, output)
            losses['loss_total'].backward()
            gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
            optimizer.step()
            results.append((output, losses, gradients, torch.get_rng_state().clone()))
        for attr in ('accepted_box', 'selected_index', 'selected_quality'):
            assert torch.equal(getattr(results[0][0], attr), getattr(results[1][0], attr)), attr
        assert results[0][1].keys() == results[1][1].keys()
        for name, loss in results[0][1].items():
            assert torch.equal(loss, results[1][1][name]), name
        for name, gradient in results[0][2].items():
            other = results[1][2][name]
            assert (gradient is None and other is None) or torch.equal(gradient, other), name
        assert torch.equal(results[0][3], results[1][3])
        for name, value in old.state_dict().items():
            assert torch.equal(value, new.state_dict()[name]), name
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in left:
                assert torch.equal(left[key], right[key]), key


@pytest.mark.parametrize('training', [False, True])
def test_shadow_and_full_train_identical_candidates_losses_gradients_and_only_change_policy(training):
    torch.manual_seed(42)
    shadow = JointTracker(config('b1_b2')).train(training)
    torch.manual_seed(42)
    full = JointTracker(config('full')).train(training)
    assert shadow.state_dict().keys() == full.state_dict().keys()
    for name, value in shadow.state_dict().items():
        assert torch.equal(value, full.state_dict()[name]), name
    values = batch()
    rng = torch.get_rng_state().clone()
    outputs, rng_after = [], []
    for model in (shadow, full):
        torch.set_rng_state(rng)
        outputs.append(model(values))
        rng_after.append(torch.get_rng_state().clone())
    assert torch.equal(*rng_after)
    for left, right in ((outputs[0].candidates.boxes, outputs[1].candidates.boxes),
                        (outputs[0].verification.logits, outputs[1].verification.logits)):
        assert torch.equal(left, right)
    loss_sets = []
    for model, output in zip((shadow, full), outputs):
        output.loss_statistics = {}
        losses = model.compute_losses(values, output)
        losses['loss_total'].backward()
        loss_sets.append(losses)
    for name in loss_sets[0]:
        assert torch.equal(loss_sets[0][name], loss_sets[1][name]), name
    for (name, left), (_, right) in zip(shadow.named_parameters(), full.named_parameters()):
        assert (left.grad is None and right.grad is None) or torch.equal(left.grad, right.grad), name
    forced = replace(outputs[0].verification, logits=outputs[0].verification.logits.detach().clone())
    forced.logits[:, 1, :2] = 10.
    assert (select_candidates(outputs[0].candidates, forced, apply_policy=False)[1] == 0).all()
    selected = select_candidates(outputs[0].candidates, forced, apply_policy=True)
    assert (selected[1] == 1).all() and (selected[2] == 0).all()


def _decoder_inputs():
    values = batch(1, 24)
    observation = B0Observation(8, query_context=True, coarse_condition=True).eval()(values)
    prior = PhysicalTimePrior().eval()(values)
    evidence = EvidenceHypotheses.empty(values['target_box'])
    evidence.point_valid[:, :8] = True
    evidence.point_ids[:, :8] = torch.arange(1000, 1008)
    evidence.point_xyz[:, :8] = values['points'][:, -1, :8, :3]
    evidence.point_features[:, :8] = torch.randn(1, 8, 64)
    evidence.selected_identity_logits[:, :8] = 3.
    evidence.mode_valid[:] = True
    evidence.members[:, 0, :4] = True
    evidence.members[:, 1, 4:8] = True
    evidence.centers_xyz = torch.tensor([[[.3, 0., 0.], [2., 0., 0.]]], requires_grad=True)
    return values, observation, prior, evidence


@pytest.mark.parametrize('training', [False, True])
def test_q0_history_do_not_read_added_evidence_or_memory_and_use_lwh_neighbors(training):
    values, observation, prior, evidence = _decoder_inputs()
    decoder = SharedHypothesisDecoder(dropout=.2).train(training)
    torch.nn.init.normal_(decoder.pose_head.weight, std=.03)
    torch.nn.init.normal_(decoder.mode_pose_head.weight, std=.03)
    torch.nn.init.normal_(decoder.local_reader.output.weight, std=.03)
    rng = torch.get_rng_state().clone()
    first = decoder(observation, evidence, values, prior)
    validity = evidence.point_valid.clone()
    validity[:, 2:6] = False
    members = evidence.members.clone()
    members[:, 0, 1] = False
    members[:, 1, 7] = False
    changed = replace(evidence, point_features=evidence.point_features + 5,
        point_xyz=evidence.point_xyz + 20, memory_features=evidence.memory_features + 99,
        centers_xyz=evidence.centers_xyz + 3, point_valid=validity, members=members)
    torch.set_rng_state(rng)
    second = decoder(observation, changed, values, prior)
    assert torch.equal(first.boxes[:, 0], second.boxes[:, 0])
    assert torch.equal(first.history_boxes, second.history_boxes)
    assert not torch.equal(first.boxes[:, 2:], second.boxes[:, 2:])
    assert decoder.local_reader.__class__.__module__ == 'models.ct_v31.local_observation'
    assert not hasattr(decoder, 'memory_projection')


def test_mode_query_detached_but_center_output_and_identity_weights_remain_live():
    values, observation, prior, evidence = _decoder_inputs()
    output = SharedHypothesisDecoder(dropout=0.).eval()(observation, evidence, values, prior)
    feature_grad = torch.autograd.grad(output.features.sum(), evidence.centers_xyz,
        retain_graph=True, allow_unused=True)[0]
    assert feature_grad is None
    center_grad = torch.autograd.grad(output.boxes[:, 2:, :3].sum(), evidence.centers_xyz)[0]
    assert torch.equal(center_grad, torch.ones_like(center_grad))
    votes = torch.tensor([[[0., 0., 0.], [.2, .1, 0.], [3., 0., 0.]]], requires_grad=True)
    weights = torch.ones(1, 3, requires_grad=True)
    centers, valid, members, _ = build_live_modes(votes, weights, torch.ones(1, 3, dtype=torch.bool), torch.arange(3)[None])
    assert valid.all() and not (members[:, 0] & members[:, 1]).any()
    gradients = torch.autograd.grad(centers.square().sum(), (votes, weights))
    assert all(value.abs().sum() > 0 for value in gradients)


def test_memory_can_change_identity_but_not_fixed_id_geometry_votes():
    values, observation, _, _ = _decoder_inputs()
    model = B2IdentityEvidence().eval()
    cache = model.encode_current(observation, values)
    indices = torch.full((1, 256), -1, dtype=torch.long)
    indices[:, :24] = torch.arange(24)
    first_identity = model.identify(cache, observation, values)[0]
    first_geometry = model.geometry_votes(cache, indices)
    other = dict(values, memory_points=values['memory_points'] + 5.)
    assert not torch.equal(first_identity, model.identify(cache, observation, other)[0])
    for left, right in zip(first_geometry, model.geometry_votes(cache, indices)):
        assert torch.equal(left, right)


def test_verifier_has_no_upstream_gradient_after_nonzero_exit_and_permissions_are_current_only():
    values, observation, prior, evidence = _decoder_inputs()
    candidates = SharedHypothesisDecoder(dropout=0.).eval()(observation, evidence, values, prior)
    verifier = FinalBoxVerifier().eval()
    torch.nn.init.normal_(verifier.output.weight, std=.02)
    output = verifier(candidates, observation, evidence, values, prior)
    upstream = torch.autograd.grad(output.logits.sum(),
        (observation.point_features, observation.coarse_features, candidates.boxes,
         prior.feature, evidence.centers_xyz), retain_graph=True, allow_unused=True)
    assert all(value is None for value in upstream)
    output.logits.sum().backward()
    assert evidence.centers_xyz.grad is None
    assert verifier.local_reader.output.weight.grad.abs().sum() > 0
    n = values['points'].shape[2]
    assert not output.authorized[:, 2, n + 4:n + 8].any()
    assert not output.authorized[:, 3, n:n + 4].any()
    assert output.support_ids.shape[-1] == n + 256


def test_full_empty_fallback_has_zero_commit_quality_and_no_measurement_alias():
    values = batch(1)
    values['point_valid'].zero_()
    values['extension_valid'].zero_()
    model = JointTracker(config()).train()
    output = model(values)
    assert output.candidates.valid.tolist() == [[False, True, False, False]]
    assert output.selected_index.tolist() == [1]
    assert output.selected_quality.tolist() == [0.]
    assert output.q0_is_fallback.all()
    losses = model.compute_losses(values, output)
    assert losses['loss_modes'] == 0 and losses['loss_vote'] == 0
    losses['loss_total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_gt_changes_labels_only_and_background_modes_keep_scoring_supervision():
    values = batch(1)
    model = JointTracker(config()).eval()
    first = model(values)
    changed = copy.deepcopy(values)
    for key in ('target_box', 'history_target_boxes', 'physical_displacement', 'segmentation_labels',
                'bc_targets', 'extension_labels', 'acquisition_target'):
        changed[key] = changed[key] * 0 + (0 if key == 'extension_labels' else 99)
    second = model(changed)
    assert torch.equal(first.candidates.boxes, second.candidates.boxes)
    assert torch.equal(first.verification.logits, second.verification.logits)
    changed['target_box'] = values['target_box']
    changed['segmentation_labels'].zero_()
    losses = model.compute_losses(changed, second)
    assert losses['loss_modes'] == 0 and losses['loss_vote'] == 0
    assert losses['loss_mode_commit_quality'] > 0


def test_pairwise_ties_return_graph_zero_and_gap_ranking_prefers_true_order():
    score = torch.tensor([[.8, .2, .4]], requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    tied = pairwise_gap_loss(score, torch.full_like(score, .5), valid)
    assert tied == 0 and tied.requires_grad
    tied.backward()
    assert torch.equal(score.grad, torch.zeros_like(score))
    target = torch.tensor([[1., 0., .5]])
    assert pairwise_gap_loss(score, target, valid) < pairwise_gap_loss(1. - score, target, valid)


def test_common_b0_b1_initialization_is_identical_across_all_four_arms():
    models = []
    for arm in ('b0', 'b1', 'b1_b2', 'full'):
        torch.manual_seed(413)
        models.append(JointTracker(config(arm)))
    reference = models[1].state_dict()
    for model in models:
        for name, value in model.state_dict().items():
            if name.startswith(('observation.', 'decoder.', 'prior.')) and name in reference:
                assert torch.equal(value, reference[name]), name


def test_mode_seed_loss_reaches_vote_identity_and_shared_current_b0_even_with_zero_pose_head():
    model = JointTracker(config()).eval()
    values = batch(1)
    output = model(values)
    losses = model.compute_losses(values, output)
    assert losses['loss_modes'] > 0
    gradients = torch.autograd.grad(losses['loss_modes'],
        (model.evidence.vote_head[-1].weight, model.evidence.identity_head[-1].weight,
         output.observation.point_features), allow_unused=True)
    assert all(value is not None and value.abs().sum() > 0 for value in gradients)


def test_verifier_lwh_neighborhood_is_not_fixed_one_meter():
    reader = LocalBoxReader().eval()
    size = torch.tensor([[4., 2., 2.]])
    # first corner=(-2,-1,-1)，点离该角2米，但LWH归一化距离仅0.5。
    pool = dict(ids=torch.tensor([[8]]), valid=torch.tensor([[True]]),
        xyz=torch.tensor([[[0., -1., -1.]]]), features=torch.ones(1, 1, 64),
        probability=torch.ones(1, 1), source=torch.zeros(1, 1), base_count=1)
    _, stats = reader(pool, torch.zeros(1, 1, 4), torch.ones(1, 1, 1, dtype=torch.bool), size)
    assert stats['local_neighbor_count'][0, 0, 0].item() == 1


def test_auxiliary_diagnostic_contributions_match_actual_weighted_losses():
    model = JointTracker(config()).eval()
    values = batch()
    # 同batch不同FG数量，检查identity/reliability仍按原全batch FG/BG归约。
    values['extension_labels'][0, 3:24] = 0
    output = model(values)
    output.loss_statistics = {}
    losses = model.compute_losses(values, output)
    for key, coefficient in (('loss_identity', .2), ('loss_reliability', .1)):
        contribution = sum(output.loss_statistics[key + '_' + group]['batch_contribution'].sum()
                           for group in ('foreground', 'background'))
        torch.testing.assert_close(contribution, coefficient * losses[key].detach())
    for key, coefficient in (('loss_vote', 1.), ('loss_modes_center', 1.), ('loss_modes_angle', 1.),
            ('loss_success', .125), ('loss_precision', .125), ('loss_mode_commit_quality', .5), ('loss_ranking', .1)):
        torch.testing.assert_close(output.loss_statistics[key]['batch_contribution'].sum(), coefficient * losses[key].detach())


def test_final_decoded_center_deduplicates_even_when_seed_centers_are_separate():
    values, observation, prior, evidence = _decoder_inputs()
    decoder = SharedHypothesisDecoder(dropout=0.).eval()
    initial = decoder(observation, evidence, values, prior)
    assert initial.valid[:, 2:].all()
    assert (initial.seed_boxes[:, 0, :2] - initial.seed_boxes[:, 1, :2]).norm() > .75
    difference = initial.features[0, 3].detach() - initial.features[0, 2].detach()
    displacement = initial.boxes[0, 3, 0].detach() - initial.boxes[0, 2, 0].detach()
    with torch.no_grad():
        decoder.mode_pose_head.weight[0] = -(displacement - .2) * difference / difference.square().sum()
    final = decoder(observation, evidence, values, prior)
    assert (final.boxes[:, 2, :2] - final.boxes[:, 3, :2]).norm() <= .75
    assert final.valid[:, 2:].tolist() == [[True, False]]
    assert final.boxes.shape == (1, 4, 4) and evidence.mode_valid.all()


@pytest.mark.parametrize('unique_count', [1, 2])
def test_sparse_one_two_unique_points_and_duplicate_raw_ids_have_finite_loss_gradient(unique_count):
    values = batch(1)
    values['point_valid'].zero_()
    values['point_valid'][:, -1, :4] = True
    values['point_ids'][:, -1, :4] = torch.arange(4).remainder(unique_count)
    values['extension_valid'].zero_()
    values['extension_valid'][:, :4] = True
    values['extension_ids'][:, :4] = 1000 + torch.arange(4).remainder(unique_count)
    values['extension_labels'].fill_(-1)
    values['extension_labels'][:, :4] = 1
    model = JointTracker(config()).train()
    output = model(values)
    losses = model.compute_losses(values, output)
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert output.evidence.point_valid.sum() == unique_count
    assert (output.verification.support_mask.sum(-1) <= 2 * unique_count).all()
    losses['loss_total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
