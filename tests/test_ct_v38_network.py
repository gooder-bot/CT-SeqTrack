"""v38 强 B1 等价、三候选权限、独立评分梯度与可信监督合同。"""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.ct_v31.model import JointTracker as OriginalB1
from models.ct_v31.losses import compute_losses as base_losses
from models.ct_v38.config import normalize_config, as_v35_config, recipe_scope
from models.ct_v38.model import JointTracker, build_candidates, select_candidates
from models.ct_v38.losses import supervision_targets, pairwise_gap_loss
from models.ct_v38.verification import current_q0_support
from tests.test_ct_v37_network import batch


@pytest.fixture(autouse=True)
def cpu_state():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(38)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def config(arm='full'):
    return normalize_config(dict(v38_arm=arm, ct_engineering_check=True,
        point_sample_size=16, batch_size=2, workers=0, epoch=2))


def values(b=2, n=16):
    result = batch(b, n)
    for key in ('previous_is_propagation', 'consecutive_propagations',
                'consecutive_untrusted_outputs', 'trusted_velocity_age'):
        result[key] = torch.zeros(b)
    result['trusted_velocity_valid'] = torch.zeros(b, dtype=torch.bool)
    return result


def test_registered_b1_standard_retains_the_actual_strong_v37_concat_configuration():
    from utils.config import load_yaml_config
    from models.ct_v37.config import normalize_config as normalize_v37, as_v35_config as legacy_v37
    from models.ct_v38.config import load_config
    old = legacy_v37(normalize_v37(load_yaml_config('cfgs/ct_seqtrack/37_b1_dedup_concat_mini.yaml')))
    new = as_v35_config(load_config('cfgs/ct_seqtrack/38_b1_standard_mini.yaml'))
    identity_fields = {'cfg', 'tag', 'experiment_name'}
    assert {key: value for key, value in old.items() if key not in identity_fields} == {
        key: value for key, value in new.items() if key not in identity_fields}


@pytest.mark.parametrize('arm', ['b1', 'full'])
def test_common_b1_two_adam_steps_preserve_parameters_gradients_bn_rng_and_losses(arm):
    cfg = config(arm)
    torch.manual_seed(738)
    old = OriginalB1(as_v35_config(cfg)).train()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(738)
    new = JointTracker(cfg).train()
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    if new.verifier is not None:
        # 零头会掩盖上游梯度错误；此处从非零 B3 出口验证主路隔离。
        torch.nn.init.normal_(new.verifier.output.weight, std=.1)
    optimizers = [torch.optim.Adam(model.parameters(), lr=1.5e-4, betas=(.5, .999),
        eps=1e-6, foreach=False) for model in (old, new)]
    data = values()
    for _ in range(2):
        rng = torch.get_rng_state().clone()
        results = []
        for model, optimizer in zip((old, new), optimizers):
            torch.set_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
            with recipe_scope(cfg):
                output = model(data)
                losses = model.compute_losses(data, output)
            losses['loss_total'].backward()
            gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                         for name, parameter in model.named_parameters()}
            optimizer.step()
            results.append((output, losses, gradients, torch.get_rng_state().clone()))
        assert torch.equal(results[0][0].decoder.hypothesis_boxes, results[1][0].decoder.hypothesis_boxes)
        assert torch.equal(results[0][0].prior.box, results[1][0].prior.box)
        for name, loss in results[0][1].items():
            if name != 'loss_total':
                assert torch.equal(loss, results[1][1][name]), name
        assert 'loss_acquisition' in results[1][1]
        for name, gradient in results[0][2].items():
            other = results[1][2][name]
            assert (gradient is None and other is None) or torch.equal(gradient, other), name
        assert torch.equal(results[0][3], results[1][3])
        for name, value in old.state_dict().items():
            assert torch.equal(value, new.state_dict()[name]), name
        for name, parameter in old.named_parameters():
            other = dict(new.named_parameters())[name]
            for key, value in optimizers[0].state[parameter].items():
                assert torch.equal(value, optimizers[1].state[other][key]), (name, key)


def test_zero_heads_immediate_q0_tie_and_independent_legacy_quality():
    model, data = JointTracker(config()), values()
    output = model(data)
    assert torch.equal(output.verification.logits, torch.zeros_like(output.verification.logits))
    assert output.selected_index.tolist() == [0, 0]
    assert output.verification.metadata.shape == (2, 3, 45)
    assert torch.equal(output.legacy_q0_quality, output.decoder.quality_logits[:, 0].sigmoid())
    assert output.commit_quality[:, 1:].eq(0).all()
    assert output.commit_decision.trusted_score.eq(.5).all()


def test_nonzero_b3_loss_has_no_upstream_gradient_but_trains_reader():
    model, data = JointTracker(config()), values()
    torch.nn.init.normal_(model.verifier.output.weight, std=.1)
    output = model(data)
    loss = output.verification.logits.square().sum()
    variables = (output.observation.point_features, output.observation.coarse_box,
        output.candidates.boxes, output.prior.feature, output.prior.log_sigma,
        model.observation.seg_pointnet.conv1.weight if hasattr(model.observation.seg_pointnet, 'conv1')
        else next(model.observation.parameters()))
    gradients = torch.autograd.grad(loss, variables, retain_graph=True, allow_unused=True)
    assert all(gradient is None for gradient in gradients)
    loss.backward()
    assert model.verifier.local_reader.output.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for module in (model.observation, model.decoder, model.prior)
               for parameter in module.parameters())


def test_candidate_dedup_wrap_tolerances_prior_legality_and_q0_alias():
    q0 = torch.tensor([[0., 0., 0., 0.], [4., 0., 0., 0.], [9., 0., 0., 0.]])
    decoder = SimpleNamespace(hypothesis_boxes=q0[:, None].expand(-1, 4, -1).clone(),
        quality_logits=torch.zeros(3, 4))
    prior = SimpleNamespace(box=torch.tensor([[1e-5, 0., 0., 2*torch.pi],
        [1., 0., 0., 0.], [9., 0., 0., 0.]]), valid=torch.zeros(3, dtype=torch.bool))
    observation = SimpleNamespace(sequence_valid=torch.tensor([True, True, False]))
    result = build_candidates(decoder, observation,
        {'fallback_box': torch.tensor([[3., 0., 0., 0.], [1., 0., 0., 0.], [9., 0., 0., 0.]])}, prior)
    assert result.raw_valid.tolist() == [[True, True, True], [True, True, True], [False, True, True]]
    assert result.valid.tolist() == [[True, False, True], [True, True, False], [False, True, False]]
    assert result.alias_of.tolist() == [[-1, 0, -1], [-1, -1, 1], [1, -1, 1]]
    prior.box[0, 0] = 2e-4
    assert build_candidates(decoder, observation,
        {'fallback_box': torch.ones(3, 4)}, prior).valid[0, 1]


@pytest.mark.parametrize('xyz,yaw,duplicate', [(5e-5, 5e-6, True), (2e-4, 0., False), (0., 2e-5, False)])
def test_registered_duplicate_thresholds_are_separate_for_xyz_and_wrapped_yaw(xyz, yaw, duplicate):
    decoder = SimpleNamespace(hypothesis_boxes=torch.zeros(1, 4, 4), quality_logits=torch.zeros(1, 4))
    prior = SimpleNamespace(box=torch.tensor([[xyz, 0., 0., yaw]]), valid=torch.tensor([False]))
    observation = SimpleNamespace(sequence_valid=torch.tensor([True]))
    candidates = build_candidates(decoder, observation, {'fallback_box': torch.ones(1, 4)}, prior)
    assert bool(candidates.valid[0, 1]) is not duplicate


def test_strict_positive_gain_prop_tie_order_and_no_prop_trusted_write():
    model, data = JointTracker(config()), values(1)
    output = model(data)
    candidates = replace(output.candidates, valid=torch.ones(1, 3, dtype=torch.bool))
    logits = torch.zeros(1, 3, 3)
    logits[:, 1:, :2] = .001
    logits[:, :, 2] = 20.
    verification = replace(output.verification, logits=logits)
    selection, commit, quality, table = select_candidates(candidates, verification, output.observation, data)
    assert selection.selected_index.tolist() == [1]
    assert commit.trusted_score.item() == 0 and not commit.eligible.any() and not commit.support_mask.any()
    assert quality.item() == 0 and table[0, 1:].eq(0).all()
    logits.zero_()
    selection, *_ = select_candidates(candidates, verification, output.observation, data)
    assert selection.selected_index.item() == 0


def test_nonfinite_candidate_or_even_invalid_score_slot_fails_closed():
    model, data = JointTracker(config()), values(1)
    output = model(data)
    invalid = output.candidates.valid.clone()
    invalid[:, 2] = False
    candidates = replace(output.candidates, valid=invalid)
    logits = output.verification.logits.clone()
    logits[:, 2, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='verifier logits'):
        select_candidates(candidates, replace(output.verification, logits=logits), output.observation, data)
    prior = replace(output.prior, box=output.prior.box.clone())
    prior.box[:, 0] = float('inf')
    with pytest.raises(FloatingPointError, match='position candidates'):
        build_candidates(output.decoder, output.observation, data, prior)


@pytest.mark.parametrize('arm', ['b1', 'full'])
def test_all_empty_q0_is_only_alias_and_no_trusted_support(arm):
    data = values(1)
    data['point_valid'].zero_()
    output = JointTracker(config(arm))(data)
    assert output.candidates.raw_valid.tolist() == [[False, True, True]]
    assert output.selected_index.item() == 1
    assert not output.commit_decision.eligible.any()
    assert not output.commit_decision.support_mask.any()
    if arm == 'full':
        assert output.selected_quality.item() == 0


@pytest.mark.parametrize('unique_count', [0, 1, 2, 3])
def test_q0_trusted_labels_include_sparse_negatives_and_unselected_q0(unique_count):
    model, data = JointTracker(config()), values(1)
    output = model(data)
    data['points'][:, -1, :, :3] = 0.
    data['point_valid'][:, -1] = False
    data['point_ids'][:, -1] = -1
    if unique_count:
        data['point_valid'][:, -1, :6] = True
        data['point_ids'][:, -1, :6] = torch.arange(6).remainder(unique_count)
    data['target_box'] = torch.zeros_like(data['target_box'])
    data['segmentation_labels'][:, -1] = 1
    observation = replace(output.observation, foreground_probability=torch.ones_like(output.observation.foreground_probability))
    boxes = output.candidates.boxes.detach().clone()
    boxes[:, 0] = 0.
    output = replace(output, observation=observation, candidates=replace(output.candidates, boxes=boxes),
                     selected_index=torch.ones(1, dtype=torch.long))
    target = supervision_targets(data, output)
    assert target['trusted_valid'].item()
    assert target['support_count'].item() == unique_count
    assert target['trusted'].item() == float(unique_count >= 3)
    data['segmentation_labels'][:, -1] = 0
    assert supervision_targets(data, output)['trusted'].item() == 0


def test_unselected_real_q0_trusted_loss_never_supervises_propagation_heads():
    model, data = JointTracker(config()), values(1)
    data['point_valid'][:, -1] = False
    output = model(data)
    assert output.candidates.raw_valid[0, 0]
    output.selected_index.fill_(1)
    output.verification.logits.retain_grad()
    losses = model.compute_losses(data, output)
    assert losses['loss_trusted_write'] > 0
    losses['loss_trusted_write'].backward()
    gradient = output.verification.logits.grad
    assert gradient[0, 0, 2] != 0
    assert gradient[:, 1:, 2].eq(0).all()
    assert gradient[..., :2].eq(0).all()


@pytest.mark.parametrize('size,shift,expected', [
    (1., float(torch.tensor(1./3., dtype=torch.float32)), False),
    (1., float(torch.nextafter(torch.tensor(1./3.), torch.tensor(0.))), True),
    (3., 1., True)])
def test_exact_trusted_iou_boundary_agrees_with_candidate_records_before_float32_rounding(size, shift, expected):
    from models.ct_v38.candidate_records import candidate_records
    model, data = JointTracker(config()), values(1)
    output = model(data)
    data['points'][:, -1, :, :3] = 0.
    data['point_valid'][:, -1] = False
    data['point_valid'][:, -1, :3] = True
    data['point_ids'][:, -1] = -1
    data['point_ids'][:, -1, :3] = torch.arange(3)
    data['segmentation_labels'][:, -1] = 1
    data['target_box'] = torch.zeros_like(data['target_box'])
    data['box_size'] = torch.full_like(data['box_size'], size)
    data['target_box_size'] = data['box_size'].clone()
    observation = replace(output.observation, foreground_probability=torch.ones_like(output.observation.foreground_probability))
    boxes = torch.zeros_like(output.candidates.boxes)
    boxes[:, 0, 0] = shift
    output = replace(output, observation=observation, candidates=replace(output.candidates, boxes=boxes))
    target = supervision_targets(data, output)
    record = candidate_records(data, output)[0]
    assert target['exact_iou'].dtype == torch.float64
    assert target['support_count'].item() == record['support']['count'] == 3
    assert target['support_purity'].item() == record['support']['target_purity'] == 1.
    assert target['exact_iou'][0, 0].item() == record['geometry'][0]['exact_iou']
    assert bool(target['trusted'].item()) is record['trusted_target'] is expected
    if not expected:
        assert target['exact_iou'][0, 0] < .5
        assert target['exact_iou'][0, 0].float().item() == .5


def test_final_q0_reference_and_support_are_independent_of_unused_coarse_box():
    model, data = JointTracker(config()), values(1)
    output = model(data)
    changed = replace(output.observation, coarse_box=output.observation.coarse_box.detach()+100.)
    second = model.verifier(output.candidates, changed, data, output.prior)
    assert torch.equal(output.verification.metadata, second.metadata)
    support = current_q0_support(changed, data, output.candidates.boxes[:, 0])
    assert torch.equal(support['mask'], second.support_mask[:, 0])


def test_gt_and_dead_extension_changes_do_not_change_prediction():
    model, data = JointTracker(config()).eval(), values(1)
    first = model(data)
    changed = copy.deepcopy(data)
    for key in ('target_box', 'history_target_boxes', 'physical_displacement', 'segmentation_labels',
                'bc_targets', 'extension_labels', 'acquisition_target'):
        if key in changed:
            changed[key].fill_(99)
    changed['extension_points'].fill_(100.)
    changed['memory_points'].fill_(100.)
    second = model(changed)
    assert torch.equal(first.candidates.boxes, second.candidates.boxes)
    assert torch.equal(first.verification.logits, second.verification.logits)
    assert torch.equal(first.verification.support_mask, second.verification.support_mask)


def test_risk_metadata_and_invalid_velocity_age_are_finite_and_explicit():
    model, data = JointTracker(config()), values(1)
    data.update(previous_is_propagation=torch.ones(1), consecutive_propagations=torch.tensor([3.]),
        consecutive_untrusted_outputs=torch.tensor([5.]), trusted_velocity_age=torch.tensor([999.]))
    result = model(data).verification.metadata
    expected = torch.tensor([1., torch.log(torch.tensor(4.)), torch.log(torch.tensor(6.)), 0., 0.])
    torch.testing.assert_close(result[0, :, 40:], expected[None].expand(3, -1))


def test_gap_loss_all_valid_unordered_pairs_endpoint_normalization_and_ties():
    utility = torch.tensor([[.9, .1, .2], [.2, .8, .4]], requires_grad=True)
    target = torch.tensor([[.8, .3, .2], [.4, .4, .4]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    actual = pairwise_gap_loss(utility, target, valid)
    first = torch.tensor([(.8-.5)**2, (.7-.6)**2, (-.1-.1)**2]).mean()*.5
    second = torch.tensor((-.6)**2)*.5
    torch.testing.assert_close(actual, (first+second)/2)
    actual.backward()
    assert utility.grad[1, 2] == 0
    zero = pairwise_gap_loss(utility, target, torch.tensor([[True, False, False], [False, True, False]]))
    assert zero.item() == 0 and zero.requires_grad


def test_losses_record_weighted_endpoint_contributions_and_keep_original_quality():
    model, data = JointTracker(config()), values()
    output = model(data)
    output.loss_statistics = {}
    losses = model.compute_losses(data, output)
    original = base_losses(data, output, enable_b1=True, enable_b2=False, enable_b3=False)
    assert torch.equal(losses['loss_quality'], original['loss_quality'])
    for name, weight in (('loss_success', .125), ('loss_precision', .125),
                         ('loss_ranking', .1), ('loss_trusted_write', .5)):
        torch.testing.assert_close(output.loss_statistics[name]['batch_contribution'].sum(), weight*losses[name])
    expected = original['loss_total'] + sum(weight*losses[name] for name, weight in
        (('loss_success', .125), ('loss_precision', .125), ('loss_ranking', .1), ('loss_trusted_write', .5)))
    torch.testing.assert_close(losses['loss_total'], expected)
