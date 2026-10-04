"""v36 整图初始化、损失、选择与一次 Adam 的 CPU 合同。"""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from models.ct_v36.config import normalize_config, config_identity
from models.ct_v36.contracts import CandidateSet, VerificationOutput
from models.ct_v36.losses import endpoint_bce, legacy_mean_loss
from models.ct_v36.model import JointTracker, select_candidates
from models.ct_v31.motion import physical_motion_uncertainty_loss
from tests.test_ct_v31_joint import make_batch as old_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(arm='A'):
    return normalize_config(dict(v36_arm=arm, ct_engineering_check=True,
        point_sample_size=16, workers=0, batch_size=2, epoch=2, v32_reserve_windows=0))


def make_batch(b=2, n=16):
    batch = old_batch(b, n)
    batch['anchor_box'] = batch['anchor_box'].double()
    batch['history_is_initial'] = torch.tensor([[True, False, False]]).expand(b, -1)
    batch.update(event_boxes=batch['history_boxes'].double().clone(),
        event_times=torch.tensor([[0., .5, 1.]], dtype=torch.float64).expand(b, -1),
        event_valid=torch.ones(b, 3, dtype=torch.bool),
        event_edge_valid=torch.tensor([[False, True, True]]).expand(b, -1),
        event_segment=torch.zeros(b, 3, dtype=torch.long),
        event_source=torch.zeros(b, 3, dtype=torch.long),
        event_initial=batch['history_is_initial'], event_geometry=torch.zeros(b, 3, 8),
        event_frame_ids=torch.tensor([[0, 1, 2]]).expand(b, -1),
        current_time=torch.full((b,), 1.5, dtype=torch.float64),
        current_frame=torch.full((b,), 3, dtype=torch.long),
        history_geometry=torch.zeros(b, 3, 8),
        current_target_world=batch['target_box'].double().clone(),
        prior_origin_target_world=batch['history_boxes'][:, -1].double().clone())
    batch['extension_ids'] = torch.where(batch['extension_valid'], batch['extension_ids'] + 1000, -1)
    return batch


@pytest.mark.parametrize('arm', ['A', 'C', 'D', 'E', 'F'])
def test_complete_single_forward_backward_adam(arm):
    model = JointTracker(config(arm)).train()
    batch = make_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=.00015)
    calls = []
    handle = model.prior.register_forward_hook(lambda *args: calls.append(1))
    prior = model.plan_prior(batch)
    output = model(batch, prior)
    losses = model.compute_losses(batch, output)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    losses['loss_total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert len(calls) == 1
    handle.remove()
    assert output.accepted_box.shape == (2, 4)
    assert output.candidates.valid[:, 1].all()
    assert (output.measurement_index == -1).all()
    if arm == 'C':
        assert model.evidence is None and not output.candidates.valid[:, 2:].any()
        assert model.decoder.mode_pose_head is None and not hasattr(model.decoder, 'extension_projection')


def test_common_initial_tensors_do_not_depend_on_arm_or_constructor_rng():
    baseline = JointTracker(config('A')).state_dict()
    for arm in ('C', 'D', 'E', 'F'):
        torch.manual_seed(ord(arm))
        model = JointTracker(config(arm))
        for key, value in model.state_dict().items():
            if not key.startswith('prior.') or key.startswith(('prior.range_anchor_head.', 'prior.range_rate_head.')):
                torch.testing.assert_close(value, baseline[key], rtol=0, atol=0)
        if arm in ('C', 'D', 'F'):
            for key, value in model.prior.state_dict().items():
                torch.testing.assert_close(value, baseline['prior.' + key], rtol=0, atol=0)


def test_measurement_bce_is_endpoint_equal_without_class_balance():
    logits = torch.tensor([[1., 9., 9.], [-2., .5, 1.]], requires_grad=True)
    labels = torch.tensor([[0., 1., 1.], [1., 1., 0.]])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    terms = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    expected = .5 * (terms[0, 0] + terms[1].mean())
    actual = endpoint_bce(logits, labels, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert (logits.grad[0, 1:] == 0).all()
    empty = endpoint_bce(logits, labels, torch.zeros_like(mask))
    assert empty == 0 and empty.requires_grad


def test_measurement_selection_is_independent_and_no_memory_second_choice():
    candidates = CandidateSet(torch.randn(1, 4, 4), torch.ones(1, 4, dtype=torch.bool),
                             torch.zeros(1, 3, 4), torch.zeros(1, 4, 64), torch.zeros(1, 2, 4))
    logits = torch.tensor([[[0., 0., 1.], [9., 9., 9.], [2., 2., 3.], [1., 1., 2.]]])
    support = torch.tensor([[[1,1,1], [1,1,1], [1,0,0], [1,1,1]]], dtype=torch.bool)
    verification = VerificationOutput(logits, support, torch.arange(3)[None],
        torch.zeros(1, 3, 3), torch.ones(1, 3), torch.ones_like(support))
    box, _, decision = select_candidates(candidates, verification)
    assert decision.output_index.item() == 1 and decision.measurement_index.item() == 2
    assert not decision.memory_eligible.item()
    torch.testing.assert_close(box, candidates.boxes[:, 1])


def test_range_loss_cannot_update_mean_context_or_anchor_through_current_term():
    model = JointTracker(config()).eval()
    batch = make_batch()
    output = model(batch)
    losses = model.compute_losses(batch, output)
    losses['loss_range_current'].backward()
    assert all(p.grad is None for p in model.prior.encoder.parameters())
    assert all(p.grad is None for p in model.prior.velocity_head.parameters())
    assert all(p.grad is None for p in model.prior.range_anchor_head.parameters())
    assert any(p.grad is not None for p in model.prior.range_rate_head.parameters())


def test_range_pinball_uses_metres_without_extra_size_weighting():
    model = JointTracker(config()).eval()
    batch = make_batch()
    output = model(batch)
    output.prior.axes = torch.eye(2).expand(2, -1, -1)
    batch['prior_origin_target_world'] = output.prior.origin_world.detach().clone()
    batch['prior_origin_target_world'][:, :2] += torch.tensor([2., 4.])
    batch['current_target_world'] = output.prior.world_box.detach().clone()
    batch['current_target_world'][:, :2] += torch.tensor([1., 2.])
    losses = model.compute_losses(batch, output)
    # anchor=.5, rate=1, T=.5: .9*(1.5+3.5)/2 and .9*(0+1)/2.
    torch.testing.assert_close(losses['loss_range_anchor'], torch.tensor(2.25))
    torch.testing.assert_close(losses['loss_range_current'], torch.tensor(.45))
    torch.testing.assert_close(losses['loss_range'], torch.tensor(1.35))
    batch['box_size'] *= 2
    doubled = model.compute_losses(batch, output)
    torch.testing.assert_close(losses['loss_range'], doubled['loss_range'])


def test_supervision_changes_cannot_change_forward_or_actions():
    model = JointTracker(config()).eval()
    batch = make_batch()
    changed = deepcopy(batch)
    for key in ('target_box', 'history_target_boxes', 'segmentation_labels', 'extension_labels',
                'current_target_world', 'prior_origin_target_world', 'physical_displacement'):
        changed[key] = torch.ones_like(changed[key]) * 7
    with torch.no_grad():
        before, after = model(batch), model(changed)
    torch.testing.assert_close(before.hypothesis_boxes, after.hypothesis_boxes, rtol=0, atol=0)
    torch.testing.assert_close(before.verification.logits, after.verification.logits, rtol=0, atol=0)
    assert torch.equal(before.selected_index, after.selected_index)
    assert torch.equal(before.measurement_index, after.measurement_index)


def test_legacy_mean_loss_matches_old_inside_boundary_and_tail():
    prior = JointTracker(config('E')).plan_prior(make_batch())
    target = torch.tensor([[.3, -.2], [10., -10.]])
    valid = torch.ones(2, dtype=torch.bool)
    terms = physical_motion_uncertainty_loss(prior.mean_xy, target, torch.zeros(2, 2),
        prior.direction_xy, valid, kinematic_xy=prior.kinematic_xy.detach(),
        envelope_parallel_perp=prior.envelope.detach(), residual_unit_parallel_perp=prior.unit_residual)
    torch.testing.assert_close(legacy_mean_loss(prior, target, valid), terms['mean_per_sample'].mean())


def test_config_identity_rejects_unregistered_formal_and_cross_arm():
    assert len({config_identity(config(a)) for a in ('A','C','D','E','F')}) == 5
    for changed in ({'lr': .0001}, {'seed': 52}, {'epoch': 61}, {'init_checkpoint': 'old.ckpt'}):
        with pytest.raises(ValueError):
            normalize_config(changed)
