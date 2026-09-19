"""完整 v31 图的目标、梯度和输入反事实，不依赖 Lightning/数据 SDK。"""

import copy
from dataclasses import replace

import pytest
import torch

from models.ct_v31.config import normalize_config, config_identity
from models.ct_v31.contracts import DecoderOutput
from models.ct_v31.losses import box_loss, oriented_iou_labels, quality_targets
from models.ct_v31.model import JointTracker, select_hypothesis
from models.ct_v31.observation import box_corners_xyz
from models.ct_v31.prior import PhysicalTimePrior


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_batch(batch_size=2, point_count=16):
    generator = torch.Generator().manual_seed(713)
    b, n = batch_size, point_count
    points = torch.randn((b, 4, n, 5), generator=generator) * .3
    points[..., 3] = torch.tensor([-1.5, -1., -.5, 0.])[None, :, None]
    points[..., 4] = .5
    boxes = torch.tensor([[[-.6, 0., 0., 0.], [-.3, 0., 0., 0.], [0., 0., 0., 0.]]]).repeat(b, 1, 1)
    target = torch.tensor([[.5, 0., 0., .1]]).repeat(b, 1)
    size = torch.tensor([[4., 2., 1.5]]).repeat(b, 1)
    valid = torch.ones((b, 4, n), dtype=torch.bool)
    extension = torch.randn((b, 768, 5), generator=generator) * .3
    extension[..., :3] += torch.tensor([3., 0., 0.])
    ext_valid = torch.arange(768)[None].repeat(b, 1) < 24
    memory = torch.randn((b, 36, 5), generator=generator) * .1
    metadata = torch.zeros((b, 36, 8))
    metadata[..., 0] = 1.
    batch = dict(points=points, point_valid=valid,
        point_ids=torch.arange(n)[None, None].expand(b, 4, -1).clone(),
        history_boxes=boxes, history_valid=torch.ones(b, 3, dtype=torch.bool),
        history_pair_valid=torch.ones(b, 2, dtype=torch.bool),
        history_times=torch.tensor([[-1.5, -1., -.5]]).repeat(b, 1),
        frame_times=torch.tensor([[-1.5, -1., -.5, 0.]]).repeat(b, 1),
        current_dt=torch.full((b,), .5), box_size=size, target_box_size=size.clone(),
        anchor_box=torch.zeros(b, 4), fallback_box=torch.tensor([[.3, 0., 0., 0.]]).repeat(b, 1),
        target_box=target, history_target_boxes=boxes.clone(),
        segmentation_labels=torch.ones(b, 4, n, dtype=torch.long), bc_targets=torch.zeros(b, 4, n, 9),
        physical_displacement=torch.tensor([[.4, .1]]).repeat(b, 1), physical_valid=torch.ones(b, dtype=torch.bool),
        acquisition_target=torch.full((b, 2), .8), acquisition_valid=torch.ones(b, dtype=torch.bool),
        acquisition_demand=torch.ones(b, dtype=torch.bool),
        extension_points=extension, extension_valid=ext_valid,
        extension_ids=torch.arange(768)[None].expand(b, -1).clone().masked_fill(~ext_valid, -1),
        extension_partition=torch.zeros(b, 768, dtype=torch.long).masked_fill(~ext_valid, -1),
        extension_labels=ext_valid.float(), memory_points=memory, memory_valid=torch.ones(b, 36, dtype=torch.bool),
        memory_ids=torch.arange(36)[None].repeat(b, 1), memory_metadata=metadata)
    return batch


def model_config(arm='full'):
    return {'v31_arm': arm, 'ct_engineering_check': True, 'point_sample_size': 16,
            'workers': 0, 'batch_size': 2, 'epoch': 2}


@pytest.mark.parametrize('arm', ['b0', 'b1', 'b1_b2', 'full'])
def test_complete_forward_backward_and_optimizer(arm):
    torch.manual_seed(11)
    model = JointTracker(model_config(arm)).train()
    batch = make_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    prior_calls = []
    hook = model.prior.register_forward_hook(lambda *args: prior_calls.append(1)) if model.prior else None
    prior = model.plan_prior(batch)
    output = model(batch, prior=prior)
    losses = model.compute_losses(batch, output)
    assert torch.isfinite(losses['loss_total'])
    losses['loss_total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())
    optimizer.step()
    if model.prior:
        assert len(prior_calls) == 1
        hook.remove()
    assert output.accepted_box.shape == (2, 4)
    if arm != 'full':
        assert torch.equal(output.selected_index, torch.zeros(2, dtype=torch.long))


def test_sigma_cannot_update_mean_or_temporal_features():
    prior = PhysicalTimePrior()
    out = prior(make_batch())
    out.log_sigma.sum().backward()
    assert prior.sigma_head.bias.grad is not None
    assert prior.mean_head.weight.grad is None
    assert all(p.grad is None for p in prior.cfc.parameters())


def test_acquisition_gradient_reaches_cfc_after_adapter_initialization():
    prior = PhysicalTimePrior()
    with torch.no_grad():
        prior.acquisition_head[-1].weight.fill_(.01)
    out = prior(make_batch())
    out.acquisition_fraction.sum().backward()
    assert sum(p.grad.abs().sum() for p in prior.cfc.parameters() if p.grad is not None) > 0


def test_pair_mask_excludes_relocalization_without_disabling_acquisition():
    batch = make_batch(1)
    batch['history_boxes'][0, 2, 0] = 50.
    batch['history_pair_valid'][0, 1] = False
    prior = PhysicalTimePrior()
    out = prior(batch)
    torch.testing.assert_close(out.mean_xy[0], torch.tensor([.3, 0.]))
    batch['history_pair_valid'].zero_()
    out = prior(batch)
    assert not out.valid.any()
    assert out.acquisition_fraction.requires_grad
    torch.testing.assert_close(out.box, batch['fallback_box'])


def test_forward_does_not_read_current_gt():
    torch.manual_seed(7)
    model = JointTracker(model_config()).eval()
    batch = make_batch(1)
    alternate = dict(batch)
    for key in ('target_box', 'target_box_size', 'history_target_boxes', 'physical_displacement',
                'segmentation_labels', 'bc_targets', 'extension_labels', 'acquisition_target'):
        alternate[key] = batch[key] * 0 + 99
    with torch.no_grad():
        first, second = model(batch), model(alternate)
    torch.testing.assert_close(first.accepted_box, second.accepted_box, rtol=0, atol=0)
    torch.testing.assert_close(first.quality_logits, second.quality_logits, rtol=0, atol=0)


def test_no_current_measurements_produce_prior_only_without_nan():
    model = JointTracker(model_config()).train()
    batch = make_batch(1)
    batch['point_valid'][:, -1] = False
    batch['extension_valid'].zero_()
    out = model(batch)
    torch.testing.assert_close(out.accepted_box, out.prior.box)
    assert not out.hypothesis_valid[:, 1:].any()
    loss = model.compute_losses(batch, out)
    assert loss['loss_main'] == 0 and loss['loss_coarse'] == 0
    loss['loss_total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_bg_has_no_direct_mode_localization_or_bc_when_absent():
    model = JointTracker(model_config()).eval()
    batch = make_batch(1)
    batch['extension_labels'].zero_()
    batch['segmentation_labels'].zero_()
    out = model(batch)
    losses = model.compute_losses(batch, out)
    assert losses['loss_modes'] == 0
    assert losses['loss_vote'] == 0
    assert losses['loss_bc'] == 0
    assert losses['loss_identity'] > 0 and losses['loss_quality'] > 0


def test_no_fixed_action_cap_and_ties_prefer_main():
    boxes = torch.tensor([[[0., 0., 0., 0.], [8., 1., 2., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
    decoder = DecoderOutput(boxes, torch.tensor([[0., 5., 0., 0.]]),
                            torch.tensor([[True, True, False, False]]), torch.zeros(1, 3, 4), torch.zeros(1, 4, 64))
    accepted, selected, _ = select_hypothesis(decoder)
    torch.testing.assert_close(accepted, boxes[:, 1])
    assert selected.item() == 1
    _, selected, _ = select_hypothesis(replace(decoder, quality_logits=torch.zeros(1, 4)))
    assert selected.item() == 0


def test_yaw_periodic_loss_and_geometry_quality_are_not_sin_aliases():
    target = torch.tensor([[0., 0., 0., .3]])
    alias = torch.tensor([[0., 0., 0., torch.pi - .3]])
    assert box_loss(alias, target, torch.tensor([True])) > 1
    equivalent = target.clone()
    equivalent[:, 3] += 2 * torch.pi
    assert box_loss(equivalent, target, torch.tensor([True])) < 1e-5
    size = torch.tensor([[4., 2., 1.5]])
    boxes = target[:, None].expand(-1, 4, -1).clone()
    boxes[:, 1, 0] = 10
    iou = oriented_iou_labels(boxes, target, size)
    assert iou[0, 0] == pytest.approx(1) and iou[0, 1] == 0
    quality = quality_targets(boxes, target, size, torch.ones(1, 4), iou)
    assert quality[0, 0] == pytest.approx(1)
    assert 0 < quality[0, 1] < quality[0, 0]


def test_config_rejects_legacy_keys_and_resume_identity_changes():
    with pytest.raises(ValueError, match='unknown/inactive'):
        normalize_config({'motion_v3_fusion_scale': .5})
    with pytest.raises(ValueError, match='scratch_only'):
        normalize_config({'init_checkpoint': 'old.ckpt'})
    assert config_identity({'path': 'a'}) == config_identity({'path': 'b'})
    assert config_identity({'v31_arm': 'full'}) != config_identity({'v31_arm': 'b0'})
