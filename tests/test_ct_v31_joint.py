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
from models.ct_v31.cfc import FullGatedCfCCell


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


def model_config(arm='full', backend='cfc'):
    return {'v31_arm': arm, 'v31_temporal_backend': backend,
            'ct_engineering_check': True, 'point_sample_size': 16,
            'workers': 0, 'batch_size': 2, 'epoch': 2}


@pytest.mark.parametrize('arm,backend', [('b0', 'cfc'), ('b1', 'cfc'),
                                        ('b1_b2', 'cfc'), ('full', 'cfc'), ('full', 'gru')])
def test_complete_forward_backward_and_optimizer(arm, backend):
    torch.manual_seed(11)
    model = JointTracker(model_config(arm, backend)).train()
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


@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_sigma_cannot_update_mean_or_temporal_features(backend):
    prior = PhysicalTimePrior(temporal_backend=backend)
    out = prior(make_batch())
    out.log_sigma.sum().backward()
    assert prior.sigma_head.bias.grad is not None
    assert prior.mean_head.weight.grad is None
    assert all(p.grad is None for p in prior.temporal_cell.parameters())


@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_acquisition_gradient_reaches_temporal_cell_after_adapter_initialization(backend):
    prior = PhysicalTimePrior(temporal_backend=backend)
    with torch.no_grad():
        prior.acquisition_head[-1].weight.fill_(.01)
    out = prior(make_batch())
    out.acquisition_fraction.sum().backward()
    assert sum(p.grad.abs().sum() for p in prior.temporal_cell.parameters() if p.grad is not None) > 0


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
    assert config_identity({'v31_temporal_backend': 'cfc'}) != config_identity({'v31_temporal_backend': 'gru'})


def test_three_arms_have_real_temporal_backends_and_fair_common_initialization():
    torch.manual_seed(42)
    b0 = JointTracker(model_config('b0'))
    torch.manual_seed(42)
    cfc = JointTracker(model_config('full', 'cfc'))
    cfc_rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    gru = JointTracker(model_config('full', 'gru'))
    gru_rng = torch.random.get_rng_state()
    assert b0.prior is None and b0.evidence is None
    assert isinstance(cfc.prior.temporal_cell, FullGatedCfCCell)
    assert isinstance(gru.prior.temporal_cell, torch.nn.GRUCell)
    assert cfc.enable_b3 and gru.enable_b3
    assert torch.equal(cfc_rng, gru_rng)
    cfc_state, gru_state = cfc.state_dict(), gru.state_dict()
    common = [key for key in cfc_state if not key.startswith('prior.temporal_cell.')]
    assert set(common) == {key for key in gru_state if not key.startswith('prior.temporal_cell.')}
    assert all(torch.equal(cfc_state[key], gru_state[key]) for key in common)
    for name in ('observation', 'decoder'):
        expected = getattr(b0, name).state_dict()
        actual = getattr(cfc, name).state_dict()
        assert all(torch.equal(actual[key], value) for key, value in expected.items())
    # 初始 zero residual heads 可以给相同位移，但时序编码必须真有不同。
    batch = make_batch(1)
    cfc_feature = cfc.prior(batch).feature
    gru_feature = gru.prior(batch).feature
    assert cfc_feature.shape == gru_feature.shape == (1, 128)
    assert not torch.allclose(cfc_feature, gru_feature)


@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_same_time_conditioning_reaches_both_temporal_backends(backend):
    batch = make_batch(1)
    prior = PhysicalTimePrior(temporal_backend=backend)
    before = prior(batch)
    retimed = dict(batch, history_times=batch['history_times'] * 2., current_dt=batch['current_dt'] * 2.)
    after = prior(retimed)
    assert before.valid.all() and after.valid.all()
    assert not torch.allclose(before.feature, after.feature)
    # 等比例拉伸时间不改变恒速轨迹外推的物理距离。
    torch.testing.assert_close(before.kinematic_xy, after.kinematic_xy)


@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_full_selects_learned_mode_without_any_calibration_artifact(backend):
    torch.manual_seed(7)
    model = JointTracker(model_config('full', backend)).eval()
    batch = make_batch(1)
    with torch.no_grad():
        initial = model(batch)
        available = torch.where(initial.hypothesis_valid[0, 1:])[0]
        assert available.numel() > 0
        index = int(available[0]) + 1
        difference = initial.decoder.decoder_features[0, index] - initial.decoder.decoder_features[0, 0]
        assert difference.square().sum() > 0
        # 用实际 quality head 的参数构造 mode>q0，不替换 model/selector。
        model.decoder.quality_head.weight.copy_(difference[None])
        model.decoder.quality_head.bias.zero_()
        selected = model(batch)
    assert selected.selected_index.item() > 0
    torch.testing.assert_close(selected.accepted_box, selected.hypothesis_boxes[:, selected.selected_index.item()])
    assert not torch.allclose(selected.accepted_box, selected.hypothesis_boxes[:, 0])


@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_no_motion_pair_still_allows_context_to_learn_from_localization(backend):
    model = JointTracker(model_config('full', backend)).eval()
    batch = make_batch(1)
    batch['history_pair_valid'].zero_()
    with torch.no_grad():
        torch.nn.init.normal_(model.decoder.prior_adapter[-1].weight, std=.03)
        torch.nn.init.normal_(model.decoder.pose_head.weight, std=.03)
    output = model(batch)
    assert not output.prior.valid.any()
    assert output.prior.context_valid.all()
    loss = box_loss(output.hypothesis_boxes[:, 0], batch['target_box'], torch.tensor([True]))
    loss.backward()
    assert model.prior.context[0].weight.grad is not None
    assert model.prior.context[0].weight.grad.abs().sum() > 0
    # 缺速度对不能伪造可训练的物理均值/方差或历史 transition。
    assert model.prior.mean_head.weight.grad is None
    assert model.prior.sigma_head.weight.grad is None
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.prior.temporal_cell.parameters())


def test_missing_history_is_masked_consistently_for_features_prior_and_losses():
    from models.ct_v31.model import canonicalize_batch
    batch = make_batch(1)
    batch['history_valid'][:, :2] = False
    batch['points'][:, :2] = float('nan')
    prepared = canonicalize_batch(batch)
    assert not prepared['point_valid'][:, :2].any()
    assert torch.count_nonzero(prepared['points'][:, :2]) == 0
    assert prepared['point_valid'][:, 2:].all()


def test_unknown_temporal_backend_is_rejected():
    with pytest.raises(ValueError, match='temporal_backend'):
        PhysicalTimePrior(temporal_backend='lstm')


@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_zero_initialized_prior_heads_train_the_cell_on_second_optimizer_step(backend):
    torch.manual_seed(9)
    prior = PhysicalTimePrior(temporal_backend=backend).train()
    optimizer = torch.optim.Adam(prior.parameters(), lr=1e-3)
    batch = make_batch()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = prior(batch)
        loss = ((output.mean_xy - batch['physical_displacement']).square().mean()
                + (output.acquisition_fraction - batch['acquisition_target']).square().mean())
        loss.backward()
        gradients = [parameter.grad for parameter in prior.temporal_cell.parameters()
                     if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        if step == 1:
            assert sum(value.abs().sum() for value in gradients) > 0
        optimizer.step()
