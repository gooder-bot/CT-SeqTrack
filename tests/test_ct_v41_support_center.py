"""按预测支持修正 q0 的公开四框、监督隔离和系数轮换。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v41.base_losses import _base_q0
from models.ct_v41.model import JointTracker
from models.ct_v41.support_center import FEATURE_COUNT, SHIFT_CAP, SupportCenter, center_features
from tests.helpers.synthetic_batch import batch as synthetic_batch


def center_inputs():
    batch = synthetic_batch(4, 12)
    batch['points'][..., :3] = 0.
    batch['consecutive_untrusted_outputs'] = torch.tensor([0, 1, 2, 9])
    q = torch.zeros(4, 4)
    q[:, 3] = torch.pi / 2
    p = torch.full((4, 4, 12), .85)
    p[1:, -1, 9:] = .5
    batch['point_valid'][3, -1] = False
    obs = SimpleNamespace(foreground_probability=p, coarse_box=q,
                          current_valid=batch['point_valid'][:, -1].any(-1))
    prior = SimpleNamespace(box=q, valid=torch.ones(4, dtype=torch.bool), log_sigma=torch.zeros(4, 2))
    return batch, prior, obs, q


def test_feature_schema_uses_raw_box_axes_and_online_data_only():
    batch, prior, obs, q = center_inputs()
    batch['points'][0, -1, 0, :3] = torch.tensor([0., 1., .75])
    # L/W/H=4/2/1.5，yaw=90°，局部向量为 (.25, 0, .5)。
    x, count, high = center_features(batch, prior, obs, q)
    assert x.shape == (4, FEATURE_COUNT)
    torch.testing.assert_close(x[0, 0:3], torch.tensor([.25, 0., .5]) / 12)
    torch.testing.assert_close(x[0, 6:9], x[0, :3])
    torch.testing.assert_close(x[0, 12:15], torch.tensor([.25, 0., .5]))
    torch.testing.assert_close(x[0, 31:34], batch['box_size'][0] / 4.)
    assert count.tolist() == [12, 12, 12, 0] and high.tolist() == [12, 9, 9, 0]
    assert x[3, -1] == 1. and torch.isfinite(x).all()
    del batch['segmentation_labels'], batch['target_box']
    assert torch.equal(x, center_features(batch, prior, obs, q)[0])


def test_dense_sparse_exact_thresholds_caps_yaw_and_no_blend_without_fit(tmp_path):
    batch, prior, obs, raw = center_inputs()
    local = SimpleNamespace(box=raw.clone(), valid=torch.tensor([True, True, True, False]))
    local.box[:, 0] += torch.tensor([.5, 1., 1.001, .5])
    center = SupportCenter()
    cold = center.before_vote(batch, prior, obs, raw)
    center.after_vote(cold, obs, local)
    assert torch.equal(cold.box, raw) and not cold.branch.any()
    archive = tmp_path / 'calibration.npz'
    np.savez(archive, mu=np.zeros(42), scale=np.ones(42), coefs=np.zeros((42, 3)),
             bias=[2., -2., 1.], cap=SHIFT_CAP)
    with pytest.raises(ValueError, match='eval mode'):
        center.load_archive(archive)
    center.eval().load_archive(archive)
    estimate = center.before_vote(batch, prior, obs, raw)
    center.after_vote(estimate, obs, local)
    assert estimate.branch.tolist() == [1, 2, 0, 0]
    torch.testing.assert_close(estimate.box[0, :3], torch.tensor([.38, .65, .3]) * .5 * 12 / 18)
    torch.testing.assert_close(estimate.box[1, :3], torch.tensor([.3, 0., 0.]))
    assert torch.equal(estimate.box[2:], raw[2:]) and torch.equal(raw[:, 3], estimate.box[:, 3])


def test_tenth_point_exactly_at_high_confidence_and_expanded_roi_boundary_switches_route():
    batch, prior, obs, raw = center_inputs()
    raw[:, 3] = 0.
    probabilities = obs.foreground_probability[:, -1]
    probabilities.fill_(.5)
    probabilities[:, :10] = .85
    probabilities[0, 9] = torch.nextafter(torch.tensor(.85), torch.tensor(0.))
    edge = batch['box_size'][1, 0] * .8
    batch['points'][1, -1, 9, 0] = edge
    batch['points'][2, -1, 9, 0] = torch.nextafter(edge, torch.tensor(float('inf')))
    center = SupportCenter()
    center.active.fitted.fill_(True)
    estimate = center.before_vote(batch, prior, obs, raw)
    local = SimpleNamespace(box=raw.clone(), valid=torch.ones(4, dtype=torch.bool))
    local.box[:, 0] += .5
    center.after_vote(estimate, obs, local)
    assert estimate.high_count.tolist() == [9, 10, 9, 0]
    assert estimate.count.tolist() == [12, 12, 11, 0]
    assert estimate.branch.tolist() == [2, 1, 2, 0]


def test_auxiliary_calibration_keeps_active_and_pending_distinct_through_resume():
    batch, prior, obs, raw = center_inputs()
    center = SupportCenter()
    batch['target_box'] = raw + torch.tensor([0., .2, .1, 0.])
    batch['segmentation_labels'][1, -1].zero_()
    batch['segmentation_labels'][2, -1].zero_()
    batch['target_box'][2, 1] += .25
    estimate = center.before_vote(batch, prior, obs, raw)
    estimate.features = torch.zeros_like(estimate.features)
    center.observe(estimate, batch)
    audit = center.fit_pending()
    assert audit == dict(samples=3, fitted=True, active_fitted=False)
    assert torch.equal(center.before_vote(batch, prior, obs, raw).box, raw)
    # 第一轴在 Huber 区域中，第三个样本是弱权重离群点。
    assert .21 < center.pending.bias[0] < .24
    torch.testing.assert_close(center.pending.bias[2], torch.tensor(.1))
    restored = SupportCenter()
    restored.load_state_dict(center.state_dict(), strict=True)
    assert not restored.active.fitted and restored.pending.fitted
    for module in (center, restored):
        module.start_epoch()
        assert module.active.fitted
        assert module.fit_pending() == dict(samples=0, fitted=True, active_fitted=True)
    assert torch.equal(center.before_vote(batch, prior, obs, raw).box,
                       restored.before_vote(batch, prior, obs, raw).box)


def test_tracker_computes_vote_once_supervises_raw_q0_and_collects_only_natural_rows():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            tracker = JointTracker(dict(v41_local_vote=True, v41_support_center=True,
                v41_history_training='window_1448', v41_attention_backend='math', ct_engineering_check=True,
                workers=0, point_sample_size=16, batch_size=2))
            data = synthetic_batch(2, 16)
            data.update(local_template_xyz=torch.zeros(2, 128, 3),
                        local_template_valid=torch.ones(2, 128, dtype=torch.bool),
                        future_valid=torch.ones(2, 2, dtype=torch.bool),
                        future_dt=torch.tensor([[.2, .4]]).expand(2, -1),
                        future_displacement=torch.zeros(2, 2, 2),
                        consecutive_untrusted_outputs=torch.zeros(2, dtype=torch.long))
            seeds = []
            handle = tracker.local_vote_head.register_forward_pre_hook(lambda _m, args: seeds.append(args[2].detach()))
            try:
                # 首轮两臂共用参数和 dropout 抽样。校准收集不得暗改原模型的梯度。
                with torch.random.fork_rng(devices=[]):
                    original = JointTracker(dict(tracker.config, v41_support_center=False))
                native_state = {name: value for name, value in tracker.state_dict().items()
                                if not name.startswith('support_center.')}
                original.load_state_dict(native_state, strict=True)
                with torch.random.fork_rng(devices=[]):
                    native_out = original(data)
                    native_loss = original.compute_losses(data, native_out)['loss_total']
                    native_loss.backward()
                output = tracker(data)
                assert len(seeds) == 1 and tracker.local_vote_head.paired_eval_shapes
                estimate = output.decoder.support_center
                assert _base_q0(output.decoder) is estimate.raw_box and estimate.raw_box.requires_grad
                assert not estimate.box.requires_grad and torch.equal(estimate.box, seeds[0])
                losses = tracker.compute_losses(data, output)
                assert torch.autograd.grad(losses['loss_main'], estimate.raw_box, retain_graph=True)[0].abs().sum() > 0
                assert torch.equal(native_out.candidates.boxes, output.candidates.boxes)
                assert torch.equal(native_loss, losses['loss_total'])
                losses['loss_total'].backward()
                for name, param in tracker.named_parameters():
                    gradient = original.get_parameter(name).grad
                    assert (param.grad is None) == (gradient is None), name
                    if gradient is not None:
                        assert torch.equal(param.grad, gradient), name
                tracker.zero_grad(set_to_none=True)
                assert len(tracker.support_center._rows) == 0
                aux = tracker.forward_aux(data)
                losses = tracker.compute_aux_losses(data, aux)
                assert len(seeds) == 2 and len(tracker.support_center._rows) == 1
                losses['loss_total'].backward()
                assert tracker.local_vote_head.offset.weight.grad is not None
                assert all(p.grad is None for p in tracker.decoder.parameters())
                tracker.eval()
                with torch.no_grad():
                    first = tracker(data)
                    poisoned = dict(data, target_box=data['target_box'] * float('nan'),
                                    segmentation_labels=1 - data['segmentation_labels'])
                    second = tracker(poisoned)
                    assert torch.equal(first.candidates.boxes, second.candidates.boxes)
                    assert torch.equal(first.commit_decision.support_mask, second.commit_decision.support_mask)
                    assert torch.equal(first.selected_index, second.selected_index)
                    assert len(tracker.support_center._rows) == 1
            finally:
                handle.remove()
    finally:
        torch.set_num_threads(threads)
