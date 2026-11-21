"""横向点投票、采用比例及真实候选/梯度边界。"""
from types import SimpleNamespace

import torch

from models.ct_v41.geometry import rotate_xy
from models.ct_v41.model import JointTracker
from models.ct_v41.q0_lateral import Q0LateralHead, lateral_losses
from models.ct_v41.verification import build_current_pool, final_candidate_support
from tests.helpers.synthetic_batch import batch


def point_fixture():
    data = batch(b=3, n=5)
    data['points'][:, -1, :, :3] = 0.
    data['points'][:, -1, :, 1] = torch.tensor([-2., .5, .8, 1.5, .8])
    data['point_ids'][:, -1] = torch.tensor([0, 1, 2, 3, 2])
    data['point_valid'][1, -1, 2:] = False
    data['point_valid'][2] = False
    base = torch.zeros(3, 4, requires_grad=True)
    dec = SimpleNamespace(q0_box=base, q0_context=torch.zeros(3, 64))
    prob = torch.ones(3, 4, 5)
    prob[:, -1, :] = torch.tensor([.5, .5, 1., .9, 1.])
    obs = SimpleNamespace(sequence_valid=torch.tensor([True, True, False]),
                          foreground_probability=prob, point_features=torch.zeros(3, 4, 5, 64),
                          bc_prediction=torch.zeros(3, 4, 5, 9))
    return data, dec, obs


def test_lateral_vote_rotates_caps_and_requires_unique_foreground():
    data, dec, obs = point_fixture()
    head = Q0LateralHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    def read(apply=True):
        return head(dec, obs, build_current_pool(obs, data), data['box_size'], apply=apply)

    warm = read(False)
    assert warm.box is dec.q0_box and warm.applied.eq(0).all()
    est = read()
    assert est.support_count.tolist() == [4, 2, 0]
    torch.testing.assert_close(est.suggestion, torch.tensor([.8, 0., 0.]))
    torch.testing.assert_close(est.applied, torch.tensor([.4, 0., 0.]))
    torch.testing.assert_close(est.box[0], torch.tensor([0., .4, 0., 0.]))
    torch.testing.assert_close(est.box[1:], dec.q0_box[1:], rtol=0, atol=0)

    obs.foreground_probability[:, -1] = .5
    torch.testing.assert_close(read().suggestion, torch.tensor([.5, 0., 0.]))
    obs.foreground_probability[:, -1] = .49
    assert read().support_count.eq(0).all()
    torch.testing.assert_close(read().box, dec.q0_box, rtol=0, atol=0)
    obs.foreground_probability[:, -1] = torch.tensor([.5, .5, 1., .9, 1.])

    angle = torch.full((3,), 1.19)
    data['points'][:, -1, :, :2] = rotate_xy(data['points'][:, -1, :, :2], angle, to_world=True)
    dec.q0_box = torch.cat((dec.q0_box[:, :3], angle[:, None]), -1)
    rotated = read()
    torch.testing.assert_close(rotated.box[:, :2], rotate_xy(est.box[:, :2], angle, to_world=True))
    torch.testing.assert_close(rotated.box[:, 2:], dec.q0_box[:, 2:])
    with torch.no_grad():
        head.point_head[-1].bias.fill_(10.)
    assert read().suggestion[0] == data['box_size'][0, 1]


def test_lateral_gate_target_and_loss_do_not_train_vote_or_pose():
    data, dec, obs = point_fixture()
    head = Q0LateralHead()
    # 真实点投票可以训练门；全假前景时用显式拒绝监督，即使建议本身恰好为零。
    for foreground in (True, False):
        data['segmentation_labels'].fill_(int(foreground))
        data['target_box'][:, 1] = .85
        est = head(dec, obs, build_current_pool(obs, data), data['box_size'], apply=False)
        losses = lateral_losses(est, data)
        gate_loss = losses['loss_q0_lateral_gate']
        point_grad, pose_grad, gate_grad = torch.autograd.grad(
            gate_loss, (head.point_head[-1].bias, dec.q0_box, head.gate_head[-1].bias),
            allow_unused=True, retain_graph=True)
        assert point_grad is None and pose_grad is None
        if foreground:
            point_grad, = torch.autograd.grad(losses['loss_q0_lateral_point'], head.point_head[-1].bias)
            assert point_grad.abs().sum() > 0
        else:
            assert gate_grad.item() > 0 and losses['loss_q0_lateral_point'].item() == 0


def test_full_lateral_uniform_warmup_and_final_b3_support():
    config = dict(ct_engineering_check=True, v41_attention_backend='math', point_sample_size=16,
                  dataset='kitti_mf', batch_size=2, workers=0)
    data = batch()
    data['points'][:, -1, 0, :3] = torch.tensor([0., 1.05, 0.])
    data['recovery_seed_box'] = data['fallback_box'].clone()
    data['recovery_seed_valid'] = torch.ones(2, dtype=torch.bool)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(774)
        uniform = JointTracker(config).eval()
        end_rng = torch.random.get_rng_state()
        torch.manual_seed(774)
        updated = JointTracker({**config, 'v41_q0_lateral_replay': True}).eval()
        assert torch.equal(torch.random.get_rng_state(), end_rng)
    with torch.no_grad():
        ref = uniform(data)
        updated.q0_epoch = 10
        warm = updated(data)
        torch.testing.assert_close(warm.candidates.boxes, ref.candidates.boxes, rtol=0, atol=0)
        assert torch.equal(warm.commit_decision.support_mask, ref.commit_decision.support_mask)
        # 强制点为预测前景以激活新头，检验 correction 真正进入 B3 最终候选。
        updated.observation.seg_pointnet.fc.bias[1].add_(100.)
        updated.q0_lateral.point_head[-1].bias.add_(5.)
        updated.q0_epoch = 11
        active = updated(data)
        lat = active.decoder.lateral
        assert lat.applied.abs().min() > 0
        torch.testing.assert_close(active.candidates.boxes[:, 0], lat.box, rtol=0, atol=0)
        support = final_candidate_support(active.observation, data, active.candidates.boxes)['mask']
        assert torch.equal(active.verification.support_mask, support & active.candidates.valid[..., None])
        raw_support = final_candidate_support(active.observation, data, lat.base_box[:, None])['mask']
        assert support[:, 0, 0].all() and not raw_support[:, 0, 0].any()
        loss = updated.compute_losses(data, active)
        expected = 2 * torch.nn.functional.smooth_l1_loss(lat.base_box[:, :3], data['target_box'][:, :3])
        torch.testing.assert_close(loss['loss_main_center'], expected)
    aux = updated.forward_aux(data)
    assert not aux.decoder.lateral.box.requires_grad
    assert not aux.decoder.lateral.point_displacement.requires_grad

    updated.train()
    updated.q0_epoch = 1
    data.update(future_valid=torch.ones(2, 2, dtype=torch.bool),
                future_dt=torch.tensor([[1., 2.], [1., 2.]]), future_displacement=torch.ones(2, 2, 2))
    out = updated(data)
    assert out.decoder.lateral.applied.eq(0).all()
    train_loss = updated.compute_losses(data, out)['loss_total']
    train_loss.backward()
    for parameter in (updated.q0_lateral.point_head[-1].bias, updated.q0_lateral.gate_head[-1].bias):
        assert parameter.grad.abs().sum() > 0

    updated.q0_epoch = 11
    active = updated(data)
    parameters = [updated.verifier.output.bias, updated.q0_lateral.point_head[-1].bias,
                  updated.q0_lateral.gate_head[-1].bias, updated.decoder.pose_head.bias]
    verifier_grad, *upstream = torch.autograd.grad(active.verification.logits.sum(), parameters,
                                                   allow_unused=True)
    assert verifier_grad.abs().sum() > 0 and all(grad is None for grad in upstream)
