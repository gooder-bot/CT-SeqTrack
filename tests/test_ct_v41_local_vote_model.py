"""四槽局部投票的几何、选择、拟合、梯度及检查点合同。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v41.joint_observation import JointSearchObservation
from models.ct_v41.local_vote import LocalGainGate, LocalVoteHead, GATE_FEATURES, local_vote_losses
from models.ct_v41.model import JointTracker, select_candidates
from tests.helpers.synthetic_batch import batch as synthetic_batch


@pytest.fixture(autouse=True)
def restore_threads():
    count = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(count)


def test_local_gate_matches_archive_ridge_and_retains_fit_on_empty_epoch():
    rng = np.random.default_rng(96)
    x = rng.normal(size=(50, 18))
    x[:, 0] = 1
    y = rng.uniform(-.4, .25, size=len(x))
    valid = np.arange(len(x)) % 3 != 0
    gate = LocalGainGate()
    for idx in np.array_split(np.arange(len(x)), 4):
        result = SimpleNamespace(features=torch.tensor(x[idx]), valid=torch.tensor(valid[idx]),
                                 rival_index=torch.zeros(len(idx), dtype=torch.long))
        utility = torch.zeros(len(idx), 4, dtype=torch.float64)
        utility[:, 3] = torch.tensor(y[idx])
        gate.observe(result, dict(utility=utility))
    assert gate.fit() == dict(samples=int(valid.sum()), fitted=True)
    x_ok, y_ok = x[valid], y[valid]
    mu, sd = x_ok.mean(0), np.maximum(x_ok.std(0), .001)
    z = (x_ok - mu) / sd
    weight = np.linalg.solve(z.T @ z + 20. * np.eye(18), z.T @ (y_ok - y_ok.mean()))
    ref = y_ok.mean() + ((x - mu) / sd) @ weight
    np.testing.assert_allclose(gate(torch.tensor(x)).numpy(), ref, rtol=1e-11, atol=1e-12)
    cloned = LocalGainGate()
    cloned.load_state_dict(gate.state_dict(), strict=True)
    edge = cloned(torch.tensor(x))
    assert cloned.fit() == dict(samples=0, fitted=True)
    torch.testing.assert_close(cloned(torch.tensor(x)), edge)
    archive = dict(features=GATE_FEATURES, penalty=20., threshold=0., w=weight,
                   mu=mu, sd=sd, mean_y=y_ok.mean())
    fresh = LocalGainGate()
    fresh.load_archive(archive)
    torch.testing.assert_close(edge, fresh(torch.tensor(x)))


def test_transplanted_head_matches_original_geometry_and_retains_actual_refinement_region():
    b, n = 2, 17
    torch.manual_seed(128)
    donor = JointSearchObservation()
    with torch.no_grad():
        donor.offset.weight.normal_(std=.1)
        donor.refine[-1].bias.fill_(.6)
    head = LocalVoteHead()
    head.load_donor({'state_dict': {'tracker.joint_search.'+k: v for k, v in donor.state_dict().items()}})
    xyz = torch.randn(b, n, 3)
    feature = torch.randn(b, n, 64, requires_grad=True)
    valid = torch.ones(b, n, dtype=torch.bool)
    valid[1] = False
    seed = torch.randn(b, 4, requires_grad=True)
    size = torch.tensor([[4., 2., 1.5]]).expand(b, -1)
    batch = dict(local_template_valid=torch.ones(b, 128, dtype=torch.bool),
                 local_template_xyz=torch.randn(b, 128, 3), box_size=size, anchor_box=torch.randn(b, 4))
    pool = dict(xyz=xyz, valid=valid, features=feature, ids=torch.arange(n).expand(b, -1))
    actual = head(pool, batch, seed)
    template, ready = donor._templates(
        [dict(template_xyz=x.numpy()) for x in batch['local_template_xyz']], size)
    expected = donor._measure(xyz, pool['ids'], valid, size, seed.detach(), batch['anchor_box'],
                              template, ready, feature)
    for name in ('box', 'valid', 'vote', 'foreground_logit'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0., atol=0.)
    assert not actual.valid[1] and not actual.refine_input_mask[1].any()
    actual.box.sum().backward()
    assert feature.grad is None and seed.grad is None


def test_paired_shapes_share_one_eval_call_and_preserve_four_slot_decision():
    torch.manual_seed(41)
    writer = JointTracker(dict(v41_local_vote=True, v41_history_training='window_1448',
                               v41_attention_backend='math', ct_engineering_check=True,
                               workers=0, point_sample_size=16, batch_size=3)).eval()
    head = writer.local_vote_head
    assert head.paired_eval_shapes is False
    weights = tuple(head.state_dict())
    batch = synthetic_batch(3, 16)
    batch.update(local_template_xyz=torch.randn(3, 128, 3),
                 local_template_valid=torch.ones(3, 128, dtype=torch.bool))
    batch['local_template_valid'][1] = False
    batch['point_valid'][2, -1] = False
    batch['point_ids'][0, -1, 1] = batch['point_ids'][0, -1, 0]
    sizes = []
    hook = head.shape.compare.register_forward_pre_hook(lambda mod, args: sizes.append(len(args[0])))
    try:
        with torch.no_grad():
            writer.local_gate.w.normal_(std=.03)
            original = writer(batch)
            head.paired_eval_shapes = True
            paired = writer(batch)
        assert sizes == [3, 3, 6]
        assert tuple(head.state_dict()) == weights
        for field in ('rival_index', 'shape_members'):
            assert torch.equal(getattr(original.local_vote, field), getattr(paired.local_vote, field))
        for field in ('shape_logits', 'features', 'relative_gain'):
            torch.testing.assert_close(getattr(original.local_vote, field), getattr(paired.local_vote, field),
                                       atol=1e-5, rtol=1e-5)
        assert torch.equal(original.selected_index, paired.selected_index)
        assert torch.equal(original.accepted_box, paired.accepted_box)
        assert torch.equal(original.commit_decision.support_mask, paired.commit_decision.support_mask)
        # 可求导的 eval 探针和训练/辅助流继续沿用原来的两个形状图。
        writer(batch).local_vote.shape_logits.sum().backward()
        assert sizes[-2:] == [3, 3] and head.shape.compare[-1].weight.grad is not None
        writer.train()
        with torch.no_grad():
            writer(batch)
        assert sizes[-2:] == [3, 3]
    finally:
        hook.remove()


def test_refinement_loss_uses_pre_refinement_mask_with_one_observed_target_point():
    result = SimpleNamespace(point_valid=torch.tensor([[True, True]]), foreground_logit=torch.zeros(1, 2),
        vote=torch.zeros(1, 2, 3), shape_logits=torch.zeros(1, 2),
        shape_members=torch.tensor([[[False, True], [False, True]]]),
        box=torch.zeros(1, 4), refine_input_mask=torch.tensor([[True, False]]))
    batch = dict(segmentation_labels=torch.tensor([[[1, 0]]]),
                 target_box=torch.tensor([[1., 0., 0., 0.]]), box_size=torch.ones(1, 3))
    assert local_vote_losses(result, batch)['local_refine_supervised_count'].item() == 1
    result.refine_input_mask = ~result.refine_input_mask
    assert local_vote_losses(result, batch)['local_refine_supervised_count'].item() == 0


def test_local_winner_keeps_position_when_its_own_write_fails_and_q0_needs_no_identity_vote():
    batch = synthetic_batch(4, 3)
    batch['points'][..., :3] = 0.
    batch['point_ids'][1, -1] = torch.tensor([0, 0, 1])
    observation = SimpleNamespace(foreground_probability=torch.ones(4, 4, 3))
    boxes = torch.zeros(4, 4, 4)
    boxes[:, 3, 0] = .5
    valid = torch.ones(4, 4, dtype=torch.bool)
    candidates = SimpleNamespace(boxes=boxes, valid=valid, raw_valid=valid,
        q0_quality_logits=torch.zeros(4),
        measurement_candidate=torch.tensor([[True, False, False, True]]).expand(4, -1))
    logits = torch.zeros(4, 4, 3)
    logits[2, 3, 2] = -1.  # q0 达 T 门，局部票自己的 T 未达门。
    verification = SimpleNamespace(logits=logits, utility=logits[..., :2].sigmoid().mean(-1))
    local = SimpleNamespace(rival_index=torch.zeros(4, dtype=torch.long),
                            relative_gain=torch.tensor([.1, .1, .1, 0.]))

    decision, commit, _, _ = select_candidates(candidates, verification, observation, batch, local_vote=local)
    assert decision.selected_index.tolist() == [3, 3, 3, 0]
    assert decision.accepted_box[:, 0].tolist() == [.5, .5, .5, 0.]
    assert commit.support_mask.sum(-1).tolist() == [3, 2, 3, 3]
    assert commit.eligible.tolist() == [True, False, False, True]


def test_four_slot_forward_auxiliary_gradients_and_decision_precedence():
    config = dict(v41_local_vote=True, v41_history_training='window_1448',
                  v41_attention_backend='math', ct_engineering_check=True,
                  workers=0, point_sample_size=16, batch_size=2)
    m = JointTracker(config)
    raw = synthetic_batch(2, 16)
    raw.update(local_template_xyz=torch.zeros(2, 128, 3),
               local_template_valid=torch.ones(2, 128, dtype=torch.bool),
               recovery_seed_box=raw['fallback_box'].clone(), recovery_seed_valid=torch.ones(2).bool())
    assert m.recovery_head is m.joint_search is m.q0_lateral is None
    m.train()
    out = m.forward_aux(raw)
    assert out.candidates.boxes.shape == (2, 4, 4) and out.recovery is None and out.joint is None
    assert torch.equal(out.selected_index, out.local_vote.rival_index)
    assert out.local_vote.features.shape == (2, len(GATE_FEATURES))
    losses = m.compute_aux_losses(raw, out)
    assert 'loss_recovery_xyz' not in losses
    losses['loss_total'].backward()
    assert all(p.grad is None for module in (m.decoder, m.observation, m.prior)
               for p in module.parameters())
    assert all(p.grad is not None for p in m.local_vote_head.parameters())
    assert all(p.grad is not None for p in m.verifier.parameters())
    assert m.fit_local_gate()['samples'] == 2
    m.reset_local_gate_statistics()
    m.eval()
    with torch.no_grad():
        m.local_gate.mean_y.fill_(.2)
        m.local_gate.w.zero_()
        raw['point_valid'][1, -1] = False
        out = m(raw)
        assert out.selected_index[0].item() == 3 and out.selected_index[1].item() != 3
        assert out.commit_decision.source[0] == 3
        expected = out.verification.support_mask[0, 3]
        assert torch.equal(out.commit_decision.support_mask[0], expected)
        assert out.commit_decision.eligible[0] == ((expected.sum() >= 3)
                                                   & (out.commit_decision.trusted_score[0] >= .5))
        m.compute_losses(raw, out)
        assert m.local_gate.moments[0, 0] == 0

        from models.ct_v41.candidate_records import candidate_records
        # 核对离线 oracle 的平局次序，先验和局部票同时完美时优先先验。
        out.candidates.boxes[0, [0, 2], 0] = 20.
        out.candidates.boxes[0, [1, 3]] = raw['target_box'][0]
        out.candidates.valid[0] = True
        assert candidate_records(raw, out)[0]['oracle']['index'] == 1
