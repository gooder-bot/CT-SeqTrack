"""联合搜索点、空匹配和各候选独立监督的回归。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v41.contracts import CandidateSet, VerificationOutput
from models.ct_v41.joint_candidates import combine_joint_candidates
from models.ct_v41.joint_grid import FINE_POINTS, GRID_FEATURES, fine_crop, grid_labels, make_sparse_scan
from models.ct_v41.joint_observation import JOINT_LOSS_WEIGHTS, JointSearchObservation, joint_search_losses
from models.ct_v41.model import select_candidates


def test_grid_supervises_only_returned_vehicles_and_crops_unique_raw_ids():
    poses = np.asarray([[1., 0., 0., 0.], [8., 0., 0., 0.]], np.float32)
    size = np.asarray([4., 2., 2.], np.float32)
    xs = np.r_[np.linspace(1.1, 1.8, 300), 8.4, 20.]
    points = np.column_stack((xs, np.zeros((len(xs), 2)))).astype(np.float32)
    ids = np.arange(len(points), dtype=np.int64)
    scan = make_sparse_scan(np.vstack((points, points[0])), np.r_[ids, ids[0]])
    assert len(scan.ids) == len(ids) and scan.features.shape == (3, GRID_FEATURES)
    _, raw = fine_crop(scan, poses[0], size)
    assert len(np.unique(raw)) == len(raw) == FINE_POINTS
    target = poses[0].copy()
    target[0] = 80.
    vehicle, own, offset = grid_labels(scan, target, size, [np.r_[poses[1], size]])
    assert not own.any() and vehicle.tolist() == [False, True, False]
    np.testing.assert_allclose(offset[1], poses[1, :3] - scan.centers[1])
    assert make_sparse_scan(np.empty((0, 3)), np.empty(0, dtype=int)) is None


@pytest.mark.parametrize(('left_count', 'right_count'), [(1, 3), (3, 1), (2, 2)])
def test_mixed_cell_instance_label_matches_its_center_for_either_query(left_count, right_count):
    left = np.array([.2, 0., 0., 0.])
    right = np.array([.8, 0., 0., 0.])
    size = np.array([.3, .4, .4])
    xs = np.r_[np.linspace(.18, .22, left_count), np.linspace(.78, .82, right_count)]
    scan = make_sparse_scan(np.column_stack((xs, xs * 0., xs * 0.)), np.arange(len(xs)))
    owner = left if left_count >= right_count else right
    for target, other in ((left, right), (right, left)):
        vehicle, own, offsets = grid_labels(scan, target, size, [np.r_[other, size]])
        assert vehicle.tolist() == [True]
        assert own.tolist() == [target is owner]
        np.testing.assert_allclose(scan.centers[0] + offsets[0], owner[:3], atol=1e-7)


def _example(*, remote, template=True):
    rows = []
    for active in remote:
        points = np.array([[19.8, 0., 0.], [20., 0., 0.], [20.2, 0., 0.],
                           [20.2, 0., 0.], [40., 0., 0.]], np.float32)
        rows.append(dict(template_xyz=points[:3] - [20., 0., 0.] if template else points[:0],
                         scan_xyz=points if active else points[:0],
                         scan_ids=np.array([21, 32, 47, 47, 63]) if active else np.empty(0, int),
                         pending_valid=active, pending_box=np.array([20., 0., 0., 0.]),
                         main_box=np.zeros(4), backup_box=np.zeros(4), backup_valid=False))
    b = len(rows)
    boxes = torch.zeros((b, 4, 4))
    xyz = torch.tensor([[[-.2, 0., 0.], [0., 0., 0.], [.2, 0., 0.]]]).expand(b, -1, -1)
    ids = torch.arange(3, dtype=torch.long).expand(b, -1)
    pool = dict(xyz=xyz, ids=ids, valid=ids >= 0, features=torch.zeros(b, 3, 64, requires_grad=True))
    batch = dict(joint_rows=rows, box_size=torch.tensor([4., 2., 2.]).expand(b, -1),
                 anchor_box=boxes[:, 0])
    return (batch, None, SimpleNamespace(q0_box=boxes[:, 0]), SimpleNamespace(box=boxes[:, 0]), pool, boxes)


def _zero_head():
    model = JointSearchObservation()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _select_with_highest_local_utility(model, args, legacy_valid):
    pool, boxes = args[-2:]
    flags = torch.tensor([legacy_valid], dtype=torch.bool)
    measured = flags.clone()
    measured[:, 1:3] = False
    candidates = CandidateSet(boxes, flags, flags, torch.full_like(flags, -1, dtype=torch.long),
                              ~flags[:, 0], torch.zeros(1), measured)
    support = pool['valid'][:, None].expand(-1, 4, -1)
    verifier = VerificationOutput(torch.zeros(1, 4, 3), support, torch.zeros(1, 4, 46))
    with torch.no_grad():
        result = model(*args, legacy_valid=flags)
        result.local.logits[:, :2] = 9.
        result.remote.logits[:, :2] = 5.
        six, scores, evidence = combine_joint_candidates(candidates, verifier, result, pool)
        decision, commit, _, _ = select_candidates(six, scores, None, {}, joint=evidence)
    return result, six, decision, commit


def test_single_uncontested_cold_local_can_win_position_without_writing_state():
    model = _zero_head().eval()
    args = _example(remote=[False], template=False)
    args[-2]['valid'][:, 1:] = False
    flags = [False, True, False, False]
    for has_member, winner in ((True, 4), (False, 1)):
        if not has_member:
            with torch.no_grad():
                model.offset.bias.fill_(8.)  # 投票落到唯一真实回波的框外。
        result, six, decision, commit = _select_with_highest_local_utility(model, args, flags)
        assert result.local.members.sum().item() == int(has_member) and six.raw_valid[0, 4]
        assert not result.local.match_wins.item() and bool(six.valid[0, 4]) == has_member
        assert decision.selected_index.item() == winner and not commit.eligible.item()
        torch.testing.assert_close(decision.accepted_box, six.boxes[:, winner])


@pytest.mark.parametrize('rival_slot', [0, 3, 5])
def test_identity_winner_at_another_position_cannot_be_displaced_by_weak_local(monkeypatch, rival_slot):
    model = _zero_head().eval()
    args = _example(remote=[rival_slot == 5])
    pool = args[-2]
    boxes = args[-1].clone()
    args = (*args[:-1], boxes)
    flags = [False, True, False, False]
    if rival_slot != 5:
        flags[rival_slot] = True
        boxes[:, rival_slot, 0] = 8.
        pool['xyz'] = torch.cat((pool['xyz'], pool['xyz'] + torch.tensor([8., 0., 0.])), 1)
        pool['ids'] = torch.arange(6)[None]
        pool['valid'] = pool['ids'] >= 0
        pool['features'] = torch.zeros(1, 6, 64)
    shape_at = model._shape_at
    def shape_with_known_rival(xyz, point_valid, box, *rest):
        descriptor, members, _ = shape_at(xyz, point_valid, box, *rest)
        return descriptor, members, (box[:, 0] > 7.).to(box) * 2.
    monkeypatch.setattr(model, '_shape_at', shape_with_known_rival)
    result, six, decision, commit = _select_with_highest_local_utility(model, args, flags)
    assert result.local.members.any() and not result.local.match_wins.item()
    assert not six.valid[0, 4] and six.valid[0, rival_slot]
    assert decision.selected_index.item() == rival_slot and commit.eligible.item()


@pytest.mark.parametrize('reason', ['first_scan', 'few_points', 'poor_geometry', 'coarse_rival'])
@pytest.mark.parametrize('local_score', [0., 1.])
def test_remote_without_takeover_evidence_cannot_veto_a_local_output_or_write(monkeypatch, reason, local_score):
    model = _zero_head().eval()
    args = _example(remote=[True])
    args[0]['joint_rows'][0]['pending_valid'] = reason != 'first_scan'
    shape_at, remote_forward = model._shape_at, model._remote

    def shape_with_strong_remote(xyz, point_valid, box, *rest):
        descriptor, members, _ = shape_at(xyz, point_valid, box, *rest)
        score = torch.where(box[:, 0] > 7., 2., local_score)
        return descriptor, members, score

    def remote_with_missing_evidence(*inputs):
        remote, coarse, rival = remote_forward(*inputs)
        if reason == 'few_points':
            remote.support_mask[:, 2:] = False
        if reason == 'poor_geometry':
            remote.logits[:, 2] = -1.
        if reason == 'coarse_rival':
            rival = torch.full_like(rival, 3.)
        return remote, coarse, rival

    monkeypatch.setattr(model, '_shape_at', shape_with_strong_remote)
    monkeypatch.setattr(model, '_remote', remote_with_missing_evidence)
    result, six, decision, commit = _select_with_highest_local_utility(model, args, [True, True, False, True])
    assert result.remote.match_logit.item() == 2.
    assert bool(result.remote.match_wins) == (reason != 'coarse_rival')
    assert result.legacy_match_wins.tolist() == [[local_score > 0., local_score > 0.]]
    assert six.valid[0, 4] and not six.valid[0, 5] and decision.selected_index.item() == 4
    assert commit.eligible.item() == (local_score > 0.)


def test_mixed_batch_one_external_crop_with_its_own_ids_and_coordinates():
    model = _zero_head().eval()
    batch_args = _example(remote=[False, True])
    result = model(*batch_args)
    assert [grid.row for grid in result.coarse] == [1]
    proposals = result.coarse[0].proposals_sensor
    assert proposals.dtype == np.float32 and len(proposals) <= 3 and proposals.shape[1] == 3
    assert result.remote.confirmed.tolist() == [False, True]
    assert result.remote.ids[0].eq(-1).all()
    assert result.remote.ids[1, result.remote.support_mask[1]].tolist() == [21, 32, 47]
    assert result.local.ids[1, result.local.support_mask[1]].tolist() == [0, 1, 2]
    assert result.remote.box[1, 0].item() == pytest.approx(20.)
    assert result.remote.match_wins.tolist() == [False, False]  # identity 与无匹配平分
    cold = model(*_example(remote=[True], template=False))
    assert cold.remote.valid.item() and not cold.remote.confirmed.item()
    assert not cold.remote.match_wins.item()


@pytest.mark.parametrize('template_points', [0, 1, 2])
def test_sparse_template_can_remember_a_remote_vote_without_selecting_it(template_points):
    model = _zero_head().eval()
    args = _example(remote=[True], template=False)
    row = args[0]['joint_rows'][0]
    row['pending_valid'] = False
    row['template_xyz'] = np.zeros((template_points, 3), dtype=np.float32)
    # 仅一个真实远端点。首扫只生成暂存位置，复查时仍须同时具备身份模板。
    row['scan_xyz'], row['scan_ids'] = row['scan_xyz'][:1], row['scan_ids'][:1]
    flags = torch.ones(1, 4, dtype=torch.bool)
    candidates = CandidateSet(args[-1], flags, flags, torch.full((1, 4), -1), ~flags[:, 0],
                              torch.zeros(1), torch.tensor([[True, False, False, True]]))
    scores = VerificationOutput(torch.ones(1, 4, 3), torch.ones(1, 4, 3, dtype=torch.bool),
                                torch.zeros(1, 4, 46))
    for pending in (False, True):
        row['pending_valid'] = pending
        result = model(*args)
        six, _, evidence = combine_joint_candidates(candidates, scores, result, args[-2])
        assert not result.template_valid.item() and result.remote.support_mask.sum().item() == 1
        assert evidence.remote_proposed.item() and not evidence.remote_valid.item()
        assert six.raw_valid[0, 5] and not six.valid[0, 5] and not evidence.identity_ok[0, 5]
    result.remote.support_mask.zero_()
    assert not combine_joint_candidates(candidates, scores, result, args[-2])[2].remote_proposed.item()


def test_empty_external_branch_skips_grid_and_keeps_local_supervision(monkeypatch):
    model = _zero_head()
    batch_args = _example(remote=[False])
    def forbidden(*args, **kwargs):
        pytest.fail("unscheduled row entered the coarse network")
    monkeypatch.setattr(model, '_outside_candidates', forbidden)
    output = model(*batch_args)
    batch = dict(batch_args[0], target_box=batch_args[-1][:, 0], joint_track_id=torch.tensor([7]))
    losses = joint_search_losses(output, batch)
    assert losses['loss_joint_foreground'].item() == pytest.approx(np.log(2.))
    expected = sum(losses[k] * value for k, value in JOINT_LOSS_WEIGHTS.items())
    torch.testing.assert_close(losses['loss_joint_total'], expected)
    losses['loss_joint_total'].backward()
    assert batch_args[-2]['features'].grad is None


def test_no_local_or_remote_returns_produces_no_observation_or_vote_loss():
    model = _zero_head()
    with torch.no_grad():
        model.refine[-1].bias.fill_(1.)
    args = _example(remote=[False], template=False)
    args[-2]['valid'] = torch.zeros_like(args[-2]['valid'])
    output = model(*args)
    assert not output.local.valid.item() and not output.remote.valid.item()
    assert not output.local.match_wins.item() and not output.remote.match_wins.item()
    torch.testing.assert_close(output.local.box, args[2].q0_box)
    batch = dict(args[0], target_box=args[-1][:, 0], joint_track_id=torch.tensor([3]))
    losses = joint_search_losses(output, batch)
    assert losses['loss_joint_total'].item() == 0.


def test_cold_bootstrap_rejects_second_coherent_local_vote_mode():
    votes = torch.tensor([[[0., 0., 0.], [.2, 0., 0.], [0., .2, 0.],
                           [8., 0., 0.], [8.1, .1, 0.], [8.2, .2, 0.]]])
    local = SimpleNamespace(vote=votes, point_valid=torch.ones((1, 6), dtype=torch.bool),
                            probability=torch.ones((1, 6)))
    box = torch.zeros((1, 1, 4))
    size = torch.tensor([[4., 2., 2.]])
    assert JointSearchObservation._competing_local_mode(local, box, size).item()
    local.point_valid[0, -1] = False
    assert not JointSearchObservation._competing_local_mode(local, box, size).item()


def test_isolated_higher_score_does_not_hide_competing_three_point_cluster():
    votes = torch.tensor([[[0., 0., 0.], [30., 0., 0.], [8., 0., 0.],
                           [8.1, .1, 0.], [8.2, .2, 0.]]])
    local = SimpleNamespace(vote=votes, point_valid=torch.ones((1, 5), dtype=torch.bool),
                            probability=torch.tensor([[.9, .99, .85, .87, .88]]))
    box = torch.zeros(1, 1, 4)
    size = torch.tensor([[4., 2., 2.]])
    check = JointSearchObservation._competing_local_mode
    assert check(local, box, size, cold_rows=[0]).item()
    assert not check(local, box, size, cold_rows=[]).item()
    local.point_valid[0, -1] = False
    assert not check(local, box, size, cold_rows=[0]).item()


def test_invalid_legacy_region_cannot_veto_valid_local_identity():
    model = _zero_head().eval()
    with torch.no_grad():
        model.shape.compare[-1].bias.fill_(1.)
    batch, obs, decoder, prior, pool, boxes = _example(remote=[False])
    pool['xyz'] = torch.cat((pool['xyz'], pool['xyz'] + torch.tensor([8., 0., 0.])), 1)
    pool['ids'] = torch.arange(6)[None]
    pool['valid'] = pool['ids'] >= 0
    pool['features'] = torch.zeros((1, 6, 64))
    boxes[:, 3, 0] = 8.
    legacy_valid = torch.ones((1, 4), dtype=torch.bool)
    contested = model(batch, obs, decoder, prior, pool, boxes, legacy_valid)
    assert not contested.local.match_wins.item()
    legacy_valid[:, 3] = False
    batch.update(target_box=torch.full((1, 4), 700.), segmentation_labels=torch.zeros((1, 4, 6)))
    result = model(batch, obs, decoder, prior, pool, boxes, legacy_valid)
    assert result.local.match_wins.item() and not result.legacy_match_wins[:, 1].item()
