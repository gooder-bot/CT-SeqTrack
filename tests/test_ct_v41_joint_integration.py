"""联合分支从配置、公开输入、候选选择到逐点监督的合同。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v41.config import config_identity, normalize_config, recipe_identity
from models.ct_v41.candidate_records import candidate_records
from models.ct_v41.contracts import CandidateSet, VerificationOutput
from models.ct_v41.identity import JOINT_CANDIDATES
from models.ct_v41.input import online_batch
from models.ct_v41.joint_candidates import combine_joint_candidates
from models.ct_v41.model import JointTracker, select_candidates
from tests.helpers.synthetic_batch import batch as synthetic_batch


def measurement(center, ids, score):
    box = torch.tensor([[center, 0., 0., 0.]])
    ids = torch.tensor([ids])
    real = ids >= 0
    return SimpleNamespace(box=box, ids=ids, xyz=box[:, None, :3].expand(-1, ids.shape[1], -1),
        support_mask=real, valid=real.any(-1), match_wins=real.any(-1),
        logits=torch.full((1, 3), float(score)), confirmed=torch.tensor([False]))


def test_remote_point_set_and_two_scan_selection_are_independent_of_local_support():
    boxes = torch.zeros(1, 4, 4)
    flags = torch.ones(1, 4, dtype=torch.bool)
    candidates = CandidateSet(boxes, flags, flags, torch.full((1, 4), -1),
        torch.tensor([False]), torch.zeros(1), torch.tensor([[True, False, False, True]]))
    verification = VerificationOutput(torch.zeros(1, 4, 3), torch.ones(1, 4, 4, dtype=torch.bool),
                                      torch.zeros(1, 4, 46))
    pool = dict(xyz=torch.zeros(1, 4, 3), ids=torch.arange(4)[None])
    remote = measurement(18., [81, 93, 111], 9.)
    observed = SimpleNamespace(local=measurement(0., [0, 1, 2, 3], -2.), remote=remote,
        local_position_valid=torch.tensor([True]),
        legacy_match_wins=torch.tensor([[False, False]]), template_valid=torch.tensor([True]))
    for confirmed, winner, eligible in ((False, 0, False), (True, 5, True)):
        remote.confirmed.fill_(confirmed)
        six, scores, joint = combine_joint_candidates(candidates, verification, observed, pool)
        decision, commit, _, _ = select_candidates(six, scores, None, {}, joint=joint)
        assert six.raw_valid[0, 5] and bool(six.valid[0, 5]) == confirmed
        assert decision.selected_index.item() == winner and commit.eligible.item() == eligible
        target = dict(target_box=remote.box, box_size=torch.ones(1, 3))
        members = joint.target_members(target)
        assert not members[0, :5].any() and members[0, 5].sum() == 3
        if confirmed:
            assert joint.support_ids[0, 5, commit.support_mask[0]].tolist() == [81, 93, 111]
    for rejected in ('template', 'identity', 'geometry', 'support'):
        observed.template_valid.fill_(rejected != 'template')
        observed.remote.match_wins.fill_(rejected != 'identity')
        observed.remote.logits[0, 2] = -1. if rejected == 'geometry' else 1.
        observed.remote.support_mask[0, -1] = rejected != 'support'
        six, _, _ = combine_joint_candidates(candidates, verification, observed, pool)
        assert not six.valid[0, 5]


@pytest.mark.parametrize('winner', range(len(JOINT_CANDIDATES)), ids=JOINT_CANDIDATES)
def test_each_joint_slot_can_win_but_only_measurements_can_write(winner):
    boxes = torch.zeros(1, 4, 4)
    boxes[0, :, 0] = torch.arange(4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    measurement_slots = torch.tensor([[True, False, False, True]])
    candidates = CandidateSet(boxes, valid, valid, torch.full((1, 4), -1),
                              torch.tensor([False]), torch.zeros(1), measurement_slots)
    evidence = VerificationOutput(torch.zeros(1, 4, 3), valid[:, :, None].expand(-1, -1, 3),
                                  torch.zeros(1, 4, 46))
    local = measurement(4., [0, 1, 2], 0.)
    remote = measurement(5., [100, 101, 102], 0.)
    remote.confirmed.fill_(True)
    observed = SimpleNamespace(local=local, remote=remote, local_position_valid=local.valid,
        legacy_match_wins=torch.ones(1, 2, dtype=torch.bool), template_valid=torch.tensor([True]))
    pool = dict(xyz=local.xyz, ids=local.ids)
    six, scores, joint = combine_joint_candidates(candidates, evidence, observed, pool)
    scores.logits[0, winner, :2] = 5.
    decision, commit, _, _ = select_candidates(six, scores, None, {}, joint=joint)
    assert six.valid.all() and decision.selected_index.item() == winner
    assert commit.eligible.item() == (winner in (0, 3, 4, 5))
    assert commit.support_mask.sum().item() == (3 if commit.eligible.item() else 0)


def test_joint_public_forward_and_auxiliary_loss():
    cfg = dict(v41_joint_search=True, ct_engineering_check=True, v41_attention_backend='math',
               workers=0, batch_size=2, point_sample_size=16)
    tracker = JointTracker(cfg)
    batch = synthetic_batch(2, 16)
    batch.update(recovery_seed_box=batch['fallback_box'].clone(), recovery_seed_valid=torch.ones(2).bool(),
        joint_motion_boxes=batch['history_boxes'], joint_motion_times=batch['history_times'],
        joint_motion_valid=batch['history_valid'], joint_motion_pairs=batch['history_pair_valid'],
        joint_motion_origin=torch.zeros(2, 4), trusted_velocity=torch.zeros(2, 3),
        joint_track_id=torch.arange(2), joint_other_cars=(np.empty((0, 7)),) * 2)
    row = dict(template_xyz=np.zeros((3, 3), np.float32), scan_xyz=np.empty((0, 3), np.float32),
               scan_ids=np.empty(0, np.int64), pending_valid=False, pending_box=np.zeros(4),
               main_box=np.zeros(4), backup_valid=False, backup_box=np.zeros(4), label_sentinel=123)
    batch['joint_rows'] = (row, row)
    online = online_batch(batch, joint_search=True)
    assert not {'joint_track_id', 'joint_other_cars', 'target_box'} & online.keys()
    assert 'label_sentinel' not in online['joint_rows'][0]
    tracker.eval()
    with torch.no_grad():
        output = tracker(batch)
        main_losses = tracker.compute_losses(batch, output)
        records = candidate_records(batch, output)
    assert output.candidates.boxes.shape == (2, 6, 4)
    assert not output.candidates.raw_valid[:, 5].any()
    assert torch.isfinite(main_losses['loss_total'])
    assert records[0]['candidate_order'] == list(JOINT_CANDIDATES)
    assert records[0]['coarse_hits_target_bev'] == [] and records[0]['remote_sampled_target_count'] == 0
    tracker.train()
    auxiliary = tracker.forward_aux(batch)
    losses = tracker.compute_aux_losses(batch, auxiliary)
    assert torch.isfinite(losses['loss_total']) and 'loss_geometry_quality' in losses
    losses['loss_total'].backward()
    assert any(p.grad is not None for p in tracker.joint_search.parameters())
    assert all(p.grad is None for p in tracker.observation.parameters())


def test_joint_recipe_identity_and_legacy_configuration():
    base = normalize_config()
    cfg = normalize_config(dict(v41_joint_search=True))
    assert config_identity(base) != config_identity(cfg)
    assert 'local_vote' not in recipe_identity(base)['candidates']
    assert recipe_identity(cfg)['candidates'] == list(JOINT_CANDIDATES)
    assert normalize_config(cfg) == cfg
    with pytest.raises(ValueError):
        normalize_config(dict(v41_joint_search=True, v41_q0_center_weighting='inverse_lwh'))
    with pytest.raises(ValueError):
        normalize_config(dict(v41_joint_latency=True))


def test_joint_model_and_builder_commit_two_physical_scans():
    from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
    from tests.helpers.raw_tracking import TinySource

    cfg = dict(v41_joint_search=True, ct_engineering_check=True, v41_attention_backend='math',
               workers=0, batch_size=1)
    dataset = RawEndpointDataset(TinySource((3,), dt=1.), joint_search=True)
    builder = BatchBuilder(cfg)
    builder.detailed_diagnostics = True
    tracker = JointTracker(cfg).eval()
    with torch.no_grad():
        for frame in (1, 2):
            rows = [dataset[EndpointRequest(0, 0, 4, 1, 3, frame)]]
            batch = builder.prepare(rows)
            result = tracker(batch)
            records = builder.commit(result, batch, diagnostics=True, defer_joint_diagnostics=True)
            builder.complete_joint_diagnostics(rows, records, dataset)
            assert records[0]['joint_scan_called']
            assert 0 <= records[0]['joint_extra_points'] <= records[0]['joint_fine_points'] <= 256
            assert records[0]['joint_max_other_support_count'] == 0
            assert '_joint_support_xyz' not in records[0]
    assert not builder.states
