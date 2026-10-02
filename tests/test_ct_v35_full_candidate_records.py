"""完整候选台账与实际 forward 一致，且不改变 RNG、梯度及 Adam 更新。"""
from copy import deepcopy
from dataclasses import replace
import json
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v31.candidate_records import candidate_records, CANDIDATE_RECORD_SCHEMA
from models.ct_v31.data import BatchBuilder, RawEndpointDataset
from models.ct_v31.model import JointTracker
from models.ct_v31.runtime import TrackingEvaluation, capture_rng_state
from tests.test_ct_v31_joint import make_batch
from tests.test_ct_v34_resume import _assert_identical
from tests.test_ct_v35_data_diagnostics import v35_config
from tests.test_ct_v31_runtime import TinySource, fake_prior, request


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _batch():
    batch = make_batch(1, 8)
    batch['history_is_initial'] = torch.tensor([[True, False, False]])
    batch['anchor_box'][:] = torch.tensor([100., -200., 4., 1.2])
    # 不同 GT 成员标签可验证 purity，而不影响实际前向。
    batch['extension_labels'][:, 1:24:2] = 0
    return batch


def _raw():
    return [dict(request=SimpleNamespace(branch=4, frame=1), tracklet_key='candidate/test',
                 first_frame=dict(timestamp=0., scene_id='test'),
                 frames={1: dict(timestamp=.5, scene_id='test')})]


@pytest.mark.parametrize('arm', ['b1', 'b1_b2', 'full'])
def test_candidate_ledger_matches_actual_boxes_members_and_padding(arm):
    torch.manual_seed(713)
    config = v35_config(v31_arm=arm)
    batch = _batch()
    model = JointTracker(config).eval()
    output = model(batch)
    records = candidate_records(batch, output)
    json.dumps(records, allow_nan=False)
    record = records[0]
    assert record['schema'] == CANDIDATE_RECORD_SCHEMA
    assert record['acquisition']['search_target_count'] is None
    np.testing.assert_array_equal(record['boxes_anchor_relative'], output.hypothesis_boxes[0].detach())
    world = output.hypothesis_boxes[0].detach().double().numpy().copy()
    world[:, :3] += batch['anchor_box'][0, :3].double().numpy()
    np.testing.assert_array_equal(record['boxes_world'], world)
    # anchor yaw 不再旋转已经是世界轴的公开 XYZ，也不重复加到 absolute yaw。
    np.testing.assert_array_equal(np.asarray(record['boxes_world'])[:, 3], world[:, 3])
    np.testing.assert_array_equal(record['quality'], output.quality_logits[0].detach().sigmoid())
    assert record['valid'] == output.hypothesis_valid[0].tolist()
    chosen = record['selected_index']
    assert record['valid'][chosen]
    assert record['selected_quality'] == record['quality'][chosen]
    ledger, evidence = record['extension'], output.evidence
    assert ledger['capacity'] == 256
    slots = torch.tensor(ledger['slots'], dtype=torch.long)
    assert ledger['slots'] == evidence.point_valid[0].nonzero().flatten().tolist()
    assert ledger['raw_ids'] == evidence.point_ids[0, slots].tolist()
    assert ledger['acquisition_indices'] == evidence.point_indices[0, slots].tolist()
    assert ledger['target_labels'] == batch['extension_labels'][0, evidence.point_indices[0, slots]].tolist()
    assert len(ledger['raw_ids']) == len(set(ledger['raw_ids']))
    assert (evidence.point_ids[0, ~evidence.point_valid[0]] == -1).all()
    for k, mode in enumerate(record['modes']):
        member = evidence.members[0, k] & evidence.point_valid[0]
        member_slots = member.nonzero().flatten()
        assert mode['member_slots'] == member_slots.tolist()
        assert mode['member_raw_ids'] == evidence.point_ids[0, member_slots].tolist()
        assert set(mode['member_slots']).issubset(ledger['slots'])
        target_count = int((batch['extension_labels'][0, evidence.point_indices[0, member_slots]] > 0).sum())
        assert mode['target_count'] == target_count
        assert mode['target_purity'] == (target_count / len(member_slots) if len(member_slots) else None)
        assert (record['geometry'][k + 1] is not None) == record['valid'][k + 1]
    if arm == 'b1':
        assert ledger['slots'] == [] and not any(record['mode_formed'])
    elif arm == 'b1_b2':
        assert any(record['mode_formed']) and record['valid'] == [True, False, False, False]
        assert all(g is None for g in record['geometry'][1:])
    evaluation = TrackingEvaluation(config)
    evaluation.add_batch(_raw(), batch, output)
    row = evaluation.rows[-1]
    assert row['diagnostic_candidates'] == record
    assert record['geometry'][chosen]['iou'] == row['iou']
    assert record['geometry'][chosen]['center_error_m'] == row['distance']
    assert record['geometry'][0] == row['fine_geometry']


@pytest.mark.parametrize('extension_exists', [True, False])
def test_prior_fallback_and_invalid_candidates_are_explicit(extension_exists):
    torch.manual_seed(91)
    batch = _batch()
    batch['point_valid'].zero_()
    if not extension_exists:
        batch['extension_valid'].zero_()
        batch['extension_ids'].fill_(-1)
        batch['extension_labels'].fill_(-1)
    output = JointTracker(v35_config(v31_arm='full')).eval()(batch)
    record = candidate_records(batch, output)[0]
    assert record['q0_prior_fallback'] and not record['sequence_valid']
    assert record['boxes_anchor_relative'][0] == record['prior']['box_anchor_relative']
    assert record['boxes_world'][0] == record['prior']['box_world']
    if extension_exists:
        assert record['extension']['slots']
        assert any(record['mode_formed'])
    else:
        assert record['extension']['slots'] == []
        assert record['valid'] == [True, False, False, False]
        assert record['geometry'][1:] == [None, None, None]
    json.dumps(record, allow_nan=False)


def _step(with_records):
    random.seed(37)
    np.random.seed(37)
    torch.manual_seed(37)
    config = v35_config(v31_arm='full')
    model = JointTracker(config).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)
    batch = _batch()
    before_batch = deepcopy(batch)
    output = model(batch)
    before_rng = capture_rng_state()
    evaluation = TrackingEvaluation(config)
    evaluation.record_candidates = with_records
    evaluation.add_batch(_raw(), batch, output)
    _assert_identical(before_rng, capture_rng_state())
    _assert_identical(before_batch, batch)
    losses = model.compute_losses(batch, output)
    losses['loss_total'].backward()
    gradients = {name: p.grad.detach().clone() if p.grad is not None else None
                 for name, p in model.named_parameters()}
    optimizer.step()
    return dict(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                gradients=gradients, rng=capture_rng_state(),
                predictions=output.hypothesis_boxes.detach().clone(),
                loss=losses['loss_total'].detach().clone()), evaluation.rows[-1]


def test_candidate_recording_does_not_change_forward_gradients_rng_or_adam():
    plain, old_row = _step(False)
    recorded, row = _step(True)
    _assert_identical(plain, recorded)
    assert 'diagnostic_candidates' in row and 'diagnostic_candidates' not in old_row
    assert {k: v for k, v in row.items() if k != 'diagnostic_candidates'} == old_row


@pytest.mark.parametrize('version,arm', [('ctseqtrackv33', 'full'), ('ctseqtrackv34', 'full'),
                                      ('ctseqtrackv35', 'b0'), ('seqtrack_reference', 'b0')])
def test_old_versions_and_v35_b0_do_not_add_candidate_fields(version, arm):
    evaluation = TrackingEvaluation(dict(net_model=version, v31_arm=arm))
    assert not evaluation.record_candidates


class _DenseExtensionSource(TinySource):
    def get_frames(self, index, frame_ids):
        rows = super().get_frames(index, frame_ids)
        for frame, row in zip(frame_ids, rows):
            if frame:
                # 1000 个 target 点在实际 u=0 支持域内，但超过 768 池容量；
                # 另 50 个背景点只在更大的搜索域里，不能误记作实际获取可见。
                xyz = np.zeros((1050, 3))
                xyz[:1000, 0] = np.linspace(4.9, 5.1, 1000)
                xyz[:1000, 1] = np.sin(np.arange(1000)) * .1
                xyz[1000:, 0] = np.linspace(10., 10.5, 50)
                row['pc'], row['point_ids'] = xyz, np.arange(1050)
                row['3d_bbox'] = np.array([5., 0., 0., .1])
        return rows


def test_actual_search_counts_precede_pool_sampling_and_remain_passive():
    raw = RawEndpointDataset(_DenseExtensionSource((2,)))
    rows = [raw[request(1, branch=4, end=2)]]
    config = v35_config(v31_arm='full')
    builder, plain = BatchBuilder(config), BatchBuilder(config)
    batch, baseline = builder.prepare(rows), plain.prepare(rows)
    # prepare 后仅关闭新台账 gate，保持相同初始化与获取输入。
    plain.is_v35 = False
    model = JointTracker(config).eval()
    prior = replace(model.plan_prior(batch), acquisition_fraction=torch.zeros(1, 2))
    state_before = pickle.dumps(builder.states)
    rng_before = capture_rng_state()
    builder.acquire(batch, prior, training=False)
    plain.acquire(baseline, prior, training=False)
    _assert_identical(rng_before, capture_rng_state())
    assert pickle.dumps(builder.states) == state_before
    search_keys = [k for k in batch if k.startswith('diagnostic_search_')]
    assert len(search_keys) == 4
    _assert_identical({k: v for k, v in batch.items() if k not in search_keys}, baseline)
    assert batch['diagnostic_search_point_count'].item() == 1000
    assert batch['diagnostic_search_target_count'].item() == 1000
    assert batch['extension_valid'].sum().item() == 768
    assert batch['diagnostic_acquired_count'].item() == 768
    assert batch['diagnostic_search_point_count_by_partition'].sum().item() == 1000
    assert batch['diagnostic_search_target_count_by_partition'].sum().item() == 1000
    output = model(batch, prior=prior)
    ledger = candidate_records(batch, output)[0]['acquisition']
    assert ledger['search_target_count'] == 1000 and ledger['target_count'] == 768
    assert ledger['search_point_count_by_partition'] == batch['diagnostic_search_point_count_by_partition'][0].tolist()


@pytest.mark.parametrize('arm,training', [('b0', False), ('b1_b2', True), ('full', True)])
def test_search_counters_do_not_expand_b0_or_training(arm, training):
    raw = RawEndpointDataset(TinySource((2,)))
    rows = [raw[request(1, branch=4, end=2)]]
    builder = BatchBuilder(v35_config(v31_arm=arm))
    batch = builder.prepare(rows)
    builder.acquire(batch, fake_prior(batch), training=training)
    assert not any(k.startswith('diagnostic_search_') for k in batch)
