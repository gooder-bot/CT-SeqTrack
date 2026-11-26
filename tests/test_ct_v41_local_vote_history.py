from copy import deepcopy
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v41.data import (BatchBuilder, EndpointRequest, KNOWN_GT_BOX, PERTURBED_BOX,
                                RawEndpointDataset, _WindowStartCollator)
from models.ct_v41.history_augmentation import HistoryAugmentation
from models.ct_v41.local_prefix import LocalPrefixBank, SCHEMA
from tests.helpers.raw_tracking import TinySource
from tests.test_ct_v41_history_augmentation import payload, RECEIVER
from tests.test_ct_v41_training import output_for
from tests.test_ct_v41_worker_transport import kitti


class Source(TinySource):
    v41_data_role = 'train'

    def get_tracklet_key(self, index):
        return RECEIVER


def local_builder():
    return BatchBuilder(dict(v41_local_vote=True, v41_history_training='window_1448',
                             ct_engineering_check=True, v41_attention_backend='math', workers=0))


@pytest.mark.parametrize('branch', [0, 1])
def test_midtrack_reset_keeps_observed_edges_and_natural_measurement_ledger(branch):
    dataset = RawEndpointDataset(Source((8,), dt=.1), training=True, enable_m=True, local_vote=True)
    settings = payload()
    settings['receivers'][RECEIVER] = [3, 3, 10, 30, 30, 30, 30, 30]
    builder = local_builder()
    builder.history_bank = HistoryAugmentation(settings)
    # 在合成段 [1,2,3] 之前，唯一已知的状态恰好是首框。
    first = dataset[EndpointRequest(0, 0, 4, 1, 8, 1)]
    ledger = builder._new_state(first)
    def before(key, frame):
        assert (key, frame) == (RECEIVER, 1)
        return deepcopy(ledger)
    builder.local_prefix_bank = SimpleNamespace(before=before)
    mid = dataset[EndpointRequest(1, 0, branch, 4, 5, 4)]
    result = builder.prepare([mid])
    state = builder._pending[0]['state']
    assert not result['trusted_velocity_valid'].item()
    assert state.previous_measurement_eligible is False and state.consecutive_untrusted_outputs == 3
    assert state.last_supported_time == ledger.last_supported_time
    assert state.last_velocity_update_time is None
    np.testing.assert_equal(state.trusted[-1][0], ledger.trusted[-1][0])
    assert result['history_pair_valid'].tolist() == [[True, True]]
    assert result['local_template_valid'].sum().item() == 4
    assert 'joint_rows' not in result and 'recovery_seed_box' not in result
    assert state.box_provenance[3] == (KNOWN_GT_BOX if branch == 0 else PERTURBED_BOX)
    assert abs(state.seed_offset[2] - (3. if branch else 0.)) < 1e-6


def test_prefix_bank_requires_complete_matching_train_coverage(tmp_path):
    dataset = RawEndpointDataset(Source((4,), dt=.1), training=True)
    path = tmp_path / 'ledger.sqlite'
    receipt = dict(schema=SCHEMA, split='train', complete=True, donor_epoch=60, center_weighting='uniform',
                   checkpoint_sha256='1'*64, evaluator_sha256='2'*64, config_sha256='3'*64, run_id='r',
                   source_sha256=dataset.source_sha256, tracks=[[RECEIVER, 4]], endpoints=3)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE states(track TEXT, frame INTEGER, state BLOB, PRIMARY KEY(track,frame))')
        db.execute('CREATE TABLE metadata(value TEXT)')
        db.execute('INSERT INTO metadata VALUES(?)', (json.dumps(receipt),))
        db.executemany('INSERT INTO states VALUES(?,?,?)', [(RECEIVER, i, b'') for i in range(1, 4)])
    bank = LocalPrefixBank(path, dataset)
    other = RawEndpointDataset(Source((4,), dt=.5), training=True)
    with pytest.raises(ValueError, match='split'):
        bank.validate_dataset(other)
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM states WHERE frame=2')
    with pytest.raises(ValueError, match='endpoints'):
        LocalPrefixBank(path)


class SparseFirst(Source):
    def get_frames(self, index, frame_ids):
        rows = super().get_frames(index, frame_ids)
        for i, row in zip(frame_ids, rows):
            if index == 0 and i == 0:
                row['pc'] = row['pc'][:2]
                row['point_ids'] = row['point_ids'][:2]
        return rows


def test_sparse_first_frame_does_not_borrow_a_template_from_later_scans():
    data = RawEndpointDataset(SparseFirst((7,), dt=.1), local_vote=True)
    later = data[EndpointRequest(0, 0, 4, 1, 7, 6)]
    xyz, valid = later['local_template']
    assert not valid.any() and not xyz.any()
    assert data[EndpointRequest(0, 0, 4, 1, 7, 1)]['local_template'] is later['local_template']


@pytest.mark.parametrize('worker_start', [False, True])
def test_cold_template_requires_adjacent_real_writes_and_is_private_to_window(kitti, worker_start):
    first = np.array([[0., 0., 0., .5], [0., .1, 0., .5]], np.float32)
    first.tofile(kitti.KITTI_velo / '0000' / '000000.bin')
    dataset = RawEndpointDataset(kitti, training=True, enable_m=True, local_vote=True,
                                 worker_transport=worker_start)
    builder = local_builder()
    builder.detailed_diagnostics = False
    worker = _WindowStartCollator(builder.config)
    worker.builder = local_builder()
    worker.builder.detailed_diagnostics = False
    for instance in (builder, worker.builder):
        instance.history_bank = SimpleNamespace(sha256='e57')
        instance.local_prefix_bank = SimpleNamespace(sha256='e60')
    for frame in range(1, 6):
        req = EndpointRequest(0, 0, 0, 1, 6, frame)
        rows = dataset.__getitems__([req])
        if worker_start:
            rows = worker(rows)
            assert ('_prepared_start' in rows[0]) == (frame == 1)
        batch = builder.prepare(rows)
        if frame < 5:
            assert not batch['local_template_valid'].any()
        else:
            actual = batch['local_template_xyz'][0, batch['local_template_valid'][0]].numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0.)
        if frame == 4:
            # 模拟测量端缩短点轴、调换顺序并重复一个物理 ID；按当前 mask 的顺序构造实际测量。
            order = torch.cat((torch.arange(63, -1, -1), torch.tensor([17])))
            for name in ('points', 'point_ids', 'point_valid'):
                batch[name] = batch[name][:, :, order]
        out = output_for(batch, selected=3 if frame in (2, 4) else 0, score=.2 if frame == 1 else .9)
        if frame == 3:
            out.prior.box[:, 0] += 10.  # 可信位置仍可写入，但旧测速拒绝本次创新。
        if frame == 4:
            support = out.commit_decision.support_mask[0]
            assert support.sum() >= 3
            assert len(torch.unique(batch['point_ids'][0, -1, support])) == support.sum()
            xyz = batch['points'][0, -1, support, :3].numpy().astype(np.float64)
            box = out.accepted_box[0].numpy().astype(np.float64)
            x, y, z = (xyz - box[:3]).T
            c, s = np.cos(box[3]), np.sin(box[3])
            expected = np.column_stack((c*x + s*y, -s*x + c*y, z))
        builder.commit(out, batch)
        if frame in (1, 2, 3):
            assert builder.states[req.state_key].local_vote_template is None
    assert not builder.states
    new_request = EndpointRequest(1, 0, 1, 1, 3, 1)
    independent = builder.prepare(dataset.__getitems__([new_request]))
    assert not independent['local_template_valid'].any()
    assert not dataset._local_templates[0][1].any()


def test_ready_and_recovered_templates_remain_fixed_in_mixed_batch():
    dataset = RawEndpointDataset(SparseFirst((6, 6)), local_vote=True)
    builder = local_builder()
    builder.detailed_diagnostics = False
    for frame in range(1, 5):
        # 把待补模板放在第二行，检查从压缩后的支持行取点时不串轨。
        requests = [EndpointRequest(0, t, 4, 1, 6, frame) for t in (1, 0)]
        batch = builder.prepare(dataset.__getitems__(requests))
        initial_xyz, initial_valid = dataset._local_templates[1]
        np.testing.assert_array_equal(batch['local_template_xyz'][0], initial_xyz)
        np.testing.assert_array_equal(batch['local_template_valid'][0], initial_valid)
        assert bool(batch['local_template_valid'][1].any()) == (frame > 2)
        if frame > 2:
            np.testing.assert_array_equal(batch['local_template_xyz'][1], recovered[0])
        builder.commit(output_for(batch, selected=3), batch)
        cold = builder.states[requests[1].state_key]
        if frame == 2:
            recovered = cold.local_vote_template
        elif frame > 2:
            assert cold.local_vote_template is recovered
    assert not dataset._local_templates[0][1].any()


def test_previous_measured_velocity_ages_through_a_clean_midtrack_reset():
    ds = RawEndpointDataset(Source((10,), dt=.1), training=True, enable_m=True, local_vote=True)
    builder = local_builder()
    initial = ds[EndpointRequest(0, 0, 4, 1, 10, 1)]
    prior = builder._new_state(initial)
    prior.next_frame = 4
    prior.times = {i: ds.source.get_frame_metadata(0, i)['timestamp'] for i in (1, 2, 3)}
    prior.boxes = {i: np.array([-.2*i, .1*i, .02*i, 0.]) for i in prior.times}
    prior.trusted = [(prior.boxes[i].copy(), prior.times[i]) for i in (2, 3)]
    prior.trusted_velocity[:] = [-2., 1., .2]
    prior.trusted_velocity_valid = True
    prior.last_velocity_update_time = prior.times[3]
    prior.last_supported_time = prior.times[3]
    builder.local_prefix_bank = SimpleNamespace(before=lambda key, index: deepcopy(prior) if index == 4 else None)
    batch = builder.prepare([ds[EndpointRequest(9, 0, 0, 7, 8, 7)]])
    np.testing.assert_allclose(batch['trusted_velocity'][0].numpy(), [-2., 1., .2])
    assert batch['trusted_velocity_valid'].item()
    assert batch['trusted_velocity_age'].item() == pytest.approx(.4)
    assert batch['supported_age'].item() == pytest.approx(.4)
