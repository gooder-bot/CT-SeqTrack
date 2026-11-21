from dataclasses import replace

import numpy as np
import pytest
import torch

from models.ct_v41.config import normalize_config
from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
from models.ct_v41.fixed_replay import FixedReplayBank, _decode, _encode, weights_sha256
from models.ctseqtrackv41 import CTSEQTRACKV41
from tests.test_ct_v41_worker_transport import WorkerKITTI, kitti


class TrainKITTI(WorkerKITTI):
    v41_source_role = 'train'

    def get_tracklet_key(self, track):
        return f'kitti_v41/kitti_tracking/train/0000/{track}'


def test_fixed_catalog_generated_by_real_joint_forward_pairing_and_epoch_resume(tmp_path, kitti, monkeypatch):
    root = kitti.KITTI_velo
    # 第一预测帧无回波，第二帧重现。真实 KITTI bin 和联合 tracker 产生可重放前态。
    np.full((8, 4), 100., np.float32).tofile(root / '0000' / '000001.bin')
    monkeypatch.setattr('datasets.kitti_v41.get_v41_raw_dataset',
                        lambda cfg, role: TrainKITTI(root, role))
    dataset = RawEndpointDataset(TrainKITTI(root), training=True, joint_search=True, enable_m=True)
    bank = FixedReplayBank(tmp_path / 'natural.sqlite', 42)
    cfg = normalize_config(dict(v41_joint_search=True, v41_history_training='window_1448',
                                v41_fixed_replay_bank=str(bank.path), ct_engineering_check=True,
                                batch_size=1, workers=0, v41_attention_backend='math'))
    host = CTSEQTRACKV41(cfg)
    host._completed_epoch = 2
    host.train()
    host.tracker.observation.eval()
    modes = [m.training for m in host.tracker.modules()]
    digest = weights_sha256(host.tracker)
    rng = torch.get_rng_state()
    bank.begin_epoch(0)
    request = EndpointRequest(0, 0, 0, 2, 3, 2)
    row = dataset[request]
    assert bank.replace([row], dataset)[0] is row and not bank.path.exists()
    bank.finish_epoch()
    with pytest.raises(ValueError, match='E3'):
        bank.begin_epoch(2)
    bank.ensure_catalog(host, dataset)
    assert bank.manifest_sha256 and bank.donor_weights_sha256 == digest == weights_sha256(host.tracker)
    assert modes == [m.training for m in host.tracker.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert (0, 2) in bank.groups['reappearance']
    bank.begin_epoch(2)
    rows = [dataset[EndpointRequest(2, 0, 0, i, i + 1, i)] for i in range(1, 6)]
    other_branch = dataset[replace(request, epoch=2, branch=1)]
    replayed = bank.replace(rows + [other_branch], dataset)
    bank.finish_epoch()
    assert [r['request'] for r in replayed] == [r['request'] for r in rows + [other_branch]]
    assert replayed[-1] is other_branch
    prefix = replayed[1]['_joint_replay_state']
    assert prefix.next_frame == 2 and prefix.joint.motion_time == prefix.times[1]
    assert prefix.box_provenance[1] != prefix.box_provenance[0]
    builder = BatchBuilder(cfg)
    builder.prepare([replayed[1]])
    assert builder._pending[0]['state'] is not prefix
    with pytest.raises(ValueError, match='duplicate'):
        bank.replace([rows[1]], dataset)
    saved = bank.checkpoint()
    resumed = FixedReplayBank(bank.path, 42)
    resumed.restore(saved, 3)
    resumed.bind(dataset)
    for active in (resumed, bank):
        active.begin_epoch(3)
    assert resumed.audit()['plan_sha256'] == bank.audit()['plan_sha256']
    with pytest.raises(ValueError, match='weights'):
        FixedReplayBank(bank.path, 42).bind(dataset, expected_weights_sha256='1' * 64)
    changed = FixedReplayBank(bank.path, 42)
    changed.restore(dict(saved, manifest_sha256='f' * 64), 3)
    with pytest.raises(ValueError, match='differs'):
        changed.bind(dataset)
    prefix.joint.motion_time = 10.
    with pytest.raises(ValueError, match='noncausal'):
        _decode(_encode(prefix), 2, 2)


def test_fixed_catalog_needs_both_complete_shared_epochs(tmp_path):
    bank = FixedReplayBank(tmp_path / 'missing.sqlite', 42)
    bank.begin_epoch(1)
    untouched = bank.checkpoint()
    bank.restore(untouched, 2)
    with pytest.raises(ValueError, match='provenance'):
        bank.restore(dict(untouched, completed_epoch=3), 3)
