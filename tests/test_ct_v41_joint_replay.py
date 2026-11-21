"""真实辅助前缀的轮界、主流槽位及配对恢复指标合同。"""
from types import SimpleNamespace

import numpy as np
import pytest

from models.ct_v41.data import EndpointRequest
from models.ct_v41.joint_replay import JointReplayBank
from tests.test_ct_v41_worker_transport import kitti
from tools.compare_v41_joint_search import paired_events


class Rows:
    def __getitem__(self, request):
        return dict(request=request, tracklet_key=str(request.tracklet))


def test_auxiliary_prefix_is_published_next_epoch_and_resumes_at_original_slots():
    bank = JointReplayBank(42)
    bank.begin_epoch(0)
    state = SimpleNamespace(next_frame=1, position=np.array([10.]))
    req = EndpointRequest(0, 3, 5, 1, 2, 1)
    bank.collect([dict(row=Rows()[req], sampled_xyz=np.zeros((3,3)), state=state)], [.09], [3])
    state.position[0] = 100.
    assert not bank.available
    bank.finish_epoch()
    assert bank.available[3,1]['state'].position.tolist() == [10.]
    restored = JointReplayBank(42)
    restored.restore(bank.checkpoint(), completed_epoch=1)
    with pytest.raises(ValueError):
        restored.restore(bank.checkpoint(), completed_epoch=2)
    slots = [Rows()[EndpointRequest(1,99,0,i,i+1,i)] for i in range(1,9)]
    for active in (bank, restored):
        active.begin_epoch(1)
        replaced = active.replace(slots, Rows())
        assert len(replaced) == len(slots)
        assert [i for i,r in enumerate(replaced) if '_joint_replay_state' in r] == [3]
        chosen = replaced[3]
        assert chosen['request'].state_key == slots[3]['request'].state_key
        assert chosen['request'].frame == chosen['_joint_replay_state'].next_frame == 1
        assert chosen['request'].tracklet == 3 and chosen['request'].branch == 0
        assert active.audit()['used'] == {'visible_drift': 1}
        active.finish_epoch()
        assert not active.available


def test_reappearance_requires_three_adjacent_physical_scans_and_keeps_censored_events():
    def rows(scores):
        return [dict(frame=i, scan=scan, timestamp=.1*scan, raw=raw, iou=score,
                     success=score, precision=score)
                for i,(scan,raw,score) in enumerate(zip([0,1,2,3,5,6,7,8,9],
                                              [0,0,1,1,1,0,1,1,1],scores))]
    key = ('0020', 10)
    a = {key: rows([1.,0.,.5,.6,.7,0.,0.,0.,0.])}
    b = {key: rows([1.,0.,.5,.6,.7,0.,.55,.6,.7])}
    result = paired_events(a,b)
    assert result['baseline_lost_raw_returns'] == 2
    assert result['sustained_recovery']['uniform']['censored'] == 2
    assert result['sustained_recovery']['joint']['recovered'] == 1
    assert result['sustained_recovery']['joint']['median_seconds_to_third_good_frame'] == pytest.approx(.2)


def test_replay_donor_uses_raw_ref_outside_gt10_and_keeps_other_main_slots(kitti):
    from models.ct_v41.config import normalize_config
    from models.ct_v41.data import BatchBuilder, build_loaders
    cfg = normalize_config(dict(v41_joint_search=True, ct_engineering_check=True,
                                workers=1, v41_attention_backend='math'))
    loader = build_loaders(cfg, roles=('train',), sources={'train': kitti})['train']
    dataset = loader.dataset
    assert dataset.worker_transport
    req = lambda frame: EndpointRequest(1, 0, 0, frame, frame+1, frame)
    natural = dataset[req(5)]
    seed_builder = BatchBuilder(cfg)
    seed_builder.prepare([natural])
    prefix = seed_builder._pending[0]['state']
    prefix.joint.motion_box[0] = 24.
    prefix.previous_measurement_eligible = True
    bank = JointReplayBank(42)
    bank.available = {(0,5): dict(kind='crop_miss', tracklet_key='0', state=prefix)}
    bank.begin_epoch(1)
    slots = dataset.__getitems__([req(f) for f in (2,3,4,1,5)])
    rows = bank.replace(slots, dataset)
    donor = rows[3]
    assert donor['request'].frame == 5 and donor['request'].state_key == req(1).state_key
    assert donor['request'].state_key != rows[4]['request'].state_key
    assert all('_v41_raw_ref' in f and 'points' in f['_v41_normal_train_cache']
               for f in donor['frames'].values())
    assert 'future_supervision' in donor and '_prepared_start' not in donor
    actual = BatchBuilder(cfg)
    batch = actual.prepare(rows)
    assert actual._pending[3]['state'] is not prefix
    assert batch['anchor_box'][3,0].item() == pytest.approx(24.)
    assert batch['anchor_box'][4,0].item() < 1.
    assert not batch['diagnostic_train_cache_used'][3,-1].item()
    assert batch['point_valid'][3,-1].any()
