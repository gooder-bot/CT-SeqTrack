"""单训：只替换成对的单步槽，前态可跨 worker 运输重建真实裁点。"""
import json

import pytest
import torch

from models.ct_v41.data import (BatchBuilder, EndpointRequest, RawEndpointDataset,
                                _WindowStartCollator)
from models.ct_v41.config import normalize_config
from models.ct_v41.q0_replay import Q0ReplayBank, SCHEMA
from tests.test_ct_v41_worker_transport import kitti


def _item(frame, dataset, x=0.):
    indices = range(max(0, frame - 3), frame)
    state = dict(
        boxes={i: [x, 0., 0., .05] for i in indices}, times={i: i * .1 for i in indices},
        next_frame=frame, size=[4., 2., 2.], trusted=[([0., 0., 0., .05], 0.)],
        transitions={i: True for i in indices}, trusted_velocity=[0., 0., 0.],
        trusted_velocity_valid=False, last_strong_time=0., last_supported_time=0.,
        initialization_time=0., box_provenance={i: 3 if i else 1 for i in indices},
        seed_offset=[0., 0., 0., 0.], previous_is_propagation=False, consecutive_propagations=0,
        consecutive_untrusted_outputs=0, last_supported_quality=1.,
        last_velocity_update_time=None, previous_measurement_eligible=True)
    return dict(tracklet=0, frame=frame, donor_epoch=10, tracklet_key=dataset.source.get_tracklet_key(0),
                state=state)


def _manifest(dataset, path):
    hard, normal = _item(5, dataset, x=24.), _item(2, dataset)
    record = dict(schema=SCHEMA, split='train', source_sha256=dataset.source_sha256,
                  donors=[dict(epoch=e, checkpoint_sha256=str(e % 10) * 64, source_sha256='a'*64)
                          for e in (10, 60)],
                  pairs=[dict(event_id='e10/event1', hard=hard, normal=normal)])
    path.write_text(json.dumps(record), encoding='utf-8')
    return record


def test_replay_replaces_only_complete_pairs_and_restores_raw_crop(kitti, tmp_path):
    raw = RawEndpointDataset(kitti, training=True, enable_m=True, worker_transport=True)
    plain = RawEndpointDataset(kitti, training=True, enable_m=True, worker_transport=False)
    path = tmp_path / 'prefix.json'
    _manifest(raw, path)
    cfg = normalize_config(dict(ct_engineering_check=True, v41_attention_backend='math',
                                v41_q0_lateral_replay=True, v41_q0_replay_manifest=str(path)))
    worker = _WindowStartCollator(cfg)
    requests = [EndpointRequest(0, 0, 0, f, f+1, f) for f in (1, 2, 3, 4)]
    mixed = EndpointRequest(0, 0, 1, 1, 3, 1)
    slots = [requests[0], mixed, *requests[1:]]
    transported = worker(raw.__getitems__(slots))
    assert all('_prepared_start' in row for row in transported)
    bank = Q0ReplayBank(path, 42, raw)
    bank.begin_epoch(0, 4)
    actual = bank.replace(transported[:3], raw) + bank.replace(transported[3:], raw)
    bank.finish_epoch()
    assert len(actual) == len(slots)
    assert actual[:3] == transported[:3]
    assert [r['q0_replay_event']['kind'] for r in actual[-2:]] == ['hard', 'normal']
    assert [r['request'].state_key for r in actual] == [r.state_key for r in slots]
    assert [r['request'].frame for r in actual[-2:]] == [5, 2]
    assert all('_prepared_start' not in r for r in actual[-2:])
    assert bank.audit()['used'] == dict(hard=1, normal=1)
    restored = Q0ReplayBank(path, 42, plain)
    restored.begin_epoch(0, 4)
    control = restored.replace(plain.__getitems__(slots), plain)
    builders = [BatchBuilder(cfg), BatchBuilder(cfg)]
    for builder in builders:
        builder.detailed_diagnostics = False
    got, expected = builders[0].prepare(actual), builders[1].prepare(control)
    assert torch.equal(got['point_ids'], expected['point_ids'])
    assert torch.equal(got['points'], expected['points'])
    assert got['anchor_box'][-2, 0].item() == 24.
    assert got['point_valid'][-2, -1].any()
    assert builders[0]._pending[-2]['state'] is not bank.pairs['e10/event1']['hard']['state']
    bank.begin_epoch(1, 3)
    rows = raw.__getitems__([EndpointRequest(1, 0, 0, f, f+1, f) for f in (1, 2, 3)])
    assert bank.replace(rows, raw) == rows
    bank.finish_epoch()


@pytest.mark.parametrize('error', ['split', 'digest', 'donor', 'noncausal', 'reset', 'event'])
def test_replay_rejects_mismatched_or_leaking_prefix(kitti, tmp_path, error):
    dataset = RawEndpointDataset(kitti, training=True)
    path = tmp_path / 'pairs.json'
    record = _manifest(dataset, path)
    if error == 'split':
        record['split'] = 'test'
    elif error == 'digest':
        record['source_sha256'] = 'different'
    elif error == 'donor':
        record['donors'][0]['epoch'] = 9
    elif error == 'noncausal':
        record['pairs'][0]['hard']['state']['times'][5] = .5
    elif error == 'reset':
        record['pairs'][0]['hard']['state']['box_provenance'][3] = 1
    else:
        record['pairs'].append(record['pairs'][0])
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError):
        Q0ReplayBank(path, 42, dataset)
