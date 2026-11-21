"""供体选样、物理事件配对与前态复制无需 KITTI / CUDA。"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tools.build_v41_q0_replay import (E60_RECEIPT, collect, difficulty, match_events,
                                      prefix_state, publish, track_role, validate_donor)


def test_difficulty_uses_gt_width_axis_and_rectangular_yaw():
    target = np.array([1., 2., 3., math.pi / 2])
    pred = target + np.array([.31, 2., .16, math.pi - math.radians(11)])
    errors, reasons = difficulty(pred, target, .49)
    assert errors['lateral_m'] == pytest.approx(.31)
    assert errors['axis_yaw_deg'] == pytest.approx(11.)
    assert set(reasons) == {'lateral_m', 'height_m', 'axis_yaw_deg', 'coarse_iou'}
    assert difficulty(target, target, .5)[1] == []


def row(track, frame, epoch, iou, *, raw=None, distance=10):
    return dict(tracklet=track, tracklet_key=str(track), frame=frame, donor_epoch=epoch,
                raw_frame=frame if raw is None else raw, distance_m=distance, target_points=12,
                errors=dict(coarse_iou=iou), reasons=['coarse_iou'] if iou < .5 else [])


def test_contiguous_cross_donor_events_have_deterministic_distinct_controls():
    rows = [row(0, 1, 10, .6, distance=15), row(0, 1, 60, .7, distance=15),
            row(0, 3, 10, .3), row(0, 3, 60, .9), row(0, 4, 60, .1),
            row(0, 5, 60, .2, raw=20), row(1, 2, 60, .8)]
    pairs, summary = match_events(rows)
    assert summary == dict(events=2, matched_events=2, unmatched_events=0)
    assert [(p['hard']['frame'], p['normal']['tracklet'], p['normal']['frame']) for p in pairs] == [
        (4, 0, 1), (5, 1, 2)]
    assert pairs == match_events(list(reversed(rows)))[0]
    assert match_events([row(0, 1, 10, .3)])[1]['unmatched_events'] == 1
    assert match_events([row(0, 1, 10, .3), row(0, 2, 10, .8, distance=40)])[0] == []
    assert match_events([])[0] == []


def test_precommit_snapshot_is_detached_and_future_or_wrong_role_is_rejected():
    state = SimpleNamespace(next_frame=3, boxes={i: np.arange(4.) for i in range(3)},
                            trusted=[(np.zeros(4), np.float64(2))])
    record = prefix_state(state, 3)
    state.boxes[2][0] = 99
    assert record['boxes']['2'][0] == 0.
    assert json.loads(json.dumps(record))['trusted'][0] == [[0., 0., 0., 0.], 2.]
    state.boxes[3] = np.zeros(4)
    with pytest.raises(ValueError, match='prefix'):
        prefix_state(state, 3)
    for invalid in ('kitti_v41/kitti_tracking/dev/0017/1', 'kitti_v41/kitti_tracking/train/0020/1'):
        with pytest.raises(ValueError, match='train scenes'):
            track_role(SimpleNamespace(get_tracklet_key=lambda _: invalid), 0)


def test_manifest_publication_is_atomic_and_exclusive(tmp_path):
    path = tmp_path / 'bank' / 'manifest.json'
    publish(path, {'value': 1})
    with pytest.raises(FileExistsError):
        publish(path, {'value': 2})
    assert json.loads(path.read_text(encoding='utf-8')) == {'value': 1}
    with pytest.raises(ValueError):
        publish(path.with_name('invalid.json'), {'number': float('nan')})
    assert list(path.parent.iterdir()) == [path]


def test_known_e60_receipt_binds_weight_evaluator_and_dataset(monkeypatch):
    from models.ct_v41 import runtime

    receipt = E60_RECEIPT
    dataset = dict(dataset_manifest={'version': 1}, input_metadata={'files': 3})
    manifest = dict(dataset, source={'sha256': receipt['evaluator_source']},
                    run_id=receipt['run_id'], config_sha256=receipt['config_sha256'])
    payload = dict(manifest, dataset_identity=dataset, epoch_complete=True, completed_epoch=60,
                   source={'sha256': receipt['saved_source']}, sampler={'source_sha256': 'train'},
                   execution_identity={'source_sha256': receipt['saved_source']}, auxiliary={})
    calls = []
    monkeypatch.setattr(runtime, 'dataset_identity', lambda cfg: dataset)
    monkeypatch.setattr(runtime, 'validate_auxiliary_payload', lambda *args: calls.append(args))
    policy = validate_donor(payload, None, manifest, 60, receipt['checkpoint_sha256'], 'train')
    assert policy == 'audited_uniform_e60_fixed_test_receipt' and len(calls) == 1
    with pytest.raises(ValueError, match='receipt'):
        validate_donor(payload, None, manifest, 60, 'a'*64, 'train')
    bad_source = dict(manifest, source={'sha256': 'b'*64})
    with pytest.raises(ValueError, match='receipt'):
        validate_donor(payload, None, bad_source, 60, receipt['checkpoint_sha256'], 'train')
    monkeypatch.setattr(runtime, 'dataset_identity', lambda cfg: dict(dataset, input_metadata={'files': 2}))
    with pytest.raises(ValueError, match='KITTI data'):
        validate_donor(payload, None, manifest, 60, receipt['checkpoint_sha256'], 'train')


def test_collect_reads_full_tracks_with_actual_builder_and_decoder():
    import torch
    from models.ct_v41.config import normalize_config
    from models.ct_v41.data import RawEndpointDataset
    from models.ctseqtrackv41 import CTSEQTRACKV41
    from tests.helpers.raw_tracking import TinySource

    class DenseSource(TinySource):
        def get_frames(self, index, frame_ids):
            rows = super().get_frames(index, frame_ids)
            for frame, index_in_track in zip(rows, frame_ids):
                center = frame['3d_bbox'][:3]
                frame['pc'] = np.tile(center, (16, 1))
                frame['pc'][:, 1] += np.linspace(-.1, .1, 16)
                frame['point_ids'] = np.arange(16)
                frame['raw_frame_id'] = index_in_track + 10
            return rows

    cfg = normalize_config(dict(ct_engineering_check=True, v41_attention_backend='math',
                               workers=0, batch_size=2, test=True,
                               checkpoint='epoch=010.ckpt'))
    dataset = RawEndpointDataset(DenseSource(lengths=(4, 3)))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(42)
            host = CTSEQTRACKV41(cfg).eval()
            records, visited = collect(host, dataset, cfg, 10)
    finally:
        torch.set_num_threads(threads)
    assert visited == len(dataset) == 5
    assert {r['tracklet'] for r in records} == {0, 1}
    assert any(r['frame'] > 1 for r in records)
    for record in records:
        assert record['target_points'] == 16
        assert max(int(k) for k in record['state']['boxes']) == record['frame'] - 1
