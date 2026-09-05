"""v27 不等长轨迹完整遍历和固定 B0 预算下的有序机制 ticks。"""
from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest

from tools.preflight_ct_v27 import inspect_scheduler_coverage
from utils.dual_stream import DualStreamLoader
from utils.recursive_state import OnlineRecursiveBatchSampler, stable_tracklet_partition


class FrameDataset:
    def __init__(self, lengths, v27=True):
        self.lengths = lengths
        self.keys = [f'train/{index}' for index in range(1000)
                     if stable_tracklet_partition(f'train/{index}') == 'train'][:len(lengths)]
        if v27:
            self.ct_scene_manifest = {'schema': 'ct_seqtrack.scene_protocol.v27'}

    def get_num_tracklets(self):
        return len(self.lengths)

    def get_num_frames_tracklet(self, index):
        return self.lengths[index]

    def get_tracklet_key(self, index):
        return self.keys[index]


def _sampler(lengths, slots=4, v27=True, candidates=1):
    dataset = SimpleNamespace(dataset=FrameDataset(lengths, v27), num_candidates=candidates)
    return OnlineRecursiveBatchSampler(dataset, slots=slots, candidate_views=candidates, seed=42)


@pytest.mark.parametrize('lengths,slots', [([11, 8, 6, 4, 3, 1], 4), ([8, 2], 16), ([100], 16)])
def test_v27_visits_every_eligible_endpoint_once_including_partial_and_single_slot_tails(lengths, slots):
    sampler = _sampler(lengths, slots)
    declared = len(sampler)
    batches = list(sampler)
    assert declared == len(batches) == max(sampler.slot_prediction_frames)
    visits = Counter((row[3], row[4]) for batch in batches for row in batch)
    expected = {(index, frame) for index, length in enumerate(lengths) for frame in range(1, length)}
    assert set(visits) == expected and set(visits.values()) == {1}
    assert sum(map(len, batches)) == sampler.prediction_frames
    assert all(1 <= len(batch) <= slots for batch in batches)
    assert any(len(batch) < slots for batch in batches)
    assert all(len({row[2] for row in batch}) == len(batch) for batch in batches)
    assert all(sum(bool(row[6]) for row in batch) <= 1 for batch in batches)


def test_v27_epoch_replay_preserves_slot_state_order_and_resets_each_tracklet_to_frame_one():
    sampler = _sampler([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], slots=3)
    sampler.set_epoch(7)
    first = list(sampler)
    assert sampler.epoch == 8
    sampler.set_epoch(7)
    assert first == list(sampler)
    sampler.set_epoch(8)
    next_epoch = list(sampler)
    for epoch, batches in ((7, first), (8, next_epoch)):
        per_tracklet = defaultdict(list)
        tracklet_slot = {}
        for index, batch in enumerate(batches):
            for row_epoch, batch_index, slot, tracklet, frame, candidate, _ in batch:
                assert row_epoch == epoch and batch_index == index and candidate == 0
                assert tracklet_slot.setdefault(tracklet, slot) == slot
                per_tracklet[tracklet].append(frame)
        for tracklet, frames in per_tracklet.items():
            assert frames == list(range(1, sampler.dataset.dataset.lengths[tracklet]))
    assert sum(map(len, first)) == sum(map(len, next_epoch)) == sampler.prediction_frames


def test_legacy_keeps_full_slot_batches_and_drops_unbalanced_tail():
    sampler = _sampler([11, 8, 6, 4, 3], slots=4, v27=False, candidates=4)
    batches = list(sampler)
    assert len(batches) == len(sampler) == min(sampler.slot_prediction_frames)
    assert all(len(batch) == 16 for batch in batches)
    assert sum(len(batch) // 4 for batch in batches) < sampler.prediction_frames
    with pytest.raises(ValueError, match='fewer tracklets'):
        _sampler([8, 2], slots=16, v27=False)


def test_preflight_actually_walks_coverage_and_restores_sampler_epoch():
    sampler = _sampler([11, 8, 6, 4, 3, 1], slots=4)
    sampler.set_epoch(5)
    result = inspect_scheduler_coverage(sampler)
    assert sampler.epoch == 5
    assert result['expected_endpoints'] == result['visited_endpoints'] == sampler.prediction_frames
    assert result['missing_endpoints'] == 0 and result['complete_tracklets'] == 5
    assert result['visited_batches'] == result['declared_batches'] == len(sampler)
    assert result['partial_slot_batches'] > 0 and result['minimum_active_slots'] == 1
    assert result['full_epoch_coverage']


@pytest.mark.parametrize('observation_steps,mechanism_steps', [(4, 0), (4, 2), (4, 4), (4, 9), (25, 99)])
def test_v4_tick_packing_keeps_observation_budget_and_each_mechanism_tick_in_order(observation_steps, mechanism_steps):
    mechanism = list(range(mechanism_steps)) if mechanism_steps else None
    loader = DualStreamLoader(list(range(observation_steps)), mechanism, schema='ct_seqtrack.train.v4',
                              isolate_mechanism_rng=True)
    rows = list(loader)
    assert len(loader) == len(rows) == observation_steps
    assert [row['observation'] for row in rows] == list(range(observation_steps))
    flattened = []
    for step, row in enumerate(rows):
        ticks = ((step + 1) * mechanism_steps // observation_steps
                 - step * mechanism_steps // observation_steps)
        value = row['mechanism']
        if ticks == 0:
            assert value is None
        elif ticks == 1:
            assert not isinstance(value, dict)
            flattened.append(value)
        else:
            assert set(value) == {'ct_mechanism_sequence'}
            assert len(value['ct_mechanism_sequence']) == ticks
            flattened.extend(value['ct_mechanism_sequence'])
    assert flattened == list(range(mechanism_steps))


def test_single_long_tracklet_runs_all_endpoints_inside_unchanged_observation_epoch():
    sampler = _sampler([100], slots=16)
    loader = DualStreamLoader(list(range(25)), sampler, schema='ct_seqtrack.train.v4', isolate_mechanism_rng=True)
    endpoints = []
    for item in loader:
        sequence = item['mechanism']['ct_mechanism_sequence']
        for tick in sequence:
            assert len(tick) == 1
            endpoints.append(tick[0][4])
    assert endpoints == list(range(1, 100))
    assert len(loader) == 25


def test_legacy_schema_still_rejects_mechanism_longer_than_observation():
    with pytest.raises(ValueError, match='may not exceed'):
        DualStreamLoader([0], [0, 1], schema='ct_seqtrack.train.v3')
