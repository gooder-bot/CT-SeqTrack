"""v31 端点预算、因果状态、raw 主进程准备与恢复的 CPU 合同。"""
from copy import deepcopy
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from models.ct_v31.data import (BatchBuilder, EndpointRequest, RawEndpointDataset,
    ReadyQueueBatchSampler, box_corners, build_loaders)
from models.ct_v31.runtime import (capture_rng_state, restore_rng_state, resume_payload,
    validate_resume_payload, TrackingEvaluation)


class TinySource:
    def __init__(self, lengths=(10, 6), dt=.5):
        self.lengths, self.dt = lengths, dt

    def get_num_tracklets(self):
        return len(self.lengths)

    def get_num_frames_tracklet(self, index):
        return self.lengths[index]

    def get_tracklet_key(self, index):
        return 'tiny/' + str(index)

    def get_frames(self, index, frame_ids):
        rows = []
        for i in frame_ids:
            center = np.asarray([i * .2, index * 20., 0.])
            cloud = np.asarray([[0., 0., 0.], [.2, .1, 0.], [-.2, -.1, 0.],
                                [0., -.2, .1], [5., 0., 0.], [8., 0., 0.]]) + center
            rows.append(dict(pc=cloud, point_ids=np.arange(len(cloud)),
                             **{'3d_bbox': np.r_[center, .1]}, box_size=np.asarray([4., 2., 2.]),
                             timestamp=100. + i * self.dt, scene_id='tiny_scene'))
        return rows

    def get_frame_metadata(self, index, frame):
        return self.get_frames(index, [frame])[0]


def cfg(**updates):
    return dict(ct_engineering_check=True, point_sample_size=8, workers=0,
                batch_size=4, v31_arm='full', **updates)


def request(frame, *, branch=3, start=1, end=9, track=0):
    return EndpointRequest(9, track, branch, start, end, frame)


def fake_prior(batch):
    return SimpleNamespace(box=batch['fallback_box'].clone(),
                           acquisition_fraction=torch.full((len(batch['points']), 2), .3))


def fake_output(batch, *, delta=0., quality=.9):
    accepted = batch['fallback_box'].clone()
    accepted[:, 0] += delta
    return SimpleNamespace(accepted_box=accepted,
        selected_quality=torch.full((len(accepted),), quality),
        selected_index=torch.zeros(len(accepted), dtype=torch.long),
        prior=fake_prior(batch),
        observation=SimpleNamespace(foreground_probability=torch.ones_like(batch['point_valid'], dtype=torch.float)),
        evidence=SimpleNamespace(identity_logits=torch.zeros_like(batch['extension_valid'], dtype=torch.float)))


@pytest.mark.parametrize('epoch', [0, 4, 9, 59])
def test_four_branch_full_coverage_and_causal_ready_queue(epoch):
    lengths = [1, 2, 4, 11, 19]
    sampler = ReadyQueueBatchSampler(lengths, 7)
    sampler.set_epoch(epoch)
    rows = [row for batch in sampler for row in batch]
    assert len(rows) == 4 * sum(n - 1 for n in lengths)
    assert len({(r.tracklet, r.branch, r.frame) for r in rows}) == len(rows)
    previous = {}
    for batch in sampler:
        assert len({r.state_key for r in batch}) == len(batch)
        for r in batch:
            assert r.frame == previous.get(r.state_key, r.window_start - 1) + 1
            previous[r.state_key] = r.frame
    assert len(list(sampler)) == len(sampler)
    assert sum(r.branch == 0 for r in rows) * 4 == len(rows)
    assert max(r.window_end - r.window_start for r in rows if r.branch == 3) <= 8
    assert sampler.state_dict() == deepcopy(sampler).state_dict()


def test_curriculum_lengths_and_offset_windows():
    sampler = ReadyQueueBatchSampler([30], 4)
    assert sampler.window_length(8) == 1
    sampler.set_epoch(9)
    rows = [r for batch in sampler for r in batch]
    assert sampler.window_length(8) == 8
    starts1 = {r.window_start for r in rows if r.branch == 1}
    starts2 = {r.window_start for r in rows if r.branch == 2}
    assert starts1 != starts2
    assert len(rows) == 116


def test_raw_dataset_and_loader_do_not_own_model_state():
    sources = {'train': TinySource(), 'test': TinySource()}
    loaders = build_loaders(cfg(), roles=('train', 'test'), sources=sources)
    train = list(loaders['train'])
    assert sum(map(len, train)) == 4 * (9 + 5)
    assert all('points' not in row and 'frames' in row for batch in train for row in batch)
    evaluation = [r for batch in loaders['test'] for r in batch]
    assert len(evaluation) == 14
    assert all(r['request'].branch == 4 and r['request'].window_start == 1 for r in evaluation)


def test_prepare_uses_only_past_seed_and_unique_sparse_points():
    raw = RawEndpointDataset(TinySource())
    builder = BatchBuilder(cfg())
    rows = [raw[request(1)]]
    batch = builder.prepare(rows)
    assert batch['points'].shape == (1, 4, 8, 5)
    assert batch['history_valid'].tolist() == [[False, False, True]]
    assert batch['history_pair_valid'].tolist() == [[False, False]]
    assert batch['point_valid'][0, -1].sum() == 4
    assert (batch['point_ids'][~batch['point_valid']] == -1).all()
    assert (batch['history_times'] < 0).all()
    assert torch.equal(batch['box_size'][0], torch.tensor([4., 2., 2.]))
    with pytest.raises(RuntimeError, match='not been committed'):
        builder.prepare(rows)
    builder.acquire(batch, fake_prior(batch))
    assert batch['extension_points'].shape == (1, 768, 5)
    assert batch['memory_points'].shape == (1, 36, 5)
    assert batch['acquisition_demand'].dtype == torch.bool
    assert set(batch['point_ids'][0, -1][batch['point_valid'][0, -1]].tolist()).isdisjoint(
        set(batch['extension_ids'][0][batch['extension_valid'][0]].tolist()))


def test_accepted_state_commit_is_unique_detached_and_no_gt_reset():
    raw = RawEndpointDataset(TinySource())
    builder = BatchBuilder(cfg())
    first = builder.prepare([raw[request(1)]])
    builder.acquire(first, fake_prior(first))
    output = fake_output(first, delta=6., quality=.1)
    output.accepted_box.requires_grad_()
    builder.commit(output, first)
    with pytest.raises(RuntimeError, match='duplicate commit'):
        builder.commit(output, first)
    rows = [raw[request(2)]]
    # 更改当前 GT 不允许改变 accepted 历史、crop 或 recovery。
    alternative = deepcopy(rows)
    alternative[0]['frames'][2]['3d_bbox'][:3] += 90
    left, right = deepcopy(builder), deepcopy(builder)
    a, b = left.prepare(rows), right.prepare(alternative)
    for key in ('points', 'point_valid', 'history_boxes', 'history_pair_valid', 'fallback_box', 'box_size'):
        assert torch.equal(a[key], b[key]), key
    assert a['anchor_box'][0, 0] == 6.
    assert not a['history_pair_valid'][0, -1]
    assert not a['history_boxes'].requires_grad
    assert not torch.equal(a['target_box'], b['target_box'])
    with pytest.raises(RuntimeError, match='missing accepted predecessor'):
        BatchBuilder(cfg()).prepare(rows)


def test_branch_states_are_independent_and_window_seeds_are_past():
    raw = RawEndpointDataset(TinySource())
    builder = BatchBuilder(cfg())
    a = raw[request(4, branch=1, start=4, end=7)]
    b = raw[request(4, branch=2, start=4, end=7)]
    batch = builder.prepare([a, b])
    assert len(builder.states) == 2
    assert set(next(iter(builder.states.values())).boxes) == {1, 2, 3}
    builder.acquire(batch, fake_prior(batch))
    out = fake_output(batch, quality=.1)
    out.accepted_box[0, 0] += 2
    builder.commit(out, batch)
    next_batch = builder.prepare([raw[request(5, branch=1, start=4, end=7)],
                                  raw[request(5, branch=2, start=4, end=7)]])
    assert next_batch['anchor_box'][0, 0] - next_batch['anchor_box'][1, 0] == pytest.approx(2.)


def test_bc_corner_channel_order_matches_observation():
    from models.ct_v31.observation import box_corners_xyz
    boxes = torch.tensor([[2., -1., .4, .7]])
    sizes = torch.tensor([[4., 2., 1.5]])
    expected = box_corners_xyz(boxes, sizes).numpy()[0]
    assert np.allclose(box_corners(boxes.numpy()[0], sizes.numpy()[0]), expected, atol=1e-6)


def test_rng_and_resume_require_complete_compatible_epoch():
    state = capture_rng_state()
    a = random.random(), np.random.random(), torch.rand(3)
    restore_rng_state(state)
    b = random.random(), np.random.random(), torch.rand(3)
    assert a[:2] == b[:2] and torch.equal(a[2], b[2])
    sampler = ReadyQueueBatchSampler([6], 4)
    payload = resume_payload(cfg(), completed_epoch=1, complete=True, rows=20, steps=5,
                             sampler=sampler.state_dict())
    assert validate_resume_payload(payload, cfg()) is payload
    with pytest.raises(ValueError, match='configuration identity'):
        validate_resume_payload(payload, cfg(v31_short_window=2))
    payload['epoch_complete'] = False
    with pytest.raises(ValueError, match='complete epoch boundaries'):
        validate_resume_payload(payload, cfg())


def test_evaluation_counts_first_frame_once():
    raw = RawEndpointDataset(TinySource(lengths=(3,)))
    builder, evaluation = BatchBuilder(cfg()), TrackingEvaluation()
    for frame in (1, 2):
        rows = [raw[request(frame, branch=4, end=3)]]
        batch = builder.prepare(rows)
        builder.acquire(batch, fake_prior(batch), training=False)
        output = fake_output(batch)
        output.accepted_box = batch['target_box'].clone()
        evaluation.add_batch(rows, batch, output)
        builder.commit(output, batch)
    summary = evaluation.summary()
    assert summary['frames'] == 3 and summary['prediction_frames'] == 2
    assert summary['success'] == pytest.approx(100.) and summary['precision'] == pytest.approx(100.)
    assert not builder.states


def test_missing_lightning_is_explicit_at_training_boundary():
    import models.ctseqtrackv31 as host
    if host.pl is not None:
        pytest.skip('Lightning is installed')
    model = host.CTSEQTRACKV31(cfg(), tracker=torch.nn.Linear(2, 1))
    with pytest.raises(RuntimeError, match='requires pytorch-lightning'):
        model.train_dataloader()
