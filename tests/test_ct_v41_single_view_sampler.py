"""单视角主队列、初始化/点采样与四独立候选的逐输入对照。"""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.ct_v41.data import BatchBuilder, EndpointRequest, PERTURBED_BOX, build_loaders, stable_seed
from tests.helpers.raw_tracking import TinySource
from tests.test_ct_v41_history_pair import ExampleBank, arm


class DenseSource(TinySource):
    def get_frames(self, index, frame_ids):
        rows = super().get_frames(index, frame_ids)
        for frame in rows:
            # 超过 1,024 个 ROI 内物理点，使点采样 RNG 的差异可观测。
            x = np.linspace(-1.5, 1.5, 1536)
            offsets = np.column_stack((x, np.sin(5 * x) * .3, np.cos(x) * .2))
            frame['pc'] = frame['3d_bbox'][:3] + offsets
            frame['point_ids'] = np.arange(len(x))
        return rows


def test_single_view_visits_each_endpoint_once_and_keeps_eval_full_track():
    source = TinySource((11, 7), dt=.1)
    roles = {'train': source, 'test': source}
    loaders = build_loaders(arm('single_view_cgp'), tuple(roles), roles)
    baseline = build_loaders(arm('four_candidates'), tuple(roles), roles)
    sampler = loaders['train'].batch_sampler
    for epoch in (0, 9):
        sampler.set_epoch(epoch)
        rows = [row['request'] for batch in loaders['train'] for row in batch]
        assert all(r.branch == 0 and r.starts_window and r.ends_window for r in rows)
        assert Counter((r.tracklet, r.frame) for r in rows) == {
            (track, frame): 1 for track, length in enumerate(source.lengths) for frame in range(1, length)}
        manifest = sampler.state_dict()
        assert manifest['history_training'] == 'single_view_cgp'
        assert manifest['rows'] == 16 and manifest['batches'] == 6
    assert 'history_training' not in baseline['train'].batch_sampler.state_dict()
    assert loaders['test'].batch_sampler.state_dict() == baseline['test'].batch_sampler.state_dict()
    assert all(r.branch == 4 and r.window_start == 1 for batch in loaders['test'].batch_sampler for r in batch)


@pytest.mark.parametrize('frame', [1, 4])
def test_single_view_draw_matches_one_existing_candidate_including_its_crops(frame):
    source = DenseSource((12,), dt=.1)
    builders, datasets = [], []
    for mode in ('single_view_cgp', 'four_candidates'):
        config = arm(mode)
        datasets.append(build_loaders(config, ('train',), {'train': source})['train'].dataset)
        builder = BatchBuilder(config)
        builder.history_bank = ExampleBank()
        builder.detailed_diagnostics = False
        builders.append(builder)
    # 相同物理端点跨轮抽样，覆盖干净及三个扰动视角。首个预测始终由合法首框启动。
    observed = set()
    for epoch in range(32):
        view = stable_seed(42, epoch, 'tiny/0', frame, 'single_view') % 4
        if view in observed:
            continue
        request = EndpointRequest(epoch, 0, 0, frame, frame + 1, frame)
        raw = datasets[0][request]
        baseline = datasets[1][replace(request, branch=view)]
        single, one_of_four = [builder.prepare([row]) for builder, row in zip(builders, (raw, baseline))]
        assert single['branch_id'].item() == 0
        for field in ('points', 'point_ids', 'history_boxes', 'history_box_provenance', 'target_box',
                      'anchor_box', 'joint_motion_boxes', 'window_seed_offset', 'trusted_velocity'):
            torch.testing.assert_close(single[field], one_of_four[field], atol=0, rtol=0)
        assert bool(single['history_box_provenance'][0, -1] == PERTURBED_BOX) == (frame > 1 and view != 0)
        if frame == 1:
            assert not single['window_seed_offset'].any()
        observed.add(view)
        for builder in builders:
            builder.reset()
    assert observed == {0, 1, 2, 3}


def test_replayed_endpoint_uses_original_snapshot_and_branch_zero_point_sampling():
    source = DenseSource((8,), dt=.1)
    single = BatchBuilder(arm('single_view_cgp'))
    baseline = BatchBuilder(arm('four_candidates'))
    ds = build_loaders(single.config, ('train',), {'train': source})['train'].dataset
    epoch = next(e for e in range(2, 32) if stable_seed(42, e, 'tiny/0', 4, 'single_view') % 4 != 0)
    row = ds[EndpointRequest(epoch, 0, 0, 4, 5, 4)]
    snapshot = baseline._new_state(row)
    replay = dict(row, _joint_replay_state=snapshot)
    actual = single.prepare([replay])
    expected = baseline.prepare([replay])
    for field in ('history_boxes', 'history_box_provenance', 'point_ids', 'points', 'joint_motion_boxes'):
        torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
    assert not single.history_counts
