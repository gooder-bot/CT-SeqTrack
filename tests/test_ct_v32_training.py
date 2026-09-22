"""v32 实际 mini 预算、连贯种子扰动、尾部 BN 和 worker 独立随机流。"""

from copy import deepcopy
from dataclasses import replace
from collections import Counter

import numpy as np
import pytest
import torch

from models.ct_v31.data import (BatchBuilder, RawEndpointDataset, ReadyQueueBatchSampler,
    KNOWN_GT_BOX, PERTURBED_BOX, PREDICTED_BOX, build_loaders)
from utils.bn_policy import running_batch_norm
from tests.test_ct_v31_runtime import TinySource, cfg, request, fake_prior, fake_output


# 已完成 mini 正式 B0 epoch060 checkpoint 的 sampler.lengths，274 轨迹；
# 固定真实预算回归夹具，不在测试时读取大权重或依赖本地 output 目录。
MINI_TRAIN_LENGTHS = (
    37,3,24,39,5,39,32,39,13,19,35,36,26,1,34,32,17,23,21,31,41,41,19,17,10,
    27,26,41,41,29,41,16,41,27,41,25,28,30,23,41,11,41,11,38,37,25,10,9,18,16,
    26,23,9,21,19,12,9,5,7,41,9,32,12,12,15,2,14,13,6,10,30,16,5,19,6,21,22,
    3,13,27,4,9,14,2,26,8,22,29,15,14,17,7,36,15,24,24,39,34,10,24,12,10,36,
    8,28,14,13,27,4,10,7,32,19,6,7,21,26,10,23,4,17,22,13,13,10,13,9,21,11,
    4,11,14,11,10,4,13,8,15,41,11,41,41,10,13,10,41,14,5,2,11,8,13,13,4,40,
    9,39,19,23,39,2,12,40,10,10,8,15,15,11,1,10,1,31,17,40,8,1,11,4,10,21,
    4,12,7,18,7,18,10,14,14,13,17,7,7,13,9,18,3,19,8,12,17,8,21,13,15,6,6,
    5,23,15,41,21,11,41,9,15,8,6,22,17,5,13,10,7,2,16,14,6,14,8,17,1,14,5,
    34,30,40,29,23,6,18,36,40,27,1,35,26,27,19,23,21,5,23,34,40,10,24,36,15,
    15,12,19,40,31,31,40,40,7,34,4,16,27,11,
)


@pytest.mark.parametrize('seed', [42, 52])
@pytest.mark.parametrize('epoch', [0, 9, 59])
def test_real_mini_reserve_budget_causality_and_rebuild(seed, epoch):
    sampler = ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS, 16, seed=seed)
    sampler.set_epoch(epoch)
    normal, reserve = sampler._queues()
    assert len(reserve) == 112
    assert Counter(row[1] for row in reserve) == {0: 28, 1: 28, 2: 28, 3: 28}
    assert all(end - start == 1 for _, _, start, end in reserve)
    batches = list(sampler)
    assert len(batches) == 1195
    assert list(map(len, batches)) == [16] * 1194 + [4]
    seen, previous, drain = set(), {}, False
    for batch in batches:
        assert len({r.state_key for r in batch}) == len(batch)
        assert len({r.drain for r in batch}) == 1
        assert not drain or batch[0].drain
        drain = batch[0].drain
        for row in batch:
            key = row.tracklet, row.branch, row.frame
            assert key not in seen
            seen.add(key)
            assert row.frame == previous.get(row.state_key, row.window_start - 1) + 1
            previous[row.state_key] = row.frame
    assert seen == {(t, b, f) for t, n in enumerate(MINI_TRAIN_LENGTHS)
                    for b in range(4) for f in range(1, n)}
    assert len(seen) == 19108 and drain
    rebuilt = ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS, 16, seed=seed)
    rebuilt.set_epoch(epoch)
    assert list(rebuilt) == batches
    assert rebuilt.state_dict() == sampler.state_dict()
    assert sampler.state_dict()['batches'] == 1195


def test_sampler_identity_includes_seed_and_drain_policies():
    original = ReadyQueueBatchSampler([30, 15], 16).state_dict()
    for changes in ({'reserve_windows': 0}, {'seed_translation': .2}, {'seed_yaw_degrees': 2.}):
        changed = ReadyQueueBatchSampler([30, 15], 16, **changes).state_dict()
        assert changed['manifest_sha256'] != original['manifest_sha256']
    assert original['schema'] == 'ct_seqtrack.v32.ready_queue.v2'
    assert original['drain_policy'] and original['seed_policy']


def test_coherent_seed_perturbation_preserves_physics_and_first_memory():
    raw = RawEndpointDataset(TinySource())
    canonical_row = raw[request(4, branch=0, start=4, end=7)]
    augmented_row = raw[request(4, branch=2, start=4, end=7)]
    snapshot = deepcopy(augmented_row)
    canonical, augmented = BatchBuilder(cfg()), BatchBuilder(cfg())
    clean, noisy = canonical.prepare([canonical_row]), augmented.prepare([augmented_row])
    left, right = next(iter(canonical.states.values())), next(iter(augmented.states.values()))
    for frame in left.boxes:
        np.testing.assert_allclose(right.boxes[frame][:3] - left.boxes[frame][:3], right.seed_offset[:3])
        delta = right.boxes[frame][3] - left.boxes[frame][3]
        assert np.arctan2(np.sin(delta), np.cos(delta)) == pytest.approx(right.seed_offset[3])
        np.testing.assert_array_equal(augmented_row['frames'][frame]['3d_bbox'],
                                      snapshot['frames'][frame]['3d_bbox'])
    yaw = left.boxes[3][3]
    rotation = np.asarray([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    assert (np.abs(rotation @ right.seed_offset[:2]) <= .3).all()
    assert abs(right.seed_offset[3]) <= np.deg2rad(1.5)
    assert right.seed_offset[2] == 0
    np.testing.assert_allclose(left.trusted_velocity, right.trusted_velocity, atol=1e-12)
    assert torch.equal(clean['physical_displacement'], noisy['physical_displacement'])
    assert torch.equal(clean['current_dt'], noisy['current_dt'])
    for key, value in left.memory.export(102.).items():
        np.testing.assert_array_equal(value, right.memory.export(102.)[key])
    assert noisy['history_box_provenance'].tolist() == [[PERTURBED_BOX] * 3]
    assert clean['history_box_provenance'].tolist() == [[KNOWN_GT_BOX] * 3]
    repeated = BatchBuilder(cfg()).prepare([raw[request(4, branch=2, start=4, end=7)]])
    assert torch.equal(noisy['window_seed_offset'], repeated['window_seed_offset'])


def test_known_perturbed_and_accepted_hints_have_explicit_provenance():
    raw = RawEndpointDataset(TinySource())
    for branch, expected in ((0, {0., 1.}), (1, {.2, .8}), (4, {0., 1.})):
        builder = BatchBuilder(cfg())
        batch = builder.prepare([raw[request(4, branch=branch, start=4, end=7)]])
        hints = batch['points'][0, :3, :, 4][batch['point_valid'][0, :3]]
        assert all(any(abs(float(value) - v) < 1e-6 for v in expected) for value in hints)
        assert (batch['points'][0, -1, :, 4][batch['point_valid'][0, -1]] == .5).all()
        builder.acquire(batch, fake_prior(batch))
        output = fake_output(batch, delta=2., quality=.1)
        builder.commit(output, batch)
        following = builder.prepare([raw[request(5, branch=branch, start=4, end=7)]])
        assert following['history_box_provenance'][0, -1] == PREDICTED_BOX
        hints = following['points'][0, -2, :, 4][following['point_valid'][0, -2]]
        assert all(abs(float(value) - .2) < 1e-6 or abs(float(value) - .8) < 1e-6 for value in hints)
        expected_anchor = batch['anchor_box'][0, :3] + output.accepted_box[0, :3]
        torch.testing.assert_close(following['anchor_box'][0, :3], expected_anchor)


def test_running_bn_keeps_affine_input_gradients_and_restores_on_error():
    module = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.2)).train()
    bn = module[0]
    old = deepcopy(bn.state_dict())
    values = torch.tensor([[2., 3., 4.]], requires_grad=True)
    with running_batch_norm(module):
        assert not bn.training and module[1].training
        output = bn(values)
        output.square().sum().backward()
    assert bn.training
    assert values.grad.abs().sum() > 0 and bn.weight.grad.abs().sum() > 0 and bn.bias.grad.abs().sum() > 0
    for key in ('running_mean', 'running_var', 'num_batches_tracked'):
        assert torch.equal(bn.state_dict()[key], old[key])
    with pytest.raises(RuntimeError, match='synthetic'):
        with running_batch_norm(module):
            raise RuntimeError('synthetic forward failure')
    assert bn.training and module[1].training


class BNProbeTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.observation = torch.nn.Sequential(torch.nn.BatchNorm1d(3))
        self.other = torch.nn.Dropout(.1)
        self.observed_modes = None

    def plan_prior(self, batch):
        return fake_prior(batch)

    def forward(self, batch, prior=None):
        self.observed_modes = self.observation[0].training, self.other.training
        values = self.observation(batch['points'][:, -1, :, :3].mean(1))
        output = fake_output(batch)
        output.accepted_box = output.accepted_box + torch.cat((values, values[:, :1] * 0), -1)
        return output


@pytest.mark.parametrize('count,drain,training_bn', [(4, False, True), (4, True, False), (1, False, False)])
def test_host_scopes_running_bn_to_observation(count, drain, training_bn):
    from models.ctseqtrackv31 import CTSEQTRACKV31
    model = CTSEQTRACKV31(cfg(batch_size=4, v31_arm='b0'), tracker=BNProbeTracker()).train()
    raw = RawEndpointDataset(TinySource((3,) * count))
    rows = [raw[replace(request(1, branch=0, end=3, track=t), drain=drain)] for t in range(count)]
    batch, output = model._forward_raw(rows, model.train_builder, training=True)
    output.accepted_box.sum().backward()
    assert model.tracker.observed_modes == (training_bn, True)
    bn = model.tracker.observation[0]
    assert bn.training and int(bn.num_batches_tracked) == int(training_bn)
    assert bn.bias.grad.abs().sum() > 0


def test_workers_do_not_change_requests_perturbations_or_accepted_input():
    sequences = []
    for workers in (0, 2):
        config = cfg(v31_arm='b0', workers=workers)
        loader = build_loaders(config, roles=('train',), sources={'train': TinySource((6, 4))})['train']
        loader.batch_sampler.set_epoch(9)
        builder, batches = BatchBuilder(config), []
        for rows in loader:
            batch = builder.prepare(rows)
            batches.append(([row['request'] for row in rows],
                            {key: value.clone() for key, value in batch.items()}))
            builder.acquire(batch, fake_prior(batch), training=False)
            builder.commit(fake_output(batch, delta=.1), batch)
        assert not builder.states
        sequences.append(batches)
    assert len(sequences[0]) == len(sequences[1])
    for (left_rows, left), (right_rows, right) in zip(*sequences):
        assert left_rows == right_rows
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key]), key
