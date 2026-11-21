"""实际数据入口的配对物理队列、独立初始化、梯度观测和训练身份。"""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v41.base_losses import local_observation_box_terms
from models.ct_v41.config import config_identity, load_config, normalize_config
from models.ct_v41.data import BatchBuilder, EndpointRequest, build_loaders
from models.ct_v41.history_training import HistoryTrainingAudit
from tests.helpers.raw_tracking import TinySource


class ExampleBank:
    def sample(self, key, ids, times, boxes, rng):
        delta = np.zeros((len(ids), 3))
        delta[:-1, 0] = rng.uniform(.03, .07)
        return delta, 'full_density'


def arm(mode):
    return normalize_config(dict(v41_history_training=mode, v41_joint_search=True,
        ct_engineering_check=True, v41_attention_backend='math', batch_size=3, workers=0,
        v32_reserve_windows=3))


def test_pair_uses_same_batches_and_drain_with_four_genuinely_independent_inputs():
    source = TinySource((11, 7), dt=.1)
    modes = ('window_1448', 'four_candidates')
    loaders = [build_loaders(arm(m), ('train',), {'train': source})['train'] for m in modes]
    for epoch in (0, 1, 2, 9):
        for loader in loaders:
            loader.batch_sampler.set_epoch(epoch)
        assert loaders[0].batch_sampler.state_dict() == loaders[1].batch_sampler.state_dict()
        starts = [0, 0]
        endpoints = Counter()
        for rows_a, rows_b in zip(*loaders):
            for a, b in zip(rows_a, rows_b):
                ra, rb = a['request'], b['request']
                assert (ra.tracklet, ra.branch, ra.frame, ra.drain) == (rb.tracklet, rb.branch, rb.frame, rb.drain)
                endpoints[rb.tracklet, rb.frame] += 1
                starts[0] += ra.starts_window
                starts[1] += rb.starts_window
                assert rb.starts_window and rb.ends_window and 'first_frame' in b
                assert set(range(max(0, rb.frame - 3), rb.frame + 1)) <= b['frames'].keys()
        assert set(endpoints.values()) == {4}
        if epoch < 2:
            assert starts[0] == starts[1]
        else:
            assert starts[0] < starts[1]

    builders = [BatchBuilder(arm(m)) for m in modes]
    for builder in builders:
        builder.history_bank = ExampleBank()
        builder.detailed_diagnostics = False
    # 同一窗口起点在 B 变成单步，末框、各历史框、重裁点与运动输入仍要一致。
    start = EndpointRequest(9, 0, 1, 4, 8, 4)
    batches = [b.prepare([loader.dataset[start]]) for b, loader in zip(builders, loaders)]
    for field in ('points', 'point_ids', 'anchor_box', 'history_boxes', 'joint_motion_boxes', 'trusted_velocity'):
        torch.testing.assert_close(batches[0][field], batches[1][field], rtol=0, atol=0)
    assert all(builder.history_counts['full_density'] == 1 for builder in builders)
    for builder in builders:
        builder.reset()
    continuation = EndpointRequest(9, 0, 1, 4, 8, 5)
    with pytest.raises(RuntimeError, match='predecessor'):
        builders[0].prepare([loaders[0].dataset[continuation]])
    builders[1].prepare([loaders[1].dataset[continuation]])
    assert builders[1].history_counts['full_density'] == 1


def test_history_audit_measures_weighted_center_output_gradient_including_background():
    valid = torch.zeros(4, 4, 3, dtype=torch.bool)
    valid[:3, :, :2] = True
    labels = torch.zeros_like(valid, dtype=torch.long)
    labels[0, -1, 0], labels[1, 0, 0] = 1, 1
    active = valid.flatten(1).any(1)
    boxes = torch.arange(16.).reshape(4, 4).div(9).requires_grad_()
    target = torch.zeros_like(boxes)
    anchor = torch.ones_like(boxes) * .6
    batch = dict(point_valid=valid, segmentation_labels=labels, target_box=target, anchor_box=anchor)
    output = SimpleNamespace(observation=SimpleNamespace(coarse_box=boxes, sequence_valid=active),
                             decoder=SimpleNamespace(q0_box=boxes))
    audit = HistoryTrainingAudit(3)
    audit.localization(batch, output)
    center_loss, _ = local_observation_box_terms(boxes, target, active, anchor[:, 3])
    actual = torch.autograd.grad(center_loss, boxes)[0][:, :3].norm(dim=1)
    torch.testing.assert_close(audit.gradient_sums[:, 2].float(), actual)
    assert audit.gradient_sums[:, 0].tolist() == [1.] * 4
    assert actual[2] > 0 and actual[3] == 0
    output.decoder = SimpleNamespace(q0_box=boxes + 20., lateral=SimpleNamespace(base_box=boxes))
    fused = HistoryTrainingAudit(11)
    fused.localization(batch, output)
    torch.testing.assert_close(fused.gradient_sums[:, 2].float(), actual)


def test_default_pair_identity_and_eval_do_not_need_training_artifacts(tmp_path):
    a, b = [load_config('cfgs/ct_seqtrack/41_history_' + name + '.yaml')
            for name in ('window_1448', 'four_candidates')]
    assert config_identity(a) != config_identity(b)
    assert a.eval_split == b.eval_split == 'dev'
    assert config_identity(dict(a, v41_history_bank='moved.json')) == config_identity(a)
    from models.ctseqtrackv41 import CTSEQTRACKV41
    host = CTSEQTRACKV41(dict(a, test=True, checkpoint='eval.ckpt', v41_history_bank=None,
                              v41_fixed_replay_bank=None), tracker=torch.nn.Linear(1, 1))
    assert host.joint_replay is None and host.evaluation_builder.history_bank is None
