"""候选监督的静态 GT 仅在一次训练事务中生效；不改变几何阈值。"""
import numpy as np
import torch

from models.ct_v41.input import HOST_GEOMETRY, loss_batch, online_batch, retained_geometry_sources
from models.ct_v41.runtime import move_tensors
from tests.test_ct_v41_recovery import values
from utils.tracking_metrics import BoxMetricsTarget, LocalYawBox


def test_only_loss_sees_unchanged_host_geometry():
    batch = values(b=2)
    batch['target_box_size'] = batch['box_size'].clone()
    assert HOST_GEOMETRY not in move_tensors(batch, 'cpu', retain_loss_geometry=True)
    assert HOST_GEOMETRY not in move_tensors(batch, 'meta')
    moved = move_tensors(batch, 'meta', retain_loss_geometry=True)
    for k, v in zip(('target_box', 'box_size', 'target_box_size'),
                    retained_geometry_sources(moved, moved['box_size'])):
        assert v is batch[k]
    assert retained_geometry_sources(moved, moved['box_size'].double()) is None
    batch[HOST_GEOMETRY] = moved[HOST_GEOMETRY]
    assert HOST_GEOMETRY not in online_batch(batch)
    assert loss_batch(batch)[HOST_GEOMETRY] is moved[HOST_GEOMETRY]
    moved['box_size'].add_(1)
    assert retained_geometry_sources(moved, moved['box_size']) is None
    batch.pop(HOST_GEOMETRY)
    moved = move_tensors(batch, 'meta', retain_loss_geometry=True)
    batch['target_box'].add_(1)
    assert retained_geometry_sources(moved, moved['box_size']) is None
    with torch.inference_mode():
        assert HOST_GEOMETRY not in move_tensors(batch, 'meta', retain_loss_geometry=True)


def test_identical_poses_share_geometry_but_nearby_boxes_keep_exact_gate(monkeypatch):
    gt = LocalYawBox(np.zeros(4), np.array([2., 3., 1.]))
    target = BoxMetricsTarget(gt, up_axis=(0, 0, 1))
    original = target.metrics_3d
    calls = []
    def observed(box):
        calls.append(box)
        return original(box)
    monkeypatch.setattr(target, 'metrics_3d', observed)
    poses = np.array([[1., 0., 0., 0.]] * 4)
    poses[1, 0] += 1e-10
    poses[2, 0] -= 1e-10
    metrics = target.local_yaw_metrics(poses, gt.wlh)
    assert len(calls) == 3
    assert metrics[0] == metrics[3]
    assert [exact >= .5 for exact, _, _ in metrics] == [True, False, True, True]
