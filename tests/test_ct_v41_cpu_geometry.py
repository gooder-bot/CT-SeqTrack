"""V41 单遍裁点、训练几何离散标签和旧 q0 质量的数值边界。"""
import numpy as np
import pytest

from models.ct_v41 import cpu_geometry, search
from utils.tracking_metrics import metric_contributions


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_closed_crop_matches_numpy_for_strided_cloud_and_ulp_faces(dtype):
    rng = np.random.default_rng(419)
    backing = rng.uniform(-4, 4, (8192, 7)).astype(dtype)
    points = backing[::2, 1:7:2]
    size = np.array([4., 2., 1.])
    box = np.array([1., -2., .5, 0.])
    points[:3] = [[3., -1., 1.], [np.nextafter(dtype(3), dtype(4)), -1., 1.],
                  [3., -1., np.nextafter(dtype(1), dtype(2))]]
    for yaw in (0., .325, -np.pi / 2):
        box[3] = yaw
        for scale, offset in ((1., 0.), (1.25, 2.)):
            actual = search.inside_box(points, box, size, scale=scale, offset=offset)
            expected = search._inside_box_numpy(points, box, size, scale=scale, offset=offset)
            np.testing.assert_array_equal(actual, expected)
            if yaw == 0. and scale == 1.:
                assert actual[:3].tolist() == [True, False, False]
    assert search.inside_box(points[:0], box, size).size == 0


def test_general_crop_and_optional_backend_keep_numpy_contract(monkeypatch):
    points = np.arange(24.).reshape(2, 3, 4)
    box = np.zeros(4, np.float32)
    size = np.ones((2, 1, 3))
    np.testing.assert_array_equal(search.inside_box(points, box, size), search._inside_box_numpy(points, box, size))
    monkeypatch.setattr(search, 'compiled_backend', lambda: None)
    xyz, box, size = points[0, :, :3], box.astype(np.float64), size[0, 0]
    np.testing.assert_array_equal(search.inside_box(xyz, box, size), search._inside_box_numpy(xyz, box, size))


def test_fast_candidates_preserve_discrete_labels_and_reference_quality():
    rng = np.random.default_rng(506)
    gt = rng.uniform(-1, 1, size=(192, 4))
    boxes = gt[:, None] + rng.normal(0., .9, size=(len(gt), 4, 4))
    sizes = rng.uniform(.3, 5., size=(len(gt), 3))
    target_sizes = sizes * rng.uniform(.8, 1.2, sizes.shape)
    boxes[::3, 1] = boxes[::3, 0]
    q0 = boxes[:, 0].copy()
    q0[1::4] = gt[1::4]
    options = dict(q0=q0, quality_sizes=sizes)
    actual, quality = cpu_geometry.training_box_metrics(boxes, gt, sizes, target_sizes, **options)
    expected, old = cpu_geometry.training_box_metrics(boxes, gt, sizes, target_sizes, reference=True, **options)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(actual[..., 0] >= .5, expected[..., 0] >= .5)
    np.testing.assert_array_equal(metric_contributions(actual[..., 1], actual[..., 2]),
                                  metric_contributions(expected[..., 1], expected[..., 2]))
    np.testing.assert_array_equal(quality.view(np.uint32), old.view(np.uint32))


def test_precise_thresholds_and_far_small_boxes_use_reference(monkeypatch):
    backend = cpu_geometry.compiled_backend()
    if backend is None:
        pytest.skip('requires compiled geometry backend')
    # 大世界坐标/小框曾使无保护的 q0 浮点标签相差一 ULP。
    gt = np.array([[0., 0., 0., 0.], [219.6722412109375, 219.30055236816406, 0., 1.679668664932251]])
    boxes = np.repeat(gt[:, None, :], 4, axis=1)
    boxes[0, :, 0] = [1., np.nextafter(1., 0.), np.nextafter(1., 2.), .23]
    boxes[1, :, :] = [219.65589904785156, 219.31439208984375, -.0027908796910196543, 1.6530544757843018]
    sizes = np.array([[3., 2., 1.], [.13453465700149536, .051378265023231506, .1169297844171524]])
    class Recorder:
        def candidate_metrics(self, boxes, *args):
            assert len(boxes) == 1
            return backend.candidate_metrics(boxes, *args)
    monkeypatch.setattr(cpu_geometry, 'compiled_backend', lambda: Recorder())
    actual, quality = cpu_geometry.training_box_metrics(boxes, gt, sizes, sizes, q0=boxes[:, 0], quality_sizes=sizes)
    expected, old = cpu_geometry.training_box_metrics(boxes, gt, sizes, sizes, q0=boxes[:, 0],
        quality_sizes=sizes, reference=True)
    np.testing.assert_array_equal(actual[0, :3], expected[0, :3])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(quality.view(np.uint32), old.view(np.uint32))
    monkeypatch.setattr(cpu_geometry, 'compiled_backend', lambda: None)
    numpy_values, _ = cpu_geometry.training_box_metrics(boxes, gt, sizes, sizes)
    np.testing.assert_array_equal(numpy_values, expected)
