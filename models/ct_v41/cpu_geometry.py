"""V41 CPU 裁点与四候选监督共享的后端选择和精度边界。"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def compiled_backend():
    try:
        import numba
    except ModuleNotFoundError as error:
        if error.name != 'numba':
            raise
        return None
    if numba.config.DISABLE_JIT:
        return None
    from . import _numba_geometry
    return _numba_geometry


def backend_identity():
    from importlib.metadata import PackageNotFoundError, version
    try:
        numba_version = version('numba')
    except PackageNotFoundError:
        numba_version = None
    return dict(backend='numba' if compiled_backend() is not None else 'numpy',
                numba=numba_version, numpy=np.__version__)


@lru_cache(maxsize=1)
def _metric_boundaries():
    from utils.tracking_metrics import _SUCCESS_THRESHOLDS, _PRECISION_THRESHOLDS
    def midpoints(thresholds, direction):
        neighbor = np.nextafter(thresholds, np.float32(direction))
        return (neighbor.astype(np.float64) + thresholds.astype(np.float64)) / 2
    # S/P 在 float32 舍入后比较；跳变点是相邻 float32 值的中点。
    return midpoints(_SUCCESS_THRESHOLDS[1:], -np.inf), midpoints(_PRECISION_THRESHOLDS, np.inf)


def _uncertain_slots(metrics):
    exact, compat, distance = metrics.transpose(2, 0, 1)
    success, precision = _metric_boundaries()
    tolerance = 1e-8
    return (~np.isfinite(metrics).all(-1) | (np.abs(exact - .5) <= tolerance)
            | (np.abs(compat[..., None] - success) <= tolerance).any(-1)
            | ((distance > 0) & (np.abs(distance[..., None] - precision) <= tolerance).any(-1)))


def _well_conditioned_rows(boxes, targets, sizes, target_sizes):
    # 大世界坐标与小 footprint 的面积差容易相消；这些行沿用标量参考。
    positions = np.maximum(np.abs(boxes).max((1, 2)), np.abs(targets).max(1))
    horizontal = np.maximum(np.abs(boxes[..., :2]).max((1, 2)), np.abs(targets[:, :2]).max(1))
    minimum_width = np.minimum(sizes[:, :2].min(1), target_sizes[:, :2].min(1))
    return ((positions <= 256.) & (horizontal <= 16 * minimum_width)
            & (sizes >= .05).all(1) & (target_sizes >= .05).all(1)
            & (sizes <= 16.).all(1) & (target_sizes <= 16.).all(1))


def training_box_metrics(boxes, targets, sizes, target_sizes, *, q0=None, quality_sizes=None,
                         explicit_target_size=True, reference=False):
    """返回 FP64 exact/compat/distance；旧 q0 quality 逐框使用原参考再舍入。"""
    from utils.tracking_metrics import BoxMetricsTarget, LocalYawBox

    batch_size, slots = boxes.shape[:2]
    measured = np.empty((batch_size, slots, 3), dtype=np.float64)
    needs_reference = np.ones((batch_size, slots), dtype=bool)
    shared = np.zeros(batch_size, dtype=bool)
    quality = None
    if q0 is not None:
        shared = (q0 == boxes[:, 0]).all(-1) & (quality_sizes == sizes).all(-1)
        quality = np.empty((batch_size, 1), np.float32)
    backend = None if reference else compiled_backend()
    if backend is not None and batch_size and slots:
        eligible = _well_conditioned_rows(boxes, targets, sizes, target_sizes)
        if eligible.any():
            rows = slice(None) if eligible.all() else eligible
            skip = np.zeros((int(eligible.sum()), slots), dtype=bool)
            skip[:, 0] = shared[rows]
            fast = backend.candidate_metrics(boxes[rows], targets[rows], sizes[rows], target_sizes[rows], skip)
            measured[rows] = fast
            needs_reference[rows] = skip | _uncertain_slots(fast)

    for row in range(batch_size):
        select = needs_reference[row]
        if quality is None and not select.any():
            continue
        truth = BoxMetricsTarget(LocalYawBox(targets[row], target_sizes[row][[1, 0, 2]]), up_axis=(0, 0, 1))
        if select.any():
            measured[row, select] = truth.local_yaw_metrics(boxes[row, select], sizes[row][[1, 0, 2]])
        if quality is not None:
            if shared[row]:
                quality[row, 0] = measured[row, 0, 0]
            else:
                quality_wlh = quality_sizes[row][[1, 0, 2]]
                if not explicit_target_size and not np.array_equal(quality_sizes[row], sizes[row]):
                    truth = BoxMetricsTarget(LocalYawBox(targets[row], quality_wlh), up_axis=(0, 0, 1))
                quality[row, 0] = truth.metrics_3d(LocalYawBox(q0[row], quality_wlh))[0]
    return measured, quality
