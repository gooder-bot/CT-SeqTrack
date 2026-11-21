"""联合搜索的稀疏 BEV 几何。在线构造与监督入口分开，坐标为传感器轴。"""
from dataclasses import dataclass

import numpy as np

from .cpu_geometry import compiled_backend


CELL = 1.0
GRID_FEATURES = 18
FINE_POINTS = 256


def _groups(xy):
    backend = compiled_backend()
    if backend is not None:
        x, y = xy.T
        xmin, ymin = int(x.min()), int(y.min())
        stride = int(y.max()) - ymin + 1
        capacity = (int(x.max()) - xmin + 1) * stride
        # 两张 int64 桶表合计最多 16 MB；稀疏大跨度时直接排序真实格键。
        if capacity <= min(1_000_000, max(4096, 16 * len(xy))):
            return backend.bounded_xy_groups(xy, xmin, ymin, stride, capacity)
    cells = xy - xy.min(axis=0)
    stride = int(cells[:, 1].max()) + 1
    keys = cells[:, 0] * stride + cells[:, 1]
    _, first, inverse, count = np.unique(keys, return_index=True, return_inverse=True,
                                        return_counts=True)
    return first, inverse, count


def _extrema(inverse, lows, highs, groups):
    backend = compiled_backend()
    if backend is not None:
        return backend.sequential_extrema(inverse, lows, highs, groups)
    lower = np.full(groups, np.inf, dtype=np.float32)
    upper = np.full(groups, -np.inf, dtype=np.float32)
    np.minimum.at(lower, inverse, lows)
    np.maximum.at(upper, inverse, highs)
    return lower, upper


@dataclass
class SparseScan:
    xyz: np.ndarray
    ids: np.ndarray
    cell_index: np.ndarray
    centers: np.ndarray
    features: np.ndarray


def make_sparse_scan(xyz, ids):
    points = np.asarray(xyz, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    if points.shape != (len(ids), 3) or ids.ndim != 1:
        raise ValueError("joint scan points must align with per-scan raw IDs")
    if not len(points):
        return None
    if (ids < 0).any() or not np.isfinite(points).all():
        raise ValueError("joint scan requires finite points and real nonnegative raw IDs")
    if not (ids[1:] > ids[:-1]).all():
        _, first = np.unique(ids, return_index=True)
        first.sort()
        points, ids = points[first], ids[first]
    xy = np.floor(points[:, :2] / CELL).astype(np.int64)
    first, inv, count = _groups(xy)
    def means(values):
        return (np.bincount(inv, weights=values) / count).astype(np.float32)
    centers = np.stack([means(axis) for axis in points.T], axis=-1)
    z = points[:, 2]
    low, high = _extrema(inv, z, z, len(count))
    xy_cell = xy[first]
    _, wide_inv, _ = _groups(np.floor_divide(xy_cell, 4))
    wide_count = np.bincount(wide_inv, weights=count)
    wide_xy = np.stack([np.bincount(wide_inv, weights=count * axis) / wide_count
                        for axis in centers[:, :2].T], axis=-1)
    wide_var = np.stack([np.bincount(wide_inv, weights=count * axis**2) / wide_count
                         for axis in centers[:, :2].T], axis=-1) - wide_xy**2
    wide_low, wide_high = _extrema(wide_inv, low, high, len(wide_count))
    dxy = centers[:, :2] / CELL - (xy_cell + .5)
    radius = np.linalg.norm(centers[:, :2], axis=-1)
    bearing = centers[:, :2] / np.maximum(radius[:, None], 1e-6)
    features = np.column_stack((dxy, centers[:, 2], low, high,
                                np.sqrt(np.maximum(means(z*z) - centers[:, 2]**2, 0)),
                                np.log1p(count), np.log1p(wide_count[wide_inv]),
                                centers[:, 2] - wide_low[wide_inv], wide_high[wide_inv] - centers[:, 2],
                                np.log1p(radius), np.log1p(high - low), bearing,
                                wide_xy[wide_inv] - centers[:, :2],
                                np.sqrt(np.maximum(wide_var[wide_inv], 0.)))).astype(np.float32)
    return SparseScan(points, ids, inv, centers, features)


def inside(points, pose, half_size):
    dx, dy, dz = (points - pose[:3]).T
    c, s = np.cos(pose[3]), np.sin(pose[3])
    return ((np.abs(c*dx + s*dy) <= half_size[0])
            & (np.abs(-s*dx + c*dy) <= half_size[1])
            & (np.abs(dz) <= half_size[2]))


def fine_crop(scan, sensor_pose, size):
    selected = np.flatnonzero(inside(scan.xyz, sensor_pose, .625 * size + 2.))
    if len(selected) > FINE_POINTS:
        selected = selected[np.linspace(0, len(selected)-1, FINE_POINTS, dtype=np.int64)]
    return scan.xyz[selected], scan.ids[selected]


def grid_labels(scan, target, target_size, other_cars):
    """仅由 loss 调用。有框内真实回波的体素监督类别及中心偏移。"""
    owner = np.full(len(scan.centers), -1, dtype=np.int64)
    weight = np.zeros(len(owner), dtype=np.int64)
    boxes = np.vstack((np.r_[target, target_size], np.asarray(other_cars).reshape(-1, 7)))
    # 平票按几何顺序归属，同一扫描换一辆车作查询也得到相同的中心偏移。
    for i in np.lexsort(boxes.T[::-1]):
        box = boxes[i]
        # 先按格心安全剔除远格；同格 XY 点与格心各轴至多相差 CELL。
        radius = np.hypot(box[4], box[5]) * .5 + CELL
        near = np.all(np.abs(scan.centers[:, :2] - box[:2]) <= radius, axis=-1)
        point_indices = np.flatnonzero(near[scan.cell_index])
        members = inside(scan.xyz[point_indices], box[:4], box[4:] * .5)
        cells, counts = np.unique(scan.cell_index[point_indices[members]], return_counts=True)
        wins = counts > weight[cells]
        owner[cells[wins]], weight[cells[wins]] = i, counts[wins]
    assigned = owner >= 0
    offsets = np.zeros_like(scan.centers)
    offsets[assigned] = boxes[owner[assigned], :3] - scan.centers[assigned]
    return assigned, owner == 0, offsets
