"""跟踪指标：主评测兼容积分与定位质量监督使用的精确几何。"""

from __future__ import annotations

import numpy as np
import torch


METRIC_MODES = ("benchmark_compat", "geometry_exact")
_SUCCESS_THRESHOLDS = torch.linspace(0, 1, 21, dtype=torch.float32).numpy()
_PRECISION_THRESHOLDS = torch.linspace(0, 2, 21, dtype=torch.float32).numpy()


def _mode(mode):
    if mode not in METRIC_MODES:
        raise ValueError(f"unknown metric mode: {mode}")


def _axis(up_axis):
    axis = np.flatnonzero(np.asarray(up_axis) != 0)
    if len(axis) != 1 or axis[0] not in (1, 2):
        raise ValueError("up_axis must select y or z")
    return int(axis[0])


def metric_contributions(iou, distance, mode="benchmark_compat"):
    """精确复现 21 个阈值、含端点的梯形积分，返回 0..1 的 (S,P)。"""
    _mode(mode)
    iou, distance = np.broadcast_arrays(np.asarray(iou, dtype=np.float32),
                                        np.asarray(distance, dtype=np.float32))
    if not np.isfinite(iou).all() or not np.isfinite(distance).all():
        raise ValueError("metric inputs must be finite")
    # Match torch.linspace(..., dtype=float32), including threshold rounding.
    s_axis = _SUCCESS_THRESHOLDS
    p_axis = _PRECISION_THRESHOLDS
    sw = np.diff(s_axis.astype(np.float64))
    pw = np.diff(p_axis.astype(np.float64)) / 2
    s = (iou[..., None] >= s_axis).astype(np.float64)
    p = (distance[..., None] <= p_axis).astype(np.float64)
    return (((s[..., :-1] + s[..., 1:]) * .5 * sw).sum(-1),
            ((p[..., :-1] + p[..., 1:]) * .5 * pw).sum(-1))


def _area(poly):
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return 0.0
    return abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                     - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))) / 2


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _ccw(poly):
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    signed = np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(
        poly[:, 1], np.roll(poly[:, 0], -1))
    return poly if signed >= 0 else poly[::-1]


def _intersection(first, second):
    if _area(first) <= 1e-14 or _area(second) <= 1e-14:
        return 0.0
    out = list(_ccw(first))
    clip = _ccw(second)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        inp, out = out, []
        if not inp:
            return 0.0
        previous = inp[-1]
        prev_inside = _cross(b - a, previous - a) >= -1e-12
        for current in inp:
            inside = _cross(b - a, current - a) >= -1e-12
            if inside != prev_inside:
                direction = current - previous
                denominator = _cross(direction, b - a)
                if abs(denominator) > 1e-14:
                    ratio = _cross(a - previous, b - a) / denominator
                    out.append(previous + ratio * direction)
            if inside:
                out.append(current)
            previous, prev_inside = current, inside
    return _area(out)


def box_metrics(box, target, *, up_axis, mode="benchmark_compat", dim=3):
    """所有调用显式给 up_axis；compat 故意保留历史高度/二维轴偏差。"""
    _mode(mode)
    axis = _axis(up_axis)
    if dim not in (2, 3):
        raise ValueError("metric dim must be 2 or 3")
    plane = [i for i in range(3) if i != axis]
    centers = [np.asarray(x.center, dtype=np.float64) for x in (box, target)]
    sizes = [np.asarray(x.wlh, dtype=np.float64) for x in (box, target)]
    if any(not np.isfinite(x).all() for x in centers + sizes):
        raise ValueError("box geometry must be finite")
    if any((x <= 0).any() for x in sizes):
        raise ValueError("box dimensions must be positive")
    polygons = []
    for item in (box, target):
        corners = np.asarray(item.corners(), dtype=np.float64)
        indices = [2, 3, 7, 6] if axis == 2 else [0, 1, 5, 4]
        polygons.append(corners[plane][:, indices].T)
    intersection_area = _intersection(*polygons)
    if dim == 2:
        union = _area(polygons[0]) + _area(polygons[1]) - intersection_area
        overlap = intersection_area / union if union > 0 else 0.0
        selected = [axis] if mode == "benchmark_compat" else plane
        distance = np.linalg.norm((centers[0] - centers[1])[selected])
    else:
        if mode == "benchmark_compat":
            upper = min(c[axis] for c in centers)
            lower = max(c[axis] - s[2] for c, s in zip(centers, sizes))
        else:
            upper = min(c[axis] + s[2] / 2 for c, s in zip(centers, sizes))
            lower = max(c[axis] - s[2] / 2 for c, s in zip(centers, sizes))
        intersection = intersection_area * max(upper - lower, 0.0)
        union = np.prod(sizes[0]) + np.prod(sizes[1]) - intersection
        overlap = intersection / union if union > 0 else 0.0
        distance = np.linalg.norm(centers[0] - centers[1])
    return float(np.clip(overlap, 0, 1)), float(distance)


class LocalYawBox:
    """局部 nuScenes z-up [x,y,z,yaw] 与各自真实 wlh 的轻量包装。"""
    def __init__(self, local, wlh):
        self.center = np.asarray(local[:3], dtype=np.float64)
        self.wlh = np.asarray(wlh, dtype=np.float64)
        self.yaw = float(local[3])

    def corners(self):
        w, l, h = self.wlh
        corners = np.asarray([
            l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1]),
            w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1]),
            h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return rotation @ corners + self.center[:, None]

    def bottom_corners(self):
        return self.corners()[:, [2, 3, 7, 6]]
