"""跟踪指标：主评测兼容积分与定位质量监督使用的精确几何。"""

from __future__ import annotations

import numpy as np
import torch


METRIC_MODES = ("benchmark_compat", "geometry_exact")
_SUCCESS_THRESHOLDS = torch.linspace(0, 1, 21, dtype=torch.float32).numpy()
_PRECISION_THRESHOLDS = torch.linspace(0, 2, 21, dtype=torch.float32).numpy()
_SUCCESS_WIDTHS = np.diff(_SUCCESS_THRESHOLDS.astype(np.float64))
_PRECISION_WIDTHS = np.diff(_PRECISION_THRESHOLDS.astype(np.float64)) / 2


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
    s = (iou[..., None] >= s_axis).astype(np.float64)
    p = (distance[..., None] <= p_axis).astype(np.float64)
    return (((s[..., :-1] + s[..., 1:]) * .5 * _SUCCESS_WIDTHS).sum(-1),
            ((p[..., :-1] + p[..., 1:]) * .5 * _PRECISION_WIDTHS).sum(-1))


def _area(poly):
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return 0.0
    return abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                     - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))) / 2


def _oriented_polygon(poly):
    signed = np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(
        poly[:, 1], np.roll(poly[:, 0], -1))
    if abs(signed) / 2 <= 1e-14:
        return []
    return (poly if signed >= 0 else poly[::-1]).tolist()


def _clip_edges(polygon):
    return [(a[0], a[1], b[0] - a[0], b[1] - a[1])
            for a, b in zip(polygon, polygon[1:] + polygon[:1])]


def _clip_area(polygon, edges):
    if not edges:
        return 0.0
    for ax, ay, ex, ey in edges:
        if not polygon:
            return 0.0
        incoming, polygon = polygon, []
        px, py = incoming[-1]
        prev_inside = ex * (py - ay) - ey * (px - ax) >= -1e-12
        for x, y in incoming:
            inside = ex * (y - ay) - ey * (x - ax) >= -1e-12
            if inside != prev_inside:
                dx, dy = x - px, y - py
                denominator = dx * ey - dy * ex
                if abs(denominator) > 1e-14:
                    ratio = ((ax - px) * ey - (ay - py) * ex) / denominator
                    polygon.append((px + ratio * dx, py + ratio * dy))
            if inside:
                polygon.append((x, y))
            px, py, prev_inside = x, y, inside
    return _area(polygon)


def _box_geometry(box, plane, indices):
    center = np.asarray(box.center, dtype=np.float64)
    size = np.asarray(box.wlh, dtype=np.float64)
    if not np.isfinite(center).all() or not np.isfinite(size).all():
        raise ValueError("box geometry must be finite")
    if (size <= 0).any():
        raise ValueError("box dimensions must be positive")
    corners = np.asarray(box.corners(), dtype=np.float64)
    return center, size, corners[plane][:, indices].T


def _volume_overlap(area, volume, top, bottom, target_top, target_bottom):
    intersection = area * max(min(top, target_top) - max(bottom, target_bottom), 0.0)
    union = volume - intersection
    return float(np.clip(intersection / union if union > 0 else 0.0, 0, 1))


class BoxMetricsTarget:
    """一个端点的 GT；重复测量不同预测时共享裁剪边、体积和两种高度区间。"""
    def __init__(self, target, *, up_axis):
        self.axis = _axis(up_axis)
        self.plane = [i for i in range(3) if i != self.axis]
        self.indices = [2, 3, 7, 6] if self.axis == 2 else [0, 1, 5, 4]
        self.center, size, self.polygon = _box_geometry(target, self.plane, self.indices)
        self.edges = _clip_edges(_oriented_polygon(self.polygon))
        self.volume = np.prod(size)
        elevation, height = self.center[self.axis], size[2]
        self.exact_top, self.exact_bottom = elevation + height / 2, elevation - height / 2
        self.compat_top, self.compat_bottom = elevation, elevation - height

    def _prediction(self, box):
        center, size, polygon = _box_geometry(box, self.plane, self.indices)
        area = _clip_area(_oriented_polygon(polygon), self.edges)
        return center, size, polygon, area

    def metrics_3d(self, box):
        center, size, _, area = self._prediction(box)
        volume = np.prod(size) + self.volume
        elevation, height = center[self.axis], size[2]
        exact = _volume_overlap(area, volume, elevation + height / 2, elevation - height / 2,
                                self.exact_top, self.exact_bottom)
        compat = _volume_overlap(area, volume, elevation, elevation - height,
                                 self.compat_top, self.compat_bottom)
        return exact, compat, float(np.linalg.norm(center - self.center))

    def local_yaw_metrics(self, boxes, wlh):
        """同尺寸局部候选保留全部槽；只有位形字节完全相同才共享同一几何结果。"""
        measured = {}
        results = []
        for pose in np.asarray(boxes):
            key = pose.tobytes()
            if key not in measured:
                measured[key] = self.metrics_3d(LocalYawBox(pose, wlh))
            results.append(measured[key])
        return results


def paired_box_metrics_3d(box, target, *, up_axis):
    """同一对框的 exact IoU、compat IoU 和三维中心距。"""
    return BoxMetricsTarget(target, up_axis=up_axis).metrics_3d(box)


def box_metrics(box, target, *, up_axis, mode="benchmark_compat", dim=3):
    """所有调用显式给 up_axis；compat 故意保留历史高度/二维轴偏差。"""
    _mode(mode)
    axis = _axis(up_axis)
    if dim not in (2, 3):
        raise ValueError("metric dim must be 2 or 3")
    plane = [i for i in range(3) if i != axis]
    indices = [2, 3, 7, 6] if axis == 2 else [0, 1, 5, 4]
    center, size, polygon = _box_geometry(box, plane, indices)
    truth, target_size, target_polygon = _box_geometry(target, plane, indices)
    intersection_area = _clip_area(_oriented_polygon(polygon), _clip_edges(_oriented_polygon(target_polygon)))
    if dim == 3:
        volume = np.prod(size) + np.prod(target_size)
        top, target_top = center[axis], truth[axis]
        if mode == "geometry_exact":
            bottom, target_bottom = top - size[2] / 2, target_top - target_size[2] / 2
            top, target_top = top + size[2] / 2, target_top + target_size[2] / 2
        else:
            bottom, target_bottom = top - size[2], target_top - target_size[2]
        overlap = _volume_overlap(intersection_area, volume, top, bottom, target_top, target_bottom)
        return overlap, float(np.linalg.norm(center - truth))
    union = _area(polygon) + _area(target_polygon) - intersection_area
    overlap = intersection_area / union if union > 0 else 0.0
    selected = [axis] if mode == "benchmark_compat" else plane
    distance = np.linalg.norm((center - truth)[selected])
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
