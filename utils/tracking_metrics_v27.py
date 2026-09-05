"""v27 指标适配器。兼容分数用于训练/主表，正确几何仅用于复核。

geometry_exact 的 3D IoU 适用于跟踪协议中的直立有向框（yaw only）。
此模块不依赖 nuScenes、Shapely 或 torchmetrics，便于 CPU 合同测试。
"""
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


def box_metric_contributions(box, target, *, up_axis, mode="benchmark_compat", dim=3):
    return metric_contributions(*box_metrics(
        box, target, up_axis=up_axis, mode=mode, dim=dim), mode=mode)


def action_metric_gains(observation, action, target, *, up_axis,
                        mode="benchmark_compat", dim=3):
    oi, od = box_metrics(observation, target, up_axis=up_axis, mode=mode, dim=dim)
    ai, ad = box_metrics(action, target, up_axis=up_axis, mode=mode, dim=dim)
    os, op = metric_contributions(oi, od, mode)
    acs, acp = metric_contributions(ai, ad, mode)
    return {"success_gain": float(acs - os), "precision_gain": float(acp - op),
            "utility_gain": float((acs - os + acp - op) / 2),
            "center_gain": od - ad, "iou_gain": ai - oi,
            "observation_success": float(os), "observation_precision": float(op),
            "candidate_success": float(acs), "candidate_precision": float(acp)}


def batch_action_metric_gains(observations, actions, targets, *, up_axis,
                              mode="benchmark_compat", dim=3):
    if not (len(observations) == len(actions) == len(targets)):
        raise ValueError("action box batches must have equal lengths")
    rows = [action_metric_gains(o, a, g, up_axis=up_axis, mode=mode, dim=dim)
            for o, a, g in zip(observations, actions, targets)]
    return {key: np.asarray([r[key] for r in rows], dtype=np.float64)
            for key in (rows[0] if rows else ("success_gain", "precision_gain", "utility_gain"))}


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


def local_boxes_metric_gains(observation, action, target, prediction_wlh,
                             target_wlh, *, up_axis=(0, 0, 1),
                             mode="benchmark_compat", dim=3):
    if _axis(up_axis) != 2:
        raise ValueError("local [x,y,z,yaw] helper requires z-up")
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    observation, action, target, prediction_wlh, target_wlh = map(
        array, (observation, action, target, prediction_wlh, target_wlh))
    n = len(observation)
    if any(len(x) != n for x in (action, target, prediction_wlh, target_wlh)):
        raise ValueError("local metric batches must align")
    return batch_action_metric_gains(
        [LocalYawBox(b, s) for b, s in zip(observation, prediction_wlh)],
        [LocalYawBox(b, s) for b, s in zip(action, prediction_wlh)],
        [LocalYawBox(b, s) for b, s in zip(target, target_wlh)],
        up_axis=up_axis, mode=mode, dim=dim)
