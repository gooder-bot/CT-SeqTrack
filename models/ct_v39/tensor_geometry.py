"""v39 设备端直立框几何；训练标签与兼容评测语义分开，无 CPU 往返。"""
from __future__ import annotations

import torch
from torch import Tensor


def cross2(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rotate_xy(points: Tensor, yaw: Tensor) -> Tensor:
    """yaw 的维数应已经广播到 points 的非坐标轴。"""
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c * points[..., 0] - s * points[..., 1],
                        s * points[..., 0] + c * points[..., 1]), -1)


def local_points(xyz: Tensor, boxes: Tensor) -> Tensor:
    delta = xyz[:, None] - boxes[..., None, :3]
    return torch.cat((rotate_xy(delta[..., :2], -boxes[..., None, 3]), delta[..., 2:3]), -1)


def inside_boxes(xyz: Tensor, boxes: Tensor, size: Tensor, scale: float = 1.,
                 offset: float = 0.) -> Tensor:
    return (local_points(xyz.float(), boxes.float()).abs()
            <= size.float()[:, None, None] * (0.5 * float(scale)) + float(offset)).all(-1)


def rectangle_corners(boxes: Tensor, size: Tensor) -> Tensor:
    signs = boxes.new_tensor(((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)))
    corners = signs * size[..., None, :2] * .5
    return rotate_xy(corners, boxes[..., None, 3]) + boxes[..., None, :2]


def _inside_polygon(points: Tensor, polygon: Tensor) -> Tensor:
    edges = polygon.roll(-1, -2) - polygon
    delta = points[..., :, None, :] - polygon[..., None, :, :]
    return (cross2(edges[..., None, :, :], delta) >= -1e-6).all(-1)


def rectangle_intersection_area(first: Tensor, second: Tensor) -> Tensor:
    """四角+16条边交点的固定张量求交，适用于任意前导批次轴。"""
    first_in = _inside_polygon(first, second)
    second_in = _inside_polygon(second, first)
    a = first[..., :, None, :]
    b = second[..., None, :, :]
    r = (first.roll(-1, -2) - first)[..., :, None, :]
    s = (second.roll(-1, -2) - second)[..., None, :, :]
    denominator = cross2(r, s)
    nonparallel = denominator.abs() > 1e-8
    safe = torch.where(nonparallel, denominator, torch.ones_like(denominator))
    delta = b - a
    t, u = cross2(delta, s) / safe, cross2(delta, r) / safe
    edge_valid = nonparallel & (t >= -1e-6) & (t <= 1. + 1e-6) & (u >= -1e-6) & (u <= 1. + 1e-6)
    intersections = (a + t[..., None] * r).flatten(-3, -2)
    points = torch.cat((first, second, intersections), -2)
    valid = torch.cat((first_in, second_in, edge_valid.flatten(-2)), -1)
    count = valid.sum(-1)
    centroid = (points * valid[..., None]).sum(-2) / count.clamp_min(1)[..., None]
    delta = points - centroid[..., None, :]
    angles = torch.atan2(delta[..., 1], delta[..., 0]).masked_fill(~valid, 10.)
    order = angles.argsort(dim=-1, stable=True)
    ordered = points.gather(-2, order[..., None].expand(*order.shape, 2))
    indices = torch.arange(points.shape[-2], device=points.device)
    next_indices = torch.where(indices + 1 < count[..., None], indices + 1, 0)
    following = ordered.gather(-2, next_indices[..., None].expand(*next_indices.shape, 2))
    areas = cross2(ordered, following) * (indices < count[..., None])
    return torch.where(count >= 3, areas.sum(-1).abs() * .5, 0.)


def metric_contributions(iou: Tensor, distance: Tensor) -> tuple[Tensor, Tensor]:
    """原 21 阈值、含端点梯形积分；不能替换成连续 IoU/距离。"""
    iou, distance = iou.float(), distance.float()
    s_axis = torch.linspace(0., 1., 21, device=iou.device, dtype=torch.float32)
    p_axis = torch.linspace(0., 2., 21, device=iou.device, dtype=torch.float32)
    success = (iou[..., None] >= s_axis).float()
    precision = (distance[..., None] <= p_axis).float()
    s = ((success[..., :-1] + success[..., 1:]) * .5 * s_axis.diff()).sum(-1)
    p = ((precision[..., :-1] + precision[..., 1:]) * .25 * p_axis.diff()).sum(-1)
    return s, p


@torch.no_grad()
def pair_box_metrics(boxes: Tensor, target: Tensor, size: Tensor,
                     target_size: Tensor | None = None) -> dict[str, Tensor]:
    """[B,C,4] 对 [B,4]；size 为 XYZ/LWH，返回全设备端 float32 标签。"""
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or target.shape != (boxes.shape[0], 4):
        raise ValueError('pair_box_metrics expects boxes[B,C,4], target[B,4]')
    target_size = size if target_size is None else target_size
    boxes, target, size, target_size = (x.detach().float() for x in (boxes, target, size, target_size))
    # 求交先平移到 GT 中心附近，避免大世界坐标的消减误差。
    origin = target[:, None, :3]
    estimates = torch.cat((boxes[..., :3] - origin, boxes[..., 3:]), -1)
    truths = torch.cat((target[:, None, :3] - origin, target[:, None, 3:]), -1).expand_as(estimates)
    sizes = size[:, None].expand(*boxes.shape[:2], 3)
    true_sizes = target_size[:, None].expand_as(sizes)
    area = rectangle_intersection_area(rectangle_corners(estimates, sizes),
                                       rectangle_corners(truths, true_sizes))
    area = area.clamp_min(0.).minimum(sizes[..., :2].prod(-1)).minimum(true_sizes[..., :2].prod(-1))
    exact_height = (torch.minimum(estimates[..., 2] + sizes[..., 2] * .5,
                                  truths[..., 2] + true_sizes[..., 2] * .5)
                    - torch.maximum(estimates[..., 2] - sizes[..., 2] * .5,
                                    truths[..., 2] - true_sizes[..., 2] * .5)).clamp_min(0.)
    # benchmark_compat 刻意保留原 z-h..z 区间，不能用 exact 高度代替。
    compat_height = (torch.minimum(estimates[..., 2], truths[..., 2])
                     - torch.maximum(estimates[..., 2] - sizes[..., 2],
                                     truths[..., 2] - true_sizes[..., 2])).clamp_min(0.)
    volume = sizes.prod(-1) + true_sizes.prod(-1)
    exact_intersection, compat_intersection = area * exact_height, area * compat_height
    exact = (exact_intersection / (volume - exact_intersection).clamp_min(1e-8)).clamp(0., 1.)
    compat = (compat_intersection / (volume - compat_intersection).clamp_min(1e-8)).clamp(0., 1.)
    distance = estimates[..., :3].norm(dim=-1)
    success, precision = metric_contributions(compat, distance)
    return dict(exact_iou=exact, benchmark_iou=compat, distance=distance,
                success=success, precision=precision)
