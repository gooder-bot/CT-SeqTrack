"""v36 原始点获取：prior 优先，重叠域只排已选 ID，固定 768 槽。"""
from __future__ import annotations

import numpy as np

from models.ct_v31.acquisition import (
    _rotation, _vector, points_in_support, spatial_coverage_indices,
)


def _axes(value):
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (2, 2) or not np.isfinite(result).all()
            or not np.allclose(result.T @ result, np.eye(2), atol=1e-6)
            or np.linalg.det(result) <= 0):
        raise ValueError('v36 axes must be finite right-handed orthonormal columns')
    return result


def build_dual_support(*, b0_box, box_size, prior_box, range_xy, axes,
                       crop_scale=1.25, crop_offset=2., b0_crop_half=None):
    """两个独立位置域；range 在事件轴内截断，仅限制获取、不截断运动。"""
    b0, prior = _vector(b0_box, 4, 'b0_box'), _vector(prior_box, 4, 'prior_box')
    size, extent = _vector(box_size, 3, 'box_size'), _vector(range_xy, 2, 'range_xy')
    basis = _axes(axes)
    if (size <= 0).any() or (extent < 0).any():
        raise ValueError('v36 size must be positive and position range nonnegative')
    crop_half = (size * .5 * float(crop_scale) + float(crop_offset)
                 if b0_crop_half is None else _vector(b0_crop_half, 3, 'b0_crop_half'))
    if not np.isfinite(crop_half).all() or (crop_half <= 0).any():
        raise ValueError('v36 actual B0 crop half widths must be finite positive')
    margin = np.clip(extent, .25, [12., 8.])
    projected_object = np.abs(basis.T @ _rotation(prior[3])) @ (size[:2] * .5)
    z_margin = max(0., float(crop_half[2] - size[2] * .5))
    return {
        'prior': dict(center=prior[:3].copy(),
                      half=np.r_[projected_object + margin, size[2] * .5 + z_margin],
                      yaw=float(np.arctan2(basis[1, 0], basis[0, 0]))),
        'local': dict(center=b0[:3].copy(), half=crop_half + [4., 3., 0.], yaw=float(b0[3])),
        'range_xy': extent.copy(), 'search_range_xy': margin,
        'range_clipped': extent > np.array([12., 8.]),
    }


def support_membership(points, support, b0_raw_ids, raw_ids):
    """返回可重叠的 prior/local 成员；仅原 B0 raw crop 是共同排除域。"""
    points = np.asarray(points, dtype=np.float64)
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) != len(ids):
        raise ValueError('v36 raw points/IDs must have matching point counts')
    eligible = (ids >= 0) & np.isfinite(points[:, :3]).all(1)
    eligible &= ~np.isin(ids, np.asarray(b0_raw_ids, dtype=np.int64).reshape(-1))
    positions = np.flatnonzero(eligible)
    _, unique = np.unique(ids[positions], return_index=True)
    keep = np.zeros(len(ids), dtype=bool)
    keep[positions[unique]] = True
    return (keep & points_in_support(points, support['prior']),
            keep & points_in_support(points, support['local']))


def acquire_extension(cloud_points, raw_ids, *, b0_raw_ids, anchor, b0_box,
                      box_size, prior_box, range_xy, axes, crop_scale=1.25,
                      crop_offset=2., b0_crop_half=None, seed=0):
    """prior 384 -> local 排已选ID后384 -> union余点借位；空槽不复制。"""
    points = np.asarray(cloud_points, dtype=np.float64)
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    support = build_dual_support(b0_box=b0_box, box_size=box_size,
        prior_box=prior_box, range_xy=range_xy, axes=axes, crop_scale=crop_scale,
        crop_offset=crop_offset, b0_crop_half=b0_crop_half)
    prior, local = support_membership(points, support, b0_raw_ids, ids)
    chosen = np.zeros(len(ids), dtype=bool)
    selected, partitions = [], []
    for part, membership in enumerate((prior, local)):
        candidates = np.flatnonzero(membership & ~chosen)
        picked = candidates[spatial_coverage_indices(points[candidates], ids[candidates], 384, seed)]
        chosen[picked] = True
        selected.extend(picked.tolist())
        partitions.extend([part] * len(picked))
    if len(selected) < 768:
        candidates = np.flatnonzero((prior | local) & ~chosen)
        extra = candidates[spatial_coverage_indices(points[candidates], ids[candidates],
                                                    768 - len(selected), seed)]
        selected.extend(extra.tolist())
        # 一侧未填满时其全部点已耗尽；余点只来自另一侧，来源不随借位改变。
        partitions.extend(np.where(prior[extra], 0, 1).tolist())
    out = dict(extension_points=np.zeros((768, 5), np.float32),
               extension_ids=np.full(768, -1, np.int64),
               extension_valid=np.zeros(768, bool),
               extension_partition=np.full(768, -1, np.int64),
               extension_membership=np.zeros((768, 2), bool))
    anchor_xyz = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if anchor_xyz.size not in (3, 4):
        raise ValueError('anchor must be world XYZ or world XYZ/yaw')
    anchor_xyz = _vector(anchor_xyz[:3], 3, 'anchor')
    count = len(selected)
    if count:
        selected = np.asarray(selected, np.int64)
        channels = min(points.shape[1], 5)
        out['extension_points'][:count, :channels] = np.nan_to_num(points[selected, :channels])
        # 大地图坐标必须先 double 相减。
        out['extension_points'][:count, :3] = points[selected, :3] - anchor_xyz
        out['extension_ids'][:count] = ids[selected]
        out['extension_valid'][:count] = True
        out['extension_partition'][:count] = partitions
        out['extension_membership'][:count] = np.stack((prior[selected], local[selected]), -1)
    return out
