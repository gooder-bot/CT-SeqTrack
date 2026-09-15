"""v30：真实 B0 边界与预测终点的有界外部获取带。

实际 crop、最大可达范围和 9x9 标签共享同一矩形族。只有标签函数
接收当前 GT 前景；在线几何不接收标签，也不以 GT 决定获取范围。
"""

import copy
import math

import numpy as np

from utils.acquisition_v29 import (
    Z_CONTRACT, _box_geometry, _margins, acquisition_margin_grid_target_v29,
    apply_b0_vertical_hull, maximum_acquisition_supports_v29,
    support_membership, support_vertical_interval)
from utils.b1_acquisition import acquisition_axis, projected_box_half_extents


XY_CONTRACT = 'b0_boundary_endpoint_band_v1'
BAND_MIN = (.25, .25)
BAND_MAX = (4., 3.)
BAND_INITIAL = (.75, .5)
BAND_TIGHT_MAX = (2., 1.5)


def build_acquisition_supports_v30(
        *, b0_crop_box, endpoint_center, object_wlh, object_yaw,
        band_margins=BAND_INITIAL, support_yaw=None,
        band_margin_min=BAND_MIN, band_margin_max=BAND_MAX,
        b0_crop_scale=1.25, b0_crop_offset=2.0):
    """返回 endpoint、B0/endpoint 包络 tube、几何诊断（均世界坐标）。

    endpoint 的核心为初始尺寸框在运动轴上的投影；tube 的核心是该
    endpoint 与真实 B0 crop 投影区间的并包络。二者使用同一个 band。
    不在此函数扣除 B0 点；采样器仍按整个 B0 raw IDs 做差集。
    """
    if b0_crop_box is None:
        raise ValueError('v30 supports require the actual B0 crop anchor')
    anchor, crop_size, crop_rotation = _box_geometry(b0_crop_box)
    endpoint = np.asarray(endpoint_center, dtype=np.float64)
    if endpoint.shape != (3,) or not np.isfinite(endpoint).all():
        raise ValueError('v30 endpoint must contain three finite coordinates')
    scale, offset = float(b0_crop_scale), float(b0_crop_offset)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset) or offset < 0:
        raise ValueError('v30 B0 scale/offset must be finite and nonnegative')
    band = _margins(band_margins, 'band_margins')
    minimum = _margins(band_margin_min, 'band_margin_min')
    maximum = _margins(band_margin_max, 'band_margin_max')
    if np.any(maximum <= minimum) or np.any(band < minimum - 1e-6) or np.any(band > maximum + 1e-6):
        raise ValueError('v30 band must lie within registered min/max bounds')
    if support_yaw is None:
        axis, yaw, _ = acquisition_axis(endpoint[:2] - anchor[:2], object_yaw)
    else:
        yaw = float(support_yaw)
        if not math.isfinite(yaw):
            raise ValueError('v30 support yaw must be finite')
        axis = np.asarray((math.cos(yaw), math.sin(yaw)))
    rotation = np.asarray(((axis[0], -axis[1], 0.),
                           (axis[1], axis[0], 0.), (0., 0., 1.)))
    crop_half = .5 * crop_size[[1, 0, 2]] * scale + offset
    # abs(R_support^T R_crop) h gives exactly the projected corner extrema,
    # including a possible tilted crop; no width/length or world/local mixing.
    projected_crop_half = np.abs(rotation.T @ crop_rotation) @ crop_half
    object_half = projected_box_half_extents(object_wlh, object_yaw, yaw)
    endpoint_local = rotation.T @ (endpoint - anchor)
    lower = np.minimum(-projected_crop_half[:2], endpoint_local[:2] - object_half[:2])
    upper = np.maximum(projected_crop_half[:2], endpoint_local[:2] + object_half[:2])
    core_center_local = np.r_[.5 * (lower + upper), endpoint_local[2]]
    core_half = .5 * (upper - lower)

    def make_box(center, half_xy):
        box = copy.deepcopy(b0_crop_box)
        box.center = np.asarray(center, dtype=np.float64).copy()
        box.orientation = b0_crop_box.orientation.__class__(axis=[0, 0, 1], radians=yaw)
        box.wlh = np.asarray((2. * half_xy[1], 2. * half_xy[0], 2. * object_half[2]))
        return apply_b0_vertical_hull(box, b0_crop_box, scale=scale, offset=offset)

    endpoint_box = make_box(endpoint, object_half[:2] + band)
    tube_box = make_box(anchor + rotation @ core_center_local, core_half + band)
    diagnostic = dict(
        support_xy_contract=XY_CONTRACT, support_z_contract=Z_CONTRACT,
        acquisition_margin_parallel_perp=band.copy(),
        acquisition_direction_world_xy=axis.copy(),
        base_projected_length=2. * object_half[0],
        base_projected_width=2. * object_half[1],
        b0_projected_half_xy=projected_crop_half[:2].copy(),
        acquisition_core_lower_xy=lower.copy(), acquisition_core_upper_xy=upper.copy(),
        acquisition_support_half_xy=core_half + band,
        endpoint_center=endpoint.copy(), endpoint_support_center=endpoint_box.center.copy(),
        tube_support_center=tube_box.center.copy(),
        endpoint_support_z_interval=support_vertical_interval(endpoint_box),
        tube_support_z_interval=support_vertical_interval(tube_box))
    return endpoint_box, tube_box, diagnostic


def maximum_acquisition_supports_v30(
        endpoint_box, tube_box, corridor_box=None, *, actual_margins,
        b0_crop_box, margin_max=BAND_MAX, b0_crop_scale=1.25, b0_crop_offset=2.0):
    """固定实际 v30 中心/轴/core，只将 band 提升到注册上界。"""
    return maximum_acquisition_supports_v29(
        endpoint_box, tube_box, corridor_box, actual_margins=actual_margins,
        b0_crop_box=b0_crop_box, margin_max=margin_max,
        b0_crop_scale=b0_crop_scale, b0_crop_offset=b0_crop_offset)


def acquisition_margin_grid_target_v30(
        points_xyz, point_ids, target_mask, baseline_ids, *, endpoint_box,
        tube_box, corridor_box=None, actual_margins, b0_crop_box,
        b0_crop_scale=1.25, b0_crop_offset=2., margin_min=BAND_MIN,
        margin_max=BAND_MAX, coverage=.90, grid_size=9):
    """v30 band 标签；输入框必须来自 build_acquisition_supports_v30。

    复用同一严格边界/unique-ID 枚举内核。该内核只对固定中心框的半宽
    加 band，因此对 v30 的 B0/core 包络与实际 crop 完全一致。
    """
    result = acquisition_margin_grid_target_v29(
        points_xyz, point_ids, target_mask, baseline_ids,
        endpoint_box=endpoint_box, tube_box=tube_box, corridor_box=corridor_box,
        actual_margins=actual_margins, b0_crop_box=b0_crop_box,
        b0_crop_scale=b0_crop_scale, b0_crop_offset=b0_crop_offset,
        margin_min=margin_min, margin_max=margin_max, coverage=coverage, grid_size=grid_size)
    result.update(support_xy_contract=XY_CONTRACT,
                  demand=bool(result['max_reachable_target_count'] > 0))
    return result
