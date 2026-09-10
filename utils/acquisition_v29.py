"""v29 共享获取几何：B0 垂直包络与只读原始 ID 可达性监督。"""

import copy
import math

import numpy as np


Z_CONTRACT = 'b0_vertical_hull_v1'


def _box_geometry(box):
    center = np.asarray(box.center, dtype=np.float64)
    size = np.asarray(box.wlh, dtype=np.float64)
    rotation = np.asarray(box.rotation_matrix, dtype=np.float64)
    if (center.shape != (3,) or size.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(center).all() or not np.isfinite(size).all()
            or not np.isfinite(rotation).all() or np.any(size <= 0)):
        raise ValueError('v29 support boxes require finite centers/rotations and positive wlh')
    return center, size, rotation


def b0_vertical_interval(b0_crop_box, *, scale=1.25, offset=2.0):
    """实际 B0 有向 crop 的世界 Z 包络；不读取点密度或当前 GT。"""
    if b0_crop_box is None:
        raise ValueError('v29 Z hull requires the actual B0 crop anchor')
    center, size, rotation = _box_geometry(b0_crop_box)
    scale, offset = float(scale), float(offset)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset) or offset < 0:
        raise ValueError('v29 B0 crop scale/offset must be finite and nonnegative')
    local_half = .5 * size[[1, 0, 2]] * scale + offset
    world_half_z = float(np.abs(rotation[2]) @ local_half)
    return np.asarray((center[2] - world_half_z, center[2] + world_half_z))


def support_vertical_interval(box):
    center, size, rotation = _box_geometry(box)
    # 所有正式 endpoint/tube/corridor 都是直立框。不能用修改高度暗改 XY。
    if not np.allclose(rotation[2], (0., 0., 1.), atol=1e-6, rtol=0):
        raise ValueError('v29 acquisition supports must be upright before applying the Z hull')
    return np.asarray((center[2] - .5 * size[2], center[2] + .5 * size[2]))


def apply_b0_vertical_hull(box, b0_crop_box, *, scale=1.25, offset=2.0):
    """复制 support，只改 Z center/height；原 XY、yaw 和输入对象保持不变。"""
    if box is None:
        return None
    actual = support_vertical_interval(box)
    baseline = b0_vertical_interval(b0_crop_box, scale=scale, offset=offset)
    lower, upper = min(actual[0], baseline[0]), max(actual[1], baseline[1])
    result = copy.deepcopy(box)
    result.center = np.asarray(box.center, dtype=np.float64).copy()
    result.wlh = np.asarray(box.wlh, dtype=np.float64).copy()
    result.center[2] = .5 * (lower + upper)
    result.wlh[2] = upper - lower
    return result


def _margins(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError(f'{name} must contain positive finite parallel/perpendicular margins')
    return value


def _with_margins(box, actual_margins, new_margins):
    if box is None:
        return None
    result = copy.deepcopy(box)
    result.wlh = np.asarray(box.wlh, dtype=np.float64).copy()
    result.wlh[[1, 0]] += 2. * (new_margins - actual_margins)
    _box_geometry(result)
    return result


def maximum_acquisition_supports_v29(
        endpoint_box, tube_box, corridor_box, *, actual_margins,
        b0_crop_box, margin_max=(6., 3.), b0_crop_scale=1.25,
        b0_crop_offset=2.0):
    """同一中心/方向下最大合法 margin；不搜索其它中心，不接受目标标签。"""
    actual = _margins(actual_margins, 'actual_margins')
    maximum = _margins(margin_max, 'margin_max')
    if np.any(actual > maximum + 1e-5):
        raise ValueError('actual acquisition margins exceed the registered maximum')
    local = [_with_margins(box, actual, maximum) for box in (endpoint_box, tube_box)]
    return tuple(apply_b0_vertical_hull(box, b0_crop_box,
                 scale=b0_crop_scale, offset=b0_crop_offset)
                 for box in (*local, corridor_box))


def support_membership(points, box, *, ignore_z=False):
    if box is None:
        return np.zeros(len(points), dtype=bool)
    center, size, rotation = _box_geometry(box)
    local = (np.asarray(points, dtype=np.float64)[:, :3] - center) @ rotation
    axes = 2 if ignore_z else 3
    return (np.abs(local[:, :axes]) < .5 * size[[1, 0, 2]][:axes]).all(axis=1)


def acquisition_margin_grid_target_v29(
        points_xyz, point_ids, target_mask, baseline_ids, *, endpoint_box,
        tube_box, corridor_box=None, actual_margins, b0_crop_box,
        b0_crop_scale=1.25, b0_crop_offset=2., margin_min=(2., 1.),
        margin_max=(6., 3.), coverage=.90, grid_size=9):
    """只生成监督：81 候选沿实际 support 方向改 margin，Z 与真实 crop 共用。"""
    points = np.asarray(points_xyz, dtype=np.float64)
    ids = np.asarray(point_ids)
    labels = np.asarray(target_mask, dtype=bool).reshape(-1)
    base_ids = np.asarray(baseline_ids)
    if (points.ndim != 2 or points.shape[1] != 3 or ids.shape != (len(points),)
            or labels.shape != (len(points),) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError('v29 margin labels require aligned unique raw point IDs and XYZ/targets')
    if not 0 < float(coverage) <= 1 or int(grid_size) != 9:
        raise ValueError('v29 margin labels require a 9x9 grid and valid coverage')
    actual = _margins(actual_margins, 'actual_margins')
    minimum, maximum = _margins(margin_min, 'margin_min'), _margins(margin_max, 'margin_max')
    if np.any(maximum <= minimum) or np.any(actual < minimum - 1e-5):
        raise ValueError('v29 margin bounds or actual margins are inconsistent')
    supports = tuple(apply_b0_vertical_hull(box, b0_crop_box,
                     scale=b0_crop_scale, offset=b0_crop_offset)
                     for box in (endpoint_box, tube_box, corridor_box))
    max_supports = maximum_acquisition_supports_v29(*supports,
        actual_margins=actual, b0_crop_box=b0_crop_box, margin_max=maximum,
        b0_crop_scale=b0_crop_scale, b0_crop_offset=b0_crop_offset)
    finite = np.isfinite(points).all(axis=1)
    novel = ~np.isin(ids, base_ids) & finite
    global_count = int(np.sum(novel & labels))
    max_member = np.logical_or.reduce([support_membership(points, box) for box in max_supports])
    table_mask = novel & max_member
    reachable = int(np.sum(table_mask & labels))
    result = dict(target_margin=minimum.astype(np.float32), valid=True,
                  reason='no_novel_target', global_novel_target_count=global_count,
                  max_reachable_target_count=reachable, selected_target_count=0,
                  selected_background_count=0, grid_index=np.asarray((0, 0), dtype=np.int64),
                  support_z_contract=Z_CONTRACT)
    if not any(box is not None for box in supports):
        result.update(valid=False, reason='no_structural_geometry')
        return result
    if global_count == 0:
        return result
    if reachable == 0:
        result.update(valid=False, reason='outside_maximum_support')
        return result
    table, truth = points[table_mask], labels[table_mask]
    parallel, perpendicular = np.meshgrid(np.linspace(minimum[0], maximum[0], 9),
                                        np.linspace(minimum[1], maximum[1], 9), indexing='ij')
    margins = np.stack((parallel.ravel(), perpendicular.ravel()), axis=1)
    foreground, background = np.zeros(81, dtype=np.int64), np.zeros(81, dtype=np.int64)
    for start in range(0, len(table), 8192):
        chunk, target = table[start:start + 8192], truth[start:start + 8192]
        member = np.zeros((81, len(chunk)), dtype=bool)
        for box in supports[:2]:
            if box is None:
                continue
            center, size, rotation = _box_geometry(box)
            local = np.abs((chunk - center) @ rotation)
            base_half = .5 * size[[1, 0]] - actual
            member |= ((local[None, :, 0] < base_half[0] + margins[:, 0:1])
                       & (local[None, :, 1] < base_half[1] + margins[:, 1:2])
                       & (local[None, :, 2] < .5 * size[2]))
        member |= support_membership(chunk, supports[2])[None, :]
        foreground += np.sum(member & target[None, :], axis=1)
        background += np.sum(member & ~target[None, :], axis=1)
    feasible = np.flatnonzero(foreground >= math.ceil(float(coverage) * reachable))
    if not len(feasible):
        raise RuntimeError('v29 maximum support and margin-grid membership diverged')
    index = min(feasible.tolist(), key=lambda k: (int(background[k]), -int(foreground[k]),
        float(((margins[k] - minimum) / (maximum - minimum)).sum()), k // 9, k % 9))
    result.update(target_margin=margins[index].astype(np.float32), reason='ok',
                  selected_target_count=int(foreground[index]),
                  selected_background_count=int(background[index]),
                  grid_index=np.asarray((index // 9, index % 9), dtype=np.int64))
    return result
