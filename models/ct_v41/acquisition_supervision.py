"""B1 acquisition 辅助监督的原双支持域标签；无扩展点获取。"""
from __future__ import annotations

import numpy as np


def _rotation(yaw):
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _vector(value, count, name):
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != count or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {count} finite values")
    return result


def build_dual_support(*, b0_box, box_size, prior_center, recovery_center, u,
                       support_yaw=None, recovery_yaw=None, crop_scale=1.25,
                       crop_offset=2.0, b0_crop_half=None):
    """框为 world XYZ/yaw，尺寸为 LWH；L 包含 B0/prior，R 只有可信终点。

    b0_crop_half 可显式传实际 crop 半宽。Z margin 等于实际 B0 crop
    半高减初始物体半高；R 的 Z 中心与范围不会被漂移 B0 裁剪。
    """
    geometry = _dual_geometry(b0_box=b0_box, box_size=box_size, prior_center=prior_center,
        recovery_center=recovery_center, support_yaw=support_yaw, recovery_yaw=recovery_yaw,
        crop_scale=crop_scale, crop_offset=crop_offset, b0_crop_half=b0_crop_half)
    return _support_at(geometry, u)


def _dual_geometry(*, b0_box, box_size, prior_center, recovery_center,
                   support_yaw=None, recovery_yaw=None, crop_scale=1.25,
                   crop_offset=2., b0_crop_half=None):
    """同一 9×9 网格共享旋转/中心，float64 运算顺序保持原几何定义。"""
    b0 = _vector(b0_box, 4, "b0_box")
    size = _vector(box_size, 3, "box_size")
    if np.any(size <= 0):
        raise ValueError("box_size must be positive LWH")
    prior = _vector(prior_center, 3, "prior_center")
    recovery = _vector(recovery_center, 3, "recovery_center")
    local_yaw = float(b0[3] if support_yaw is None else support_yaw)
    recover_yaw = float(local_yaw if recovery_yaw is None else recovery_yaw)
    crop_half = (size * .5 * float(crop_scale) + float(crop_offset)
                 if b0_crop_half is None else _vector(b0_crop_half, 3, "b0_crop_half"))
    if np.any(crop_half <= 0):
        raise ValueError("actual crop half widths must be positive")
    local_rotation = _rotation(local_yaw)
    box_rotation = _rotation(b0[3])
    crop_xy = np.abs(local_rotation.T @ box_rotation) @ crop_half[:2]
    object_xy = np.abs(local_rotation.T @ box_rotation) @ (size[:2] * .5)
    endpoint_xy = (prior[:2] - b0[:2]) @ local_rotation
    lower = np.minimum(-crop_xy, endpoint_xy - object_xy)
    upper = np.maximum(crop_xy, endpoint_xy + object_xy)
    midpoint = (lower + upper) * .5
    z_margin = max(0., float(crop_half[2] - size[2] * .5))
    z_lower = min(b0[2] - crop_half[2], prior[2] - size[2] * .5 - z_margin)
    z_upper = max(b0[2] + crop_half[2], prior[2] + size[2] * .5 + z_margin)
    local_base_xy = (upper - lower) * .5 + [.25, .25]
    local_center = np.r_[b0[:2] + midpoint @ local_rotation.T, (z_lower + z_upper) * .5]
    # R 的 core 是可信传播终点框，recovery_yaw 同时定义其物体轴。
    # 不能再用漂移 B0 的 yaw 投影，否则 R 的尺寸会随不可信框旋转。
    recovery_half_xy = size[:2] * .5
    return dict(local_center=local_center, local_base_xy=local_base_xy,
                local_half_z=(z_upper - z_lower) * .5, local_yaw=local_yaw,
                local_rotation=local_rotation, recovery_center=recovery.copy(),
                recovery_base_xy=recovery_half_xy + [4., 3.],
                recovery_half_z=size[2] * .5 + z_margin, recovery_yaw=recover_yaw,
                recovery_rotation=_rotation(recover_yaw))


def _support_at(geometry, u):
    fraction = np.clip(_vector(u, 2, "u"), 0., 1.)
    local_half = np.r_[geometry['local_base_xy'] + fraction * [3.75, 2.75], geometry['local_half_z']]
    recovery_half = np.r_[geometry['recovery_base_xy'] + fraction * [8., 5.], geometry['recovery_half_z']]
    return {"local": {"center": geometry['local_center'], "half": local_half, "yaw": geometry['local_yaw']},
            "recovery": {"center": geometry['recovery_center'], "half": recovery_half,
                         "yaw": geometry['recovery_yaw']},
            "u": fraction.astype(np.float32)}


def points_in_support(points, region):
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    delta = xyz - region["center"]
    xy = delta[:, :2] @ _rotation(region["yaw"])
    return (np.isfinite(xyz).all(1) & (np.abs(xy) <= region["half"][:2] + 1e-7).all(1)
            & (np.abs(delta[:, 2]) <= region["half"][2] + 1e-7))


def _unique_mask(raw_ids):
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    good = ids >= 0
    _, first = np.unique(ids[good], return_index=True)
    result = np.zeros(len(ids), dtype=bool)
    result[np.flatnonzero(good)[first]] = True
    return result


def support_membership(points, support, b0_raw_ids, raw_ids):
    """A=L\\B0raw；B=R\\(L union B0raw)，partition 保留真实来源。"""
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    if len(ids) != len(points):
        raise ValueError("raw_ids and cloud_points must have equal length")
    excluded = np.isin(ids, np.asarray(b0_raw_ids, dtype=np.int64).reshape(-1))
    eligible = _unique_mask(ids) & ~excluded
    local = points_in_support(points, support["local"])
    recovery = points_in_support(points, support["recovery"])
    return eligible & local, eligible & recovery & ~local


def band_grid_target(cloud_points, raw_ids, *, target_mask, b0_raw_ids, b0_box, box_size,
                     prior_center, recovery_center, support_yaw=None, recovery_yaw=None,
                     crop_scale=1.25, crop_offset=2., b0_crop_half=None, target_fraction=.9):
    """9x9 同族支持域：覆盖最大可达 unique FG 的 q90 后最少背景。

    这是获取监督，不参与真实获取或候选排序。无可达 FG 时目标 u=0，
    demand=False，但有效监督仍为 True（与有需求样本分组归一化）。
    """
    target = np.asarray(target_mask, dtype=bool).reshape(-1)
    if len(target) != len(cloud_points):
        raise ValueError("target_mask must match raw cloud")
    kwargs = dict(b0_box=b0_box, box_size=box_size, prior_center=prior_center,
                  recovery_center=recovery_center, support_yaw=support_yaw,
                  recovery_yaw=recovery_yaw, crop_scale=crop_scale,
                  crop_offset=crop_offset, b0_crop_half=b0_crop_half)
    geometry = _dual_geometry(**kwargs)
    maximum = _support_at(geometry, [1., 1.])
    xyz = np.asarray(cloud_points, dtype=np.float64)[:, :3]
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    if len(ids) != len(xyz):
        raise ValueError('raw_ids and cloud_points must have equal length')
    eligible = _unique_mask(ids) & ~np.isin(ids, np.asarray(b0_raw_ids, dtype=np.int64).reshape(-1))
    local_delta, recovery_delta = xyz - geometry['local_center'], xyz - geometry['recovery_center']
    local_xy = np.abs(local_delta[:, :2] @ geometry['local_rotation'])
    recovery_xy = np.abs(recovery_delta[:, :2] @ geometry['recovery_rotation'])
    local_z, recovery_z = np.abs(local_delta[:, 2]), np.abs(recovery_delta[:, 2])
    finite = np.isfinite(xyz).all(1)
    local = ((local_xy <= maximum['local']['half'][:2] + 1e-7).all(1)
             & (local_z <= maximum['local']['half'][2] + 1e-7))
    recovery = ((recovery_xy <= maximum['recovery']['half'][:2] + 1e-7).all(1)
                & (recovery_z <= maximum['recovery']['half'][2] + 1e-7))
    maximum_members = eligible & finite & (local | recovery)
    target_count_max = int((maximum_members & target).sum())
    if not target_count_max:
        return dict(acquisition_target=np.zeros(2, np.float32), acquisition_valid=True,
                    acquisition_demand=False,
                    maximum_target_count=0, target_count=0, background_count=0)
    # 大全云只做一次最大支持域裁剪；枚举只读取最大域内唯一 extension 点。
    local_xy, recovery_xy = local_xy[maximum_members], recovery_xy[maximum_members]
    local_z, recovery_z = local_z[maximum_members], recovery_z[maximum_members]
    target = target[maximum_members]
    required = int(np.ceil(float(target_fraction) * target_count_max))
    # 81 个假设只广播 membership；点维分块，避免构造随全云增大的 81×N 临时张量。
    grid = [(u0, u1) for u0 in np.linspace(0., 1., 9) for u1 in np.linspace(0., 1., 9)]
    supports = [_support_at(geometry, u) for u in grid]
    local_half = np.stack([s['local']['half'] for s in supports])
    recovery_half = np.stack([s['recovery']['half'] for s in supports])
    fg_counts = np.zeros(len(grid), dtype=np.int64)
    bg_counts = np.zeros(len(grid), dtype=np.int64)
    for start in range(0, len(target), 4096):
        stop = min(start + 4096, len(target))
        local = ((local_xy[None, start:stop] <= local_half[:, None, :2] + 1e-7).all(-1)
                 & (local_z[None, start:stop] <= local_half[:, None, 2] + 1e-7))
        recovery = ((recovery_xy[None, start:stop] <= recovery_half[:, None, :2] + 1e-7).all(-1)
                    & (recovery_z[None, start:stop] <= recovery_half[:, None, 2] + 1e-7))
        included = local | recovery
        fg_counts += (included & target[None, start:stop]).sum(1, dtype=np.int64)
        bg_counts += (included & ~target[None, start:stop]).sum(1, dtype=np.int64)
    choices = []
    for index, ((u0, u1), support) in enumerate(zip(grid, supports)):
        fg, bg = int(fg_counts[index]), int(bg_counts[index])
        if fg >= required:
            # 保持旧版逐标量面积运算及 lexicographic tie 顺序。
            area = sum(4. * np.prod(support[key]["half"][:2]) for key in ("local", "recovery"))
            choices.append(((bg, area, u0 + u1, u0, u1), fg, u0, u1))
    choice = min(choices, key=lambda row: row[0])
    return dict(acquisition_target=np.asarray(choice[2:4], np.float32), acquisition_valid=True,
                acquisition_demand=True,
                maximum_target_count=target_count_max, target_count=choice[1],
                background_count=choice[0][0])
