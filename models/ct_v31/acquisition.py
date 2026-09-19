"""v31 原始点获取：两个独立支持域、唯一 ID 预算与同几何获取监督。"""
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
    b0 = _vector(b0_box, 4, "b0_box")
    size = _vector(box_size, 3, "box_size")
    if np.any(size <= 0):
        raise ValueError("box_size must be positive LWH")
    prior = _vector(prior_center, 3, "prior_center")
    recovery = _vector(recovery_center, 3, "recovery_center")
    fraction = np.clip(_vector(u, 2, "u"), 0., 1.)
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
    local_half = np.r_[(upper - lower) * .5 + [.25, .25] + fraction * [3.75, 2.75],
                       (z_upper - z_lower) * .5]
    local_center = np.r_[b0[:2] + midpoint @ local_rotation.T, (z_lower + z_upper) * .5]
    # R 的 core 是可信传播终点框，recovery_yaw 同时定义其物体轴。
    # 不能再用漂移 B0 的 yaw 投影，否则 R 的尺寸会随不可信框旋转。
    recovery_half_xy = size[:2] * .5
    recovery_half = np.r_[recovery_half_xy + [4., 3.] + fraction * [8., 5.],
                          size[2] * .5 + z_margin]
    return {"local": {"center": local_center, "half": local_half, "yaw": local_yaw},
            "recovery": {"center": recovery.copy(), "half": recovery_half, "yaw": recover_yaw},
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


def hash_order(raw_ids, seed=0):
    """无 RNG 状态的固定 ID 散列；相同 hash 以原始 ID 决定。"""
    ids = np.asarray(raw_ids, dtype=np.int64)
    value = ids.astype(np.uint64) ^ np.uint64(int(seed) & ((1 << 64) - 1))
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    value ^= value >> np.uint64(31)
    return np.lexsort((ids, value))


def spatial_coverage_indices(points, ids, count, seed=0):
    """确定性 XY 最远点覆盖；只返回唯一输入槽，不复制稀疏点。"""
    xyz = np.asarray(points, dtype=np.float64)[:, :2]
    count = min(max(0, int(count)), len(xyz))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = hash_order(ids, seed)
    # 固定 tie 顺序与原输入排列无关。
    xyz = xyz[order]
    selected = np.empty(count, dtype=np.int64)
    distance = np.full(len(xyz), np.inf)
    available = np.ones(len(xyz), dtype=bool)
    current = 0
    for index in range(count):
        selected[index] = current
        available[current] = False
        distance = np.minimum(distance, ((xyz - xyz[current]) ** 2).sum(1))
        current = int(np.argmax(np.where(available, distance, -1.)))
    return order[selected]


def acquire_extension(cloud_points, raw_ids, *, b0_raw_ids, anchor, b0_box, box_size,
                      prior_center, recovery_center, u, support_yaw=None,
                      recovery_yaw=None, crop_scale=1.25, crop_offset=2.,
                      b0_crop_half=None, seed=0):
    points = np.asarray(cloud_points, dtype=np.float64)
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("cloud_points must be [N, >=3]")
    support = build_dual_support(b0_box=b0_box, box_size=box_size, prior_center=prior_center,
        recovery_center=recovery_center, u=u, support_yaw=support_yaw, recovery_yaw=recovery_yaw,
        crop_scale=crop_scale, crop_offset=crop_offset, b0_crop_half=b0_crop_half)
    membership = support_membership(points, support, b0_raw_ids, ids)
    selected, partitions = [], []
    remaining = []
    for partition, mask in enumerate(membership):
        candidates = np.flatnonzero(mask)
        picked = candidates[spatial_coverage_indices(points[candidates], ids[candidates], 384, seed)]
        selected.extend(picked.tolist())
        partitions.extend([partition] * len(picked))
        picked_set = set(picked.tolist())
        remaining.extend([(int(i), partition) for i in candidates if i not in picked_set])
    if len(selected) < 768 and remaining:
        candidates = np.asarray([row[0] for row in remaining], dtype=np.int64)
        extra = spatial_coverage_indices(points[candidates], ids[candidates], 768 - len(selected), seed)
        selected.extend(candidates[extra].tolist())
        partitions.extend([remaining[i][1] for i in extra])
    out_points = np.zeros((768, 5), dtype=np.float32)
    out_ids = np.full(768, -1, dtype=np.int64)
    out_partition = np.full(768, -1, dtype=np.int64)
    valid = np.zeros(768, dtype=bool)
    count = len(selected)
    if count:
        selected = np.asarray(selected, dtype=np.int64)
        channels = min(points.shape[1], 5)
        out_points[:count, :channels] = points[selected, :channels]
        anchor_xyz = np.asarray(anchor, dtype=np.float64).reshape(-1)
        if len(anchor_xyz) not in (3, 4):
            raise ValueError("anchor must be world XYZ or world XYZ/yaw")
        # 世界坐标先以 float64 平移再落 float32，避免大地图坐标相减丢精度。
        out_points[:count, :3] = points[selected, :3] - _vector(anchor_xyz[:3], 3, "anchor")
        out_points[:count, 3:] = np.nan_to_num(out_points[:count, 3:])
        out_ids[:count], out_partition[:count], valid[:count] = ids[selected], partitions, True
    return dict(extension_points=out_points, extension_ids=out_ids, extension_valid=valid,
                extension_partition=out_partition)


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
    maximum = build_dual_support(u=[1., 1.], **kwargs)
    a, b = support_membership(cloud_points, maximum, b0_raw_ids, raw_ids)
    target_count_max = int(((a | b) & target).sum())
    if not target_count_max:
        return dict(acquisition_target=np.zeros(2, np.float32), acquisition_valid=True,
                    acquisition_demand=False,
                    maximum_target_count=0, target_count=0, background_count=0)
    # 大全云只做一次最大支持域裁剪；枚举只读取最大域内唯一 extension 点。
    maximum_members = a | b
    cloud_points = np.asarray(cloud_points)[maximum_members]
    raw_ids = np.asarray(raw_ids)[maximum_members]
    target = target[maximum_members]
    b0_raw_ids = np.empty(0, dtype=np.int64)
    required = int(np.ceil(float(target_fraction) * target_count_max))
    choices = []
    for u0 in np.linspace(0., 1., 9):
        for u1 in np.linspace(0., 1., 9):
            support = build_dual_support(u=[u0, u1], **kwargs)
            a, b = support_membership(cloud_points, support, b0_raw_ids, raw_ids)
            included = a | b
            fg, bg = int((included & target).sum()), int((included & ~target).sum())
            if fg >= required:
                # 面积只在背景数相同时打破并列；不冒称两个旋转框 union 的精确面积。
                area = sum(4. * np.prod(support[key]["half"][:2]) for key in ("local", "recovery"))
                choices.append(((bg, area, u0 + u1, u0, u1), fg, u0, u1))
    choice = min(choices, key=lambda row: row[0])
    return dict(acquisition_target=np.asarray(choice[2:4], np.float32), acquisition_valid=True,
                acquisition_demand=True,
                maximum_target_count=target_count_max, target_count=choice[1],
                background_count=choice[0][0])
