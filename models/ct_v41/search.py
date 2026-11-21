"""仅决定当前搜索点集；不改变 accepted、物理先验或可信状态时间。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .cpu_geometry import compiled_backend
from .identity import JOINT_SEARCH_POLICY


def _unique_mask(raw_ids):
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    good = ids >= 0
    _, first = np.unique(ids[good], return_index=True)
    result = np.zeros(len(ids), dtype=bool)
    result[np.flatnonzero(good)[first]] = True
    return result


@dataclass(frozen=True)
class SearchPlan:
    main_box: np.ndarray
    backup_box: np.ndarray | None
    triggered: bool
    backup_elapsed: float
    trusted_age: float
    reason: str
    main_quota: int
    backup_quota: int


@dataclass(frozen=True)
class SearchSample:
    indices: np.ndarray
    union_indices: np.ndarray
    main_count: int
    backup_only_count: int
    sampled_main: int
    sampled_backup: int
    main_indices: np.ndarray


def make_search_plan(*, anchor, trusted_box, trusted_time, velocity, velocity_valid,
                     timestamp, previous_measurement_eligible, policy='accepted_only',
                     max_age=1.5, main_quota=768, backup_quota=256):
    """None 表示合法窗口初始化；只有实际提交的 False 才触发保底搜索。"""
    main = np.asarray(anchor, dtype=np.float64).copy()
    trusted = np.asarray(trusted_box, dtype=np.float64).copy()
    velocity = np.asarray(velocity, dtype=np.float64)
    if main.shape != (4,) or trusted.shape != (4,) or velocity.shape != (3,):
        raise ValueError('search geometry requires XYZ/yaw boxes and XYZ velocity')
    if not (np.isfinite(main).all() and np.isfinite(trusted).all()
            and np.isfinite(velocity).all() and np.isfinite(timestamp)
            and np.isfinite(trusted_time) and np.isfinite(max_age) and max_age > 0):
        raise ValueError('search geometry/time must be finite with positive max_age')
    if policy not in ('accepted_only', 'dual_roi_guard_v1', JOINT_SEARCH_POLICY):
        raise ValueError('unknown v41 search policy')
    if (type(main_quota) is not int or type(backup_quota) is not int
            or min(main_quota, backup_quota) < 0 or main_quota + backup_quota < 1):
        raise ValueError('search plan requires nonnegative integer quotas and a positive budget')
    if previous_measurement_eligible is not None and not isinstance(
            previous_measurement_eligible, (bool, np.bool_)):
        raise TypeError('previous measurement eligibility must be bool or initialization None')
    age = float(timestamp) - float(trusted_time)
    if age < 0:
        raise ValueError('trusted position cannot be in the future')
    reason = ('policy_disabled' if policy == 'accepted_only' else
              'initialization' if previous_measurement_eligible is None else
              'accepted_eligible' if bool(previous_measurement_eligible) else 'accepted_untrusted')
    triggered = reason == 'accepted_untrusted'
    elapsed = min(age, float(max_age)) if bool(velocity_valid) else 0.
    main.setflags(write=False)
    if not triggered:
        return SearchPlan(main, None, False, 0., age, reason, main_quota, backup_quota)
    trusted[:3] += velocity * elapsed
    trusted.setflags(write=False)
    return SearchPlan(main, trusted, True, elapsed, age, reason, main_quota, backup_quota)


def inside_box(points, box, size, *, scale=1., offset=0.):
    """搜索、种子、监督和提交共用的闭区间框测试；保留原世界坐标精度。"""
    if (type(points) is np.ndarray and points.ndim == 2 and points.shape[1] == 3
            and points.dtype in (np.float32, np.float64)
            and type(box) is np.ndarray and box.shape == (4,) and box.dtype == np.float64
            and type(size) is np.ndarray and size.shape == (3,) and size.dtype == np.float64):
        backend = compiled_backend()
        if backend is not None:
            limits = size * (float(scale) / 2) + float(offset)
            return backend.inside_box(points, box, np.cos(box[3]), np.sin(box[3]), limits)
    return _inside_box_numpy(points, box, size, scale=scale, offset=offset)


def _inside_box_numpy(points, box, size, *, scale=1., offset=0.):
    delta = np.asarray(points)[..., :3] - np.asarray(box)[:3]
    c, s = np.cos(box[3]), np.sin(box[3])
    x = c * delta[..., 0] + s * delta[..., 1]
    y = -s * delta[..., 0] + c * delta[..., 1]
    limits = np.asarray(size) * (float(scale) / 2) + float(offset)
    # 展开坐标轴时维持原 stack -> abs -> 与 limits 广播比较的 dtype 提升。
    local_dtype = np.result_type(x.dtype, y.dtype, delta.dtype)
    compare_dtype = np.result_type(local_dtype, limits.dtype)
    limits = np.broadcast_to(limits, np.broadcast_shapes(limits.shape, (3,)))

    def within(values, axis):
        magnitude = np.abs(values.astype(local_dtype, copy=False))
        return magnitude.astype(compare_dtype, copy=False) <= limits[..., axis]

    mask = within(x, 0) & within(y, 1)
    return mask & within(delta[..., 2], 2)


def sample_search_points(points, raw_ids, plan, size, *, count=1024, main_quota=768,
                         backup_quota=256, seed=0, scale=1.25, offset=2., known_unique_ids=False):
    """主域优先拥有重叠 ID；配额不足互相回填，不重复点，不连接两域之间背景。"""
    xyz, ids = np.asarray(points), np.asarray(raw_ids, dtype=np.int64)
    if xyz.shape != (len(ids), 3) or ids.ndim != 1:
        raise ValueError('search raw points/IDs must align')
    if count < 1 or min(main_quota, backup_quota) < 0 or main_quota + backup_quota != count:
        raise ValueError('search quotas must partition the fixed point budget')
    if (main_quota, backup_quota) != (plan.main_quota, plan.backup_quota):
        raise ValueError('sampling quotas differ from the immutable search plan')
    # 内部 KITTI 视图来自 bin 行号的有序子集；公共入口保留首次出现去重规则。
    unique = None if known_unique_ids else _unique_mask(ids)
    main_mask = inside_box(xyz, plan.main_box, size, scale=scale, offset=offset)
    if unique is not None:
        main_mask &= unique
    main = np.flatnonzero(main_mask)
    backup = np.empty(0, dtype=np.int64)
    if plan.backup_box is not None:
        backup_mask = inside_box(xyz, plan.backup_box, size, scale=scale, offset=offset) & ~main_mask
        if unique is not None:
            backup_mask &= unique
        backup = np.flatnonzero(backup_mask)
    if not len(backup):
        # 无新增备用点时连抽样顺序也保持原版，不因“触发过”改变正常输入。
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(main, count, replace=False)) if len(main) > count else main.copy()
        return SearchSample(selected, main, len(main), 0, len(selected), 0, main)
    union = np.sort(np.concatenate((main, backup)))
    # 双域独立随机命名空间（0x563430 = V41）；不消费原单域抽样流。
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x563430, 1]))
    take_main, take_backup = min(len(main), main_quota), min(len(backup), backup_quota)
    spare = count - take_main - take_backup
    extra_main = min(spare, len(main) - take_main)
    take_main += extra_main
    take_backup += min(spare - extra_main, len(backup) - take_backup)
    chosen_main = (np.sort(rng.choice(main, take_main, replace=False))
                   if len(main) > take_main else main.copy())
    chosen_backup = (np.sort(rng.choice(backup, take_backup, replace=False))
                     if len(backup) > take_backup else backup.copy())
    return SearchSample(np.sort(np.concatenate((chosen_main, chosen_backup))), union,
                        len(main), len(backup), len(chosen_main), len(chosen_backup), main)
