"""联合搜索的独立测量历史、模板及单次复核状态（传感器坐标、物理时间）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


EMPTY_XYZ = np.empty((0, 3), dtype=np.float32)
EMPTY_IDS = np.empty(0, dtype=np.int64)
MAX_TEMPLATE_POINTS = 256


def template_points(xyz, box):
    """模板始终在其创建时的车体轴表达，固定地缩减到可缓存的点数。"""
    if len(xyz) < 3:
        return EMPTY_XYZ.copy()
    if len(xyz) > MAX_TEMPLATE_POINTS:
        xyz = xyz[np.linspace(0, len(xyz) - 1, MAX_TEMPLATE_POINTS, dtype=np.int64)]
    delta = np.asarray(xyz) - box[:3]
    c, s = np.cos(box[3]), np.sin(box[3])
    return np.column_stack((c*delta[:, 0] + s*delta[:, 1],
                            -s*delta[:, 0] + c*delta[:, 1], delta[:, 2])).astype(np.float32)


def relative(box, anchor):
    value = np.asarray(box, dtype=np.float64).copy()
    value[:3] -= anchor[:3]
    return value.astype(np.float32)


def physical_frame(frame, fallback):
    for candidate in ('raw_frame_id', 'frame_id'):
        value = frame.get(candidate)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError, OverflowError):
                continue
    return int(fallback)


@dataclass
class Measurement:
    box: np.ndarray
    time: float
    scan: int
    adjacent: bool
    can_start_pair: bool = True


@dataclass
class JointSearchState:
    # 观测按新测量写入。连续缺测可以推进 motion_box，但不消耗这三个实测槽。
    measurements: list
    motion_box: np.ndarray
    motion_time: float
    template_xyz: np.ndarray
    last_scan_time: float
    template_streak: int = 0
    pending_box: np.ndarray | None = None
    pending_scan: int | None = None
    history_start: int = 0
    weak_history_boxes: dict = field(default_factory=dict)

    @classmethod
    def from_seed(cls, state, frames, first_frame, first_xyz, first_box, supported_seeds, *, patrol_seconds=1.):
        samples = []
        indices = sorted(state.boxes)
        seed_times = np.asarray([state.times[i] for i in indices])
        if not np.isfinite(seed_times).all() or (np.diff(seed_times) <= 0).any():
            raise ValueError('joint seed times must strictly increase')
        for index in indices:
            if not supported_seeds[index]:
                continue
            scan = physical_frame(frames[index], index)
            adjacent = bool(samples and scan == samples[-1].scan + 1)
            if not adjacent:
                samples.clear()
            samples.append(Measurement(state.boxes[index].copy(), state.times[index], scan, adjacent))
        end = indices[-1]
        end_time = state.times[end]
        if not samples:
            # 知道合法种子的位置，但没有测量速度所需的物理证据。
            samples.append(Measurement(state.boxes[end].copy(), end_time, physical_frame(frames[end], end),
                                       adjacent=False, can_start_pair=False))
        last = samples[-1]
        state.trusted = [(sample.box.copy(), sample.time) for sample in samples[-2:]]
        state.trusted_velocity = np.zeros(3, dtype=np.float64)
        state.trusted_velocity_valid = last.adjacent
        state.last_velocity_update_time = last.time if last.adjacent else None
        state.last_supported_time = last.time
        if last.adjacent:
            prev = samples[-2]
            state.trusted_velocity = (last.box[:3] - prev.box[:3]) / (last.time - prev.time)
        motion = last.box.copy()
        motion[:3] += (end_time - last.time) * state.trusted_velocity
        if last.time < end_time:
            state.previous_measurement_eligible = False
        origin = float(first_frame['timestamp'])
        # 窗口可能从轨迹中途重置，健康巡查仍沿合法首框的物理时钟对齐。
        intervals = np.floor((max(0., end_time - origin) + 1e-7) / patrol_seconds)
        previous_scan = min(end_time, origin + intervals * patrol_seconds)
        return cls(samples[-3:], motion, end_time,
                   template_points(first_xyz, first_box), previous_scan)

    def crop_center(self, state, timestamp):
        if state.previous_measurement_eligible is not False:
            return self.motion_box.copy()
        box = self.motion_box.copy()
        if state.trusted_velocity_valid:
            box[:3] += state.trusted_velocity * (timestamp - self.motion_time)
        return box

    def pending_at(self, scan):
        return self.pending_scan is not None and scan == self.pending_scan + 1

    def search_due(self, state, timestamp, scan, *, patrol_seconds=1., weak_patrol_seconds=.2):
        interval = weak_patrol_seconds if state.consecutive_untrusted_outputs >= 2 else patrol_seconds
        return self.pending_at(scan) or timestamp - self.last_scan_time >= interval - 1e-7

    def motion_fields(self, anchor, now):
        boxes = np.zeros((3, 4), dtype=np.float32)
        times = np.zeros(3, dtype=np.float32)
        valid = np.zeros(3, dtype=bool)
        pairs = np.zeros(2, dtype=bool)
        start = 3 - len(self.measurements)
        for j, sample in enumerate(self.measurements, start):
            boxes[j] = relative(sample.box, anchor)
            times[j], valid[j] = sample.time - now, sample.can_start_pair
            if j > start:
                pairs[j-1] = sample.adjacent
        return dict(joint_motion_boxes=boxes, joint_motion_times=times,
                    joint_motion_valid=valid, joint_motion_pairs=pairs,
                    joint_motion_origin=relative(self.motion_box, anchor))

    def advance(self, state, *, frame, scan, now, box, prior, source, supported,
                support_xyz, searched, proposal):
        previous = self.measurements[-1]
        remote = source == 5
        jump = np.linalg.norm(box[:2] - prior[:2]) > .5 * np.hypot(*state.size[:2])
        if not np.isfinite(now) or now <= self.motion_time:
            raise ValueError('joint update time must strictly increase')
        adjacent = supported and not remote and not jump and previous.can_start_pair and previous.scan + 1 == scan
        if supported:
            if not adjacent:
                self.measurements.clear()
            if remote or jump:
                self.history_start = frame
            self.measurements.append(Measurement(box.copy(), now, scan, adjacent))
            self.measurements = self.measurements[-3:]
            self.motion_box = box.copy()
            if adjacent:
                state.trusted_velocity = (box[:3] - previous.box[:3]) / (now - previous.time)
                state.trusted_velocity_valid = True
                state.last_velocity_update_time = now
            else:
                state.trusted_velocity = np.zeros(3, dtype=np.float64)
                state.trusted_velocity_valid = False
            state.trusted.append((box.copy(), now))
            state.trusted[:] = state.trusted[-2:]
            state.last_supported_time = now
        else:
            self.motion_box = prior.copy()
            self.weak_history_boxes[frame] = self.motion_box.copy()
        self.motion_time = now
        self.weak_history_boxes = {i: b for i, b in self.weak_history_boxes.items() if i >= frame - 2}
        if not len(self.template_xyz):
            if supported and not remote:
                self.template_streak = self.template_streak + 1 if adjacent else 1
                if self.template_streak >= 2:
                    self.template_xyz = template_points(support_xyz, box)
            else:
                self.template_streak = 0
        # 每个提议只能拿下一扫描的一次独立复核。复核失败不会在此步自动续约。
        was_pending = self.pending_at(scan)
        self.pending_box = self.pending_scan = None
        if searched:
            self.last_scan_time = now
        if searched and proposal is not None and not was_pending and not remote:
            self.pending_box, self.pending_scan = proposal.copy(), scan
        return bool(adjacent)
