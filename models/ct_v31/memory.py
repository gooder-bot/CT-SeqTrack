"""无学习特征、无跨帧计算图的 12 + 2*(8+4) 原始身份记忆。"""
from __future__ import annotations

import numpy as np

from .acquisition import _rotation, _unique_mask, _vector, spatial_coverage_indices


class RawIdentityMemory:
    """缓存各保存框内的规范化点；当前 forward 使用当前参数重新编码。

    metadata 八维依次为 persistent, FG, BG, age_seconds, L, W, H,
    slot_present。ID 只在其来源帧内标识点，跨帧相同 ID 合法。
    """
    def __init__(self, box_size):
        self.box_size = _vector(box_size, 3, "box_size").copy()
        if np.any(self.box_size <= 0):
            raise ValueError("box_size must be positive LWH")
        self.initial = None
        self.recent = []

    def _pack(self, points_world, raw_ids, fg_mask, box, timestamp, *, initial=False):
        points = np.asarray(points_world, dtype=np.float64)
        ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
        fg = np.asarray(fg_mask, dtype=bool).reshape(-1)
        if points.ndim != 2 or points.shape[1] < 3 or not (len(points) == len(ids) == len(fg)):
            raise ValueError("points, IDs and foreground mask must have matching point counts")
        valid = _unique_mask(ids) & np.isfinite(points[:, :3]).all(1)
        state_box = _vector(box, 4, "box")
        selected, flags = [], []
        for mask, budget, is_fg in ((valid & fg, 12 if initial else 8, True),
                                    (valid & ~fg, 0 if initial else 4, False)):
            candidates = np.flatnonzero(mask)
            picked = candidates[spatial_coverage_indices(points[candidates], ids[candidates], budget)]
            selected.extend(picked.tolist())
            flags.extend([is_fg] * len(picked))
        selected = np.asarray(selected, dtype=np.int64)
        normalized = np.zeros((len(selected), 5), dtype=np.float32)
        if len(selected):
            channels = min(points.shape[1], 5)
            normalized[:, :channels] = points[selected, :channels]
            delta = points[selected, :3] - state_box[:3]
            delta[:, :2] = delta[:, :2] @ _rotation(state_box[3])
            normalized[:, :3] = delta / self.box_size
            normalized[:, 3:] = np.nan_to_num(normalized[:, 3:])
        return dict(points=normalized, ids=ids[selected].copy(), fg=np.asarray(flags, dtype=bool),
                    timestamp=float(timestamp), box=state_box.copy(), initial=bool(initial))

    def initialize(self, points_world, raw_ids, target_mask, box, timestamp):
        """首帧 GT 唯一入口；即使稀疏也初始化，空槽绝不复制点。"""
        self.initial = self._pack(points_world, raw_ids, target_mask, box, timestamp, initial=True)
        self.recent = []

    def update(self, points_world, raw_ids, foreground_probability, box, timestamp, quality):
        """仅已接受预测帧，quality>=.5 且至少三个 unique predicted FG。"""
        if self.initial is None:
            raise RuntimeError("initialize first-frame memory before prediction updates")
        probability = np.asarray(foreground_probability, dtype=np.float64).reshape(-1)
        ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
        points = np.asarray(points_world)
        if not (len(probability) == len(ids) == len(points)):
            raise ValueError("foreground probabilities must match the raw points")
        foreground = np.isfinite(probability) & (probability >= .5)
        good = _unique_mask(ids) & np.isfinite(points[:, :3]).all(1)
        if not np.isfinite(quality) or float(quality) < .5 or int((foreground & good).sum()) < 3:
            return False
        if self.recent and float(timestamp) <= self.recent[-1]["timestamp"]:
            raise ValueError("accepted memory timestamps must strictly increase")
        if float(timestamp) <= self.initial["timestamp"]:
            raise ValueError("accepted frame must follow the initial template")
        self.recent.append(self._pack(points, ids, foreground, box, timestamp))
        self.recent = self.recent[-2:]
        return True

    def export(self, timestamp):
        points = np.zeros((36, 5), np.float32)
        valid = np.zeros(36, bool)
        ids = np.full(36, -1, np.int64)
        metadata = np.zeros((36, 8), np.float32)
        records = [(0, self.initial)] + [(12 + 12 * i, row) for i, row in enumerate(self.recent)]
        for start, record in records:
            if record is None:
                continue
            count = len(record["ids"])
            slots = slice(start, start + count)
            points[slots], ids[slots], valid[slots] = record["points"], record["ids"], True
            metadata[slots, 0] = float(record["initial"])
            metadata[slots, 1] = record["fg"]
            metadata[slots, 2] = ~record["fg"]
            metadata[slots, 3] = max(0., float(timestamp) - record["timestamp"])
            metadata[slots, 4:7] = self.box_size
            metadata[slots, 7] = 1.
        return dict(memory_points=points, memory_valid=valid, memory_ids=ids, memory_metadata=metadata)
