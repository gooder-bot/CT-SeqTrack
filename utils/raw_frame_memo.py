"""单个 prepare/acquire/commit 事务内的只读原帧物化。

只缓存 raw 转换、GT 静态几何和 raw-ID 索引；不缓存 crop、采样或网络值。
保留对象强引用，避免事务内 Python id 复用；目标框改变身份时不共享标签。
"""
from functools import wraps

import numpy as np


class RawFrameMemo:
    def __init__(self, point_arrays, box_array, box_size, box_corners, inside_box):
        self._point_arrays, self._box_array = point_arrays, box_array
        self._box_size, self._box_corners, self._inside_box = box_size, box_corners, inside_box
        self._frames, self._geometry, self._indices, self._target_masks = {}, {}, {}, {}

    def points(self, frame):
        key = id(frame)
        if key not in self._frames:
            self._frames[key] = (frame, self._point_arrays(frame))
        return self._frames[key][1]

    def geometry(self, frame):
        target, size_object = frame['3d_bbox'], frame.get('box_size')
        key = (id(frame), id(target), id(size_object))
        if key not in self._geometry:
            box, size = self._box_array(target), self._box_size(target, frame)
            centers = np.vstack((box[:3], self._box_corners(box, size)))
            self._geometry[key] = (frame, target, size_object, box, size, centers)
        _, _, _, box, size, centers = self._geometry[key]
        # 调用方随后将 GT 平移到 anchor-local，不能改写缓存的世界框。
        return box.copy(), size, centers

    def target_mask(self, frame):
        """仅已有整云 GT 诊断需要时计算；历史采样标签不触发整云扫描。"""
        key = (id(frame), id(frame['3d_bbox']), id(frame.get('box_size')))
        if key not in self._target_masks:
            box, size, _ = self.geometry(frame)
            xyz, _ = self.points(frame)
            self._target_masks[key] = self._inside_box(xyz, box, size)
        return self._target_masks[key]

    def index(self, frame):
        key = id(frame)
        if key not in self._indices:
            _, ids = self.points(frame)
            self._indices[key] = {int(value): index for index, value in enumerate(ids)}
        return self._indices[key]


def raw_transaction(method):
    """异常及 commit 退出立即释放缓存；网络阶段由 host 的异常路径负责。"""
    @wraps(method)
    def call(builder, *args, **kwargs):
        if method.__name__ == 'prepare' and builder._pending is None:
            builder._start_materialization_cache()
        try:
            return method(builder, *args, **kwargs)
        except BaseException:
            builder.clear_materialization_cache()
            raise
        finally:
            if method.__name__ == 'commit':
                builder.clear_materialization_cache()
    return call
