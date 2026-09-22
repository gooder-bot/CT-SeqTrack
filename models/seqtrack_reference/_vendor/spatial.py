"""无需 nuscenes-devkit 的有向框 membership，边界为闭区间。"""
import numpy as np


def points_in_box(box, points, wlh_factor=1.0):
    corners = box.corners(wlh_factor=wlh_factor)
    origin = corners[:, 0]
    axes = (corners[:, 4] - origin, corners[:, 1] - origin, corners[:, 3] - origin)
    delta = np.asarray(points) - origin[:, None]
    inside = np.ones(delta.shape[1], dtype=bool)
    for axis in axes:
        projection = axis @ delta
        inside &= (projection >= 0.) & (projection <= axis @ axis)
    return inside
