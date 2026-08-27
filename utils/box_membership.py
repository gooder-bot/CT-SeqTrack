"""Shared strict point/box membership primitives."""

import copy

import numpy as np


def axis_aligned_box_membership_mask(
        point_columns, box, offset=0, scale=1.0, ignore_z=False):
    """Return strict membership for points and a box in the same frame."""
    point_columns = np.asarray(point_columns)
    if point_columns.ndim != 2 or point_columns.shape[0] < 3:
        raise ValueError("point columns must have shape [C>=3,N]")
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * float(scale)
    if hasattr(box_tmp, "corners"):
        maxi = np.max(box_tmp.corners(), 1) + float(offset)
        mini = np.min(box_tmp.corners(), 1) - float(offset)
    else:
        # Minimal test/diagnostic boxes still obey nuScenes wlh semantics:
        # crop-local x/y/z correspond to length/width/height.
        half_extent = 0.5 * np.asarray(
            box_tmp.wlh, dtype=np.float64)[[1, 0, 2]] + float(offset)
        maxi = half_extent
        mini = -half_extent
    close = np.logical_and(
        point_columns[0, :] < maxi[0],
        point_columns[0, :] > mini[0])
    close = np.logical_and(close, point_columns[1, :] < maxi[1])
    close = np.logical_and(close, point_columns[1, :] > mini[1])
    if not bool(ignore_z):
        close = np.logical_and(close, point_columns[2, :] < maxi[2])
        close = np.logical_and(close, point_columns[2, :] > mini[2])
    return close
