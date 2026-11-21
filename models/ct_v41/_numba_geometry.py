"""V41 CPU 几何内核。输入类型与阈值回退由 cpu_geometry 控制。"""
import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def inside_box(points, box, cosine, sine, limits):
    result = np.empty(len(points), dtype=np.bool_)
    for i in range(len(points)):
        dx = points[i, 0] - box[0]
        dy = points[i, 1] - box[1]
        dz = points[i, 2] - box[2]
        x = cosine * dx + sine * dy
        y = -sine * dx + cosine * dy
        result[i] = abs(x) <= limits[0] and abs(y) <= limits[1] and abs(dz) <= limits[2]
    return result


@njit(cache=True, fastmath=False)
def _footprint(pose, size):
    # 与 LocalYawBox 的底面顶点 [2,3,7,6] 同向；size 采用 LWH。
    polygon = np.empty((4, 2), np.float64)
    cosine, sine = math.cos(pose[3]), math.sin(pose[3])
    for i in range(4):
        x = size[0] * (.5 if i < 2 else -.5)
        y = size[1] * (.5 if i in (1, 2) else -.5)
        polygon[i, 0] = pose[0] + (cosine * x - sine * y)
        polygon[i, 1] = pose[1] + (sine * x + cosine * y)
    return polygon


@njit(cache=True, fastmath=False)
def _intersection_area(subject, clip):
    # 两矩形的凸交集最多八个顶点；留出边界重合时的重复点空间。
    incoming = np.empty((16, 2), np.float64)
    outgoing = np.empty_like(incoming)
    incoming[:4] = subject
    count = 4
    for j in range(4):
        if count == 0:
            return 0.
        ax, ay = clip[j]
        ex, ey = clip[(j + 1) % 4] - clip[j]
        px, py = incoming[count - 1]
        previous_inside = ex * (py - ay) - ey * (px - ax) >= -1e-12
        written = 0
        for k in range(count):
            x, y = incoming[k]
            inside = ex * (y - ay) - ey * (x - ax) >= -1e-12
            if inside != previous_inside:
                dx, dy = x - px, y - py
                denominator = dx * ey - dy * ex
                if abs(denominator) > 1e-14:
                    fraction = ((ax - px) * ey - (ay - py) * ex) / denominator
                    outgoing[written, 0] = px + fraction * dx
                    outgoing[written, 1] = py + fraction * dy
                    written += 1
            if inside:
                outgoing[written, 0] = x
                outgoing[written, 1] = y
                written += 1
            px, py, previous_inside = x, y, inside
        incoming, outgoing = outgoing, incoming
        count = written
    if count < 3:
        return 0.
    twice_area = 0.
    for i in range(count):
        j = (i + 1) % count
        twice_area += incoming[i, 0] * incoming[j, 1] - incoming[j, 0] * incoming[i, 1]
    return abs(twice_area) * .5


@njit(cache=True, fastmath=False, nogil=True)
def candidate_metrics(boxes, targets, sizes, target_sizes, skip):
    result = np.zeros((len(boxes), boxes.shape[1], 3), np.float64)
    for row in range(len(boxes)):
        target, size, target_size = targets[row], sizes[row], target_sizes[row]
        footprint = _footprint(target, target_size)
        volume = size[0] * size[1] * size[2] + target_size[0] * target_size[1] * target_size[2]
        for slot in range(boxes.shape[1]):
            if skip[row, slot]:
                continue
            pose = boxes[row, slot]
            area = _intersection_area(_footprint(pose, size), footprint)
            for mode in range(2):
                # exact 以 z 为中心，benchmark_compat 以 z 为顶面。
                shift = .5 if mode == 0 else 0.
                top = min(pose[2] + shift * size[2], target[2] + shift * target_size[2])
                bottom = max(pose[2] - (1. - shift) * size[2], target[2] - (1. - shift) * target_size[2])
                intersection = area * max(top - bottom, 0.)
                result[row, slot, mode] = min(max(intersection / (volume - intersection), 0.), 1.)
            dx, dy, dz = pose[:3] - target[:3]
            result[row, slot, 2] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return result


@njit(cache=True, fastmath=False, nogil=True)
def bounded_xy_groups(xy, xmin, ymin, stride, capacity):
    first_at = np.full(capacity, -1, dtype=np.int64)
    counts_at = np.zeros(capacity, dtype=np.int64)
    inverse = np.empty(len(xy), dtype=np.int64)
    groups = 0
    for i in range(len(xy)):
        key = (xy[i, 0] - xmin) * stride + (xy[i, 1] - ymin)
        inverse[i] = key
        if first_at[key] < 0:
            first_at[key] = i
            groups += 1
        counts_at[key] += 1
    first = np.empty(groups, dtype=np.int64)
    counts = np.empty(groups, dtype=np.int64)
    # 与 np.unique 一样按格键升序编号，保留首个输入点和每点的反向索引。
    group = 0
    for key in range(capacity):
        if first_at[key] >= 0:
            first[group] = first_at[key]
            counts[group] = counts_at[key]
            first_at[key] = group
            group += 1
    for i in range(len(inverse)):
        inverse[i] = first_at[inverse[i]]
    return first, inverse, counts


@njit(cache=True, fastmath=False, nogil=True)
def sequential_extrema(inverse, lows, highs, groups):
    lower = np.full(groups, np.inf, dtype=np.float32)
    upper = np.full(groups, -np.inf, dtype=np.float32)
    for i in range(len(inverse)):
        key = inverse[i]
        # 与 numpy.minimum/maximum.at 同序，平局取后值，保留 ±0 的符号。
        if lows[i] <= lower[key]:
            lower[key] = lows[i]
        if highs[i] >= upper[key]:
            upper[key] = highs[i]
    return lower, upper
