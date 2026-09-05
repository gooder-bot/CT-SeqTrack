"""v27 causal B1 primitives and isolated acquisition-label geometry.

Only ``acquisition_margin_grid_target`` accepts target labels.  Its outputs
are loss/diagnostic targets and must never replace actual acquired supports.
"""

import hashlib
import math

import numpy as np


def projected_box_half_extents(wlh, object_yaw, support_yaw):
    size = np.asarray(wlh, dtype=np.float64).reshape(3)
    if not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('first-frame wlh must be finite and positive')
    delta = float(object_yaw) - float(support_yaw)
    if not math.isfinite(delta):
        raise ValueError('support/object yaw must be finite')
    c, s = abs(math.cos(delta)), abs(math.sin(delta))
    return np.asarray(((c * size[1] + s * size[0]) * .5,
                       (s * size[1] + c * size[0]) * .5,
                       size[2] * .5), dtype=np.float64)


def acquisition_axis(displacement_xy, fallback_yaw=0.0):
    vector = np.asarray(displacement_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(vector).all():
        raise ValueError('acquisition displacement must be finite')
    norm = float(np.linalg.norm(vector))
    yaw = math.atan2(vector[1], vector[0]) if norm > 1e-6 else float(fallback_yaw)
    return np.asarray((math.cos(yaw), math.sin(yaw))), yaw, norm


def _yaw(box):
    return float(box.orientation.radians * box.orientation.axis[-1])


def build_b1_input_arrays(history_boxes, delta_t, valid_mask, *,
                          history_quality=None, recursive_age=0.,
                          first_frame_wlh=None, degrees=False,
                          time_scale=.5):
    """Build one unbatched causal, planar input; no current GT is accepted.

    Quality rows are [raw count, predicted foreground probability,
    segmentation entropy in nats, metadata-valid].  Output angles follow the
    legacy ``degrees`` flag; v27 formal callers use radians consistently.
    """
    boxes = list(history_boxes)
    if len(boxes) != 3:
        raise ValueError('v27 B1 requires exactly three history boxes')
    gaps = np.asarray(delta_t, dtype=np.float32).reshape(3)
    valid = np.asarray(valid_mask, dtype=np.float32).reshape(3)
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise ValueError('B1 effective intervals must be finite and positive')
    if not np.isfinite(valid).all() or np.any((valid != 0) & (valid != 1)):
        raise ValueError('B1 history validity must be binary')
    yaw = _yaw(boxes[0])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, s, 0.), (-s, c, 0.), (0., 0., 1.)))
    anchor = np.asarray(boxes[0].center, dtype=np.float64)
    rows = []
    for box in boxes:
        local = rotation @ (np.asarray(box.center, dtype=np.float64) - anchor)
        angle = (_yaw(box) - yaw + math.pi) % (2 * math.pi) - math.pi
        rows.append(np.r_[local, math.degrees(angle) if degrees else angle])
    quality = (np.zeros((3, 4), dtype=np.float64) if history_quality is None
               else np.asarray(history_quality, dtype=np.float64).reshape(3, 4).copy())
    if not np.isfinite(quality).all():
        raise ValueError('historical quality must be finite')
    quality_valid = ((quality[:, 3] > .5) & (valid > 0)).astype(np.float64)
    quality[:, 0] = np.clip(np.log1p(np.maximum(quality[:, 0], 0.)) / np.log(1025.), 0., 2.)
    quality[:, 1] = np.clip(quality[:, 1], 0., 1.)
    quality[:, 2] = np.clip(quality[:, 2] / np.log(2.), 0., 1.)
    quality[:, :3] *= quality_valid[:, None]
    quality[:, 3] = quality_valid
    size = np.asarray(boxes[0].wlh if first_frame_wlh is None
                      else first_frame_wlh, dtype=np.float64).reshape(3)
    if not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('first-frame dimensions must be finite and positive')
    pair_valid = valid[:-1] * valid[1:]
    nominal = float(np.sum(gaps[1:] * pair_valid) / max(pair_valid.sum(), 1.))
    ratio = float(gaps[0]) / max(nominal, 1e-3)
    if not math.isfinite(float(recursive_age)) or float(recursive_age) < 0:
        raise ValueError('recursive age must be finite and non-negative')
    features = np.r_[quality.reshape(-1),
                     np.log1p(min(float(recursive_age), 64.)) / np.log(65.),
                     np.log1p(size[[1, 0]]),
                     np.log1p(float(gaps[0]) / max(float(time_scale), 1e-3)),
                     np.log1p(ratio)].astype(np.float32)
    result = dict(ref_boxs=np.asarray(rows, dtype=np.float32), delta_t=gaps,
                  valid_mask=valid, current_delta_t=np.float32(gaps[0]),
                  acquisition_features=features)
    if not np.isfinite(result['ref_boxs']).all():
        raise ValueError('B1 history boxes must be finite')
    return result


def b1_input_digest(inputs):
    digest = hashlib.sha256()
    for name in ('ref_boxs', 'delta_t', 'valid_mask', 'current_delta_t',
                 'acquisition_features'):
        value = np.ascontiguousarray(inputs[name], dtype='<f4')
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _strict_box_mask(points, center, axis, half):
    relative = points - np.asarray(center, dtype=np.float64)
    perpendicular = np.asarray((-axis[1], axis[0]))
    return ((np.abs(relative[:, :2] @ axis) < half[0])
            & (np.abs(relative[:, :2] @ perpendicular) < half[1])
            & (np.abs(relative[:, 2]) < half[2]))


def acquisition_margin_grid_target(points_xyz, point_ids, target_mask,
                                   baseline_ids, *, anchor_center,
                                   endpoint_center, object_wlh, object_yaw,
                                   corridor_box=None, coverage=.90, grid_size=9):
    """Select a loss-only 9x9 margin target on one maximum-support point table.

    All positions and yaw use the same frame.  No RNG, crop object, model, or
    sampler is called.  Ties prefer fewer novel background points, more novel
    target points, less normalized margin, then deterministic grid indices.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    ids = np.asarray(point_ids)
    labels = np.asarray(target_mask, dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or ids.shape != (len(points),) or len(labels) != len(points):
        raise ValueError('label points, ids and targets must align')
    if ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
        raise ValueError('label table requires unique integer raw point IDs')
    if not 0 < float(coverage) <= 1 or int(grid_size) != 9:
        raise ValueError('v27 labels require a 9x9 grid and valid coverage')
    anchor = np.asarray(anchor_center, dtype=np.float64).reshape(3)
    endpoint = np.asarray(endpoint_center, dtype=np.float64).reshape(3)
    axis, yaw, distance = acquisition_axis(endpoint[:2] - anchor[:2], object_yaw)
    half = projected_box_half_extents(object_wlh, object_yaw, yaw)
    center = (anchor + endpoint) * .5
    finite = np.isfinite(points).all(axis=1)
    novel = ~np.isin(ids, np.asarray(baseline_ids)) & finite
    global_targets = int(np.sum(novel & labels))
    corridor = np.zeros(len(points), dtype=bool)
    if corridor_box is not None:
        corridor_yaw = _yaw(corridor_box)
        corridor_axis = np.asarray((math.cos(corridor_yaw), math.sin(corridor_yaw)))
        corridor_half = np.asarray(corridor_box.wlh, dtype=np.float64)[[1, 0, 2]] * .5
        corridor = _strict_box_mask(points, corridor_box.center, corridor_axis, corridor_half)
    maximum = _strict_box_mask(points, center, axis, half + np.asarray((distance * .5 + 6., 3., 0.)))
    maximum |= _strict_box_mask(points, endpoint, axis, half + np.asarray((6., 3., 0.)))
    table_mask = novel & (maximum | corridor)
    table = points[table_mask]
    table_targets = labels[table_mask]
    reachable = int(table_targets.sum())
    result = dict(target_margin=np.asarray((2., 1.), dtype=np.float32),
                  valid=True, reason='no_novel_target',
                  global_novel_target_count=global_targets,
                  max_reachable_target_count=reachable,
                  selected_target_count=0, selected_background_count=0,
                  grid_index=np.asarray((0, 0), dtype=np.int64))
    if global_targets == 0:
        return result
    if reachable == 0:
        result.update(valid=False, reason='outside_maximum_support')
        return result
    parallel, perpendicular = np.meshgrid(np.linspace(2., 6., 9),
                                         np.linspace(1., 3., 9), indexing='ij')
    margins = np.stack((parallel.ravel(), perpendicular.ravel()), axis=1)
    target_counts = np.zeros(81, dtype=np.int64)
    background_counts = np.zeros(81, dtype=np.int64)
    perpendicular_axis = np.asarray((-axis[1], axis[0]))
    corridor_table = corridor[table_mask]
    # Chunk only the boolean membership matrix, never duplicate point clouds.
    for start in range(0, len(table), 8192):
        stop = start + 8192
        chunk = table[start:stop]
        def membership(box_center, parallel_base):
            relative = chunk - box_center
            x = np.abs(relative[:, :2] @ axis)
            y = np.abs(relative[:, :2] @ perpendicular_axis)
            z = np.abs(relative[:, 2])
            return ((x[None, :] < parallel_base + margins[:, 0:1])
                    & (y[None, :] < half[1] + margins[:, 1:2])
                    & (z[None, :] < half[2]))
        member = membership(center, half[0] + distance * .5)
        member |= membership(endpoint, half[0])
        member |= corridor_table[start:stop][None, :]
        truth = table_targets[start:stop]
        target_counts += np.sum(member & truth[None, :], axis=1)
        background_counts += np.sum(member & ~truth[None, :], axis=1)
    feasible = np.flatnonzero(target_counts >= math.ceil(float(coverage) * reachable))
    index = min(feasible.tolist(), key=lambda k: (
        int(background_counts[k]), -int(target_counts[k]),
        float((margins[k, 0] - 2.) / 4. + (margins[k, 1] - 1.) / 2.),
        k // 9, k % 9))
    result.update(target_margin=margins[index].astype(np.float32), reason='ok',
                  selected_target_count=int(target_counts[index]),
                  selected_background_count=int(background_counts[index]),
                  grid_index=np.asarray((index // 9, index % 9), dtype=np.int64))
    return result

