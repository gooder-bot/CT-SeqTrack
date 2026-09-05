"""v27 日志专用获取漏斗：统一在原始 point ID 上计数，不回流模型。"""

import numpy as np


def _point_table(pc):
    if pc is None:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    if isinstance(pc, dict):
        xyz, ids = pc['points'], pc['point_ids']
    elif isinstance(pc, tuple) and len(pc) == 2:
        xyz, ids = pc
    else:
        if getattr(pc, 'point_ids', None) is None:
            raise ValueError('v27 diagnostics requires existing original point_ids')
        xyz, ids = np.asarray(pc.points)[:3].T, pc.point_ids
    xyz, ids = np.asarray(xyz), np.asarray(ids)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or ids.shape != (len(xyz),):
        raise ValueError('diagnostic coordinates/IDs must align')
    if ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('raw diagnostic point IDs must be nonnegative integers')
    return xyz[:, :3], ids.astype(np.int64, copy=False)


def _in_box(world_xyz, box, *, scale=1., offset=0., ignore_z=False,
            inclusive=False):
    if box is None:
        return np.zeros(len(world_xyz), dtype=bool)
    local = ((np.asarray(world_xyz, dtype=np.float64)
              - np.asarray(box.center, dtype=np.float64))
             @ np.asarray(box.rotation_matrix, dtype=np.float64))
    half = np.asarray(box.wlh, dtype=np.float64)[[1, 0, 2]] * (.5 * scale) + offset
    axes = 2 if ignore_z else 3
    return (np.abs(local[:, :axes]) <= half[:axes] if inclusive
            else np.abs(local[:, :axes]) < half[:axes]).all(axis=1)


def build_acquisition_diagnostics(
        global_pc, gt_box, anchor_box, baseline_pc,
        sampled_base_xyz, sampled_base_ids, endpoint_pc, tube_pc,
        corridor_pc, pool_xyz, pool_ids, prepool_xyz, prepool_ids,
        prepool_valid, source, *, support_boxes=(), margin_target=None):
    """Return loss-independent, exact-ID stage counts for one endpoint.

    ``global_pc`` coordinates and GT/support boxes are world-frame; all other
    point tables are canonical-frame.  Tables can be PointCloud objects,
    ``(xyz[N,3], ids[N])`` tuples, or dictionaries with points/point_ids.
    Targets are identified once on the global world table, then joined by ID.
    No coordinates, labels, sampler state, or input objects are modified.
    """
    del anchor_box  # Exact ID joins avoid a second coordinate transform.
    global_xyz, global_ids = _point_table(global_pc)
    if len(np.unique(global_ids)) != len(global_ids):
        raise ValueError('global diagnostic table requires unique raw IDs')
    finite = np.isfinite(global_xyz).all(1)
    global_set = set(global_ids[finite].tolist())
    target_global_mask = finite & _in_box(global_xyz, gt_box, inclusive=True)
    target_set = set(global_ids[target_global_mask].tolist())
    _, baseline_ids = _point_table(baseline_pc)
    base = set(baseline_ids.tolist())
    branch = {}
    for name, pc in (('endpoint', endpoint_pc), ('tube', tube_pc), ('corridor', corridor_pc)):
        _, ids = _point_table(pc)
        branch[name] = set(ids.tolist())
    expansion = set().union(*branch.values())

    def sampled_ids(xyz, ids, valid=None, *, unique=False):
        values = np.asarray(xyz)
        ids = np.asarray(ids)
        if values.ndim != 2 or values.shape[1] < 3 or ids.shape != (len(values),):
            raise ValueError('sampled diagnostic coordinates/IDs must align')
        if ids.dtype.kind not in 'iu':
            raise ValueError('sampled diagnostic IDs must be integers')
        keep = (ids >= 0) & np.isfinite(values[:, :3]).all(1)
        if valid is not None:
            valid = np.asarray(valid).reshape(-1)
            if valid.shape != ids.shape:
                raise ValueError('sample validity and IDs must align')
            keep &= valid > 0
        selected = ids[keep].astype(np.int64, copy=False)
        if unique and len(np.unique(selected)) != len(selected):
            raise RuntimeError('v27 extension diagnostic stage has duplicate raw IDs')
        return selected, keep

    base_sample_ids, _ = sampled_ids(sampled_base_xyz, sampled_base_ids)
    pool_id_rows, _ = sampled_ids(pool_xyz, pool_ids, unique=True)
    prepool_id_rows, prepool_keep = sampled_ids(
        prepool_xyz, prepool_ids, prepool_valid, unique=True)
    base_sample, pool, prepool = (set(v.tolist()) for v in
                                  (base_sample_ids, pool_id_rows, prepool_id_rows))
    if not (base | expansion | base_sample | pool | prepool).issubset(global_set):
        raise RuntimeError('acquisition diagnostics contains IDs absent from the global frame')
    if not base_sample.issubset(base):
        raise RuntimeError('sampled B0 IDs are not a subset of its raw crop')
    novel = expansion - base
    if pool != novel:
        raise RuntimeError('v27 novel pool differs from the actual raw support ID union')
    if not prepool.issubset(pool):
        raise RuntimeError('v27 prepool IDs are not a subset of the novel pool')
    sources = np.asarray(source).reshape(-1)
    if sources.shape != np.asarray(prepool_ids).shape:
        raise ValueError('prepool provenance and IDs must align')
    selected_sources = sources[prepool_keep].astype(np.int64, copy=False)
    source_by_id = {}
    for name, bit in (('endpoint', 1), ('tube', 2), ('corridor', 4)):
        for point_id in branch[name]:
            source_by_id[point_id] = source_by_id.get(point_id, 0) | bit
    if any(int(bit) != source_by_id[int(pid)] for pid, bit in zip(prepool_id_rows, selected_sources)):
        raise RuntimeError('v27 prepool source bits differ from actual raw support membership')

    def count_targets(values):
        return len(values & target_set)

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.

    total_targets = len(target_set)
    global_novel_targets = target_set - base
    result = dict(acquisition_schema_version='ct_acquisition.v4',
                  global_target_count_exact=total_targets,
                  global_target_count_label=total_targets,
                  global_raw_point_count=len(global_set),
                  global_novel_target_count=len(global_novel_targets),
                  target_visible=int(total_targets > 0),
                  global_target_denominator_valid=int(total_targets > 0),
                  global_novel_target_denominator_valid=int(bool(global_novel_targets)),
                  base_target_count=count_targets(base),
                  base_raw_target_count=count_targets(base),
                  base_sampled_target_count=count_targets(base_sample),
                  base_raw_point_count=len(base),
                  base_sampled_point_count=len(base_sample),
                  base_sampled_slot_count=len(base_sample_ids),
                  expansion_target_count=count_targets(expansion),
                  support_union_target_count=count_targets(expansion),
                  support_union_raw_point_count=len(expansion),
                  support_union_background_count=len(expansion - target_set),
                  pool_target_count=count_targets(pool), extension_pool_count=len(pool),
                  pool_background_count=len(pool - target_set),
                  sampled_target_count=count_targets(prepool), sampled_count=len(prepool),
                  sampled_background_count=len(prepool - target_set),
                  raw_target_bearing=int(bool(expansion & target_set)),
                  novel_target_bearing=int(bool(novel & target_set)),
                  pool_target_bearing=int(bool(pool & target_set)),
                  prepool_target_bearing=int(bool(prepool & target_set)),
                  corridor_valid=int(corridor_pc is not None))
    for prefix, values in (('support_raw', expansion), ('support_novel', novel),
                           ('novel_pool', pool), ('prepool', prepool)):
        result.update({f'{prefix}_target_count': count_targets(values),
                       f'{prefix}_point_count': len(values),
                       f'{prefix}_background_count': len(values - target_set)})
    for name, values in branch.items():
        missing = values - base
        result.update({f'{name}_raw_target_count': count_targets(values),
                       f'{name}_raw_point_count': len(values),
                       f'{name}_support_raw_target_count': count_targets(values),
                       f'{name}_support_raw_point_count': len(values),
                       f'{name}_support_raw_background_count': len(values - target_set),
                       f'{name}_support_novel_target_count': count_targets(missing),
                       f'{name}_support_novel_point_count': len(missing),
                       f'{name}_support_novel_background_count': len(missing - target_set),
                       f'{name}_raw_target_bearing': int(bool(values & target_set)),
                       f'{name}_novel_target_bearing': int(bool(missing & target_set)),
                       f'pool_{name}_target_count': count_targets(pool & values),
                       f'sampled_{name}_target_count': count_targets(prepool & values),
                       f'pool_{name}_point_count': len(pool & values),
                       f'sampled_{name}_point_count': len(prepool & values)})
    result.update(base_raw_target_recall=ratio(count_targets(base), total_targets),
                  support_raw_target_recall=ratio(count_targets(expansion), total_targets),
                  support_novel_target_recall=ratio(count_targets(novel), len(global_novel_targets)),
                  pool_target_recall=ratio(count_targets(pool), len(global_novel_targets)),
                  prepool_target_recall=ratio(count_targets(prepool), len(global_novel_targets)),
                  prepool_target_retention=ratio(count_targets(prepool), count_targets(pool)),
                  prepool_target_retention_denominator_valid=int(count_targets(pool) > 0),
                  pool_target_fraction=ratio(count_targets(pool), len(pool)),
                  prepool_target_fraction=ratio(count_targets(prepool), len(prepool)))
    masks_xy, masks_xyz = [], []
    for spec in support_boxes:
        if spec is None:
            continue
        box, scale, offset = spec if isinstance(spec, tuple) else (spec, 1., 0.)
        masks_xy.append(_in_box(global_xyz, box, scale=scale, offset=offset, ignore_z=True))
        masks_xyz.append(_in_box(global_xyz, box, scale=scale, offset=offset))
    if masks_xy:
        xy_count = int(np.sum(np.logical_or.reduce(masks_xy) & target_global_mask))
        xyz_count = int(np.sum(np.logical_or.reduce(masks_xyz) & target_global_mask))
    else:
        xy_count = xyz_count = count_targets(expansion)
    result.update(support_xy_target_count=xy_count, support_xyz_target_count=xyz_count,
                  support_z_clip_target_count=max(xy_count - xyz_count, 0))
    if margin_target is not None:
        for key in ('valid', 'reason', 'global_novel_target_count',
                    'max_reachable_target_count', 'selected_target_count',
                    'selected_background_count'):
            if key in margin_target:
                result[f'margin_label_{key}'] = margin_target[key]
    return result
