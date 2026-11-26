"""当前四帧观测的唯一 raw-ID 输入合同。"""
import torch
from utils.batch_materialization import materialization_cache, require_finite
from .observation import B0Observation


# 在线路径只接收已提交状态和当前/历史观测。未来标签即使出现在调用者的
# batch 中，也不会传给 B0/B1/B3、候选或提交决策。
ONLINE_FIELDS = frozenset((
    'points', 'point_valid', 'point_ids', 'history_boxes', 'history_valid',
    'history_pair_valid', 'history_times', 'current_dt', 'frame_times',
    'box_size', 'anchor_box', 'fallback_box', 'history_box_provenance',
    'recovery_seed_box', 'recovery_seed_valid',
    'history_is_initial', 'window_seed_offset', 'branch_id',
    'trusted_velocity', 'trusted_velocity_valid', 'trusted_velocity_age',
    'local_template_xyz', 'local_template_valid',
    'weak_age', 'supported_age',
    'previous_is_propagation', 'consecutive_propagations',
    'joint_rows', 'joint_motion_boxes', 'joint_motion_times', 'joint_motion_valid',
    'joint_motion_pairs', 'joint_motion_origin',
    'consecutive_untrusted_outputs', '_ct_materialization_cache',
))
HOST_GEOMETRY = '_ct_loss_geometry_host'
_GEOMETRY_FIELDS = ('target_box', 'box_size', 'target_box_size')
LOSS_FIELDS = frozenset((
    'target_box', 'target_box_size', 'history_target_boxes',
    'segmentation_labels', 'bc_targets', 'physical_displacement', 'physical_valid',
    'future_displacement', 'future_dt', 'future_valid', HOST_GEOMETRY,
    'joint_other_cars', 'joint_track_id',
))


def retain_geometry_sources(cpu, moved):
    if any(key not in cpu or not torch.is_tensor(cpu[key]) or cpu[key].device.type != 'cpu'
           or moved[key].device.type == 'cpu' or cpu[key].requires_grad for key in _GEOMETRY_FIELDS):
        return
    try:
        moved[HOST_GEOMETRY] = tuple((cpu[key], cpu[key]._version, moved[key], moved[key]._version)
                                    for key in _GEOMETRY_FIELDS)
    except RuntimeError:  # inference tensors 没有版本；按设备实况取值。
        return


def retained_geometry_sources(batch, boxes):
    saved = batch.get(HOST_GEOMETRY)
    if saved is not None:
        for key, (cpu, before, device, after) in zip(_GEOMETRY_FIELDS, saved):
            if (batch.get(key) is not device or device.dtype != boxes.dtype or device.device != boxes.device
                    or cpu._version != before or device._version != after):
                return None
        return [row[0] for row in saved]
    return None


JOINT_ROW_FIELDS = ('scan_xyz', 'scan_ids', 'template_xyz', 'pending_valid', 'pending_box',
                    'main_box', 'backup_valid', 'backup_box')


def online_batch(batch, *, joint_search=None):
    prepared = {key: value for key, value in batch.items() if key in ONLINE_FIELDS
                and (joint_search is not False or not key.startswith('joint_'))}
    if joint_search and not all(key in prepared for key in ONLINE_FIELDS if key.startswith('joint_')):
        raise ValueError('joint search requires scan rows and independent motion history')
    if 'joint_rows' in prepared:
        prepared['joint_rows'] = tuple({**{key: row[key] for key in JOINT_ROW_FIELDS},
                                       **({'sparse_scan': row['sparse_scan']} if 'sparse_scan' in row else {})}
                                       for row in prepared['joint_rows'])
        if len(prepared['joint_rows']) != len(prepared['points']):
            raise ValueError('joint scan rows must align with the batch')
    return canonicalize_batch(prepared)


def loss_batch(batch):
    return canonicalize_batch({key: value for key, value in batch.items()
                               if key in ONLINE_FIELDS or key in LOSS_FIELDS})


def canonicalize_batch(batch):
    """统一 raw-ID 有效性，防止 feature/监督使用不同的重复点分母。"""
    result = dict(batch)
    points = batch['points']
    if points.ndim != 4 or points.shape[1] != 4 or points.shape[-1] != 5:
        raise ValueError('v41 points must be [B,4,N,5]')
    if 'recovery_seed_box' in batch or 'recovery_seed_valid' in batch:
        if 'recovery_seed_box' not in batch or 'recovery_seed_valid' not in batch:
            raise ValueError('v41 recovery seed geometry and validity must be supplied together')
        seed, seed_valid = batch['recovery_seed_box'], batch['recovery_seed_valid']
        if (seed.shape != (len(points), 4) or seed_valid.shape != (len(points),)
                or seed_valid.dtype != torch.bool):
            raise ValueError('v41 recovery requires finite seed [B,4] and bool validity [B]')
        require_finite(batch, 'recovery_seed_box', seed, 'v41 recovery seed must be finite')
    cache = materialization_cache(batch)
    cached = None if cache is None else cache.canonical_values(batch)
    # 历史不存在时其点槽也不存在；先合并存在性再去重，使 B0/B1/损失
    # 使用同一真实测量集合，而非仅由 observation 在内部额外过滤。
    if cached is None:
        valid = B0Observation.measurement_mask(batch)
        if valid.shape != points.shape[:-1]:
            raise ValueError('point_valid must align with points')
        clean = points.masked_fill(~valid[..., None], 0.)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError('real v41 point observations must be finite')
        result['point_valid'] = valid
        result['points'] = clean
        if cache is not None:
            cache.remember_canonical(batch, result)
    else:
        result['point_valid'] = cached['point_valid']
        result['points'] = cached['points']
    return result
