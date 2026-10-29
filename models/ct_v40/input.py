"""当前四帧观测的唯一 raw-ID 输入合同。"""
import torch
from utils.batch_materialization import materialization_cache
from .observation import B0Observation


# 在线路径只接收已提交状态和当前/历史观测。未来标签即使出现在调用者的
# batch 中，也不会传给 B0/B1/B3、候选或提交决策。
ONLINE_FIELDS = frozenset((
    'points', 'point_valid', 'point_ids', 'history_boxes', 'history_valid',
    'history_pair_valid', 'history_times', 'current_dt', 'frame_times',
    'box_size', 'anchor_box', 'fallback_box', 'history_box_provenance',
    'history_is_initial', 'window_seed_offset', 'branch_id',
    'trusted_velocity', 'trusted_velocity_valid', 'trusted_velocity_age',
    'previous_quality', 'weak_age', 'supported_age', 'previous_innovation',
    'previous_is_propagation', 'consecutive_propagations',
    'consecutive_untrusted_outputs', '_ct_materialization_cache',
))
LOSS_FIELDS = frozenset((
    'target_box', 'target_box_size', 'history_target_boxes',
    'segmentation_labels', 'bc_targets', 'physical_displacement', 'physical_valid',
    'acquisition_target', 'acquisition_valid', 'acquisition_demand',
    'future_displacement', 'future_dt', 'future_valid',
))


def online_batch(batch):
    return canonicalize_batch({key: value for key, value in batch.items() if key in ONLINE_FIELDS})


def loss_batch(batch):
    return canonicalize_batch({key: value for key, value in batch.items()
                               if key in ONLINE_FIELDS or key in LOSS_FIELDS})


def canonicalize_batch(batch):
    """统一 raw-ID 有效性，防止 feature/监督使用不同的重复点分母。"""
    result = dict(batch)
    points = batch['points']
    if points.ndim != 4 or points.shape[1] != 4 or points.shape[-1] != 5:
        raise ValueError('v40 points must be [B,4,N,5]')
    cache = materialization_cache(batch)
    cached = None if cache is None else cache.canonical_values(batch)
    # 历史不存在时其点槽也不存在；先合并存在性再去重，使 B0/B1/损失
    # 使用同一真实测量集合，而非仅由 observation 在内部额外过滤。
    if cached is None:
        valid = B0Observation.measurement_mask(batch)
        if valid.shape != points.shape[:-1]:
            raise ValueError('point_valid must align with points')
        if not bool(torch.isfinite(points.masked_select(valid[..., None].expand_as(points))).all()):
            raise ValueError('real v40 point observations must be finite')
        result['point_valid'] = valid
        result['points'] = points.masked_fill(~valid[..., None], 0.)
        if cache is not None:
            cache.remember_canonical(batch, result)
    else:
        result['point_valid'] = cached['point_valid']
        result['points'] = cached['points']
    return result
