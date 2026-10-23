"""当前四帧观测的唯一 raw-ID 输入合同。"""
import torch
from utils.batch_materialization import materialization_cache
from .observation import B0Observation


def canonicalize_batch(batch):
    """统一 raw-ID 有效性，防止 feature/监督使用不同的重复点分母。"""
    result = dict(batch)
    points = batch['points']
    if points.ndim != 4 or points.shape[1] != 4 or points.shape[-1] != 5:
        raise ValueError('v38 points must be [B,4,N,5]')
    cache = materialization_cache(batch)
    cached = None if cache is None else cache.canonical_values(batch)
    # 历史不存在时其点槽也不存在；先合并存在性再去重，使 B0/B1/损失
    # 使用同一真实测量集合，而非仅由 observation 在内部额外过滤。
    if cached is None:
        valid = B0Observation.measurement_mask(batch)
        if valid.shape != points.shape[:-1]:
            raise ValueError('point_valid must align with points')
        if not bool(torch.isfinite(points.masked_select(valid[..., None].expand_as(points))).all()):
            raise ValueError('real v38 point observations must be finite')
        result['point_valid'] = valid
        result['points'] = points.masked_fill(~valid[..., None], 0.)
        if cache is not None:
            cache.remember_canonical(batch, result)
    else:
        result['point_valid'] = cached['point_valid']
        result['points'] = cached['points']
    return result
