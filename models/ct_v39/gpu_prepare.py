"""v39 设备端裁剪、无状态采样和空间点池；不读取 GT 决定在线输入。"""
import torch
from models.ct_v38.observation import box_corners_xyz


def inside(points, boxes, size, scale=1., offset=0.):
    delta = points - boxes[..., None, :3]
    c, s = boxes[..., 3].cos()[..., None], boxes[..., 3].sin()[..., None]
    x, y, z = delta.unbind(-1)
    local = torch.stack((c*x+s*y, -s*x+c*y, z), -1)
    return (local.abs() <= size[..., None, :] * (float(scale)/2.) + float(offset)).all(-1)


def canonical_points(points, box):
    delta = points - box[..., None, :3]
    c, s = box[..., 3].cos()[..., None], box[..., 3].sin()[..., None]
    x, y, z = delta.unbind(-1)
    return torch.stack((c*x+s*y, -s*x+c*y, z), -1)


def sample_indices(ids, mask, maximum, seed):
    if ids.ndim != 1 or mask.shape != ids.shape or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('sampling requires one-dimensional integer raw IDs and matching mask')
    if isinstance(maximum, bool) or int(maximum) != maximum or maximum < 1:
        raise ValueError('sampling maximum must be a positive integer')
    maximum = int(maximum)
    index = torch.nonzero(mask, as_tuple=False).flatten()
    if len(index) > maximum:
        # 所有乘数及 ID 在 int64 安全范围；同 CPU/GPU 的整数优先级。
        keys = ((ids[index] % 2147483647) * 1103515245 + int(seed) + 12345) & 2147483647
        selected = torch.argsort(keys, stable=True)[:maximum]
        index = index[selected].sort().values
    return index


def voxel_indices(xyz, ids, mask, maximum, seed, voxel=.5):
    if voxel <= 0:
        raise ValueError('voxel width must be positive')
    index = torch.nonzero(mask, as_tuple=False).flatten()
    if not len(index):
        return index
    cell = torch.floor(xyz[index] / voxel).long()
    _, inverse = torch.unique(cell, dim=0, sorted=True, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    sorted_ids = inverse[order]
    first = torch.cat((torch.ones(1, device=xyz.device, dtype=torch.bool), sorted_ids[1:] != sorted_ids[:-1]))
    representatives = index[order[first]].sort().values
    choose = sample_indices(ids[representatives], torch.ones_like(representatives, dtype=torch.bool), maximum, seed)
    return representatives[choose]


def pad_pool(values, maximum=None):
    """values = [(xyz, raw_ids), ...]；每行保持真实 raw 顺序，至少一个空占位。"""
    if not values:
        raise ValueError('point pool requires at least one row')
    count = max(1, max((len(ids) for _, ids in values), default=0)) if maximum is None else int(maximum)
    if count < 1:
        raise ValueError('point pool capacity must be positive')
    reference = values[0][0]
    xyz = reference.new_zeros((len(values), count, 3))
    ids = torch.full((len(values), count), -1, device=reference.device, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, (points, identity) in enumerate(values):
        n = min(count, len(identity))
        xyz[i, :n], ids[i, :n], valid[i, :n] = points[:n], identity[:n], True
    return dict(xyz=xyz.detach(), point_ids=ids, valid=valid)


def merge_pools(*pools):
    values = []
    for row in range(len(pools[0]['xyz'])):
        points = torch.cat([p['xyz'][row, p['valid'][row]] for p in pools])
        ids = torch.cat([p['point_ids'][row, p['valid'][row]] for p in pools])
        order = torch.argsort(ids, stable=True)
        sorted_ids = ids[order]
        first = torch.ones_like(sorted_ids, dtype=torch.bool)
        first[1:] = sorted_ids[1:] != sorted_ids[:-1]
        keep = order[first]
        values.append((points[keep], ids[keep]))
    return pad_pool(values)


def point_targets(xyz, valid, target, target_size):
    labels = inside(xyz, target, target_size).long().masked_fill(~valid, -1)
    centers = torch.cat((target[..., None, :3], box_corners_xyz(target, target_size)), -2)
    bc = torch.linalg.vector_norm(xyz[..., :, None, :] - centers[..., None, :, :], dim=-1)
    return labels, torch.where(valid[..., None], bc, 0.)
