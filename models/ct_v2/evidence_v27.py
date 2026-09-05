"""v27 新增测量的身份合同与选点前局部几何。

这里只处理当前 extension；B0 和预测历史仍由调用边界 detach。
"""

import torch
from torch import nn


def validate_point_ids(point_ids, valid, *, name, require_unique=False):
    """检查原始点 ID；padding=-1，同坐标的不同测量不合并。"""
    if point_ids is None or point_ids.shape != valid.shape:
        raise ValueError(f"{name} must have shape {tuple(valid.shape)}")
    if point_ids.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer original point IDs")
    ids = point_ids.detach().to(device=valid.device, dtype=torch.long)
    valid = valid.to(torch.bool)
    if bool(((ids < 0) & valid).any()):
        raise ValueError(f"{name} has a negative valid point ID")
    if bool(((ids != -1) & ~valid).any()):
        raise ValueError(f"{name} padding IDs must be -1")
    if require_unique:
        for row_ids, row_valid in zip(
                ids.reshape(-1, ids.shape[-1]),
                valid.reshape(-1, valid.shape[-1])):
            selected = row_ids[row_valid]
            if torch.unique(selected).numel() != selected.numel():
                raise ValueError(f"{name} valid original point IDs must be unique")
    return ids


def unique_point_mask(point_ids, valid, explicit_mask=None):
    """每个有效原始 ID 只保留一个槽；不改变 B0 的重复采样输入。"""
    valid = valid.to(torch.bool)
    result = torch.zeros_like(valid)
    rows_result = result.reshape(-1, result.shape[-1])
    for out, ids, mask in zip(
            rows_result, point_ids.reshape_as(rows_result),
            valid.reshape_as(rows_result)):
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        order = torch.argsort(ids.index_select(0, rows), stable=True)
        rows = rows.index_select(0, order)
        ordered_ids = ids.index_select(0, rows)
        first = torch.ones_like(ordered_ids, dtype=torch.bool)
        first[1:] = ordered_ids[1:] != ordered_ids[:-1]
        out[rows[first]] = True
    if explicit_mask is not None:
        if explicit_mask.shape != valid.shape:
            raise ValueError("unique point mask must be point aligned")
        explicit_mask = explicit_mask.detach().to(
            device=valid.device, dtype=torch.bool)
        if bool((explicit_mask & ~valid).any()):
            raise ValueError("unique point mask selected invalid points")
        for ids, raw, selected in zip(
                point_ids.reshape_as(rows_result),
                valid.reshape_as(rows_result),
                explicit_mask.reshape_as(rows_result)):
            original_ids = torch.unique(ids[raw], sorted=True)
            chosen_ids = torch.sort(ids[selected]).values
            if (original_ids.shape != chosen_ids.shape
                    or not torch.equal(original_ids, chosen_ids)):
                raise ValueError("unique point mask must represent every valid ID once")
        result = explicit_mask
    return result


def evidence_box_geometry(first_box_size_wlh, batch_size, reference):
    """nuScenes wlh 转显式 lwh，并给出跨类别一致的局部/投票尺度。"""
    if first_box_size_wlh is None or first_box_size_wlh.shape != (batch_size, 3):
        raise ValueError("v27 first_box_size_wlh must have shape [B,3]")
    size_wlh = first_box_size_wlh.detach().to(reference)
    if not bool(torch.isfinite(size_wlh).all()) or bool((size_wlh <= 0).any()):
        raise ValueError("v27 first-frame box dimensions must be finite and positive")
    size_lwh = size_wlh[:, [1, 0, 2]]
    local_radius = (0.5 * size_lwh[:, :2].amin(dim=1)).clamp(0.25, 1.0)
    vote_radius = (0.5 * torch.linalg.vector_norm(
        size_lwh[:, :2], dim=1) + 0.5).clamp_min(4.0)
    return size_lwh, local_radius, vote_radius


class ExtensionLocalGeometry(nn.Module):
    """k=16 半径约束邻域的单层 131->64->64 EdgeConv 残差。

    邻域按原始 ID 稳定打破距离并列；稀疏点保留自身特征，不补远点。
    按 query 小块计算距离，避免创建 B*768*768*131 的 edge tensor。
    """

    def __init__(self, neighbor_count=16, query_chunk_size=128):
        super().__init__()
        self.neighbor_count = int(neighbor_count)
        self.query_chunk_size = int(query_chunk_size)
        if self.neighbor_count <= 0 or self.query_chunk_size <= 0:
            raise ValueError("local geometry budgets must be positive")
        self.edge_mlp = nn.Sequential(
            nn.Linear(131, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))
        self.output_norm = nn.LayerNorm(64)

    def forward(self, features, xyz, valid, point_ids, radius):
        if features.shape[:-1] != xyz.shape[:-1] or features.shape[-1] != 64:
            raise ValueError("v27 local geometry features must be point-aligned 64d")
        output = torch.zeros_like(features)
        counts = torch.zeros_like(valid, dtype=torch.long)
        for batch_index in range(features.shape[0]):
            rows = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            rows = rows.index_select(0, torch.argsort(
                point_ids[batch_index].index_select(0, rows), stable=True))
            points = xyz[batch_index].index_select(0, rows)
            values = features[batch_index].index_select(0, rows)
            take = min(self.neighbor_count, rows.numel())
            row_chunks = []
            count_chunks = []
            for start in range(0, rows.numel(), self.query_chunk_size):
                stop = min(start + self.query_chunk_size, rows.numel())
                delta = points.unsqueeze(0) - points[start:stop].unsqueeze(1)
                distance_squared = delta.square().sum(dim=2)
                query_index = torch.arange(start, stop, device=rows.device)
                neighbor_mask = (
                    distance_squared <= radius[batch_index].square())
                neighbor_mask[torch.arange(stop - start, device=rows.device),
                              query_index] = False
                order = torch.argsort(distance_squared.masked_fill(
                    ~neighbor_mask, float("inf")), dim=1, stable=True)[:, :take]
                selected_valid = torch.gather(neighbor_mask, 1, order)
                neighbors = values[order]
                centers = values[start:stop].unsqueeze(1).expand(-1, take, -1)
                relative_xyz = torch.gather(
                    delta, 1, order.unsqueeze(2).expand(-1, -1, 3)) / (
                        radius[batch_index])
                edges = torch.cat((centers, neighbors - centers, relative_xyz), dim=2)
                encoded = self.edge_mlp(edges).masked_fill(
                    ~selected_valid.unsqueeze(2), float("-inf"))
                aggregated = encoded.amax(dim=1)
                aggregated = torch.where(
                    selected_valid.any(dim=1, keepdim=True), aggregated,
                    torch.zeros_like(aggregated))
                row_chunks.append(self.output_norm(values[start:stop] + aggregated))
                count_chunks.append(selected_valid.sum(dim=1))
            output[batch_index, rows] = torch.cat(row_chunks, dim=0)
            counts[batch_index, rows] = torch.cat(count_chunks, dim=0)
        return output, counts
