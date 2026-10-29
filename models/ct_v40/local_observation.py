"""v40：q0 八角点读取当前唯一真实 B0 点，保留原几何梯度边界。"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .contracts import ObservationFeatures
from .observation import B0Observation, box_corners_xyz


class LocalObservationReader(nn.Module):
    """八角点、LWH归一化半径1、每角最多16点；仅特征保持live。"""

    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(69, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(130, 64)
        for layer in (self.point_encoder[0], self.point_encoder[3]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, observation: ObservationFeatures, batch: Mapping[str, Tensor],
                seeds: Tensor, query_valid: Tensor):
        """只读取当前 B0 点；seeds 为公开世界轴的单个 q0。"""
        points = batch['points']
        b = points.shape[0]
        if seeds.shape != (b, 1, 4) or query_valid.shape != (b, 1):
            raise ValueError('local observation requires one q0 candidate')
        ids = batch.get('point_ids')
        if ids is None or ids.shape != points.shape[:-1]:
            raise ValueError('v40 local observation requires raw point IDs')
        pool_valid = B0Observation.measurement_mask(batch)[:, -1]
        pool_ids = ids[:, -1]
        xyz = points[:, -1, :, :3].detach()
        features = observation.point_features[:, -1]
        support = observation.foreground_probability[:, -1].detach()
        if (features.shape != (*pool_valid.shape, 64) or xyz.shape != (*pool_valid.shape, 3)
                or support.shape != pool_valid.shape):
            raise ValueError('local observation real point fields must align')
        if not all(bool(torch.isfinite(value[pool_valid]).all()) for value in (xyz, features, support)):
            raise ValueError('valid local observation measurements must be finite')
        xyz = torch.where(pool_valid[..., None], xyz, 0.)
        features = torch.where(pool_valid[..., None], features, 0.)
        support = torch.where(pool_valid, support, 0.)
        # 保留原69维输入中的来源通道；当前全部是 B0，因此恒为0。
        source_type = torch.zeros_like(pool_valid)
        allowed = pool_valid[:, None] & query_valid.detach().bool()[..., None]
        # 先按raw ID排，再稳定按距离排；等距邻居不取决于点槽排列。
        id_order = torch.argsort(pool_ids.masked_fill(~pool_valid, torch.iinfo(torch.long).max),
                                 dim=-1, stable=True)
        xyz = xyz.gather(1, id_order[..., None].expand(-1, -1, 3))
        features = features.gather(1, id_order[..., None].expand(-1, -1, 64))
        support = support.gather(1, id_order)
        source_type = source_type.gather(1, id_order).to(features)
        allowed = allowed.gather(2, id_order[:, None].expand(-1, 1, -1))
        size = batch['box_size'].detach().to(points)
        if size.shape != (b, 3) or not bool(torch.isfinite(size).all()) or not bool((size > 0).all()):
            raise ValueError('local observation requires positive first-frame L/W/H')
        # 新增几何分支全detach；旧q0角点query仍按原图向coarse反传。
        seed = seeds.detach()
        seed = torch.where(query_valid[..., None], seed, 0.)
        corners = box_corners_xyz(seed, size[:, None])
        delta = xyz[:, None, None] - corners[..., None, :]
        sine = seed[..., 3, None, None].sin()
        cosine = seed[..., 3, None, None].cos()
        x, y, z = delta.unbind(-1)
        relative = torch.stack((cosine * x + sine * y, -sine * x + cosine * y, z), dim=-1)
        relative = relative / size[:, None, None, None]
        distance = relative.square().sum(-1)
        neighborhood = allowed[:, :, None] & (distance <= 1.)
        neighbor_count = neighborhood.sum(-1)
        mean_support = (support[:, None, None] * neighborhood).sum(-1) / neighbor_count.clamp_min(1)
        ordered = torch.argsort(distance.masked_fill(~neighborhood, torch.inf), dim=-1, stable=True)
        selected = ordered[..., :min(16, id_order.shape[1])]
        selected_valid = neighborhood.gather(-1, selected)
        selected_count = selected_valid.sum(-1)
        selected_features = features[:, None, None].expand(-1, 1, 8, -1, -1).gather(
            3, selected[..., None].expand(-1, -1, -1, -1, 64))
        selected_relative = relative.gather(3, selected[..., None].expand(-1, -1, -1, -1, 3))
        selected_support = support[:, None, None].expand(-1, 1, 8, -1).gather(3, selected)
        selected_type = source_type[:, None, None].expand(-1, 1, 8, -1).gather(3, selected)
        value = torch.cat((selected_features, selected_relative,
                           selected_support[..., None], selected_type[..., None]), dim=-1)
        value = torch.where(selected_valid[..., None], value, 0.)
        encoded = self.point_encoder(value)
        mean = torch.where(selected_valid[..., None], encoded, 0.).sum(-2)
        mean = mean / selected_count.clamp_min(1)[..., None]
        maximum = encoded.masked_fill(~selected_valid[..., None], -torch.inf).max(-2).values
        present = selected_count > 0
        maximum = torch.where(present[..., None], maximum, 0.)
        statistics = torch.stack((torch.log1p(neighbor_count.to(features)) / math.log(1281.),
                                  mean_support), dim=-1)
        increment = self.output(torch.cat((maximum, mean, statistics), dim=-1))
        increment = torch.where(present[..., None], increment, 0.)
        return increment, dict(local_neighbor_count=neighbor_count[:, 0].detach(),
            local_selected_count=selected_count[:, 0].detach(), local_mean_support=mean_support[:, 0].detach(),
            local_delta_norm=increment[:, 0].detach().norm(dim=-1))
