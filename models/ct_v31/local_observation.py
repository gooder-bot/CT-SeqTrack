"""v35：现有候选角点读取唯一真实点，保留 pre-vote 和几何梯度边界。"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .contracts import EvidenceHypotheses, ObservationFeatures
from .observation import B0Observation, box_corners_xyz, unique_valid_mask


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

    def forward(self, observation: ObservationFeatures, evidence: EvidenceHypotheses,
                batch: Mapping[str, Tensor], seeds: Tensor, query_valid: Tensor,
                members: Tensor):
        """seeds是公开世界轴四候选；不读取vote、reliability、GT或memory点。"""
        points = batch['points']
        b = points.shape[0]
        if seeds.shape != (b, 4, 4) or query_valid.shape != (b, 4):
            raise ValueError('local observation requires four current candidates')
        ids = batch.get('point_ids')
        if ids is None or ids.shape != points.shape[:-1]:
            raise ValueError('v35 local observation requires raw point IDs')
        base_valid = B0Observation.measurement_mask(batch)[:, -1]
        extension_valid = evidence.point_valid.bool()
        if members.shape != (b, 3, extension_valid.shape[1]):
            raise ValueError('local observation mode members must align with extension points')
        pool_ids = torch.cat((ids[:, -1], evidence.point_ids), dim=1)
        # 同帧跨池去重，B0优先；正式B2仍保证extension-only。
        pool_valid = unique_valid_mask(torch.cat((base_valid, extension_valid), dim=1), pool_ids)
        xyz = torch.cat((points[:, -1, :, :3], evidence.point_xyz), dim=1).detach()
        features = torch.cat((observation.point_features[:, -1], evidence.point_features), dim=1)
        support = torch.cat((observation.foreground_probability[:, -1],
                             evidence.selected_identity_logits.sigmoid()), dim=1).detach()
        if (features.shape != (*pool_valid.shape, 64) or xyz.shape != (*pool_valid.shape, 3)
                or support.shape != pool_valid.shape):
            raise ValueError('local observation real point fields must align')
        if not all(bool(torch.isfinite(value[pool_valid]).all()) for value in (xyz, features, support)):
            raise ValueError('valid local observation measurements must be finite')
        xyz = torch.where(pool_valid[..., None], xyz, 0.)
        features = torch.where(pool_valid[..., None], features, 0.)
        support = torch.where(pool_valid, support, 0.)
        source_type = torch.cat((torch.zeros_like(base_valid), torch.ones_like(extension_valid)), dim=1)
        extension_allowed = torch.cat((extension_valid[:, None], members.detach().bool()), dim=1)
        allowed = torch.cat((base_valid[:, None].expand(-1, 4, -1), extension_allowed), dim=-1)
        allowed = allowed & pool_valid[:, None] & query_valid.detach().bool()[..., None]
        # 先按raw ID排，再稳定按距离排；等距邻居不取决于点槽排列。
        id_order = torch.argsort(pool_ids.masked_fill(~pool_valid, torch.iinfo(torch.long).max),
                                 dim=-1, stable=True)
        xyz = xyz.gather(1, id_order[..., None].expand(-1, -1, 3))
        features = features.gather(1, id_order[..., None].expand(-1, -1, 64))
        support = support.gather(1, id_order)
        source_type = source_type.gather(1, id_order).to(features)
        allowed = allowed.gather(2, id_order[:, None].expand(-1, 4, -1))
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
        selected_features = features[:, None, None].expand(-1, 4, 8, -1, -1).gather(
            3, selected[..., None].expand(-1, -1, -1, -1, 64))
        selected_relative = relative.gather(3, selected[..., None].expand(-1, -1, -1, -1, 3))
        selected_support = support[:, None, None].expand(-1, 4, 8, -1).gather(3, selected)
        selected_type = source_type[:, None, None].expand(-1, 4, 8, -1).gather(3, selected)
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
        return increment, dict(local_neighbor_count=neighbor_count.detach(),
            local_selected_count=selected_count.detach(), local_mean_support=mean_support.detach(),
            local_delta_norm=increment.detach().norm(dim=-1))
