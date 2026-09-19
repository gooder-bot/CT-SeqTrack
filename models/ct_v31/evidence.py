"""v31 单一身份分数、唯一点预算和连续三维测量模式。

离散采样/聚类成员不反传；固定成员中的 XYZ 重心反传。decoder 的
跨假设 KV 使用 pre-vote 特征，禁止把 live vote 坐标拼到共享上下文。
"""
from __future__ import annotations

import torch
from torch import nn

from .contracts import EvidenceHypotheses


def _gather(value, indices):
    suffix = value.shape[2:]
    index = indices.clamp_min(0).reshape(*indices.shape, *((1,) * len(suffix)))
    return value.gather(1, index.expand(*indices.shape, *suffix))


@torch.no_grad()
def select_evidence_points(points, identity_logits, valid, point_ids, partition):
    """每分区 64 identity + 48 XY coverage + 16 hash；不足从另一分区借。

    返回 [B,256] 原池索引，无效为 -1。所有阶段排除已选原始 ID。
    """
    batch, count = valid.shape
    if count == 0:
        return torch.full((batch, 256), -1, dtype=torch.long, device=points.device)
    good = (valid.bool() & (point_ids >= 0) & torch.isfinite(points[..., :3]).all(-1)
            & torch.isfinite(identity_logits) & ((partition == 0) | (partition == 1)))
    order = torch.argsort(point_ids.masked_fill(~good, torch.iinfo(torch.long).max), stable=True)
    ordered_ids = point_ids.gather(1, order)
    unique = torch.cat((torch.ones(batch, 1, device=points.device, dtype=torch.bool),
                        ordered_ids[:, 1:] != ordered_ids[:, :-1]), 1)
    good = (good.gather(1, order) & unique).gather(1, torch.argsort(order, stable=True))
    indices = torch.arange(count, device=points.device)[None]
    xyz = torch.nan_to_num(points[..., :2].detach())
    score = identity_logits.detach()
    # 整数算子与 stable 排序可在 strict deterministic CUDA 下执行。
    hashed = ((point_ids ^ (point_ids >> 16)) * 1103515245 + 12345).remainder(2147483647)
    outputs = []
    already = torch.zeros_like(good)
    for part in (0, 1):
        eligible = good & (partition == part)
        ranked = order.gather(1, torch.argsort(score.masked_fill(~eligible, -torch.inf).gather(1, order),
                                              descending=True, stable=True))
        top = ranked[:, :min(64, count)]
        top_valid = eligible.gather(1, top)
        top = top.masked_fill(~top_valid, -1)
        if top.shape[1] < 64:
            top = torch.cat((top, top.new_full((batch, 64 - top.shape[1]), -1)), 1)
        chosen = ((indices[:, :, None] == top[:, None]) & (top[:, None] >= 0)).any(-1)
        nearest = torch.linalg.vector_norm(xyz[:, :, None] - _gather(xyz, top)[:, None], dim=-1).square()
        nearest = nearest.masked_fill(top[:, None] < 0, torch.inf).amin(-1)
        coverage = []
        for _ in range(48):
            available = eligible & ~chosen
            # 输入 raw ID 已固定 tie 顺序，避免排列改变覆盖点。
            ranked_distance = nearest.masked_fill(~available, -torch.inf).gather(1, order)
            pick = order.gather(1, ranked_distance.argmax(1, keepdim=True)).squeeze(1)
            okay = available.any(1)
            pick = pick.masked_fill(~okay, -1)
            coverage.append(pick)
            chosen |= (indices == pick[:, None]) & okay[:, None]
            delta = xyz - _gather(xyz, pick[:, None]).squeeze(1)[:, None]
            nearest = torch.minimum(nearest, delta.square().sum(-1))
        available = eligible & ~chosen
        hash_rank = order.gather(1, torch.argsort(hashed.masked_fill(~available, torch.iinfo(torch.long).max)
                                                .gather(1, order), stable=True))
        exploration = hash_rank[:, :min(16, count)]
        exploration = exploration.masked_fill(~available.gather(1, exploration), -1)
        if exploration.shape[1] < 16:
            exploration = torch.cat((exploration, exploration.new_full((batch, 16 - exploration.shape[1]), -1)), 1)
        result = torch.cat((top, torch.stack(coverage, 1), exploration), 1)
        outputs.append(result)
        already |= ((indices[:, :, None] == result[:, None]) & (result[:, None] >= 0)).any(-1)
    selected = torch.cat(outputs, 1)
    spare_good = good & ~already
    spare = order.gather(1, torch.argsort(score.masked_fill(~spare_good, -torch.inf).gather(1, order),
                                        descending=True, stable=True))
    holes = selected < 0
    position = holes.long().cumsum(1) - 1
    replacement = spare.gather(1, position.clamp(0, count - 1))
    can_borrow = holes & (position < spare_good.sum(1, keepdim=True))
    return torch.where(can_borrow, replacement, selected)


@torch.no_grad()
def _mode_members(votes, weights, valid, point_ids):
    batch, count, _ = votes.shape
    good = valid & torch.isfinite(votes).all(-1) & torch.isfinite(weights) & (weights > 0) & (point_ids >= 0)
    order = torch.argsort(point_ids.masked_fill(~good, torch.iinfo(torch.long).max), stable=True)
    inverse = torch.argsort(order, stable=True)
    ids = point_ids.gather(1, order)
    unique = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=votes.device), ids[:, 1:] != ids[:, :-1]), 1)
    good = good.gather(1, order) & unique
    pts = _gather(torch.nan_to_num(votes.detach()), order)
    weight = weights.detach().gather(1, order).masked_fill(~good, 0.)
    xy = pts[..., :2]
    distances = torch.linalg.vector_norm(xy[:, :, None] - xy[:, None], dim=-1)
    mass = ((distances <= 1.) * weight[:, None]).sum(-1)
    remaining = good.clone()
    members, hubers, valids = [], [], []
    for _ in range(3):
        seed = mass.masked_fill(~remaining, -torch.inf).argmax(1)
        active = remaining.any(1)
        center = _gather(xy, seed[:, None]).squeeze(1)
        seed_distance = distances.gather(1, seed[:, None, None].expand(-1, 1, count)).squeeze(1)
        remaining &= seed_distance > .75
        for _ in range(3):
            distance = torch.linalg.vector_norm(xy - center[:, None], dim=-1)
            inside = good & active[:, None] & (distance <= 1.)
            robust = weight * inside * (.5 / distance.clamp_min(.5))
            center = (robust[..., None] * xy).sum(1) / robust.sum(1).clamp_min(1e-8)[:, None]
        distance = torch.linalg.vector_norm(xy - center[:, None], dim=-1)
        inside = good & active[:, None] & (distance <= 1.)
        huber = .5 / distance.clamp_min(.5)
        members.append(inside.gather(1, inverse))
        hubers.append(huber.gather(1, inverse))
        valids.append(active & inside.any(1))
    return torch.stack(members, 1), torch.stack(hubers, 1), torch.stack(valids, 1)


def build_live_modes(votes, weights, valid, point_ids):
    """返回 live XYZ center，detached 成员/XY covariance，三个固定槽。"""
    members, huber, mode_valid = _mode_members(votes, weights, valid, point_ids)
    clean_votes = torch.where(valid[..., None], votes, torch.zeros_like(votes))
    clean_weights = torch.where(valid, weights, torch.zeros_like(weights))
    robust = clean_weights[:, None] * members * huber
    denom = robust.sum(-1).clamp_min(1e-8)
    centers = (robust[..., None] * clean_votes[:, None]).sum(2) / denom[..., None]
    centers = torch.where(mode_valid[..., None], centers, torch.zeros_like(centers))
    with torch.no_grad():
        delta = clean_votes[:, None, :, :2].detach() - centers[:, :, None, :2].detach()
        covariance = (robust.detach()[..., None, None] * delta[..., :, None] * delta[..., None, :]).sum(2)
        covariance /= denom.detach()[..., None, None]
    return centers, mode_valid, members, covariance


class RawLocalGeometry(nn.Module):
    """旧 k=16 半径邻域的原始几何描述子，索引/坐标全部停止梯度。

    仅编码邻域相对原始 XYZ、距离及两个原始属性差，不读取预测 vote。
    固定 query chunk 控制内存，孤立点提供零局部描述。
    """
    def __init__(self, neighbors=16, query_chunk_size=128):
        super().__init__()
        self.neighbors = int(neighbors)
        self.query_chunk_size = int(query_chunk_size)
        self.projection = nn.Sequential(nn.Linear(6, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64))

    def forward(self, points, valid, point_ids, box_size):
        batch, count, _ = points.shape
        radius = (.5 * box_size[:, :2].amin(-1)).clamp(.25, 1.)
        with torch.no_grad():
            raw = points.detach()
            order = torch.argsort(point_ids.masked_fill(~valid, torch.iinfo(torch.long).max), stable=True)
            ordered = _gather(raw, order)
            ordered_valid = valid.gather(1, order)
            ordered_ids = point_ids.gather(1, order)
        chunks = []
        for start in range(0, count, self.query_chunk_size):
            stop = min(count, start + self.query_chunk_size)
            with torch.no_grad():
                delta = ordered[:, None, :, :3] - raw[:, start:stop, None, :3]
                distance2 = delta.square().sum(-1)
                neighbor_valid = (ordered_valid[:, None] & valid[:, start:stop, None]
                    & (ordered_ids[:, None] != point_ids[:, start:stop, None])
                    & (distance2 <= radius[:, None, None].square()))
                selected = torch.argsort(distance2.masked_fill(~neighbor_valid, torch.inf), stable=True)[..., :self.neighbors]
                selected_valid = neighbor_valid.gather(2, selected)
                relative = delta.gather(2, selected[..., None].expand(-1, -1, -1, 3)) / radius[:, None, None, None]
                distance = relative.square().sum(-1, keepdim=True).sqrt()
                attribute_delta = ordered[:, None, :, 3:] - raw[:, start:stop, None, 3:]
                attributes = attribute_delta.gather(2, selected[..., None].expand(-1, -1, -1, 2))
                descriptor = torch.cat((relative, distance, attributes), -1)
                descriptor = torch.where(selected_valid[..., None], descriptor, 0.)
            encoded = self.projection(descriptor).masked_fill(~selected_valid[..., None], -torch.inf)
            pooled = encoded.max(2).values
            chunks.append(torch.where(selected_valid.any(-1)[..., None], pooled, 0.))
        return torch.cat(chunks, 1)


class B2IdentityEvidence(nn.Module):
    def __init__(self, config=None, feature_dim=64):
        super().__init__()
        if feature_dim != 64:
            raise ValueError("v31 evidence feature_dim is fixed at 64")
        self.point_encoder = nn.Sequential(nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
                                           nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU())
        self.memory_metadata = nn.Linear(8, 64)
        self.local_geometry = RawLocalGeometry()
        self.memory_attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.identity_norm = nn.LayerNorm(64)
        self.b0_summary = nn.Linear(64, 64)
        self.identity_head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.observation_attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.enrichment_norm = nn.LayerNorm(64)
        self.prior_projection = nn.Linear(128, 64)
        self.vote_head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        self.reliability_head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def _attention(module, query, key, valid):
        # 一个固定 zero token 保证全空样本不会产生 softmax(-inf)=NaN。
        zero = key.new_zeros((len(key), 1, key.shape[-1]))
        key = torch.cat((key, zero), 1)
        valid = torch.cat((valid, torch.ones(len(key), 1, dtype=torch.bool, device=key.device)), 1)
        return module(query, key, key, key_padding_mask=~valid, need_weights=False)[0]

    def forward(self, observation, batch, prior=None):
        points = batch["extension_points"]
        valid = batch["extension_valid"].bool()
        point_ids = batch["extension_ids"].long()
        partition = batch["extension_partition"].long()
        if points.shape[1:] != (768, 5):
            raise ValueError("v31 extension pool must have shape [B,768,5]")
        valid = valid & (point_ids >= 0) & torch.isfinite(points).all(-1)
        clean = torch.where(valid[..., None], points, torch.zeros_like(points))
        size = batch.get("box_size", points.new_ones((len(points), 3))).to(points).clamp_min(1e-3)
        normalized_points = torch.cat((clean[..., :3] / size[:, None], clean[..., 3:]), -1)
        raw_feature = self.point_encoder(normalized_points)
        raw_feature = raw_feature + self.local_geometry(clean, valid, point_ids, size)
        memory_points = batch["memory_points"]
        memory_valid = batch["memory_valid"].bool() & torch.isfinite(memory_points).all(-1)
        metadata = torch.nan_to_num(batch["memory_metadata"])
        # 年龄只用于身份上下文，不把秒数未经缩放送入线性层。
        metadata = torch.cat((metadata[..., :3], torch.log1p(metadata[..., 3:4].clamp_min(0)),
                              metadata[..., 4:]), -1)
        memory_features = self.point_encoder(torch.where(memory_valid[..., None], memory_points,
                                                         torch.zeros_like(memory_points)))
        memory_features = (memory_features + self.memory_metadata(metadata)) * memory_valid[..., None]
        source_shape = observation.point_features.shape
        point_features = observation.point_features.reshape(len(points), -1, 64)
        observed_valid = batch.get("point_valid")
        if observed_valid is not None and observed_valid.numel() == point_features.shape[0] * point_features.shape[1]:
            observed_valid = observed_valid.reshape(point_features.shape[:2]).bool()
        else:
            observed_valid = torch.isfinite(point_features).all(-1)
            observed_valid &= observation.current_valid[:, None]
        point_features = torch.where(observed_valid[..., None], point_features, torch.zeros_like(point_features))
        summary = point_features.sum(1) / observed_valid.sum(1).clamp_min(1)[:, None]
        memory_read = self._attention(self.memory_attention, raw_feature, memory_features, memory_valid)
        identity_feature = self.identity_norm(raw_feature + memory_read + self.b0_summary(summary)[:, None])
        identity_logits = self.identity_head(identity_feature).squeeze(-1).masked_fill(~valid, 0.)
        selected = select_evidence_points(clean, identity_logits, valid, point_ids, partition)
        selected_valid = selected >= 0
        selected_feature = _gather(identity_feature, selected)
        current_features = point_features.reshape(source_shape)[:, -1]
        current_valid = observed_valid.reshape(source_shape[:3])[:, -1]
        enrichment_keys = torch.cat((current_features, memory_features), 1)
        enrichment_valid = torch.cat((current_valid, memory_valid), 1)
        observation_read = self._attention(self.observation_attention, selected_feature,
                                            enrichment_keys, enrichment_valid)
        prior_feature = (points.new_zeros((len(points), 128)) if prior is None else prior.feature)
        enriched = self.enrichment_norm(selected_feature + observation_read
                                       + self.prior_projection(prior_feature)[:, None]) * selected_valid[..., None]
        selected_xyz = _gather(clean[..., :3], selected) * selected_valid[..., None]
        selected_identity = _gather(identity_logits, selected).masked_fill(~selected_valid, 0.)
        selected_ids = _gather(point_ids, selected).masked_fill(~selected_valid, -1)
        # 无 tanh/固定半径截断；尺度只作预条件，预测仍可到达远处测量中心。
        votes = (selected_xyz + self.vote_head(enriched) * size[:, None]) * selected_valid[..., None]
        # sibling head 不能读取 live votes；q0 KV 的梯度不会旁路进入 vote_head。
        reliability = self.reliability_head(enriched).squeeze(-1).masked_fill(~selected_valid, 0.)
        weights = selected_identity.sigmoid() * reliability.sigmoid() * selected_valid
        centers, mode_valid, members, covariance = build_live_modes(votes, weights, selected_valid, selected_ids)
        return EvidenceHypotheses(centers, mode_valid, members, covariance, enriched, selected_xyz,
            selected_valid, selected_ids, selected, identity_logits, selected_identity, votes,
            reliability, memory_features, memory_valid)
