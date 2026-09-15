"""v30 多模式证据：局部加权支持选种子，几何停止梯度，质量学习。"""
from __future__ import annotations

import torch
from torch import nn

from models.ct_v2.pipeline_contracts import EvidenceModeSet


@torch.no_grad()
def extract_mode_geometry(votes, weights, valid, observation_xy,
                          identity_margin, point_ids, max_modes=3):
    """1m 加权支持排序、0.75m NMS、三次 Huber 重心；不含点数比乘子。"""
    if max_modes != 3:
        raise ValueError("v30 fixes the evidence budget at three modes")
    votes, weights = votes.detach(), weights.detach()
    batch, count, _ = votes.shape
    if count == 0:
        raise ValueError('v30 requires fixed padded point slots')
    good = (valid.detach().bool() & torch.isfinite(votes).all(-1)
            & torch.isfinite(weights) & (weights > 0) & (point_ids >= 0))
    # 真实 ID 先排序，精确并列由固定槽 first-max 决定；无 CPU 候选循环。
    order = torch.argsort(point_ids.detach().masked_fill(~good, torch.iinfo(torch.long).max), stable=True)
    inverse = torch.argsort(order, stable=True)
    ids = point_ids.detach().gather(1, order)
    good = good.gather(1, order)
    unique = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=votes.device),
                        ids[:, 1:] != ids[:, :-1]), 1)
    good &= unique
    pts = torch.nan_to_num(votes, nan=0., posinf=0., neginf=0.).gather(1, order[..., None].expand(-1, -1, 2))
    w = torch.nan_to_num(weights, nan=0., posinf=0., neginf=0.).gather(1, order).clamp_min(0)
    w = w.masked_fill(~good, 0.)
    identity_values = torch.nan_to_num(identity_margin.detach(), nan=0., posinf=0., neginf=0.).gather(1, order)
    distances = torch.linalg.norm(pts[:, :, None, :] - pts[:, None, :, :], dim=-1)
    local_mass = ((distances <= 1.).to(w) * w[:, None, :]).sum(-1)
    remaining = good.clone()
    total_mass = w.sum(1).clamp_min(1e-6)
    collected = {key: [] for key in ('centers_xy', 'covariance_xy', 'valid', 'member_mask',
                                    'unique_count', 'mass', 'identity', 'targetness_mean',
                                    'support_score', 'seed_point_ids')}
    for _ in range(max_modes):
        seed = local_mass.masked_fill(~remaining, -torch.inf).max(1).indices
        seed_valid = remaining.any(1)
        seed_center = pts.gather(1, seed[:, None, None].expand(-1, 1, 2)).squeeze(1)
        seed_distances = distances.gather(1, seed[:, None, None].expand(-1, 1, count)).squeeze(1)
        remaining &= seed_distances > .75
        center = seed_center
        for _ in range(3):
            distance = torch.linalg.norm(pts - center[:, None], dim=-1)
            inlier = good & seed_valid[:, None] & (distance <= 1.)
            robust = w * inlier * (.5 / distance.clamp_min(.5))
            numerator = (robust[..., None] * pts).sum(1)
            center = numerator / robust.sum(1).clamp_min(1e-6)[:, None]
        inlier = good & seed_valid[:, None] & (torch.linalg.norm(pts - center[:, None], dim=-1) <= 1.)
        weighted = w * inlier
        mass = weighted.sum(1)
        mode_valid = seed_valid & (mass > 0)
        center = torch.where(mode_valid[:, None], center, observation_xy.detach())
        delta = pts - torch.nan_to_num(center, nan=0., posinf=0., neginf=0.)[:, None]
        covariance = (weighted[:, :, None, None] * delta[:, :, :, None]
                      * delta[:, :, None, :]).sum(1) / mass.clamp_min(1e-6)[:, None, None]
        count_unique = inlier.sum(1)
        values = dict(centers_xy=center, covariance_xy=covariance, valid=mode_valid,
            member_mask=inlier.gather(1, inverse), unique_count=count_unique, mass=mass,
            identity=(identity_values * inlier).sum(1) / count_unique.clamp_min(1),
            targetness_mean=mass / count_unique.clamp_min(1),
            support_score=mass / total_mass * torch.exp(-covariance.diagonal(dim1=-2, dim2=-1).sum(-1)),
            seed_point_ids=ids.gather(1, seed[:, None]).squeeze(1).masked_fill(~mode_valid, -1))
        for key, value in values.items():
            collected[key].append(value)
    return {key: torch.stack(value, 1) for key, value in collected.items()}


class ModeEvidenceBuilder(nn.Module):
    """模式质量只更新模式特征与本头；不驱动投票坐标/成员或 targetness。"""
    def __init__(self, mode_count=3):
        super().__init__()
        if int(mode_count) not in (1, 3):
            raise ValueError('v30 mode_count must be 1 or 3')
        self.mode_count = int(mode_count)
        self.quality_head = nn.Sequential(nn.Linear(75, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, votes, weights, valid, features, observation_xy,
                identity_margin, point_ids, b1_center_xy, b1_direction_xy,
                b1_sigma_parallel_perp, support_half_size_parallel_perp):
        geometry = extract_mode_geometry(votes, weights, valid, observation_xy,
                                         identity_margin, point_ids)
        pool_weight = (geometry['member_mask'].to(features)
                       * torch.nan_to_num(weights.detach()).clamp_min(0)[:, None, :])
        # 固定槽加权和，无重复 ID / scatter 原子归约。
        pooled = (pool_weight[..., None] * features[:, None]).sum(2)
        pooled = pooled / pool_weight.sum(2).clamp_min(1e-6)[..., None]
        cov = geometry['covariance_xy']
        direction = b1_direction_xy.detach()
        norm = torch.linalg.norm(direction, dim=-1, keepdim=True)
        fallback = torch.zeros_like(direction)
        fallback[:, 0] = 1
        direction = torch.where(norm > 1e-6, direction / norm.clamp_min(1e-6), fallback)
        perp = torch.stack((-direction[:, 1], direction[:, 0]), -1)
        delta = geometry['centers_xy'] - b1_center_xy.detach()[:, None]
        half = support_half_size_parallel_perp.detach().clamp_min(1e-6)
        normalized_center = torch.stack(((delta * direction[:, None]).sum(-1) / half[:, :1],
                                         (delta * perp[:, None]).sum(-1) / half[:, 1:]), -1)
        scalar = torch.stack((torch.log1p(geometry['mass']),
                              torch.log1p(geometry['unique_count'].to(features)),
                              geometry['targetness_mean'], geometry['identity'] / 2,
                              torch.log1p(cov[..., 0, 0].clamp_min(0)),
                              cov[..., 0, 1].sign() * torch.log1p(cov[..., 0, 1].abs()),
                              torch.log1p(cov[..., 1, 1].clamp_min(0))), -1)
        log_sigma = b1_sigma_parallel_perp.detach().clamp_min(1e-6).log()
        quality_input = torch.cat((pooled, scalar, normalized_center,
                                  log_sigma[:, None].expand(-1, 3, -1)), -1)
        logits = self.quality_head(torch.nan_to_num(quality_input, nan=0., posinf=0., neginf=0.)).squeeze(-1)
        quality = logits.sigmoid()
        # 模式按纯加权支持×紧致度排序，质量预测不参与排序/always 控制。
        with torch.no_grad():
            order = torch.argsort(geometry['seed_point_ids'].masked_fill(~geometry['valid'], torch.iinfo(torch.long).max), stable=True)
            ranking = geometry['support_score'].masked_fill(~geometry['valid'], -torch.inf)
            order = order.gather(1, torch.argsort(ranking.gather(1, order), descending=True, stable=True))
            top = torch.zeros(len(order), dtype=torch.long, device=order.device)
            top = top.masked_fill(~geometry['valid'].any(1), -1)
        def reorder(value):
            indices = order.reshape(*order.shape, *((1,) * (value.ndim - 2))).expand_as(value)
            return value.gather(1, indices)
        geometry = {key: reorder(value) for key, value in geometry.items()}
        pooled, logits, quality = map(reorder, (pooled, logits, quality))
        if self.mode_count == 1:
            geometry['valid'][:, 1:] = False
            geometry['member_mask'][:, 1:] = False
        return EvidenceModeSet(**geometry, features=pooled, quality_logit=logits,
                               quality=quality, evidence_top_index=top)


def mode_set_consensus(modes):
    """为旧日志/原始候选保留 evidence-top 的标量接口；决策仍读取全部模式。"""
    index = modes.evidence_top_index.clamp_min(0)
    rows = torch.arange(len(index), device=index.device)
    pick = lambda value: value[rows, index]
    sorted_scores = modes.quality.detach().masked_fill(~modes.valid, 0).sort(1, descending=True).values
    return dict(center=pick(modes.centers_xy), consistency=pick(modes.support_score),
                covariance=pick(modes.covariance_xy),
                inlier_ratio=pick(modes.unique_count).to(modes.mass) / modes.member_mask.any(1).sum(1).clamp_min(1),
                effective_mass=pick(modes.mass), margin=sorted_scores[:, 0] - sorted_scores[:, 1],
                compatible_count=modes.valid.sum(1).to(modes.mass),
                mode_unique_count=pick(modes.unique_count).to(modes.mass),
                mode_mean_targetness=pick(modes.targetness_mean),
                mode_mean_identity_margin=pick(modes.identity),
                mode_inlier_mask=pick(modes.member_mask))
