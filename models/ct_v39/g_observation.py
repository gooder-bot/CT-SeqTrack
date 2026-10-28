"""v39 独立当前观测 G：空间对应、无匹配、有限纠偏及有预算的粗搜索。

不读取 B0 特征或 GT；所有外部几何在入口 detach。返回的 soft_boxes 保留
G 自身的对应梯度。世界轴 XYZ 均相对本帧 anchor，yaw 为世界绝对角。
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.feature_precision import feature_autocast
from .tensor_geometry import inside_boxes, local_points, rotate_xy


def unique_valid(point_ids: Tensor, valid: Tensor) -> Tensor:
    """同一池中一个真实 raw ID 最多计一次；不把 ID 用作跨帧对应。"""
    if point_ids.shape[-1] == 0:
        return valid.bool()
    valid = valid.bool() & (point_ids >= 0)
    maximum = torch.iinfo(torch.long).max
    ids = point_ids.long().masked_fill(~valid, maximum)
    order = ids.argsort(dim=-1, stable=True)
    sorted_ids = ids.gather(-1, order)
    previous = torch.cat((sorted_ids[..., :1].new_full((*sorted_ids.shape[:-1], 1), -1),
                          sorted_ids[..., :-1]), -1)
    keep = (sorted_ids != maximum) & (sorted_ids != previous)
    return torch.zeros_like(valid).scatter(-1, order, keep)


def _masked_mean(values: Tensor, valid: Tensor, dim: int) -> Tensor:
    return (values * valid[..., None]).sum(dim) / valid.sum(dim).clamp_min(1)[..., None]


def _coarse_vote_max(weights: Tensor, inverse: Tensor) -> Tensor:
    """unique 生成的连续组号按组取 max，包含原 zero 初值。

    torch 2.0.1 的 CUDA scatter_reduce(amax) 在严格确定性模式直接报错。
    稳定分组后 segment max 保留同 raw-ID/空间/yaw 格的最大票语义；
    不改变其余票数、质量与高度求和，也不关闭全局确定性。
    """
    order = torch.argsort(inverse, stable=True)
    _, lengths = torch.unique_consecutive(inverse[order], return_counts=True)
    if lengths.numel() == 0:
        return weights.new_zeros((0,))
    return torch.segment_reduce(weights[order], 'max', lengths=lengths, initial=0.)


class ContiguousCpuGELU(nn.GELU):
    """兼容 torch 2.0 CPU oneDNN 的非连续 grad_output 缺陷，CUDA 数学不变。"""
    @staticmethod
    def _contiguous_gradient(gradient):
        return gradient.contiguous()

    def forward(self, value):
        output = super().forward(value)
        if output.device.type == 'cpu' and output.requires_grad:
            output.register_hook(self._contiguous_gradient)
        return output


class GObservation(nn.Module):
    def __init__(self, feature_dim=64, k=8, match_radius_min=.15,
                 match_radius_max=.5, match_radius_fraction=.1,
                 match_probability_threshold=.2, sinkhorn_iterations=5, feature_bf16=False):
        super().__init__()
        if feature_dim != 64:
            raise ValueError('v39 registered G feature_dim is 64')
        self.feature_dim, self.k = feature_dim, int(k)
        self.match_radius_min, self.match_radius_max = match_radius_min, match_radius_max
        self.match_radius_fraction = match_radius_fraction
        self.match_probability_threshold = match_probability_threshold
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.feature_bf16 = bool(feature_bf16)
        self.checkpoint_matching = True
        self.edge_mlp = nn.Sequential(nn.Linear(4, 32), nn.LayerNorm(32), ContiguousCpuGELU(),
                                      nn.Linear(32, 64), nn.LayerNorm(64), ContiguousCpuGELU())
        self.point_projection = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), ContiguousCpuGELU())
        self.targetness = nn.Sequential(nn.Linear(132, 64), nn.LayerNorm(64), ContiguousCpuGELU(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Linear(149, 128), nn.LayerNorm(128), ContiguousCpuGELU(),
                                  nn.Linear(128, 64), ContiguousCpuGELU(), nn.Linear(64, 5))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(5.)))
        self.log_geometry_scale = nn.Parameter(torch.tensor(0.))
        self.dustbin_score = nn.Parameter(torch.tensor(1.))

    def _logit_head(self, module, value):
        with feature_autocast(value, enabled=self.feature_bf16):
            hidden = value
            for layer in tuple(module.children())[:-1]:
                hidden = layer(hidden)
        # 分类/质量输出保持 FP32；外层也不得强制覆盖这一边界。
        with torch.autocast(device_type=value.device.type, enabled=False):
            return module[-1](hidden.float())

    def encode(self, pool: Mapping[str, Tensor], size: Tensor) -> dict[str, Tensor]:
        xyz, ids = pool['xyz'].detach().float(), pool['point_ids'].detach().long()
        valid = unique_valid(ids, pool['valid'].detach())
        xyz = torch.where(valid[..., None], xyz, 0.)
        size = size.detach().float().clamp_min(1e-3)
        b, n, _ = xyz.shape
        if n == 0:
            return dict(xyz=xyz, point_ids=ids, valid=valid, features=xyz.new_zeros(b, 0, 64))
        pieces = []
        count = min(self.k, n)
        row = torch.arange(b, device=xyz.device)[:, None, None]
        xy_scale, z_scale = size[:, :2].norm(dim=-1).clamp_min(.1), size[:, 2].clamp_min(.1)
        all_scale = size.norm(dim=-1).clamp_min(.1)
        # 固定块防止全原云粗代表点产生 N×N 的峰值临时张量。
        for start in range(0, n, 256):
            query = xyz[:, start:start + 256]
            distance = torch.cdist(query, xyz).masked_fill(~valid[:, None], torch.inf)
            same = ids[:, start:start + 256, None] == ids[:, None]
            distance = distance.masked_fill(same, torch.inf)
            nearest_distance, index = distance.topk(count, dim=-1, largest=False, sorted=True)
            neighbor_valid = torch.isfinite(nearest_distance) & valid[:, start:start + 256, None]
            neighbors = xyz[row, index]
            delta = torch.where(neighbor_valid[..., None], neighbors - query[..., None, :], 0.)
            density = (neighbor_valid & (nearest_distance <= 1.)).float().sum(-1) / self.k
            edge = torch.stack((delta[..., :2].norm(dim=-1) / xy_scale[:, None, None],
                                delta[..., 2] / z_scale[:, None, None],
                                delta.norm(dim=-1) / all_scale[:, None, None],
                                density[..., None].expand_as(nearest_distance)), -1)
            with feature_autocast(edge, enabled=self.feature_bf16):
                features = self.edge_mlp(edge)
            features = features.float()
            maximum = features.masked_fill(~neighbor_valid[..., None], -1e4).amax(-2)
            maximum = torch.where(neighbor_valid.any(-1)[..., None], maximum, 0.)
            mean = (features * neighbor_valid[..., None]).sum(-2) / neighbor_valid.sum(-1).clamp_min(1)[..., None]
            pooled = torch.cat((maximum, mean), -1)
            with feature_autocast(pooled, enabled=self.feature_bf16):
                pieces.append(self.point_projection(pooled))
        features = torch.cat(pieces, 1) * valid[..., None]
        return dict(xyz=xyz, point_ids=ids, valid=valid, features=features)

    def encode_templates(self, templates: Mapping[str, Tensor], size: Tensor) -> dict[str, Tensor]:
        xyz, valid = templates['xyz'].detach().float(), templates['valid'].detach().bool()
        b, t, m, _ = xyz.shape
        ids = torch.arange(m, device=xyz.device)[None, None].expand(b, t, m)
        encoded = self.encode(dict(xyz=xyz.reshape(b * t, m, 3), point_ids=ids.reshape(b * t, m),
                                   valid=valid.reshape(b * t, m)),
                              size[:, None].expand(b, t, 3).reshape(b * t, 3))
        return dict(xyz=encoded['xyz'].reshape(b, t, m, 3), valid=encoded['valid'].reshape(b, t, m),
                    features=encoded['features'].reshape(b, t, m, 64),
                    ages=templates.get('ages', xyz.new_zeros(b, t)).detach().float())

    def _matching(self, pool, templates, boxes, size):
        # 对应、Sinkhorn、刚体几何统一 FP32；网络特征可由外层 autocast 产生。
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            pf = F.normalize(pool['features'].float(), dim=-1, eps=1e-6)
            tf = F.normalize(templates['features'].float(), dim=-1, eps=1e-6)
            b, t, m, _ = tf.shape
            c, n = boxes.shape[1], pf.shape[1]
            cosine = torch.einsum('btmd,bnd->btmn', tf, pf)
            local = local_points(pool['xyz'], boxes) / size[:, None, None]
            tx = templates['xyz'] / size[:, None, None]
            distance = torch.cdist(tx[:, None].expand(b, c, t, m, 3).reshape(b * c, t * m, 3),
                                   local.reshape(b * c, n, 3)).reshape(b, c, t, m, n)
            scores = (cosine[:, None] * self.log_temperature.exp().clamp(1., 30.)
                      - .5 * distance.square().clamp_max(16.) * self.log_geometry_scale.exp().clamp(.1, 10.))
            tv, pv = templates['valid'], pool['valid']
            pair_valid = tv[:, None, :, :, None] & pv[:, None, None, None]
            scores = scores.masked_fill(~pair_valid, -1e4)
            augmented = torch.cat((scores, self.dustbin_score.expand(b, c, t, m, 1)), -1)
            augmented = torch.cat((augmented, self.dustbin_score.expand(b, c, t, 1, n + 1)), -2)
            nt, np = tv.sum(-1).float(), pv.sum(-1).float()[:, None].expand(b, t)
            total = (nt + np).clamp_min(1.)
            mu = torch.cat((tv.float(), np[..., None]), -1) / total[..., None]
            nu = torch.cat((pv[:, None].expand(b, t, n).float(), nt[..., None]), -1) / total[..., None]
            log_mu = mu.clamp_min(1e-12).log()[:, None].expand(b, c, t, m + 1)
            log_nu = nu.clamp_min(1e-12).log()[:, None].expand(b, c, t, n + 1)
            u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
            for _ in range(self.sinkhorn_iterations):
                u = log_mu - torch.logsumexp(augmented + v[..., None, :], -1)
                v = log_nu - torch.logsumexp(augmented + u[..., :, None], -2)
            transport = augmented + u[..., :, None] + v[..., None, :]
            row_log = F.log_softmax(transport[..., :m, :], -1)
            column_log = F.log_softmax(transport[..., :, :n], -2)
            row_probability = row_log[..., :n].exp().clamp_max(1.) * pair_valid
            column_probability = column_log[..., :m, :].exp().clamp_max(1.) * pair_valid
            weights = row_probability * column_probability
            if n:
                row_choice = row_log.argmax(-1)
                column_choice = column_log.argmax(-2)
                ni = torch.arange(n, device=boxes.device)
                mi = torch.arange(m, device=boxes.device)
                mutual = ((row_choice[..., None] == ni)
                          & (column_choice[..., None, :] == mi[:, None]) & pair_valid)
                hard = mutual & (row_probability >= self.match_probability_threshold)
                hard &= column_probability >= self.match_probability_threshold
            else:
                hard = torch.zeros_like(weights, dtype=torch.bool)
            return dict(weights=weights, hard=hard, row_log_probabilities=row_log,
                        column_log_probabilities=column_log, pair_valid=pair_valid,
                        local=local, normalized_distance=distance)

    def _prepare(self, pool, templates, boxes, size, encoded_pool, encoded_templates):
        boxes, size = boxes.detach().float(), size.detach().float().clamp_min(1e-3)
        encoded_pool = self.encode(pool, size) if encoded_pool is None else encoded_pool
        encoded_templates = self.encode_templates(templates, size) if encoded_templates is None else encoded_templates
        return encoded_pool, encoded_templates, boxes, size

    def _fit(self, matching, pool, templates, boxes, size, reacquire=False):
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            w = matching['weights'].float()
            mass = w.sum((-3, -2, -1)).clamp_min(1e-8)
            wp, wq = w.sum(-1), w.sum((-3, -2))
            p, q = templates['xyz'].float(), pool['xyz'].float()
            mp = torch.einsum('bctm,btmd->bcd', wp, p) / mass[..., None]
            mq = torch.einsum('bcn,bnd->bcd', wq, q) / mass[..., None]
            cross = torch.einsum('bctmn,btmd,bne->bcde', w, p[..., :2], q[..., :2]) / mass[..., None, None]
            cross = cross - mp[..., :2, None] * mq[..., None, :2]
            a, bb = cross[..., 0, 0] + cross[..., 1, 1], cross[..., 0, 1] - cross[..., 1, 0]
            identifiable = (a.square() + bb.square()) > 1e-12
            angle = torch.atan2(torch.where(identifiable, bb, 0.), torch.where(identifiable, a, 1.))
            centered = p[:, None, :, :, :2] - mp[..., None, None, :2]
            covariance = torch.einsum('bctm,bctmd,bctme->bcde', wp, centered, centered) / mass[..., None, None]
            trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
            disc = ((covariance[..., 0, 0] - covariance[..., 1, 1]).square()
                    + 4 * covariance[..., 0, 1].square()).clamp_min(1e-12).sqrt()
            ratio = ((trace - disc).clamp_min(0.) / (trace + disc).clamp_min(1e-8))
            support = matching['hard'].any(dim=2).any(dim=2)
            count = support.sum(-1)
            geometry_valid = count >= 3
            yaw_valid = (count >= 6) & (ratio >= .05) & identifiable
            delta_yaw = torch.atan2((angle - boxes[..., 3]).sin(), (angle - boxes[..., 3]).cos())
            if not reacquire:
                delta_yaw = delta_yaw.clamp(-math.pi / 6., math.pi / 6.)
            yaw = boxes[..., 3] + torch.where(yaw_valid, delta_yaw, 0.)
            translation = torch.cat((mq[..., :2] - rotate_xy(mp[..., :2], yaw),
                                     mq[..., 2:3] - mp[..., 2:3]), -1)
            if not reacquire:
                delta = translation - boxes[..., :3]
                xy_limit = (.5 * size[:, :2].norm(dim=-1)).clamp_max(2.)[:, None]
                xy = delta[..., :2] * (xy_limit / delta[..., :2].norm(dim=-1).clamp_min(1e-8)).clamp_max(1.)[..., None]
                z_limit = (.25 * size[:, 2]).clamp_max(.5)[:, None]
                z = delta[..., 2].maximum(-z_limit).minimum(z_limit)
                translation = boxes[..., :3] + torch.cat((xy, z[..., None]), -1)
            soft = torch.cat((translation, torch.atan2(yaw.sin(), yaw.cos())[..., None]), -1)
            return dict(boxes=torch.where(geometry_valid[..., None], soft, boxes), soft_boxes=soft,
                        valid=geometry_valid, support_mask=support, support_count=count,
                        geometry_valid=geometry_valid, yaw_valid=yaw_valid,
                        covariance_ratio=ratio, matching=matching)

    def refine(self, pool, templates, boxes, size, *, encoded_pool=None,
               encoded_templates=None, reacquire=False):
        ep, et, boxes, size = self._prepare(pool, templates, boxes, size, encoded_pool, encoded_templates)
        if self.training and torch.is_grad_enabled() and self.checkpoint_matching:
            result = checkpoint(self._refine_encoded, ep, et, boxes, size, reacquire,
                                use_reentrant=False, preserve_rng_state=False)
        else:
            result = self._refine_encoded(ep, et, boxes, size, reacquire)
        result.update(encoded_pool=ep, encoded_templates=et, input_boxes=boxes, reacquire=bool(reacquire))
        return result

    def _refine_encoded(self, ep, et, boxes, size, reacquire=False):
        matching = self._matching(ep, et, boxes, size)
        result = self._fit(matching, ep, et, boxes, size, reacquire)
        # 拟合监督只需 soft pose，不能把未使用的 M×N 概率表存到每个候选。
        result.pop('matching')
        return result

    def verify(self, pool, templates, boxes, size, *, candidate_valid=None,
               measurement_eligible=None, encoded_pool=None, encoded_templates=None):
        ep, et, boxes, size = self._prepare(pool, templates, boxes, size, encoded_pool, encoded_templates)
        if self.training and torch.is_grad_enabled() and self.checkpoint_matching:
            result = checkpoint(self._verify_encoded, ep, et, boxes, size, candidate_valid,
                                measurement_eligible, use_reentrant=False, preserve_rng_state=False)
        else:
            result = self._verify_encoded(ep, et, boxes, size, candidate_valid, measurement_eligible)
        result.update(encoded_pool=ep, encoded_templates=et)
        return result

    def _verify_encoded(self, ep, et, boxes, size, candidate_valid, measurement_eligible):
        matching = self._matching(ep, et, boxes, size)
        fit = self._fit(matching, ep, et, boxes, size)
        w = matching['weights']
        mass = w.sum((2, 3)).clamp_min(1e-8)
        target_context = torch.einsum('bctmn,btmd->bcnd', w, et['features'].float()) / mass[..., None]
        b, c, n = target_context.shape[:3]
        point_features = ep['features'][:, None].expand(b, c, n, 64).float()
        strength = w.amax((2, 3)) if w.shape[2] and w.shape[3] else w.new_zeros(b, c, n)
        targetness_logits = self._logit_head(self.targetness, torch.cat((point_features, target_context,
                                                       matching['local'].clamp(-10., 10.), strength[..., None]), -1)).squeeze(-1)
        inside = inside_boxes(ep['xyz'], boxes, size)
        valid_points = ep['valid'][:, None].expand(b, c, n)
        support = valid_points & inside & (targetness_logits.detach().sigmoid() >= .5)
        count = support.sum(-1)
        probabilities = targetness_logits.sigmoid() * valid_points
        pooled = (point_features * probabilities[..., None]).sum(-2) / probabilities.sum(-1).clamp_min(1e-6)[..., None]
        context = (target_context * probabilities[..., None]).sum(-2) / probabilities.sum(-1).clamp_min(1e-6)[..., None]
        hard = matching['hard']
        template_coverage = hard.any(-1).float().sum(-1) / et['valid'].sum(-1).clamp_min(1)[:, None]
        overall_coverage = template_coverage.mean(-1)
        initial_coverage = template_coverage[..., 0]
        recent_coverage = template_coverage[..., 1:].mean(-1) if template_coverage.shape[-1] > 1 else initial_coverage.new_zeros(b, c)
        residual = (w * matching['normalized_distance']).sum((2, 3, 4)) / w.sum((2, 3, 4)).clamp_min(1e-8)
        dustbin = matching['row_log_probabilities'][..., -1].exp()
        unmatched = (dustbin * et['valid'][:, None]).sum((2, 3)) / et['valid'].sum((1, 2)).clamp_min(1)[:, None]
        point_count = valid_points.sum(-1)
        probability_mean = probabilities.sum(-1) / point_count.clamp_min(1)
        inside_count = (inside & valid_points).sum(-1)
        age = (et['ages'].clamp_min(0.).log1p() * et['valid'].any(-1)).sum(-1) / et['valid'].any(-1).sum(-1).clamp_min(1)
        metadata = torch.stack((point_count.float().log1p(), count.float().log1p(), fit['support_count'].float().log1p(),
                                overall_coverage, initial_coverage, recent_coverage, residual, unmatched,
                                fit['covariance_ratio'], fit['yaw_valid'].float(), probability_mean,
                                inside_count.float().log1p(), age[:, None].expand(b, c),
                                et['valid'].sum((1, 2)).float().log1p()[:, None].expand(b, c),
                                (count >= 3).float(), (fit['support_count'] >= 3).float()), -1)
        geometry = torch.cat(((boxes[..., :3] / size[:, None]).clamp(-20., 20.),
                              boxes[..., 3:4].sin(), boxes[..., 3:4].cos()), -1)
        prediction = self._logit_head(self.head, torch.cat((pooled, context, metadata, geometry), -1))
        logits, q_logits, region_no_match = prediction[..., :3], prediction[..., 3], prediction[..., 4]
        candidate_valid = torch.ones_like(count, dtype=torch.bool) if candidate_valid is None else candidate_valid.detach().bool()
        measurement_eligible = candidate_valid if measurement_eligible is None else measurement_eligible.detach().bool() & candidate_valid
        utility = logits[..., :2].sigmoid().mean(-1)
        trust = logits[..., 2].sigmoid() * measurement_eligible
        return dict(logits=logits, utility=utility, q_logits=q_logits,
                    region_no_match_logits=region_no_match, targetness_logits=targetness_logits,
                    support_mask=support, support_count=count, matched_support_count=fit['support_count'],
                    candidate_valid=candidate_valid, measurement_eligible=measurement_eligible,
                    trust=trust, metadata=metadata, geometry_valid=fit['geometry_valid'],
                    yaw_valid=fit['yaw_valid'], matching={key: matching[key] for key in
                        ('row_log_probabilities', 'column_log_probabilities')}, boxes=boxes, size=size)

    @torch.no_grad()
    def coarse_proposals(self, pool, templates, size, reference_yaw, *,
                         encoded_pool=None, encoded_templates=None, voxel_size=.5, peaks=2):
        """4 个 yaw 的模板中心投票；只生成 proposal，必须再读真实 crop 核验。"""
        if peaks != 2 or voxel_size <= 0:
            raise ValueError('registered coarse search requires two peaks and positive voxel size')
        size = size.detach().float()
        ep = self.encode(pool, size) if encoded_pool is None else encoded_pool
        et = self.encode_templates(templates, size) if encoded_templates is None else encoded_templates
        b, n = ep['valid'].shape
        flat_xyz, flat_valid = et['xyz'].flatten(1, 2), et['valid'].flatten(1, 2)
        flat_features = F.normalize(et['features'].flatten(1, 2).float(), dim=-1)
        features = F.normalize(ep['features'].float(), dim=-1)
        score = torch.einsum('bnd,bmd->bnm', features, flat_features).masked_fill(~flat_valid[:, None], -1e4)
        results, validity, scores_out = [], [], []
        top = min(2, flat_xyz.shape[1])
        if n == 0 or top == 0:
            return dict(boxes=size.new_zeros(b, 2, 4), valid=torch.zeros(b, 2, dtype=torch.bool, device=size.device),
                        scores=size.new_zeros(b, 2))
        similarity, match_index = score.topk(top, -1)
        row = torch.arange(b, device=size.device)[:, None, None]
        matched = flat_xyz[row, match_index]
        correspondence_valid = flat_valid[row, match_index] & ep['valid'][..., None]
        weight = ((similarity + 1.) * .5).clamp(0., 1.) * correspondence_valid
        for batch_index in range(b):
            yaw = reference_yaw.detach().float()[batch_index] + torch.arange(4, device=size.device) * (math.pi / 2.)
            rotated = rotate_xy(matched[batch_index, None, ..., :2], yaw[:, None, None])
            # [4,N,top,3] 世界轴中心票。
            xy = ep['xyz'][batch_index, None, :, None, :2] - rotated
            z = ep['xyz'][batch_index, :, None, 2] - matched[batch_index, ..., 2]
            votes = torch.cat((xy, z[None, ..., None].expand(4, n, top, 1)), -1).reshape(-1, 3)
            weights = weight[batch_index][None].expand(4, n, top).reshape(-1)
            yaw_id = torch.arange(4, device=size.device)[:, None, None].expand(4, n, top).reshape(-1)
            raw_id = ep['point_ids'][batch_index][None, :, None].expand(4, n, top).reshape(-1)
            cells = torch.floor(votes[:, :2] / voxel_size).long()
            # 同一点对同一个空间/yaw 格仅保留最大票，避免两个模板近邻重复加票。
            combined = torch.cat((cells, yaw_id[:, None], raw_id[:, None]), -1)
            unique, inverse = torch.unique(combined, dim=0, return_inverse=True)
            maximum = _coarse_vote_max(weights, inverse)
            keys, group = torch.unique(unique[:, :3], dim=0, return_inverse=True)
            masses = weights.new_zeros(len(keys)).scatter_add_(0, group, maximum)
            counts = weights.new_zeros(len(keys)).scatter_add_(0, group, (maximum > 0).float())
            raw_mass = weights.new_zeros(len(unique)).scatter_add_(0, inverse, weights)
            raw_height = weights.new_zeros(len(unique)).scatter_add_(0, inverse, weights * votes[:, 2])
            raw_height = raw_height / raw_mass.clamp_min(1e-6)
            heights = weights.new_zeros(len(keys)).scatter_add_(0, group, maximum * raw_height)
            heights = heights / masses.clamp_min(1e-6)
            centers = (keys[:, :2].float() + .5) * voxel_size
            all_yaw = yaw[keys[:, 2]]
            first = masses.argmax()
            separated = (centers - centers[first]).norm(dim=-1) >= .5 * size[batch_index, :2].norm()
            second = masses.masked_fill(~separated, -1.).argmax()
            selected = torch.stack((first, second))
            ok = (masses[selected] > 0) & (counts[selected] >= 3)
            ok[1] &= separated[second]
            # 高度仅使用该峰的票，其他车辆/背景峰不得拉偏搜索 crop 的 Z。
            results.append(torch.cat((centers[selected], heights[selected, None], all_yaw[selected, None]), -1))
            validity.append(ok)
            scores_out.append(masses[selected])
        return dict(boxes=torch.stack(results), valid=torch.stack(validity), scores=torch.stack(scores_out))
