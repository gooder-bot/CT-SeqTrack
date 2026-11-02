"""v41 独立 B3：只核验已采样 B0 当前点；所有上游输入在边界 detach。"""
import math

import torch
from torch import nn

from .observation import B0Observation, box_corners_xyz
from .geometry import inside_final_boxes
from .contracts import VerificationOutput


def _current_measurements(observation, batch):
    """支持集合无需复制64维特征；与核验点池共用同一有效性定义。"""
    valid = B0Observation.measurement_mask(batch)[:, -1].detach()
    ids = batch['point_ids'][:, -1].detach()
    xyz = batch['points'][:, -1, :, :3].detach()
    probability = observation.foreground_probability[:, -1].detach()
    if not all(bool(torch.isfinite(value[valid]).all()) for value in (xyz, probability)):
        raise ValueError('v41 current B0 evidence must be finite')
    return dict(ids=ids, valid=valid, xyz=torch.where(valid[..., None], xyz, 0.),
        probability=torch.where(valid, probability, 0.),
        source=torch.zeros_like(probability), base_count=valid.shape[1])


def build_current_pool(observation, batch):
    pool = _current_measurements(observation, batch)
    features = observation.point_features[:, -1].detach()
    if not bool(torch.isfinite(features[pool['valid']]).all()):
        raise ValueError('v41 current B0 features must be finite')
    pool['features'] = torch.where(pool['valid'][..., None], features, 0.)
    return pool


@torch.no_grad()
def current_candidate_support(observation, batch, boxes):
    """标签、硬门与提交共享的最终候选支持集合；不读取 GT。"""
    pool = _current_measurements(observation, batch)
    inside = inside_final_boxes(pool['xyz'], boxes.detach(), batch['box_size'])
    mask = pool['valid'][:, None] & inside & (pool['probability'][:, None] >= .5)
    return dict(mask=mask, ids=pool['ids'], xyz=pool['xyz'],
        probabilities=pool['probability'], valid=pool['valid'], count=mask.sum(-1))


@torch.no_grad()
def current_q0_support(observation, batch, q0_box):
    result = current_candidate_support(observation, batch, q0_box[:, None])
    result['mask'], result['count'] = result['mask'][:, 0], result['count'][:, 0]
    return result


class LocalBoxReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(69, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(130, 64)
        self.norm = nn.Sequential(nn.LayerNorm(64), nn.GELU())
        for layer in (self.point_encoder[0], self.point_encoder[3], self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, pool, boxes, allowed, size):
        b, q, _ = boxes.shape
        if size.shape != (b, 3) or not bool(torch.isfinite(size).all()) or not bool((size > 0).all()):
            raise ValueError('v41 local reader requires positive finite LWH')
        ids, valid = pool['ids'], pool['valid']
        if allowed.shape != (b, q, ids.shape[1]):
            raise ValueError('v41 local permissions do not align')
        order = torch.argsort(ids.masked_fill(~valid, torch.iinfo(torch.long).max), stable=True)
        xyz = pool['xyz'].gather(1, order[..., None].expand(-1, -1, 3)).detach()
        features = pool['features'].gather(1, order[..., None].expand(-1, -1, 64)).detach()
        probability = pool['probability'].gather(1, order).detach()
        source = pool['source'].gather(1, order).detach()
        allowed = allowed.detach().bool().gather(2, order[:, None].expand(-1, q, -1))
        boxes = boxes.detach()
        corners = box_corners_xyz(boxes, size.detach()[:, None])
        delta = xyz[:, None, None] - corners[..., None, :]
        sine, cosine = boxes[..., 3, None, None].sin(), boxes[..., 3, None, None].cos()
        x, y, z = delta.unbind(-1)
        relative = torch.stack((cosine*x+sine*y, -sine*x+cosine*y, z), -1)
        relative = relative / size.detach()[:, None, None, None]
        distance = relative.square().sum(-1)
        neighborhood = allowed[:, :, None] & (distance <= 1.)
        count = neighborhood.sum(-1)
        support = (probability[:, None, None] * neighborhood).sum(-1) / count.clamp_min(1)
        ranked = torch.argsort(distance.masked_fill(~neighborhood, torch.inf), stable=True)
        selected = ranked[..., :min(16, ids.shape[1])]
        mask = neighborhood.gather(-1, selected)
        selected_count = mask.sum(-1)
        pf = features[:, None, None].expand(-1, q, 8, -1, -1).gather(
            3, selected[..., None].expand(-1, -1, -1, -1, 64))
        rel = relative.gather(3, selected[..., None].expand(-1, -1, -1, -1, 3))
        prob = probability[:, None, None].expand(-1, q, 8, -1).gather(3, selected)
        src = source[:, None, None].expand(-1, q, 8, -1).gather(3, selected)
        encoded = self.point_encoder(torch.where(mask[..., None],
            torch.cat((pf, rel, prob[..., None], src[..., None]), -1), 0.))
        mean = torch.where(mask[..., None], encoded, 0.).sum(-2) / selected_count.clamp_min(1)[..., None]
        present = selected_count > 0
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).max(-2).values
        maximum = torch.where(present[..., None], maximum, 0.)
        statistics = torch.stack((torch.log1p(count.to(features))/math.log1p(ids.shape[1]), support), -1)
        value = self.norm(self.output(torch.cat((maximum, mean, statistics), -1)))
        value = torch.where(present[..., None], value, 0.)
        # 返回实际 8×16 读取点的唯一并集，并还原成 batch 原始点顺序。
        selected_mask = torch.zeros_like(neighborhood).scatter(-1, selected, mask)
        point_mask = torch.zeros_like(allowed).scatter(
            -1, order[:, None].expand(-1, q, -1), selected_mask.any(-2))
        return value, dict(local_neighbor_count=count.detach(), local_selected_count=selected_count.detach(),
            local_mean_support=support.detach(), local_delta_norm=value.detach().norm(dim=-1),
            local_point_mask=point_mask.detach())


class FinalBoxVerifier(nn.Module):
    def __init__(self, time_scale=.5):
        super().__init__()
        self.time_scale = float(time_scale)
        if not math.isfinite(self.time_scale) or self.time_scale <= 0:
            raise ValueError('v41 time_scale must be finite and positive')
        self.local_reader = LocalBoxReader()
        self.box_encoder = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(110, 64), nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(64, 3)
        for layer in (self.box_encoder[0], self.fusion[0]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # S/P/T 均从 logit 0 开始；立即部署，平分按 q0 优先。
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def metadata(self, candidates, pool, inside, support, observation, batch, prior):
        box = candidates.boxes.detach()
        size = batch['box_size'].detach().to(box)
        pbox = prior.box.detach().to(box)
        qbox = box[:, 0]
        fallback = batch['fallback_box'].detach().to(box)
        pd, qd = (box[..., :3]-pbox[:, None, :3])/size[:, None], (box[..., :3]-qbox[:, None, :3])/size[:, None]
        pa, qa = box[..., 3]-pbox[:, None, 3], box[..., 3]-qbox[:, None, 3]
        source = torch.eye(4, dtype=box.dtype, device=box.device)[None].expand(len(box), -1, -1)
        history_times = batch['history_times'].detach().to(box)
        history_age = torch.where(batch['history_valid'].bool(), -history_times, 0.).clamp_min(0).amax(-1)
        def column(key, default=0.):
            value = batch.get(key)
            return box.new_full((len(box),), default) if value is None else value.detach().to(box).reshape(-1)
        ages = torch.stack((column('weak_age'), column('supported_age'), history_age), -1)
        ages = torch.log1p(ages.clamp_min(0)/self.time_scale)
        shared = torch.cat((prior.log_sigma.detach().to(box), prior.direction_xy.detach().to(box),
            torch.log1p(batch['current_dt'].detach().to(box).reshape(-1, 1)/self.time_scale),
            prior.valid.detach().to(box)[:, None], observation.quality.detach().to(box),
            observation.sequence_valid.detach().to(box)[:, None], observation.current_valid.detach().to(box)[:, None],
            size.log(), ages), -1)
        count, foreground = inside.sum(-1), support.sum(-1)
        probability = pool['probability'][:, None]
        mean = torch.where(inside, probability, 0.).sum(-1)/count.clamp_min(1)
        probability_bounded = probability.clamp(1e-6, 1.-1e-6)
        entropy = -(probability_bounded*probability_bounded.log()+(1-probability_bounded)*(1-probability_bounded).log())/math.log(2.)
        entropy = torch.where(inside, entropy, 0.).sum(-1)/count.clamp_min(1)
        counts = torch.stack((torch.log1p(count.to(box))/math.log1p(pool['base_count']),
            torch.log1p(foreground.to(box))/math.log1p(pool['base_count']), mean, entropy), -1)
        fd = (box[..., :3]-fallback[:, None, :3])/size[:, None]
        fa = box[..., 3]-fallback[:, None, 3]
        velocity_valid = column('trusted_velocity_valid').bool()
        velocity_age = torch.where(velocity_valid, column('trusted_velocity_age').clamp_min(0), 0.)
        risk = torch.stack((column('previous_is_propagation'),
            torch.log1p(column('consecutive_propagations').clamp_min(0)),
            torch.log1p(column('consecutive_untrusted_outputs').clamp_min(0)),
            velocity_valid.to(box), torch.log1p(velocity_age/self.time_scale)), -1)
        result = torch.cat((pd, qd, pa.sin()[..., None], pa.cos()[..., None], qa.sin()[..., None], qa.cos()[..., None],
            source, shared[:, None].expand(-1, 4, -1), counts, fd, fa.sin()[..., None], fa.cos()[..., None],
            risk[:, None].expand(-1, 4, -1)), -1)
        if result.shape != (len(box), 4, 46) or not bool(torch.isfinite(result).all()):
            raise ValueError('v41 verifier requires finite [B,4,46] metadata')
        return result.detach()

    def read_local(self, pool, boxes, valid, size):
        authorized = pool['valid'][:, None] & valid.detach()[..., None]
        local, statistics = self.local_reader(pool, boxes.detach(), authorized, size.detach())
        feature = self.box_encoder(local.reshape(len(boxes), boxes.shape[1], 512))
        return feature, statistics

    def forward(self, candidates, observation, batch, prior, *, pool=None):
        pool = build_current_pool(observation, batch) if pool is None else pool
        authorized = pool['valid'][:, None] & candidates.valid.detach()[..., None]
        inside = authorized & inside_final_boxes(pool['xyz'], candidates.boxes.detach(), batch['box_size'].detach())
        support = inside & (pool['probability'][:, None] >= .5)
        feature, statistics = self.read_local(pool, candidates.boxes, candidates.valid, batch['box_size'])
        metadata = self.metadata(candidates, pool, inside, support, observation, batch, prior)
        logits = self.output(self.fusion(torch.cat((feature, metadata), -1)))
        return VerificationOutput(logits=logits, support_mask=support.detach(),
                                  metadata=metadata, **statistics)
