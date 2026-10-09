"""最终框效用核验；唯一梯度出口是本模块自身参数。"""
import math

import torch
from torch import nn

from models.ct_v31.observation import box_corners_xyz
from models.ct_v36.verification import (build_current_pool,
    candidate_authorization, inside_final_boxes)
from .contracts import VerificationOutput


def read_lwh_local(reader, pool, boxes, allowed, size):
    """v35邻域数学与稳定raw-ID排序；读取参数由调用者独立持有。"""
    b, q, _ = boxes.shape
    if size.shape != (b, 3) or not bool(torch.isfinite(size).all()) or not bool((size > 0).all()):
        raise ValueError('v37 local reader requires positive finite first-frame LWH')
    ids, valid = pool['ids'], pool['valid']
    if allowed.shape != (b, q, ids.shape[1]):
        raise ValueError('v37 local permissions do not align')
    order = torch.argsort(ids.masked_fill(~valid, torch.iinfo(torch.long).max), stable=True)
    xyz = pool['xyz'].gather(1, order[..., None].expand(-1, -1, 3)).detach()
    features = pool['features'].gather(1, order[..., None].expand(-1, -1, 64))
    probability = pool['probability'].gather(1, order).detach()
    source = pool['source'].gather(1, order).detach()
    allowed = allowed.detach().bool().gather(2, order[:, None].expand(-1, q, -1))
    seed = boxes.detach()
    corners = box_corners_xyz(seed, size.detach()[:, None])
    delta = xyz[:, None, None] - corners[..., None, :]
    sine, cosine = seed[..., 3, None, None].sin(), seed[..., 3, None, None].cos()
    x, y, z = delta.unbind(-1)
    relative = torch.stack((cosine * x + sine * y, -sine * x + cosine * y, z), -1)
    relative = relative / size.detach()[:, None, None, None]
    # 定位与核验各自独立权重，但均保留v35的LWH归一化邻域。
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
    encoded = reader.point_encoder(torch.where(mask[..., None],
        torch.cat((pf, rel, prob[..., None], src[..., None]), -1), 0.))
    mean = torch.where(mask[..., None], encoded, 0.).sum(-2) / selected_count.clamp_min(1)[..., None]
    present = selected_count > 0
    maximum = encoded.masked_fill(~mask[..., None], -torch.inf).max(-2).values
    maximum = torch.where(present[..., None], maximum, 0.)
    statistics = torch.stack((torch.log1p(count.to(features)) / math.log(1281.), support), -1)
    value = reader.output(torch.cat((maximum, mean, statistics), -1))
    if hasattr(reader, 'norm'):
        value = reader.norm(value)
    value = torch.where(present[..., None], value, 0.)
    return value, dict(local_neighbor_count=count.detach(), local_selected_count=selected_count.detach(),
        local_mean_support=support.detach(), local_delta_norm=value.detach().norm(dim=-1))


class LocalBoxReader(nn.Module):
    """独立核验读取：每角LWH半径1、最多16唯一点，不共享定位参数。"""
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
        return read_lwh_local(self, pool, boxes, allowed, size)


class FinalBoxVerifier(nn.Module):
    def __init__(self, time_scale=.5):
        super().__init__()
        self.time_scale = float(time_scale)
        self.local_reader = LocalBoxReader()
        self.box_encoder = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(99, 64), nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(64, 3)
        for layer in (self.box_encoder[0], self.fusion[0]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor([0., 0., math.log(.1 / .9)]))

    def metadata(self, candidates, pool, inside, support, observation, batch, prior):
        """35维 v35 同源统计；没有事件状态、GT或神经评分输入。"""
        box = candidates.boxes.detach()
        size = batch['box_size'].detach().to(box)
        pbox, coarse = prior.box.detach().to(box), observation.coarse_box.detach()
        pd = (box[..., :3] - pbox[:, None, :3]) / size[:, None]
        cd = (box[..., :3] - coarse[:, None, :3]) / size[:, None]
        pa, ca = box[..., 3] - pbox[:, None, 3], box[..., 3] - coarse[:, None, 3]
        source = box.new_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])[None].expand(len(box), -1, -1)
        history_times = batch['history_times'].detach().to(box)
        history_age = torch.where(batch['history_valid'].bool(), -history_times, 0.).clamp_min(0).amax(-1)
        ages = torch.stack((batch.get('weak_age', history_age.new_zeros(len(box))).detach().to(box).reshape(-1),
            batch.get('supported_age', history_age.new_zeros(len(box))).detach().to(box).reshape(-1), history_age), -1)
        ages = torch.log1p(ages.clamp_min(0) / self.time_scale)
        shared = torch.cat((prior.log_sigma.detach().to(box), prior.direction_xy.detach().to(box),
            torch.log1p(batch['current_dt'].detach().to(box).reshape(-1, 1) / self.time_scale),
            prior.valid.detach().to(box)[:, None], observation.quality.detach().to(box),
            observation.sequence_valid.detach().to(box)[:, None], observation.current_valid.detach().to(box)[:, None],
            size.log(), ages), -1)
        count, foreground = inside.sum(-1), support.sum(-1)
        probability = pool['probability'][:, None]
        mean = torch.where(inside, probability, 0.).sum(-1) / count.clamp_min(1)
        p = probability.clamp(1e-6, 1. - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2.)
        entropy = torch.where(inside, entropy, 0.).sum(-1) / count.clamp_min(1)
        counts = torch.stack((torch.log1p(count.to(box)) / math.log(1281.),
            torch.log1p(foreground.to(box)) / math.log(1281.), mean, entropy), -1)
        result = torch.cat((pd, cd, pa.sin()[..., None], pa.cos()[..., None],
            ca.sin()[..., None], ca.cos()[..., None], source,
            shared[:, None].expand(-1, 4, -1), counts), -1)
        if result.shape != (len(box), 4, 35) or not bool(torch.isfinite(result).all()):
            raise ValueError('v37 verifier requires finite [B,4,35] metadata')
        return result.detach()

    def forward(self, candidates, observation, evidence, batch, prior):
        raw_pool = build_current_pool(observation, evidence, batch)
        # 边界只阻断评分目标对定位/身份特征的反推；它们仍由各自loss正常学习。
        pool = {key: value.detach() if torch.is_tensor(value) else value
                for key, value in raw_pool.items()}
        authorized = candidate_authorization(pool, evidence.members.detach(), candidates.valid, q0_extension=True)
        inside = authorized & inside_final_boxes(pool['xyz'], candidates.boxes.detach(), batch['box_size'].detach())
        support = inside & (pool['probability'][:, None] >= .5)
        local, statistics = self.local_reader(pool, candidates.boxes.detach(), authorized, batch['box_size'].detach())
        features = self.box_encoder(local.reshape(len(local), 4, 512))
        metadata = self.metadata(candidates, pool, inside, support, observation, batch, prior)
        logits = self.output(self.fusion(torch.cat((features, metadata), -1)))
        # 写入资格与评分读取明确分开；host仍独立复算最终框支持作为事务核对。
        commit_support = support.detach().clone()
        commit_support[:, 0, pool['base_count']:] = False
        commit_support[:, 0] &= ~candidates.q0_is_fallback[:, None]
        commit_support[:, 1] = False
        return VerificationOutput(logits, support.detach(), pool['ids'], pool['xyz'],
            pool['probability'], authorized.detach(), commit_support_mask=commit_support,
            support_source=pool['source'], metadata=metadata, **statistics)


__all__ = ['FinalBoxVerifier', 'LocalBoxReader', 'build_current_pool',
           'candidate_authorization', 'inside_final_boxes']
