"""v36 最终框核验：当前唯一点、候选权限和 detached 几何。"""
from __future__ import annotations

import math

import torch
from torch import nn

from models.ct_v31.observation import B0Observation, box_corners_xyz, unique_valid_mask
from .contracts import VerificationOutput


def projected_xy_size(size, yaw, axes):
    """物体L/W在给定列轴中的包围尺寸；只作detached几何规范化。"""
    c,s = yaw.detach().cos(),yaw.detach().sin()
    rotation = torch.stack((c,-s,s,c),-1).reshape(-1,2,2)
    return (axes.detach().transpose(-1,-2).matmul(rotation).abs()
            .matmul(size.detach()[:,:2,None]).squeeze(-1)).clamp_min(1e-6)


def build_current_pool(observation, evidence, batch):
    """B0 当前采样点 + 已选256扩展点；memory/768未选点没有测量权限。"""
    base = B0Observation.measurement_mask(batch)[:, -1]
    ids = torch.cat((batch['point_ids'][:, -1], evidence.point_ids), 1).detach()
    valid = unique_valid_mask(torch.cat((base, evidence.point_valid.bool()), 1), ids)
    xyz = torch.cat((batch['points'][:, -1, :, :3], evidence.point_xyz), 1).detach()
    features = torch.cat((observation.point_features[:, -1], evidence.point_features), 1)
    probability = torch.cat((observation.foreground_probability[:, -1],
                             evidence.selected_identity_logits.sigmoid()), 1).detach()
    if not all(bool(torch.isfinite(x[valid]).all()) for x in (xyz, features, probability)):
        raise ValueError('v36 valid current evidence must be finite')
    return dict(ids=ids, valid=valid, xyz=torch.where(valid[..., None], xyz, 0.),
        features=torch.where(valid[..., None], features, 0.),
        probability=torch.where(valid, probability, 0.),
        source=torch.cat((torch.zeros_like(base), torch.ones_like(evidence.point_valid)), 1).to(features),
        base_count=base.shape[1])


def candidate_authorization(pool, members, candidate_valid, *, q0_extension):
    """顺序固定q0/prior/mode0/mode1；不同来源不伪造相同信息量。"""
    n = pool['base_count']
    base, extension = pool['valid'][:, :n], pool['valid'][:, n:]
    if members.shape != (len(base), 2, extension.shape[1]):
        raise ValueError('v36 requires two extension member sets')
    q0 = extension if q0_extension else torch.zeros_like(extension)
    ext = torch.cat((q0[:, None], extension[:, None], members.detach().bool()), 1)
    allowed = torch.cat((base[:, None].expand(-1, 4, -1), ext), -1)
    return allowed & pool['valid'][:, None] & candidate_valid.detach().bool()[..., None]


def inside_final_boxes(xyz, boxes, size):
    """当前输入框内判断；不依赖GT，也不向最终pose回传。"""
    boxes, xyz, size = boxes.detach(), xyz.detach(), size.detach()
    delta = xyz[:, None] - boxes[..., None, :3]
    c, s = boxes[..., 3, None].cos(), boxes[..., 3, None].sin()
    x, y, z = delta.unbind(-1)
    local = torch.stack((c * x + s * y, -s * x + c * y, z), -1)
    return (local.abs() <= size[:, None, None] * .5).all(-1)


class LocalBoxReader(nn.Module):
    """每角读取最多16唯一点；69维点输入，几何/概率/索引均detach。"""
    def __init__(self, *, zero_output=False):
        super().__init__()
        self.point_encoder = nn.Sequential(nn.Linear(69, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(130, 64)
        self.norm = nn.Identity() if zero_output else nn.Sequential(nn.LayerNorm(64), nn.GELU())
        for layer in (self.point_encoder[0], self.point_encoder[3], self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if zero_output:
            nn.init.zeros_(self.output.weight)

    def forward(self, pool, boxes, allowed, size):
        b, q, _ = boxes.shape
        if size.shape != (b, 3) or not bool(torch.isfinite(size).all()) or not bool((size > 0).all()):
            raise ValueError('v36 local reader requires positive first-frame LWH')
        if allowed.shape != (b, q, pool['ids'].shape[1]):
            raise ValueError('candidate point permissions do not align')
        ids, valid = pool['ids'], pool['valid']
        order = torch.argsort(ids.masked_fill(~valid, torch.iinfo(torch.long).max), stable=True)
        xyz = pool['xyz'].gather(1, order[..., None].expand(-1, -1, 3)).detach()
        feature = pool['features'].gather(1, order[..., None].expand(-1, -1, 64))
        probability = pool['probability'].gather(1, order).detach()
        source = pool['source'].gather(1, order).detach()
        allowed = allowed.detach().bool().gather(2, order[:, None].expand(-1, q, -1))
        seed = boxes.detach()
        corners = box_corners_xyz(seed, size.detach()[:, None])
        delta = xyz[:, None, None] - corners[..., None, :]
        c, s = seed[..., 3, None, None].cos(), seed[..., 3, None, None].sin()
        x, y, z = delta.unbind(-1)
        relative = torch.stack((c*x+s*y, -s*x+c*y, z), -1) / size.detach()[:, None, None, None]
        # 半径和近邻顺序使用真实米；LWH仅规范化输入MLP的相对坐标。
        distance = delta.square().sum(-1)
        neighborhood = allowed[:, :, None] & (distance <= 1.)
        count = neighborhood.sum(-1)
        support = (probability[:, None, None] * neighborhood).sum(-1) / count.clamp_min(1)
        ranked = torch.argsort(distance.masked_fill(~neighborhood, torch.inf), stable=True)
        selected = ranked[..., :min(16, ids.shape[1])]
        mask = neighborhood.gather(-1, selected)
        selected_count = mask.sum(-1)
        pf = feature[:, None, None].expand(-1, q, 8, -1, -1).gather(
            3, selected[..., None].expand(-1, -1, -1, -1, 64))
        rel = relative.gather(3, selected[..., None].expand(-1, -1, -1, -1, 3))
        ps = probability[:, None, None].expand(-1, q, 8, -1).gather(3, selected)
        src = source[:, None, None].expand(-1, q, 8, -1).gather(3, selected)
        encoded = self.point_encoder(torch.where(mask[..., None],
            torch.cat((pf, rel, ps[..., None], src[..., None]), -1), 0.))
        mean = torch.where(mask[..., None], encoded, 0.).sum(-2) / selected_count.clamp_min(1)[..., None]
        present = selected_count > 0
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).max(-2).values
        maximum = torch.where(present[..., None], maximum, 0.)
        statistics = torch.stack((torch.log1p(count.to(feature))/math.log(1281.), support), -1)
        value = self.norm(self.output(torch.cat((maximum, mean, statistics), -1)))
        value = torch.where(present[..., None], value, 0.)
        return value, dict(local_neighbor_count=count.detach(),
            local_selected_count=selected_count.detach(), local_mean_support=support.detach(),
            local_delta_norm=value.detach().norm(dim=-1))


class FinalBoxVerifier(nn.Module):
    def __init__(self, time_scale=.5):
        super().__init__()
        self.time_scale = float(time_scale)
        if self.time_scale <= 0:
            raise ValueError('time_scale must be positive')
        self.local_reader = LocalBoxReader()
        self.box_encoder = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(96, 64), nn.LayerNorm(64), nn.GELU())
        self.output = nn.Linear(64, 3)
        for layer in (self.box_encoder[0], self.fusion[0]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor([0., 0., math.log(.1/.9)]))

    def metadata(self, candidates, pool, allowed, inside, support, batch, prior):
        """32维同源摘要；没有历史神经评分、教师标记或GT。"""
        box = candidates.boxes.detach()
        size = batch['box_size'].detach().to(box)
        axes = prior.axes.detach().to(box)
        # world差分在float64中完成，避免大世界坐标先转float32损失精度。
        anchor = batch['anchor_box'].detach().double()
        origin = prior.origin_world.detach().double()
        origin_local = origin.clone()
        origin_local[:, :3] -= anchor[:, :3]
        origin_local = origin_local.to(box)
        delta = box[..., :3] - origin_local[:, None, :3]
        xy = torch.einsum('bqi,bij->bqj', delta[..., :2], axes)
        event_size = torch.cat((projected_xy_size(size,origin_local[:,3],axes),size[:,2:3]),-1)
        relative_event = torch.cat((xy, delta[..., 2:3]), -1) / event_size[:, None]
        axis_yaw = torch.atan2(axes[:, 1, 0], axes[:, 0, 0])
        angle = box[..., 3] - origin_local[:, None, 3]
        event_yaw = torch.stack((angle.sin(), angle.cos()), -1)
        pbox = prior.box.detach().to(box)
        pd = box[..., :3] - pbox[:, None, :3]
        pxy = torch.einsum('bqi,bij->bqj', pd[..., :2], axes)
        # 物体半尺寸沿范围轴投影；range决定相对prior的实际核验尺度。
        theta = pbox[:, 3] - axis_yaw
        c, s = theta.cos().abs(), theta.sin().abs()
        half = .5 * torch.stack((c*size[:, 0]+s*size[:, 1], s*size[:, 0]+c*size[:, 1]), -1)
        ranges = prior.range_xy.detach().to(box)
        relative_prior = torch.cat((pxy/(ranges+half).clamp_min(1e-6)[:, None],
                                    pd[..., 2:3]/size[:, None, 2:3]), -1)
        angle = box[..., 3] - pbox[:, None, 3]
        prior_yaw = torch.stack((angle.sin(), angle.cos()), -1)
        source = box.new_tensor([[1,0,0], [0,1,0], [0,0,1], [0,0,1]])[None].expand(len(box), -1, -1)
        range_summary = torch.log1p(ranges / half.clamp_min(1e-6))
        age = torch.stack((prior.horizon, prior.velocity_age), -1).detach().to(box)
        age = torch.log1p(age.clamp_min(0)/self.time_scale)
        flags = torch.stack((prior.prior_valid, prior.motion_valid), -1).detach().to(box)
        count = inside.sum(-1)
        fg_count = support.sum(-1)
        raw_probability = pool['probability'][:, None].expand_as(inside)
        p = raw_probability.clamp(1e-6, 1.-1e-6)
        mean = torch.where(inside, raw_probability, 0.).sum(-1) / count.clamp_min(1)
        entropy = -(p*p.log()+(1-p)*(1-p).log()) / math.log(2.)
        entropy = torch.where(inside, entropy, 0.).sum(-1) / count.clamp_min(1)
        counts = torch.stack((torch.log1p(count.to(box))/math.log(1281.),
            torch.log1p(fg_count.to(box))/math.log(1281.), mean, entropy), -1)
        shared = torch.cat((range_summary, age, flags, prior.geometry.detach().to(box),
                            prior.initial.detach().to(box)[:, None]), -1)
        result = torch.cat((relative_event, event_yaw, relative_prior, prior_yaw, source,
                            shared[:, None, :6].expand(-1, 4, -1), counts,
                            shared[:, None, 6:].expand(-1, 4, -1)), -1)
        if result.shape != (len(box), 4, 32) or not bool(torch.isfinite(result).all()):
            raise ValueError('v36 verifier metadata must be finite [B,4,32]')
        return result.detach()

    def forward(self, candidates, observation, evidence, batch, prior):
        pool = build_current_pool(observation, evidence, batch)
        authorized = candidate_authorization(pool, evidence.members, candidates.valid, q0_extension=True)
        inside = authorized & inside_final_boxes(pool['xyz'], candidates.boxes, batch['box_size'])
        support = inside & (pool['probability'][:, None] >= .5)
        local, _ = self.local_reader(pool, candidates.boxes, authorized, batch['box_size'])
        feature = self.box_encoder(local.reshape(len(local), 4, 512))
        meta = self.metadata(candidates, pool, authorized, inside, support, batch, prior)
        logits = self.output(self.fusion(torch.cat((feature, meta), -1)))
        return VerificationOutput(logits=logits, support_mask=support.detach(),
            support_ids=pool['ids'], support_xyz=pool['xyz'],
            support_probabilities=pool['probability'], authorized=authorized.detach())
