"""B1：物理位移与同帧可微上下文，不承担锚点误差回归。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.ct_v2.cfc import FullGatedCfCCell
from models.ct_v2.motion import motion_aligned_axes, motion_aligned_covariance
from .contracts import PriorContext


def _column(batch, key, reference, default=0.0):
    value = batch.get(key)
    if value is None:
        return reference.new_full((reference.shape[0], 1), float(default))
    return value.to(reference).reshape(reference.shape[0], 1)


def acquisition_features(batch, pair_valid, time_scale):
    """21 个因果统计，只看已获得的 B0 crop/历史，绝不读取 GT。"""
    points, valid = batch['points'], batch['point_valid'].bool()
    counts = valid.sum(-1).to(points.dtype)
    xyz = points[:, -1, :, :2]
    current = valid[:, -1, :, None]
    low = xyz.masked_fill(~current, float('inf')).amin(1)
    high = xyz.masked_fill(~current, -float('inf')).amax(1)
    extent = torch.where(current.any(1), high - low, torch.zeros_like(low))
    size = batch['box_size'].to(points).clamp_min(.1)
    velocity = batch.get('trusted_velocity', points.new_zeros((points.shape[0], 3)))
    stats = torch.cat((
        torch.log1p(counts), torch.log1p(size),
        batch['history_valid'].to(points), pair_valid.to(points),
        torch.log1p(batch['current_dt'].to(points).reshape(-1, 1) / time_scale),
        _column(batch, 'previous_quality', points),
        torch.log1p(_column(batch, 'weak_age', points).clamp_min(0)),
        torch.log1p(_column(batch, 'supported_age', points).clamp_min(0)),
        velocity.to(points)[:, :2] / 10.,
        _column(batch, 'previous_innovation', points), extent / size[:, :2],
    ), -1)
    if stats.shape[1] != 21:
        raise ValueError('v31 acquisition statistics must have 21 columns')
    return stats.detach()


class PhysicalTimePrior(nn.Module):
    """沿用 CfC/运动学包络，显式支持独立 transition mask。"""

    def __init__(self, time_scale=.5):
        super().__init__()
        if time_scale <= 0:
            raise ValueError('time_scale must be positive')
        self.time_scale = float(time_scale)
        self.step_projection = nn.Sequential(nn.Linear(9, 64), nn.LayerNorm(64), nn.ReLU())
        self.cfc = FullGatedCfCCell(input_size=64, hidden_size=128, backbone_units=105)
        self.context = nn.Sequential(nn.Linear(130, 128), nn.ReLU())
        self.mean_head = nn.Linear(128, 2)
        self.sigma_head = nn.Linear(128, 2)
        self.acquisition_head = nn.Sequential(
            nn.Linear(149, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, math.log(math.expm1(.5 - .1)))
        nn.init.zeros_(self.acquisition_head[-1].weight)
        with torch.no_grad():
            self.acquisition_head[-1].bias.copy_(torch.logit(torch.tensor((.5 / 3.75, .25 / 2.75))))

    def forward(self, batch):
        boxes = batch['history_boxes']
        times = batch['history_times'].to(boxes)
        exists = batch['history_valid'].bool()
        if boxes.ndim != 3 or boxes.shape[1:] != (3, 4) or times.shape != boxes.shape[:2]:
            raise ValueError('v31 B1 requires oldest-first history_boxes[B,3,4] and history_times[B,3]')
        raw_dt = batch['current_dt'].to(boxes).reshape(-1)
        if (not torch.isfinite(boxes).all() or not torch.isfinite(times).all()
                or not torch.isfinite(raw_dt).all() or bool((raw_dt <= 0).any())):
            raise ValueError('v31 physical history and positive dt must be finite')
        gap = times[:, 1:] - times[:, :-1]
        pair_valid = exists[:, :-1] & exists[:, 1:] & (gap > 0)
        if 'history_pair_valid' in batch:
            if batch['history_pair_valid'].shape != pair_valid.shape:
                raise ValueError('history_pair_valid must be [B,2]')
            pair_valid = pair_valid & batch['history_pair_valid'].bool()
        gap = gap.clamp_min(1e-3)
        dt = raw_dt.clamp_min(1e-3)
        displacement = boxes[:, 1:, :2] - boxes[:, :-1, :2]
        velocity = displacement / gap[..., None]
        angle = boxes[:, 1:, 3] - boxes[:, :-1, 3]
        steps = torch.cat((velocity, displacement, angle.sin()[..., None],
                           (angle.cos() - 1)[..., None],
                           torch.log1p(gap / self.time_scale)[..., None],
                           (dt[:, None] / gap)[..., None],
                           pair_valid.to(boxes)[..., None]), -1)
        projected = self.step_projection(steps * pair_valid[..., None])
        hidden = boxes.new_zeros((boxes.shape[0], 128))
        for index in range(2):
            proposed = self.cfc(projected[:, index], hidden, gap[:, index] / self.time_scale)
            hidden = torch.where(pair_valid[:, index, None], proposed, hidden)
        count = pair_valid.sum(1)
        nominal_gap = (gap * pair_valid).sum(1) / count.clamp_min(1)
        ratio = dt / nominal_gap.clamp_min(self.time_scale * .1)
        context = self.context(torch.cat((hidden, torch.log1p(dt / self.time_scale)[:, None],
                                          torch.log1p(ratio)[:, None]), -1))
        valid = count > 0
        latest_index = torch.where(pair_valid[:, 1], 1, 0)
        row = torch.arange(boxes.shape[0], device=boxes.device)
        base_velocity = velocity[row, latest_index] * valid[:, None]
        both = pair_valid.all(1)
        accel_gap = .5 * (gap[:, 0] + gap[:, 1])
        acceleration = (velocity[:, 1] - velocity[:, 0]) / accel_gap[:, None]
        acceleration = acceleration * both[:, None]
        acceleration = acceleration * (8. / acceleration.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
        kinematic = base_velocity * dt[:, None] + .25 * acceleration * dt[:, None].square()
        kinematic = kinematic * (12. / kinematic.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
        spread = (velocity[:, 1] - velocity[:, 0]).norm(dim=1) * both
        envelope = torch.stack(((.25 + .25 * dt + .5 * spread * dt).clamp_max(4),
                                (.20 + .15 * dt + .25 * spread * dt).clamp_max(3)), -1)
        direction, perpendicular, _ = motion_aligned_axes(base_velocity)
        unit = torch.tanh(self.mean_head(context)) * valid[:, None]
        mean = kinematic + direction * unit[:, :1] * envelope[:, :1] + perpendicular * unit[:, 1:] * envelope[:, 1:]
        sigma = (.1 + F.softplus(self.sigma_head(context.detach()))).clamp_max(math.exp(2.5))
        uncertainty = motion_aligned_covariance(base_velocity, sigma.log(), direction_xy=direction)
        # 获取 context 保持图；raw crop/状态统计不携带跨帧图。
        fraction = torch.sigmoid(self.acquisition_head(torch.cat((
            context, acquisition_features(batch, pair_valid, self.time_scale)), -1)))
        fallback = batch['fallback_box'].to(boxes).detach()
        endpoint = torch.cat((mean.detach(), fallback[:, 2:]), -1)
        endpoint = torch.where(valid[:, None], endpoint, fallback)
        return PriorContext(context, mean, uncertainty['log_sigma_parallel_perp'], valid,
                            fraction, direction, kinematic, envelope, unit, endpoint)


def empty_prior(batch):
    """B0 arm 无可学习 B1；仅保留可信恒速/保持位置的缺测 fallback。"""
    boxes = batch['history_boxes']
    b = boxes.shape[0]
    zeros = boxes.new_zeros((b, 2))
    direction = zeros.clone()
    direction[:, 0] = 1
    return PriorContext(boxes.new_zeros((b, 128)), zeros, zeros,
                        torch.zeros(b, dtype=torch.bool, device=boxes.device),
                        zeros, direction, zeros, zeros, zeros,
                        batch['fallback_box'].to(boxes).detach())
