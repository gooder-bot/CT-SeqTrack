"""只读取已提交状态的强 B1；不再包含 acquisition head 或点云统计。"""
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from models.ct_v38.cfc import FullGatedCfCCell
from models.ct_v38.motion import motion_aligned_axes, motion_aligned_covariance


@dataclass
class PriorContext:
    feature: torch.Tensor
    mean_xy: torch.Tensor
    log_sigma: torch.Tensor
    valid: torch.Tensor
    direction_xy: torch.Tensor
    kinematic_xy: torch.Tensor
    envelope: torch.Tensor
    unit_residual: torch.Tensor
    box: torch.Tensor
    context_valid: Optional[torch.Tensor] = None


def canonicalize_state(batch):
    """状态前向不要求 points；过滤辅助标签，避免把未来监督带入先验。"""
    required = ('history_boxes', 'history_times', 'history_valid', 'current_dt', 'fallback_box')
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError('v39 state missing: ' + ', '.join(missing))
    result = {key: batch[key] for key in required}
    if 'history_pair_valid' in batch:
        result['history_pair_valid'] = batch['history_pair_valid']
    boxes = result['history_boxes']
    if not torch.is_tensor(boxes) or boxes.ndim != 3 or boxes.shape[1:] != (3, 4):
        raise ValueError('v39 requires history_boxes[B,3,4]')
    b = len(boxes)
    for key in ('history_times', 'history_valid'):
        if not torch.is_tensor(result[key]) or result[key].shape != (b, 3):
            raise ValueError(key + ' must be [B,3]')
    if result['current_dt'].numel() != b or result['fallback_box'].shape != (b, 4):
        raise ValueError('current_dt/fallback_box must be [B]/[B,4]')
    if 'history_pair_valid' in result and result['history_pair_valid'].shape != (b, 2):
        raise ValueError('history_pair_valid must be [B,2]')
    for key in ('history_boxes', 'history_times', 'current_dt', 'fallback_box'):
        result[key] = result[key].to(device=boxes.device, dtype=torch.float32)
        if not bool(torch.isfinite(result[key]).all()):
            raise ValueError(key + ' must be finite')
    if bool((result['current_dt'] <= 0).any()):
        raise ValueError('current_dt must be positive')
    return result


class PhysicalTimePrior(nn.Module):
    """保留 v38 的一步运动数学，live 128d context 供 B0 和 M 共用。"""
    def __init__(self, time_scale=.5, temporal_backend='cfc'):
        super().__init__()
        if time_scale <= 0 or temporal_backend not in ('cfc', 'gru'):
            raise ValueError('positive time_scale and cfc/gru backend required')
        self.time_scale = float(time_scale)
        self.temporal_backend = str(temporal_backend)
        self.step_projection = nn.Sequential(nn.Linear(9, 64), nn.LayerNorm(64), nn.ReLU())
        with torch.random.fork_rng(devices=[]):
            self.temporal_cell = (FullGatedCfCCell(input_size=64, hidden_size=128, backbone_units=105)
                                 if temporal_backend == 'cfc' else nn.GRUCell(64, 128))
        self.context = nn.Sequential(nn.Linear(130, 128), nn.ReLU())
        self.mean_head = nn.Linear(128, 2)
        self.sigma_head = nn.Linear(128, 2)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, math.log(math.expm1(.5 - .1)))

    def forward(self, batch):
        batch = canonicalize_state(batch)
        # 几何/运动计算始终 FP32；B0 的局部 BF16 scope 不得污染本模块。
        with torch.autocast(device_type=batch['history_boxes'].device.type, enabled=False):
            return self._forward(batch)

    def _forward(self, batch):
        boxes = batch['history_boxes']
        times = batch['history_times'].to(boxes)
        exists = batch['history_valid'].bool()
        raw_dt = batch['current_dt'].to(boxes).reshape(-1)
        gap = times[:, 1:] - times[:, :-1]
        pair_valid = exists[:, :-1] & exists[:, 1:] & (gap > 0)
        if 'history_pair_valid' in batch:
            pair_valid = pair_valid & batch['history_pair_valid'].bool()
        gap = gap.clamp_min(1e-3)
        dt = raw_dt.clamp_min(1e-3)
        displacement = boxes[:, 1:, :2] - boxes[:, :-1, :2]
        velocity = displacement / gap[..., None]
        angle = boxes[:, 1:, 3] - boxes[:, :-1, 3]
        steps = torch.cat((velocity, displacement, angle.sin()[..., None],
            (angle.cos() - 1)[..., None], torch.log1p(gap / self.time_scale)[..., None],
            (dt[:, None] / gap)[..., None], pair_valid.to(boxes)[..., None]), -1)
        projected = self.step_projection(steps * pair_valid[..., None])
        hidden = boxes.new_zeros((boxes.shape[0], 128))
        for index in range(2):
            proposed = (self.temporal_cell(projected[:, index], hidden, gap[:, index] / self.time_scale)
                        if self.temporal_backend == 'cfc'
                        else self.temporal_cell(projected[:, index], hidden))
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
        fallback = batch['fallback_box'].to(boxes).detach()
        endpoint = torch.cat((mean.detach(), fallback[:, 2:]), -1)
        endpoint = torch.where(valid[:, None], endpoint, fallback)
        return PriorContext(context, mean, uncertainty['log_sigma_parallel_perp'], valid,
                            direction, kinematic, envelope, unit, endpoint,
                            context_valid=torch.ones_like(valid))
