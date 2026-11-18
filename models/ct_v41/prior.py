"""B1：物理位移与同帧可微上下文，不承担锚点误差回归。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cfc import FullGatedCfCCell
from .motion import motion_aligned_axes, motion_aligned_covariance
from .contracts import PriorContext


def empty_prior(batch):
    """B0 的无参数占位协议；只提供合法传播框，跳过 B1。"""
    fallback = batch['fallback_box'].detach()
    if fallback.ndim != 2 or fallback.shape[-1] != 4 or not bool(torch.isfinite(fallback).all()):
        raise ValueError('B0 fallback_box must be finite [B,4]')
    b = len(fallback)
    zero = fallback.new_zeros((b, 2))
    invalid = torch.zeros(b, dtype=torch.bool, device=fallback.device)
    return PriorContext(feature=fallback.new_zeros((b, 128)), mean_xy=zero,
        log_sigma=zero.clone(), valid=invalid,
        direction_xy=zero.clone(), kinematic_xy=zero.clone(), envelope=zero.clone(),
        unit_residual=zero.clone(), box=fallback, context_valid=invalid.clone())


class PhysicalTimePrior(nn.Module):
    """共用物理时间输入和目标，仅替换 CfC/GRU 时序单元。"""

    def __init__(self, time_scale=.5, temporal_backend='cfc'):
        super().__init__()
        if time_scale <= 0:
            raise ValueError('time_scale must be positive')
        if temporal_backend not in ('cfc', 'gru'):
            raise ValueError('v31 temporal_backend must be cfc or gru')
        self.time_scale = float(time_scale)
        self.temporal_backend = str(temporal_backend)
        self.step_projection = nn.Sequential(nn.Linear(9, 64), nn.LayerNorm(64), nn.ReLU())
        # 两种 cell 参数量不同；构造时只恢复 CPU RNG，避免改变公共 heads/B2
        # 初始化，也避免为创建普通 CPU 模块而触发 CUDA 设备初始化。
        with torch.random.fork_rng(devices=[]):
            self.temporal_cell = (FullGatedCfCCell(input_size=64, hidden_size=128, backbone_units=105)
                                  if self.temporal_backend == 'cfc' else nn.GRUCell(64, 128))
        self.context = nn.Sequential(nn.Linear(130, 128), nn.ReLU())
        self.mean_head = nn.Linear(128, 2)
        self.sigma_head = nn.Linear(128, 2)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, math.log(math.expm1(.5 - .1)))

    def forward(self, batch):
        boxes = batch['history_boxes']
        times = batch['history_times'].to(boxes)
        exists = batch['history_valid'].bool()
        if boxes.ndim != 3 or boxes.shape[1:] != (3, 4) or times.shape != boxes.shape[:2]:
            raise ValueError('v31 B1 requires oldest-first history_boxes[B,3,4] and history_times[B,3]')
        raw_dt = batch['current_dt'].to(boxes).reshape(-1)
        if not bool(torch.isfinite(boxes).all() & torch.isfinite(times).all()
                    & (torch.isfinite(raw_dt) & (raw_dt > 0)).all()):
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
            # GRU 同样读取含 log(gap)、dt/gap 的公共 64d step；仅 CfC
            # 还在其定义的连续时间 gate 中直接使用 elapsed time。
            proposed = (self.temporal_cell(projected[:, index], hidden, gap[:, index] / self.time_scale)
                        if self.temporal_backend == 'cfc' else self.temporal_cell(projected[:, index], hidden))
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
