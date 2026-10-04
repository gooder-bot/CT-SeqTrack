"""测量事件物理传播、独立位置范围及两个预登记运动对照。"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from models.ct_v31.cfc import FullGatedCfCCell
from models.ct_v31.motion import motion_aligned_axes
from .contracts import PriorPrediction
from .randomness import component_rng


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('nonfinite v36 ' + name)
    return value


def _last(value, index):
    return value[torch.arange(len(index), device=index.device), index]


def projected_box_xy(box_size, yaw, axes):
    """物体完整 L/W 在冻结运动轴上的矩形投影；位移/速度/损失共用量纲。"""
    size = box_size.to(axes)
    yaw = yaw.to(axes)
    c, s = yaw.cos(), yaw.sin()
    object_axes = torch.stack((c, -s, s, c), -1).reshape(-1, 2, 2)
    return torch.bmm(axes.transpose(1, 2), object_axes).abs().bmm(size[:, :2, None]).squeeze(-1)


def decayed_acceleration_displacement(velocity, acceleration, horizon, tau):
    """秒制解析积分；小 x 展开避免 expm1 后的二次相消。"""
    horizon = horizon.reshape(-1, 1).to(velocity)
    tau = torch.as_tensor(tau, device=velocity.device, dtype=velocity.dtype)
    if bool((horizon < 0).any()) or bool((tau <= 0).any()):
        raise ValueError('v36 horizon must be nonnegative and tau positive')
    x = horizon / tau
    small = x < 1e-3
    # 大 x 分支不计算高次幂，避免未选分支的 inf 在 backward 中污染梯度。
    xs = torch.where(small, x, torch.zeros_like(x))
    series = xs.square() * (.5 - xs / 6. + xs.square() / 24.)
    integral = torch.where(small, tau.square() * series,
                           tau * (horizon + tau * torch.expm1(-x)))
    return _finite(velocity * horizon + acceleration * integral, 'motion displacement')


class EventEncoder(nn.Module):
    """固定三事件重新编码；输入中没有当前查询 T、accepted anchor 或 GT。"""
    def __init__(self, time_scale=.5, temporal_backend='cfc', seed=42):
        super().__init__()
        self.time_scale, self.temporal_backend = float(time_scale), temporal_backend
        with component_rng(seed, 'prior.event_projection'):
            self.step_projection = nn.Sequential(nn.Linear(50, 64), nn.LayerNorm(64), nn.GELU())
        with component_rng(seed, 'prior.temporal'):
            self.temporal_cell = (FullGatedCfCCell(64, 128, 105) if temporal_backend == 'cfc'
                                  else nn.GRUCell(64, 128))
        with component_rng(seed, 'prior.event_context'):
            self.context = nn.Sequential(nn.Linear(151, 128), nn.LayerNorm(128), nn.GELU())

    @staticmethod
    def prepare(batch, dtype, time_scale):
        """纯几何事件摘要；E 的 CV fallback 调用它，不构造/调用事件网络。"""
        boxes = batch['event_boxes'].detach().double()
        times = batch['event_times'].detach().double()
        valid = batch['event_valid'].bool().detach()
        segments = batch['event_segment'].long().detach()
        incoming = batch['event_edge_valid'].bool().detach()
        geometry = batch['event_geometry'].detach().to(dtype=dtype)
        source = batch['event_source'].long().detach()
        initial = batch['event_initial'].bool().detach()
        frames = batch['event_frame_ids'].long().detach()
        size = batch['box_size'].detach().to(dtype=dtype)
        b = len(boxes)
        if (boxes.shape != (b, 3, 4) or times.shape != (b, 3)
                or geometry.shape != (b, 3, 8) or size.shape != (b, 3)
                or any(x.shape != (b, 3) for x in (valid, segments, incoming, source, initial, frames))):
            raise ValueError('invalid v36 event tensor shapes')
        if not bool(valid.any(1).all()) or bool((size <= 0).any()):
            raise ValueError('v36 requires a legal position event and positive box size')
        if bool((frames[valid] < 0).any()):
            raise ValueError('valid events require nonnegative frame IDs')
        _finite(boxes[valid], 'event boxes'); _finite(times[valid], 'event times')
        _finite(geometry[valid], 'event geometry'); _finite(size, 'box size')
        if bool(((source[valid] < 0) | (source[valid] > 1)).any()):
            raise ValueError('event source must be observation=0 or extension=1')
        slots = torch.arange(3, device=boxes.device)[None].expand(b, -1)
        latest = slots.masked_fill(~valid, -1).amax(1)
        origin, origin_time = _last(boxes, latest), _last(times, latest)
        origin_segment = _last(segments, latest)
        gaps = times[:, 1:] - times[:, :-1]
        adjacent = valid[:, 1:] & valid[:, :-1]
        if bool((gaps[adjacent] <= 0).any()):
            raise ValueError('event timestamps must strictly increase')
        same_segment = (segments[:, 1:] == segments[:, :-1]) & (segments[:, 1:] == origin_segment[:, None])
        edge = adjacent & incoming[:, 1:] & same_segment
        gaps_safe = torch.where(adjacent, gaps, torch.ones_like(gaps))
        delta = torch.where(adjacent[..., None], boxes[:, 1:, :2] - boxes[:, :-1, :2], 0.)
        velocities = (delta / gaps_safe[..., None]).to(dtype=dtype)
        edge_index = torch.arange(2, device=boxes.device)[None].expand(b, -1).masked_fill(~edge, -1).amax(1)
        motion_valid = edge_index >= 0
        velocity = _last(velocities, edge_index.clamp_min(0)) * motion_valid[:, None]
        direction, perpendicular, _ = motion_aligned_axes(velocity)
        axes = torch.stack((direction, perpendicular), -1).detach()
        xy_size = projected_box_xy(size, origin[:, 3], axes)
        projected_size = torch.cat((xy_size, size[:, 2:]), -1)
        relative = torch.where(valid[..., None], boxes[..., :3] - origin[:, None, :3], 0.).to(dtype=dtype)
        relative_xy = torch.einsum('bki,bij->bkj', relative[..., :2], axes)
        relative = torch.cat((relative_xy, relative[..., 2:]), -1) / projected_size[:, None]
        angle = torch.where(valid, boxes[..., 3] - origin[:, None, 3], 0.).to(dtype=dtype)
        age = torch.where(valid, origin_time[:, None] - times, 0.).to(dtype=dtype)
        onehot = F.one_hot(source.clamp(0, 1), 2).to(dtype=dtype)
        descriptor = torch.cat((relative, angle.sin()[..., None], angle.cos()[..., None],
            torch.log1p(age / time_scale)[..., None], torch.where(valid[..., None], geometry, 0.),
            initial.to(dtype)[..., None], onehot,
            torch.log(size)[:, None].expand(-1, 3, -1), valid.to(dtype)[..., None]), -1)
        descriptor = torch.where(valid[..., None], descriptor, 0.).detach()
        aligned_delta = torch.einsum('bki,bij->bkj', delta.to(dtype=dtype), axes) / xy_size[:, None]
        aligned_velocity = torch.einsum('bki,bij->bkj', velocities, axes) * time_scale / xy_size[:, None]
        yaw_delta = (boxes[:, 1:, 3] - boxes[:, :-1, 3]).to(dtype=dtype)
        yaw_delta = torch.where(adjacent, yaw_delta, 0.)
        steps = torch.cat((descriptor[:, :-1], descriptor[:, 1:], aligned_delta, aligned_velocity,
            yaw_delta.sin()[..., None], (yaw_delta.cos() - 1)[..., None],
            torch.log1p(gaps_safe.to(dtype=dtype) / time_scale)[..., None], edge.to(dtype)[..., None]), -1)
        steps = torch.where(edge[..., None], steps, 0.)
        newest = _last(descriptor, latest)
        midpoint = .5 * (times[:, 1:] + times[:, :-1])
        velocity_event_age = torch.where(motion_valid,
            origin_time - _last(midpoint, edge_index.clamp_min(0)), 0.).to(dtype=dtype)
        return dict(descriptor=newest, origin=origin, origin_time=origin_time,
            origin_frame=_last(frames, latest), geometry=_last(geometry, latest),
            initial=_last(initial, latest), velocity=velocity, axes=axes, motion_valid=motion_valid,
            velocity_event_age=velocity_event_age, steps=steps, gaps=gaps_safe.to(dtype), edge=edge,
            event_valid=valid)

    def forward(self, batch, dtype):
        result = self.prepare(batch, dtype, self.time_scale)
        projected = self.step_projection(result['steps'])
        hidden = projected.new_zeros((len(projected), 128))
        for index in range(2):
            proposed = (self.temporal_cell(projected[:, index], hidden, result['gaps'][:, index] / self.time_scale)
                        if self.temporal_backend == 'cfc' else self.temporal_cell(projected[:, index], hidden))
            # 断边不携带旧 segment hidden；尾部 padding 不抹除已有有效边。
            without_edge = torch.where(result['event_valid'][:, index + 1, None], torch.zeros_like(hidden), hidden)
            hidden = torch.where(result['edge'][:, index, None], proposed, without_edge)
        valid = result['motion_valid']
        result['context'] = self.context(torch.cat((hidden, result['descriptor'], valid.to(dtype)[:, None],
                                        torch.ones_like(valid, dtype=dtype)[:, None]), -1))
        return result


class LegacyMotionCore(nn.Module):
    """E 保留旧 accepted 三框 mean/context 数学；不构造已撤销 sigma/u 头。"""
    def __init__(self, time_scale, temporal_backend, seed):
        super().__init__()
        self.time_scale, self.temporal_backend = float(time_scale), temporal_backend
        with component_rng(seed, 'prior.legacy_core'):
            self.step_projection = nn.Sequential(nn.Linear(9, 64), nn.LayerNorm(64), nn.ReLU())
            self.temporal_cell = (FullGatedCfCCell(64, 128, 105) if temporal_backend == 'cfc'
                                  else nn.GRUCell(64, 128))
            self.context = nn.Sequential(nn.Linear(130, 128), nn.ReLU())
            self.mean_head = nn.Linear(128, 2)
            nn.init.zeros_(self.mean_head.weight); nn.init.zeros_(self.mean_head.bias)

    def forward(self, batch):
        boxes = batch['history_boxes']
        times = batch['history_times'].to(boxes)
        exists = batch['history_valid'].bool()
        dt = batch['current_dt'].to(boxes).reshape(-1)
        _finite(boxes, 'legacy history boxes'); _finite(times, 'legacy history times')
        _finite(dt, 'legacy current dt')
        if bool((dt <= 0).any()):
            raise ValueError('legacy current dt must be positive')
        dt = dt.clamp_min(1e-3)
        gap = times[:, 1:] - times[:, :-1]
        pair = exists[:, :-1] & exists[:, 1:] & (gap > 0) & batch['history_pair_valid'].bool()
        gap = gap.clamp_min(1e-3)
        displacement = boxes[:, 1:, :2] - boxes[:, :-1, :2]
        velocity = displacement / gap[..., None]
        angle = boxes[:, 1:, 3] - boxes[:, :-1, 3]
        steps = torch.cat((velocity, displacement, angle.sin()[..., None], (angle.cos() - 1)[..., None],
            torch.log1p(gap / self.time_scale)[..., None], (dt[:, None] / gap)[..., None],
            pair.to(boxes)[..., None]), -1)
        projected = self.step_projection(steps * pair[..., None])
        hidden = boxes.new_zeros((len(boxes), 128))
        for i in range(2):
            proposed = (self.temporal_cell(projected[:, i], hidden, gap[:, i] / self.time_scale)
                        if self.temporal_backend == 'cfc' else self.temporal_cell(projected[:, i], hidden))
            hidden = torch.where(pair[:, i, None], proposed, hidden)
        count = pair.sum(1)
        nominal_gap = (gap * pair).sum(1) / count.clamp_min(1)
        ratio = dt / nominal_gap.clamp_min(self.time_scale * .1)
        context = self.context(torch.cat((hidden, torch.log1p(dt / self.time_scale)[:, None],
                                         torch.log1p(ratio)[:, None]), -1))
        valid = count > 0
        latest = torch.where(pair[:, 1], 1, 0)
        base = _last(velocity, latest) * valid[:, None]
        both = pair.all(1)
        acceleration = (velocity[:, 1] - velocity[:, 0]) / (.5 * (gap[:, 0] + gap[:, 1]))[:, None]
        acceleration = acceleration * both[:, None]
        acceleration = acceleration * (8. / acceleration.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
        kinematic = base * dt[:, None] + .25 * acceleration * dt[:, None].square()
        kinematic = kinematic * (12. / kinematic.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
        spread = (velocity[:, 1] - velocity[:, 0]).norm(dim=1) * both
        envelope = torch.stack(((.25 + .25 * dt + .5 * spread * dt).clamp_max(4),
                                (.20 + .15 * dt + .25 * spread * dt).clamp_max(3)), -1)
        direction, perpendicular, _ = motion_aligned_axes(base)
        unit = torch.tanh(self.mean_head(context)) * valid[:, None]
        mean = kinematic + direction * unit[:, :1] * envelope[:, :1] + perpendicular * unit[:, 1:] * envelope[:, 1:]
        return dict(feature=context, mean=mean, valid=valid, kinematic=kinematic, envelope=envelope,
                    unit=unit, velocity=base, acceleration=acceleration)


class EventPrior(nn.Module):
    def __init__(self, time_scale=.5, backend='learned', seed=42, temporal_backend='cfc'):
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0 or backend not in ('learned', 'cv', 'legacy') or temporal_backend not in ('cfc', 'gru'):
            raise ValueError('unsupported v36 prior configuration')
        self.time_scale, self.backend = float(time_scale), backend
        if backend != 'legacy':
            self.encoder = EventEncoder(time_scale, temporal_backend, seed)
            with component_rng(seed, 'prior.query_context'):
                self.query_context = nn.Sequential(nn.Linear(130, 128), nn.LayerNorm(128), nn.GELU())
        if backend == 'learned':
            with component_rng(seed, 'prior.motion_heads'):
                self.velocity_head, self.acceleration_head = nn.Linear(128, 2), nn.Linear(128, 2)
                for head in (self.velocity_head, self.acceleration_head):
                    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
                self.tau_parameter = nn.Parameter(torch.tensor(math.log(math.expm1((1. - 1e-4) / time_scale))))
        if backend == 'legacy':
            self.legacy = LegacyMotionCore(time_scale, temporal_backend, seed)
        for name, initial in (('range_anchor', .5), ('range_rate', 1.)):
            with component_rng(seed, 'prior.' + name):
                head = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 2))
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, math.log(math.expm1(initial)))
            setattr(self, name + '_head', head)

    def forward(self, batch):
        dtype = batch['points'].dtype if 'points' in batch else next(self.parameters()).dtype
        event = (EventEncoder.prepare(batch, dtype, self.time_scale) if self.backend == 'legacy'
                 else self.encoder(batch, dtype))
        current = batch['current_time'].detach().double().reshape(-1)
        anchor = batch['anchor_box'].detach().double()
        _finite(current, 'query timestamp'); _finite(anchor, 'accepted anchor')
        horizon = (current - event['origin_time']).to(dtype=dtype)
        if bool((horizon < 0).any()):
            raise ValueError('query time cannot precede latest event')
        velocity_age = horizon + event['velocity_event_age']
        origin, origin_frame = event['origin'], event['origin_frame']
        context, axes, valid = event.get('context'), event['axes'], event['motion_valid']
        velocity, acceleration = event['velocity'], torch.zeros_like(event['velocity'])
        tau = velocity.new_ones((len(velocity),))
        if self.backend == 'learned':
            size_scale = projected_box_xy(batch['box_size'], origin[:, 3], axes)
            velocity = velocity + torch.einsum('bij,bj->bi', axes,
                self.velocity_head(context) * size_scale / self.time_scale) * valid[:, None]
            acceleration = torch.einsum('bij,bj->bi', axes,
                self.acceleration_head(context) * size_scale / self.time_scale ** 2) * valid[:, None]
            tau = (1e-4 + self.time_scale * F.softplus(self.tau_parameter)).expand(len(velocity))
        mean = decayed_acceleration_displacement(velocity, acceleration, horizon, tau[:, None])
        feature = (None if self.backend == 'legacy' else self.query_context(torch.cat((
            context, torch.log1p(horizon / self.time_scale)[:, None],
            torch.log1p(velocity_age / self.time_scale)[:, None]), -1)))
        geometry, initial = event['geometry'], event['initial']
        range_context, legacy = context, None
        world_z_yaw = origin[:, 2:].clone()
        if self.backend == 'legacy':
            legacy = self.legacy(batch)
            live = legacy['valid']
            if 'current_frame' not in batch:
                raise KeyError('legacy range origin requires current_frame')
            origin = torch.where(live[:, None], anchor, origin)
            origin_frame = torch.where(live, batch['current_frame'].long().reshape(-1) - 1, origin_frame)
            horizon = torch.where(live, batch['current_dt'].to(mean).reshape(-1), horizon)
            mean = torch.where(live[:, None], legacy['mean'], mean)
            # E 明确保留原 query-dependent context；不额外事件网络重编码。
            feature = legacy['feature']
            range_context = legacy['feature']
            velocity = torch.where(live[:, None], legacy['velocity'], velocity)
            acceleration = torch.where(live[:, None], legacy['acceleration'], acceleration)
            valid = torch.where(live, live, valid)
            direction, perpendicular, _ = motion_aligned_axes(velocity)
            axes = torch.stack((direction, perpendicular), -1).detach()
            history_times = batch['history_times'].to(mean)
            pair = batch['history_pair_valid'].bool() & batch['history_valid'][:, :-1].bool() & batch['history_valid'][:, 1:].bool()
            edge_index = torch.where(pair[:, 1], 1, 0)
            midpoint_age = -.5 * (history_times[:, 1:] + history_times[:, :-1])
            velocity_age = torch.where(live, _last(midpoint_age, edge_index), velocity_age)
            if 'history_geometry' in batch:
                geometry = torch.where(live[:, None], batch['history_geometry'][:, -1].to(geometry), geometry)
            initial = torch.where(live, batch.get('history_is_initial',
                torch.zeros_like(batch['history_valid']))[:, -1].bool(), initial)
        range_input = range_context.detach()
        r_anchor = F.softplus(self.range_anchor_head(range_input))
        r_rate = F.softplus(self.range_rate_head(range_input))
        range_for_loss = r_anchor.detach() + horizon[:, None] * r_rate
        range_xy = (r_anchor + horizon[:, None] * r_rate).detach()
        world = torch.cat((origin[:, :2] + mean.detach().double(), world_z_yaw), -1).detach()
        local = world.clone()
        local[:, :3] -= anchor[:, :3]
        local = local.to(dtype=dtype)
        for name, value in (('prior world', world), ('position range', range_xy), ('prior feature', feature),
                            ('velocity', velocity), ('acceleration', acceleration), ('tau', tau)):
            _finite(value, name)
        return PriorPrediction(feature=feature, event_feature=range_context, box=local,
            world_box=world, mean_xy=mean, origin_world=origin.detach(), origin_frame=origin_frame,
            horizon=horizon, velocity_age=velocity_age.detach(), axes=axes,
            prior_valid=torch.ones_like(valid), motion_valid=valid,
            range_anchor=r_anchor, range_rate=r_rate, range_xy=range_xy, range_for_loss=range_for_loss,
            geometry=geometry.detach(), initial=initial.detach(), velocity=velocity,
            acceleration=acceleration, tau=tau,
            kinematic_xy=None if legacy is None else legacy['kinematic'],
            envelope=None if legacy is None else legacy['envelope'],
            unit_residual=None if legacy is None else legacy['unit'],
            legacy_valid=None if legacy is None else legacy['valid'])


__all__ = ['EventPrior', 'EventEncoder', 'LegacyMotionCore', 'decayed_acceleration_displacement', 'projected_box_xy']
