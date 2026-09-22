"""v32 B0：anchor 局部测量上的粗定位与 SeqTrack PointNet 特征。

三种 PointNet 保留 SeqTrack 的层宽与布局，
只计算有效测量，不依赖 PointNet++ CUDA 扩展。
前景概率仅作为 detached 聚合权重；XYZ 从不乘概率。公开框还原世界轴。
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from utils.masked_observation import mask_values, masked_max_pool, masked_sequence
from .contracts import ObservationFeatures
from .geometry import anchor_yaw, rotate_xyz, transform_boxes


def unique_valid_mask(valid: Tensor, point_ids: Tensor | None = None) -> Tensor:
    """同一帧相同 raw ID 仅首槽有效；不同帧的 ID 互不比较。"""
    valid = valid.bool()
    if point_ids is None:
        return valid
    if point_ids.shape != valid.shape:
        raise ValueError("point_ids and point_valid must have the same shape")
    if point_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("raw point IDs must be integer tensors")
    ids = point_ids.to(device=valid.device, dtype=torch.long)
    valid = valid & (ids >= 0)
    keys = ids.masked_fill(~valid, torch.iinfo(torch.long).max)
    order = torch.argsort(keys, dim=-1, stable=True)
    sorted_ids = keys.gather(-1, order)
    sorted_valid = valid.gather(-1, order)
    first = torch.cat((torch.ones_like(sorted_valid[..., :1]),
                       sorted_ids[..., 1:] != sorted_ids[..., :-1]), dim=-1)
    inverse = torch.argsort(order, dim=-1)
    return (sorted_valid & first).gather(-1, inverse)


def box_corners_xyz(boxes: Tensor, box_size: Tensor) -> Tensor:
    """世界轴平移坐标；size 为物体轴 L/W/H，yaw 为绝对弧度。

    boxes [...,4]、box_size [...,3] 可广播；返回 [...,8,3]。
    只对物体角点偏移应用 yaw，不旋转已经在世界轴中的中心。
    """
    signs = boxes.new_tensor([[-1., -1., -1.], [-1., -1., 1.],
                              [-1., 1., -1.], [-1., 1., 1.],
                              [1., -1., -1.], [1., -1., 1.],
                              [1., 1., -1.], [1., 1., 1.]])
    offsets = box_size.unsqueeze(-2) * signs * .5
    sine, cosine = boxes[..., 3:4].sin(), boxes[..., 3:4].cos()
    ox, oy, oz = offsets.unbind(-1)
    rx, ry = cosine * ox - sine * oy, sine * ox + cosine * oy
    rotated = torch.stack((rx, ry, oz.expand_as(rx)), dim=-1)
    return boxes[..., None, :3] + rotated


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, 1),
                         nn.BatchNorm1d(out_channels), nn.ReLU())


def _linear_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_channels, out_channels),
                         nn.BatchNorm1d(out_channels), nn.ReLU())


def _pooled_tokens(value: Tensor, valid: Tensor, count: int):
    """真实点压紧后做不重叠 ragged max；稀疏帧一真实点对应一 token。

    按原始有效槽顺序分桶，不复制点；floor 边界避免 AdaptiveMaxPool
    在不整除时让同一点进入相邻两桶。固定 gather/max 保持 CUDA 确定性。
    """
    count = int(count)
    if count < 1 or value.ndim != 3 or valid.shape != (value.shape[0], value.shape[-1]):
        raise ValueError('ragged pooling requires [B,C,N], [B,N] and positive token count')
    batch, channels, slots = value.shape
    if slots < 1:
        raise ValueError('ragged pooling requires positive point slots')
    valid = valid.bool()
    order = torch.argsort((~valid).long(), dim=-1, stable=True)
    compact = value.gather(2, order[:, None].expand(-1, channels, -1))
    measured = valid.sum(-1)
    token_count = measured.clamp_max(count)
    token = torch.arange(count, device=value.device)[None]
    denominator = token_count.clamp_min(1)[:, None]
    start = torch.div(token * measured[:, None], denominator, rounding_mode='floor')
    stop = torch.div((token + 1) * measured[:, None], denominator, rounding_mode='floor')
    width = (slots + count - 1) // count
    positions = start[..., None] + torch.arange(width, device=value.device)
    token_valid = token < token_count[:, None]
    member_valid = token_valid[..., None] & (positions < stop[..., None])
    indices = positions.clamp(0, slots - 1).reshape(batch, 1, -1).expand(-1, channels, -1)
    grouped = compact.gather(2, indices).reshape(batch, channels, count, width)
    pooled = grouped.masked_fill(~member_valid[:, None], -torch.inf).max(-1).values
    return mask_values(pooled, token_valid), token_valid


class SegPointNet(nn.Module):
    """原 SegPointNet：14→64/64/64/128/1024→512/256/128/128→11。"""

    def __init__(self):
        super().__init__()
        widths = (14, 64, 64, 64, 128, 1024)
        self.seq_per_point = nn.ModuleList(
            _conv_block(i, o) for i, o in zip(widths[:-1], widths[1:]))
        widths = (1088, 512, 256, 128, 128)
        self.seq_per_point2 = nn.ModuleList(
            _conv_block(i, o) for i, o in zip(widths[:-1], widths[1:]))
        self.fc = nn.Conv1d(128, 11, 1)

    def forward(self, value: Tensor, valid: Tensor):
        for index, layer in enumerate(self.seq_per_point):
            value, _ = masked_sequence(layer, value, valid)
            if index == 1:
                point_features = value
        pooled, _ = masked_max_pool(value, valid)
        value = torch.cat((point_features, pooled.expand(-1, -1, value.shape[-1])), dim=1)
        for layer in self.seq_per_point2:
            value, _ = masked_sequence(layer, value, valid)
        return mask_values(self.fc(value), valid), point_features


class MiniPointNet(nn.Module):
    """原 MiniPointNet 层宽；概率加权在 per-point latent 后、max前。"""

    def __init__(self):
        super().__init__()
        widths = (13, 64, 128, 256, 512)
        self.per_point = nn.ModuleList(
            _conv_block(i, o) for i, o in zip(widths[:-1], widths[1:]))
        self.hidden = nn.ModuleList((_linear_block(512, 512), _linear_block(512, 256)))

    def forward(self, value: Tensor, valid: Tensor, foreground: Tensor):
        for layer in self.per_point:
            value, _ = masked_sequence(layer, value, valid)
        # value 已经过 ReLU；此处加权保持原始几何输入，不产生缩放后的伪坐标。
        value = value * foreground.detach()[:, None]
        value, _ = masked_max_pool(value, valid)
        value = value.squeeze(-1)
        row_valid = valid.any(-1)
        for layer in self.hidden:
            value, _ = masked_sequence(layer, value, row_valid)
        return value


class FeaturePointNet(nn.Module):
    """原 FeaturePointNet 的 128 维 token；池化桶数与通道数分开定义。"""

    def __init__(self, token_count: int = 128):
        super().__init__()
        self.token_count = int(token_count)
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        widths = (14, 64, 64, 64, 128, 1024)
        self.seq_per_point = nn.ModuleList(
            _conv_block(i, o) for i, o in zip(widths[:-1], widths[1:]))
        widths = (1088, 512, 256, 128, 128)
        self.seq_per_point2 = nn.ModuleList(
            _conv_block(i, o) for i, o in zip(widths[:-1], widths[1:]))
        self.fc = nn.Conv1d(128, 128, 1)

    def forward(self, value: Tensor, valid: Tensor):
        for index, layer in enumerate(self.seq_per_point):
            value, _ = masked_sequence(layer, value, valid)
            if index == 1:
                point_features = value
        pooled, _ = masked_max_pool(value, valid)
        tokens, token_valid = _pooled_tokens(point_features, valid, self.token_count)
        value = torch.cat((tokens, pooled.expand(-1, -1, self.token_count)), dim=1)
        for layer in self.seq_per_point2:
            value, _ = masked_sequence(layer, value, token_valid)
        return mask_values(self.fc(value), token_valid), token_valid


class B0Observation(nn.Module):
    """一次 B0 前向产生粗定位和两条后续分支需要的真实点特征。"""

    def __init__(self, token_count: int = 128):
        super().__init__()
        self.seg_pointnet = SegPointNet()
        self.mini_pointnet = MiniPointNet()
        self.feature_pointnet = FeaturePointNet(token_count)
        # 与 SeqTrack 相同的 256 维粗定位输入；质量摘要仅保留给后续接口。
        self.coarse_box_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 5))
        # XYZ + sin/cos；初始化 yaw=0，消除零向量 atan2 和仅 sin 的角度歧义。
        with torch.no_grad():
            self.coarse_box_head[-1].weight[3:].zero_()
            self.coarse_box_head[-1].bias[3:].copy_(torch.tensor([0., 1.]))

    @staticmethod
    def measurement_mask(batch: Mapping[str, Tensor]) -> Tensor:
        points, valid = batch['points'], batch['point_valid']
        if points.ndim != 4 or points.shape[1] != 4 or points.shape[-1] != 5:
            raise ValueError("B0 points must be [B,4,N,5]")
        if valid.shape != points.shape[:-1] or points.shape[2] < 1:
            raise ValueError("B0 point_valid must match nonempty point slots")
        history_valid = batch['history_valid'].to(points.device).bool()
        if history_valid.shape != (len(points), 3):
            raise ValueError("history_valid must be [B,3]")
        exists = torch.cat((history_valid, torch.ones_like(history_valid[:, :1])), dim=1)
        return unique_valid_mask(valid.to(points.device).bool() & exists[..., None],
                                 batch.get('point_ids'))

    def forward(self, batch: Mapping[str, Tensor]) -> ObservationFeatures:
        points = batch['points']
        valid = self.measurement_mask(batch)
        batch_size, frames, count, _ = points.shape
        boxes = batch['history_boxes'].to(points)
        size = batch['box_size'].to(points)
        if boxes.shape != (batch_size, 3, 4) or size.shape != (batch_size, 3):
            raise ValueError("history_boxes/box_size must be [B,3,4]/[B,3]")
        if not bool(torch.isfinite(points[valid]).all()):
            raise ValueError("valid measurements must be finite")
        if not bool(torch.isfinite(size).all()) or not bool((size > 0).all()):
            raise ValueError("box_size must be finite positive L/W/H")
        history_valid = batch['history_valid'].to(points.device).bool()
        if not bool(torch.isfinite(boxes[history_valid]).all()):
            raise ValueError("valid history boxes must be finite")
        yaw_anchor = anchor_yaw(batch, points)
        boxes = torch.where(history_valid[..., None], boxes, 0.).detach()
        boxes = transform_boxes(boxes, yaw_anchor)
        boxes = torch.where(history_valid[..., None], boxes, 0.)
        clean = torch.where(valid[..., None], points, 0.)
        clean = torch.cat((rotate_xyz(clean[..., :3], yaw_anchor), clean[..., 3:]), dim=-1)
        # BC 的前向先验仅来自输入历史框；当前未知框没有 GT BC 输入。
        corners = box_corners_xyz(boxes, size[:, None])
        landmarks = torch.cat((boxes[..., None, :3], corners), dim=-2)
        history_bc = torch.linalg.vector_norm(
            clean[:, :3, :, None, :3] - landmarks[:, :, None], dim=-1)
        candidate_bc = torch.cat((history_bc, points.new_zeros(batch_size, 1, count, 9)), dim=1)
        candidate_bc = torch.where(valid[..., None], candidate_bc, 0.)
        point_input = torch.cat((clean, candidate_bc), dim=-1)
        flat_valid = valid.reshape(batch_size, frames * count)
        flat_input = point_input.reshape(batch_size, frames * count, 14).transpose(1, 2)
        segmentation, features = self.seg_pointnet(flat_input, flat_valid)
        logits = segmentation[:, :2].transpose(1, 2).reshape(batch_size, frames, count, 2)
        bc = segmentation[:, 2:].transpose(1, 2).reshape(batch_size, frames, count, 9)
        foreground = logits.softmax(-1)[..., 1] * valid.to(points.dtype)
        mini_input = torch.cat((clean[..., :4], bc), dim=-1)
        pooled = self.mini_pointnet(
            mini_input.reshape(batch_size, frames * count, 13).transpose(1, 2),
            flat_valid, foreground.reshape(batch_size, frames * count))
        per_frame_count = valid.sum(-1).to(points.dtype)
        current_count = per_frame_count[:, -1]
        current_valid = current_count > 0
        sequence_valid = flat_valid.any(-1)
        probability = foreground[:, -1].detach().clamp(1e-6, 1. - 1e-6)
        entropy = -(probability * probability.log() + (1. - probability) * (1. - probability).log())
        entropy = (entropy * valid[:, -1]).sum(-1) / current_count.clamp_min(1.)
        quality = torch.stack((
            torch.log1p(current_count) / math.log1p(count),
            torch.log1p(per_frame_count[:, :3].sum(-1)) / math.log1p(3 * count),
            entropy / math.log(2.), history_valid.to(points.dtype).mean(-1)), dim=-1).detach()
        coarse, _ = masked_sequence(self.coarse_box_head, pooled, sequence_valid)
        nonzero = coarse[:, 3].square() + coarse[:, 4].square() > 1e-12
        yaw = torch.atan2(torch.where(nonzero, coarse[:, 3], 0.),
                          torch.where(nonzero, coarse[:, 4], 1.))
        coarse = torch.cat((coarse[:, :3], yaw[:, None]), dim=-1)
        coarse = transform_boxes(coarse, yaw_anchor, to_world=True)
        coarse = torch.where(sequence_valid[:, None], coarse, 0.)
        source, source_valid = self.feature_pointnet(
            point_input.reshape(batch_size * frames, count, 14).transpose(1, 2),
            valid.reshape(batch_size * frames, count))
        return ObservationFeatures(
            coarse_box=coarse,
            point_features=features.transpose(1, 2).reshape(batch_size, frames, count, 64),
            source_tokens=source.transpose(1, 2).reshape(batch_size, frames, -1, 128),
            source_valid=source_valid.reshape(batch_size, frames, -1),
            segmentation_logits=logits, bc_prediction=bc,
            foreground_probability=foreground, quality=quality, current_valid=current_valid,
            sequence_valid=sequence_valid)
