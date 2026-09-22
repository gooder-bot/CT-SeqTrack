"""v32 B0 的局部几何视图；公开 XYZ 仍是 anchor 平移后的世界轴。

这里仅换轴，不再次平移。B1/B2 的原始输入、物理量和监督保持世界轴，
64/128 维学习特征不是几何向量，不经过这些变换。
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def anchor_yaw(batch: Mapping[str, Tensor], reference: Tensor) -> Tensor:
    anchor = batch['anchor_box'].to(reference)
    if anchor.shape != (len(reference), 4) or not bool(torch.isfinite(anchor).all()):
        raise ValueError('anchor_box must be finite [B,4] world XYZ/yaw')
    return anchor[:, 3].detach()


def rotate_xy(value: Tensor, yaw: Tensor, *, to_world: bool = False) -> Tensor:
    """转换 [...,2] 平面向量，batch 轴为首轴；保留输入梯度。"""
    if value.shape[0] != len(yaw) or value.shape[-1] != 2 or yaw.ndim != 1:
        raise ValueError('XY values and anchor yaw must share their batch axis')
    angle = yaw.to(value).reshape(len(yaw), *([1] * (value.ndim - 2)))
    sine, cosine = angle.sin(), angle.cos()
    if not to_world:
        sine = -sine
    x, y = value.unbind(-1)
    return torch.stack((cosine * x - sine * y, sine * x + cosine * y), dim=-1)


def rotate_xyz(value: Tensor, yaw: Tensor, *, to_world: bool = False) -> Tensor:
    if value.shape[-1] != 3:
        raise ValueError('XYZ values must have three coordinates')
    return torch.cat((rotate_xy(value[..., :2], yaw, to_world=to_world), value[..., 2:3]), -1)


def transform_boxes(boxes: Tensor, yaw: Tensor, *, to_world: bool = False) -> Tensor:
    """框中心换轴，朝向在 absolute world yaw 与 anchor-relative yaw 间转换。"""
    if boxes.shape[-1] != 4:
        raise ValueError('boxes must end in XYZ/yaw')
    offset = yaw.to(boxes).reshape(len(yaw), *([1] * (boxes.ndim - 2)))
    angle = boxes[..., 3] + offset if to_world else boxes[..., 3] - offset
    angle = torch.atan2(angle.sin(), angle.cos())
    return torch.cat((rotate_xyz(boxes[..., :3], yaw, to_world=to_world), angle[..., None]), -1)
