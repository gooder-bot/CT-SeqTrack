"""B0 coarse 与 decoder 共用的合法历史描述；不读取训练标签。"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def history_descriptor(batch: Mapping[str, Tensor], local_history: Tensor,
                       history_valid: Tensor, size: Tensor, support: Tensor,
                       time_scale: float, *, include_initial: bool = False) -> Tensor:
    """每槽几何/真实年龄/预测支持/存在性10维，v35追加真实首框位。"""
    b = len(size)
    if (local_history.shape != (b, 3, 4) or history_valid.shape != (b, 3)
            or support.shape != (b, 3, 3)):
        raise ValueError('history context requires three aligned history slots')
    times = batch.get('frame_times')
    if times is None or times.shape != (b, 4):
        raise ValueError('history context requires physical frame_times[B,4]')
    times = times.detach().to(size)
    age = times[:, -1:] - times[:, :3]
    if time_scale <= 0 or not bool(torch.isfinite(times).all()) or bool((age[history_valid] <= 0).any()):
        raise ValueError('history ages and time_scale must be finite and positive')
    history = local_history.detach()
    angle = history[..., 3]
    descriptor = torch.cat((history[..., :3] / size.detach()[:, None],
        angle.sin()[..., None], angle.cos()[..., None],
        torch.log1p(age.clamp_min(0.) / time_scale)[..., None], support.detach()), dim=-1)
    descriptor = torch.where(history_valid[..., None], descriptor, 0.)
    descriptor = torch.cat((descriptor, history_valid[..., None].to(descriptor)), dim=-1)
    if include_initial:
        initial = batch.get('history_is_initial')
        if initial is None or initial.shape != (b, 3):
            raise ValueError('v35 requires history_is_initial[B,3]')
        initial = initial.detach().to(device=size.device).bool() & history_valid
        descriptor = torch.cat((descriptor, initial[..., None].to(descriptor)), dim=-1)
    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError('valid history context must be finite')
    return descriptor.reshape(b, -1)
