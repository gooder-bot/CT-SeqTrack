"""v30：真实测量 mask 的确定性 PointNet 归一化与固定分桶池化。"""

import torch
from torch import nn
from torch.nn import functional as F

from utils.deterministic_pooling import DeterministicMaxPool1d


def mask_values(value, valid):
    mask = valid[:, None] if value.ndim == 3 else valid.reshape(-1, 1)
    return torch.where(mask, value, torch.zeros_like(value))


def masked_batch_norm(value, valid, module):
    """复用原 BN 参数/状态；无效值连统计与梯度都不进入网络。"""
    valid = valid.bool()
    count = int(valid.sum().item())
    if count == valid.numel() and (not module.training or count >= 2):
        return module(value)
    clean = mask_values(value, valid)
    if not module.training or count < 2:
        result = F.batch_norm(clean, module.running_mean, module.running_var,
                              module.weight, module.bias, False, 0., module.eps)
        return mask_values(result, valid)
    # [B,N,C] 的确定性 sum，不使用 scatter。统计分母只计真实测量槽。
    rows = clean.movedim(1, -1).reshape(-1, clean.shape[1]) if clean.ndim == 3 else clean
    weights = valid.reshape(-1, 1).to(clean.dtype)
    mean = rows.sum(0) / count
    variance = ((rows - mean).square() * weights).sum(0) / count
    if module.track_running_stats:
        with torch.no_grad():
            module.num_batches_tracked.add_(1)
            factor = (1. / float(module.num_batches_tracked)
                      if module.momentum is None else module.momentum)
            module.running_mean.lerp_(mean.detach(), factor)
            module.running_var.lerp_(variance.detach() * count / (count - 1), factor)
    shape = (1, -1, 1) if value.ndim == 3 else (1, -1)
    result = (clean - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + module.eps)
    if module.affine:
        result = result * module.weight.reshape(shape) + module.bias.reshape(shape)
    return mask_values(result, valid)


def masked_max_pool(value, valid, output_size=1):
    """固定不重叠桶，first-max 梯度；空桶产生零值与 False token mask。"""
    output_size = int(output_size)
    if value.shape[-1] % output_size:
        raise ValueError('masked pooling requires fixed divisible buckets')
    token_valid = valid.reshape(valid.shape[0], output_size, -1).any(-1)
    masked = value.masked_fill(~valid[:, None], -torch.inf)
    pooled = masked.reshape(*value.shape[:-1], output_size, -1).max(-1).values
    return mask_values(pooled, token_valid), token_valid


def masked_sequence(module, value, valid):
    """运行已有 Sequential，保留 state_dict 键和全有效普通 BN 路径。"""
    for layer in module:
        if isinstance(layer, nn.BatchNorm1d):
            value = masked_batch_norm(value, valid, layer)
        elif isinstance(layer, (nn.AdaptiveMaxPool1d, DeterministicMaxPool1d)):
            value, valid = masked_max_pool(value, valid, layer.output_size)
        elif isinstance(layer, nn.Flatten):
            value = layer(value)
            valid = valid.any(-1)
        else:
            value = mask_values(layer(mask_values(value, valid)), valid)
    return value, valid


def masked_mean(value, valid):
    """按元素展开有效性，空项为保留计算图的零。"""
    mask = valid.bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)
