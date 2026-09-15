"""v30：真实测量 mask 的确定性 PointNet 归一化与固定分桶池化。"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from utils.deterministic_pooling import DeterministicMaxPool1d


def mask_values(value, valid):
    mask = valid[:, None] if value.ndim == 3 else valid.reshape(-1, 1)
    # 标量零避免分配同尺寸 zeros_like；保留原 where 反向的布局与归约路径。
    return torch.where(mask, value, 0.)


def _training_bn_values(value, valid, weight, bias, count, eps):
    """纯计算：与首次 v30 相同的布局、统计与运算顺序，不更新 BN 状态。"""
    clean = mask_values(value, valid)
    rows = clean.movedim(1, -1).reshape(-1, clean.shape[1]) if clean.ndim == 3 else clean
    weights = valid.reshape(-1, 1).to(clean.dtype)
    mean = rows.sum(0) / count
    variance = ((rows - mean).square() * weights).sum(0) / count
    shape = (1, -1, 1) if value.ndim == 3 else (1, -1)
    result = (clean - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + eps)
    if weight is not None:
        result = result * weight.reshape(shape) + bias.reshape(shape)
    return mask_values(result, valid), mean, variance


class _RecomputedMaskedBatchNorm(torch.autograd.Function):
    """仅保留输入；反向逐层重建原 BN 图，不让所有层同时持有大激活副本。

    使用原运算的 autograd，而非另写近似/改归约顺序的 BN 梯度。
    running statistics 在外层更新一次；这里重算不消耗 RNG、不修改 buffer。
    正式 Adam 只需一阶导数。
    """

    @staticmethod
    def forward(ctx, value, valid, weight, bias, count, eps):
        ctx.save_for_backward(value, valid, weight, bias)
        ctx.count, ctx.eps = count, eps
        result, mean, variance = _training_bn_values(value, valid, weight, bias, count, eps)
        ctx.mark_non_differentiable(mean, variance)
        return result, mean, variance

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_result, _grad_mean, _grad_variance):
        value, valid, weight, bias = ctx.saved_tensors
        source = (value, weight, bias)
        needs = (ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3])
        recreated = tuple(item.detach().requires_grad_(need) if item is not None else None
                          for item, need in zip(source, needs))
        with torch.enable_grad():
            result, _, _ = _training_bn_values(
                recreated[0], valid, recreated[1], recreated[2], ctx.count, ctx.eps)
            gradients = iter(torch.autograd.grad(
                result, tuple(item for item, need in zip(recreated, needs) if need), grad_result))
        dx, dw, db = (next(gradients) if need else None for need in needs)
        return dx, None, dw, db, None, None


def masked_batch_norm(value, valid, module):
    """复用原 BN 参数/状态；无效值连统计与梯度都不进入网络。"""
    valid = valid.bool()
    count = int(valid.sum().item())
    if count == valid.numel() and (not module.training or count >= 2):
        return module(value)
    if not module.training or count < 2:
        clean = mask_values(value, valid)
        result = F.batch_norm(clean, module.running_mean, module.running_var,
                              module.weight, module.bias, False, 0., module.eps)
        return mask_values(result, valid)
    # 用户允许适度增加显存：默认直接计算，省去反向重算；容量受限时显式开启。
    compute = (_RecomputedMaskedBatchNorm.apply
               if getattr(module, 'ct_b0_masked_bn_recompute', False) else _training_bn_values)
    result, mean, variance = compute(
        value, valid, module.weight, module.bias, count, module.eps)
    if module.track_running_stats:
        with torch.no_grad():
            module.num_batches_tracked.add_(1)
            factor = (1. / float(module.num_batches_tracked)
                      if module.momentum is None else module.momentum)
            module.running_mean.lerp_(mean.detach(), factor)
            module.running_var.lerp_(variance.detach() * count / (count - 1), factor)
    return result


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
    # 输入只清理一次。Conv 后紧跟 masked BN，后者负责清理 bias；ReLU 保持零。
    value = mask_values(value, valid)
    layers = list(module)
    for index, layer in enumerate(layers):
        if isinstance(layer, nn.BatchNorm1d):
            value = masked_batch_norm(value, valid, layer)
        elif isinstance(layer, (nn.AdaptiveMaxPool1d, DeterministicMaxPool1d)):
            value, valid = masked_max_pool(value, valid, layer.output_size)
        elif isinstance(layer, nn.Flatten):
            value = layer(value)
            valid = valid.any(-1)
        elif isinstance(layer, nn.ReLU):
            value = layer(value)
        elif (isinstance(layer, (nn.Conv1d, nn.Linear)) and index + 1 < len(layers)
              and isinstance(layers[index + 1], nn.BatchNorm1d)):
            value = layer(value)
        else:
            value = mask_values(layer(value), valid)
    return value, valid


def masked_mean(value, valid):
    """按元素展开有效性，空项为保留计算图的零。"""
    mask = valid.bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)
