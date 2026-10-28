"""v39 逐实例有效点执行：保留 v38 参数名、BN 总体与 first-max 梯度。

只压紧计算槽；输入采样、点顺序、层宽和训练目标不在本模块改变。
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils.feature_precision import feature_autocast


@dataclass
class PackedPoints:
    values: Tensor                 # [真实点数,C]，batch/原点槽顺序
    indices: Tensor                # 原 [B*N] 索引
    counts: Tensor                 # [B]
    rows: Tensor                   # 每个真实点的 batch 行
    batch: int
    slots: int

    @classmethod
    def from_dense(cls, value: Tensor, valid: Tensor):
        if value.ndim != 3 or valid.shape != (value.shape[0], value.shape[2]):
            raise ValueError('packed points require [B,C,N] and [B,N]')
        valid = valid.bool()
        batch, channels, slots = value.shape
        indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        values = value.transpose(1, 2).reshape(batch * slots, channels).index_select(0, indices)
        return cls(values, indices, valid.sum(-1), torch.div(indices, slots, rounding_mode='floor'),
                   batch, slots)

    def scatter(self, values: Tensor):
        dense = values.new_zeros(self.batch * self.slots, values.shape[-1])
        return dense.index_copy(0, self.indices, values).reshape(
            self.batch, self.slots, values.shape[-1]).transpose(1, 2)


def packed_batch_norm(value: Tensor, module: nn.BatchNorm1d, *, population_size=None):
    """正式 FP32 统计；double 参数/输入允许 CPU 严格语义参考测试。

    一个原 BN 调用仍只更新一次；全空/单点不更新 running statistics。
    """
    if value.ndim != 2:
        raise ValueError('packed BN expects [M,C]')
    with torch.autocast(device_type=value.device.type, enabled=False):
        reference = module.running_mean if module.running_mean is not None else module.weight
        is_double = value.dtype == torch.float64 and (reference is None or reference.dtype == torch.float64)
        value = value.to(torch.float64 if is_double else torch.float32)
        count = value.shape[0]
        if module.training and count >= 2:
            if population_size is None or count == population_size:
                return module(value)
            # 部分有效的旧 masked-BN 使用显式总体方差；压紧后保持同一公式。
            mean = value.sum(0) / count
            variance = (value - mean).square().sum(0) / count
            result = (value - mean) * torch.rsqrt(variance + module.eps)
            if module.weight is not None:
                result = result * module.weight + module.bias
            if module.track_running_stats:
                with torch.no_grad():
                    module.num_batches_tracked.add_(1)
                    factor = (1. / float(module.num_batches_tracked)
                              if module.momentum is None else module.momentum)
                    module.running_mean.lerp_(mean.detach(), factor)
                    module.running_var.lerp_(variance.detach() * count / (count - 1), factor)
            return result
        return F.batch_norm(value, module.running_mean, module.running_var,
                            module.weight, module.bias, False, 0., module.eps)


def _project(value: Tensor, module: nn.Module, *, bf16: bool):
    if isinstance(module, nn.Conv1d):
        if module.kernel_size != (1,) or module.groups != 1 or module.stride != (1,):
            raise ValueError('packed PointNet only supports ordinary 1x1 convolutions')
        weight = module.weight[:, :, 0]
    elif isinstance(module, nn.Linear):
        weight = module.weight
    else:
        raise TypeError('packed projection requires Linear or Conv1d')
    if bf16:
        with feature_autocast(value, enabled=True):
            return F.linear(value, weight, module.bias)
    # 正式参数为 FP32，回归出口也为 FP32；显式 double 模型仅供高精度参考。
    # 不继承调用者意外开启的全局 autocast。
    with torch.autocast(device_type=value.device.type, enabled=False):
        if weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('packed projections require FP32 master parameters (or FP64 reference)')
        return F.linear(value.to(weight.dtype), weight, module.bias)


def _sequence(module, value, *, bf16, population_size=None):
    for layer in module:
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            value = _project(value, layer, bf16=bf16)
        elif isinstance(layer, nn.BatchNorm1d):
            value = packed_batch_norm(value, layer, population_size=population_size)
        elif isinstance(layer, nn.ReLU):
            value = layer(value)
        else:
            raise TypeError('unsupported packed PointNet layer: ' + type(layer).__name__)
    return value


def grouped_first_max(value: Tensor, lengths: Tensor):
    """连续 segment 的 max；并列只反传首槽，空 segment 为零。

    segment_reduce(max) 自身会平分并列梯度；这里仅用它找无梯度最大值和
    首索引，再 gather 原值，保持原 torch.max(dim) 的梯度定义。
    """
    if value.ndim != 2 or lengths.ndim != 1:
        raise ValueError('grouped max expects [M,C] and [groups]')
    if value.shape[0] == 0:
        return value.new_zeros((len(lengths), value.shape[1])) + value.sum() * 0.
    if not hasattr(torch, 'segment_reduce'):
        raise RuntimeError('v39 packed pooling requires torch.segment_reduce')
    lengths = lengths.to(device=value.device, dtype=torch.long)
    groups = torch.repeat_interleave(torch.arange(len(lengths), device=value.device), lengths,
                                    output_size=value.shape[0])
    detached = value.detach()
    maximum = torch.segment_reduce(detached, 'max', lengths=lengths,
                                   initial=-float('inf'), unsafe=True)
    # float32 索引在 2**24 以下精确；更大工作集用 float64 索引。
    index_dtype = torch.float32 if value.shape[0] < 2**24 else torch.float64
    positions = torch.arange(value.shape[0], device=value.device, dtype=index_dtype)[:, None]
    candidate = torch.where(detached == maximum.index_select(0, groups), positions, float('inf'))
    first = torch.segment_reduce(candidate, 'min', lengths=lengths,
                                 initial=float('inf'), unsafe=True)
    nonempty = lengths[:, None] > 0
    first = torch.where(nonempty, first, 0).long()
    result = value.gather(0, first)
    return torch.where(nonempty, result, 0.)


def _check_count(packed, valid_count):
    if valid_count is not None and (type(valid_count) is not int or valid_count != len(packed.values)):
        raise ValueError('packed valid_count differs from the actual valid point population')


def packed_seg_forward(self, value, valid, *, valid_count=None, feature_bf16=False):
    packed = PackedPoints.from_dense(value, valid)
    _check_count(packed, valid_count)
    features = packed.values
    point_features = None
    for index, layer in enumerate(self.seq_per_point):
        features = _sequence(layer, features, bf16=feature_bf16,
                             population_size=packed.batch * packed.slots)
        if index == 1:
            point_features = features
    pooled = grouped_first_max(features, packed.counts)
    features = torch.cat((point_features, pooled.index_select(0, packed.rows)), -1)
    for layer in self.seq_per_point2:
        features = _sequence(layer, features, bf16=feature_bf16,
                             population_size=packed.batch * packed.slots)
    prediction = _project(features, self.fc, bf16=False)
    return packed.scatter(prediction), packed.scatter(point_features.float() if feature_bf16 else point_features)


def packed_mini_forward(self, value, valid, foreground, *, valid_count=None,
                        row_count=None, feature_bf16=False):
    packed = PackedPoints.from_dense(value, valid)
    _check_count(packed, valid_count)
    features = packed.values
    for layer in self.per_point:
        features = _sequence(layer, features, bf16=feature_bf16,
                             population_size=packed.batch * packed.slots)
    probability = foreground.detach().reshape(-1).index_select(0, packed.indices)
    features = features * probability[:, None]
    pooled = grouped_first_max(features, packed.counts)
    rows = (packed.counts > 0).nonzero(as_tuple=False).flatten()
    if row_count is not None and (type(row_count) is not int or row_count != len(rows)):
        raise ValueError('packed row_count differs from the actual valid row population')
    pooled = pooled.index_select(0, rows)
    for layer in self.hidden:
        pooled = _sequence(layer, pooled, bf16=feature_bf16, population_size=packed.batch)
    result = pooled.new_zeros((packed.batch, pooled.shape[-1])).index_copy(0, rows, pooled)
    return result.float() if feature_bf16 else result


def packed_feature_forward(self, value, valid, *, valid_count=None, feature_bf16=False):
    packed = PackedPoints.from_dense(value, valid)
    _check_count(packed, valid_count)
    features = packed.values
    point_features = None
    for index, layer in enumerate(self.seq_per_point):
        features = _sequence(layer, features, bf16=feature_bf16,
                             population_size=packed.batch * packed.slots)
        if index == 1:
            point_features = features
    pooled = grouped_first_max(features, packed.counts)
    # 原 _pooled_tokens 的 floor 边界；稀疏帧一真实点一 token，绝不复制点。
    capacity = self.token_count
    tokens = packed.counts.clamp_max(capacity)
    column = torch.arange(capacity, device=value.device)[None]
    denominator = tokens.clamp_min(1)[:, None]
    start = torch.div(column * packed.counts[:, None], denominator, rounding_mode='floor')
    stop = torch.div((column + 1) * packed.counts[:, None], denominator, rounding_mode='floor')
    token_valid = column < tokens[:, None]
    lengths = torch.where(token_valid, stop - start, 0).reshape(-1)
    token_values = grouped_first_max(point_features, lengths)
    indices = token_valid.reshape(-1).nonzero(as_tuple=False).flatten()
    rows = torch.div(indices, capacity, rounding_mode='floor')
    features = torch.cat((token_values.index_select(0, indices), pooled.index_select(0, rows)), -1)
    for layer in self.seq_per_point2:
        features = _sequence(layer, features, bf16=feature_bf16, population_size=packed.batch * capacity)
    features = _project(features, self.fc, bf16=feature_bf16)
    if feature_bf16:
        features = features.float()
    dense = features.new_zeros((packed.batch * capacity, features.shape[-1]))
    dense = dense.index_copy(0, indices, features).reshape(packed.batch, capacity, -1).transpose(1, 2)
    return dense, token_valid


def configure_efficient_modules(observation, decoder, config):
    """逐实例安装；不替换 Parameter，不改变原 state_dict 路径。

    config: v39_point_packing / v39_feature_bf16 / v39_attention_backend /
    v39_attention_deterministic。正式 FA2 不会退回 math；CPU 显式选 math_reference。
    """
    from models.ct_v38.observation import SegPointNet, MiniPointNet, FeaturePointNet
    from .attention import configure_attention_modules

    packing = config.get('v39_point_packing', True)
    bf16 = config.get('v39_feature_bf16', True)
    if type(packing) is not bool or type(bf16) is not bool:
        raise ValueError('v39 packing and feature BF16 switches must be bool')
    if 'factorized_projection' in config.get('v38_engineering_options', ()):
        raise ValueError('v39 packed/BF16 execution does not combine factorized_projection')
    backend = config.get('v39_attention_backend', 'fa2_varlen')
    if backend == 'fa2_varlen' and not bf16:
        raise ValueError('v39 fa2_varlen requires v39_feature_bf16=True')
    choices = ((observation.seg_pointnet, SegPointNet, packed_seg_forward),
               (observation.mini_pointnet, MiniPointNet, packed_mini_forward),
               (observation.feature_pointnet, FeaturePointNet, packed_feature_forward))
    for module, expected, forward in choices:
        if not isinstance(module, expected):
            raise TypeError('v39 packing requires the registered v38 PointNet layer layout')
        module.forward = MethodType(forward if packing else expected.forward, module)
    profile = 'optimized_bf16' if bf16 else 'optimized_fp32'
    observation.execution_profile = profile
    decoder.execution_profile = profile
    configure_attention_modules(decoder, backend=backend,
                                deterministic=config.get('v39_attention_deterministic', True))
    return observation, decoder
