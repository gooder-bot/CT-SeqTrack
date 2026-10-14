"""独立工程投影：保留原参数，分解重复全局项；不声称 FP32 训练逐位等价。"""
from torch import nn
import torch
from torch.nn import functional as F

from utils.masked_observation import mask_values, masked_batch_norm


def factorized_global_block(block, local, pooled, valid, *, valid_count=None):
    """原 1088→512 Conv/BN/ReLU；global 每个 batch/frame group 只投影一次。

    权重使用原 Parameter 的可微视图，无新参数、跨 batch 缓存或 detach。
    无效点先将 local/global 输入置零，bias 只加一次；其后原 BN 规则不变。
    FP64 仅用于代数探针，工程执行只登记 FP32，不支持 BF16/FP16。
    """
    if len(block) != 3:
        raise ValueError('factorized projection requires Conv1d/BatchNorm1d/ReLU')
    conv, bn, activation = block
    if (not isinstance(conv, nn.Conv1d) or not isinstance(bn, nn.BatchNorm1d)
            or not isinstance(activation, nn.ReLU) or conv.in_channels != 1088
            or conv.out_channels != 512 or conv.groups != 1 or conv.kernel_size != (1,)
            or conv.stride != (1,) or conv.padding != (0,) or conv.dilation != (1,)):
        raise ValueError('factorized projection requires the original 1088-to-512 pointwise block')
    if (local.ndim != 3 or local.shape[1] != 64
            or pooled.shape != (local.shape[0], 1024, 1)
            or valid.shape != (local.shape[0], local.shape[-1])):
        raise ValueError('factorized projection requires group-local features and group-specific pooled input')
    if local.dtype not in (torch.float32, torch.float64) or pooled.dtype != local.dtype:
        raise ValueError('factorized projection supports FP32 and FP64 probes only')
    with torch.autocast(device_type=local.device.type, enabled=False):
        local_value = F.conv1d(mask_values(local, valid), conv.weight[:, :64], bias=None)
        global_value = F.conv1d(pooled, conv.weight[:, 64:], bias=None)
        value = local_value + torch.where(valid[:, None], global_value, 0.)
        if conv.bias is not None:
            value = value + conv.bias[None, :, None]
        value = masked_batch_norm(value, valid, bn, valid_count=valid_count)
    return activation(value)
