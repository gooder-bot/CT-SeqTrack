"""仅特征计算使用的局部 BF16；不改变主参数、优化器或几何精度。"""
from contextlib import nullcontext

import torch


def feature_autocast(value, *, enabled=False):
    """FP32 路径不增加算子；BF16 必须在支持的 CUDA 设备上显式运行。"""
    if not enabled:
        return nullcontext()
    if value.device.type != 'cuda':
        raise RuntimeError('optimized_bf16 requires a CUDA device with native BF16 support')
    with torch.cuda.device(value.device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('optimized_bf16 requires native CUDA BF16 support')
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def feature_fp32(value, *, enabled=False):
    """只在 BF16 分支恢复模块出口；legacy 不插入额外转换。"""
    return value.float() if enabled else value
