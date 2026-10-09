"""被动张量按原 dtype/device 一次传回；不做归约或改变累计顺序。"""
from collections import defaultdict

import torch


def tensors_to_numpy(values, *, optimized=False):
    """保持键、形状和 dtype；结果在返回前完成 D2H，不跨事务持有张量。"""
    if not optimized:
        return {key: value.detach().cpu().numpy() for key, value in values.items()}
    groups = defaultdict(list)
    for key, value in values.items():
        groups[(value.device, value.dtype)].append((key, value.detach()))
    result = {}
    for items in groups.values():
        packet = torch.cat([value.reshape(-1) for _, value in items]).cpu()
        offset = 0
        for key, value in items:
            count = value.numel()
            result[key] = packet[offset:offset + count].reshape(value.shape).numpy()
            offset += count
    return {key: result[key] for key in values}
