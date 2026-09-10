"""v28 的 SeqTrack 观测采样兼容层；不改变新增测量采样。"""
from contextlib import contextmanager
import random

import numpy as np
import torch


class B0EmptyHistoryError(AssertionError):
    """只有原 SeqTrack 历史 GT 全空条件可以触发观测重抽。"""


def regularize_b0_seqtrack_compat(points, sample_size, seed=None):
    """原 B0 槽语义：不足三个测量清零，返回 None 表示无真实槽身份。"""
    from datasets.points_utils import regularize_pc

    if points.shape[0] <= 2:
        return np.zeros((int(sample_size), points.shape[1]), dtype='float32'), None
    return regularize_pc(points, sample_size, seed=seed)


def regularize_b0_sparse_v29(points, sample_size, seed=None):
    """v29 保留1/2个真实测量；空输入与至少3点的原采样路径不变。"""
    from datasets.points_utils import regularize_pc

    return regularize_pc(points, sample_size, seed=seed)


@contextmanager
def isolated_observation_rng(seed):
    """原随机采样算法使用逐样本随机域，不移动调用方或其他模块 RNG。"""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        # 采样只使用 CPU；避免在 DataLoader worker 中初始化 CUDA。
        torch.random.default_generator.manual_seed(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
