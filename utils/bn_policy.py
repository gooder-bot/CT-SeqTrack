"""尾部仅切换指定网络的 BN 统计路径，参数和其余模块照常训练。"""

from contextlib import contextmanager

from torch.nn.modules.batchnorm import _BatchNorm


@contextmanager
def running_batch_norm(module, enabled=True):
    """异常退出也恢复 training 标志；使用 running 值而非仅 momentum=0。"""
    batch_norms = [layer for layer in module.modules() if isinstance(layer, _BatchNorm)] if enabled else []
    if any(layer.running_mean is None or layer.running_var is None for layer in batch_norms):
        raise ValueError('running BN policy requires tracked running statistics')
    previous = [layer.training for layer in batch_norms]
    try:
        for layer in batch_norms:
            layer.training = False
        yield
    finally:
        for layer, training in zip(batch_norms, previous):
            layer.training = training
