"""v28 等价最大池化：固定、不重叠分桶与首个最大值的梯度归属。"""

from torch import nn


class DeterministicMaxPool1d(nn.Module):
    """替代 global/divisible adaptive max pool，不修改参数或归约定义。

    正式 PointNet 只使用全局池化与 1024→128。非整除形状没有等价的
    固定分桶合同，显式拒绝，不能静默退回非确定性 adaptive backward。
    """

    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = int(output_size)
        if self.output_size <= 0:
            raise ValueError("deterministic max pooling requires positive output_size")

    def forward(self, value):
        if value.ndim != 3 or value.shape[-1] <= 0:
            raise ValueError("deterministic max pooling requires nonempty [B,C,N]")
        count = value.shape[-1]
        if count % self.output_size:
            raise ValueError("deterministic max pooling requires N divisible by output_size")
        if self.output_size == 1:
            return value.max(dim=-1, keepdim=True).values
        return value.reshape(
            *value.shape[:-1], self.output_size,
            count // self.output_size).max(dim=-1).values

    def extra_repr(self):
        return f"output_size={self.output_size}"
