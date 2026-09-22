"""独立的原数学 CUDA 确定性适配；不复用 production B0。"""
import torch
from torch import nn
from torch.nn import functional as F


class _OverlappingAdaptiveMax(torch.autograd.Function):
    """工程非整除输入保留原 adaptive 区间和首个最大值，按输出桶顺序累加。"""
    @staticmethod
    def forward(ctx, value, count):
        length = value.shape[-1]
        outputs, indices = [], []
        for index in range(count):
            start = index * length // count
            stop = ((index + 1) * length + count - 1) // count
            maximum, selected = value[..., start:stop].max(-1)
            outputs.append(maximum)
            indices.append(selected + start)
        ctx.save_for_backward(torch.stack(indices, -1))
        ctx.input_shape = tuple(value.shape)
        return torch.stack(outputs, -1)

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        result = gradient.new_zeros(ctx.input_shape)
        positions = torch.arange(ctx.input_shape[-1], device=gradient.device)
        # 不用 gather/scatter_add 的重叠原子写；每次 dense add 只有唯一写入者。
        for index in range(indices.shape[-1]):
            result = result + (indices[..., index, None] == positions).to(gradient) * gradient[..., index, None]
        return result, None


class ReferenceAdaptiveMaxPool1d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = int(output_size)
        if self.output_size < 1:
            raise ValueError('positive adaptive output_size required')

    def forward(self, value):
        if value.ndim != 3 or value.shape[-1] < 1:
            raise ValueError('reference pooling needs nonempty [B,C,N]')
        length, count = value.shape[-1], self.output_size
        if length % count == 0:
            return value.reshape(*value.shape[:-1], count, length // count).max(-1).values
        return _OverlappingAdaptiveMax.apply(value, count)


def replace_adaptive_max_pool(module):
    """不新增参数、不消耗 RNG，保持原 state_dict 路径。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveMaxPool1d):
            setattr(module, name, ReferenceAdaptiveMaxPool1d(child.output_size))
        else:
            replace_adaptive_max_pool(child)


def original_weighted_segmentation_loss(logits, labels):
    """原整批 weighted CE 分母，避免 CUDA 空间 NLL mean 的非确定性内核。"""
    probabilities = F.log_softmax(logits, dim=1)
    rows = probabilities.movedim(1, -1).reshape(-1, 2)
    target = labels.reshape(-1)
    weights = logits.new_tensor([.5, 2.])
    losses = F.nll_loss(rows, target, weight=weights, reduction='none')
    valid = target != -100  # 原 cross_entropy 默认 ignore_index，实际 teacher 标签只含0/1。
    return losses.sum() / (weights[target.masked_fill(~valid, 0)] * valid).sum()
