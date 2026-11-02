"""当前点 XYZ 恢复测量；局部特征复用 B3，不添加骨干或在线 GT。"""
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import RecoveryOutput
from .base_losses import frame_endpoint_mean


class RecoveryResidualHead(nn.Module):
    """唯一新增定位参数：64→3，零初始化，每轴限制在一个 L/W/H。"""
    def __init__(self):
        super().__init__()
        self.xyz = nn.Linear(64, 3)
        nn.init.zeros_(self.xyz.weight)
        nn.init.zeros_(self.xyz.bias)

    def forward(self, feature, seed_box, size, point_mask, seed_valid):
        b = len(seed_box)
        if (feature.shape != (b, 64) or seed_box.shape != (b, 4)
                or size.shape != (b, 3) or point_mask.ndim != 2
                or point_mask.shape[0] != b or seed_valid.shape != (b,)):
            raise ValueError('v41 recovery feature/seed/size/point-mask shape mismatch')
        if (not bool(torch.isfinite(seed_box).all())
                or not bool(torch.isfinite(size).all()) or not bool((size > 0).all())):
            raise ValueError('v41 recovery requires finite seed and positive LWH')
        seed = seed_box.detach().to(feature)
        valid = seed_valid.detach().bool() & point_mask.detach().bool().any(-1)
        normalized = torch.tanh(self.xyz(feature))
        normalized = torch.where(valid[:, None], normalized, 0.)
        local = normalized * size.detach().to(feature)
        sine, cosine = seed[:, 3].sin(), seed[:, 3].cos()
        x, y, z = local.unbind(-1)
        world = torch.stack((cosine*x-sine*y, sine*x+cosine*y, z), -1)
        box = torch.cat((seed[:, :3] + world, seed[:, 3:]), -1)
        return RecoveryOutput(box, valid, seed, normalized, point_mask.detach().bool())


def recovery_loss_terms(batch, recovery):
    """训练 GT 只定义监督资格；点掩码来自实际读取的当前唯一 raw 点。"""
    target = batch['target_box'].detach().to(recovery.seed_box)
    size = batch['box_size'].detach().to(target)
    labels = batch['segmentation_labels'][:, -1].detach()
    if labels.shape != recovery.point_mask.shape:
        raise ValueError('v41 recovery supervision must align with current point IDs')
    displacement = target[:, :3] - recovery.seed_box[:, :3]
    sine, cosine = recovery.seed_box[:, 3].sin(), recovery.seed_box[:, 3].cos()
    x, y, z = displacement.unbind(-1)
    normalized_target = torch.stack((cosine*x+sine*y, -sine*x+cosine*y, z), -1) / size
    target_count = (recovery.point_mask & (labels > .5)).sum(-1)
    valid = (recovery.valid & (target_count >= 3)
             & torch.isfinite(normalized_target).all(-1)
             & (normalized_target.abs() <= 1.).all(-1))
    safe_target = torch.where(valid[:, None], normalized_target, 0.)
    terms = F.smooth_l1_loss(recovery.normalized_delta, safe_target, reduction='none', beta=1.).mean(-1)
    return dict(loss=frame_endpoint_mean(terms, valid), terms=terms, valid=valid,
                target=normalized_target, target_count=target_count)
