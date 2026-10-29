"""按实际 Adam 更新升温，按已完成 epoch 衰减；两种计数均进入 checkpoint。"""

from bisect import bisect_right
import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupMultiStepLR(LRScheduler):
    """last_epoch 在 step 调度器中表示已完成的更新数。

    初始化即设置第 1 次更新的学习率；每次 Adam 后 step() 准备下一次。
    host 在 step() 前提供 completed_epochs，避免把轮数换算成固定 batch 数。
    """

    def __init__(self, optimizer, *, warmup_steps, milestones, gamma):
        self.warmup_steps = int(warmup_steps)
        self.milestones = tuple(milestones)
        self.gamma = float(gamma)
        self.completed_epochs = 0
        super().__init__(optimizer)

    def get_lr(self):
        warmup = min((self.last_epoch + 1) / self.warmup_steps, 1.)
        decay = self.gamma ** bisect_right(self.milestones, self.completed_epochs)
        return [base_lr * warmup * decay for base_lr in self.base_lrs]


class AbsolutePiecewiseLR(LRScheduler):
    """按已完成 epoch 设置下一轮的绝对学习率；不做浮点连乘。

    初始化 last_epoch=0，对应第1轮；第20轮结束后的 step() 将
    last_epoch 置20，第21轮开始使用第二段。全部状态沿用标准checkpoint。
    """

    def __init__(self, optimizer, *, milestones, stage_values):
        self.milestones = tuple(milestones)
        self.stage_values = tuple(stage_values)
        if (not self.milestones or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                for v in self.milestones) or list(self.milestones) != sorted(set(self.milestones))):
            raise ValueError('piecewise milestones must be strictly increasing positive integers')
        if (len(self.stage_values) != len(self.milestones) + 1
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v <= 0 for v in self.stage_values)):
            raise ValueError('piecewise needs one positive finite absolute LR per stage')
        if any(group['lr'] != self.stage_values[0] for group in optimizer.param_groups):
            raise ValueError('piecewise optimizer LR must equal the first stage')
        super().__init__(optimizer)

    def get_lr(self):
        value = self.stage_values[bisect_right(self.milestones, self.last_epoch)]
        return [value for _ in self.base_lrs]
