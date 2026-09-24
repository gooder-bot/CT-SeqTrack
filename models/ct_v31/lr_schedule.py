"""按实际 Adam 更新升温，按已完成 epoch 衰减；两种计数均进入 checkpoint。"""

from bisect import bisect_right

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
