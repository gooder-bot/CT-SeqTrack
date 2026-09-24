"""E 的实际更新学习率、不同 epoch 长度与 checkpoint 连续性。"""
from copy import deepcopy

import pytest
import torch

from models.ct_v31.config import load_config
from models.ctseqtrackv31 import CTSEQTRACKV31


def make_optimizer():
    config = load_config('cfgs/ct_seqtrack/33_b0_x3_lr_warmup_mini.yaml')
    model = CTSEQTRACKV31(config, tracker=torch.nn.Linear(1, 1))
    configured = model.configure_optimizers()
    assert configured['lr_scheduler']['interval'] == 'step'
    return configured['optimizer'], configured['lr_scheduler']['scheduler']


def test_e_all_71700_update_rates_and_resume_within_warmup_and_at_decay():
    optimizer, scheduler = make_optimizer()
    observed = []
    for completed in range(71700):
        epoch = completed // 1195
        expected = (3e-4 * min((completed + 1) / 2000, 1.)
                    * .1 ** ((epoch >= 20) + (epoch >= 50)))
        actual = optimizer.param_groups[0]['lr']
        assert actual == pytest.approx(expected)
        if completed + 1 in (1, 1999, 2000, 2001, 23900, 23901, 59750, 59751, 71700):
            observed.append((completed + 1, actual))
        optimizer.step()
        scheduler.completed_epochs = (completed + 1) // 1195
        scheduler.step()
        if completed + 1 in (1195, 23900, 59750):
            restored_optimizer, restored_scheduler = make_optimizer()
            restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
            restored_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
            assert restored_scheduler.state_dict() == scheduler.state_dict()
            assert restored_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
            optimizer, scheduler = restored_optimizer, restored_scheduler
    assert [step for step, _ in observed] == [1, 1999, 2000, 2001, 23900, 23901, 59750, 59751, 71700]
    assert scheduler.last_epoch == 71700 and scheduler.completed_epochs == 60


def test_epoch_decay_does_not_assume_fixed_batch_count():
    optimizer, scheduler = make_optimizer()
    completed = 0
    for epoch in range(60):
        count = 3 + epoch % 4
        for batch in range(count):
            expected = (3e-4 * min((completed + 1) / 2000, 1.)
                        * .1 ** ((epoch >= 20) + (epoch >= 50)))
            assert optimizer.param_groups[0]['lr'] == pytest.approx(expected)
            optimizer.step()
            completed += 1
            scheduler.completed_epochs = epoch + int(batch + 1 == count)
            scheduler.step()
    assert scheduler.last_epoch == completed
