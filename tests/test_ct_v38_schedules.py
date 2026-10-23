"""保留工程调度能力，正式三组仍由配置门禁限定。"""
from copy import deepcopy
import pytest
import torch

from models.ct_v38.config import normalize_config
from models.ctseqtrackv38 import CTSEQTRACKV38


def configured(**changes):
    config = normalize_config(dict(ct_engineering_check=True, **changes))
    return CTSEQTRACKV38(config, tracker=torch.nn.Linear(1,1)).configure_optimizers()


@pytest.mark.parametrize('recipe,expected', [
    ('b1_standard',[1.5e-4]*20+[1.5e-5]*30+[1.5e-6]*10),
    ('b1_hold',[1.5e-4]*20+[1.5e-5]*40),
    ('full_standard',[1.5e-4]*20+[1.5e-5]*30+[1.5e-6]*10)])
def test_registered_sixty_epoch_rates(recipe, expected):
    value = configured(v38_training_recipe=recipe)
    optimizer,scheduler = value['optimizer'],value['lr_scheduler']['scheduler']
    actual=[]
    for epoch in range(60):
        actual.append(optimizer.param_groups[0]['lr'])
        optimizer.step(); scheduler.step()
        if epoch == 19:
            restored = configured(v38_training_recipe=recipe)
            restored['optimizer'].load_state_dict(deepcopy(optimizer.state_dict()))
            restored['lr_scheduler']['scheduler'].load_state_dict(deepcopy(scheduler.state_dict()))
            optimizer,scheduler = restored['optimizer'],restored['lr_scheduler']['scheduler']
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('changes', [
    dict(lr_schedule='step', lr_milestones=[]),
    dict(lr_schedule='multistep', lr_warmup_steps=5),
    dict(lr_schedule='piecewise', lr=5e-5, lr_stage_values=[5e-5,1e-5,5e-6])])
def test_independent_optional_schedulers_remain_usable(changes):
    value = configured(**changes)
    optimizer,scheduler = value['optimizer'],value['lr_scheduler']['scheduler']
    for _ in range(3):
        optimizer.step(); scheduler.step()
        assert 0 < optimizer.param_groups[0]['lr'] < 1.
