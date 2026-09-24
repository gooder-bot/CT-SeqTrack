"""三份正式配方的实际 Adam/LR 序列和恢复身份。"""
from copy import deepcopy

import pytest
import torch

from models.ct_v31.config import config_identity, load_config, normalize_config
from models.ctseqtrackv31 import CTSEQTRACKV31


RECIPES = [
    ('b0', [1e-4] * 20 + [1e-5] * 20 + [1e-6] * 20),
    ('b0_late_decay', [1e-4] * 20 + [1e-5] * 30 + [1e-6] * 10),
    ('b0_half_lr', [5e-5] * 20 + [5e-6] * 30 + [5e-7] * 10),
]


def setup_optimizer(config):
    model = CTSEQTRACKV31(config, tracker=torch.nn.Linear(1, 1))
    configured = model.configure_optimizers()
    return model, configured['optimizer'], configured['lr_scheduler']['scheduler']


@pytest.mark.parametrize('name,expected', RECIPES)
def test_formal_lr_sequence_and_midrun_restore(name, expected):
    config = load_config(f'cfgs/ct_seqtrack/33_{name}_mini.yaml')
    model, optimizer, scheduler = setup_optimizer(config)
    observed = []
    for epoch in range(60):
        observed.append(optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        model.tracker(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
        if epoch == 39:
            checkpoint = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    assert observed == pytest.approx(expected)
    resumed, resumed_optimizer, resumed_scheduler = setup_optimizer(config)
    resumed.load_state_dict(checkpoint[0])
    resumed_optimizer.load_state_dict(checkpoint[1])
    resumed_scheduler.load_state_dict(checkpoint[2])
    for epoch in range(40, 60):
        assert resumed_optimizer.param_groups[0]['lr'] == pytest.approx(expected[epoch])
        resumed_optimizer.zero_grad()
        resumed.tracker(torch.ones(1, 1)).sum().backward()
        resumed_optimizer.step()
        resumed_scheduler.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    assert optimizer.param_groups[0]['betas'] == (.5, .999)
    assert optimizer.param_groups[0]['eps'] == 1e-6


def test_only_registered_recipe_fields_differ_and_all_resume_identities_are_distinct():
    configs = [load_config(f'cfgs/ct_seqtrack/33_{name}_mini.yaml') for name, _ in RECIPES]
    recipe_fields = {'cfg', 'experiment_name', 'tag', 'lr', 'lr_schedule', 'lr_milestones'}
    common = [{key: value for key, value in cfg.items() if key not in recipe_fields} for cfg in configs]
    assert common[0] == common[1] == common[2]
    assert len({config_identity(cfg) for cfg in configs}) == 3
    # 同种子模型初始化不读取配方，三组模型权重必须相同。
    from models.ct_v31.model import JointTracker
    states = []
    for config in configs:
        torch.manual_seed(config.seed)
        states.append(JointTracker(config).state_dict())
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
        assert torch.equal(states[0][key], states[2][key])


@pytest.mark.parametrize('change', [dict(lr=2e-4), dict(lr_schedule='multistep', lr_milestones=[25, 50]),
                                  dict(lr=5e-5), dict(epoch=61)])
def test_formal_run_does_not_use_engineering_to_bypass_recipe(change):
    with pytest.raises(ValueError, match='formal v33'):
        normalize_config(change)
