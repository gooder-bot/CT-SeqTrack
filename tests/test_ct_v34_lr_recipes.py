"""八组单seed正式配方：LR轨迹、配置差异及跨LR恢复身份。"""
import pytest
import torch

from models.ct_v31.config import config_identity, load_config
from models.ct_v31.entry import parse_config
from models.ct_v31.data import ReadyQueueBatchSampler
from models.ct_v31.runtime import resume_payload, validate_resume_payload
from models.ctseqtrackv31 import CTSEQTRACKV31


RECIPES = [
    ('33_seqtrack_ref_mini', 1e-4, [20, 40]),
    ('33_seqtrack_ref_half_lr_mini', 5e-5, [20, 40]),
    ('34_b0_context_normal_lr_mini', 1e-4, [20, 50]),
    ('34_b0_context_mini', 5e-5, [20, 50]),
    ('34_b0_context_quarter_lr_mini', 2.5e-5, [20, 50]),
    ('34_b0_context_w4_normal_lr_mini', 1e-4, [20, 50]),
    ('34_b0_context_w4_mini', 5e-5, [20, 50]),
    ('34_b0_context_w4_quarter_lr_mini', 2.5e-5, [20, 50]),
]


def recipe(name):
    return load_config(f'cfgs/ct_seqtrack/{name}.yaml')


@pytest.mark.parametrize('name,peak,milestones', RECIPES)
def test_all_eight_launch_recipes_and_actual_60_epoch_learning_rates(name, peak, milestones):
    cfg = parse_config(['--cfg', f'cfgs/ct_seqtrack/{name}.yaml',
        '--path', '/home/lishengjie/data/nuscenes-mini', '--batch_size', '16',
        '--epoch', '60', '--workers', '4', '--check_val_every_n_epoch', '5',
        '--accelerator', 'gpu', '--trainer_devices', '1'])
    assert cfg.lr == peak and cfg.seed == 42 and cfg.precision == 32
    assert cfg.lr_warmup_steps == 0 and cfg.v31_evaluate_late3
    assert config_identity(cfg) == config_identity(recipe(name))
    host = CTSEQTRACKV31(cfg, tracker=torch.nn.Linear(1, 1))
    optimization = host.configure_optimizers()
    optimizer, scheduler = optimization['optimizer'], optimization['lr_scheduler']['scheduler']
    for epoch in range(60):
        assert optimizer.param_groups[0]['lr'] == pytest.approx(
            peak * .1 ** sum(epoch >= boundary for boundary in milestones))
        optimizer.zero_grad()
        host.tracker(torch.ones(1, 1)).square().sum().backward()
        optimizer.step()
        scheduler.step()


def test_eight_recipes_only_change_registered_factors_and_keep_existing_identities():
    configs = [recipe(name) for name, _, _ in RECIPES]
    labels = {'cfg', 'experiment_name', 'tag', 'lr'}
    def strip(cfg, extra=()):
        return {k: v for k, v in cfg.items() if k not in labels | set(extra)}
    assert strip(configs[0]) == strip(configs[1])
    assert all(strip(cfg, ('v31_short_window',)) == strip(configs[2], ('v31_short_window',))
               for cfg in configs[2:])
    assert [cfg.v31_short_window for cfg in configs[2:]] == [3, 3, 3, 4, 4, 4]
    assert len({config_identity(cfg) for cfg in configs}) == 8
    assert config_identity(configs[3]) == '08e2d2c72eb55ea69b0ccc7da3a7faf374940084743d42ce58061cdf5e72a81e'
    assert config_identity(configs[6]) == '84d802446044e0aa0d4b22676458bf8ac574a39f4500ea9cbe4caae897588f61'


@pytest.mark.parametrize('left,right', [
    ('33_seqtrack_ref_mini', '33_seqtrack_ref_half_lr_mini'),
    ('34_b0_context_mini', '34_b0_context_normal_lr_mini'),
    ('34_b0_context_mini', '34_b0_context_quarter_lr_mini'),
    ('34_b0_context_w4_mini', '34_b0_context_w4_quarter_lr_mini')])
def test_learning_rate_is_part_of_resume_identity(left, right):
    source, target = recipe(left), recipe(right)
    # config身份校验先于sampler，测试不需要伪造参考teacher的排程。
    sampler = ReadyQueueBatchSampler([6], 16)
    payload = resume_payload(source, completed_epoch=1, complete=True,
        rows=sampler.row_count, steps=len(sampler), sampler=sampler.state_dict())
    with pytest.raises(ValueError, match='configuration identity'):
        validate_resume_payload(payload, target)


def test_new_reference_keeps_original_scheduler_and_rejects_other_recipes():
    for override in (dict(lr=2.5e-5), dict(lr_schedule='multistep', lr_milestones=[20, 50])):
        with pytest.raises(ValueError, match='learning-rate recipe'):
            load_config('cfgs/ct_seqtrack/33_seqtrack_ref_half_lr_mini.yaml', override)
