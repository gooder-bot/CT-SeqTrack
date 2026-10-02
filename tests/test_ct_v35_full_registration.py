"""v35 六组完整链正式配置：固定配方、同臂恢复与历史 B0 兼容。"""
from itertools import product

import pytest
import torch

from models.ct_v31.config import config_identity, load_config, normalize_config
from models.ct_v31.data import ReadyQueueBatchSampler
from models.ct_v31.entry import parse_config
from models.ct_v31.identity import model_schema, runtime_key
from models.ct_v31.runtime import resume_payload, validate_resume_payload


ARMS = ('b1', 'b1_b2', 'full')
RECIPES = {'normal': 1e-4, 'scaled': 1.5e-4}
B0_HASHES = {
    'normal': 'dbae6a081d7012830c19aed4dbfe7ed1989f332704ec18ab7bdf9d6ca4eebd8e',
    'scaled': 'fc644d15d2fbb59a81cfea754ceacdf13ca22f3dbbe9b56d96abb4c7991c3ab5',
}


def config_for(arm, recipe='scaled', **updates):
    return load_config(f'cfgs/ct_seqtrack/35_{arm}_w_{recipe}_lr_mini.yaml', updates)


def test_six_formal_configs_differ_from_reusable_b0_only_in_arm_and_labels():
    identities = set()
    allowed = {'cfg', 'experiment_name', 'tag', 'v31_arm'}
    for recipe, lr in RECIPES.items():
        baseline = config_for('b0', recipe)
        assert config_identity(baseline) == B0_HASHES[recipe]
        for arm in ARMS:
            cfg = config_for(arm, recipe)
            assert {k: v for k, v in cfg.items() if k not in allowed} == {
                k: v for k, v in baseline.items() if k not in allowed}
            assert cfg.v31_arm == arm and cfg.lr == lr and cfg.seed == 42
            assert cfg.ct_partition_seed == 42 and cfg.v31_temporal_backend == 'cfc'
            assert (cfg.v31_short_window, cfg.v31_long_window) == (4, 8)
            assert cfg.dynamics_time_mode == 'true' and not cfg.ct_engineering_check
            assert cfg.epoch == 60 and cfg.v35_train_diagnostics
            assert model_schema(cfg) == 'ct_seqtrack.joint_identity.v35'
            assert runtime_key(cfg) == 'ct_v35_runtime'
            parsed = parse_config(['--cfg', cfg.cfg, '--batch_size', '16', '--epoch', '60',
                                   '--workers', '4', '--check_val_every_n_epoch', '5'])
            assert config_identity(parsed) == config_identity(cfg)
            identities.add(config_identity(cfg))
    assert len(identities) == 6


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('updates', [
    dict(seed=52), dict(seed=True), dict(lr=2.5e-5), dict(lr=5e-5), dict(lr=2e-4),
    dict(v31_temporal_backend='gru'), dict(dynamics_time_mode='fixed'),
    dict(v31_short_window=3), dict(v31_long_window=9), dict(ct_partition_seed=52),
    dict(lr_schedule='piecewise', lr=5e-5, lr_stage_values=[5e-5, 1e-5, 5e-6]),
    dict(lr_schedule='step', lr_milestones=[]), dict(lr_milestones=[20, 40]),
    dict(lr_warmup_steps=2000), dict(version='v1.0-trainval'), dict(epoch=61),
    dict(v35_train_diagnostics=False), dict(v31_evaluate_late3=False),
])
def test_new_formal_arms_reject_unregistered_seed_budget_backend_and_recipe(arm, updates):
    with pytest.raises(ValueError, match='formal v35'):
        config_for(arm, **updates)


@pytest.mark.parametrize('arm,recipe', product(ARMS, RECIPES))
def test_registered_identity_resumes_itself_and_rejects_other_arms_lr_seed(arm, recipe):
    cfg = config_for(arm, recipe)
    sampler = ReadyQueueBatchSampler([6], batch_size=16, seed=42, short_window=4,
                                    initial_seed_policy=cfg.v35_seed_policy)
    payload = resume_payload(cfg, completed_epoch=1, complete=True,
                             rows=sampler.row_count, steps=len(sampler),
                             sampler=sampler.state_dict())
    migrated = normalize_config(dict(cfg, path='/new/data/root', tag='resumed',
                                     log_dir='/new/output/path'))
    assert validate_resume_payload(payload, migrated) is payload
    others = [config_for(other, recipe) for other in ('b0', *ARMS) if other != arm]
    others.append(config_for(arm, 'normal' if recipe == 'scaled' else 'scaled'))
    # seed52 本身不在此次正式登记中；工程配置仍不得跨身份读取正式 checkpoint。
    others.append(normalize_config(dict(cfg, ct_engineering_check=True, seed=52)))
    for other in others:
        with pytest.raises(ValueError, match='configuration identity'):
            validate_resume_payload(payload, other)
    with pytest.raises(ValueError, match='complete epoch boundaries'):
        validate_resume_payload(dict(payload, epoch_complete=False), cfg)


@pytest.mark.parametrize('recipe,lr', RECIPES.items())
def test_two_recipes_keep_actual_adam_and_epoch20_epoch50_schedule(recipe, lr):
    from tests.test_ct_v33_recipes import setup_optimizer
    cfg = config_for('full', recipe)
    host, optimizer, scheduler = setup_optimizer(cfg)
    assert type(scheduler) is torch.optim.lr_scheduler.MultiStepLR
    expected = [lr] * 20 + [lr * .1] * 30 + [lr * .01] * 10
    observed = []
    for _ in range(60):
        observed.append(optimizer.param_groups[0]['lr'])
        optimizer.zero_grad(set_to_none=True)
        host.tracker(torch.ones(1, 1)).square().mean().backward()
        optimizer.step()
        scheduler.step()
    assert observed == pytest.approx(expected)
    assert optimizer.param_groups[0]['betas'] == (.5, .999)
    assert optimizer.param_groups[0]['eps'] == 1e-6


def test_previous_version_does_not_gain_new_formal_arms():
    for arm in ARMS:
        with pytest.raises(ValueError, match='formal v34'):
            load_config('cfgs/ct_seqtrack/34_b0_context_w4_normal_lr_mini.yaml',
                        dict(v31_arm=arm))
