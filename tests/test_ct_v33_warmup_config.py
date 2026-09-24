"""E 配方注册与旧 R/A/B/C/D 配置身份兼容，不运行模型。"""

import pytest

from models.ct_v31.config import config_identity, load_config, normalize_config


# 加入 lr_warmup_steps 字段之前，从现有五份配置实际捕获的摘要。
LEGACY_IDENTITIES = {
    'seqtrack_ref': 'bf78862407cfaa5d5368ca574929a34ea86d86ccfc1ca2c636d0fd1b401f3bbc',
    'b0': 'cb674ad3bc04c27058c96039ddc9cca31429f05f93c176719799d1760e1fdf7d',
    'b0_late_decay': '2a14698f78dd71678a2f400bdaf9294d4270ef189e42f83f68d8897b996de834',
    'b0_half_lr': '5f4a4de5cd9c12d055bae47134bc97f3d8ab6a095f8af9f3e6c61974dcc56a02',
    'b0_scaled_lr': '1dae2405bbaa46b7224446092d789abd70a5714ec125492b950281a4955fb240',
}
E_CONFIG = 'cfgs/ct_seqtrack/33_b0_x3_lr_warmup_mini.yaml'


@pytest.mark.parametrize('name,expected', LEGACY_IDENTITIES.items())
def test_zero_warmup_preserves_captured_legacy_identity(name, expected):
    path = f'cfgs/ct_seqtrack/33_{name}_mini.yaml'
    implicit, explicit = load_config(path), load_config(path, {'lr_warmup_steps': 0})
    assert implicit.lr_warmup_steps == explicit.lr_warmup_steps == 0
    assert config_identity(implicit) == config_identity(explicit) == expected


def test_e_inherits_b_except_learning_rate_warmup_and_labels():
    base = load_config('cfgs/ct_seqtrack/33_b0_late_decay_mini.yaml')
    warmup = load_config(E_CONFIG)
    assert warmup.lr == pytest.approx(3 * base.lr)
    assert warmup.lr_warmup_steps == 2000
    assert warmup.lr_schedule == 'multistep' and warmup.lr_milestones == [20, 50]
    assert warmup.trainer_devices == 1
    changed = {'cfg', 'experiment_name', 'tag', 'lr', 'lr_warmup_steps'}
    assert {key: value for key, value in warmup.items() if key not in changed} == {
        key: value for key, value in base.items() if key not in changed}
    assert config_identity(warmup) not in LEGACY_IDENTITIES.values()


@pytest.mark.parametrize('value', [-1, True, False, .5, 2000., '2000', None])
def test_warmup_steps_must_be_a_nonnegative_integer(value):
    with pytest.raises(ValueError, match='lr_warmup_steps must be a nonnegative integer'):
        normalize_config(dict(ct_engineering_check=True, lr_warmup_steps=value))


@pytest.mark.parametrize('overrides', [
    {'lr_warmup_steps': 0}, {'lr_warmup_steps': 1000}, {'lr_warmup_steps': 2001},
    {'lr': 1e-4}, {'lr': 1.5e-4},
])
def test_formal_warmup_only_allows_the_registered_e_recipe(overrides):
    with pytest.raises(ValueError, match='unregistered formal v33 learning-rate recipe'):
        load_config(E_CONFIG, overrides)


@pytest.mark.parametrize('engineering', [False, True])
def test_positive_warmup_requires_multistep_including_engineering(engineering):
    with pytest.raises(ValueError, match='positive lr_warmup_steps requires lr_schedule=multistep'):
        normalize_config(dict(ct_engineering_check=engineering, lr_warmup_steps=2000,
                              lr_schedule='step', lr_milestones=[]))


@pytest.mark.parametrize('overrides', [{'net_model': 'seqtrack_reference'}, {'v31_arm': 'full'}])
def test_reference_and_full_cannot_use_e_recipe(overrides):
    with pytest.raises(ValueError, match='unregistered formal v33 learning-rate recipe'):
        load_config(E_CONFIG, overrides)


def test_positive_warmup_step_count_changes_configuration_identity():
    # 工程配置允许检查相邻步数的身份变化，不放宽正式注册。
    first = load_config(E_CONFIG, {'ct_engineering_check': True})
    second = load_config(E_CONFIG, {'ct_engineering_check': True, 'lr_warmup_steps': 2001})
    assert config_identity(first) != config_identity(second)
