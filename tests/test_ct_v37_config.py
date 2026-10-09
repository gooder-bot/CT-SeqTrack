"""v37 独立身份与五组固定配方，不改写 v35 摘要。"""
from pathlib import Path

import pytest

from models.ct_v37.config import as_v35_config, config_identity, load_config, normalize_config
from models.ct_v31.config import config_identity as old_identity, load_config as old_load

ROOT = Path(__file__).resolve().parents[1]


def test_five_configs_have_distinct_pinned_identities_and_correct_arms():
    identities = []
    for label in ('b0', 'b1', 'b1_b2', 'full', 'full_bf16'):
        cfg = load_config(ROOT / f'cfgs/ct_seqtrack/37_{label}_mini.yaml')
        assert cfg.v37_arm == cfg.v31_arm == ('full' if label == 'full_bf16' else label)
        assert cfg.execution_profile == ('optimized_bf16' if label == 'full_bf16' else 'optimized_fp32')
        assert (cfg.epoch, cfg.batch_size, cfg.lr, cfg.v31_short_window, cfg.v31_long_window) == (60, 16, .00015, 4, 8)
        identities.append(config_identity(cfg))
    assert len(set(identities)) == 5


@pytest.mark.parametrize('change', [dict(lr=.0001), dict(seed=52), dict(epoch=59),
    dict(v31_short_window=3), dict(execution_profile='legacy'),
    dict(v37_arm='b1', execution_profile='optimized_bf16'), dict(precision='bf16-mixed'),
    dict(lr_warmup_steps=1), dict(init_checkpoint='old.ckpt'), dict(v36_arm='A')])
def test_reject_unregistered_formal_recipe(change):
    with pytest.raises(ValueError):
        normalize_config(change)


def test_adapter_isolated_old_api_and_bf16_not_leaked_into_old_formal_contract():
    cfg = normalize_config(dict(execution_profile='optimized_bf16'))
    legacy = as_v35_config(cfg)
    assert legacy.net_model == 'ctseqtrackv35' and legacy.v31_arm == 'b1'
    assert not any(key.startswith('v37_') for key in legacy)
    assert legacy.execution_profile == 'optimized_fp32' and legacy.ct_engineering_check
    assert cfg.net_model == 'ctseqtrackv37' and cfg.execution_profile == 'optimized_bf16'
    assert as_v35_config(cfg, arm='full').v31_arm == 'full'


def test_paths_and_checkpoint_do_not_change_identity_but_precision_does():
    cfg = normalize_config()
    checksum = config_identity(cfg)
    assert config_identity(dict(cfg, path='different', log_dir='output/new', checkpoint='same.ckpt')) == checksum
    assert config_identity(dict(cfg, execution_profile='optimized_bf16')) != checksum


def test_v35_normalization_and_hash_unmodified_by_adapter():
    path = ROOT / 'cfgs/ct_seqtrack/35_b0_w_scaled_lr_mini.yaml'
    before = old_load(path)
    checksum = old_identity(before)
    as_v35_config(normalize_config())
    after = old_load(path)
    assert before == after and checksum == old_identity(after)
