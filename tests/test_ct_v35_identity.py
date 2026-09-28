"""v35 独立版本身份与登记预算；旧配置不注入新版默认。"""
from pathlib import Path

import pytest

from models.ct_v31.config import V35_DEFAULTS, config_identity, load_config, normalize_config
from models.ct_v31.identity import model_schema, runtime_key
from models.ct_v31.entry import parse_config


def config35(recipe='quarter', seed=42, **updates):
    suffix = '' if seed == 42 else '_seed52'
    return load_config(f'cfgs/ct_seqtrack/35_b0_w_{recipe}_lr_mini{suffix}.yaml', dict(seed=seed, **updates))


def test_versions_and_new_fields_do_not_leak_into_legacy():
    from tests.test_ct_v34_identity import V33_HASHES
    for name, checksum in V33_HASHES.items():
        old = load_config(f'cfgs/ct_seqtrack/33_{name}_mini.yaml')
        assert not set(V35_DEFAULTS) & set(old)
        assert config_identity(old) == checksum
    old = load_config('cfgs/ct_seqtrack/34_b0_context_w4_mini.yaml')
    assert not set(V35_DEFAULTS) & set(old)
    new = config35()
    assert model_schema(new) == 'ct_seqtrack.joint_identity.v35'
    assert runtime_key(new) == 'ct_v35_runtime'
    assert new.v35_seed_policy == 'initial_exact_other_perturbed_v1'


def test_all_recipes_seed52_reference_and_cli():
    identities = set()
    for recipe, lr in [('quarter', 2.5e-5), ('half', 5e-5), ('normal', 1e-4), ('scaled', 1.5e-4)]:
        for seed in (42, 52):
            cfg = config35(recipe, seed)
            assert cfg.lr == lr and cfg.seed == seed and cfg.ct_partition_seed == 42
            assert (cfg.v31_short_window, cfg.v31_long_window, cfg.epoch) == (4, 8, 60)
            assert cfg.lr_milestones == [20, 50] and cfg.lr_warmup_steps == 0
            identities.add(config_identity(cfg))
            assert parse_config(['--cfg', cfg.cfg]).net_model == 'ctseqtrackv35'
    assert len(identities) == 8
    reference = load_config('cfgs/ct_seqtrack/35_seqtrack_ref_seed52_mini.yaml')
    assert reference.seed == 52 and reference.net_model == 'seqtrack_reference'
    assert model_schema(reference) == 'ct_seqtrack.joint_identity.v33'


@pytest.mark.parametrize('update', [dict(seed=43), dict(seed=True), dict(v31_short_window=3),
    dict(ct_partition_seed=52), dict(v35_train_diagnostics=False), dict(v31_arm='full'),
    dict(lr=1.25e-5), dict(lr_milestones=[20, 40]), dict(lr_warmup_steps=2000),
    dict(version='v1.0-trainval'), dict(v31_evaluate_late3=False), dict(dynamics_time_mode='fixed')])
def test_formal_registration_rejects_changes(update):
    with pytest.raises(ValueError, match='formal v35'):
        config35(**update)


@pytest.mark.parametrize('key,value', [('v35_local_neighbors', 32), ('v35_local_radius', .5),
    ('v35_history_descriptor_dim', 30), ('v35_local_input_dim', 68), ('v35_seed_policy', 'old')])
def test_engineering_cannot_silently_change_architecture_identity(key, value):
    with pytest.raises(ValueError, match='architecture identity'):
        config35(ct_engineering_check=True, **{key: value})


def test_diagnostic_flag_is_engineering_only_and_part_of_identity():
    on = config35(ct_engineering_check=True)
    off = config35(ct_engineering_check=True, v35_train_diagnostics=False)
    assert config_identity(on) != config_identity(off)
    with pytest.raises(ValueError, match='unknown'):
        normalize_config(dict(v35_train_diagnostics=True))


def test_new_yaml_comments_are_utf8_not_replacement_marks():
    for path in Path('cfgs/ct_seqtrack').glob('35_*.yaml'):
        assert '?' not in path.read_text(encoding='utf-8').splitlines()[0]


def test_scaled_recipe_is_v35_only_and_changes_only_learning_rate_and_labels():
    normal, scaled = config35('normal'), config35('scaled')
    ignored = {'cfg', 'experiment_name', 'tag', 'lr'}
    assert {k: v for k, v in normal.items() if k not in ignored} == {
        k: v for k, v in scaled.items() if k not in ignored}
    for name in ('context', 'context_w4'):
        with pytest.raises(ValueError, match='unregistered formal v34'):
            load_config(f'cfgs/ct_seqtrack/34_b0_{name}_mini.yaml', dict(lr=1.5e-4))
