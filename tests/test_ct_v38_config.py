"""新三组身份与旧强 B1 适配边界。"""
import json
from pathlib import Path

import pytest

from models.ct_v38.config import (RECIPES, REGISTRATION_SCHEMA,
    config_identity, load_config, normalize_config, recipe_identity, recipe_scope)
from models.ct_v38.entry import parse_config, validate_formal_registration


def test_three_recipes_share_b1_math_except_registered_schedule_and_policy():
    configs = [load_config(f'cfgs/ct_seqtrack/38_{name}_mini.yaml') for name in RECIPES]
    assert len({config_identity(cfg) for cfg in configs}) == 3
    assert [cfg.lr_milestones for cfg in configs] == [[20, 50], [20], [20, 50]]
    assert [cfg.v38_arm for cfg in configs] == ['b1', 'b1', 'full']
    allowed = {'experiment_name','tag','cfg','lr_milestones','v38_arm','v38_training_recipe'}
    common = [{key:value for key,value in cfg.items() if key not in allowed} for cfg in configs]
    assert common[0] == common[1] == common[2]
    assert all(cfg.v38_execution_revision == 'clean1' for cfg in configs)



@pytest.mark.parametrize('change', [dict(seed=52), dict(epoch=61), dict(batch_size=8),
    dict(workers=2), dict(check_val_every_n_epoch=5), dict(lr=1e-4),
    dict(v31_arm='full'), dict(v38_arm='b1_b2'), dict(v38_shadow_epochs=10),
    dict(v38_gain_margin=.025), dict(v38_trust_threshold=.7), dict(v38_min_support=1),
    dict(v38_duplicate_xyz_tolerance=1e-6), dict(init_checkpoint='old.ckpt'),
    dict(execution_profile='optimized_bf16'), dict(lr_milestones=[20]),
    dict(v38_training_recipe='b1_hold', v38_arm='full'),
    dict(v37_training_recipe='b1_concat'), dict(v37_centroid_recipe='full_mass_detach')])
def test_formal_identity_rejects_unregistered_variants(change):
    with pytest.raises(ValueError):
        normalize_config(change)


def test_engineering_adjustment_is_distinct_and_paths_devices_do_not_change_algorithm():
    cfg = normalize_config()
    baseline = config_identity(cfg)
    assert config_identity(dict(cfg, devices=1, path='new_data', log_dir='new_run')) == baseline
    engineered = normalize_config(dict(ct_engineering_check=True, epoch=2, batch_size=2,
        workers=0, v31_curriculum_epochs=1))
    assert config_identity(engineered) != baseline
    assert engineered.v31_arm == 'b1'


def test_recipe_scope_never_inherits_factorized_or_b2_options():
    from utils.execution_lab import execution_scope, enabled
    with execution_scope(('factorized_projection',), engineering=True):
        with recipe_scope(normalize_config()):
            assert enabled('unique_mask')
            assert not enabled('factorized_projection')


def test_cli_rejects_unknown_abbreviated_and_training_override():
    path = 'cfgs/ct_seqtrack/38_full_standard_mini.yaml'
    assert parse_config(['--cfg', path, '--devices', '0']).devices == 0
    with pytest.raises(SystemExit):
        parse_config(['--cfg', path, '--bat', '8'])
    with pytest.raises(SystemExit):
        parse_config(['--cfg', path, '--unknown', 'x'])
    with pytest.raises(ValueError):
        parse_config(['--cfg', path, '--lr', '.0001'])


def test_formal_registration_binds_exact_three_configs_and_source(tmp_path):
    source = {'sha256': 'f' * 64, 'files': {'model.py': 'a' * 64}}
    configs = [load_config(f'cfgs/ct_seqtrack/38_{name}_mini.yaml') for name in RECIPES]
    rows = [dict(label=cfg.v38_training_recipe, arm=cfg.v38_arm,
        execution_profile=cfg.execution_profile, config_sha256=config_identity(cfg),
        training_execution_recipe=recipe_identity(cfg)) for cfg in configs]
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(dict(schema=REGISTRATION_SCHEMA, source=source, runs=rows)), encoding='utf-8')
    for cfg in configs:
        assert validate_formal_registration(cfg, source, registration_path=path)['label'] == cfg.v38_training_recipe
    with pytest.raises(ValueError, match='source differs'):
        validate_formal_registration(configs[0], {'sha256': 'changed'}, registration_path=path)
    drift = dict(configs[0], experiment_name='unregistered')
    with pytest.raises(ValueError, match='not exactly one'):
        validate_formal_registration(normalize_config(drift), source, registration_path=path)
    path.write_text(json.dumps(dict(schema=REGISTRATION_SCHEMA, source=source, runs=rows[:1])), encoding='utf-8')
    with pytest.raises(ValueError, match='exactly three'):
        validate_formal_registration(configs[0], source, registration_path=path)


def test_current_training_recipe_matches_frozen_resolved_strong_b1():
    import hashlib
    fixture=json.loads(Path('tests/fixtures/strong_b1_preclean_resolved.json').read_text(encoding='utf-8'))
    old=fixture['config']
    assert hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()==fixture['resolved_sha256']
    new=load_config('cfgs/ct_seqtrack/38_b1_standard_mini.yaml')
    identity={'net_model','experiment_family','experiment_name','tag','cfg','ct_engineering_check','log_dir'}
    for key,value in new.items():
        if key in old and key not in identity:
            assert value == old[key], key
