"""B2 重心三组的配置隔离、空执行作用域、登记和恢复合同。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.ct_v31.data import ReadyQueueBatchSampler
from models.ct_v37 import entry
from models.ct_v37.config import DEFAULTS, as_v35_config, config_identity, load_config, normalize_config
from models.ct_v37.runtime import resume_payload, validate_resume_payload
from models.ctseqtrackv31 import CTSEQTRACKV31
from utils.b1_training_recipes import RECIPE_KEY as B1_RECIPE_KEY
from utils.b1_training_recipes import recipe_options as b1_recipe_options
from utils.b2_centroid_recipes import (RECIPE_KEY, RECIPES, REGISTRATION_FILE,
    REGISTRATION_SCHEMA, recipe_spec, recipe_identity, detach_mass_weights, recipe_scope)
from utils.execution_lab import active_options, execution_scope


ROOT = Path(__file__).resolve().parents[1]
FILES = {name: '37_' + name + '_mini.yaml' for name in (
    'b1_b2_mass_detach', 'full_mass_detach', 'full_mass_detach_hold_lr')}


def config(name='full_mass_detach'):
    return load_config(ROOT / 'cfgs/ct_seqtrack' / FILES[name])


def test_three_yaml_recipes_bind_exact_arms_schedule_and_old_execution_path():
    assert set(RECIPES) == set(FILES)
    assert REGISTRATION_FILE == '37_b2_centroid_registration.json'
    identities = set()
    for name in FILES:
        cfg = config(name)
        arm = 'b1_b2' if name.startswith('b1_b2') else 'full'
        assert cfg[RECIPE_KEY] == name and cfg.v37_arm == cfg.v31_arm == arm
        assert B1_RECIPE_KEY not in cfg and detach_mass_weights(cfg)
        assert (cfg.epoch, cfg.batch_size, cfg.seed, cfg.workers) == (60, 16, 42, 4)
        assert cfg.precision == 32 and cfg.execution_profile == 'optimized_fp32'
        assert cfg.check_val_every_n_epoch == 5 and cfg.v31_evaluate_late3 is True
        assert cfg.lr_milestones == ([20] if name.endswith('hold_lr') else [20, 50])
        assert cfg.lr == .00015 and cfg.lr_decay_rate == .1 and cfg.lr_warmup_steps == 0
        assert not cfg.ct_engineering_check and cfg.checkpoint is None and cfg.init_checkpoint is None
        identity = recipe_identity(cfg)
        assert identity['options'] == [] and identity['projection'] == 'concat'
        assert identity['centroid_mass_gradient'] == 'detached_identity_reliability'
        assert identity['arm'] == arm and identity['check_val_every_n_epoch'] == 5
        assert identity['lr_milestones'] == cfg.lr_milestones
        assert RECIPE_KEY not in as_v35_config(cfg)
        identities.add(config_identity(cfg))
    assert len(identities) == 3


def test_all_eleven_old_config_hashes_remain_exact_and_no_optional_default_is_added():
    assert RECIPE_KEY not in DEFAULTS and B1_RECIPE_KEY not in DEFAULTS
    count = 0
    for filename in ('37_registration.json', '37_b1_execution_registration.json'):
        registration = json.loads((ROOT / 'cfgs/ct_seqtrack' / filename).read_text(encoding='utf-8'))
        for row in registration['runs']:
            cfg = load_config(ROOT / row['config'])
            assert RECIPE_KEY not in cfg and recipe_spec(cfg) is None
            assert recipe_identity(cfg) is None and not detach_mass_weights(cfg)
            assert config_identity(cfg) == row['config_sha256']
            count += 1
    assert count == 11


def test_only_recipe_identity_changes_from_old_arm_and_only_schedule_changes_between_fulls():
    for name, arm in [('b1_b2_mass_detach', 'b1_b2'), ('full_mass_detach', 'full')]:
        old = load_config(ROOT / f'cfgs/ct_seqtrack/37_{arm}_mini.yaml')
        new = config(name)
        changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        assert changed == {'cfg', 'experiment_name', 'tag', RECIPE_KEY}
    base, hold = config(), config('full_mass_detach_hold_lr')
    changed = {k for k in base.keys() | hold.keys() if base.get(k) != hold.get(k)}
    assert changed == {'cfg', 'experiment_name', 'tag', RECIPE_KEY, 'lr_milestones'}


@pytest.mark.parametrize('name', [None, '', 'full_mass_live', 'b1_mass_detach', 1, True, []])
def test_invalid_explicit_recipe_is_rejected(name):
    with pytest.raises(ValueError, match='recipe'):
        normalize_config(dict(config(), **{RECIPE_KEY: name}))


@pytest.mark.parametrize('name', ['b1_concat', None, 'unknown'])
def test_b1_and_centroid_recipe_keys_are_mutually_exclusive_even_when_value_is_invalid(name):
    value = dict(config(), **{B1_RECIPE_KEY: name})
    for function in (normalize_config, recipe_spec, recipe_identity, detach_mass_weights, recipe_scope):
        with pytest.raises(ValueError, match='mutually exclusive'):
            function(value)


@pytest.mark.parametrize('change', [
    dict(v37_arm='b1', v31_arm='b1'), dict(v37_arm='b1_b2', v31_arm='b1_b2'),
    dict(execution_profile='optimized_bf16'), dict(execution_profile='legacy'),
    dict(seed=52), dict(batch_size=4), dict(epoch=59), dict(workers=0),
    dict(lr=.0001), dict(lr_milestones=[20]), dict(lr_warmup_steps=1),
    dict(check_val_every_n_epoch=1), dict(v31_evaluate_late3=False),
    dict(precision='bf16-mixed'), dict(v31_short_window=3),
    dict(init_checkpoint='old.ckpt'), dict(limit_train_batches=1),
    dict(v37_detach_mass=False), dict(execution_options=['unique_mask']),
])
def test_new_formal_recipe_rejects_unregistered_overrides(change):
    with pytest.raises(ValueError):
        normalize_config(dict(config(), **change))


def test_hold_lr_does_not_relax_old_or_other_centroid_schedules():
    with pytest.raises(ValueError):
        normalize_config(dict(config('full_mass_detach_hold_lr'), lr_milestones=[20, 50]))
    with pytest.raises(ValueError):
        normalize_config(dict(config('b1_b2_mass_detach'), lr_milestones=[20]))
    with pytest.raises(ValueError):
        normalize_config(dict(lr_milestones=[20], ct_engineering_check=True))


@pytest.mark.parametrize('name', list(FILES))
def test_old_scheduler_preserves_first_fifty_epochs_and_exact_resume_boundaries(name):
    cfg = config(name)
    parameter = torch.nn.Parameter(torch.tensor(1.))
    host = SimpleNamespace(config=cfg, parameters=lambda: iter((parameter,)))
    result = CTSEQTRACKV31.configure_optimizers(host)
    optimizer, scheduler = result['optimizer'], result['lr_scheduler']['scheduler']
    assert result['lr_scheduler']['interval'] == 'epoch'
    learning_rates = []
    for epoch in range(1, 61):
        learning_rates.append(optimizer.param_groups[0]['lr'])
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        if epoch in (20, 50):
            saved_optimizer = deepcopy(optimizer.state_dict())
            saved_scheduler = deepcopy(scheduler.state_dict())
            replacement = CTSEQTRACKV31.configure_optimizers(host)
            replacement['optimizer'].load_state_dict(saved_optimizer)
            replacement['lr_scheduler']['scheduler'].load_state_dict(saved_scheduler)
            assert replacement['optimizer'].param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
            assert replacement['lr_scheduler']['scheduler'].state_dict() == scheduler.state_dict()
    assert learning_rates[:20] == [cfg.lr] * 20
    assert learning_rates[20:50] == [cfg.lr * .1] * 30
    assert learning_rates[50:] == [cfg.lr * .1 * (1. if name.endswith('hold_lr') else .1)] * 10


@pytest.mark.parametrize('name', list(FILES))
@pytest.mark.parametrize('mode', ['train', 'test', 'resume'])
def test_entry_enforces_empty_scope_for_whole_lifecycle_and_restores_outer_scope(monkeypatch, name, mode):
    cfg = config(name)
    if mode != 'train':
        cfg.checkpoint = 'same-run.ckpt'
        cfg.test = mode == 'test'
    calls = []
    injected = {'sentinel': object()}
    def run_inside(actual, *, loaders):
        calls.append((active_options(), actual.test, actual.checkpoint))
        assert loaders is injected
        return 'done'
    monkeypatch.setattr(entry, '_run', run_inside)
    with execution_scope(('labels', 'pool', 'factorized_projection'), engineering=True):
        before = active_options()
        assert entry.run(cfg, loaders=injected) == 'done'
        assert active_options() == before
    assert active_options() == ()
    assert calls == [((), cfg.test, cfg.checkpoint)]


def test_scope_restores_after_failure_and_preserves_old_recipe_and_ambient_behavior(monkeypatch):
    def fail(*args, **kwargs):
        assert active_options() == ()
        raise RuntimeError('injected centroid failure')
    monkeypatch.setattr(entry, '_run', fail)
    with execution_scope(('labels',), engineering=True):
        with pytest.raises(RuntimeError, match='injected'):
            entry.run(config())
        assert active_options() == ('labels',)
        with recipe_scope(load_config(ROOT / 'cfgs/ct_seqtrack/37_full_mini.yaml')):
            assert active_options() == ('labels',)
        b1 = load_config(ROOT / 'cfgs/ct_seqtrack/37_b1_dedup_concat_mini.yaml')
        with recipe_scope(b1):
            assert active_options() == b1_recipe_options(b1)
        assert active_options() == ('labels',)


def test_new_registration_binds_schema_source_full_config_label_and_centroid_algorithm(tmp_path):
    cfg = config()
    source = dict(sha256='a' * 64, files={'source.py': 'b' * 64})
    row = dict(label=cfg[RECIPE_KEY], arm=cfg.v37_arm, execution_profile='optimized_fp32',
               config_sha256=config_identity(cfg), centroid_recipe_identity=recipe_identity(cfg))
    registration = dict(schema=REGISTRATION_SCHEMA, source=source, runs=[row])
    path = tmp_path / 'centroid_registration.json'
    def write(value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(registration)
    result = entry.validate_formal_registration(cfg, source, registration_path=path)
    assert result['schema'] == REGISTRATION_SCHEMA and result['label'] == cfg[RECIPE_KEY]
    with pytest.raises(ValueError, match='source differs'):
        entry.validate_formal_registration(cfg, dict(source, sha256='wrong'), registration_path=path)
    with pytest.raises(ValueError, match='not exactly one registered'):
        entry.validate_formal_registration(config('full_mass_detach_hold_lr'), source, registration_path=path)
    for field, value, message in [
        ('schema', 'ct_seqtrack.v37.registration.v1', 'schema mismatch'),
        ('centroid_recipe_identity', dict(recipe_identity(cfg), options=['unique_mask']), 'recipe identity mismatch'),
        ('label', 'full_mass_detach_hold_lr', 'recipe identity mismatch'),
        ('duplicate', None, 'not exactly one registered'),
    ]:
        invalid = deepcopy(registration)
        if field == 'schema':
            invalid[field] = value
        elif field == 'duplicate':
            invalid['runs'].append(deepcopy(row))
        else:
            invalid['runs'][0][field] = value
        write(invalid)
        with pytest.raises(ValueError, match=message):
            entry.validate_formal_registration(cfg, source, registration_path=path)


def test_runtime_keeps_schema_but_rejects_cross_recipe_and_old_run_checkpoints(monkeypatch):
    source = dict(sha256='a' * 64, files={'source.py': 'b' * 64})
    monkeypatch.setattr(entry, 'source_identity', lambda: source)
    configs = {name: config(name) for name in FILES}
    configs['old_full'] = load_config(ROOT / 'cfgs/ct_seqtrack/37_full_mini.yaml')
    configs['old_shadow'] = load_config(ROOT / 'cfgs/ct_seqtrack/37_b1_b2_mini.yaml')
    configs['old_b1_concat'] = load_config(ROOT / 'cfgs/ct_seqtrack/37_b1_dedup_concat_mini.yaml')
    for name, cfg in configs.items():
        sampler = ReadyQueueBatchSampler((3, 2), cfg.batch_size, short_window=4,
            long_window=8, reserve_windows=112, initial_seed_policy=cfg.v35_seed_policy)
        sampler.set_epoch(19)
        payload = resume_payload(cfg, completed_epoch=20, complete=True, rows=sampler.row_count,
            steps=len(sampler), sampler=sampler.state_dict(), run_id='same-run-' + name)
        assert payload['schema'] == 'ct_seqtrack.v37.epoch_boundary.v1'
        for training in (True, False):
            assert validate_resume_payload(payload, cfg, training=training) is payload
            for other, other_cfg in configs.items():
                if other != name:
                    with pytest.raises(ValueError, match='identity'):
                        validate_resume_payload(payload, other_cfg, training=training)


def test_main_parser_uses_yaml_recipe_and_rejects_cli_cadence_or_late3_drift():
    path = str(ROOT / 'cfgs/ct_seqtrack' / FILES['full_mass_detach_hold_lr'])
    parsed = entry.parse_config(['--cfg', path, '--path', 'local-data-root', '--tag', 'new-run'])
    assert parsed.lr_milestones == [20] and parsed[RECIPE_KEY] == 'full_mass_detach_hold_lr'
    assert config_identity(parsed) == config_identity(config('full_mass_detach_hold_lr'))
    for args in (['--check_val_every_n_epoch', '1'], ['--no_late3']):
        with pytest.raises(ValueError):
            entry.parse_config(['--cfg', path, *args])
