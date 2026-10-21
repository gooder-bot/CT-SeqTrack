"""B2 六项 CLI 覆盖：登记预设、实际运行身份和同 run 恢复保持分离。"""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from models.ct_v31.data import ReadyQueueBatchSampler
from models.ct_v37 import entry
from models.ct_v37.config import config_identity, load_config, normalize_config
from models.ct_v37.identity import model_schema, runtime_key
from models.ct_v37.runtime import resume_payload
from utils.b2_centroid_recipes import (OVERRIDABLE_FIELDS, REGISTRATION_SCHEMA,
    RECIPE_KEY, override_policy, recipe_identity)
from utils.execution_profile import execution_identity


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'cfgs/ct_seqtrack/37_full_mass_detach_mini.yaml'
SOURCE = dict(sha256='a' * 64, files={'source.py': 'b' * 64})


@pytest.fixture
def registered(tmp_path):
    cfg = load_config(CONFIG_PATH)
    row = dict(label=cfg[RECIPE_KEY], arm=cfg.v37_arm,
        execution_profile=cfg.execution_profile, config_sha256=config_identity(cfg),
        centroid_recipe_identity=recipe_identity(cfg),
        override_defaults={key: cfg[key] for key in OVERRIDABLE_FIELDS})
    registration = dict(schema=REGISTRATION_SCHEMA, source=deepcopy(SOURCE),
                        override_policy=override_policy(), runs=[row])
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(registration), encoding='utf-8')
    return cfg, registration, path


def validate(cfg, path):
    return entry.validate_formal_registration(cfg, SOURCE, registration_path=path)


def test_omitted_cli_parameters_and_explicit_defaults_have_the_same_identity(registered):
    baseline, _, path = registered
    omitted = entry.parse_config(['--cfg', str(CONFIG_PATH)])
    args = ['--cfg', str(CONFIG_PATH)]
    for key in OVERRIDABLE_FIELDS:
        args.extend(['--' + key, str(baseline[key])])
    explicit = entry.parse_config(args)
    assert omitted == explicit == baseline
    assert config_identity(omitted) == config_identity(explicit)
    assert validate(omitted, path) == validate(explicit, path)
    assert validate(explicit, path)['run_kind'] == 'registered'
    assert validate(explicit, path)['overrides'] == {}


def test_validation_1_5_3_share_one_registered_template_but_keep_distinct_actual_hashes(registered):
    baseline, _, path = registered
    results = []
    for cadence in (1, 5, 3):
        cfg = entry.parse_config(['--cfg', str(CONFIG_PATH),
                                  '--check_val_every_n_epoch', str(cadence)])
        result = validate(cfg, path)
        assert result['registered_config_sha256'] == config_identity(baseline)
        assert result['actual_config_sha256'] == config_identity(cfg)
        assert result['run_kind'] == ('registered' if cadence == 1 else 'runtime_override')
        results.append(result)
    assert len({result['actual_config_sha256'] for result in results}) == 3


@pytest.mark.parametrize('key,value,kind', [
    ('workers', 0, 'runtime_override'),
    ('check_val_every_n_epoch', 3, 'runtime_override'),
    ('batch_size', 4, 'recipe_variant'),
    ('epoch', 30, 'recipe_variant'),
    ('lr', .0001, 'recipe_variant'),
    ('seed', 52, 'recipe_variant'),
])
def test_six_cli_overrides_are_effective_and_classified(registered, key, value, kind):
    baseline, _, path = registered
    cfg = entry.parse_config(['--cfg', str(CONFIG_PATH), '--' + key, str(value)])
    assert cfg[key] == value
    result = validate(cfg, path)
    assert result['run_kind'] == kind
    assert result['overrides'] == {key: dict(registered=baseline[key], effective=value)}
    assert result['actual_config_sha256'] != result['registered_config_sha256']


def test_training_override_takes_precedence_over_runtime_classification(registered):
    _, _, path = registered
    cfg = entry.parse_config(['--cfg', str(CONFIG_PATH), '--workers', '0',
                             '--check_val_every_n_epoch', '3', '--lr', '.0001'])
    result = validate(cfg, path)
    assert result['run_kind'] == 'recipe_variant'
    assert set(result['overrides']) == {'workers', 'check_val_every_n_epoch', 'lr'}


def test_registered_defaults_do_not_come_from_the_current_yaml(registered, tmp_path):
    baseline, _, path = registered
    values = dict(baseline, workers=0, check_val_every_n_epoch=5)
    values.pop('cfg', None)
    yaml_path = tmp_path / 'changed_defaults.yaml'
    yaml_path.write_text(yaml.safe_dump(values), encoding='utf-8')
    inherited = entry.parse_config(['--cfg', str(yaml_path)])
    result = validate(inherited, path)
    assert result['run_kind'] == 'runtime_override'
    assert result['registered_config_sha256'] == config_identity(baseline)
    assert result['overrides'] == {
        'workers': dict(registered=4, effective=0),
        'check_val_every_n_epoch': dict(registered=1, effective=5)}
    explicit = entry.parse_config(['--cfg', str(yaml_path), '--check_val_every_n_epoch', '3'])
    assert explicit.workers == 0 and explicit.check_val_every_n_epoch == 3


def test_source_and_unlisted_config_drift_still_fail(registered):
    baseline, _, path = registered
    cfg = normalize_config(dict(baseline, workers=0))
    with pytest.raises(ValueError, match='source differs'):
        entry.validate_formal_registration(cfg, dict(SOURCE, sha256='wrong'), registration_path=path)
    # experiment_name 不在模型固定字段中；仍必须由完整回填配置 hash 拦住。
    drifted = normalize_config(dict(cfg, experiment_name='unregistered_experiment'))
    with pytest.raises(ValueError, match='not exactly one registered'):
        validate(drifted, path)
    with pytest.raises(ValueError):
        normalize_config(dict(cfg, v31_short_window=3))
    with pytest.raises(ValueError, match='unknown'):
        normalize_config(dict(cfg, unlisted_override=True))


@pytest.mark.parametrize('tamper', ['policy', 'missing_default', 'extra_default',
                                   'changed_default', 'algorithm', 'duplicate'])
def test_registration_policy_defaults_and_algorithm_are_checked(registered, tamper):
    cfg, registration, path = registered
    invalid = deepcopy(registration)
    row = invalid['runs'][0]
    if tamper == 'policy':
        invalid['override_policy']['runtime_fields'].append('point_sample_size')
    elif tamper == 'missing_default':
        row['override_defaults'].pop('workers')
    elif tamper == 'extra_default':
        row['override_defaults']['point_sample_size'] = 1024
    elif tamper == 'changed_default':
        row['override_defaults']['seed'] = 52
    elif tamper == 'algorithm':
        row['centroid_recipe_identity']['options'] = ['unique_mask']
    else:
        invalid['runs'].append(deepcopy(row))
    path.write_text(json.dumps(invalid), encoding='utf-8')
    with pytest.raises(ValueError):
        validate(cfg, path)


def test_registration_without_override_policy_keeps_exact_legacy_matching(registered):
    cfg, registration, path = registered
    registration.pop('override_policy')
    registration['runs'][0].pop('override_defaults')
    path.write_text(json.dumps(registration), encoding='utf-8')
    assert set(validate(cfg, path)) == {'schema', 'label', 'sha256'}
    with pytest.raises(ValueError, match='not exactly one registered'):
        validate(normalize_config(dict(cfg, check_val_every_n_epoch=5)), path)


@pytest.mark.parametrize('name', ['--unknown_parameter', '--dev'])
def test_unknown_or_abbreviated_command_line_parameter_is_not_silently_ignored(name):
    with pytest.raises(SystemExit) as error:
        entry.parse_config(['--cfg', str(CONFIG_PATH), name, '3'])
    assert error.value.code == 2


def test_yaml_defaults_and_cli_paths_device_values_have_correct_precedence(tmp_path, monkeypatch):
    values = dict(load_config(CONFIG_PATH), path='yaml-data', log_dir='output/yaml-run', devices=2)
    values.pop('cfg', None)
    yaml_path = tmp_path / 'chosen.yaml'
    yaml_path.write_text(yaml.safe_dump(values), encoding='utf-8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    inherited = entry.parse_config(['--cfg', str(yaml_path)])
    assert inherited.cfg == str(yaml_path)
    assert (inherited.path, inherited.log_dir, inherited.devices) == ('yaml-data', 'output/yaml-run', 2)
    explicit = entry.parse_config(['--cfg', str(yaml_path), '--path', 'cli-data',
                                  '--log_dir', 'output/cli-run', '--devices', '1'])
    assert (explicit.path, explicit.log_dir, explicit.devices) == ('cli-data', 'output/cli-run', 1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'  # parse 本身不操作 GPU 环境。
    assert config_identity(inherited) == config_identity(explicit)
    without_device = entry.parse_config(['--cfg', str(CONFIG_PATH)])
    assert 'devices' not in without_device


@pytest.mark.parametrize('existing,requested,initialized,expected', [
    ('7', None, False, '7'), (None, None, True, None),
    ('7', 1, False, '1'), (None, 0, False, '0'), ('1', 1, True, '1'),
])
def test_device_selection_preserves_default_environment_or_applies_one_physical_id(
        monkeypatch, existing, requested, initialized, expected):
    # 先登记一个环境值，确保 helper 后续新增变量也由 monkeypatch 完整还原。
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'test-previous')
    if existing is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', existing)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: initialized)
    cfg = {} if requested is None else {'devices': requested}
    entry.apply_device_selection(cfg)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == expected


def test_yaml_null_device_inherits_environment_even_after_cuda_initialization(tmp_path, monkeypatch):
    values = dict(load_config(CONFIG_PATH), devices=None)
    values.pop('cfg', None)
    yaml_path = tmp_path / 'inherit_device.yaml'
    yaml_path.write_text(yaml.safe_dump(values), encoding='utf-8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    cfg = entry.parse_config(['--cfg', str(yaml_path)])
    assert cfg.devices is None
    entry.apply_device_selection(cfg)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'
    assert config_identity(cfg) == config_identity(load_config(CONFIG_PATH))


def test_device_selection_rejects_changing_an_initialized_cuda_context(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(RuntimeError, match='already initialized'):
        entry.apply_device_selection({'devices': 1})
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'


@pytest.mark.parametrize('value', ['-1', '1,2'])
def test_device_cli_does_not_allow_negative_or_multiple_gpu_ids(value):
    with pytest.raises((ValueError, SystemExit)):
        entry.parse_config(['--cfg', str(CONFIG_PATH), '--devices', value])


@pytest.mark.parametrize('key,value', [('workers', 2), ('check_val_every_n_epoch', 5),
    ('batch_size', 4), ('epoch', 30), ('lr', .0001), ('seed', 52)])
def test_same_actual_run_can_resume_but_parameter_changes_cannot(tmp_path, monkeypatch, key, value):
    monkeypatch.setattr(entry, 'source_identity', lambda: SOURCE)
    cfg = normalize_config(dict(load_config(CONFIG_PATH), workers=0,
                                check_val_every_n_epoch=3, log_dir=str(tmp_path)))
    sampler = ReadyQueueBatchSampler((3, 2), cfg.batch_size, seed=cfg.seed,
        short_window=4, long_window=8, reserve_windows=112,
        initial_seed_policy=cfg.v35_seed_policy)
    payload = resume_payload(cfg, completed_epoch=1, complete=True, rows=sampler.row_count,
        steps=len(sampler), sampler=sampler.state_dict(), run_id='same-actual-run')
    checkpoint = tmp_path / 'epoch=001.ckpt'
    torch.save({runtime_key(cfg): payload}, checkpoint)
    manifest = dict(schema=model_schema(cfg), config_sha256=config_identity(cfg),
        model=cfg.net_model, source=SOURCE, execution_identity=execution_identity(cfg, SOURCE),
        run_id='same-actual-run')
    entry.write_run_metadata(cfg, tmp_path, manifest, preserve_existing=False)
    original = {name: (tmp_path / name).read_bytes()
                for name in ('resolved_config.yaml', 'run_manifest.json')}
    resumed = normalize_config(dict(cfg, checkpoint=str(checkpoint)))
    assert entry.validate_run_destination(resumed, tmp_path) is True
    with pytest.raises(ValueError, match='identity mismatch'):
        entry.validate_run_destination(normalize_config(dict(resumed, **{key: value})), tmp_path)
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())


def test_formal_manifest_and_console_show_effective_parameters_and_variant(registered, tmp_path, monkeypatch, capsys):
    pl = pytest.importorskip('pytorch_lightning')
    from pytorch_lightning import loggers
    import models.ctseqtrackv37 as host_module

    baseline, _, registration_path = registered
    cfg = normalize_config(dict(baseline, lr=.0001, workers=0,
                                log_dir=str(tmp_path / 'actual_run')))
    real_validate = entry.validate_formal_registration
    monkeypatch.setattr(entry, 'validate_formal_registration',
        lambda config, source: real_validate(config, source, registration_path=registration_path))
    monkeypatch.setattr(entry, 'source_identity', lambda: SOURCE)
    monkeypatch.setattr(entry, 'configure_numerics', lambda: None)
    monkeypatch.setattr(entry, 'initialization_identity', lambda tracker: {})
    tracker = SimpleNamespace(enable_b1=True, enable_b2=True, enable_b3=True)
    monkeypatch.setattr(host_module, 'CTSEQTRACKV37', lambda config, loaders: SimpleNamespace(
        tracker=tracker, parameters=lambda: iter(())))
    monkeypatch.setattr(loggers, 'CSVLogger', lambda *args, **kwargs: None)
    monkeypatch.setattr(loggers, 'TensorBoardLogger', lambda *args, **kwargs: None)
    class ReachedTrainer(Exception):
        pass
    def stop_before_training(**kwargs):
        raise ReachedTrainer
    monkeypatch.setattr(pl, 'Trainer', stop_before_training)
    with pytest.raises(ReachedTrainer):
        entry.run(cfg)
    manifest = json.loads((Path(cfg.log_dir) / 'run_manifest.json').read_text(encoding='utf-8'))
    resolved = yaml.safe_load((Path(cfg.log_dir) / 'resolved_config.yaml').read_text(encoding='utf-8'))
    assert manifest['run_kind'] == 'recipe_variant'
    assert manifest['run_parameters'] == {key: cfg[key] for key in OVERRIDABLE_FIELDS}
    assert manifest['config_sha256'] == config_identity(cfg) == manifest['registration']['actual_config_sha256']
    assert manifest['registration']['registered_config_sha256'] == config_identity(baseline)
    assert resolved['lr'] == .0001 and resolved['workers'] == 0
    console = capsys.readouterr().out
    assert '"run_kind": "recipe_variant"' in console and '"run_parameters"' in console
