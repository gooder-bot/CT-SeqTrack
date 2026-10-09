"""执行优化不污染旧配置/参数身份，且新入口不能静默加载旧来源。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.execution_profile import (effective_execution_profile, optimized_execution,
                                     execution_identity, validate_execution_identity)


CONFIGS = {
    'v35': 'cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml',
    'v36': 'cfgs/ct_seqtrack/36_a_full_mini.yaml',
}


def modules(version):
    package = 'models.ct_v31' if version == 'v35' else 'models.ct_v36'
    return tuple(importlib.import_module(package + '.' + name)
                 for name in ('config', 'runtime', 'entry'))


def engineering_config(version, **updates):
    config, _, _ = modules(version)
    return config.load_config(CONFIGS[version], dict(ct_engineering_check=True,
        batch_size=2, point_sample_size=8, workers=0, v32_reserve_windows=0, **updates))


def payload_for(version, config):
    _, runtime, _ = modules(version)
    if version == 'v35':
        from models.ct_v31.data import ReadyQueueBatchSampler
        sampler = ReadyQueueBatchSampler([3, 2], batch_size=2, short_window=4,
            reserve_windows=0, initial_seed_policy=config.v35_seed_policy)
        extra = {}
    else:
        from models.ct_v36.sampler import ReadyQueueBatchSampler
        sampler = ReadyQueueBatchSampler([3, 2], batch_size=2, reserve_windows=0)
        extra = dict(run_id='execution-test')
    return runtime.resume_payload(config, completed_epoch=1, complete=True,
        rows=sampler.row_count, steps=len(sampler), sampler=sampler.state_dict(), **extra)


@pytest.fixture
def fixed_source(monkeypatch):
    source = dict(sha256='a' * 64, files={'models/example.py': 'b' * 64})
    for version in CONFIGS:
        monkeypatch.setattr(modules(version)[2], 'source_identity', lambda: deepcopy(source))
    return source


@pytest.mark.parametrize('version', CONFIGS)
def test_default_profile_preserves_registered_hash_and_explicit_default(version):
    config, _, entry = modules(version)
    cfg = config.load_config(CONFIGS[version])
    assert 'execution_profile' not in cfg and 'execution_profile' not in config.DEFAULTS
    registration_path = ('cfgs/ct_seqtrack/35_full_seed42_registration.json'
                         if version == 'v35' else 'cfgs/ct_seqtrack/36_registration.json')
    registered = json.loads(Path(registration_path).read_text(encoding='utf-8'))
    spec = next(row for row in registered['runs']
                if (row.get('label') == 'scaled_full' if version == 'v35' else row.get('arm') == 'A'))
    assert config.config_identity(cfg) == spec['config_sha256']
    explicit = entry.parse_config(['--cfg', CONFIGS[version], '--execution_profile', 'optimized_fp32'])
    assert explicit.execution_profile == 'optimized_fp32'
    assert config.config_identity(explicit) == config.config_identity(cfg)
    assert effective_execution_profile(cfg) == 'optimized_fp32' and optimized_execution(cfg)
    legacy = config.normalize_config(dict(cfg, execution_profile='legacy'))
    assert not optimized_execution(legacy)
    assert config.config_identity(legacy) != config.config_identity(cfg)


@pytest.mark.parametrize('version', CONFIGS)
def test_bf16_is_explicit_engineering_only_and_has_its_own_identity(version):
    config, _, entry = modules(version)
    with pytest.raises(ValueError, match='requires ct_engineering_check'):
        entry.parse_config(['--cfg', CONFIGS[version], '--execution_profile', 'optimized_bf16'])
    cfg = engineering_config(version, execution_profile='optimized_bf16')
    assert effective_execution_profile(cfg) == 'optimized_bf16'
    assert config.config_identity(cfg) != config.config_identity(engineering_config(version))
    for invalid in ('fast', None, True):
        with pytest.raises(ValueError, match='execution_profile'):
            engineering_config(version, execution_profile=invalid)


@pytest.mark.parametrize('model,family', [
    ('ctseqtrackv33', 'ct_seqtrack_v33'), ('ctseqtrackv34', 'ct_seqtrack_v34'),
    ('seqtrack_reference', 'ct_seqtrack_v33'),
])
def test_older_versions_keep_legacy_defaults_and_reject_profile(model, family):
    from models.ct_v31.config import normalize_config
    cfg = normalize_config(dict(net_model=model, experiment_family=family, ct_engineering_check=True))
    assert effective_execution_profile(cfg) == 'legacy' and not optimized_execution(cfg)
    with pytest.raises(ValueError, match='unknown/inactive'):
        normalize_config(dict(cfg, execution_profile='legacy'))
    from models.ct_v31.runtime import resume_payload, validate_resume_payload
    payload = resume_payload(cfg, completed_epoch=1, complete=True, rows=0, steps=0)
    assert 'execution_identity' not in payload
    assert validate_resume_payload(payload, cfg, training=False) is payload


@pytest.mark.parametrize('version', CONFIGS)
@pytest.mark.parametrize('training', [True, False])
def test_new_checkpoint_roundtrip_rejects_missing_tampered_and_cross_profile(version, training, fixed_source):
    cfg = engineering_config(version)
    config, runtime, _ = modules(version)
    payload = payload_for(version, cfg)
    assert payload['execution_identity'] == execution_identity(cfg, fixed_source)
    assert runtime.validate_resume_payload(payload, cfg, training=training) is payload
    explicit = config.normalize_config(dict(cfg, execution_profile='optimized_fp32'))
    assert runtime.validate_resume_payload(payload, explicit, training=training) is payload
    old = deepcopy(payload)
    del old['execution_identity']
    with pytest.raises(ValueError, match='original source'):
        runtime.validate_resume_payload(old, cfg, training=training)
    for field, wrong in (('profile', 'legacy'), ('source_sha256', 'c' * 64), ('schema', 'unknown')):
        broken = deepcopy(payload)
        broken['execution_identity'][field] = wrong
        with pytest.raises(ValueError, match='identity'):
            runtime.validate_resume_payload(broken, cfg, training=training)
    with pytest.raises(ValueError, match='identity'):
        runtime.validate_resume_payload(payload,
            config.normalize_config(dict(cfg, execution_profile='legacy')), training=training)


@pytest.mark.parametrize('version', CONFIGS)
def test_standalone_eval_requires_new_execution_identity_without_writing_destination(version, tmp_path, fixed_source):
    cfg = engineering_config(version)
    config, _, entry = modules(version)
    payload = payload_for(version, cfg)
    checkpoint = tmp_path / 'new.ckpt'
    key = 'ct_' + version + '_runtime'
    torch.save({key: payload}, checkpoint)
    eval_cfg = config.normalize_config(dict(cfg, test=True, checkpoint=str(checkpoint)))
    destination = tmp_path / 'evaluation'
    assert entry.validate_run_destination(eval_cfg, destination) is False
    old = deepcopy(payload)
    del old['execution_identity']
    torch.save({key: old}, checkpoint)
    with pytest.raises(ValueError, match='original source'):
        entry.validate_run_destination(eval_cfg, destination)
    assert not destination.exists()


def test_execution_identity_does_not_accept_extra_unvalidated_fields(fixed_source):
    cfg = engineering_config('v35')
    value = execution_identity(cfg, fixed_source)
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_execution_identity(dict(value, allow_source_mismatch=True), cfg, fixed_source)


@pytest.mark.parametrize('version', CONFIGS)
@pytest.mark.parametrize('failure', ['forward', 'loss', 'commit', None])
def test_host_transaction_releases_private_cache_on_commit_or_error(version, failure):
    class Tracker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen = None

        def plan_prior(self, batch):
            self.seen = batch
            return None

        def forward(self, batch, prior=None):
            if failure == 'forward':
                raise RuntimeError('injected forward failure')
            return SimpleNamespace()

        def compute_losses(self, batch, output):
            if failure == 'loss':
                raise RuntimeError('injected loss failure')
            return dict(loss_total=self.weight.square())

    class Builder:
        def __init__(self):
            self.released = 0

        def prepare(self, rows):
            return {'value': torch.ones(1)}

        def acquire(self, batch, prior, *, training):
            pass

        def commit(self, output, batch):
            if failure == 'commit':
                raise RuntimeError('injected commit failure')

        def clear_materialization_cache(self):
            self.released += 1

    cfg = engineering_config(version, **({'v35_train_diagnostics': False} if version == 'v35'
                                        else {'v36_diagnostics': False}))
    host_module = importlib.import_module('models.ctseqtrackv31' if version == 'v35' else 'models.ctseqtrackv36')
    host_type = host_module.CTSEQTRACKV31 if version == 'v35' else host_module.CTSEQTRACKV36
    tracker = Tracker()
    host = host_type(cfg, tracker=tracker)
    host.train_builder = builder = Builder()
    host.log = lambda *args, **kwargs: None
    rows = [dict(request=SimpleNamespace(drain=False)) for _ in range(2)]
    if failure in ('forward', 'loss'):
        with pytest.raises(RuntimeError, match='injected ' + failure):
            host.training_step(rows, 0)
    else:
        host.training_step(rows, 0)
        assert '_ct_materialization_cache' in tracker.seen
        if failure == 'commit':
            with pytest.raises(RuntimeError, match='injected commit'):
                host.on_train_batch_end(None, rows, 0)
        else:
            host.on_train_batch_end(None, rows, 0)
            assert host._pending_train is None and host._epoch_steps == 1
    assert '_ct_materialization_cache' not in tracker.seen
    assert builder.released >= 1
