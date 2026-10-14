"""B1 从零训练的配置、算法作用域、调度与恢复身份合同。"""
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
from models.ctseqtrackv37 import CTSEQTRACKV37
from utils.b1_training_recipes import (RECIPE_KEY, RECIPES, REGISTRATION_SCHEMA,
                                       recipe_identity, recipe_options)
from utils.execution_lab import active_options, execution_scope


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'b1_factorized': '37_b1_dedup_factorized_mini.yaml',
    'b1_concat': '37_b1_dedup_concat_mini.yaml',
    'b1_factorized_hold_lr': '37_b1_dedup_factorized_hold_lr_mini.yaml',
    'b1_factorized_bs128': '37_b1_dedup_factorized_bs128_mini.yaml',
    'b1_factorized_bs8': '37_b1_dedup_factorized_bs8_mini.yaml',
}


def config(name='b1_factorized'):
    return load_config(ROOT / 'cfgs/ct_seqtrack' / FILES[name])


def test_yaml_recipes_lock_exact_arms_budget_algorithms_and_distinct_identities():
    assert set(RECIPES) == set(FILES)
    identities = set()
    for name in FILES:
        cfg = config(name)
        assert cfg[RECIPE_KEY] == name
        assert cfg.v37_arm == cfg.v31_arm == 'b1'
        assert (cfg.epoch, cfg.seed, cfg.precision, cfg.check_val_every_n_epoch) == (60, 42, 32, 1)
        assert not cfg.ct_engineering_check and cfg.checkpoint is None and cfg.init_checkpoint is None
        assert cfg.execution_profile == 'optimized_fp32'
        assert cfg.batch_size == {'b1_factorized_bs128': 128, 'b1_factorized_bs8': 8}.get(name, 16)
        assert cfg.lr_milestones == ([20] if name.endswith('hold_lr') else [20, 50])
        assert cfg.lr == .00015 and cfg.lr_decay_rate == .1 and cfg.lr_warmup_steps == 0
        identity = recipe_identity(cfg)
        assert identity['prior_read_applicable'] is False
        assert 'prior_read' not in recipe_options(cfg)
        effective = ['unique_mask'] + ([] if name == 'b1_concat' else ['factorized_projection'])
        assert identity['effective_b1_options'] == effective
        assert RECIPE_KEY not in as_v35_config(cfg)
        identities.add(config_identity(cfg))
    assert len(identities) == len(FILES)


def test_batch8_changes_only_batch_size_and_run_identity_from_factorized16():
    baseline, small = config('b1_factorized'), config('b1_factorized_bs8')
    changed = {key for key in baseline.keys() | small.keys() if baseline.get(key) != small.get(key)}
    assert changed == {'cfg', 'experiment_name', 'tag', RECIPE_KEY, 'batch_size'}
    assert recipe_options(small) == recipe_options(baseline)
    assert small.batch_size == 8 and baseline.batch_size == 16
    with pytest.raises(ValueError, match='batch_size'):
        normalize_config(dict(small, batch_size=16))


def test_original_five_config_hashes_remain_exactly_the_registered_values():
    registration = json.loads((ROOT / 'cfgs/ct_seqtrack/37_registration.json').read_text(encoding='utf-8'))
    assert RECIPE_KEY not in DEFAULTS
    for row in registration['runs']:
        cfg = load_config(ROOT / row['config'])
        assert RECIPE_KEY not in cfg and recipe_identity(cfg) is None
        assert config_identity(cfg) == row['config_sha256']


@pytest.mark.parametrize('name', [None, '', 'b1_prior_read', 'full_factorized', 1, True, []])
def test_unknown_or_null_explicit_recipe_is_rejected(name):
    with pytest.raises(ValueError, match='recipe'):
        normalize_config(dict(config(), **{RECIPE_KEY: name}))


@pytest.mark.parametrize('change', [
    dict(seed=52), dict(epoch=59), dict(batch_size=128), dict(lr=.0001),
    dict(lr_milestones=[20]), dict(check_val_every_n_epoch=5),
    dict(v37_arm='full', v31_arm='full'), dict(execution_profile='optimized_bf16'),
    dict(precision='bf16-mixed'), dict(v31_short_window=3), dict(lr_warmup_steps=1),
    dict(init_checkpoint='old.ckpt'), dict(limit_train_batches=1),
])
def test_registered_recipe_rejects_unregistered_parameter_changes(change):
    with pytest.raises(ValueError):
        normalize_config(dict(config(), **change))


def test_hold_lr_and_large_batch_are_not_silent_overrides_to_other_recipes():
    with pytest.raises(ValueError):
        normalize_config(dict(config('b1_factorized_hold_lr'), lr_milestones=[20, 50]))
    with pytest.raises(ValueError):
        normalize_config(dict(config('b1_factorized_bs128'), batch_size=16))
    with pytest.raises(ValueError):
        normalize_config(dict(ct_engineering_check=True, lr_milestones=[20]))


@pytest.mark.parametrize('name', list(FILES))
def test_original_host_scheduler_has_exact_epoch20_and_epoch50_boundaries(name):
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
            # 边界保存的调度器/Adam状态应精确恢复下一轮学习率。
            saved_optimizer, saved_scheduler = deepcopy(optimizer.state_dict()), deepcopy(scheduler.state_dict())
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
def test_entry_run_activates_recipe_for_entire_lifecycle_and_restores_ambient_scope(monkeypatch, name, mode):
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
    with execution_scope(('prior_read',), engineering=True):
        assert entry.run(cfg, loaders=injected) == 'done'
        assert active_options() == ('prior_read',)
    assert active_options() == ()
    assert calls == [(recipe_options(cfg), cfg.test, cfg.checkpoint)]


def test_entry_recipe_scope_restores_on_exception_and_does_not_override_old_engineering_scope(monkeypatch):
    def fail(*args, **kwargs):
        assert active_options() == recipe_options(config())
        raise RuntimeError('injected lifecycle failure')
    monkeypatch.setattr(entry, '_run', fail)
    with execution_scope(('prior_read',), engineering=True):
        with pytest.raises(RuntimeError, match='injected'):
            entry.run(config())
        assert active_options() == ('prior_read',)
    monkeypatch.setattr(entry, '_run', lambda *args, **kwargs: active_options())
    with execution_scope(('labels',), engineering=True):
        assert entry.run(load_config(ROOT / 'cfgs/ct_seqtrack/37_b1_mini.yaml')) == ('labels',)


def test_checkpoint_identity_rejects_every_other_recipe_for_resume_and_test(monkeypatch):
    source = dict(sha256='a' * 64, files={'source.py': 'b' * 64})
    monkeypatch.setattr(entry, 'source_identity', lambda: source)
    for name in FILES:
        cfg = config(name)
        sampler = ReadyQueueBatchSampler((3, 2), cfg.batch_size, short_window=4,
            long_window=8, reserve_windows=112, initial_seed_policy=cfg.v35_seed_policy)
        sampler.set_epoch(19)
        payload = resume_payload(cfg, completed_epoch=20, complete=True, rows=sampler.row_count,
            steps=len(sampler), sampler=sampler.state_dict(), run_id='same-run-' + name)
        for training in (True, False):
            assert validate_resume_payload(payload, cfg, training=training) is payload
            for other in set(FILES) - {name}:
                with pytest.raises(ValueError, match='identity'):
                    validate_resume_payload(payload, config(other), training=training)


def test_new_registration_binds_source_config_and_actual_execution_recipe(tmp_path):
    cfg = config()
    source = dict(sha256='a' * 64, files={'source.py': 'b' * 64})
    row = dict(label=cfg[RECIPE_KEY], arm='b1', execution_profile='optimized_fp32',
               config_sha256=config_identity(cfg), training_execution_recipe=recipe_identity(cfg))
    registration = dict(schema=REGISTRATION_SCHEMA, source=source, runs=[row])
    path = tmp_path / 'registered.json'
    path.write_text(json.dumps(registration), encoding='utf-8')
    assert entry.validate_formal_registration(cfg, source, registration_path=path)['label'] == cfg[RECIPE_KEY]
    with pytest.raises(ValueError, match='source differs'):
        entry.validate_formal_registration(cfg, dict(source, sha256='wrong'), registration_path=path)
    with pytest.raises(ValueError, match='not exactly one registered'):
        entry.validate_formal_registration(config('b1_concat'), source, registration_path=path)
    row['training_execution_recipe']['projection'] = 'concat'
    path.write_text(json.dumps(registration), encoding='utf-8')
    with pytest.raises(ValueError, match='recipe identity mismatch'):
        entry.validate_formal_registration(cfg, source, registration_path=path)


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('name', ['b1_factorized', 'b1_concat'])
def test_direct_model_forward_uses_registered_projection_and_restores_scope(monkeypatch, training, name):
    from models.ct_v31 import observation
    from models.ct_v31.decoder import SharedHypothesisDecoder as V35Decoder
    from models.ct_v37.model import JointTracker
    from tests.test_ct_v37_network import batch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actual_projection, calls = observation.factorized_global_block, []
        def record(*args, **kwargs):
            calls.append(active_options())
            return actual_projection(*args, **kwargs)
        monkeypatch.setattr(observation, 'factorized_global_block', record)
        cfg = config(name)
        model = JointTracker(cfg).train(training)
        # B1 仍是原decoder，prior_read不是这组对照的独立干预。
        assert type(model.decoder) is V35Decoder
        with execution_scope(('prior_read',), engineering=True), torch.no_grad():
            result = model(batch())
            assert active_options() == ('prior_read',)
        assert torch.isfinite(result.accepted_box).all()
        assert calls == ([recipe_options(cfg)] * 2 if name == 'b1_factorized' else [])
        assert active_options() == ()
    finally:
        torch.set_num_threads(previous)


def test_direct_old_model_forward_keeps_outer_lab_projection_scope(monkeypatch):
    from models.ct_v31 import observation
    from models.ct_v37.model import JointTracker
    from tests.test_ct_v37_network import batch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actual_projection, calls = observation.factorized_global_block, []
        def record(*args, **kwargs):
            calls.append(active_options())
            return actual_projection(*args, **kwargs)
        monkeypatch.setattr(observation, 'factorized_global_block', record)
        model = JointTracker(load_config(ROOT / 'cfgs/ct_seqtrack/37_b1_mini.yaml')).eval()
        with execution_scope(('factorized_projection',), engineering=True), torch.no_grad():
            model(batch())
            assert active_options() == ('factorized_projection',)
        assert calls == [('factorized_projection',)] * 2
    finally:
        torch.set_num_threads(previous)


def test_each_validation_epoch_preserves_existing_numbered_output_and_collision_guard(tmp_path):
    class ValidationProbe(CTSEQTRACKV37):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(log_dir=str(tmp_path))
            self.probe_epoch = 0
            self.probe_value = 50.

        @property
        def current_epoch(self):
            return self.probe_epoch

        def _evaluation_end(self, role):
            assert role == 'val'
            self.evaluation_results = dict(success=self.probe_value, precision=60., complete_coverage=True)

    model = ValidationProbe()
    for epoch in range(60):
        model.probe_epoch = epoch
        model.on_validation_epoch_end()
    paths = sorted((tmp_path / 'validation').glob('epoch=*.json'))
    assert len(paths) == 60 and paths[0].name == 'epoch=001.json' and paths[-1].name == 'epoch=060.json'
    assert all(json.loads(path.read_text(encoding='utf-8'))['complete_coverage'] for path in paths)
    original = paths[-1].read_bytes()
    model.on_validation_epoch_end()
    assert paths[-1].read_bytes() == original
    model.probe_value = 49.
    with pytest.raises(FileExistsError, match='validation differs'):
        model.on_validation_epoch_end()
