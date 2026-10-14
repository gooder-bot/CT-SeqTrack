"""工程课程、禁用基线与数值报告的针对性 CPU 回归。"""
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v31.data import stable_seed
from models.ct_v37.config import normalize_config
from tests.test_ct_v31_runtime import TinySource
from tools import execution_lab_runtime as runtime
from tools import check_execution_performance as checks
from utils.execution_lab import active_options, execution_scope


def config():
    return normalize_config(dict(ct_engineering_check=True, workers=0,
        point_sample_size=8, batch_size=4, v32_reserve_windows=0))


def lightning_202_sampler_epoch(loader, epoch):
    # PL 2.0.2 fabric.utilities.data._set_sampler_epoch 的对象选择顺序。
    # 自定义 batch_sampler 本身不在调用集合；不在本地安装/升级 Lightning。
    objects = {}
    for sampler in (getattr(loader, 'sampler', None),
                    getattr(getattr(loader, 'batch_sampler', None), 'sampler', None)):
        if sampler is not None:
            objects[id(sampler)] = sampler
    for sampler in objects.values():
        setter = getattr(sampler, 'set_epoch', None)
        if callable(setter):
            setter(epoch)


@pytest.mark.parametrize('epoch', [0, 10])
def test_course_loader_seed_requests_and_diagnostics_agree_after_lightning_reset(epoch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = config()
        loader = runtime.train_loader(cfg, epoch, {'train': TinySource((17, 13))})
        model, _ = runtime.make_model(cfg, torch.device('cpu'), epoch)
        model.set_loaders({'train': loader})
        assert model._loader('train') is loader
        assert loader.generator.initial_seed() == stable_seed(cfg.seed, 'train', epoch)
        # Trainer 自身只跑 epoch0，必须保留工程 host 的 course epoch。
        lightning_202_sampler_epoch(loader, 0)
        model.on_train_epoch_start()
        requests = [request for rows in loader.batch_sampler for request in rows]
        assert model.current_epoch == epoch
        assert model.training_diagnostics.epoch == epoch
        assert loader.batch_sampler.epoch == epoch
        assert {request.epoch for request in requests} == {epoch}
        maximum_depth = max(request.frame - request.window_start for request in requests)
        assert maximum_depth == (0 if epoch == 0 else 7)
        assert {request.branch for request in requests} == {0, 1, 2, 3}
    finally:
        torch.set_num_threads(previous)


def test_disabled_replay_clears_ambient_options_and_restores_caller_scope(monkeypatch):
    row = dict(request=SimpleNamespace(branch=0, frame=1, window_start=1, drain=False))
    model = SimpleNamespace(train_builder=SimpleNamespace(reset=lambda: None),
                            clear_pending_materialization=lambda: None)
    monkeypatch.setattr(checks, 'module_set', lambda config: {})
    monkeypatch.setattr(runtime, 'train_loader', lambda *args: [[row]])
    monkeypatch.setattr(runtime, 'make_model', lambda *args: (model, object()))
    monkeypatch.setattr(checks, '_step', lambda *args: dict(state={}, options=active_options()))
    with execution_scope(('labels',), engineering=True):
        result = runtime.replay_disabled(SimpleNamespace(batch_size=1), 'cpu', steps=1, epochs=(10,))
        assert active_options() == ('labels',)
    assert result['phases'][0]['snapshot_sha256'] == [runtime.snapshot_digest(dict(state={}, options=()))]
    assert active_options() == ()


def test_snapshot_digest_cross_process_and_noncontiguous_signed_zero():
    value = dict(array=np.array([0., -0.], dtype=np.float32),
                 tensor=torch.arange(12, dtype=torch.float32).reshape(3, 4).T)
    program = (
        'import json, numpy as np, torch; '
        'from tools.execution_lab_runtime import snapshot_digest; '
        'value=dict(array=np.array([0.,-0.],dtype=np.float32),'
        'tensor=torch.arange(12,dtype=torch.float32).reshape(3,4).T); '
        'print(json.dumps(snapshot_digest(value)))'
    )
    actual = json.loads(subprocess.check_output([sys.executable, '-c', program], text=True))
    assert actual == runtime.snapshot_digest(value)
    assert actual == runtime.snapshot_digest(dict(array=value['array'].copy(), tensor=value['tensor'].contiguous()))
    assert runtime.snapshot_digest(torch.tensor(0.)) != runtime.snapshot_digest(torch.tensor(-0.))


def test_numpy_float_state_delta_has_real_tolerance_and_magnitude():
    left = dict(state={1: dict(box=np.array([1., 2.], dtype=np.float64))})
    right = dict(state={1: dict(box=np.array([1. + 1e-6, 2.], dtype=np.float64))})
    report = runtime.numerical_report(left, right)
    assert report['changed_fields'] == 1
    assert report['tolerance_failures'] == 0
    assert report['max_abs'] == pytest.approx(1e-6)
    assert report['representative_changes'] == ['transaction/state/1/box']


def test_numeric_report_keeps_tensor_shape_and_finite_mask_as_contract_failures():
    with pytest.raises(AssertionError, match='interface changed'):
        runtime.numerical_report(torch.zeros(2), torch.zeros(3))
    with pytest.raises(FloatingPointError, match='finite mask'):
        runtime.numerical_report(torch.zeros(1), torch.tensor([float('nan')]))


@pytest.mark.parametrize('field', ['output', 'losses', 'gradients', 'parameters'])
def test_numeric_report_fixed_dictionary_interfaces_are_hard_failures(field):
    with pytest.raises(AssertionError, match='interface|structure|keys'):
        runtime.numerical_report({field: {'original': torch.zeros(1)}}, {field: {}})
    with pytest.raises(AssertionError, match='interface|structure|keys'):
        runtime.numerical_report({field: {}}, {field: []})


def test_numeric_report_top_level_interface_is_fixed_but_recursive_state_can_branch():
    with pytest.raises(AssertionError, match='interface|structure|keys'):
        runtime.numerical_report({'state': {}}, {'state': {}, 'new': 1})
    report = runtime.numerical_report({'state': {1: {}}}, {'state': {2: {}}})
    assert report['changed_fields'] == 1


def pair_fixture(monkeypatch, *, prepared_changes=False, grad_changes=False, adam_changes=False):
    row = dict(request=SimpleNamespace(branch=0, frame=1, window_start=1, drain=False))
    api = {'runtime': SimpleNamespace(capture_rng_state=lambda: None, restore_rng_state=lambda state: None)}
    monkeypatch.setattr(checks, 'module_set', lambda config: api)
    monkeypatch.setattr(runtime, 'train_loader', lambda *args: [[row]])

    def model(*args):
        value = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            value.weight.zero_()
        value.train_builder = SimpleNamespace(reset=lambda: None)
        value.clear_pending_materialization = lambda: None
        return value, object()
    monkeypatch.setattr(runtime, 'make_model', model)
    index = iter((0, 1))

    def step(*args):
        candidate = next(index) == 1
        return dict(prepared={'points': torch.tensor([float(candidate and prepared_changes)])},
            output={'score': torch.tensor([.5])}, losses={'loss_total': torch.tensor(.5)},
            gradients={'weight': None if candidate and grad_changes else torch.ones(1, 1)},
            parameters={'weight': torch.zeros(1, 1)},
            optimizer={'state': {} if candidate and adam_changes else {0: {'step': torch.tensor(1.)}},
                       'param_groups': [{'params': [0]}]},
            state={}, rng={'same': 1}, host_steps=1, candidates={}, diagnostics={})
    monkeypatch.setattr(checks, '_step', step)


@pytest.mark.parametrize('change', ['grad_changes', 'adam_changes'])
def test_numeric_pair_connectivity_and_adam_coverage_are_hard_contracts_on_same_input(monkeypatch, change):
    pair_fixture(monkeypatch, **{change: True})
    with pytest.raises(AssertionError):
        runtime.run_pair(SimpleNamespace(batch_size=1), 'cpu', steps=1, epochs=(10,), numerical=True)


def test_numeric_pair_reports_connectivity_after_recursive_input_divergence(monkeypatch):
    pair_fixture(monkeypatch, prepared_changes=True, grad_changes=True, adam_changes=True)
    result = runtime.run_pair(SimpleNamespace(batch_size=1), 'cpu', steps=1, epochs=(10,), numerical=True)
    report = result['phases'][0]['numerical_differences'][0]
    assert not report['prepared_equal']
    assert not report['gradient_connectivity_equal']
    assert not report['adam_step_coverage_equal']
    assert result['first_divergence']['epoch_index'] == 10
