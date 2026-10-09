"""批量 D2H 保 dtype、完整报告、CPU 累加顺序与固定 IoU 抽样。"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.ct_v31 import data as data35
from models.ct_v31.model import JointTracker as Model35
from models.ct_v31.training_diagnostics import TrainingDiagnostics as Diagnostics35
from models.ct_v36 import data as data36
from models.ct_v36.model import JointTracker as Model36
from models.ct_v36.training_diagnostics import TrainingDiagnostics as Diagnostics36
from tests.test_ct_v31_runtime import TinySource, request as request35
from tests.test_ct_v35_data_diagnostics import v35_config
from tests.test_ct_v36_data import cfg as config36, request as request36
from utils.diagnostic_transfer import tensors_to_numpy


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA transfer verification requires server'))])
def test_transfer_preserves_shapes_dtype_order_int64_and_values(device):
    values = dict(f32=torch.arange(12., device=device).reshape(3, 4).T.requires_grad_(),
        integer=torch.tensor([2**60 + 123, -1], dtype=torch.int64, device=device),
        flag=torch.tensor([True, False], device=device),
        f64=torch.tensor([1.00000000001], dtype=torch.float64, device=device),
        scalar=torch.tensor(1.25, device=device), empty=torch.empty(0, 2, device=device))
    original = tensors_to_numpy(values)
    optimized = tensors_to_numpy(values, optimized=True)
    assert list(original) == list(optimized)
    for key, value in original.items():
        assert optimized[key].dtype == value.dtype and optimized[key].shape == value.shape
        np.testing.assert_array_equal(optimized[key], value)
    assert int(optimized['integer'][0]) == 2**60 + 123
    assert all(value.grad is None for value in values.values())
    assert tensors_to_numpy({}, optimized=True) == {}


def _baseline_diagnostics(version):
    family = 'ct_v31' if version == 35 else 'ct_v36'
    path = (Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
            / '20261009-142559_execution_optimization' / 'baseline_source' / 'models'
            / family / 'training_diagnostics.py')
    if not path.exists():
        pytest.skip('historical execution audit snapshot is not installed in this checkout')
    spec = importlib.util.spec_from_file_location('models.' + family + '._execution_diagnostics_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainingDiagnostics


@pytest.mark.parametrize('version', [35, 36])
@pytest.mark.parametrize('historical', [False, True])
def test_full_training_report_matches_legacy_without_mutating_gradients_or_rng(version, historical):
    module, config, request, model_type, diagnostic_type = ((data35, v35_config, request35, Model35, Diagnostics35)
        if version == 35 else (data36, config36, request36, Model36, Diagnostics36))
    expected_type = _baseline_diagnostics(version) if historical else diagnostic_type
    expected, actual = expected_type(seed=42), diagnostic_type(seed=42, optimized=True)
    legacy = diagnostic_type(seed=42)
    expected.reset(0); actual.reset(0); legacy.reset(0)
    cfg = config(execution_profile='legacy')
    model = model_type(cfg).train()
    builder = module.BatchBuilder(cfg)
    raw = module.RawEndpointDataset(TinySource((3, 3)))
    chosen = next('sample/' + str(i) for i in range(1000)
        if data35.stable_seed(42, 0, 'sample/' + str(i), 3, 1, 1, 'v35_training_iou') % 16 == 0)
    for frame in (1, 2):
        rows = [raw[replace(request(frame, branch=3, end=3), tracklet=i, epoch=0)] for i in range(2)]
        rows[0]['tracklet_key'] = chosen
        batch = builder.prepare(rows)
        prior = model.plan_prior(batch)
        builder.acquire(batch, prior)
        output = model(batch, prior=prior)
        output.loss_statistics = {}
        losses = model.compute_losses(batch, output)
        parameters = tuple(model.parameters())
        before = torch.autograd.grad(losses['loss_total'], parameters, allow_unused=True, retain_graph=True)
        rng = torch.get_rng_state().clone()
        prediction = output.accepted_box.detach().clone()
        expected.add_batch(rows, batch, output, losses)
        legacy.add_batch(rows, batch, output, losses)
        actual.add_batch(rows, batch, output, losses)
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(prediction, output.accepted_box)
        after = torch.autograd.grad(losses['loss_total'], parameters, allow_unused=True)
        for a, b in zip(before, after):
            assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
        assert json.dumps(expected.summary(), sort_keys=True, allow_nan=False) == json.dumps(
            actual.summary(), sort_keys=True, allow_nan=False)
        assert json.dumps(expected.summary(), sort_keys=True, allow_nan=False) == json.dumps(
            legacy.summary(), sort_keys=True, allow_nan=False)
        builder.commit(output, batch)
    if version == 35:
        assert actual.summary()['sampled_geometry']
