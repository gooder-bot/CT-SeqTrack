"""工程标签去重：两种几何定义、旧质量归约和完整训练图逐位回归。"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v31.losses import compute_losses as legacy_losses, oriented_iou_labels
from models.ct_v37 import engineering_geometry as geometry
from models.ct_v37.losses import _engineering_base_losses, supervision_targets
from models.ct_v37.model import JointTracker
from tests.test_ct_v37_network import batch, config
from tools.check_execution_performance import assert_identical, freeze
from utils.execution_lab import enabled, execution_scope
from utils.tracking_metrics import LocalYawBox, box_metrics


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20261014)
    yield
    torch.set_num_threads(previous)


def test_paired_metrics_preserve_both_height_formulas_and_float_outputs():
    rng = np.random.default_rng(37)
    pairs = []
    for _ in range(256):
        box, target = rng.normal(size=(2, 4)) * np.array([3., 3., 1., 3.])
        first_size, second_size = rng.uniform(.01, 6., size=(2, 3))
        pairs.append((box, target, first_size, second_size))
    pairs.extend([
        ([0., 0., 0., 0.], [0., 0., 0., 0.], [4., 2., 2.], [4., 2., 2.]),
        ([4., 0., 0., 0.], [0., 0., 0., 0.], [4., 2., 2.], [4., 2., 2.]),
        ([0., 0., .6, .3], [0., 0., 0., .3], [4., 2., .8], [4., 2., 3.]),
        ([0., 0., 0., np.pi / 2], [0., 0., 0., 0.], [4., 2., 2.], [4., 2., 2.]),
        ([100., 100., 100., 0.], [0., 0., 0., 0.], [4., 2., 2.], [4., 2., 2.]),
        ([0., 0., 0., 0.], [.0001, 0., 0., 0.], [.0002, .002, .003], [.0004, .002, .003]),
    ])
    different_heights = False
    for box, target, size, target_size in pairs:
        prediction = LocalYawBox(box, np.asarray(size)[[1, 0, 2]])
        truth = LocalYawBox(target, np.asarray(target_size)[[1, 0, 2]])
        exact, compat, distance = geometry.paired_box_metrics(prediction, truth)
        exact_expected = box_metrics(prediction, truth, up_axis=(0, 0, 1), mode='geometry_exact')
        compat_expected = box_metrics(prediction, truth, up_axis=(0, 0, 1), mode='benchmark_compat')
        assert_identical((exact, distance), exact_expected)
        assert_identical((compat, distance), compat_expected)
        different_heights |= exact != compat
    assert different_heights


@pytest.mark.parametrize('bad_size', [[0., 2., 1.], [-1., 2., 1.], [np.nan, 2., 1.]])
def test_paired_metrics_preserve_invalid_geometry_rejection(bad_size):
    box = LocalYawBox(np.zeros(4), bad_size)
    with pytest.raises(ValueError):
        geometry.paired_box_metrics(box, LocalYawBox(np.zeros(4), [4., 2., 1.]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('small_size,explicit_target_size', [(False, True), (True, True), (True, False)])
def test_label_cache_matches_original_dtype_and_legacy_clamp(dtype, small_size, explicit_target_size):
    boxes = torch.tensor([[[0., 0., 0., 0.], [0., 0., .1, 0.],
                           [.3, .2, 0., .2], [0., 0., 0., 1.2]]], dtype=dtype)
    size = torch.tensor([[.0002, .002, .003] if small_size else [4., 2., 1.5]], dtype=torch.float64)
    values = dict(target_box=torch.tensor([[.0001, 0., 0., .13]], dtype=torch.float64), box_size=size)
    if explicit_target_size:
        values['target_box_size'] = size.clone() * torch.tensor([[1.01, .99, 1.4]])
    legacy_boxes = torch.stack((boxes[:, 0], boxes[:, 2], boxes[:, 3], torch.zeros_like(boxes[:, 0])), 1)
    output = SimpleNamespace(candidates=SimpleNamespace(boxes=boxes),
                             decoder=SimpleNamespace(hypothesis_boxes=legacy_boxes))
    expected = supervision_targets(values, output)
    actual, q0_exact = geometry.supervision_geometry(values, output)
    assert_identical(expected, actual)
    expected_q0 = oriented_iou_labels(legacy_boxes[:, :1], values['target_box'],
        values['box_size'].to(values['target_box']).clamp_min(1e-3), values.get('target_box_size'))
    assert_identical(expected_q0, q0_exact)
    assert all(not value.requires_grad for value in actual.values())


def test_common_geometry_only_visits_four_pairs_per_endpoint(monkeypatch):
    model = JointTracker(config()).eval()
    values = batch()
    output = model(values)
    calls = []
    fallback_calls = []
    original = geometry.paired_box_metrics
    original_fallback = geometry.box_metrics
    def counted(*args):
        calls.append(1)
        return original(*args)
    def counted_fallback(*args, **kwargs):
        fallback_calls.append(1)
        return original_fallback(*args, **kwargs)
    monkeypatch.setattr(geometry, 'paired_box_metrics', counted)
    monkeypatch.setattr(geometry, 'box_metrics', counted_fallback)
    geometry.supervision_geometry(values, output)
    assert len(calls) == 4 * len(values['target_box'])
    assert not fallback_calls


@pytest.mark.parametrize('case', ['ordinary', 'fallback', 'background', 'mixed_history'])
def test_base_adapter_preserves_every_legacy_key_quality_denominator_and_statistics(case):
    model = JointTracker(config()).eval()
    values = batch()
    if case == 'fallback':
        values['point_valid'].zero_()
        values['extension_valid'].zero_()
    elif case == 'background':
        values['segmentation_labels'].zero_()
        values['extension_labels'].zero_()
    elif case == 'mixed_history':
        values['history_valid'][0, :2] = False
        values['history_pair_valid'][0].zero_()
        values['point_valid'][0, :2].zero_()
        values['physical_valid'][0] = False
        values['acquisition_valid'][0] = False
    output = model(values)
    output.loss_statistics = {}
    expected = legacy_losses(values, output, enable_b1=True, enable_b2=False, enable_b3=False)
    expected_statistics = freeze(output.loss_statistics)
    _, q0_exact = geometry.supervision_geometry(values, output)
    output.loss_statistics = {}
    actual = _engineering_base_losses(values, output, q0_exact)
    assert_identical(freeze(expected), freeze(actual))
    assert_identical(expected_statistics, freeze(output.loss_statistics))
    if case == 'fallback':
        assert not output.candidates.valid[:, 0].any()
        assert output.decoder.hypothesis_valid[:, 0].all()
        assert actual['loss_quality'] > 0
    elif case == 'ordinary':
        # 有效 mode 虽然基础质量标签为零，仍占旧 quality_target_mean 的分母。
        assert output.decoder.hypothesis_valid[:, 1:].any()


@pytest.mark.parametrize('arm', ['b1_b2', 'full'])
def test_labels_scope_two_adam_steps_preserve_outputs_all_losses_gradients_bn_adam_and_rng(arm):
    original = JointTracker(config(arm)).train()
    candidate = copy.deepcopy(original).train()
    optimizers = [torch.optim.Adam(model.parameters(), lr=1.5e-4) for model in (original, candidate)]
    values = batch()
    assert not enabled('labels')
    for step in range(2):
        if step:
            values['point_valid'][0].zero_()
            values['extension_valid'][0].zero_()
            values['extension_labels'][1, 4:] = 0
        rng = torch.get_rng_state().clone()
        results = []
        for model, optimizer, options in zip((original, candidate), optimizers, ((), ('labels',))):
            torch.set_rng_state(rng)
            optimizer.zero_grad(set_to_none=True)
            with execution_scope(options, engineering=True):
                output = model(values)
                output.loss_statistics = {}
                losses = model.compute_losses(values, output)
                losses['loss_total'].backward()
            gradients = {name: freeze(parameter.grad) for name, parameter in model.named_parameters()}
            optimizer.step()
            results.append(dict(output=freeze(output), losses=freeze(losses), gradients=gradients,
                                rng=torch.get_rng_state().clone(), state=freeze(model.state_dict()),
                                optimizer=freeze(optimizer.state_dict())))
        assert_identical(results[0], results[1])
    assert not enabled('labels')
