"""v28 保留观测图的特征导出、确定性池化及结构候选语义。"""

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from models.ct_v2.action_v27 import bounded_residual_xy
from models.ct_v2.evidence_memory import B2EvidenceAcquirer
from tests.test_ct_v27_evidence import _inputs
from tests.test_ct_v27_evaluation import FakeHost, sequence
from utils.deterministic_pooling import DeterministicMaxPool1d
from utils.v27_evaluation import evaluate_sequence_v27
from utils.v27_eval_reporting import summarize_endpoint_diagnostics


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def pointnet_module(monkeypatch):
    # 真实三个 PointNet；只隔离未被使用的 PointNet++ CUDA 导入。
    shell = types.ModuleType('pointnet2.utils.pointnet2_modules')
    shell.PointnetSAModule = None
    monkeypatch.setitem(sys.modules, shell.__name__, shell)
    path = Path(__file__).resolve().parents[1] / 'models/backbone/pointnet.py'
    spec = importlib.util.spec_from_file_location('ct_v28_pointnet_components', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('output_size', [1, 128])
@pytest.mark.parametrize('kind', ['random', 'ties', 'zero', 'noncontiguous'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_pool_matches_adaptive_values_and_first_max_gradients(output_size, kind, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA verification requires a CUDA runtime')
    generator = torch.Generator().manual_seed(812)
    if kind == 'noncontiguous':
        value = torch.randn(2, 1024, 3, generator=generator).transpose(1, 2)
    elif kind == 'ties':
        value = torch.randint(0, 3, (2, 3, 1024), generator=generator).float()
    elif kind == 'zero':
        value = torch.zeros(2, 3, 1024)
    else:
        value = torch.randn(2, 3, 1024, generator=generator)
    original = value.to(device).requires_grad_()
    replacement = value.to(device).detach().clone().requires_grad_()
    expected = nn.AdaptiveMaxPool1d(output_size)(original)
    actual = DeterministicMaxPool1d(output_size)(replacement)
    assert torch.equal(expected, actual)
    cotangent = torch.randn(expected.shape, generator=generator).to(device)
    expected.backward(cotangent)
    actual.backward(cotangent)
    assert torch.equal(original.grad, replacement.grad)


def test_pool_rejects_unsupported_shape_and_runs_under_strict_determinism():
    with pytest.raises(ValueError, match='divisible'):
        DeterministicMaxPool1d(128)(torch.zeros(2, 3, 1023))
    with pytest.raises(ValueError, match='nonempty'):
        DeterministicMaxPool1d(1)(torch.zeros(2, 3, 0))
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        value = torch.randn(2, 3, 1024, requires_grad=True)
        DeterministicMaxPool1d(128)(value).square().sum().backward()
        assert torch.isfinite(value.grad).all()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _pointnet(module, kind, deterministic):
    if kind == 'mini':
        return module.MiniPointNet(5, [16, 32], [12], output_size=4,
                                   deterministic_pooling=deterministic)
    if kind == 'seg':
        return module.SegPointNet(5, [64, 64, 128], [32, 16], output_size=11,
                                  deterministic_pooling=deterministic)
    return module.FeaturePointNet(5, [64, 64, 128], [32, 16], output_size=128,
                                  deterministic_pooling=deterministic)


@pytest.mark.parametrize('kind', ['mini', 'seg', 'feature'])
def test_three_pointnets_keep_parameters_outputs_gradients_and_bn(pointnet_module, kind):
    torch.manual_seed(283)
    original = _pointnet(pointnet_module, kind, False).train()
    replacement = _pointnet(pointnet_module, kind, True).train()
    replacement.load_state_dict(original.state_dict(), strict=True)
    first = torch.randn(2, 5, 1024, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    expected, actual = original(first), replacement(second)
    assert torch.equal(expected, actual)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert torch.equal(first.grad, second.grad)
    for (name, value), (other_name, other) in zip(original.named_parameters(), replacement.named_parameters()):
        assert name == other_name
        assert (value.grad is None and other.grad is None) or torch.equal(value.grad, other.grad), name
    for name, value in original.state_dict().items():
        assert torch.equal(value, replacement.state_dict()[name]), name


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('intermediate', [False, True])
def test_seg_export_is_same_computation_with_aligned_second_layer(pointnet_module, training, intermediate):
    torch.manual_seed(928)
    original = _pointnet(pointnet_module, 'seg', True).train(training)
    original.return_intermediate = intermediate
    exported = copy.deepcopy(original)
    first = torch.randn(2, 5, 4 * 1024, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    observed = []
    hook = exported.seq_per_point[1].register_forward_hook(
        lambda module, args, output: observed.append(output))
    rng = torch.get_rng_state().clone()
    expected = original(first)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual, points = exported(second, return_point_features=True)
    hook.remove()
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert points is observed[0]
    assert points.shape == (2, 64, 4096) and points.requires_grad
    aligned = points.transpose(1, 2).reshape(2, 4, 1024, 64)
    assert torch.equal(aligned[:, 2, 19], points[:, :, 2 * 1024 + 19])
    expected_parts = expected if intermediate else (expected,)
    actual_parts = actual if intermediate else (actual,)
    for a, b in zip(expected_parts, actual_parts):
        assert torch.equal(a, b)
    sum(value.square().sum() for value in expected_parts).backward()
    sum(value.square().sum() for value in actual_parts).backward()
    assert torch.equal(first.grad, second.grad)
    for (name, value), (_, other) in zip(original.named_parameters(), exported.named_parameters()):
        assert torch.equal(value.grad, other.grad), name
    for name, value in original.state_dict().items():
        assert torch.equal(value, exported.state_dict()[name]), name


def _evidence(v28):
    torch.manual_seed(127)
    module = B2EvidenceAcquirer(v27_enabled=True, v28_enabled=v28,
        relation_aware_sampling=True, robust_consensus_voting=True).eval()
    with torch.no_grad():
        module.extension_presence_head[-1].weight.zero_()
        module.extension_presence_head[-1].bias.fill_(-20.)
    return module


def test_v28_low_presence_is_still_a_structural_bounded_candidate():
    values = _inputs()
    legacy = _evidence(False)(**values)
    output = _evidence(True)(**values)
    assert output['ct_b2_extension_presence_probability'].item() < 1e-6
    assert legacy['ct_search_candidate_valid'].item() == 0
    assert output['ct_b2_available'].item() == output['ct_search_candidate_valid'].item() == 1
    bounded, geometry = bounded_residual_xy(
        values['observation_box'], output['ct_b2_raw_box'], values['query_delta_t'])
    assert geometry['finite'].all()
    assert torch.linalg.vector_norm(bounded, dim=1).le(geometry['radius'] + 1e-6).all()


@pytest.mark.parametrize('bad', ['empty', 'nonfinite_points', 'nonfinite_center', 'negative_time', 'nonfinite_gap'])
def test_v28_rejects_nonstructural_evidence(bad):
    values = _inputs(extension_count=0 if bad == 'empty' else 4)
    if bad == 'nonfinite_points':
        values['extension_points'][values['extension_valid_mask']] = float('nan')
    elif bad == 'nonfinite_center':
        values['b1_center_xy'].fill_(float('nan'))
    elif bad == 'negative_time':
        values['query_delta_t'].fill_(-1.)
    elif bad == 'nonfinite_gap':
        values['gap_ratio'].fill_(float('inf'))
    output = _evidence(True)(**values)
    assert output['ct_b2_available'].item() == output['ct_search_candidate_valid'].item() == 0
    assert torch.equal(output['ct_b2_raw_box'], values['observation_box'])


def test_v28_still_requires_unique_extension_ids():
    values = _inputs()
    values['extension_point_ids'][0, 1] = values['extension_point_ids'][0, 0]
    with pytest.raises(ValueError, match='must be unique'):
        _evidence(True)(**values)
    with pytest.raises(ValueError, match='v27 point identity'):
        B2EvidenceAcquirer(v28_enabled=True)


def test_v28_evaluator_and_report_use_structural_validity_even_if_legacy_presence_gate_is_false():
    class Host(FakeHost):
        def _build_ct_joint_diagnostic_row(self, *args, previous_target_box=None, **kwargs):
            assert previous_target_box is not None
            return super()._build_ct_joint_diagnostic_row(*args, **kwargs)

        def evaluate_one_sample(self, batch, ref_box):
            final, valid, output = super().evaluate_one_sample(batch, ref_box)
            output['ct_b2_available'] = output['ct_policy_candidate_valid'].clone()
            output['ct_policy_candidate_valid'] = torch.zeros(1)
            return final, valid, output
    host = Host(True)
    host.config.ct_enable_v28 = True
    evaluate_sequence_v27(host, sequence())
    rows = host._ct_v27_sequence_endpoints
    assert rows[1]['structural_available'] and rows[1]['action_applied']
    assert rows[1]['presence_score'] == 0.
    assert not rows[2]['structural_available']
    summary = summarize_endpoint_diagnostics(rows)
    assert summary['schema'] == 'ct_seqtrack.endpoint_diagnostics.v28'
    assert summary['funnel']['structural_frames'] == 2
    assert summary['funnel']['action_frames'] == 2
