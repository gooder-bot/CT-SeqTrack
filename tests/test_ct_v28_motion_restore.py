"""v28 物理运动诊断与按已构造模块恢复 checkpoint。"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.checkpoint_loading import load_initial_weights
from utils.v27_evaluation import evaluate_sequence_v27
from tests.test_ct_v27_evaluation import FakeHost, sequence


def _diagnostic_host(degrees=False):
    source = ast.parse(Path('models/base_model.py').read_text(encoding='utf-8'))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'BaseModelMF')
    selected = [copy.deepcopy(node) for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {'_proposal_scalar', '_build_v28_motion_diagnostics'}]
    namespace = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'actual_v28_motion_diagnostics', 'exec'), namespace)
    host = SimpleNamespace(config=SimpleNamespace(degrees=degrees))
    host._proposal_scalar = namespace['_proposal_scalar'].__func__
    host.diagnostics = namespace['_build_v28_motion_diagnostics'].__get__(host)
    return host


def _box(center, yaw=0.):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return SimpleNamespace(center=np.asarray(center, dtype=float),
                           rotation_matrix=np.asarray(((cosine, -sine, 0.),
                                                       (sine, cosine, 0.), (0., 0., 1.))))


@pytest.mark.parametrize('degrees', [False, True])
def test_motion_error_and_nll_exclude_recursive_drift_and_rotate_source_axes(degrees):
    host = _diagnostic_host(degrees)
    reference = _box([7., 3., 0.], np.pi / 3)
    previous, current = _box([1., 2., 0.]), _box([3., 2., 0.])
    # source yaw=90°，局部(0,-2)恰是世界坐标(+2,0)物理位移。
    source_yaw = 90. if degrees else np.pi / 2
    data = dict(motion_source_anchor=torch.tensor([[6., 2., 0., source_yaw]], dtype=torch.float64))
    output = dict(motion_prior_xy=torch.tensor([[0., -2.]]),
                  motion_prior_kinematic_xy=torch.tensor([[0., -1.]]),
                  motion_prior_direction_xy=torch.tensor([[0., -1.]]),
                  motion_prior_log_sigma_parallel_perp=torch.zeros(1, 2),
                  motion_prior_envelope_parallel_perp=torch.ones(1, 2),
                  motion_prior_valid=torch.ones(1))
    original_data, original_output = copy.deepcopy(data), copy.deepcopy(output)
    row = host.diagnostics(output, data, current, reference, previous)
    assert row['b1_target_kind'] == 'physical_displacement'
    assert row['learned_motion_error'] == pytest.approx(0., abs=1e-12)
    assert row['kinematic_error'] == pytest.approx(1.)
    assert row['learned_endpoint_error'] == pytest.approx(5.)
    assert row['kinematic_endpoint_error'] == pytest.approx(4.)
    assert row['recursive_anchor_error'] == pytest.approx(5.)
    assert row['b1_nll'] == pytest.approx(0., abs=1e-12)
    assert row['b1_mahalanobis_sq'] == pytest.approx(0., abs=1e-12)
    assert row['target_residual_unit_parallel'] == pytest.approx(1.)
    assert row['target_residual_unit_perpendicular'] == pytest.approx(0., abs=1e-12)
    assert row['b1_coverage_95'] == 1
    for values, snapshot in ((data, original_data), (output, original_output)):
        assert all(torch.equal(value, snapshot[key]) for key, value in values.items())
    with pytest.raises(ValueError, match='previous target'):
        host.diagnostics(output, data, current, reference, None)


def test_endpoint_collection_runs_motion_diagnostics_without_heavy_export():
    helper = _diagnostic_host().diagnostics

    class Host(FakeHost):
        def build_input_dict(self, *args, **kwargs):
            batch, anchor = super().build_input_dict(*args, **kwargs)
            anchor.rotation_matrix = np.eye(3)
            batch['motion_source_anchor'] = torch.tensor([[*anchor.center, 0.]], dtype=torch.float64)
            return batch, anchor

        def evaluate_one_sample(self, *args, **kwargs):
            final, valid, output = super().evaluate_one_sample(*args, **kwargs)
            output['motion_prior_xy'] = torch.zeros(1, 2)
            output['motion_prior_valid'] = torch.ones(1)
            output['ct_b2_available'] = output['ct_policy_candidate_valid']
            return final, valid, output

        def _build_v28_motion_diagnostics(self, *args):
            return helper(*args)

    host = Host(False)
    host.config.ct_enable_v28 = True
    host.config.export_proposal_diagnostics = False
    frames = sequence()
    evaluate_sequence_v27(host, frames)
    rows = host._ct_v27_sequence_endpoints
    assert not host._proposal_sequence_diagnostics
    assert all(row['learned_motion_error'] == 0. for row in rows[1:])
    assert rows[2]['learned_endpoint_error'] == pytest.approx(.5)


class RestoreProbe(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.config = dict(ct_enable_v27=True, ct_enable_v28=True,
                           ct_enable_b1=arm not in ('b0', 'reference'),
                           ct_enable_b2=arm == 'full', ct_enable_b3=arm == 'full')
        for name in ('seg_pointnet', 'mini_pointnet', 'motion_mlp', 'feature_pointnet', 'Transformer'):
            setattr(self, name, torch.nn.Linear(1, 1))
        if self.config['ct_enable_b1']:
            self.physical_motion_encoder = torch.nn.Linear(1, 1)
        if arm == 'full':
            self.ct_joint_search_refiner = torch.nn.Linear(1, 1)
            self.ct_joint_router = torch.nn.Linear(1, 1)


@pytest.mark.parametrize('arm', ['b0', 'reference', 'b1', 'full'])
def test_complete_restore_requires_only_existing_enabled_modules(arm, tmp_path):
    original, restored = RestoreProbe(arm), RestoreProbe(arm)
    path = tmp_path / 'same_run.ckpt'
    torch.save(dict(state_dict=original.state_dict(), hyper_parameters=dict(config=original.config)), path)
    assert load_initial_weights(restored, path, require_complete=True)['complete']
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


@pytest.mark.parametrize('prefix', ['physical_motion_encoder.', 'ct_joint_search_refiner.', 'ct_joint_router.'])
def test_partial_restore_rejects_missing_enabled_plugin(prefix, tmp_path):
    model = RestoreProbe('full')
    path = tmp_path / 'incomplete.ckpt'
    state = {key: value for key, value in model.state_dict().items() if not key.startswith(prefix)}
    torch.save(dict(state_dict=state, hyper_parameters=dict(config=model.config)), path)
    with pytest.raises(RuntimeError, match=prefix):
        load_initial_weights(model, path)
