"""当前依赖闭包、真实张量尺寸与保留模块的执行合同。"""
import ast
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from models.ct_v38.config import normalize_config
from models.ct_v38.contracts import VerificationOutput
from models.ct_v38.model import JointTracker
from tests.test_ct_v38_network import values
from tests.test_ct_v38_data import cfg, output_for, next_batch
from tests.helpers.raw_tracking import TinySource
from models.ct_v38.data import BatchBuilder, RawEndpointDataset


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_activity_tree_has_no_legacy_python_imports():
    root = Path(__file__).resolve().parents[1]
    banned = ('models.ct_v31', 'models.ct_v36', 'models.ct_v37',
              'models.ctseqtrackv31', 'models.ctseqtrackv36', 'models.ctseqtrackv37')
    for folder in ('models', 'datasets', 'utils', 'tools', 'tests'):
        for path in (root/folder).rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            for node in ast.walk(tree):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else (
                    [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                assert not any(name.startswith(banned) for name in names), (path, names)


def test_verification_contract_exports_only_scores_support_and_diagnostics():
    assert {field.name for field in fields(VerificationOutput)} == {
        'logits', 'support_mask', 'metadata', 'local_neighbor_count',
        'local_selected_count', 'local_mean_support', 'local_delta_norm',
    }


@pytest.mark.parametrize('recipe', ['b1_standard', 'b1_hold', 'full_standard'])
def test_actual_formal_shape_has_only_b0_sources_and_four_box_queries(recipe):
    model = JointTracker(normalize_config(dict(v38_training_recipe=recipe))).eval()
    data = values(1, 1024)
    shapes = []
    local_shapes = []
    handle = model.decoder.decoder.register_forward_pre_hook(
        lambda module, args: shapes.append((tuple(args[0].shape), tuple(args[2].shape), tuple(args[3].shape))))
    local_handle = model.decoder.local_reader.point_encoder.register_forward_pre_hook(
        lambda module, args: local_shapes.append(tuple(args[0].shape)))
    with torch.no_grad():
        output = model(data)
    handle.remove()
    local_handle.remove()
    assert shapes == [((1,32,64),(1,1024,64),(1,32,1024))]
    assert local_shapes == [(1,1,8,16,69)]
    assert output.decoder.q0_box.shape == (1,4)
    assert output.decoder.history_boxes.shape == (1,3,4)
    assert output.decoder.local_selected_count.shape == (1,8)
    assert not hasattr(output, 'evidence') and not hasattr(model, 'evidence')
    assert not hasattr(output.candidates, 'decoder')
    assert not any('extension' in name or 'memory' in name for name in model.state_dict())


@pytest.mark.parametrize('support', [2,3])
@pytest.mark.parametrize('trust', [.499999,.5,.500001])
def test_commit_threshold_is_inclusive_and_support_is_unique(support, trust):
    builder = BatchBuilder(cfg())
    _, batch = next_batch(builder, RawEndpointDataset(TinySource()), 1)
    probability = torch.zeros_like(batch['point_valid'][:, -1], dtype=torch.float)
    probability[:, :support] = 1.
    result = builder.commit(output_for(batch, trusted=trust, probabilities=probability), batch, diagnostics=True)[0]
    assert result['supported'] is (support >= 3 and trust >= .5)


def test_empty_q0_still_trains_original_quality_but_not_main_box():
    model = JointTracker(normalize_config(dict(ct_engineering_check=True, v38_arm='b1')))
    data = values(1)
    data['point_valid'].zero_()
    output = model(data)
    output.decoder.q0_quality_logits.retain_grad()
    losses = model.compute_losses(data, output)
    assert losses['loss_main'].item() == losses['loss_coarse'].item() == 0.
    assert losses['loss_quality'].item() > 0.
    losses['loss_quality'].backward()
    assert output.decoder.q0_quality_logits.grad.abs().sum() > 0
