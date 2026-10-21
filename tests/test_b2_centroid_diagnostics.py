"""新 centroid 配方仅追加被动分子/计数，不改变训练图或旧诊断。"""
from copy import deepcopy
from dataclasses import is_dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from models.ct_v31.runtime import capture_rng_state
from models.ct_v37.config import normalize_config
from models.ct_v37.model import JointTracker
from models.ct_v37.training_diagnostics import TrainingDiagnostics
from tests.test_ct_v34_resume import _assert_identical
from tests.test_ct_v37_network import batch, config


PREFIX = 'v37_centroid_'


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(3721)
    yield
    torch.set_num_threads(previous)


def _forward(count=2):
    cfg = normalize_config(dict(config(), v37_centroid_recipe='full_mass_detach'))
    model = JointTracker(cfg).train()
    values = batch(count, 8)
    values['history_box_provenance'] = torch.zeros(count, 3, dtype=torch.long)
    rows = [dict(tracklet_key=f'centroid/{i}', request=SimpleNamespace(
        epoch=0, branch=i, frame=1, window_start=1)) for i in range(count)]
    output = model(values)
    output.loss_statistics = {}
    losses = model.compute_losses(values, output)
    return model, rows, values, output, losses


def _tensor_snapshot(value, path=''):
    if torch.is_tensor(value):
        return {path: value.detach().clone()}
    if is_dataclass(value):
        value = vars(value)
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, child in value.items():
        result.update(_tensor_snapshot(child, path + '/' + str(name)))
    return result


def _without_centroid_statistics(report):
    report = deepcopy(report)
    for bucket in (report['totals'], *report['groups'].values()):
        bucket['statistics'] = {key: value for key, value in bucket['statistics'].items()
                                if not key.startswith(PREFIX)}
    return report


@pytest.mark.parametrize('optimized', [False, True])
def test_centroid_diagnostics_preserve_outputs_losses_gradients_rng_and_legacy_report(optimized):
    model, rows, values, output, losses = _forward()
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(losses['loss_total'], parameters,
                                    allow_unused=True, retain_graph=True)
    snapshot = _tensor_snapshot(dict(batch=values, output=output, losses=losses))
    rng = capture_rng_state()
    default = TrainingDiagnostics(seed=42, optimized=optimized)
    disabled = TrainingDiagnostics(seed=42, optimized=optimized, centroid_semantics=False)
    enabled = TrainingDiagnostics(seed=42, optimized=optimized, centroid_semantics=True)
    for diagnostics in (default, disabled, enabled):
        diagnostics.add_batch(rows, values, output, losses)
        _assert_identical(rng, capture_rng_state())
        _assert_identical(snapshot, _tensor_snapshot(dict(batch=values, output=output, losses=losses)))
    after = torch.autograd.grad(losses['loss_total'], parameters, allow_unused=True)
    _assert_identical(gradients, after)
    _assert_identical(default.summary(), disabled.summary())
    _assert_identical(default.summary(), _without_centroid_statistics(enabled.summary()))
    assert any(name.startswith(PREFIX) for name in enabled.summary()['totals']['statistics'])
    assert not any(name.startswith(PREFIX) for name in default.summary()['totals']['statistics'])


def test_centroid_semantics_count_all_valid_identity_points_and_only_selected_reliability_points():
    _, rows, values, output, losses = _forward(3)
    # 第768槽也计入全池身份摘要；高分/NaN padding 及未选点不能进入可靠性摘要。
    values['extension_valid'].zero_()
    values['extension_labels'].fill_(1.)
    values['extension_valid'][0, [0, 1, 2, 767]] = True
    values['extension_labels'][0, [2, 767]] = 0.
    values['extension_valid'][1, [4, 767]] = True
    values['extension_labels'][1, [4, 767]] = 0.
    identity = torch.full_like(output.evidence.identity_logits, torch.nan)
    identity[0, [0, 1, 2, 767]] = torch.logit(torch.tensor([.8, .6, .25, .75]))
    identity[1, [4, 767]] = torch.logit(torch.tensor([.2, .9]))
    indices = torch.full_like(output.evidence.point_indices, -1)
    indices[0, :2] = torch.tensor([0, 2])
    indices[1, 0] = 767
    selected = indices >= 0
    reliability = torch.full_like(output.evidence.reliability_logits, torch.nan)
    reliability[0, :2] = torch.logit(torch.tensor([.7, .4]))
    reliability[1, 0] = torch.logit(torch.tensor(.3))
    votes = values['target_box'][:, None, :3].expand(-1, 256, -1).clone()
    radius = .5 * values['box_size'][0, :2].norm()
    votes[0, 0, 0] += radius  # FG 软目标=1/(1+r/r)=0.5；BG 即使票在中心目标仍为0。
    output.evidence = replace(output.evidence, identity_logits=identity,
        point_indices=indices, point_valid=selected, vote_xyz=votes,
        reliability_logits=reliability, members=torch.zeros_like(output.evidence.members),
        mode_valid=torch.zeros_like(output.evidence.mode_valid))
    diagnostics = TrainingDiagnostics(centroid_semantics=True, optimized=True)
    diagnostics.add_batch(rows, values, output, losses)
    report = diagnostics.summary()
    stats = report['totals']['statistics']
    expected = dict(identity_fg_probability_sum=1.4, identity_fg_probability_count=2,
        identity_bg_probability_sum=2.1, identity_bg_probability_count=4,
        reliability_fg_abs_error_sum=.2, reliability_fg_abs_error_count=1,
        reliability_bg_abs_error_sum=.7, reliability_bg_abs_error_count=2)
    assert {key for key in stats if key.startswith(PREFIX)} == {PREFIX + key for key in expected}
    for name, value in expected.items():
        assert stats[PREFIX + name]['sum'] == pytest.approx(value, abs=1e-6)
    # 不能把缺失类别补作一个“均值为0”的有效观测；真实点分母必须仍为0。
    missing_fg = report['groups']['b1/d0/p0,0,0']['statistics']
    empty = report['groups']['b2/d0/p0,0,0']['statistics']
    for name in expected:
        assert empty[PREFIX + name]['sum'] == 0.
        if '_fg_' in name:
            assert missing_fg[PREFIX + name]['sum'] == 0.
    assert stats[PREFIX + 'identity_fg_probability_sum']['sum'] / stats[
        PREFIX + 'identity_fg_probability_count']['sum'] == pytest.approx(.7)
