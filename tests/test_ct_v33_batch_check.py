"""单批工程工具的事务、诊断与输出边界；不依赖真实数据或 CUDA。"""
from pathlib import Path

import pytest
import torch

from models.ct_v31.config import normalize_config
from models.ct_v31.data import build_loaders
from tests.test_ct_v31_runtime import TinyJointTracker, TinySource
from tools import check_v33_batch as check


def test_report_destination_never_overwrites_or_leaves_check_root(tmp_path, monkeypatch):
    monkeypatch.setattr(check, 'CHECK_ROOT', tmp_path / 'ct_checks')
    first, second = check.report_destination(None), check.report_destination(None)
    assert first != second and first.name == 'report.json'
    assert first.parent.parent == check.CHECK_ROOT
    with pytest.raises(ValueError, match='under artifacts'):
        check.report_destination(tmp_path / 'outside.json')
    with pytest.raises(ValueError, match='under artifacts'):
        check.report_destination(check.CHECK_ROOT / 'report.txt')
    first.parent.mkdir(parents=True)
    first.write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError, match='overwrite'):
        check.report_destination(first)


def test_gradient_summary_measures_global_norm_and_rejects_nonfinite():
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[3., 4.]])
    report = check.gradient_summary(model)
    assert report['global_l2_norm'] == pytest.approx(5.)
    assert report['tensors_with_gradient'] == 1
    assert report['parameters_without_gradient'] == ['bias']
    model.weight.grad[0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='nonfinite gradient'):
        check.gradient_summary(model)


def test_single_batch_uses_real_host_forward_adam_and_commit(monkeypatch):
    import models.ctseqtrackv31 as host

    class DiagnosticTracker(TinyJointTracker):
        def forward(self, batch, prior=None):
            output = super().forward(batch, prior=prior)
            output.observation.sequence_valid = batch['point_valid'].flatten(1).any(dim=1)
            return output

    config = normalize_config(dict(ct_engineering_check=True, workers=0))
    loaders = build_loaders(config, roles=('train',), sources={'train': TinySource((3,) * 16)})
    original, created = host.CTSEQTRACKV31, []

    def construct(config, loaders):
        model = original(config, tracker=DiagnosticTracker(), loaders=loaders)
        created.append(model)
        return model

    monkeypatch.setattr(host, 'CTSEQTRACKV31', construct)
    report = {}
    check.run_one_batch(config, 'cpu', report, loaders=loaders)
    assert report['status'] == 'passed' and report['stage'] == 'complete'
    assert report['batch']['shape'] == [16, 4, 1024, 5]
    assert report['optimizer']['state_step_values'] == [1]
    assert report['gradients']['all_finite']
    assert report['commit']['epoch_rows'] == 16 and report['commit']['epoch_steps'] == 1
    assert not report['commit']['pending_host'] and not report['commit']['pending_builder']
    assert created[0]._pending_train is None and created[0].train_builder._pending is None
    assert created[0].tracker.prior_calls == 1
    assert created[0].tracker.shift.item() != pytest.approx(.03)


def test_single_batch_refuses_reduced_budget_before_building_loaders():
    config = normalize_config(dict(ct_engineering_check=True, batch_size=2, workers=0))
    with pytest.raises(ValueError, match='batch_size=16'):
        check.run_one_batch(config, 'cpu', {}, loaders={})


def test_cli_defaults_to_registered_b0_full_budget():
    args = check.parse_args([])
    assert args.cfg == check.ROOT / 'cfgs/ct_seqtrack/33_b0_mini.yaml'
    assert args.device == 'cuda' and args.output is None
    assert check.parse_args(['--output', 'result.json']).output == Path('result.json')
