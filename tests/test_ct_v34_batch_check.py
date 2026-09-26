"""v34 单批入口复用真实事务宿主，版本记录不能回写成 v33。"""
from types import SimpleNamespace

import pytest
import torch

from models.ct_v31.config import load_config
from models.ct_v31.data import build_loaders
from tests.test_ct_v31_runtime import TinyJointTracker, TinySource
from tools import check_v33_batch as core
from tools import check_v34_batch as check


def test_v34_wrapper_selects_version_and_preserves_cli_arguments(monkeypatch):
    captured = {}

    def main(argv, *, version):
        captured.update(argv=argv, version=version)
        return 7

    monkeypatch.setattr(check, '_main', main)
    assert check.main(['--device', 'cuda']) == 7
    assert captured == {'argv': ['--device', 'cuda'], 'version': 34}


@pytest.mark.parametrize('name', ['34_b0_context_mini', '34_b0_context_w4_mini'])
def test_v34_batch_tool_runs_one_adam_and_commit_with_support(monkeypatch, name):
    import models.ctseqtrackv31 as host

    class DiagnosticTracker(TinyJointTracker):
        def forward(self, batch, prior=None):
            output = super().forward(batch, prior=prior)
            count = len(batch['points'])
            output.observation.sequence_valid = batch['point_valid'].flatten(1).any(dim=1)
            output.observation.history_support = torch.zeros(count, 3, 3)
            output.observation.current_support = torch.zeros(count, 3)
            output.decoder = SimpleNamespace(query_context_norm=torch.zeros(count))
            return output

    config = load_config('cfgs/ct_seqtrack/' + name + '.yaml',
                         dict(ct_engineering_check=True, workers=0))
    loaders = build_loaders(config, roles=('train',), sources={'train': TinySource((3,) * 16)})
    original = host.CTSEQTRACKV31
    monkeypatch.setattr(host, 'CTSEQTRACKV31', lambda config, loaders:
                        original(config, tracker=DiagnosticTracker(), loaders=loaders))
    report = {}
    core.run_one_batch(config, 'cpu', report, loaders=loaders)
    assert report['status'] == 'passed'
    assert report['optimizer']['lr'] == 5e-5
    assert report['optimizer']['state_step_values'] == [1]
    assert report['commit']['epoch_rows'] == 16 and report['commit']['epoch_steps'] == 1
    assert report['query_context']['norm'] == [0.] * 16
    assert len(report['query_context']['history_support']) == 16


def test_versioned_cli_rejects_other_model_before_importing_lightning(monkeypatch, tmp_path):
    import json
    import models.ct_v31.config as configuration

    monkeypatch.setattr(core, 'CHECK_ROOT', tmp_path)
    # 输出路径约束由独立测试覆盖；这里仅隔离版本拒绝发生在依赖导入之前。
    wrong_config = load_config('cfgs/ct_seqtrack/33_b0_mini.yaml', dict(ct_engineering_check=True))
    monkeypatch.setattr(configuration, 'load_config', lambda *args, **kwargs: wrong_config)
    destination = tmp_path / 'wrong_version.json'
    assert core.main(['--cfg', 'cfgs/ct_seqtrack/33_b0_mini.yaml',
                      '--output', str(destination), '--device', 'cpu'], version=34) == 1
    report = json.loads(destination.read_text(encoding='utf-8'))
    assert report['schema'] == 'ct_seqtrack.v34.batch_check.v1'
    assert report['error_type'] == 'ValueError'
    assert 'v34' in report['error']
    assert not report['checkpoint_saved']
