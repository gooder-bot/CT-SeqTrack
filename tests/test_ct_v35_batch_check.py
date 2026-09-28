"""新版真实单批入口沿用一次forward/backward/Adam/commit事务。"""
from types import SimpleNamespace
import pytest
import torch

from tools import check_v35_batch as check
from tests.test_ct_v35_identity import config35
from models.ct_v31.data import build_loaders
from tests.test_ct_v31_runtime import TinyJointTracker, TinySource


def test_v35_cli_defaults_and_protected_output(tmp_path, monkeypatch):
    assert check.parse_args([]).cfg.name == '35_b0_w_quarter_lr_mini.yaml'
    monkeypatch.setattr(check, 'CHECK_ROOT', tmp_path / 'ct_checks')
    with pytest.raises(ValueError):
        check.report_destination(tmp_path / 'outside.json')
    destination = check.report_destination(None)
    destination.parent.mkdir(parents=True)
    destination.write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError):
        check.report_destination(destination)


def test_v35_single_batch_host_commit_and_local_diagnostics(monkeypatch):
    import models.ctseqtrackv31 as host
    class TinyV35(TinyJointTracker):
        def forward(self, batch, prior=None):
            out = super().forward(batch, prior=prior)
            n = len(batch['points'])
            out.observation.sequence_valid = batch['point_valid'].flatten(1).any(-1)
            out.observation.history_support = torch.zeros(n, 3, 3)
            out.observation.current_support = torch.zeros(n, 3)
            out.decoder = SimpleNamespace(query_context_norm=torch.zeros(n), **{
                key: torch.zeros(n, 4, 8) for key in ('local_neighbor_count', 'local_selected_count',
                    'local_mean_support', 'local_delta_norm')})
            return out
    cfg = config35(ct_engineering_check=True, workers=0, v35_train_diagnostics=False)
    loaders = build_loaders(cfg, roles=('train',), sources={'train': TinySource((3,) * 16)})
    original = host.CTSEQTRACKV31
    monkeypatch.setattr(host, 'CTSEQTRACKV31', lambda config, loaders: original(config, tracker=TinyV35(), loaders=loaders))
    report = {}
    check.run_one_batch(cfg, 'cpu', report, loaders=loaders)
    assert report['status'] == 'passed' and report['optimizer']['steps'] == 1
    assert report['commit']['epoch_rows'] == 16 and not report['commit']['pending_builder']
    assert len(report['local_evidence']['local_neighbor_count']) == 16


def test_old_or_reduced_batch_rejected():
    from models.ct_v31.config import normalize_config
    for cfg in (normalize_config(dict(ct_engineering_check=True)),
                config35(ct_engineering_check=True, batch_size=2)):
        with pytest.raises(ValueError, match='v35 B0'):
            check.run_one_batch(cfg, 'cpu', {}, loaders={})
