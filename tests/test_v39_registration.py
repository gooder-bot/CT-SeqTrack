"""v39 部署差异诊断与 CUDA/真实数据检查的登记前置约束。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch

from tools import register_v39 as registration


def saved_registration(tmp_path):
    value = dict(schema='fixture', source=dict(sha256='registered-source', files={
        'models/ct_v39/g_observation.py': 'g-expected',
        'models/ct_v39/model.py': 'model-expected',
    }))
    path = tmp_path / '39_registration.json'
    path.write_text(json.dumps(value, indent=3) + '\n', encoding='utf-8')
    return path, value


def test_verify_matching_registration_is_read_only(tmp_path, monkeypatch):
    path, value = saved_registration(tmp_path)
    original, modified = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(registration, 'build_registration', lambda: deepcopy(value))
    assert registration.verify_registration(path) == value
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == modified
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('kind,name', [
    ('changed', 'models/ct_v39/g_observation.py'),
    ('missing', 'models/ct_v39/model.py'),
    ('unexpected', 'models/ct_v39/old_copy.py'),
])
def test_source_difference_names_files_without_rewriting_registration(tmp_path, monkeypatch, kind, name):
    path, value = saved_registration(tmp_path)
    original, modified = path.read_bytes(), path.stat().st_mtime_ns
    actual = deepcopy(value)
    actual['source']['sha256'] = 'different-source'
    files = actual['source']['files']
    if kind == 'missing':
        del files[name]
    else:
        files[name] = 'different-content'
    monkeypatch.setattr(registration, 'build_registration', lambda: actual)
    with pytest.raises(ValueError) as error:
        registration.verify_registration(path)
    message = str(error.value)
    assert 'v39 registration mismatch: source' in message
    assert kind + ': ' + name in message
    assert 'registered source: registered-source' in message
    assert 'actual source: different-source' in message
    assert 'do not refresh to hide drift' in message
    assert path.read_bytes() == original and path.stat().st_mtime_ns == modified
    assert list(tmp_path.iterdir()) == [path]


def test_real_batch_rejects_registration_before_cuda_or_data(tmp_path, monkeypatch, capsys):
    from models.ct_v39 import data, entry
    from tools import check_v39_full_batch as check

    destination = tmp_path / 'failed' / 'report.json'
    monkeypatch.setattr(check, 'report_destination', lambda *args, **kwargs: destination)
    mismatch = 'v39 registration mismatch: source\nchanged: models/ct_v39/g_observation.py'
    verify = Mock(side_effect=ValueError(mismatch))
    forbidden = Mock(side_effect=AssertionError('must stop before CUDA or real data'))
    monkeypatch.setattr(registration, 'verify_registration', verify)
    monkeypatch.setattr(entry, 'configure_numerics', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(data, 'build_loaders', forbidden)
    monkeypatch.setattr(check, 'run_check', forbidden)

    assert check.main(['--device', 'cuda:0']) == 1
    verify.assert_called_once_with()
    forbidden.assert_not_called()
    report = json.loads(destination.read_text(encoding='utf-8'))
    assert report['status'] == 'failed' and report['stage'] == 'verify_registration'
    assert report['error_type'] == 'ValueError' and report['error'] == mismatch
    printed = json.loads(capsys.readouterr().out)
    assert printed['error'] == mismatch and printed['error_type'] == 'ValueError'


def test_benchmark_rejects_registration_before_cuda_data_or_output(tmp_path, monkeypatch):
    from models.ct_v39 import entry
    from tools import benchmark_v39 as benchmark

    destination = tmp_path / 'artifacts' / 'ct_checks' / 'profile'
    cfg = Path('cfgs/ct_seqtrack/39_full_a_mini.yaml').resolve()
    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['benchmark_v39.py', '--cfg', str(cfg),
                                    '--training', '--output', str(destination)])
    mismatch = 'v39 registration mismatch: source\nmissing: models/ct_v39/model.py'
    verify = Mock(side_effect=ValueError(mismatch))
    forbidden = Mock(side_effect=AssertionError('must stop before CUDA or real data'))
    monkeypatch.setattr(registration, 'verify_registration', verify)
    monkeypatch.setattr(entry, 'apply_device_selection', forbidden)
    monkeypatch.setattr(entry, 'configure_numerics', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(benchmark, 'build_loaders', forbidden)
    monkeypatch.setattr(benchmark, 'run', forbidden)
    with pytest.raises(ValueError, match='registration mismatch') as error:
        benchmark.main()
    assert str(error.value) == mismatch
    verify.assert_called_once_with()
    forbidden.assert_not_called()
    assert not destination.exists()


def test_synthetic_batch_allows_unregistered_engineering_worktree(tmp_path, monkeypatch):
    from models.ct_v39 import config as configuration, entry
    from tools import check_v39_full_batch as check

    # 隔离工程产物根，仍保留真实 normalize_config 的输出范围检查。
    monkeypatch.setattr(configuration, '__file__', str(tmp_path / 'models/ct_v39/config.py'))
    destination = tmp_path / 'artifacts/ct_checks/synthetic/report.json'
    monkeypatch.setattr(check, 'report_destination', lambda *args, **kwargs: destination)
    verify = Mock(side_effect=ValueError('unregistered engineering edit'))
    monkeypatch.setattr(registration, 'verify_registration', verify)
    numerics = Mock()
    monkeypatch.setattr(entry, 'configure_numerics', numerics)
    monkeypatch.setattr(entry, 'source_identity', lambda: dict(sha256='engineering-working-tree'))
    calls = []

    def run(config, args, report, output):
        assert config.ct_engineering_check and args.synthetic and args.device == 'cpu'
        assert config.v39_attention_backend == 'math_reference'
        calls.append(output)
        report.update(status='passed', stage='complete')

    monkeypatch.setattr(check, 'run_check', run)
    assert check.main(['--synthetic', '--device', 'cpu', '--batch-size', '2']) == 0
    verify.assert_not_called()
    numerics.assert_called_once_with()
    assert calls == [destination]
    report = json.loads(destination.read_text(encoding='utf-8'))
    assert report['source']['sha256'] == 'engineering-working-tree'
    assert 'registered_source_sha256' not in report
