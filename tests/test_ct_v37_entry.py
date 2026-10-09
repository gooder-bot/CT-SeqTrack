"""实际随机初始化身份与正式注册入口锁。"""
import json
from datetime import datetime

import pytest
import torch

from models.ct_v37.config import config_identity, load_config
from models.ct_v37.entry import initialization_identity, validate_formal_registration
from models.ct_v37.model import JointTracker


def test_common_initialization_is_shared_and_full_precision_does_not_change_weights():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        hashes = {}
        for label in ('b0', 'b1', 'b1_b2', 'full', 'full_bf16'):
            torch.manual_seed(42)
            tracker = JointTracker(load_config(f'cfgs/ct_seqtrack/37_{label}_mini.yaml'))
            state = torch.get_rng_state().clone()
            hashes[label] = initialization_identity(tracker)
            assert torch.equal(state, torch.get_rng_state())
        assert len({value['common_parameter_hash'] for value in hashes.values()}) == 1
        assert hashes['b0']['prior_parameter_hash'] is None
        assert len({hashes[label]['prior_parameter_hash'] for label in ('b1', 'b1_b2', 'full', 'full_bf16')}) == 1
        assert hashes['full'] == hashes['full_bf16'] == hashes['b1_b2']
    finally:
        torch.set_num_threads(previous)


def test_formal_entry_rejects_config_drift_and_source_drift(tmp_path):
    cfg = load_config('cfgs/ct_seqtrack/37_full_mini.yaml')
    source = {'sha256': 'f' * 64, 'files': {'model.py': 'a' * 64}}
    registration = dict(schema='ct_seqtrack.v37.registration.v1', source=source,
        runs=[dict(label='full', arm='full', config_sha256=config_identity(cfg),
                   execution_profile='optimized_fp32')])
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(registration), encoding='utf-8')
    assert validate_formal_registration(cfg, source, registration_path=path)['label'] == 'full'
    with pytest.raises(ValueError, match='source differs'):
        validate_formal_registration(cfg, {'sha256': 'wrong'}, registration_path=path)
    # 未被数学固定字段清单覆盖的意外配置也不能穿过完整 config hash。
    cfg.experiment_name = 'unexpected_unregistered_experiment'
    with pytest.raises(ValueError, match='not exactly one registered'):
        validate_formal_registration(cfg, source, registration_path=path)


def test_same_clock_resume_events_are_unique_and_preserve_original_metadata(tmp_path, monkeypatch):
    from models.ct_v37 import entry
    cfg = load_config('cfgs/ct_seqtrack/37_full_mini.yaml')
    cfg.checkpoint = str(tmp_path / 'epoch=010.ckpt')
    original = dict(run_id='same-original-run', source={'sha256': 'a' * 64, 'files': {}})
    entry.write_run_metadata(cfg, tmp_path, original, preserve_existing=False)
    snapshots = {name: (tmp_path / name).read_bytes()
                 for name in ('run_manifest.json', 'resolved_config.yaml')}
    class FixedClock:
        @staticmethod
        def now():
            return datetime(2026, 10, 9, 16, 0, 37, 854121)
    monkeypatch.setattr(entry, 'datetime', FixedClock)
    paths = [entry.write_resume_event(cfg, tmp_path, original) for _ in range(2)]
    assert paths[0] != paths[1]
    assert len(list((tmp_path / 'resume_manifests').iterdir())) == 2
    assert all(path.name.startswith('20261009-160037-854121-') for path in paths)
    records = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    assert records[0] == records[1]
    assert records[0]['run_id'] == original['run_id']
    assert records[0]['source'] == original['source']
    assert records[0]['config_sha256'] == config_identity(cfg)
    entry.write_run_metadata(cfg, tmp_path, {'replacement': True}, preserve_existing=True)
    assert all((tmp_path / name).read_bytes() == content for name, content in snapshots.items())
