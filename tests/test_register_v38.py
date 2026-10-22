"""登记生成不按目录名识别运行，并兼容服务器 Python 3.9 的文本写入。"""
import json

import pytest

from tools import register_v38


def test_registration_refuses_nested_custom_named_existing_v38_run(tmp_path, monkeypatch):
    monkeypatch.setattr(register_v38, 'ROOT', tmp_path)
    run = tmp_path / 'output/custom/nested/run_manifest.json'
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({'model': 'ctseqtrackv38'}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='v38 run already exists'):
        register_v38.main(['--refresh'])


def test_registration_new_and_explicit_refresh_use_exact_utf8_lf(tmp_path, monkeypatch):
    monkeypatch.setattr(register_v38, 'ROOT', tmp_path)
    target = tmp_path / 'cfgs/ct_seqtrack/38_registration.json'
    target.parent.mkdir(parents=True)
    value = {'source': {'sha256': 'x'}, 'notes': '三组'}
    monkeypatch.setattr(register_v38, 'build_registration', lambda: value)
    register_v38.main([])
    first = target.read_bytes()
    assert b'\r\n' not in first and json.loads(first)['notes'] == '三组'
    with pytest.raises(FileExistsError):
        register_v38.main([])
    register_v38.main(['--refresh'])
    assert target.read_bytes() == first
