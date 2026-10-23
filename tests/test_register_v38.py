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
    target = tmp_path / 'cfgs/ct_seqtrack/38_clean1_registration.json'
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


def test_verify_is_read_only_and_works_with_existing_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(register_v38, 'ROOT', tmp_path)
    target = tmp_path / 'cfgs/ct_seqtrack/38_clean1_registration.json'
    target.parent.mkdir(parents=True)
    value = {'source': {'sha256': 'same', 'files': {'models/current.py': 'sha'}}}
    target.write_text(json.dumps(value), encoding='utf-8')
    before = target.read_bytes(), target.stat().st_mtime_ns
    manifest = tmp_path / 'output/existing/run_manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"model": "ctseqtrackv38"}', encoding='utf-8')
    monkeypatch.setattr(register_v38, 'build_registration', lambda: value)
    register_v38.main(['--verify'])
    assert 'verified' in capsys.readouterr().out
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


@pytest.mark.parametrize('section', ['source', 'runs', 'generator_sha256'])
def test_verify_rejects_mismatch_without_rewriting_registration(tmp_path, monkeypatch, section):
    monkeypatch.setattr(register_v38, 'ROOT', tmp_path)
    target = tmp_path / 'cfgs/ct_seqtrack/38_clean1_registration.json'
    target.parent.mkdir(parents=True)
    value = dict(source={'sha256': 'same', 'files': {}}, runs=[], generator_sha256='same')
    target.write_text(json.dumps(value), encoding='utf-8')
    before = target.read_bytes(), target.stat().st_mtime_ns
    actual = dict(value)
    actual[section] = ({'sha256': 'other', 'files': {'models/retired.py': 'x'}}
                       if section == 'source' else 'other')
    monkeypatch.setattr(register_v38, 'build_registration', lambda: actual)
    with pytest.raises(ValueError, match=section):
        register_v38.main(['--verify'])
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_verify_and_refresh_cannot_be_combined():
    with pytest.raises(SystemExit):
        register_v38.main(['--verify', '--refresh'])
