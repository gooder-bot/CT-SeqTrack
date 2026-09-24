"""入口在写 metadata 前拒绝错误恢复，合法恢复保留首次运行证据。"""

import json

import pytest
import torch

from models.ct_v31 import entry
from models.ct_v31.config import config_identity, normalize_config
from models.ct_v31.contracts import SCHEMA
from models.ct_v31.data import ReadyQueueBatchSampler
from models.ct_v31.runtime import resume_payload


def resume_config(tmp_path, *, test=False):
    checkpoint = tmp_path / 'epoch=001.ckpt'
    config = normalize_config(dict(checkpoint=str(checkpoint), test=test,
                                   log_dir=str(tmp_path / 'run')))
    sampler = ReadyQueueBatchSampler([3], 16)
    payload = resume_payload(config, completed_epoch=1, complete=True,
        rows=sampler.row_count, steps=len(sampler), sampler=sampler.state_dict())
    torch.save({'ct_v33_runtime': payload}, checkpoint)
    return config, payload


def existing_metadata(config, root, **overrides):
    root.mkdir()
    manifest = dict(schema=SCHEMA, model=config.net_model,
                    config_sha256=config_identity(config), source={'original': 'frozen'})
    manifest.update(overrides)
    entry.write_run_metadata(config, root, manifest, preserve_existing=False)
    (root / 'training_budget.json').write_text('{"original": true}\n', encoding='utf-8')
    return snapshot(root)


def snapshot(root):
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}


def test_invalid_checkpoint_is_rejected_before_creating_run_directory(tmp_path, monkeypatch):
    config, payload = resume_config(tmp_path)
    payload['schema'] = 'ct_seqtrack.v32.epoch_boundary.v1'
    torch.save({'ct_v33_runtime': payload}, config.checkpoint)
    monkeypatch.setattr(entry, 'configure_numerics', lambda: None)
    with pytest.raises(ValueError, match='runtime schema'):
        entry.run(config)
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('changed', [dict(schema='ct_seqtrack.joint_identity.v32'),
                                   dict(config_sha256='wrong_recipe'),
                                   dict(model='seqtrack_reference')])
def test_existing_run_identity_mismatch_never_changes_files(tmp_path, monkeypatch, changed):
    config, _ = resume_config(tmp_path)
    root = tmp_path / 'run'
    before = existing_metadata(config, root, **changed)
    monkeypatch.setattr(entry, 'configure_numerics', lambda: None)
    with pytest.raises(ValueError, match='existing run identity mismatch'):
        entry.run(config)
    assert snapshot(root) == before


def test_wrong_checkpoint_recipe_preserves_matching_directory_metadata(tmp_path, monkeypatch):
    config, payload = resume_config(tmp_path)
    root = tmp_path / 'run'
    before = existing_metadata(config, root)
    payload['config_sha256'] = 'another_recipe'
    torch.save({'ct_v33_runtime': payload}, config.checkpoint)
    monkeypatch.setattr(entry, 'configure_numerics', lambda: None)
    with pytest.raises(ValueError, match='configuration identity mismatch'):
        entry.run(config)
    assert snapshot(root) == before


def test_legal_same_run_resume_preserves_original_metadata_bytes(tmp_path):
    config, _ = resume_config(tmp_path)
    root = tmp_path / 'run'
    before = existing_metadata(config, root)
    preserve = entry.validate_run_destination(config, root)
    assert preserve
    entry.write_run_metadata(config, root, dict(source={'current': 'new_environment'}),
                             preserve_existing=preserve)
    assert snapshot(root) == before


def test_evaluation_requires_separate_empty_directory(tmp_path, monkeypatch):
    config, _ = resume_config(tmp_path, test=True)
    root = tmp_path / 'run'
    before = existing_metadata(config, root)
    monkeypatch.setattr(entry, 'configure_numerics', lambda: None)
    with pytest.raises(FileExistsError, match='evaluation requires a new empty'):
        entry.run(config)
    assert snapshot(root) == before
    config.log_dir = str(tmp_path / 'new_evaluation')
    assert not entry.validate_run_destination(config, tmp_path / 'new_evaluation')


def test_new_formal_directory_keeps_launcher_log_and_writes_metadata(tmp_path):
    config = normalize_config(dict(log_dir=str(tmp_path)))
    (tmp_path / 'train.log').write_bytes(b'launcher log\n')
    (tmp_path / 'train.pid').write_bytes(b'123\n')
    preserve = entry.validate_run_destination(config, tmp_path)
    assert not preserve
    manifest = dict(schema=SCHEMA, model=config.net_model, config_sha256=config_identity(config))
    entry.write_run_metadata(config, tmp_path, manifest, preserve_existing=preserve)
    assert json.loads((tmp_path / 'run_manifest.json').read_text(encoding='utf-8')) == manifest
    assert (tmp_path / 'resolved_config.yaml').exists()
    assert (tmp_path / 'train.log').read_bytes() == b'launcher log\n'
    assert (tmp_path / 'train.pid').read_bytes() == b'123\n'
