"""历史权重只读工程加载不绕过原来源身份或变成 resume。"""
import hashlib
import json

import pytest
import torch
import yaml

from tools.check_execution_performance import load_engineering_config, module_set, verified_engineering_weights


def original_run(tmp_path, version):
    filename = ('cfgs/ct_seqtrack/36_a_full_mini.yaml' if version == 36 else
                'cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml')
    config = load_engineering_config(filename)
    api = module_set(config)
    from importlib import import_module
    identity = import_module('models.ct_v36.identity' if version == 36 else 'models.ct_v31.identity')
    original = api['config'].load_config(filename)
    folder = tmp_path / ('v' + str(version))
    folder.mkdir()
    (folder / 'resolved_config.yaml').write_text(yaml.safe_dump(dict(original)), encoding='utf-8')
    files = {'models/model.py': '0123456789abcdef'}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    source = dict(files=files, sha256=digest)
    manifest = dict(config_sha256=api['config'].config_identity(original), source=source,
                    model=original.net_model, schema=identity.model_schema(original), run_id='original_run')
    payload = dict(schema=identity.runtime_schema(original), model_schema=identity.model_schema(original),
        config_sha256=manifest['config_sha256'], epoch_complete=True, completed_epoch=60)
    if version == 36:
        payload.update(source=source, run_id='original_run')
    manifest_path = folder / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    checkpoint_path = folder / 'epoch=060.ckpt'
    saved = {identity.runtime_key(original): payload, 'state_dict': {'tracker.example': torch.ones(1)}}
    torch.save(saved, checkpoint_path)
    return config, checkpoint_path, manifest_path, saved, identity.runtime_key(original)


@pytest.mark.parametrize('version', [35, 36])
def test_read_only_state_loading_records_original_source_and_checkpoint_hash(tmp_path, version):
    config, checkpoint, manifest, saved, key = original_run(tmp_path, version)
    before = checkpoint.read_bytes()
    state, report = verified_engineering_weights(config, checkpoint, manifest)
    assert list(state) == ['tracker.example']
    assert report['purpose'] == 'read_only_engineering_weights_not_resume'
    assert report['completed_epoch'] == 60
    assert before == checkpoint.read_bytes()


@pytest.mark.parametrize('field,value', [('run_id', 'different_run'), ('completed_epoch', 0),
                                      ('schema', 'wrong'), ('model_schema', 'wrong'),
                                      ('epoch_complete', False)])
def test_mismatched_original_runtime_is_rejected(tmp_path, field, value):
    config, checkpoint, manifest, saved, key = original_run(tmp_path, 36)
    saved[key][field] = value
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError):
        verified_engineering_weights(config, checkpoint, manifest)
