"""调度增量恢复只追加执行来源，不覆盖原四组的配置与运行证据。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from models.ct_v31 import entry
from models.ct_v31.config import config_identity, load_config


def registration():
    path = Path(__file__).resolve().parents[1] / 'cfgs/ct_seqtrack/35_piecewise_source_registration.json'
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(root, source):
    config = load_config('cfgs/ct_seqtrack/35_b0_w_half_lr_mini.yaml',
                         {'checkpoint': str(root / 'epoch=036.ckpt')})
    manifest = dict(source=source, config_sha256=config_identity(config))
    entry.write_json(root / 'run_manifest.json', manifest)
    (root / 'resolved_config.yaml').write_bytes(b'original config\n')
    return config, manifest


@pytest.mark.parametrize('initial', ['before', 'after'])
def test_registered_resume_appends_source_and_preserves_original_files(tmp_path, initial):
    sources = registration()
    config, manifest = prepare(tmp_path, sources[initial])
    protected = {p: p.read_bytes() for p in tmp_path.iterdir()}
    current = dict(manifest, source=sources['after'])
    for _ in range(2):
        entry.record_v35_resume(config, tmp_path, current, preserve_existing=True)
    events = sorted((tmp_path / 'resume_manifests').glob('*.json'))
    assert len(events) == 2
    assert {p: p.read_bytes() for p in protected} == protected
    for path in events:
        event = json.loads(path.read_text(encoding='utf-8'))
        assert event['source'] == sources['after']
        assert event['original_source_sha256'] == sources[initial]['sha256']
        assert event['config_sha256'] == config_identity(config)
        assert event['checkpoint'] == str(Path(config.checkpoint).resolve())


@pytest.mark.parametrize('change', ['unknown_source', 'downgrade'])
def test_unregistered_revision_rejected_before_writing_event(tmp_path, change):
    sources = registration()
    config, manifest = prepare(tmp_path, sources['after'] if change == 'downgrade' else sources['before'])
    current = deepcopy(manifest)
    current['source'] = deepcopy(sources['before'] if change == 'downgrade' else sources['after'])
    if change == 'unknown_source':
        current['source']['sha256'] = 'unregistered'
    with pytest.raises(ValueError, match='registered piecewise'):
        entry.record_v35_resume(config, tmp_path, current, preserve_existing=True)
    assert not (tmp_path / 'resume_manifests').exists()
    assert json.loads((tmp_path / 'run_manifest.json').read_text(encoding='utf-8')) == manifest


def test_scratch_has_no_resume_event(tmp_path):
    config = load_config('cfgs/ct_seqtrack/35_b0_w_piecewise_lr_mini.yaml')
    entry.record_v35_resume(config, tmp_path, {}, preserve_existing=False)
    assert not list(tmp_path.iterdir())
