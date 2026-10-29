"""V40 静态登记和四份正式配置合同；上传后执行，不依赖真实点云。"""
import json
from pathlib import Path

import pytest

from models.ct_v40.registration import build_registration, verify_registration, source_identity, LABELS
from models.ct_v40.config import load_config
from models.ct_v40.entry import validate_formal_registration


def fixture_root(tmp_path):
    for relative, text in {
        'main.py': '# entry\n', 'requirement.txt': '# dependencies\n',
        'models/ct_v40/model.py': '# active\n',
        'models/ct_v38/model.py': '# baseline\n',
        'models/ctseqtrackv38.py': '# baseline host\n',
        'tools/verify_ct_v40.py': '# registration tool\n',
        'tests/test_ct_v40_fixture.py': '# acceptance test\n',
        **{f'cfgs/ct_seqtrack/40_{label}_mini.yaml': f'v40_arm: {label}\n' for label in LABELS},
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return tmp_path


def test_static_registration_pins_baseline_yaml_tools_and_active_source(tmp_path):
    root = fixture_root(tmp_path)
    value = build_registration(root)
    path = root / 'cfgs/ct_seqtrack/40_registration.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    original_bytes = path.read_bytes()
    assert verify_registration(path, root) == value
    assert len(value['runs']) == 4
    assert value['waves'] == [['b0', 'b1'], ['b1_mtp', 'full']]
    assert value['local_execution'].startswith('not_run')
    tool = root / 'tools/verify_ct_v40.py'
    tool.write_text('# changed tool\n', encoding='utf-8')
    with pytest.raises(ValueError, match='engineering_files'):
        verify_registration(path, root)
    assert path.read_bytes() == original_bytes


def test_only_explicit_retired_source_paths_are_ignored(tmp_path):
    root = fixture_root(tmp_path)
    before = source_identity(root)
    retired = root / 'models/ct_v37/retired.py'
    retired.parent.mkdir()
    retired.write_text('# retired\n', encoding='utf-8')
    assert source_identity(root) == before
    unexpected = root / 'models/unknown_active.py'
    unexpected.write_text('# unknown\n', encoding='utf-8')
    assert source_identity(root)['sha256'] != before['sha256']


def test_four_pinned_yaml_recipes_match_actual_formal_entry_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    value = build_registration(root)
    registration = tmp_path / 'registration.json'
    registration.write_text(json.dumps(value), encoding='utf-8')
    for label in LABELS:
        cfg = load_config(root / f'cfgs/ct_seqtrack/40_{label}_mini.yaml')
        result = validate_formal_registration(cfg, value['source'], registration_path=registration)
        assert result['label'] == label
        assert cfg.lr == 1e-4 and cfg.lr_milestones == [20, 40] and cfg.lr_decay_rate == .2
        assert cfg.epoch == 60 and cfg.batch_size == 16 and cfg.workers == 4
    with pytest.raises(ValueError, match='formal source'):
        validate_formal_registration(cfg, dict(sha256='different'), registration_path=registration)
