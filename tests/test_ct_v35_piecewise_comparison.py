"""小型完整逐帧证据：第五组比较、固定源码增量与第二种子锁定。"""
from copy import deepcopy
import hashlib
import shutil

import pytest
import yaml

from tools import compare_v35_b0 as original
from tools import compare_v35_piecewise_b0 as compare
from tests import test_ct_v35_comparison as previous
from tests.test_ct_v35_identity import config35

old = previous.old


def source(files):
    return dict(files=files, sha256=original.digest(files))


def set_source(root, value):
    manifest = old.read(root / 'run_manifest.json')
    manifest['source'] = deepcopy(value)
    old.write(root / 'run_manifest.json', manifest)


@pytest.fixture
def registration(monkeypatch):
    files = {name: str(index + 1) * 64 for index, name in enumerate(compare.CHANGED_FILES)}
    files['models/ct_v31/local_observation.py'] = '9' * 64
    before = source(files)
    after = source(dict(files, **{name: 'a' * 64 for name in compare.CHANGED_FILES}))
    value = dict(before=before, after=after, changed_files=list(compare.CHANGED_FILES))
    monkeypatch.setattr(compare, 'BASELINE_SOURCE_SHA256', before['sha256'])
    monkeypatch.setattr(compare, 'load_source_registration', lambda: deepcopy(value))
    return value


@pytest.fixture
def runs(tmp_path, monkeypatch, registration):
    result = previous.runs.__wrapped__(tmp_path, monkeypatch)
    for recipe in original.RECIPES:
        set_source(result[recipe], registration['before'])
    target = tmp_path / 'piecewise'
    shutil.copytree(result['half'], target)
    previous.seal_run(target, config35('piecewise'), v35=True)
    set_source(target, registration['after'])
    return dict(result, piecewise=target)


def improve(root):
    old.replace_evaluation(root, 'W', dynamic_iou=.95, static_iou=.95,
                           dynamic_distance=.1, static_distance=.1)
    previous.add_local_rows(root)


def add_stage2(runs, tmp_path, recipe, source_identity):
    chosen, reference = tmp_path / 'chosen52', tmp_path / 'reference52'
    shutil.copytree(runs[recipe], chosen, ignore=shutil.ignore_patterns('resume_manifests'))
    previous.seal_run(chosen, config35(recipe, 52), v35=True)
    set_source(chosen, source_identity)
    shutil.copytree(runs['registered_reference'], reference)
    cfg = yaml.safe_load((reference / 'resolved_config.yaml').read_text(encoding='utf-8'))
    cfg['seed'] = 52
    previous.seal_run(reference, cfg, v35=False)
    set_source(reference, source_identity)
    return dict(runs, selected_seed52=chosen, reference_seed52=reference)


def add_resume(root, source_identity, number=1):
    manifest = old.read(root / 'run_manifest.json')
    path = root / 'resume_manifests' / f'{number:03d}.json'
    old.write(path, dict(schema='ct_seqtrack.v35.resume_source.v1',
        original_source_sha256=manifest['source']['sha256'], source=source_identity,
        config_sha256=manifest['config_sha256'], checkpoint='checkpoints/last.ckpt'))
    return path


def fingerprints(runs):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for root in runs.values() for path in root.rglob('*') if path.is_file()}


def test_five_equal_candidates_keep_quarter_priority_and_evidence_read_only(runs):
    before = fingerprints(runs)
    report = compare.compare_runs(**runs)
    assert report['status'] == 'passed' and report['selected'] == 'quarter'
    assert report['all_candidate_rank'] == ['quarter', 'half', 'piecewise', 'normal', 'scaled']
    assert set(report['candidates']) == set(compare.RECIPES)
    assert report['selected_schedule'] == 'multistep'
    assert report['full_planning_paused'] and report['full_pause_reasons'] == ['seed52_not_verified']
    assert report['automatic_retraining'] is False
    assert 'piecewise_minus_half' in report['raw_contributions']
    assert fingerprints(runs) == before


def test_equal_half_and_piecewise_winners_prefer_original_half(runs):
    improve(runs['half'])
    improve(runs['piecewise'])
    report = compare.compare_runs(**runs)
    assert report['all_candidate_rank'][:2] == ['half', 'piecewise']
    assert report['selected'] == 'half' and report['selected_schedule'] == 'multistep'


def test_piecewise_can_win_and_lock_its_complete_seed52_recipe(runs, registration, tmp_path):
    improve(runs['piecewise'])
    report = compare.compare_runs(**add_stage2(runs, tmp_path, 'piecewise', registration['after']))
    assert report['selected'] == 'piecewise' and report['selected_lr'] == 5e-5
    assert report['selected_schedule'] == 'piecewise'
    assert report['stage2']['locked_recipe'] == 'piecewise'
    assert report['stage2']['locked_schedule'] == 'piecewise'
    assert report['stage2']['locked_lr'] == 5e-5
    assert report['stage2']['fixed_registration']['passed']
    assert report['reproducibility_confirmed'] and not report['full_planning_paused']


def test_fifth_candidate_is_required_and_cannot_reuse_an_old_directory(runs, capsys):
    four = {name: path for name, path in runs.items() if name != 'piecewise'}
    with pytest.raises(TypeError, match='piecewise'):
        compare.compare_runs(**four)
    argv = [arg for name, path in four.items() for arg in ('--' + name.replace('_', '-'), str(path))]
    with pytest.raises(SystemExit) as caught:
        compare.main(argv)
    assert caught.value.code == 2 and '--piecewise' in capsys.readouterr().err
    with pytest.raises(compare.EvidenceError, match='五个不同'):
        compare.compare_runs(**dict(runs, piecewise=runs['half']))


@pytest.mark.parametrize('change', [dict(lr_schedule='multistep'),
    dict(lr_stage_values=[5e-5, 1e-5, 1e-6]), dict(lr_milestones=[20, 49])])
def test_wrong_piecewise_schedule_is_rejected_even_with_valid_config_checksum(runs, change):
    cfg = dict(config35('piecewise'))
    cfg.update(change)
    if cfg['lr_schedule'] == 'multistep':
        cfg.pop('lr_stage_values')
    previous.seal_run(runs['piecewise'], cfg, v35=True)
    with pytest.raises(compare.EvidenceError, match='lr_'):
        compare.compare_runs(**runs)


def test_seed52_cannot_switch_from_piecewise_to_same_initial_lr_half(runs, registration, tmp_path):
    improve(runs['piecewise'])
    both = add_stage2(runs, tmp_path, 'piecewise', registration['after'])
    previous.seal_run(both['selected_seed52'], config35('half', 52), v35=True)
    with pytest.raises(compare.EvidenceError, match='lr_schedule'):
        compare.compare_runs(**both)


def test_unrelated_network_change_cannot_be_hidden_in_self_consistent_registration(runs, registration):
    registration['after']['files']['models/ct_v31/local_observation.py'] = 'f' * 64
    registration['after']['sha256'] = original.digest(registration['after']['files'])
    set_source(runs['piecewise'], registration['after'])
    with pytest.raises(compare.EvidenceError, match='未登记源码改动'):
        compare.compare_runs(**runs)


def test_any_original_candidate_manifest_must_keep_registered_before_source(runs, registration):
    set_source(runs['normal'], registration['after'])
    with pytest.raises(compare.EvidenceError, match='normal.*原四组源码'):
        compare.compare_runs(**runs)


def test_legal_old_to_new_resume_preserves_original_manifest_and_locks_effective_source(runs, registration, tmp_path):
    manifest = (runs['quarter'] / 'run_manifest.json').read_bytes()
    resume = add_resume(runs['quarter'], registration['after'])
    both = add_stage2(runs, tmp_path, 'quarter', registration['after'])
    before = fingerprints(both)
    report = compare.compare_runs(**both)
    assert report['selected'] == 'quarter' and report['reproducibility_confirmed']
    assert (runs['quarter'] / 'run_manifest.json').read_bytes() == manifest
    assert resume.exists() and fingerprints(both) == before


def test_unknown_resume_source_is_rejected(runs):
    add_resume(runs['half'], source({'unregistered.py': 'f' * 64}))
    with pytest.raises(compare.EvidenceError, match='resume.*未登记源码'):
        compare.compare_runs(**runs)


def test_resume_cannot_return_to_before_after_using_new_source(runs, registration):
    add_resume(runs['quarter'], registration['after'], 1)
    add_resume(runs['quarter'], registration['before'], 2)
    with pytest.raises(compare.EvidenceError, match='恢复回|回退'):
        compare.compare_runs(**runs)


@pytest.mark.parametrize('target', ['selected_seed52', 'reference_seed52'])
def test_seed52_effective_source_must_match_the_locked_candidate(runs, registration, tmp_path, target):
    both = add_stage2(runs, tmp_path, 'quarter', registration['before'])
    add_resume(both[target], registration['after'])
    with pytest.raises(compare.EvidenceError, match='源码'):
        compare.compare_runs(**both)


def test_original_four_group_tool_remains_independent_and_rejects_piecewise(runs):
    four = {name: path for name, path in runs.items() if name != 'piecewise'}
    report = original.compare_runs(**four)
    assert report['selected'] == 'quarter'
    assert report['all_candidate_rank'] == ['quarter', 'half', 'normal', 'scaled']
    with pytest.raises(original.EvidenceError, match='lr_schedule'):
        original.compare_runs(**dict(four, half=runs['piecewise']))
